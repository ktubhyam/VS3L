"""
SpectralFM v2: Data Loading & Preprocessing
Handles corn, tablet datasets + wavelet decomposition + augmentation
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pywt


class SpectralPreprocessor:
    """Preprocess raw spectra: resample, normalize, wavelet decompose."""

    def __init__(self, target_length: int = 2048, wavelet: str = "db4",
                 wavelet_levels: int = 4, normalize: str = "snv"):
        self.target_length = target_length
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.normalize = normalize

    def resample(self, spectrum: np.ndarray, wavelengths: np.ndarray = None,
                 target_wavelengths: np.ndarray = None) -> np.ndarray:
        """Resample spectrum to target_length via interpolation."""
        n = len(spectrum)
        if n == self.target_length:
            return spectrum
        x_old = wavelengths if wavelengths is not None else np.linspace(0, 1, n)
        x_new = target_wavelengths if target_wavelengths is not None else np.linspace(
            x_old[0], x_old[-1], self.target_length
        )
        f = interp1d(x_old, spectrum, kind='cubic', fill_value='extrapolate')
        return f(x_new).astype(np.float32)

    def snv(self, spectrum: np.ndarray) -> np.ndarray:
        """Standard Normal Variate normalization."""
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        if std < 1e-10:
            return spectrum - mean
        return (spectrum - mean) / std

    def minmax(self, spectrum: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        mn, mx = spectrum.min(), spectrum.max()
        if mx - mn < 1e-10:
            return np.zeros_like(spectrum)
        return (spectrum - mn) / (mx - mn)

    def normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply selected normalization."""
        if self.normalize == "snv":
            return self.snv(spectrum)
        elif self.normalize == "minmax":
            return self.minmax(spectrum)
        elif self.normalize == "none":
            return spectrum
        else:
            return self.snv(spectrum)

    def wavelet_decompose(self, spectrum: np.ndarray) -> Dict[str, np.ndarray]:
        """Multi-scale wavelet decomposition."""
        coeffs = pywt.wavedec(spectrum, self.wavelet, level=self.wavelet_levels,
                              mode='symmetric')
        result = {"approx": coeffs[0]}  # Low-frequency (baseline)
        for i, detail in enumerate(coeffs[1:], 1):
            result[f"detail_{i}"] = detail
        return result

    def process(self, spectrum: np.ndarray,
                wavelengths: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Full preprocessing pipeline."""
        # Resample
        resampled = self.resample(spectrum, wavelengths)
        # Normalize
        normalized = self.normalize_spectrum(resampled)
        # Wavelet decompose
        wavelet_coeffs = self.wavelet_decompose(normalized)

        return {
            "raw": resampled.astype(np.float32),
            "normalized": normalized.astype(np.float32),
            **{k: v.astype(np.float32) for k, v in wavelet_coeffs.items()}
        }


class SpectralAugmentor:
    """Data augmentation for spectral data."""

    def __init__(self, noise_std: float = 0.01, baseline_drift_scale: float = 0.005,
                 wavelength_shift_max: int = 3, intensity_scale_range: Tuple = (0.95, 1.05)):
        self.noise_std = noise_std
        self.baseline_drift_scale = baseline_drift_scale
        self.wavelength_shift_max = wavelength_shift_max
        self.intensity_scale_range = intensity_scale_range

    def add_noise(self, spectrum: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.noise_std, spectrum.shape)
        return spectrum + noise

    def add_baseline_drift(self, spectrum: np.ndarray) -> np.ndarray:
        """Add smooth polynomial baseline drift."""
        n = len(spectrum)
        x = np.linspace(0, 1, n)
        # Random low-order polynomial
        order = np.random.randint(1, 4)
        coeffs = np.random.normal(0, self.baseline_drift_scale, order + 1)
        baseline = np.polyval(coeffs, x)
        return spectrum + baseline

    def wavelength_shift(self, spectrum: np.ndarray) -> np.ndarray:
        """Shift spectrum by a few channels (simulates wavelength calibration error)."""
        shift = np.random.randint(-self.wavelength_shift_max, self.wavelength_shift_max + 1)
        if shift == 0:
            return spectrum
        return np.roll(spectrum, shift)

    def intensity_scale(self, spectrum: np.ndarray) -> np.ndarray:
        """Random multiplicative intensity scaling."""
        scale = np.random.uniform(*self.intensity_scale_range)
        return spectrum * scale

    def augment(self, spectrum: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Apply random augmentations."""
        s = spectrum.copy()
        if np.random.random() < p:
            s = self.add_noise(s)
        if np.random.random() < p:
            s = self.add_baseline_drift(s)
        if np.random.random() < p:
            s = self.wavelength_shift(s)
        if np.random.random() < p:
            s = self.intensity_scale(s)
        return s.astype(np.float32)


class CornDataset(Dataset):
    """Corn NIR calibration transfer benchmark.
    80 samples × 3 instruments (m5, mp5, mp6) × 700 channels × 4 properties.
    """

    INSTRUMENTS = ["m5", "mp5", "mp6"]
    PROPERTIES = ["moisture", "oil", "protein", "starch"]

    def __init__(self, data_dir: str, instruments: List[str] = None,
                 preprocessor: SpectralPreprocessor = None,
                 augmentor: SpectralAugmentor = None,
                 target_property: str = "moisture",
                 split: str = "all",  # all, train, test
                 train_ratio: float = 0.75,
                 seed: int = 42):
        self.data_dir = Path(data_dir) / "processed" / "corn"
        self.instruments = instruments or self.INSTRUMENTS
        self.preprocessor = preprocessor or SpectralPreprocessor()
        self.augmentor = augmentor
        self.target_property = target_property

        # Load data
        self.wavelengths = np.load(self.data_dir / "wavelengths.npy")
        self.properties = np.load(self.data_dir / "properties.npy")  # (80, 4)

        # Property index
        self.prop_idx = self.PROPERTIES.index(target_property)

        # Load spectra per instrument
        self.spectra = {}
        for inst in self.instruments:
            self.spectra[inst] = np.load(
                self.data_dir / f"{inst}_spectra.npy"
            )  # (80, 700)

        # Train/test split
        n = self.spectra[self.instruments[0]].shape[0]
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        n_train = int(n * train_ratio)

        if split == "train":
            self.indices = indices[:n_train]
        elif split == "test":
            self.indices = indices[n_train:]
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices) * len(self.instruments)

    def __getitem__(self, idx):
        # Map flat index to (sample, instrument)
        sample_idx = idx // len(self.instruments)
        inst_idx = idx % len(self.instruments)
        real_idx = self.indices[sample_idx]
        inst = self.instruments[inst_idx]

        spectrum = self.spectra[inst][real_idx]
        target = self.properties[real_idx, self.prop_idx]

        # Preprocess
        processed = self.preprocessor.process(spectrum, self.wavelengths)

        # Augment (training only)
        if self.augmentor is not None:
            processed["normalized"] = self.augmentor.augment(processed["normalized"])

        return {
            "spectrum": torch.tensor(processed["normalized"], dtype=torch.float32),
            "raw": torch.tensor(processed["raw"], dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "instrument": inst,
            "instrument_id": torch.tensor(inst_idx, dtype=torch.long),
            "sample_id": torch.tensor(real_idx, dtype=torch.long),
            "domain": "NIR",
        }


class TabletDataset(Dataset):
    """IDRC 2002 Pharmaceutical Tablet NIR Shootout.
    655 tablets × 2 spectrometers × 650 channels × 3 properties.
    """

    def __init__(self, data_dir: str, split: str = "calibrate",
                 preprocessor: SpectralPreprocessor = None,
                 augmentor: SpectralAugmentor = None,
                 target_property: int = 0):  # 0=active, 1=weight, 2=hardness
        self.data_dir = Path(data_dir) / "processed" / "tablet"
        self.preprocessor = preprocessor or SpectralPreprocessor()
        self.augmentor = augmentor
        self.target_property = target_property

        # Load data
        self.spectra_1 = np.load(self.data_dir / f"{split}_1.npy")
        self.spectra_2 = np.load(self.data_dir / f"{split}_2.npy")
        self.targets = np.load(self.data_dir / f"{split}_Y.npy")

    def __len__(self):
        return self.spectra_1.shape[0] * 2  # Both instruments

    def __getitem__(self, idx):
        sample_idx = idx // 2
        inst_idx = idx % 2

        spectrum = self.spectra_1[sample_idx] if inst_idx == 0 else self.spectra_2[sample_idx]
        target = self.targets[sample_idx, self.target_property]

        processed = self.preprocessor.process(spectrum)

        if self.augmentor is not None:
            processed["normalized"] = self.augmentor.augment(processed["normalized"])

        return {
            "spectrum": torch.tensor(processed["normalized"], dtype=torch.float32),
            "raw": torch.tensor(processed["raw"], dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "instrument": f"spec_{inst_idx + 1}",
            "instrument_id": torch.tensor(inst_idx, dtype=torch.long),
            "sample_id": torch.tensor(sample_idx, dtype=torch.long),
            "domain": "NIR",
        }


class CalibrationTransferDataset(Dataset):
    """Paired dataset for calibration transfer fine-tuning.
    Returns (source_spectrum, target_spectrum, target_value) triplets.
    """

    def __init__(self, source_spectra: np.ndarray, target_spectra: np.ndarray,
                 targets: np.ndarray, n_transfer: int = 10,
                 preprocessor: SpectralPreprocessor = None,
                 seed: int = 42):
        self.preprocessor = preprocessor or SpectralPreprocessor()

        # Select n_transfer samples
        rng = np.random.RandomState(seed)
        n = min(n_transfer, len(targets))
        self.indices = rng.choice(len(targets), n, replace=False)

        self.source = source_spectra[self.indices]
        self.target = target_spectra[self.indices]
        self.targets = targets[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        src = self.preprocessor.process(self.source[idx])
        tgt = self.preprocessor.process(self.target[idx])

        return {
            "source_spectrum": torch.tensor(src["normalized"], dtype=torch.float32),
            "target_spectrum": torch.tensor(tgt["normalized"], dtype=torch.float32),
            "target_value": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class PretrainHDF5Dataset(Dataset):
    """Dataset for pretraining from HDF5 corpus.

    Supports both v1 and v2 corpus formats:
      v1: /spectra, /modality (int8), /source_id (int16)
      v2: /spectra, /metadata/source (str), /metadata/type (str), /metadata/sample_id (str)

    If modality/source_id are not present, falls back to metadata/type and metadata/source.
    """

    MODALITY_NAMES = ["IR", "RAMAN", "NIR", "FTIR", "UNKNOWN"]
    TYPE_TO_MODALITY = {"ir": 0, "raman": 1, "nir": 2, "ftir": 3}

    def __init__(self, h5_path: str = "data/pretrain/spectral_corpus_v2.h5",
                 augment: bool = True,
                 augmentor: SpectralAugmentor = None,
                 max_samples: int = None,
                 default_domain: str = "RAMAN",
                 filter_sources: Optional[List[str]] = None,
                 filter_types: Optional[List[str]] = None):
        """
        Args:
            h5_path: Path to HDF5 corpus file
            augment: Whether to apply augmentation
            augmentor: Custom augmentor (or uses default)
            max_samples: Limit number of samples (for debugging)
            default_domain: Default domain if not in HDF5
            filter_sources: Only include spectra from these sources (e.g. ["rruff", "chembl"])
            filter_types: Only include spectra of these types (e.g. ["raman", "ir"])
        """
        import h5py
        import logging
        log = logging.getLogger(__name__)

        self.h5_path = h5_path
        self.augment = augment
        self.augmentor = augmentor or SpectralAugmentor()
        self.default_domain = default_domain

        # Load dataset metadata (keep file closed for multiprocessing)
        with h5py.File(h5_path, 'r') as f:
            total_samples = f['spectra'].shape[0]
            self.n_channels = f['spectra'].shape[1]

            # Detect format version
            self.has_modality = 'modality' in f
            self.has_source_id = 'source_id' in f
            self.has_metadata = 'metadata' in f

            # Build index filter if needed
            self._indices = None
            if (filter_sources or filter_types) and self.has_metadata:
                sources = f['metadata/source'][:]
                types = f['metadata/type'][:]

                mask = np.ones(total_samples, dtype=bool)

                if filter_sources:
                    src_set = set(s.encode() if isinstance(s, str) else s
                                  for s in filter_sources)
                    # Also include augmented versions
                    src_set_aug = set(s + b"_aug" if isinstance(s, bytes)
                                      else (s + "_aug").encode()
                                      for s in filter_sources)
                    src_set = src_set | src_set_aug
                    mask &= np.array([s in src_set for s in sources])

                if filter_types:
                    type_set = set(t.encode() if isinstance(t, str) else t
                                   for t in filter_types)
                    mask &= np.array([t in type_set for t in types])

                self._indices = np.where(mask)[0]

            # Compute effective size
            if self._indices is not None:
                self.n_samples = len(self._indices)
            else:
                self.n_samples = total_samples

            if max_samples is not None:
                self.n_samples = min(self.n_samples, max_samples)

            # Log corpus stats
            if self.has_metadata:
                sources = f['metadata/source'][:]
                types = f['metadata/type'][:]
                from collections import Counter
                src_counts = Counter(
                    s.decode() if isinstance(s, bytes) else s for s in sources
                )
                type_counts = Counter(
                    t.decode() if isinstance(t, bytes) else t for t in types
                )
                log.info(f"Corpus: {total_samples:,} spectra, {self.n_channels} channels")
                log.info(f"Sources: {dict(src_counts)}")
                log.info(f"Types: {dict(type_counts)}")
                if self._indices is not None:
                    log.info(f"After filtering: {self.n_samples:,} spectra")

        # Will be opened lazily in worker process
        self._h5_file = None

    def __getstate__(self):
        """For pickling (multiprocessing): don't pickle the h5 handle."""
        state = self.__dict__.copy()
        state['_h5_file'] = None
        return state

    def __setstate__(self, state):
        """For unpickling: restore without h5 handle (will lazy-open)."""
        self.__dict__.update(state)

    def _get_h5(self):
        """Lazy open HDF5 file (for multiprocessing compatibility)."""
        import h5py
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def _real_idx(self, idx):
        """Map filtered index to actual HDF5 index."""
        if self._indices is not None:
            return int(self._indices[idx])
        return idx

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        h5 = self._get_h5()
        real_idx = self._real_idx(idx)

        spectrum = h5['spectra'][real_idx].astype(np.float32)

        # Get modality/domain
        if self.has_modality:
            modality = int(h5['modality'][real_idx])
            domain = self.MODALITY_NAMES[min(modality, len(self.MODALITY_NAMES) - 1)]
        elif self.has_metadata and 'type' in h5['metadata']:
            spec_type = h5['metadata/type'][real_idx]
            if isinstance(spec_type, bytes):
                spec_type = spec_type.decode()
            modality = self.TYPE_TO_MODALITY.get(spec_type.lower(), 4)
            domain = spec_type.upper()
        else:
            modality = self.MODALITY_NAMES.index(self.default_domain) if self.default_domain in self.MODALITY_NAMES else 4
            domain = self.default_domain

        # Get source_id
        if self.has_source_id:
            source_id = int(h5['source_id'][real_idx])
        else:
            source_id = 0

        # Augment if enabled
        if self.augment:
            spectrum = self.augmentor.augment(spectrum, p=0.5)

        return {
            "spectrum": torch.tensor(spectrum, dtype=torch.float32),
            "modality": torch.tensor(modality, dtype=torch.long),
            "source_id": torch.tensor(source_id, dtype=torch.long),
            "instrument_id": torch.tensor(source_id, dtype=torch.long),
            "domain": domain,
        }

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


def build_pretrain_loader(data_dir: str, batch_size: int = 64,
                          target_length: int = 2048,
                          num_workers: int = 2) -> DataLoader:
    """Build pretraining dataloader combining all available data."""
    preprocessor = SpectralPreprocessor(target_length=target_length)
    augmentor = SpectralAugmentor()

    datasets = []

    # Corn (all instruments, all splits)
    corn = CornDataset(data_dir, preprocessor=preprocessor,
                       augmentor=augmentor, split="all")
    datasets.append(corn)

    # Tablet (calibrate split, both instruments)
    tablet = TabletDataset(data_dir, split="calibrate",
                           preprocessor=preprocessor, augmentor=augmentor)
    datasets.append(tablet)

    combined = torch.utils.data.ConcatDataset(datasets)

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_hdf5_pretrain_loader(h5_path: str, batch_size: int = 64,
                                num_workers: int = 0,
                                max_samples: int = None) -> DataLoader:
    """Build pretraining dataloader from HDF5 corpus.

    Args:
        h5_path: Path to HDF5 corpus file
        batch_size: Batch size
        num_workers: Number of data loading workers (0 for main process)
        max_samples: Limit number of samples (for debugging)
    """
    dataset = PretrainHDF5Dataset(h5_path, augment=True, max_samples=max_samples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
