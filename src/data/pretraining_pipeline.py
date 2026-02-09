"""
SpectralFM v2: Pretraining Data Pipeline

Downloads, parses, normalizes, and augments spectral data from multiple
public databases into a unified HDF5 corpus for self-supervised pretraining.

Supported sources:
- RRUFF: Mineral Raman/IR spectra (~8,600)
- NIST: Chemistry WebBook IR spectra (JCAMP-DX format)
- OpenSpecy: Reference Raman/FTIR library
- Synthetic: Augmented versions of real spectra

All spectra are resampled to 2048 points with SNV normalization.
"""
import numpy as np
import h5py
import requests
import zipfile
import tarfile
import io
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SpectrumRecord:
    """Container for a single spectrum with metadata."""
    spectrum: np.ndarray  # (2048,) normalized spectrum
    source: str           # "rruff", "nist", "openspecy", "synthetic"
    spec_type: str        # "raman", "ir", "nir", "uv-vis"
    original_range: Tuple[float, float]  # (min_wn, max_wn)
    sample_id: str        # Original filename/identifier


def normalize_spectrum(wavenumbers: np.ndarray, intensities: np.ndarray,
                       target_points: int = 2048) -> Optional[np.ndarray]:
    """Normalize spectrum to common format.

    1. Sort by wavenumber (ascending)
    2. Remove NaN/Inf values
    3. Interpolate to uniform grid of target_points
    4. SNV normalization: subtract mean, divide by std

    Returns:
        (target_points,) numpy array float32, or None if spectrum is invalid
    """
    try:
        # Convert to numpy arrays
        wn = np.asarray(wavenumbers, dtype=np.float64)
        intensity = np.asarray(intensities, dtype=np.float64)

        # Check for valid data
        if len(wn) < 10 or len(intensity) < 10:
            return None
        if len(wn) != len(intensity):
            return None

        # Remove NaN/Inf
        valid = np.isfinite(wn) & np.isfinite(intensity)
        wn = wn[valid]
        intensity = intensity[valid]

        if len(wn) < 10:
            return None

        # Sort by wavenumber
        sort_idx = np.argsort(wn)
        wn = wn[sort_idx]
        intensity = intensity[sort_idx]

        # Remove duplicates (keep first)
        _, unique_idx = np.unique(wn, return_index=True)
        wn = wn[unique_idx]
        intensity = intensity[unique_idx]

        if len(wn) < 10:
            return None

        # Interpolate to uniform grid
        target_wn = np.linspace(wn.min(), wn.max(), target_points)
        interpolator = interp1d(wn, intensity, kind='linear',
                                fill_value='extrapolate', bounds_error=False)
        resampled = interpolator(target_wn)

        # SNV normalization
        mean_val = np.mean(resampled)
        std_val = np.std(resampled)
        if std_val < 1e-10:
            return None
        normalized = (resampled - mean_val) / std_val

        # Check for any remaining NaN/Inf
        if not np.all(np.isfinite(normalized)):
            return None

        return normalized.astype(np.float32)

    except Exception as e:
        logger.debug(f"Failed to normalize spectrum: {e}")
        return None


class SpectrumAugmentor:
    """Generate augmented versions of spectra for pretraining."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def add_noise(self, spectrum: np.ndarray, snr_db: float = 30) -> np.ndarray:
        """Add Gaussian noise at specified SNR."""
        signal_power = np.mean(spectrum ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), spectrum.shape)
        return spectrum + noise

    def multiplicative_scatter(self, spectrum: np.ndarray,
                                scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """Apply random multiplicative scaling per wavelength."""
        scales = self.rng.uniform(scale_range[0], scale_range[1], spectrum.shape)
        # Smooth the scales to be correlated across wavelengths
        scales = gaussian_filter1d(scales, sigma=50)
        return spectrum * scales

    def wavelength_shift(self, spectrum: np.ndarray, max_shift: int = 5) -> np.ndarray:
        """Shift spectrum by random amount (simulates calibration error)."""
        shift = self.rng.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return spectrum
        return np.roll(spectrum, shift)

    def baseline_drift(self, spectrum: np.ndarray, max_coeff: float = 0.1) -> np.ndarray:
        """Add polynomial baseline drift."""
        n = len(spectrum)
        x = np.linspace(-1, 1, n)
        degree = self.rng.randint(2, 5)
        coeffs = self.rng.uniform(-max_coeff, max_coeff, degree + 1)
        baseline = np.polyval(coeffs, x)
        return spectrum + baseline

    def augment(self, spectrum: np.ndarray, n_augmented: int = 20) -> List[np.ndarray]:
        """Generate n_augmented versions of a spectrum."""
        augmented = []

        for _ in range(n_augmented):
            s = spectrum.copy()

            # Random combination of augmentations
            if self.rng.random() < 0.7:
                snr = self.rng.uniform(20, 40)
                s = self.add_noise(s, snr)

            if self.rng.random() < 0.5:
                s = self.multiplicative_scatter(s)

            if self.rng.random() < 0.3:
                s = self.wavelength_shift(s, max_shift=3)

            if self.rng.random() < 0.4:
                s = self.baseline_drift(s, max_coeff=0.05)

            # Re-normalize after augmentation
            mean_val = np.mean(s)
            std_val = np.std(s)
            if std_val > 1e-10:
                s = (s - mean_val) / std_val
                augmented.append(s.astype(np.float32))

        return augmented


class RRUFFDownloader:
    """Download and parse RRUFF mineral spectra database."""

    # RRUFF provides bulk downloads at these URLs
    RAMAN_URL = "https://rruff.info/zipped_data_files/raman/LR-Raman.zip"
    IR_URL = "https://rruff.info/zipped_data_files/infrared/Infrared.zip"

    # Alternative: individual mineral search
    SEARCH_URL = "https://rruff.info/rruff_export.php"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_zip(self, url: str, name: str) -> Optional[Path]:
        """Download and extract a zip file."""
        zip_path = self.cache_dir / f"{name}.zip"
        extract_dir = self.cache_dir / name

        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"Using cached {name} data")
            return extract_dir

        logger.info(f"Downloading {name} from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Extracting {name}...")
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)

            return extract_dir

        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            return None

    def parse_rruff_file(self, filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        """Parse a RRUFF spectrum file.

        Format: Two columns (wavenumber, intensity), with ## header lines.
        """
        try:
            wavenumbers = []
            intensities = []
            mineral_name = filepath.stem

            with open(filepath, 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('##'):
                        continue

                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            wn = float(parts[0])
                            intensity = float(parts[1])
                            wavenumbers.append(wn)
                            intensities.append(intensity)
                        except ValueError:
                            continue

            if len(wavenumbers) > 10:
                return np.array(wavenumbers), np.array(intensities), mineral_name
            return None

        except Exception as e:
            logger.debug(f"Failed to parse {filepath}: {e}")
            return None

    def get_spectra(self) -> Generator[SpectrumRecord, None, None]:
        """Yield all RRUFF spectra."""

        # Try to download bulk files
        for url, name, spec_type in [
            (self.RAMAN_URL, "raman", "raman"),
            (self.IR_URL, "infrared", "ir"),
        ]:
            extract_dir = self.download_zip(url, name)
            if extract_dir is None:
                continue

            # Find all spectrum files
            spectrum_files = list(extract_dir.rglob("*.txt")) + list(extract_dir.rglob("*.rruff"))
            logger.info(f"Found {len(spectrum_files)} {name} files")

            for filepath in tqdm(spectrum_files, desc=f"Parsing RRUFF {name}"):
                result = self.parse_rruff_file(filepath)
                if result is None:
                    continue

                wn, intensity, mineral_name = result
                normalized = normalize_spectrum(wn, intensity)

                if normalized is not None:
                    yield SpectrumRecord(
                        spectrum=normalized,
                        source="rruff",
                        spec_type=spec_type,
                        original_range=(wn.min(), wn.max()),
                        sample_id=f"rruff_{mineral_name}_{filepath.stem}"
                    )


class JCAMPParser:
    """Parse JCAMP-DX format spectral files."""

    @staticmethod
    def parse_file(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        """Parse a JCAMP-DX file."""
        try:
            import jcamp
            data = jcamp.jcamp_read(str(filepath))

            if 'x' in data and 'y' in data:
                wn = np.array(data['x'])
                intensity = np.array(data['y'])
                name = data.get('title', filepath.stem)

                if len(wn) > 10:
                    return wn, intensity, name
            return None

        except Exception as e:
            logger.debug(f"Failed to parse JCAMP file {filepath}: {e}")
            return None


class OpenSpecyDownloader:
    """Download and parse OpenSpecy reference library."""

    # OpenSpecy provides data via GitHub releases
    GITHUB_API = "https://api.github.com/repos/wincowgerDEV/OpenSpecy-package/releases/latest"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_library(self) -> Optional[Path]:
        """Try to download OpenSpecy reference library."""
        try:
            # Check for existing data
            data_dir = self.cache_dir / "openspecy"
            if data_dir.exists() and any(data_dir.iterdir()):
                logger.info("Using cached OpenSpecy data")
                return data_dir

            # Try to get release info
            logger.info("Checking OpenSpecy GitHub releases...")
            response = requests.get(self.GITHUB_API, timeout=30)

            if response.status_code != 200:
                logger.warning("Could not access OpenSpecy releases")
                return None

            release = response.json()
            assets = release.get('assets', [])

            # Look for data files in assets
            for asset in assets:
                if 'data' in asset['name'].lower() or 'spectr' in asset['name'].lower():
                    url = asset['browser_download_url']
                    logger.info(f"Downloading OpenSpecy data from {url}")
                    # Download logic would go here

            return None  # OpenSpecy integration is optional

        except Exception as e:
            logger.warning(f"OpenSpecy download failed: {e}")
            return None

    def get_spectra(self) -> Generator[SpectrumRecord, None, None]:
        """Yield OpenSpecy spectra (placeholder for future implementation)."""
        # OpenSpecy requires R package or specific API access
        # For now, we'll rely on RRUFF and augmentation
        return
        yield  # Make this a generator


class PretrainingCorpusBuilder:
    """Build unified HDF5 corpus from multiple sources."""

    def __init__(self, output_path: Path, cache_dir: Path, target_points: int = 2048):
        self.output_path = output_path
        self.cache_dir = cache_dir
        self.target_points = target_points
        self.augmentor = SpectrumAugmentor()

        # Source handlers
        self.rruff = RRUFFDownloader(cache_dir / "rruff")
        self.openspecy = OpenSpecyDownloader(cache_dir / "openspecy")

    def collect_real_spectra(self) -> List[SpectrumRecord]:
        """Collect all real spectra from available sources."""
        all_spectra = []

        # RRUFF (primary source)
        logger.info("Collecting RRUFF spectra...")
        rruff_count = 0
        for record in self.rruff.get_spectra():
            all_spectra.append(record)
            rruff_count += 1
        logger.info(f"Collected {rruff_count} RRUFF spectra")

        # OpenSpecy (optional)
        logger.info("Checking OpenSpecy...")
        openspecy_count = 0
        for record in self.openspecy.get_spectra():
            all_spectra.append(record)
            openspecy_count += 1
        if openspecy_count > 0:
            logger.info(f"Collected {openspecy_count} OpenSpecy spectra")

        return all_spectra

    def augment_spectra(self, real_spectra: List[SpectrumRecord],
                        target_total: int = 200000) -> List[SpectrumRecord]:
        """Augment spectra to reach target count."""
        n_real = len(real_spectra)
        if n_real == 0:
            logger.error("No real spectra to augment!")
            return []

        # Calculate augmentation factor
        n_augment_per = max(1, (target_total - n_real) // n_real)
        logger.info(f"Generating {n_augment_per} augmented versions per spectrum...")

        augmented_spectra = []

        for record in tqdm(real_spectra, desc="Augmenting"):
            # Keep original
            augmented_spectra.append(record)

            # Generate augmented versions
            aug_specs = self.augmentor.augment(record.spectrum, n_augmented=n_augment_per)
            for i, aug_spec in enumerate(aug_specs):
                augmented_spectra.append(SpectrumRecord(
                    spectrum=aug_spec,
                    source="synthetic",
                    spec_type=record.spec_type,
                    original_range=record.original_range,
                    sample_id=f"{record.sample_id}_aug{i}"
                ))

        return augmented_spectra

    def save_to_hdf5(self, spectra: List[SpectrumRecord]):
        """Save all spectra to HDF5 file."""
        n_spectra = len(spectra)
        logger.info(f"Saving {n_spectra} spectra to {self.output_path}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.output_path, 'w') as f:
            # Create datasets
            spectra_data = np.stack([s.spectrum for s in spectra])
            f.create_dataset('spectra', data=spectra_data, dtype='float32',
                            chunks=(min(1000, n_spectra), self.target_points),
                            compression='gzip', compression_opts=4)

            # Metadata
            meta = f.create_group('metadata')

            # String arrays need special handling in h5py
            dt = h5py.special_dtype(vlen=str)

            sources = np.array([s.source for s in spectra], dtype=object)
            meta.create_dataset('source', data=sources, dtype=dt)

            types = np.array([s.spec_type for s in spectra], dtype=object)
            meta.create_dataset('type', data=types, dtype=dt)

            ranges = np.array([s.original_range for s in spectra], dtype='float32')
            meta.create_dataset('original_range', data=ranges)

            sample_ids = np.array([s.sample_id for s in spectra], dtype=object)
            meta.create_dataset('sample_id', data=sample_ids, dtype=dt)

            # Attributes
            f.attrs['corpus_size'] = n_spectra
            f.attrs['target_points'] = self.target_points
            f.attrs['normalization'] = 'snv'

            from datetime import datetime
            f.attrs['creation_date'] = datetime.now().isoformat()

        logger.info(f"Saved corpus to {self.output_path}")

    def validate_corpus(self):
        """Load and validate the saved corpus."""
        logger.info("Validating corpus...")

        with h5py.File(self.output_path, 'r') as f:
            spectra = f['spectra'][:]
            sources = f['metadata/source'][:]
            types = f['metadata/type'][:]

            # Basic checks
            assert spectra.shape[1] == self.target_points
            assert len(sources) == spectra.shape[0]
            assert len(types) == spectra.shape[0]

            # Check for NaN/Inf
            assert np.all(np.isfinite(spectra)), "Found NaN/Inf in spectra!"

            # Statistics
            logger.info(f"Corpus size: {spectra.shape[0]} spectra")
            logger.info(f"Spectrum shape: {spectra.shape[1]} points")
            logger.info(f"Memory: {spectra.nbytes / 1e6:.1f} MB")

            # Source breakdown
            unique_sources, counts = np.unique(sources, return_counts=True)
            for src, cnt in zip(unique_sources, counts):
                logger.info(f"  {src}: {cnt} spectra")

            # Type breakdown
            unique_types, counts = np.unique(types, return_counts=True)
            for typ, cnt in zip(unique_types, counts):
                logger.info(f"  {typ}: {cnt} spectra")

            # Spot check a few spectra
            for i in [0, len(spectra)//2, -1]:
                s = spectra[i]
                logger.info(f"  Spectrum {i}: mean={s.mean():.3f}, std={s.std():.3f}, "
                           f"range=[{s.min():.3f}, {s.max():.3f}]")

        logger.info("Validation passed!")
        return True

    def build(self, target_total: int = 200000):
        """Build the complete pretraining corpus."""
        logger.info("=" * 60)
        logger.info("Building SpectralFM Pretraining Corpus")
        logger.info("=" * 60)

        # Collect real spectra
        real_spectra = self.collect_real_spectra()
        logger.info(f"Total real spectra: {len(real_spectra)}")

        if len(real_spectra) == 0:
            logger.error("No spectra collected! Check download sources.")
            return False

        # Augment to target size
        all_spectra = self.augment_spectra(real_spectra, target_total)
        logger.info(f"Total after augmentation: {len(all_spectra)}")

        # Save to HDF5
        self.save_to_hdf5(all_spectra)

        # Validate
        return self.validate_corpus()
