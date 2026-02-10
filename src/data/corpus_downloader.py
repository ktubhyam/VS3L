"""
Corpus Downloader: Download and parse spectral data from multiple public databases.

Supported sources:
- ChEMBL: ~220K computed Raman+IR spectra (SQLite from Figshare)
- USPTO: ~177K computed IR spectra (Parquet from Zenodo)
- OpenSpecy: ~30K experimental Raman+FTIR (RDS from OSF)
- RRUFF: ~8.6K experimental Raman+IR mineral spectra (ZIP from rruff.info)

Each downloader yields SpectrumRecord objects compatible with the pretraining pipeline.
"""
import numpy as np
import logging
import time
import sqlite3
import zlib
import pickle
from pathlib import Path
from typing import Optional, Tuple, Generator, List
from dataclasses import dataclass

from tqdm import tqdm

from src.data.pretraining_pipeline import SpectrumRecord, normalize_spectrum

logger = logging.getLogger(__name__)


def _download_file(url: str, dest: Path, desc: str = "", max_retries: int = 3,
                   timeout: int = 600) -> bool:
    """Download a file with retries and progress bar."""
    import requests

    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Using cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {desc or dest.name} (attempt {attempt}/{max_retries})...")
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total = int(response.headers.get('content-length', 0))
            with open(dest, 'wb') as f:
                with tqdm(total=total, unit='B', unit_scale=True,
                          desc=desc or dest.name) as pbar:
                    for chunk in response.iter_content(chunk_size=65536):
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Validate: detect HTML responses masquerading as binary files
            with open(dest, 'rb') as f_check:
                header = f_check.read(64)
            if header.lstrip().startswith(b'<!DOCTYPE') or header.lstrip().startswith(b'<html'):
                logger.warning(f"Download returned HTML page instead of data file (attempt {attempt})")
                dest.unlink()
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                continue

            logger.info(f"Downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True

        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if dest.exists():
                dest.unlink()
            if attempt < max_retries:
                time.sleep(5 * attempt)

    logger.error(f"Failed to download {url} after {max_retries} attempts")
    return False


def _lorentzian_broadening(frequencies: np.ndarray, intensities: np.ndarray,
                            x_range: Tuple[float, float] = (400.0, 4000.0),
                            n_points: int = 2048, gamma: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct continuous spectrum from discrete vibrational modes via Lorentzian broadening."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    spectrum = np.zeros(n_points, dtype=np.float64)

    # Vectorized: sum of Lorentzians
    for freq, inten in zip(frequencies, intensities):
        if not (np.isfinite(freq) and np.isfinite(inten)):
            continue
        if inten <= 0 or freq < x_range[0] or freq > x_range[1]:
            continue
        spectrum += inten * gamma**2 / ((x - freq)**2 + gamma**2)

    return x, spectrum


# ═══════════════════════════════════════════════════════════════════════
# ChEMBL Downloader
# ═══════════════════════════════════════════════════════════════════════

class ChEMBLDownloader:
    """Download and parse ChEMBL computed Raman+IR spectra from Figshare.

    Data: SQLite DBs with discrete vibrational modes (freq, IR_inten, Raman_activ).
    We reconstruct continuous spectra via Lorentzian broadening.
    """

    # Figshare direct download URLs for SQLite DB files
    PART1_URL = "https://ndownloader.figshare.com/files/54662735"
    PART2_URL = "https://ndownloader.figshare.com/files/54662885"

    def __init__(self, cache_dir: Path, download_both_parts: bool = True):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_both_parts = download_both_parts

    def download(self) -> List[Path]:
        """Download ChEMBL SQLite DB files. Returns list of downloaded paths."""
        downloaded = []

        part1 = self.cache_dir / "Raman-ChEMBL-part1.db"
        if _download_file(self.PART1_URL, part1, "ChEMBL Part 1 (~5 GB)"):
            downloaded.append(part1)

        if self.download_both_parts:
            part2 = self.cache_dir / "Raman-ChEMBL-part2.db"
            if _download_file(self.PART2_URL, part2, "ChEMBL Part 2 (~5.4 GB)"):
                downloaded.append(part2)

        return downloaded

    def _parse_db(self, db_path: Path, spec_type: str = "raman",
                  max_molecules: int = None) -> Generator[SpectrumRecord, None, None]:
        """Parse a ChEMBL SQLite DB file and yield spectra."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Discover table name
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        if not tables:
            logger.error(f"No tables found in {db_path}")
            conn.close()
            return

        table = tables[0]
        logger.info(f"Using table '{table}' from {db_path.name}")

        # Get column info
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        logger.info(f"Columns: {columns[:10]}...")

        # Count rows
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total = cursor.fetchone()[0]
        if max_molecules:
            total = min(total, max_molecules)
        logger.info(f"Processing {total} molecules from {db_path.name}")

        # Process in batches
        batch_size = 1000
        offset = 0
        yielded = 0

        while offset < total:
            cursor.execute(f"SELECT * FROM {table} LIMIT {batch_size} OFFSET {offset}")
            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                try:
                    record = self._process_row(row, columns, spec_type, db_path.stem)
                    if record is not None:
                        yield record
                        yielded += 1
                except Exception as e:
                    logger.debug(f"Failed to process row: {e}")
                    continue

            offset += batch_size
            if offset % 10000 == 0:
                logger.info(f"  Processed {offset}/{total}, yielded {yielded} spectra")

        conn.close()
        logger.info(f"Total yielded from {db_path.name}: {yielded} {spec_type} spectra")

    def _process_row(self, row, columns, spec_type: str,
                     db_name: str) -> Optional[SpectrumRecord]:
        """Process a single row from ChEMBL DB into a SpectrumRecord."""
        # Build dict from row
        data = {}
        for col, val in zip(columns, row):
            if isinstance(val, bytes):
                try:
                    decompressed = zlib.decompress(val)
                    data[col] = pickle.loads(decompressed)
                except Exception:
                    try:
                        data[col] = pickle.loads(val)
                    except Exception:
                        data[col] = val
            else:
                data[col] = val

        # Extract frequencies
        frequencies = None
        for key in ['freq', 'frequency', 'frequencies', 'Frequency']:
            if key in data and data[key] is not None:
                frequencies = np.asarray(data[key], dtype=np.float64).ravel()
                break

        if frequencies is None or len(frequencies) < 3:
            return None

        # Extract intensities based on spec_type
        if spec_type == "raman":
            intensities = None
            for key in ['Raman Activ', 'raman_activ', 'Raman_Activ', 'raman']:
                if key in data and data[key] is not None:
                    intensities = np.asarray(data[key], dtype=np.float64).ravel()
                    break
        else:  # IR
            intensities = None
            for key in ['IR Inten', 'ir_inten', 'IR_Inten', 'ir']:
                if key in data and data[key] is not None:
                    intensities = np.asarray(data[key], dtype=np.float64).ravel()
                    break

        if intensities is None or len(intensities) < 3:
            return None

        # Ensure same length
        min_len = min(len(frequencies), len(intensities))
        frequencies = frequencies[:min_len]
        intensities = intensities[:min_len]

        # Reconstruct continuous spectrum via Lorentzian broadening
        if spec_type == "raman":
            wn_range = (100.0, 4000.0)
        else:
            wn_range = (400.0, 4000.0)

        wavenumbers, spectrum = _lorentzian_broadening(
            frequencies, intensities, x_range=wn_range, n_points=2048, gamma=10.0
        )

        if np.max(spectrum) < 1e-12:
            return None

        # Normalize
        normalized = normalize_spectrum(wavenumbers, spectrum)
        if normalized is None:
            return None

        # Get sample ID
        sample_id_val = data.get('SMILES', data.get('smiles', str(hash(tuple(frequencies[:5])))))
        if not isinstance(sample_id_val, str):
            sample_id_val = str(sample_id_val)

        return SpectrumRecord(
            spectrum=normalized,
            source="chembl",
            spec_type=spec_type,
            original_range=(wn_range[0], wn_range[1]),
            sample_id=f"chembl_{db_name}_{sample_id_val[:50]}"
        )

    def get_spectra(self, max_molecules_per_part: int = None) -> Generator[SpectrumRecord, None, None]:
        """Download and yield all ChEMBL spectra (Raman + IR)."""
        db_files = self.download()
        if not db_files:
            logger.error("No ChEMBL DB files downloaded")
            return

        for db_path in db_files:
            # Yield Raman spectra
            logger.info(f"Extracting Raman spectra from {db_path.name}...")
            yield from self._parse_db(db_path, spec_type="raman",
                                       max_molecules=max_molecules_per_part)

            # Yield IR spectra
            logger.info(f"Extracting IR spectra from {db_path.name}...")
            yield from self._parse_db(db_path, spec_type="ir",
                                       max_molecules=max_molecules_per_part)


# ═══════════════════════════════════════════════════════════════════════
# USPTO Downloader
# ═══════════════════════════════════════════════════════════════════════

class USPTODownloader:
    """Download and parse USPTO computed IR spectra from Zenodo.

    Data: Parquet files with pre-computed continuous IR spectra (12K points).
    """

    BASE_URL = "https://zenodo.org/api/records/16417648/files"
    N_CHUNKS = 9

    def __init__(self, cache_dir: Path, max_chunks: int = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_chunks = max_chunks or self.N_CHUNKS

    def download(self) -> List[Path]:
        """Download USPTO Parquet chunk files."""
        downloaded = []

        for i in range(1, self.max_chunks + 1):
            fname = f"IR_data_chunk{i:03d}_of_009.parquet"
            url = f"{self.BASE_URL}/{fname}/content"
            dest = self.cache_dir / fname

            if _download_file(url, dest, f"USPTO chunk {i}/{self.max_chunks} (~900 MB)"):
                downloaded.append(dest)

        return downloaded

    def get_spectra(self, max_per_chunk: int = None) -> Generator[SpectrumRecord, None, None]:
        """Download and yield all USPTO IR spectra."""
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas required for USPTO parsing. Install with: pip install pandas pyarrow")
            return

        parquet_files = self.download()
        if not parquet_files:
            logger.error("No USPTO parquet files downloaded")
            return

        for pf in parquet_files:
            logger.info(f"Parsing {pf.name}...")
            try:
                df = pd.read_parquet(pf)
            except Exception as e:
                logger.error(f"Failed to read {pf.name}: {e}")
                continue

            n_rows = len(df)
            if max_per_chunk:
                n_rows = min(n_rows, max_per_chunk)

            yielded = 0
            for idx in tqdm(range(n_rows), desc=f"Parsing {pf.stem}"):
                try:
                    row = df.iloc[idx]

                    # Extract wavenumber and intensity
                    wavenumbers = np.asarray(row.get("Frequency(cm^-1)",
                                                      row.get("frequency", None)))
                    intensities = np.asarray(row.get("ir_spectra",
                                                      row.get("IR_spectra", None)))

                    if wavenumbers is None or intensities is None:
                        continue

                    wavenumbers = wavenumbers.ravel().astype(np.float64)
                    intensities = intensities.ravel().astype(np.float64)

                    if len(wavenumbers) < 100 or len(intensities) < 100:
                        continue

                    # Crop to chemically relevant range (400-4000 cm^-1)
                    mask = (wavenumbers >= 400.0) & (wavenumbers <= 4000.0)
                    wn = wavenumbers[mask]
                    ints = intensities[mask]

                    if len(wn) < 100:
                        continue

                    # Normalize to 2048 points
                    normalized = normalize_spectrum(wn, ints)
                    if normalized is None:
                        continue

                    smiles = str(row.get("smiles", row.get("id", idx)))

                    yield SpectrumRecord(
                        spectrum=normalized,
                        source="uspto",
                        spec_type="ir",
                        original_range=(float(wn.min()), float(wn.max())),
                        sample_id=f"uspto_{pf.stem}_{smiles[:50]}"
                    )
                    yielded += 1

                except Exception as e:
                    logger.debug(f"Failed to parse USPTO row {idx}: {e}")
                    continue

            logger.info(f"Yielded {yielded} spectra from {pf.name}")


# ═══════════════════════════════════════════════════════════════════════
# OpenSpecy Downloader
# ═══════════════════════════════════════════════════════════════════════

class OpenSpecyDownloader:
    """Download and parse OpenSpecy experimental Raman+FTIR reference library from OSF.

    Data: RDS files containing wavenumber vectors and spectral matrices.
    """

    # OSF direct download URLs
    RAMAN_URL = "https://osf.io/download/jdwx9/"
    FTIR_URL = "https://osf.io/download/cxrsq/"
    RAMAN_META_URL = "https://osf.io/download/zyxnk/"
    FTIR_META_URL = "https://osf.io/download/z27yr/"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> List[Path]:
        """Download OpenSpecy RDS files."""
        downloaded = []

        for url, fname, desc in [
            (self.RAMAN_URL, "raman_library.rds", "OpenSpecy Raman (~48 MB)"),
            (self.FTIR_URL, "ftir_library.rds", "OpenSpecy FTIR (~6 MB)"),
            (self.RAMAN_META_URL, "raman_metadata.rds", "Raman metadata"),
            (self.FTIR_META_URL, "ftir_metadata.rds", "FTIR metadata"),
        ]:
            dest = self.cache_dir / fname
            if _download_file(url, dest, desc):
                downloaded.append(dest)

        return downloaded

    def _parse_rds(self, rds_path: Path, spec_type: str) -> Generator[SpectrumRecord, None, None]:
        """Parse an OpenSpecy RDS library file.

        Handles the long-format DataFrame with columns:
            wavenumber, intensity, sample_name, group
        Each sample_name is a unique spectrum.
        """
        try:
            import rdata
            import pandas as pd
        except ImportError:
            logger.error("rdata and pandas required for OpenSpecy.")
            return

        logger.info(f"Parsing {rds_path.name}...")

        try:
            parsed = rdata.parser.parse_file(str(rds_path))
            constructor_dict = {
                **rdata.conversion.DEFAULT_CLASS_MAP,
                "data.table": lambda obj, attrs: pd.DataFrame(obj),
                "data.frame": lambda obj, attrs: pd.DataFrame(obj),
            }
            converted = rdata.conversion.convert(parsed, constructor_dict=constructor_dict)
        except Exception as e:
            logger.error(f"Failed to parse {rds_path.name}: {e}")
            return

        # Handle long-format DataFrame: (wavenumber, intensity, sample_name, group)
        if isinstance(converted, pd.DataFrame) and 'wavenumber' in converted.columns and 'intensity' in converted.columns:
            df = converted
            sample_col = 'sample_name' if 'sample_name' in df.columns else 'group'
            unique_samples = df[sample_col].unique()
            n_spectra = len(unique_samples)
            logger.info(f"Found {n_spectra} spectra in long-format DataFrame from {rds_path.name}")

            yielded = 0
            for sample_name in tqdm(unique_samples, desc=f"Parsing {spec_type}"):
                try:
                    sub = df[df[sample_col] == sample_name]
                    wavenumbers = sub['wavenumber'].values.astype(np.float64)
                    intensities = sub['intensity'].values.astype(np.float64)

                    if len(wavenumbers) < 10:
                        continue

                    normalized = normalize_spectrum(wavenumbers, intensities)
                    if normalized is None:
                        continue

                    yield SpectrumRecord(
                        spectrum=normalized,
                        source="openspecy",
                        spec_type=spec_type,
                        original_range=(float(wavenumbers.min()), float(wavenumbers.max())),
                        sample_id=f"openspecy_{spec_type}_{sample_name}"
                    )
                    yielded += 1
                except Exception as e:
                    logger.debug(f"Failed to parse spectrum {sample_name}: {e}")
                    continue

            logger.info(f"Yielded {yielded} {spec_type} spectra from {rds_path.name}")
            return

        # Fallback: dict-based OpenSpecy object with wavenumber/spectra/metadata
        if isinstance(converted, dict):
            wavenumber = converted.get('wavenumber')
            spectra = converted.get('spectra')

            if wavenumber is not None and spectra is not None:
                wavenumber = np.asarray(wavenumber, dtype=np.float64).ravel()

                if isinstance(spectra, pd.DataFrame):
                    n = spectra.shape[1]
                    logger.info(f"Found {n} spectra (wide format) in {rds_path.name}")
                    yielded = 0
                    for col_idx in tqdm(range(n), desc=f"Parsing {spec_type}"):
                        try:
                            intensities = spectra.iloc[:, col_idx].values.astype(np.float64)
                            if len(intensities) != len(wavenumber):
                                continue
                            normalized = normalize_spectrum(wavenumber, intensities)
                            if normalized is None:
                                continue
                            col_name = str(spectra.columns[col_idx])
                            yield SpectrumRecord(
                                spectrum=normalized,
                                source="openspecy",
                                spec_type=spec_type,
                                original_range=(float(wavenumber.min()), float(wavenumber.max())),
                                sample_id=f"openspecy_{spec_type}_{col_name}"
                            )
                            yielded += 1
                        except Exception:
                            continue
                    logger.info(f"Yielded {yielded} {spec_type} spectra")
                    return

        logger.error(f"Unrecognized data format from {rds_path.name}: {type(converted)}")

    def get_spectra(self) -> Generator[SpectrumRecord, None, None]:
        """Download and yield all OpenSpecy spectra."""
        self.download()

        raman_path = self.cache_dir / "raman_library.rds"
        ftir_path = self.cache_dir / "ftir_library.rds"

        if raman_path.exists():
            yield from self._parse_rds(raman_path, "raman")

        if ftir_path.exists():
            yield from self._parse_rds(ftir_path, "ftir")


# ═══════════════════════════════════════════════════════════════════════
# RRUFF Downloader (improved from existing code)
# ═══════════════════════════════════════════════════════════════════════

class RRUFFDownloader:
    """Download and parse RRUFF mineral Raman+IR spectra.

    Improved version with longer timeouts and better error handling.
    """

    RAMAN_URL = "https://www.rruff.net/zipped_data_files/raman/LR-Raman.zip"
    IR_URL = "https://www.rruff.net/zipped_data_files/infrared/RAW.zip"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_and_extract_zip(self, url: str, name: str) -> Optional[Path]:
        """Download and extract a ZIP file with retries."""
        import zipfile
        import requests

        extract_dir = self.cache_dir / name
        zip_path = self.cache_dir / f"{name}.zip"

        if extract_dir.exists() and any(extract_dir.rglob("*.txt")):
            n_files = len(list(extract_dir.rglob("*.txt")))
            logger.info(f"Using cached RRUFF {name}: {n_files} files")
            return extract_dir

        # Delete corrupt cached zip if it exists (e.g. HTML page from wrong URL)
        if zip_path.exists():
            if not zipfile.is_zipfile(zip_path):
                logger.warning(f"Cached {zip_path.name} is not a valid ZIP, deleting for re-download")
                zip_path.unlink()

        if not _download_file(url, zip_path, f"RRUFF {name}", timeout=600):
            return None

        if not zipfile.is_zipfile(zip_path):
            logger.error(f"Downloaded {zip_path.name} is not a valid ZIP file")
            zip_path.unlink()
            return None

        try:
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            logger.info(f"Extracted RRUFF {name}")
            return extract_dir
        except Exception as e:
            logger.error(f"Failed to extract {name}: {e}")
            return None

    def _parse_rruff_file(self, filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        """Parse a RRUFF spectrum file (two-column format with ## headers)."""
        try:
            wavenumbers = []
            intensities = []
            mineral_name = filepath.stem

            with open(filepath, 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('##'):
                        continue

                    parts = line.split(',') if ',' in line else line.split()
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
        """Download and yield all RRUFF spectra."""
        for url, name, spec_type in [
            (self.RAMAN_URL, "raman", "raman"),
            (self.IR_URL, "infrared", "ir"),
        ]:
            extract_dir = self._download_and_extract_zip(url, name)
            if extract_dir is None:
                logger.warning(f"Skipping RRUFF {name} (download/extract failed)")
                continue

            spectrum_files = (list(extract_dir.rglob("*.txt")) +
                              list(extract_dir.rglob("*.rruff")))
            logger.info(f"Found {len(spectrum_files)} RRUFF {name} files")

            yielded = 0
            for filepath in tqdm(spectrum_files, desc=f"Parsing RRUFF {name}"):
                result = self._parse_rruff_file(filepath)
                if result is None:
                    continue

                wn, intensity, mineral_name = result
                normalized = normalize_spectrum(wn, intensity)

                if normalized is not None:
                    yield SpectrumRecord(
                        spectrum=normalized,
                        source="rruff",
                        spec_type=spec_type,
                        original_range=(float(wn.min()), float(wn.max())),
                        sample_id=f"rruff_{mineral_name}_{filepath.stem}"
                    )
                    yielded += 1

            logger.info(f"Yielded {yielded} RRUFF {name} spectra")
