#!/usr/bin/env python3
"""
Build SpectralFM Pretraining Corpus v2

Downloads spectra from multiple public databases, normalizes them,
applies data augmentation, and saves to a unified HDF5 file.

Usage:
    python scripts/build_corpus_v2.py                      # full build, 3x augment
    python scripts/build_corpus_v2.py --augment-ratio 5    # 5x augment
    python scripts/build_corpus_v2.py --skip-download      # rebuild from cached downloads
    python scripts/build_corpus_v2.py --sources rruff openspecy  # specific sources only
    python scripts/build_corpus_v2.py --max-per-source 5000      # limit per source (for testing)
"""
import argparse
import logging
import sys
import os
import time
import numpy as np
import h5py
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.pretraining_pipeline import SpectrumRecord, SpectrumAugmentor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def collect_spectra(sources: list, cache_dir: Path, max_per_source: int = None,
                    skip_download: bool = False) -> list:
    """Collect real spectra from all requested sources."""
    from src.data.corpus_downloader import (
        ChEMBLDownloader, USPTODownloader, OpenSpecyDownloader, RRUFFDownloader
    )

    all_spectra = []
    source_counts = Counter()
    source_errors = {}

    # ── RRUFF ──
    if "rruff" in sources:
        logger.info("=" * 50)
        logger.info("SOURCE: RRUFF (experimental Raman + IR)")
        logger.info("=" * 50)
        try:
            dl = RRUFFDownloader(cache_dir / "rruff")
            count = 0
            for record in dl.get_spectra():
                all_spectra.append(record)
                count += 1
                if max_per_source and count >= max_per_source:
                    break
            source_counts["rruff"] = count
            logger.info(f"RRUFF: collected {count} spectra")
        except Exception as e:
            logger.error(f"RRUFF failed: {e}")
            source_errors["rruff"] = str(e)

    # ── OpenSpecy ──
    if "openspecy" in sources:
        logger.info("=" * 50)
        logger.info("SOURCE: OpenSpecy (experimental Raman + FTIR)")
        logger.info("=" * 50)
        try:
            dl = OpenSpecyDownloader(cache_dir / "openspecy")
            count = 0
            for record in dl.get_spectra():
                all_spectra.append(record)
                count += 1
                if max_per_source and count >= max_per_source:
                    break
            source_counts["openspecy"] = count
            logger.info(f"OpenSpecy: collected {count} spectra")
        except Exception as e:
            logger.error(f"OpenSpecy failed: {e}")
            source_errors["openspecy"] = str(e)

    # ── USPTO ──
    if "uspto" in sources:
        logger.info("=" * 50)
        logger.info("SOURCE: USPTO (computed IR, ~177K)")
        logger.info("=" * 50)
        try:
            # Start with 1 chunk for testing, expand later
            max_chunks = None
            if max_per_source and max_per_source <= 20000:
                max_chunks = 1
            dl = USPTODownloader(cache_dir / "uspto", max_chunks=max_chunks)
            count = 0
            max_per_chunk = max_per_source if max_per_source else None
            for record in dl.get_spectra(max_per_chunk=max_per_chunk):
                all_spectra.append(record)
                count += 1
                if max_per_source and count >= max_per_source:
                    break
            source_counts["uspto"] = count
            logger.info(f"USPTO: collected {count} spectra")
        except Exception as e:
            logger.error(f"USPTO failed: {e}")
            source_errors["uspto"] = str(e)

    # ── ChEMBL ──
    if "chembl" in sources:
        logger.info("=" * 50)
        logger.info("SOURCE: ChEMBL (computed Raman + IR, ~220K)")
        logger.info("=" * 50)
        try:
            dl = ChEMBLDownloader(cache_dir / "chembl",
                                   download_both_parts=(max_per_source is None or max_per_source > 50000))
            count = 0
            max_mol = max_per_source // 2 if max_per_source else None  # /2 since we get Raman+IR
            for record in dl.get_spectra(max_molecules_per_part=max_mol):
                all_spectra.append(record)
                count += 1
                if max_per_source and count >= max_per_source:
                    break
            source_counts["chembl"] = count
            logger.info(f"ChEMBL: collected {count} spectra")
        except Exception as e:
            logger.error(f"ChEMBL failed: {e}")
            source_errors["chembl"] = str(e)

    # ── Summary ──
    logger.info("=" * 50)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total real spectra: {len(all_spectra)}")
    for src, cnt in sorted(source_counts.items()):
        logger.info(f"  {src}: {cnt}")
    if source_errors:
        logger.warning("Failed sources:")
        for src, err in source_errors.items():
            logger.warning(f"  {src}: {err}")

    return all_spectra


def augment_spectra(real_spectra: list, augment_ratio: int, seed: int = 42) -> list:
    """Augment spectra at specified ratio."""
    if augment_ratio <= 0 or len(real_spectra) == 0:
        return real_spectra

    augmentor = SpectrumAugmentor(seed=seed)
    all_spectra = list(real_spectra)  # keep originals

    logger.info(f"Augmenting {len(real_spectra)} spectra at {augment_ratio}x ratio...")

    for record in real_spectra:
        aug_specs = augmentor.augment(record.spectrum, n_augmented=augment_ratio)
        for i, aug_spec in enumerate(aug_specs):
            all_spectra.append(SpectrumRecord(
                spectrum=aug_spec,
                source=f"{record.source}_aug",
                spec_type=record.spec_type,
                original_range=record.original_range,
                sample_id=f"{record.sample_id}_aug{i}"
            ))

    logger.info(f"Total after augmentation: {len(all_spectra)}")
    return all_spectra


def save_to_hdf5(spectra: list, output_path: Path, augment_ratio: int,
                 source_counts: dict):
    """Save spectra to HDF5 in the format expected by PretrainHDF5Dataset."""
    n = len(spectra)
    logger.info(f"Saving {n} spectra to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Spectra array
        spectra_data = np.stack([s.spectrum for s in spectra])
        f.create_dataset('spectra', data=spectra_data, dtype='float32',
                         chunks=(min(1000, n), 2048),
                         compression='gzip', compression_opts=4)

        # Metadata group
        meta = f.create_group('metadata')
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
        f.attrs['corpus_size'] = n
        f.attrs['target_points'] = 2048
        f.attrs['normalization'] = 'snv'
        f.attrs['augment_ratio'] = augment_ratio

        # Real counts per source
        real_mask = np.array(["_aug" not in s for s in [rec.source for rec in spectra]])
        f.attrs['n_real_total'] = int(real_mask.sum())
        for src, cnt in source_counts.items():
            f.attrs[f'n_real_{src}'] = cnt

        from datetime import datetime
        f.attrs['creation_date'] = datetime.now().isoformat()

    file_size = output_path.stat().st_size / 1e9
    logger.info(f"Saved: {output_path} ({file_size:.2f} GB)")


def validate_corpus(h5_path: Path) -> bool:
    """Run validation checks on the built corpus."""
    logger.info("Validating corpus...")

    with h5py.File(h5_path, 'r') as f:
        spectra = f['spectra']
        sources_raw = f['metadata/source'][:]
        types_raw = f['metadata/type'][:]

        n = spectra.shape[0]
        logger.info(f"Total spectra: {n:,}")

        # Shape check
        assert spectra.shape[1] == 2048, f"Wrong shape: {spectra.shape}"

        # NaN/Inf check (sample 5000)
        sample_idx = np.random.choice(n, min(5000, n), replace=False)
        sample = spectra[sorted(sample_idx)]
        assert np.all(np.isfinite(sample)), "NaN/Inf found in spectra!"

        # Real spectra count
        real_mask = np.array([b"aug" not in s for s in sources_raw])
        n_real = int(real_mask.sum())
        logger.info(f"Unique real spectra: {n_real:,}")

        # Source diversity
        unique_sources = set()
        for s in sources_raw[real_mask]:
            src = s.decode() if isinstance(s, bytes) else s
            unique_sources.add(src)
        logger.info(f"Real sources: {unique_sources}")

        # Modality breakdown
        type_counts = Counter()
        for t in types_raw:
            typ = t.decode() if isinstance(t, bytes) else t
            type_counts[typ] += 1
        logger.info("Modality breakdown:")
        for typ, cnt in sorted(type_counts.items()):
            logger.info(f"  {typ}: {cnt:,}")

        # Source breakdown (all)
        source_counts = Counter()
        for s in sources_raw:
            src = s.decode() if isinstance(s, bytes) else s
            source_counts[src] += 1
        logger.info("Source breakdown:")
        for src, cnt in sorted(source_counts.items()):
            logger.info(f"  {src}: {cnt:,}")

        # Validation thresholds
        if n < 50000:
            logger.warning(f"WARN: Only {n:,} total spectra (target: 50,000+)")
        if n_real < 1000:
            logger.error(f"FAIL: Only {n_real:,} real spectra (minimum: 1,000)")
            return False
        if len(unique_sources) < 2:
            logger.error(f"FAIL: Only {len(unique_sources)} source(s) (minimum: 2)")
            return False

        logger.info("VALIDATION PASSED")
        return True


def print_summary(h5_path: Path, real_counts: dict, augment_ratio: int):
    """Print final build summary."""
    with h5py.File(h5_path, 'r') as f:
        n = f['spectra'].shape[0]
        sources_raw = f['metadata/source'][:]
        types_raw = f['metadata/type'][:]

    real_mask = np.array([b"aug" not in s for s in sources_raw])
    n_real = int(real_mask.sum())

    source_counts = Counter()
    for s in sources_raw:
        src = s.decode() if isinstance(s, bytes) else s
        source_counts[src] += 1

    type_counts = Counter()
    for t in types_raw:
        typ = t.decode() if isinstance(t, bytes) else t
        type_counts[typ] += 1

    file_size = h5_path.stat().st_size / 1e9

    print("\n" + "=" * 50)
    print("CORPUS V2 BUILD COMPLETE")
    print("=" * 50)
    print(f"File: {h5_path}")
    print(f"Size: {file_size:.2f} GB")
    print(f"Total spectra: {n:,}")
    print(f"Unique real spectra: {n_real:,}")
    print(f"Augmentation ratio: {augment_ratio}x")
    print("\nSources:")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt:,}")
    print("\nModalities:")
    for typ, cnt in sorted(type_counts.items()):
        print(f"  {typ}: {cnt:,}")
    print(f"\nReady for pretraining: {'YES' if n_real >= 1000 else 'NO'}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Build SpectralFM pretraining corpus v2")
    parser.add_argument("--output", type=str, default="data/pretrain/spectral_corpus_v2.h5",
                        help="Output HDF5 path")
    parser.add_argument("--cache-dir", type=str, default="data/pretrain/downloads",
                        help="Download cache directory")
    parser.add_argument("--augment-ratio", type=int, default=3,
                        help="Augmentation multiplier per real spectrum")
    parser.add_argument("--sources", nargs="+",
                        default=["rruff", "openspecy", "uspto", "chembl"],
                        choices=["rruff", "openspecy", "uspto", "chembl"],
                        help="Which sources to download")
    parser.add_argument("--max-per-source", type=int, default=None,
                        help="Max spectra per source (for testing)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloads, rebuild from cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    output_path = project_root / args.output
    cache_dir = project_root / args.cache_dir

    start = time.time()

    # Step 1: Collect real spectra
    logger.info("Step 1: Collecting real spectra from all sources...")
    real_spectra = collect_spectra(
        sources=args.sources,
        cache_dir=cache_dir,
        max_per_source=args.max_per_source,
        skip_download=args.skip_download,
    )

    if len(real_spectra) == 0:
        logger.error("FATAL: No spectra collected from any source!")
        sys.exit(1)

    if len(real_spectra) < 1000:
        logger.error(f"FATAL: Only {len(real_spectra)} real spectra collected (minimum: 1,000)")
        logger.error("Check download logs above for errors.")
        sys.exit(1)

    # Count by source
    real_counts = Counter(r.source for r in real_spectra)

    # Check source diversity
    if len(real_counts) < 2:
        logger.error(f"FATAL: Only {len(real_counts)} source(s). Need at least 2.")
        sys.exit(1)

    # Step 2: Augment
    logger.info(f"\nStep 2: Augmenting at {args.augment_ratio}x ratio...")
    all_spectra = augment_spectra(real_spectra, args.augment_ratio, seed=args.seed)

    # Step 3: Save
    logger.info(f"\nStep 3: Saving to {output_path}...")
    save_to_hdf5(all_spectra, output_path, args.augment_ratio, dict(real_counts))

    # Step 4: Validate
    logger.info("\nStep 4: Validating...")
    valid = validate_corpus(output_path)

    elapsed = time.time() - start
    logger.info(f"\nTotal build time: {elapsed / 60:.1f} minutes")

    # Print summary
    print_summary(output_path, dict(real_counts), args.augment_ratio)

    if not valid:
        logger.error("Corpus validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
