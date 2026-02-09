#!/usr/bin/env python3
"""
SpectralFM v2: Build Pretraining Corpus

Main script to download, process, and build the HDF5 pretraining corpus.

Usage:
    python src/data/build_pretrain_corpus.py [--target-size 200000]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from pathlib import Path

from src.data.pretraining_pipeline import PretrainingCorpusBuilder


def main():
    parser = argparse.ArgumentParser(description="Build SpectralFM pretraining corpus")
    parser.add_argument('--target-size', type=int, default=200000,
                        help='Target number of spectra (default: 200000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HDF5 path (default: data/pretrain/spectral_corpus.h5)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache directory for downloads (default: data/pretrain/cache)')
    args = parser.parse_args()

    # Set up paths
    project_dir = Path(__file__).parent.parent.parent
    output_path = Path(args.output) if args.output else project_dir / "data" / "pretrain" / "spectral_corpus.h5"
    cache_dir = Path(args.cache_dir) if args.cache_dir else project_dir / "data" / "pretrain" / "cache"

    print(f"Output: {output_path}")
    print(f"Cache: {cache_dir}")
    print(f"Target size: {args.target_size:,} spectra")
    print()

    # Build corpus
    builder = PretrainingCorpusBuilder(
        output_path=output_path,
        cache_dir=cache_dir,
        target_points=2048
    )

    success = builder.build(target_total=args.target_size)

    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: Pretraining corpus built!")
        print("=" * 60)
        print(f"Output file: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")
    else:
        print("\nFAILED: Could not build corpus")
        sys.exit(1)


if __name__ == "__main__":
    main()
