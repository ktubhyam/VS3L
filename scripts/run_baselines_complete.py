#!/usr/bin/env python3
"""
SpectralFM v2: Comprehensive Baseline Runner

Runs ALL classical baselines on ALL properties × ALL instrument pairs,
with multiple seeds for standard deviations.

Output: experiments/baselines_complete.json — the full Table 1 for the paper.

Usage:
    python scripts/run_baselines_complete.py
    python scripts/run_baselines_complete.py --dataset corn
    python scripts/run_baselines_complete.py --dataset tablet
    python scripts/run_baselines_complete.py --n-seeds 10
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import combinations

from src.evaluation.baselines import (
    PLSCalibration, PDS, SBC, DS, CCA, DiPLS,
    compute_metrics, run_baseline_comparison, print_results_table
)


def load_corn_transfer(data_dir, source_inst, target_inst, prop_idx,
                       n_transfer=30, seed=42):
    """Load corn data for a specific instrument pair and property."""
    corn_dir = Path(data_dir) / "processed" / "corn"
    PROPERTIES = ["moisture", "oil", "protein", "starch"]

    source_spectra = np.load(corn_dir / f"{source_inst}_spectra.npy")  # (80, 700)
    target_spectra = np.load(corn_dir / f"{target_inst}_spectra.npy")  # (80, 700)
    properties = np.load(corn_dir / "properties.npy")  # (80, 4)
    wavelengths = np.load(corn_dir / "wavelengths.npy")  # (700,)

    y = properties[:, prop_idx]

    rng = np.random.RandomState(seed)
    n_total = len(y)
    indices = rng.permutation(n_total)
    train_idx = indices[:n_transfer]
    test_idx = indices[n_transfer:]

    return {
        "X_source_train": source_spectra[train_idx],
        "X_target_train": target_spectra[train_idx],
        "X_source_test": source_spectra[test_idx],
        "X_target_test": target_spectra[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "property": PROPERTIES[prop_idx],
        "source": source_inst,
        "target": target_inst,
    }


def load_tablet_transfer(data_dir, prop_idx=0, n_transfer=30, seed=42):
    """Load tablet data for spectrometer 1 → 2 transfer."""
    tablet_dir = Path(data_dir) / "processed" / "tablet"
    PROPERTIES = ["active_ingredient", "weight", "hardness"]

    cal1 = np.load(tablet_dir / "calibrate_1.npy")  # (155, 650)
    cal2 = np.load(tablet_dir / "calibrate_2.npy")  # (155, 650)
    calY = np.load(tablet_dir / "calibrate_Y.npy")  # (155, 3)

    test1 = np.load(tablet_dir / "test_1.npy")
    test2 = np.load(tablet_dir / "test_2.npy")
    testY = np.load(tablet_dir / "test_Y.npy")

    y_cal = calY[:, prop_idx]
    y_test = testY[:, prop_idx]

    # Use first n_transfer calibration samples as transfer standards
    rng = np.random.RandomState(seed)
    transfer_idx = rng.choice(len(y_cal), min(n_transfer, len(y_cal)), replace=False)

    return {
        "X_source_train": cal1[transfer_idx],
        "X_target_train": cal2[transfer_idx],
        "X_source_test": test1,
        "X_target_test": test2,
        "y_train": y_cal[transfer_idx],
        "y_test": y_test,
        "property": PROPERTIES[prop_idx],
        "source": "spec_1",
        "target": "spec_2",
    }


def run_single_experiment(data, n_transfer):
    """Run all baselines on a single (property, instrument_pair, seed) combo."""
    return run_baseline_comparison(
        X_source_train=data["X_source_train"],
        X_target_train=data["X_target_train"],
        X_source_test=data["X_source_test"],
        X_target_test=data["X_target_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
    )


def aggregate_seeds(all_results):
    """Aggregate results across seeds: mean ± std for each metric."""
    methods = set()
    for r in all_results:
        methods.update(r.keys())

    aggregated = {}
    for method in sorted(methods):
        metrics_lists = {}
        for r in all_results:
            if method not in r or "error" in r[method]:
                continue
            for k, v in r[method].items():
                if k in ("n_components", "error"):
                    continue
                metrics_lists.setdefault(k, []).append(v)

        if not metrics_lists:
            aggregated[method] = {"error": "all seeds failed"}
            continue

        agg = {}
        for k, vals in metrics_lists.items():
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
                agg[f"{k}_n"] = len(vals)
            else:
                agg[f"{k}_mean"] = float('nan')
                agg[f"{k}_std"] = float('nan')
        aggregated[method] = agg

    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["corn", "tablet", "all"], default="all")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-transfer", type=int, default=30)
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    experiments_dir = project_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    all_experiments = {}

    # ========== CORN ==========
    if args.dataset in ("corn", "all"):
        INSTRUMENTS = ["m5", "mp5", "mp6"]
        PROPERTIES = ["moisture", "oil", "protein", "starch"]
        pairs = list(combinations(INSTRUMENTS, 2))
        # Also add reverse direction for each pair
        directed_pairs = []
        for a, b in pairs:
            directed_pairs.append((a, b))
            directed_pairs.append((b, a))

        for prop_idx, prop_name in enumerate(PROPERTIES):
            for source, target in directed_pairs:
                key = f"corn_{source}_to_{target}_{prop_name}"
                print(f"\n{'='*60}")
                print(f"Running: {key}")
                print(f"{'='*60}")

                seed_results = []
                for seed in range(args.n_seeds):
                    data = load_corn_transfer(
                        data_dir, source, target, prop_idx,
                        n_transfer=args.n_transfer, seed=42 + seed
                    )
                    results = run_single_experiment(data, args.n_transfer)
                    seed_results.append(results)

                    if seed == 0:
                        print_results_table(results, f"{key} (seed 0)")

                aggregated = aggregate_seeds(seed_results)
                all_experiments[key] = {
                    "dataset": "corn",
                    "source": source,
                    "target": target,
                    "property": prop_name,
                    "n_transfer": args.n_transfer,
                    "n_seeds": args.n_seeds,
                    "results": aggregated,
                }

    # ========== TABLET ==========
    if args.dataset in ("tablet", "all"):
        PROPERTIES = ["active_ingredient", "weight", "hardness"]

        for prop_idx, prop_name in enumerate(PROPERTIES):
            for direction in [("spec_1", "spec_2"), ("spec_2", "spec_1")]:
                source, target = direction
                key = f"tablet_{source}_to_{target}_{prop_name}"
                print(f"\n{'='*60}")
                print(f"Running: {key}")
                print(f"{'='*60}")

                seed_results = []
                for seed in range(args.n_seeds):
                    if source == "spec_1":
                        data = load_tablet_transfer(
                            data_dir, prop_idx,
                            n_transfer=args.n_transfer, seed=42 + seed
                        )
                    else:
                        # Reverse: spec_2 → spec_1
                        data = load_tablet_transfer(
                            data_dir, prop_idx,
                            n_transfer=args.n_transfer, seed=42 + seed
                        )
                        # Swap source and target
                        data["X_source_train"], data["X_target_train"] = \
                            data["X_target_train"], data["X_source_train"]
                        data["X_source_test"], data["X_target_test"] = \
                            data["X_target_test"], data["X_source_test"]
                        data["source"], data["target"] = target, source

                    results = run_single_experiment(data, args.n_transfer)
                    seed_results.append(results)

                    if seed == 0:
                        print_results_table(results, f"{key} (seed 0)")

                aggregated = aggregate_seeds(seed_results)
                all_experiments[key] = {
                    "dataset": "tablet",
                    "source": source,
                    "target": target,
                    "property": prop_name,
                    "n_transfer": args.n_transfer,
                    "n_seeds": args.n_seeds,
                    "results": aggregated,
                }

    # ========== SUMMARY ==========
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_seeds": args.n_seeds,
        "n_transfer": args.n_transfer,
        "n_experiments": len(all_experiments),
        "experiments": all_experiments,
    }

    # Save dataset-specific + combined
    suffix = args.dataset if args.dataset != "all" else "all"
    out_path = experiments_dir / f"baselines_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    # Also always write combined
    combined_path = experiments_dir / "baselines_complete.json"
    # Merge with existing if different dataset
    if combined_path.exists() and suffix != "all":
        with open(combined_path) as f:
            existing = json.load(f)
        existing.get("experiments", {}).update(all_experiments)
        existing["n_experiments"] = len(existing["experiments"])
        existing["timestamp"] = datetime.now().isoformat()
        with open(combined_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        with open(combined_path, "w") as f:
            json.dump(output, f, indent=2)
    print(f"\n\n{'='*60}")
    print(f"All results saved to {out_path}")
    print(f"Combined results in {combined_path}")
    print(f"Total experiments: {len(all_experiments)}")

    # Print summary table for paper
    print(f"\n\n{'='*80}")
    print("PAPER TABLE 1: Baseline Comparison (mean ± std across seeds)")
    print(f"{'='*80}")

    # Collect method names
    method_names = set()
    for exp in all_experiments.values():
        method_names.update(exp["results"].keys())
    method_names = sorted(method_names)

    # Print per-dataset summary
    for dataset in ["corn", "tablet"]:
        exps = {k: v for k, v in all_experiments.items() if v["dataset"] == dataset}
        if not exps:
            continue

        print(f"\n--- {dataset.upper()} Dataset ---")
        print(f"{'Experiment':<35} ", end="")
        for m in method_names:
            print(f"{m:>12}", end="")
        print()
        print("-" * (35 + 12 * len(method_names)))

        for key, exp in sorted(exps.items()):
            label = f"{exp['source']}→{exp['target']} {exp['property']}"
            print(f"{label:<35} ", end="")
            for m in method_names:
                r = exp["results"].get(m, {})
                mean = r.get("r2_mean", float('nan'))
                std = r.get("r2_std", 0)
                if np.isnan(mean):
                    print(f"{'N/A':>12}", end="")
                else:
                    print(f"{mean:>6.3f}±{std:.3f}", end="")
            print()


if __name__ == "__main__":
    main()
