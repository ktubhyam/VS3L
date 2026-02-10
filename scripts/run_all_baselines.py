#!/usr/bin/env python3
"""
SpectralFM v2: Comprehensive Baseline Runner

Runs ALL classical baselines across:
- Corn: 4 properties × 3 instrument pairs × {PLS, PDS, SBC, DS, CCA, di-PLS}
- Tablet: 3 properties × 1 instrument pair × {PLS, PDS, SBC, DS, CCA, di-PLS}

Produces the complete Table 1 for the paper with mean ± std over 5 random splits.

Usage:
    python scripts/run_all_baselines.py [--n-repeats 5] [--n-transfer 30]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from src.evaluation.baselines import (
    PLSCalibration, PDS, SBC, DS, CCA, DiPLS,
    compute_metrics, run_baseline_comparison, print_results_table
)


def load_corn_transfer(data_dir, source_inst, target_inst, prop_idx,
                       n_transfer, seed):
    """Load corn data for a specific instrument pair and property."""
    corn_dir = Path(data_dir) / "processed" / "corn"
    PROPERTIES = ["moisture", "oil", "protein", "starch"]

    src = np.load(corn_dir / f"{source_inst}_spectra.npy")   # (80, 700)
    tgt = np.load(corn_dir / f"{target_inst}_spectra.npy")   # (80, 700)
    props = np.load(corn_dir / "properties.npy")              # (80, 4)

    y = props[:, prop_idx]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(y))
    train_idx = indices[:n_transfer]
    test_idx = indices[n_transfer:]

    return {
        "X_source_train": src[train_idx],
        "X_target_train": tgt[train_idx],
        "X_source_test": src[test_idx],
        "X_target_test": tgt[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "property": PROPERTIES[prop_idx],
        "source": source_inst,
        "target": target_inst,
    }


def load_tablet_transfer(data_dir, prop_idx, n_transfer, seed):
    """Load tablet data for a specific property."""
    tab_dir = Path(data_dir) / "processed" / "tablet"
    PROPERTIES = ["active_ingredient", "weight", "hardness"]

    cal1 = np.load(tab_dir / "calibrate_1.npy")   # (155, 650)
    cal2 = np.load(tab_dir / "calibrate_2.npy")   # (155, 650)
    calY = np.load(tab_dir / "calibrate_Y.npy")   # (155, 3)

    test1 = np.load(tab_dir / "test_1.npy")       # (N_test, 650)
    test2 = np.load(tab_dir / "test_2.npy")
    testY = np.load(tab_dir / "test_Y.npy")

    y_cal = calY[:, prop_idx]
    y_test = testY[:, prop_idx]

    # Use n_transfer from calibration set for fitting transfer
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


def run_single_experiment(data, pls_components=None):
    """Run all baselines on a single transfer experiment."""
    return run_baseline_comparison(
        X_source_train=data["X_source_train"],
        X_target_train=data["X_target_train"],
        X_source_test=data["X_source_test"],
        X_target_test=data["X_target_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        pls_components=pls_components,
    )


def aggregate_results(all_results):
    """Aggregate results over multiple repeats → mean ± std."""
    methods = set()
    metrics_keys = set()
    for r in all_results:
        methods.update(r.keys())
        for m in r.values():
            metrics_keys.update(k for k in m.keys()
                                if isinstance(m.get(k), (int, float))
                                and k not in ("n_components",))

    agg = {}
    for method in sorted(methods):
        agg[method] = {}
        for metric in sorted(metrics_keys):
            vals = [r[method].get(metric, float('nan'))
                    for r in all_results if method in r]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                agg[method][f"{metric}_mean"] = float(np.mean(vals))
                agg[method][f"{metric}_std"] = float(np.std(vals))
            else:
                agg[method][f"{metric}_mean"] = float('nan')
                agg[method][f"{metric}_std"] = float('nan')
    return agg


def print_aggregated_table(agg, title=""):
    """Pretty-print aggregated results."""
    if title:
        print(f"\n{title}")
        print("=" * 85)

    print(f"{'Method':<15} {'R² (mean±std)':>18} {'RMSEP':>18} {'RPD':>14} {'Bias':>14}")
    print("-" * 85)

    for method in agg:
        r2_m = agg[method].get("r2_mean", float('nan'))
        r2_s = agg[method].get("r2_std", float('nan'))
        rmsep_m = agg[method].get("rmsep_mean", float('nan'))
        rmsep_s = agg[method].get("rmsep_std", float('nan'))
        rpd_m = agg[method].get("rpd_mean", float('nan'))
        rpd_s = agg[method].get("rpd_std", float('nan'))
        bias_m = agg[method].get("bias_mean", float('nan'))
        bias_s = agg[method].get("bias_std", float('nan'))
        print(f"{method:<15} {r2_m:>7.4f}±{r2_s:<7.4f}  {rmsep_m:>7.4f}±{rmsep_s:<7.4f}"
              f"  {rpd_m:>5.2f}±{rpd_s:<5.2f}  {bias_m:>5.4f}±{bias_s:<5.4f}")

    print("-" * 85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-transfer", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corn-only", action="store_true")
    parser.add_argument("--tablet-only", action="store_true")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    out_dir = project_dir / "experiments" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    CORN_PROPERTIES = ["moisture", "oil", "protein", "starch"]
    CORN_PAIRS = [("m5", "mp5"), ("m5", "mp6"), ("mp5", "mp6")]
    TABLET_PROPERTIES = ["active_ingredient", "weight", "hardness"]

    all_experiments = {}

    # ==================== CORN ====================
    if not args.tablet_only:
        print("\n" + "=" * 85)
        print("CORN NIR CALIBRATION TRANSFER BASELINES")
        print("=" * 85)

        for prop_idx, prop_name in enumerate(CORN_PROPERTIES):
            for src, tgt in CORN_PAIRS:
                exp_key = f"corn_{src}_{tgt}_{prop_name}"
                print(f"\n>>> {exp_key} (n_transfer={args.n_transfer}, {args.n_repeats} repeats)")

                repeat_results = []
                for rep in range(args.n_repeats):
                    seed = args.seed + rep * 1000
                    data = load_corn_transfer(data_dir, src, tgt, prop_idx,
                                              args.n_transfer, seed)
                    results = run_single_experiment(data)
                    repeat_results.append(results)
                    print(f"  repeat {rep+1}: DS R²={results['DS']['r2']:.4f}, "
                          f"PDS R²={results['PDS']['r2']:.4f}")

                agg = aggregate_results(repeat_results)
                all_experiments[exp_key] = {
                    "dataset": "corn",
                    "source": src,
                    "target": tgt,
                    "property": prop_name,
                    "n_transfer": args.n_transfer,
                    "n_repeats": args.n_repeats,
                    "aggregated": agg,
                    "raw": repeat_results,
                }

                print_aggregated_table(agg, f"Corn {src}→{tgt} ({prop_name})")

    # ==================== TABLET ====================
    if not args.corn_only:
        print("\n" + "=" * 85)
        print("TABLET NIR CALIBRATION TRANSFER BASELINES")
        print("=" * 85)

        for prop_idx, prop_name in enumerate(TABLET_PROPERTIES):
            exp_key = f"tablet_spec1_spec2_{prop_name}"
            print(f"\n>>> {exp_key} (n_transfer={args.n_transfer}, {args.n_repeats} repeats)")

            repeat_results = []
            for rep in range(args.n_repeats):
                seed = args.seed + rep * 1000
                data = load_tablet_transfer(data_dir, prop_idx,
                                            args.n_transfer, seed)
                results = run_single_experiment(data)
                repeat_results.append(results)
                print(f"  repeat {rep+1}: DS R²={results['DS']['r2']:.4f}")

            agg = aggregate_results(repeat_results)
            all_experiments[exp_key] = {
                "dataset": "tablet",
                "source": "spec_1",
                "target": "spec_2",
                "property": prop_name,
                "n_transfer": args.n_transfer,
                "n_repeats": args.n_repeats,
                "aggregated": agg,
                "raw": repeat_results,
            }

            print_aggregated_table(agg, f"Tablet spec1→spec2 ({prop_name})")

    # ==================== SAVE ====================
    # Convert NaN to None for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj

    output = {
        "timestamp": datetime.now().isoformat(),
        "n_transfer": args.n_transfer,
        "n_repeats": args.n_repeats,
        "experiments": clean_for_json(all_experiments),
    }

    out_file = out_dir / "all_baselines.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ All results saved to {out_file}")

    # ==================== SUMMARY TABLE (Paper Table 1) ====================
    print("\n" + "=" * 85)
    print("PAPER TABLE 1: Summary (R² mean ± std)")
    print("=" * 85)

    # Collect corn averages across all pairs for each property
    methods = ["No_Transfer", "Target_Direct", "PDS", "SBC", "DS", "CCA", "di-PLS"]
    header = f"{'Property':<12} {'Pair':<10}"
    for m in methods:
        header += f" {m:>10}"
    print(header)
    print("-" * (22 + 11 * len(methods)))

    for exp_key, exp in all_experiments.items():
        agg = exp["aggregated"]
        pair = f"{exp['source']}→{exp['target']}"
        row = f"{exp['property']:<12} {pair:<10}"
        for m in methods:
            if m in agg:
                r2 = agg[m].get("r2_mean", float('nan'))
                row += f" {r2:>10.3f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print("\n✓ Complete baseline evaluation finished!")


if __name__ == "__main__":
    main()
