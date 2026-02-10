#!/usr/bin/env python3
"""
SpectralFM v2: Master Experiment Runner

Orchestrates all experiments E1-E12 for the paper.
Each experiment can be run independently or all at once.

Usage:
    python scripts/run_experiments.py --experiment E3 --checkpoint checkpoints/best_pretrain.pt
    python scripts/run_experiments.py --all --checkpoint checkpoints/best_pretrain.pt
    python scripts/run_experiments.py --list  # Show experiment descriptions
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from src.config import SpectralFMConfig
from src.models.spectral_fm import SpectralFM
from src.models.lora import inject_lora
from src.evaluation.baselines import compute_metrics, run_baseline_comparison
from scripts.run_finetune import (
    load_corn_data, load_tablet_data, preprocess_spectra,
    finetune_spectral_fm, evaluate_model, run_single_transfer,
    run_sample_efficiency_sweep,
)
from scripts.run_ttt import (
    load_pretrained_model, run_ttt,
    evaluate_zero_shot_ttt, evaluate_ttt_plus_fewshot,
    run_ttt_ablation,
)


EXPERIMENTS = {
    "E1":  "Pretraining ablation (pretrained vs random init vs no pretrain)",
    "E2":  "LoRA vs full fine-tuning parameter efficiency",
    "E3":  "Sample efficiency curve (THE KEY FIGURE)",
    "E4":  "TTT step budget (0, 5, 10, 20, 50, 100 steps)",
    "E5":  "Cross-instrument generalization (train m5→mp5, test m5→mp6)",
    "E6":  "Cross-property transfer (train moisture, predict oil/protein/starch)",
    "E7":  "Corpus size scaling law (15K, 61K, ~415K, 1M+)",
    "E8":  "Physics loss ablation (remove each loss component)",
    "E9":  "Architecture ablation (no Mamba, no MoE, no Transformer, no VIB)",
    "E10": "VIB disentanglement visualization (t-SNE, MI, KL)",
    "E11": "Uncertainty calibration (MC Dropout coverage)",
    "E12": "Tablet dataset validation (second benchmark)",
}


def save_results(results, name, experiments_dir):
    out = {
        "timestamp": datetime.now().isoformat(),
        "experiment": name,
        "results": results,
    }
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    path = Path(experiments_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=convert)
    print(f"  Saved: {path}")
    return path


# ============================================================
# E1: Pretraining Ablation
# ============================================================
def run_E1(checkpoint_path, data_dir, device, experiments_dir, n_seeds=3):
    """Compare: pretrained + LoRA vs random init + LoRA vs random init + full FT."""
    print("\n" + "="*60)
    print("E1: Pretraining Ablation")
    print("="*60)

    results = {}
    data = load_corn_data(data_dir, "m5", "mp6", 0, n_transfer=30, seed=42)

    # A) Pretrained + LoRA
    print("\n  [A] Pretrained + LoRA")
    metrics_a, _ = run_single_transfer(checkpoint_path, data, device, use_lora=True)
    results["pretrained_lora"] = metrics_a
    print(f"    R² = {metrics_a['r2']:.4f}")

    # B) Random init + LoRA (no pretrain)
    print("\n  [B] Random Init + LoRA")
    config = SpectralFMConfig()
    model = SpectralFM(config)
    # Save random init as a "checkpoint"
    tmp_ckpt = Path(experiments_dir) / "tmp_random_init.pt"
    torch.save({"model_state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
                "config": config}, tmp_ckpt)
    metrics_b, _ = run_single_transfer(tmp_ckpt, data, device, use_lora=True)
    results["random_lora"] = metrics_b
    print(f"    R² = {metrics_b['r2']:.4f}")

    # C) Random init + Full fine-tuning
    print("\n  [C] Random Init + Full FT")
    metrics_c, _ = run_single_transfer(tmp_ckpt, data, device, use_lora=False, n_epochs=200)
    results["random_full_ft"] = metrics_c
    print(f"    R² = {metrics_c['r2']:.4f}")

    # D) Pretrained + Full fine-tuning
    print("\n  [D] Pretrained + Full FT")
    metrics_d, _ = run_single_transfer(checkpoint_path, data, device, use_lora=False, n_epochs=200)
    results["pretrained_full_ft"] = metrics_d
    print(f"    R² = {metrics_d['r2']:.4f}")

    tmp_ckpt.unlink(missing_ok=True)
    return save_results(results, "E1_pretrain_ablation", experiments_dir)


# ============================================================
# E2: LoRA vs Full Fine-Tuning
# ============================================================
def run_E2(checkpoint_path, data_dir, device, experiments_dir, n_seeds=3):
    """Compare LoRA ranks and full FT across sample sizes."""
    print("\n" + "="*60)
    print("E2: LoRA vs Full Fine-Tuning")
    print("="*60)

    results = {}
    for n_transfer in [5, 10, 20, 30]:
        data = load_corn_data(data_dir, "m5", "mp6", 0, n_transfer=n_transfer, seed=42)

        # LoRA rank sweep
        for rank in [2, 4, 8, 16]:
            metrics, _ = run_single_transfer(
                checkpoint_path, data, device, use_lora=True, lora_rank=rank)
            key = f"lora_r{rank}_n{n_transfer}"
            results[key] = metrics
            print(f"  LoRA r={rank}, n={n_transfer}: R²={metrics['r2']:.4f}")

        # Full FT
        metrics, _ = run_single_transfer(
            checkpoint_path, data, device, use_lora=False, n_epochs=200)
        results[f"full_ft_n{n_transfer}"] = metrics
        print(f"  Full FT, n={n_transfer}: R²={metrics['r2']:.4f}")

    return save_results(results, "E2_lora_vs_full", experiments_dir)


# ============================================================
# E3: Sample Efficiency (THE KEY FIGURE)
# ============================================================
def run_E3(checkpoint_path, data_dir, device, experiments_dir, n_seeds=5):
    """Sample efficiency: R² vs N for SpectralFM vs all baselines."""
    print("\n" + "="*60)
    print("E3: Sample Efficiency Curve (KEY FIGURE)")
    print("="*60)

    results = {}
    CORN_PROPS = {"moisture": 0, "oil": 1, "protein": 2, "starch": 3}

    for prop, idx in CORN_PROPS.items():
        print(f"\n  Property: {prop}")
        data_loader = lambda n, s: load_corn_data(data_dir, "m5", "mp6", idx, n, s)

        # SpectralFM sweep
        fm_results = run_sample_efficiency_sweep(
            checkpoint_path, data_loader, device,
            n_samples_list=[1, 3, 5, 10, 20, 30, 50],
            n_seeds=n_seeds,
        )

        # Baselines at each sample size
        baseline_sweeps = {}
        for n in [5, 10, 20, 30, 50]:
            seed_baselines = []
            for seed in range(n_seeds):
                data = data_loader(n, 42 + seed)
                bl = run_baseline_comparison(
                    data["X_source_train"], data["X_target_train"],
                    data["X_source_test"], data["X_target_test"],
                    data["y_train"], data["y_test"],
                )
                seed_baselines.append(bl)

            # Aggregate baselines across seeds
            methods = set()
            for sb in seed_baselines:
                methods.update(sb.keys())
            for method in methods:
                r2s = [sb[method]["r2"] for sb in seed_baselines
                       if method in sb and "error" not in sb[method]]
                if r2s:
                    baseline_sweeps.setdefault(method, {})[n] = {
                        "r2_mean": float(np.mean(r2s)),
                        "r2_std": float(np.std(r2s)),
                    }

        results[prop] = {
            "spectral_fm": fm_results,
            "baselines": baseline_sweeps,
        }

    return save_results(results, "E3_sample_efficiency", experiments_dir)


# ============================================================
# E4: TTT Step Budget
# ============================================================
def run_E4(checkpoint_path, data_dir, device, experiments_dir):
    """Zero-shot TTT: how many steps are needed?"""
    print("\n" + "="*60)
    print("E4: TTT Step Budget")
    print("="*60)

    data = load_corn_data(data_dir, "m5", "mp6", 0, n_transfer=30, seed=42)
    results = evaluate_zero_shot_ttt(
        checkpoint_path, data, device,
        ttt_steps_list=[0, 1, 2, 5, 10, 20, 50, 100, 200],
    )
    return save_results(results, "E4_ttt_steps", experiments_dir)


# ============================================================
# E5: Cross-Instrument Generalization
# ============================================================
def run_E5(checkpoint_path, data_dir, device, experiments_dir, n_seeds=3):
    """Train on m5→mp5, test on m5→mp6 (unseen target instrument)."""
    print("\n" + "="*60)
    print("E5: Cross-Instrument Generalization")
    print("="*60)

    results = {}
    INSTRUMENTS = ["m5", "mp5", "mp6"]

    for source in INSTRUMENTS:
        for target in INSTRUMENTS:
            if source == target:
                continue
            key = f"{source}_to_{target}"
            print(f"\n  {key}")

            data = load_corn_data(data_dir, source, target, 0, n_transfer=30, seed=42)

            # SpectralFM
            metrics, _ = run_single_transfer(checkpoint_path, data, device)
            results[f"fm_{key}"] = metrics
            print(f"    SpectralFM R² = {metrics['r2']:.4f}")

            # Best baseline
            bl = run_baseline_comparison(
                data["X_source_train"], data["X_target_train"],
                data["X_source_test"], data["X_target_test"],
                data["y_train"], data["y_test"],
            )
            results[f"baselines_{key}"] = bl

    return save_results(results, "E5_cross_instrument", experiments_dir)


# ============================================================
# E10: VIB Disentanglement
# ============================================================
def run_E10(checkpoint_path, data_dir, device, experiments_dir):
    """Extract z_chem and z_inst for t-SNE visualization."""
    print("\n" + "="*60)
    print("E10: VIB Disentanglement Analysis")
    print("="*60)

    model, config = load_pretrained_model(checkpoint_path, device)
    model.to(device).eval()

    corn_dir = Path(data_dir) / "processed" / "corn"
    wavelengths = np.load(corn_dir / "wavelengths.npy")
    properties = np.load(corn_dir / "properties.npy")

    all_z_chem, all_z_inst = [], []
    all_labels_chem, all_labels_inst = [], []

    for inst_idx, inst in enumerate(["m5", "mp5", "mp6"]):
        spectra = np.load(corn_dir / f"{inst}_spectra.npy")
        X = preprocess_spectra(spectra, wavelengths)

        with torch.no_grad():
            enc = model.encode(X.to(device), domain="NIR")

        all_z_chem.append(enc["z_chem"].cpu().numpy())
        all_z_inst.append(enc["z_inst"].cpu().numpy())
        all_labels_chem.append(properties[:, 0])  # moisture
        all_labels_inst.append(np.full(len(spectra), inst))

    results = {
        "z_chem": np.concatenate(all_z_chem).tolist(),
        "z_inst": np.concatenate(all_z_inst).tolist(),
        "labels_chem": np.concatenate(all_labels_chem).tolist(),
        "labels_inst": np.concatenate(all_labels_inst).tolist(),
    }

    # Generate visualization
    try:
        from src.evaluation.visualization import plot_tsne_disentanglement
        plot_tsne_disentanglement(
            np.concatenate(all_z_chem),
            np.concatenate(all_z_inst),
            np.concatenate(all_labels_chem),
            np.concatenate(all_labels_inst),
            chem_name="Moisture (%)",
            figures_dir=str(Path(experiments_dir).parent / "figures"),
        )
    except Exception as e:
        print(f"  Visualization failed: {e}")

    return save_results(results, "E10_disentanglement", experiments_dir)


# ============================================================
# E11: Uncertainty Calibration
# ============================================================
def run_E11(checkpoint_path, data_dir, device, experiments_dir):
    """MC Dropout uncertainty calibration analysis."""
    print("\n" + "="*60)
    print("E11: Uncertainty Calibration")
    print("="*60)

    data = load_corn_data(data_dir, "m5", "mp6", 0, n_transfer=30, seed=42)

    # Fine-tune model
    metrics, _ = run_single_transfer(checkpoint_path, data, device)

    # Now reload and evaluate with many MC samples
    model, config = load_pretrained_model(checkpoint_path, device)
    inject_lora(model, ["q_proj", "k_proj", "v_proj"], rank=8, alpha=16)

    X_train = preprocess_spectra(data["X_source_train"], data.get("wavelengths"))
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    X_test = preprocess_spectra(data["X_target_test"], data.get("wavelengths"))
    y_test = data["y_test"]

    model, _ = finetune_spectral_fm(model, X_train, y_train, device=device)

    # MC Dropout with 50 samples for good uncertainty
    output = model.predict(X_test.to(device), domain="NIR", mc_samples=50)

    y_pred = output["prediction"].cpu().numpy().flatten()
    uncertainty = output["uncertainty"].cpu().numpy().flatten()

    results = {
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "uncertainty": uncertainty.tolist(),
        "metrics": compute_metrics(y_test, y_pred, uncertainty),
    }

    # Generate visualization
    try:
        from src.evaluation.visualization import plot_calibration
        plot_calibration(
            np.array(y_test), y_pred, uncertainty,
            figures_dir=str(Path(experiments_dir).parent / "figures"),
        )
    except Exception as e:
        print(f"  Visualization failed: {e}")

    return save_results(results, "E11_calibration", experiments_dir)


# ============================================================
# E12: Tablet Dataset
# ============================================================
def run_E12(checkpoint_path, data_dir, device, experiments_dir, n_seeds=3):
    """Validation on tablet dataset (second benchmark)."""
    print("\n" + "="*60)
    print("E12: Tablet Dataset Validation")
    print("="*60)

    TABLET_PROPS = {"active_ingredient": 0, "weight": 1, "hardness": 2}
    results = {}

    for prop, idx in TABLET_PROPS.items():
        print(f"\n  Property: {prop}")
        data = load_tablet_data(data_dir, idx, n_transfer=30, seed=42)

        # SpectralFM
        metrics, _ = run_single_transfer(checkpoint_path, data, device)
        results[f"fm_{prop}"] = metrics
        print(f"    SpectralFM R² = {metrics['r2']:.4f}")

        # Baselines
        bl = run_baseline_comparison(
            data["X_source_train"], data["X_target_train"],
            data["X_source_test"], data["X_target_test"],
            data["y_train"], data["y_test"],
        )
        results[f"baselines_{prop}"] = bl

        for method, m in bl.items():
            r2 = m.get("r2", float("nan"))
            print(f"    {method}: R² = {r2:.4f}")

    return save_results(results, "E12_tablet", experiments_dir)


# ============================================================
# Main
# ============================================================

RUNNERS = {
    "E1": run_E1, "E2": run_E2, "E3": run_E3, "E4": run_E4,
    "E5": run_E5, "E10": run_E10, "E11": run_E11, "E12": run_E12,
}
# E6-E9 require special checkpoints (ablated models), handled in P4 prompt
GPU_HEAVY = {"E7"}  # Scaling requires multiple pretraining runs


def main():
    parser = argparse.ArgumentParser(description="SpectralFM Experiment Runner")
    parser.add_argument("--experiment", type=str, help="Run specific experiment (E1-E12)")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--list", action="store_true", help="List experiments")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_pretrain.pt")
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    if args.list:
        print("\nSpectralFM Experiments:")
        print("=" * 60)
        for k, v in EXPERIMENTS.items():
            gpu = " [GPU-HEAVY]" if k in GPU_HEAVY else ""
            impl = " ✓" if k in RUNNERS else " (manual)"
            print(f"  {k}: {v}{gpu}{impl}")
        return

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    experiments_dir = project_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = project_dir / args.checkpoint

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt}")

    if args.experiment:
        exp = args.experiment.upper()
        if exp in RUNNERS:
            RUNNERS[exp](ckpt, data_dir, device, experiments_dir, n_seeds=args.n_seeds)
        else:
            print(f"Experiment {exp} not implemented in auto-runner. See P4 prompt.")
    elif args.all:
        for exp, runner in RUNNERS.items():
            try:
                runner(ckpt, data_dir, device, experiments_dir, n_seeds=args.n_seeds)
            except Exception as e:
                print(f"\n  ⚠️ {exp} FAILED: {e}")
                import traceback
                traceback.print_exc()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
