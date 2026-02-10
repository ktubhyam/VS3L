#!/usr/bin/env python3
"""
SpectralFM v2: Test-Time Training Evaluation

The MOST IMPORTANT experiment: zero-shot calibration transfer via TTT.

No classical method can operate at zero labeled samples.
If TTT achieves R²>0.3 at zero-shot → new paradigm.

Experiments:
1. Zero-shot TTT: adapt using MSRP on unlabeled target spectra, then predict
2. Few-shot TTT+LoRA: TTT adaptation + LoRA fine-tuning with N samples
3. TTT ablation: vary n_steps, adapt_layers, mask_ratio, lr

Usage:
    # Zero-shot evaluation
    python scripts/run_ttt.py --checkpoint checkpoints/best_pretrain.pt \
        --dataset corn --source m5 --target mp6 --property moisture

    # TTT + few-shot sweep
    python scripts/run_ttt.py --checkpoint checkpoints/best_pretrain.pt \
        --dataset corn --sweep

    # TTT ablation
    python scripts/run_ttt.py --checkpoint checkpoints/best_pretrain.pt \
        --dataset corn --ablation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import copy
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from src.config import SpectralFMConfig
from src.models.spectral_fm import SpectralFM
from src.models.lora import inject_lora
from src.data.datasets import SpectralPreprocessor
from src.evaluation.baselines import compute_metrics, run_baseline_comparison
from scripts.run_finetune import (
    load_corn_data, load_tablet_data, preprocess_spectra,
    finetune_spectral_fm, evaluate_model,
)


def load_pretrained_model(checkpoint_path, device="cpu"):
    """Load pretrained SpectralFM from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", SpectralFMConfig())
    model = SpectralFM(config)

    state_dict = ckpt["model_state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[6:] if k.startswith("model.") else k] = v
    model.load_state_dict(cleaned, strict=False)

    return model, config


def run_ttt(model, unlabeled_spectra, n_steps=20, lr=1e-4,
            mask_ratio=0.15, adapt_layers="norm", device="cpu"):
    """Apply Test-Time Training to adapt model to new instrument.

    Args:
        model: SpectralFM (will be modified in-place)
        unlabeled_spectra: (N, 2048) preprocessed unlabeled target spectra
        n_steps: TTT gradient steps
        lr: TTT learning rate
        mask_ratio: MSRP masking ratio
        adapt_layers: which layers to adapt ("norm", "lora", "all")
    """
    model.to(device)
    unlabeled_t = unlabeled_spectra.to(device)

    model.test_time_train(
        unlabeled_t, n_steps=n_steps, lr=lr,
        mask_ratio=mask_ratio, adapt_layers=adapt_layers,
    )
    return model


def evaluate_zero_shot_ttt(checkpoint_path, data, device,
                           ttt_steps_list=None, ttt_lr=1e-4,
                           mask_ratio=0.15, adapt_layers="norm"):
    """Evaluate zero-shot TTT at different step counts.

    This is THE key experiment. We:
    1. Load pretrained model
    2. Adapt only using unlabeled target spectra (MSRP self-supervision)
    3. Predict property values on target test spectra
    4. No labeled target data used at all!
    """
    if ttt_steps_list is None:
        ttt_steps_list = [0, 5, 10, 20, 50, 100]

    # Preprocess all target spectra (unlabeled pool + test)
    X_target_all = np.concatenate([data["X_target_train"], data["X_target_test"]])
    X_target_all_t = preprocess_spectra(X_target_all, data.get("wavelengths"))
    X_test_t = preprocess_spectra(data["X_target_test"], data.get("wavelengths"))
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)

    # Also preprocess source test for "no transfer" baseline
    X_source_test_t = preprocess_spectra(data["X_source_test"], data.get("wavelengths"))

    results = {}

    for n_steps in ttt_steps_list:
        # Fresh copy each time
        model, config = load_pretrained_model(checkpoint_path, device)
        model.to(device)

        if n_steps > 0:
            run_ttt(model, X_target_all_t, n_steps=n_steps, lr=ttt_lr,
                    mask_ratio=mask_ratio, adapt_layers=adapt_layers, device=device)

        # Zero-shot prediction: encode target test spectra → predict
        model.eval()
        with torch.no_grad():
            # Predict on target test spectra
            output = model.predict(X_test_t.to(device), domain="NIR", mc_samples=10)

        preds = output["prediction"].cpu().numpy().flatten()
        uncertainty = output.get("uncertainty")
        if uncertainty is not None:
            uncertainty = uncertainty.cpu().numpy().flatten()

        metrics = compute_metrics(y_test.numpy(), preds, uncertainty)
        results[n_steps] = metrics

        print(f"  TTT steps={n_steps:>3d}: R²={metrics['r2']:.4f}, "
              f"RMSE={metrics['rmse']:.4f}, RPD={metrics['rpd']:.2f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def evaluate_ttt_plus_fewshot(checkpoint_path, data, device,
                              n_samples_list=None, n_seeds=3,
                              ttt_steps=20, ttt_lr=1e-4, ft_epochs=100):
    """Evaluate TTT + LoRA fine-tuning at different sample counts.

    Pipeline: pretrain → TTT (unlabeled) → LoRA fine-tune (N labeled samples)
    """
    if n_samples_list is None:
        n_samples_list = [0, 1, 3, 5, 10, 20, 30]

    wl = data.get("wavelengths")
    X_target_all = np.concatenate([data["X_target_train"], data["X_target_test"]])
    X_target_all_t = preprocess_spectra(X_target_all, wl)

    results = {}

    for n_samples in n_samples_list:
        seed_metrics = []

        for seed in range(n_seeds):
            # Fresh model
            model, config = load_pretrained_model(checkpoint_path, device)

            # Step 1: TTT adaptation (always, even at n=0)
            if ttt_steps > 0:
                run_ttt(model, X_target_all_t, n_steps=ttt_steps, lr=ttt_lr,
                        adapt_layers="norm", device=device)

            if n_samples == 0:
                # Zero-shot: just evaluate
                X_test_t = preprocess_spectra(data["X_target_test"], wl)
                y_test = torch.tensor(data["y_test"], dtype=torch.float32)
                metrics = evaluate_model(model, X_test_t, y_test, device=device)
            else:
                # Few-shot: LoRA fine-tune on N samples
                inject_lora(model, ["q_proj", "k_proj", "v_proj"],
                            rank=8, alpha=16)

                # Select N transfer samples
                rng = np.random.RandomState(42 + seed)
                n = min(n_samples, len(data["y_train"]))
                idx = rng.choice(len(data["y_train"]), n, replace=False)

                X_train_t = preprocess_spectra(data["X_source_train"][idx], wl)
                y_train_t = torch.tensor(data["y_train"][idx], dtype=torch.float32)
                X_test_t = preprocess_spectra(data["X_target_test"], wl)
                y_test = torch.tensor(data["y_test"], dtype=torch.float32)

                model, _ = finetune_spectral_fm(
                    model, X_train_t, y_train_t,
                    n_epochs=ft_epochs, lr=1e-4, device=device, use_lora=True,
                )
                metrics = evaluate_model(model, X_test_t, y_test, device=device)

            seed_metrics.append(metrics)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        r2s = [m["r2"] for m in seed_metrics]
        rmses = [m["rmse"] for m in seed_metrics]
        results[n_samples] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "all_metrics": seed_metrics,
        }
        print(f"  TTT+{n_samples:>2d} samples: R²={results[n_samples]['r2_mean']:.4f}"
              f"±{results[n_samples]['r2_std']:.4f}")

    return results


def run_ttt_ablation(checkpoint_path, data, device):
    """Ablation study on TTT hyperparameters.

    Varies: n_steps, adapt_layers, mask_ratio, lr
    """
    results = {}

    # 1. Vary n_steps
    print("\n--- Ablation: TTT Steps ---")
    for n_steps in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
        r = evaluate_zero_shot_ttt(
            checkpoint_path, data, device,
            ttt_steps_list=[n_steps], ttt_lr=1e-4,
        )
        results[f"steps_{n_steps}"] = r[n_steps]

    # 2. Vary adapt_layers
    print("\n--- Ablation: Adapt Layers ---")
    for layers in ["norm", "lora", "all"]:
        # Need LoRA injected for "lora" option
        model, config = load_pretrained_model(checkpoint_path, device)
        if layers == "lora":
            inject_lora(model, ["q_proj", "k_proj", "v_proj"], rank=8, alpha=16)
            # Save and reload to get consistent state
            tmp_path = "/tmp/ttt_ablation_lora.pt"
            torch.save({"model_state_dict": model.state_dict(), "config": config}, tmp_path)
            r = evaluate_zero_shot_ttt(
                tmp_path, data, device,
                ttt_steps_list=[20], adapt_layers=layers,
            )
        else:
            r = evaluate_zero_shot_ttt(
                checkpoint_path, data, device,
                ttt_steps_list=[20], adapt_layers=layers,
            )
        results[f"layers_{layers}"] = r[20]
        del model

    # 3. Vary mask_ratio
    print("\n--- Ablation: Mask Ratio ---")
    for mr in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        r = evaluate_zero_shot_ttt(
            checkpoint_path, data, device,
            ttt_steps_list=[20], mask_ratio=mr,
        )
        results[f"mask_{mr}"] = r[20]

    # 4. Vary learning rate
    print("\n--- Ablation: Learning Rate ---")
    for lr in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
        r = evaluate_zero_shot_ttt(
            checkpoint_path, data, device,
            ttt_steps_list=[20], ttt_lr=lr,
        )
        results[f"lr_{lr}"] = r[20]

    return results


def main():
    parser = argparse.ArgumentParser(description="SpectralFM TTT Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_pretrain.pt")
    parser.add_argument("--dataset", choices=["corn", "tablet"], default="corn")
    parser.add_argument("--source", type=str, default="m5")
    parser.add_argument("--target", type=str, default="mp6")
    parser.add_argument("--property", type=str, default="moisture")
    parser.add_argument("--sweep", action="store_true", help="TTT + few-shot sweep")
    parser.add_argument("--ablation", action="store_true", help="TTT ablation study")
    parser.add_argument("--ttt-steps", type=int, default=20)
    parser.add_argument("--ttt-lr", type=float, default=1e-4)
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    experiments_dir = project_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    CORN_PROPS = {"moisture": 0, "oil": 1, "protein": 2, "starch": 3}
    TABLET_PROPS = {"active_ingredient": 0, "weight": 1, "hardness": 2}
    prop_idx = CORN_PROPS.get(args.property, 0) if args.dataset == "corn" else TABLET_PROPS.get(args.property, 0)

    # Load data with full transfer set
    if args.dataset == "corn":
        data = load_corn_data(data_dir, args.source, args.target, prop_idx,
                              n_transfer=30, seed=42)
    else:
        data = load_tablet_data(data_dir, prop_idx, n_transfer=30, seed=42)

    ckpt_path = project_dir / args.checkpoint

    if args.ablation:
        print("\n=== TTT Ablation Study ===")
        results = run_ttt_ablation(ckpt_path, data, device)
        out_name = "ttt_ablation"
    elif args.sweep:
        print("\n=== TTT + Few-Shot Sweep ===")
        results = evaluate_ttt_plus_fewshot(
            ckpt_path, data, device, n_seeds=args.n_seeds,
            ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr,
        )
        out_name = "ttt_sweep"

        # Also run baselines
        baseline_results = run_baseline_comparison(
            data["X_source_train"], data["X_target_train"],
            data["X_source_test"], data["X_target_test"],
            data["y_train"], data["y_test"],
        )
        results = {"ttt_fewshot": results, "baselines": baseline_results}
    else:
        print("\n=== Zero-Shot TTT Evaluation ===")
        results = evaluate_zero_shot_ttt(
            ckpt_path, data, device,
            ttt_steps_list=[0, 5, 10, 20, 50, 100],
            ttt_lr=args.ttt_lr,
        )
        out_name = "ttt_zeroshot"

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "device": device,
        "results": results,
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    out_path = experiments_dir / f"{out_name}_{args.dataset}_{args.property}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
