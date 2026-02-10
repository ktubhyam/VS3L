#!/usr/bin/env python3
"""
SpectralFM v2: Fine-Tuning for Calibration Transfer

Loads a pretrained checkpoint, injects LoRA adapters, and fine-tunes
on calibration transfer tasks with varying numbers of transfer samples.

This is the core evaluation: SpectralFM (pretrained + LoRA) vs baselines.

Usage:
    # Fine-tune on corn m5→mp6 moisture with 10 transfer samples
    python scripts/run_finetune.py --checkpoint checkpoints/best_pretrain.pt \
        --dataset corn --source m5 --target mp6 --property moisture --n-transfer 10

    # Sample efficiency sweep (the key figure)
    python scripts/run_finetune.py --checkpoint checkpoints/best_pretrain.pt \
        --dataset corn --source m5 --target mp6 --property moisture --sweep

    # All properties
    python scripts/run_finetune.py --checkpoint checkpoints/best_pretrain.pt \
        --dataset corn --all-properties --sweep
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime

from src.config import SpectralFMConfig
from src.models.spectral_fm import SpectralFM
from src.models.lora import inject_lora, get_lora_state_dict, get_lora_optimizer_params
from src.data.datasets import SpectralPreprocessor
from src.evaluation.baselines import compute_metrics, run_baseline_comparison, print_results_table
from src.utils.logging import ExperimentLogger


# ============================================================
# Data Loading
# ============================================================

def load_corn_data(data_dir, source_inst, target_inst, prop_idx,
                   n_transfer, seed=42):
    """Load corn transfer data."""
    corn_dir = Path(data_dir) / "processed" / "corn"
    PROPERTIES = ["moisture", "oil", "protein", "starch"]

    source = np.load(corn_dir / f"{source_inst}_spectra.npy")
    target = np.load(corn_dir / f"{target_inst}_spectra.npy")
    props = np.load(corn_dir / "properties.npy")
    wl = np.load(corn_dir / "wavelengths.npy")

    y = props[:, prop_idx]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    train_idx = idx[:n_transfer]
    test_idx = idx[n_transfer:]

    return {
        "X_source_train": source[train_idx],
        "X_target_train": target[train_idx],
        "X_source_test": source[test_idx],
        "X_target_test": target[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "wavelengths": wl,
        "property": PROPERTIES[prop_idx],
    }


def load_tablet_data(data_dir, prop_idx, n_transfer, seed=42):
    """Load tablet transfer data."""
    tablet_dir = Path(data_dir) / "processed" / "tablet"
    PROPERTIES = ["active_ingredient", "weight", "hardness"]

    cal1 = np.load(tablet_dir / "calibrate_1.npy")
    cal2 = np.load(tablet_dir / "calibrate_2.npy")
    calY = np.load(tablet_dir / "calibrate_Y.npy")
    test1 = np.load(tablet_dir / "test_1.npy")
    test2 = np.load(tablet_dir / "test_2.npy")
    testY = np.load(tablet_dir / "test_Y.npy")

    rng = np.random.RandomState(seed)
    transfer_idx = rng.choice(len(calY), min(n_transfer, len(calY)), replace=False)

    return {
        "X_source_train": cal1[transfer_idx],
        "X_target_train": cal2[transfer_idx],
        "X_source_test": test1,
        "X_target_test": test2,
        "y_train": calY[transfer_idx, prop_idx],
        "y_test": testY[:, prop_idx],
        "wavelengths": None,
        "property": PROPERTIES[prop_idx],
    }


def preprocess_spectra(X, wavelengths=None, target_length=2048):
    """Preprocess raw spectra to model input format."""
    preprocessor = SpectralPreprocessor(target_length=target_length)
    processed = np.stack([
        preprocessor.process(s, wavelengths)["normalized"] for s in X
    ])
    return torch.tensor(processed, dtype=torch.float32)


# ============================================================
# Fine-Tuning Core
# ============================================================

def finetune_spectral_fm(model, X_train, y_train, X_val=None, y_val=None,
                         n_epochs=100, lr=1e-4, head_lr=1e-3,
                         batch_size=16, patience=20, device="cpu",
                         use_lora=True):
    """Fine-tune SpectralFM with LoRA on transfer samples.

    Args:
        model: SpectralFM with pretrained weights
        X_train: (N, 2048) preprocessed source spectra
        y_train: (N,) target values
        X_val, y_val: optional validation set
        n_epochs: max epochs
        lr: LoRA learning rate
        head_lr: regression head learning rate
        batch_size: batch size
        patience: early stopping patience
        device: cuda/cpu
        use_lora: whether to use LoRA (if False, fine-tune full model)

    Returns:
        fine-tuned model, training history
    """
    model = model.to(device)

    if use_lora:
        model.freeze_backbone()
        param_groups = get_lora_optimizer_params(model, lr=lr, head_lr=head_lr)
        # Also add norm layers (they're small and helpful)
        norm_params = [p for n, p in model.named_parameters()
                       if ("norm" in n or "ln" in n) and not p.requires_grad]
        for p in norm_params:
            p.requires_grad = True
        if norm_params:
            param_groups.append({"params": norm_params, "lr": lr * 0.1, "weight_decay": 0.0})
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    X_train_t = X_train.to(device)
    y_train_t = y_train.to(device)
    X_val_t = X_val.to(device) if X_val is not None else None
    y_val_t = y_val.to(device) if y_val is not None else None

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            x_batch = X_train_t[idx]
            y_batch = y_train_t[idx]

            optimizer.zero_grad()
            enc = model.encode(x_batch, domain="NIR")
            pred = model.regression_head(enc["z_chem"]).squeeze(-1)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_loss = float('nan')
        if X_val_t is not None:
            model.eval()
            with torch.no_grad():
                enc = model.encode(X_val_t, domain="NIR")
                pred = model.regression_head(enc["z_chem"]).squeeze(-1)
                val_loss = F.mse_loss(pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        else:
            # No val set — use train loss for best model
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_state = copy.deepcopy(model.state_dict())

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    if best_state is not None:
        model.load_state_dict(best_state)

    model.unfreeze_all()
    return model, history


def evaluate_model(model, X_test, y_test, device="cpu", mc_samples=10):
    """Evaluate fine-tuned model with MC dropout uncertainty."""
    model.to(device)
    X_test_t = X_test.to(device)

    output = model.predict(X_test_t, domain="NIR", mc_samples=mc_samples)

    preds = output["prediction"].cpu().numpy().flatten()
    uncertainty = output.get("uncertainty")
    if uncertainty is not None:
        uncertainty = uncertainty.cpu().numpy().flatten()

    return compute_metrics(y_test.numpy(), preds, uncertainty)


# ============================================================
# Experiment Runners
# ============================================================

def run_single_transfer(checkpoint_path, data, device, n_epochs=100,
                        lr=1e-4, use_lora=True, lora_rank=8):
    """Run SpectralFM fine-tuning for a single transfer experiment."""
    # Load pretrained model
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", SpectralFMConfig())
    model = SpectralFM(config)

    # Load pretrained weights (handle wrapper)
    state_dict = ckpt["model_state_dict"]
    # Strip 'model.' prefix if from SpectralFMForPretraining wrapper
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[6:]] = v
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)

    # Inject LoRA
    if use_lora:
        inject_lora(model, ["q_proj", "k_proj", "v_proj", "out_proj"],
                     rank=lora_rank, alpha=lora_rank * 2)

    # Preprocess
    X_train = preprocess_spectra(data["X_source_train"], data.get("wavelengths"))
    X_test = preprocess_spectra(data["X_target_test"], data.get("wavelengths"))
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)

    # Fine-tune
    model, history = finetune_spectral_fm(
        model, X_train, y_train,
        n_epochs=n_epochs, lr=lr, device=device, use_lora=use_lora,
    )

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, device=device)

    return metrics, history


def run_sample_efficiency_sweep(checkpoint_path, data_loader_fn, device,
                                n_samples_list=None, n_seeds=5,
                                n_epochs=100, use_lora=True):
    """Run sample efficiency experiment: R² vs N transfer samples.

    This generates the KEY FIGURE for the paper.
    """
    if n_samples_list is None:
        n_samples_list = [1, 3, 5, 10, 20, 30, 50]

    results = {}
    for n in n_samples_list:
        seed_metrics = []
        for seed in range(n_seeds):
            data = data_loader_fn(n_transfer=n, seed=42 + seed)

            if n < 3:
                # Very few samples — reduce epochs, increase patience
                epochs = 200
            else:
                epochs = n_epochs

            metrics, _ = run_single_transfer(
                checkpoint_path, data, device,
                n_epochs=epochs, use_lora=use_lora,
            )
            seed_metrics.append(metrics)
            print(f"  n={n}, seed={seed}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")

        # Aggregate
        r2s = [m["r2"] for m in seed_metrics]
        rmses = [m["rmse"] for m in seed_metrics]
        results[n] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "all_r2": [float(x) for x in r2s],
            "all_rmse": [float(x) for x in rmses],
        }
        print(f"  n={n}: R²={results[n]['r2_mean']:.4f}±{results[n]['r2_std']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SpectralFM Fine-Tuning")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_pretrain.pt")
    parser.add_argument("--dataset", choices=["corn", "tablet"], default="corn")
    parser.add_argument("--source", type=str, default="m5")
    parser.add_argument("--target", type=str, default="mp6")
    parser.add_argument("--property", type=str, default="moisture")
    parser.add_argument("--n-transfer", type=int, default=30)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Run sample efficiency sweep")
    parser.add_argument("--all-properties", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    experiments_dir = project_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    CORN_PROPS = {"moisture": 0, "oil": 1, "protein": 2, "starch": 3}
    TABLET_PROPS = {"active_ingredient": 0, "weight": 1, "hardness": 2}

    properties = ([args.property] if not args.all_properties else
                  list(CORN_PROPS.keys()) if args.dataset == "corn" else
                  list(TABLET_PROPS.keys()))

    all_results = {}

    for prop in properties:
        print(f"\n{'='*60}")
        print(f"Property: {prop}")
        print(f"{'='*60}")

        prop_idx = CORN_PROPS[prop] if args.dataset == "corn" else TABLET_PROPS[prop]

        if args.dataset == "corn":
            data_loader = lambda n_transfer, seed: load_corn_data(
                data_dir, args.source, args.target, prop_idx, n_transfer, seed)
        else:
            data_loader = lambda n_transfer, seed: load_tablet_data(
                data_dir, prop_idx, n_transfer, seed)

        if args.sweep:
            # Sample efficiency sweep
            sweep_results = run_sample_efficiency_sweep(
                project_dir / args.checkpoint,
                data_loader, device,
                n_seeds=args.n_seeds,
                n_epochs=args.n_epochs,
                use_lora=not args.no_lora,
            )

            # Also run baselines for comparison
            data = data_loader(n_transfer=30, seed=42)
            baseline_results = run_baseline_comparison(
                data["X_source_train"], data["X_target_train"],
                data["X_source_test"], data["X_target_test"],
                data["y_train"], data["y_test"],
            )

            all_results[prop] = {
                "sweep": sweep_results,
                "baselines": baseline_results,
            }

            # Print comparison
            print(f"\nSample Efficiency: {prop}")
            print(f"{'N':>5} {'SpectralFM R²':>15} {'DS R²':>10}")
            print("-" * 35)
            for n, r in sorted(sweep_results.items()):
                ds_r2 = baseline_results.get("DS", {}).get("r2", float('nan'))
                print(f"{n:>5} {r['r2_mean']:>8.4f}±{r['r2_std']:.4f} {ds_r2:>10.4f}")
        else:
            # Single experiment
            data = data_loader(args.n_transfer, seed=42)
            metrics, history = run_single_transfer(
                project_dir / args.checkpoint, data, device,
                n_epochs=args.n_epochs, lr=args.lr,
                use_lora=not args.no_lora, lora_rank=args.lora_rank,
            )

            # Baselines for comparison
            baseline_results = run_baseline_comparison(
                data["X_source_train"], data["X_target_train"],
                data["X_source_test"], data["X_target_test"],
                data["y_train"], data["y_test"],
            )
            baseline_results["SpectralFM"] = metrics

            all_results[prop] = baseline_results
            print_results_table(baseline_results, f"Transfer: {args.source}→{args.target} {prop}")

    # Save results
    out_name = args.run_name or f"finetune_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "device": device,
        "results": all_results,
    }

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = experiments_dir / f"{out_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
