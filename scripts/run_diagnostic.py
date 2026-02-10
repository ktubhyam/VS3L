#!/usr/bin/env python3
"""
SpectralFM v2: Diagnostic Experiment

MUST RUN FIRST before investing in full pretraining (50K steps, 2-6 hours).

Quick test (~20 minutes on GPU):
1. Pretrain ONLY on corn instruments m5 + mp5 (no modality gap!)
2. Apply TTT to instrument mp6 with zero labels
3. If R² > 0.0 → architecture works, proceed to full pretraining
4. If R² < 0.0 → debug before wasting GPU hours

This bypasses the modality gap issue (corpus is 96% Raman, evaluation is NIR)
and directly tests: does the pretrain→TTT→predict pipeline work at all?

Usage:
    python scripts/run_diagnostic.py [--max-steps 2000] [--device cuda]
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
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path
from datetime import datetime

from src.config import SpectralFMConfig
from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
from src.losses.losses import SpectralFMPretrainLoss
from src.data.datasets import SpectralPreprocessor, SpectralAugmentor
from src.evaluation.baselines import compute_metrics, run_baseline_comparison
from src.training.trainer import PretrainTrainer


class CornPretrainDataset(Dataset):
    """Simple dataset for pretraining on corn spectra from specific instruments."""

    def __init__(self, data_dir, instruments=("m5", "mp5"), target_length=2048,
                 augment=True):
        corn_dir = Path(data_dir) / "processed" / "corn"
        self.preprocessor = SpectralPreprocessor(target_length=target_length)
        self.augmentor = SpectralAugmentor() if augment else None
        wavelengths = np.load(corn_dir / "wavelengths.npy")

        spectra_list = []
        inst_ids = []
        for i, inst in enumerate(instruments):
            raw = np.load(corn_dir / f"{inst}_spectra.npy")  # (80, 700)
            for s in raw:
                processed = self.preprocessor.process(s, wavelengths)["normalized"]
                spectra_list.append(processed)
                inst_ids.append(i)

        self.spectra = np.array(spectra_list, dtype=np.float32)
        self.inst_ids = np.array(inst_ids, dtype=np.int64)
        self.n_samples = len(self.spectra)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        spectrum = self.spectra[idx].copy()
        if self.augmentor is not None:
            spectrum = self.augmentor.augment(spectrum, p=0.5)

        return {
            "spectrum": torch.tensor(spectrum, dtype=torch.float32),
            "instrument_id": torch.tensor(self.inst_ids[idx], dtype=torch.long),
            "domain": "NIR",
        }


def run_diagnostic(args):
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    experiments_dir = project_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print("=" * 60)
    print("SpectralFM DIAGNOSTIC EXPERIMENT")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Pretrain instruments: m5 + mp5 ({80*2}=160 spectra)")
    print(f"Test instrument: mp6 (zero-shot via TTT)")
    print(f"Max steps: {args.max_steps}")
    print()

    # ========== Step 1: Pretrain on m5 + mp5 ==========
    print("--- Step 1: Pretraining on m5 + mp5 ---")
    t0 = time.time()

    config = SpectralFMConfig()
    config.pretrain.max_steps = args.max_steps
    config.pretrain.batch_size = args.batch_size
    config.device = device

    dataset = CornPretrainDataset(data_dir, instruments=("m5", "mp5"))
    # 90/10 split
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    model = SpectralFM(config)
    pretrain_model = SpectralFMForPretraining(model, config)

    trainer = PretrainTrainer(
        pretrain_model, config, train_loader, val_loader,
        use_wandb=False, run_name="diagnostic",
    )
    history = trainer.train(
        max_steps=args.max_steps,
        log_every=max(1, args.max_steps // 20),
        val_every=max(1, args.max_steps // 5),
        save_every=args.max_steps,  # Only save at end
    )

    pretrain_time = time.time() - t0
    print(f"\nPretraining completed in {pretrain_time:.0f}s")
    final_loss = history[-1]["total"] if history else float('nan')
    print(f"Final loss: {final_loss:.4f}")

    # Save diagnostic checkpoint
    diag_ckpt = project_dir / "checkpoints" / "diagnostic_pretrain.pt"
    diag_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": pretrain_model.state_dict(),
        "config": config,
        "step": trainer.step,
    }, diag_ckpt)

    # ========== Step 2: Zero-shot TTT on mp6 ==========
    print("\n--- Step 2: Zero-shot TTT on mp6 ---")

    # Load mp6 data
    corn_dir = data_dir / "processed" / "corn"
    mp6_spectra = np.load(corn_dir / "mp6_spectra.npy")  # (80, 700)
    m5_spectra = np.load(corn_dir / "m5_spectra.npy")
    properties = np.load(corn_dir / "properties.npy")  # (80, 4)
    wavelengths = np.load(corn_dir / "wavelengths.npy")

    preprocessor = SpectralPreprocessor(target_length=2048)
    mp6_processed = np.stack([
        preprocessor.process(s, wavelengths)["normalized"] for s in mp6_spectra
    ])
    m5_processed = np.stack([
        preprocessor.process(s, wavelengths)["normalized"] for s in m5_spectra
    ])

    mp6_t = torch.tensor(mp6_processed, dtype=torch.float32)
    m5_t = torch.tensor(m5_processed, dtype=torch.float32)
    moisture = properties[:, 0]  # Use moisture as test property

    # Split: use all 80 mp6 spectra as unlabeled, test on all
    # (This is zero-shot — we don't use any mp6 labels for adaptation)

    ttt_results = {}
    for n_steps in [0, 5, 10, 20, 50]:
        # Fresh model copy
        model_copy = SpectralFM(config)
        # Load pretrained weights (strip wrapper prefix)
        sd = {k.replace("model.", ""): v
              for k, v in pretrain_model.state_dict().items()
              if k.startswith("model.")}
        model_copy.load_state_dict(sd, strict=False)
        model_copy.to(device)

        if n_steps > 0:
            model_copy.test_time_train(
                mp6_t.to(device), n_steps=n_steps, lr=1e-4,
                mask_ratio=0.15, adapt_layers="norm",
            )

        # Predict
        model_copy.eval()
        with torch.no_grad():
            output = model_copy.predict(mp6_t.to(device), domain="NIR", mc_samples=5)

        preds = output["prediction"].cpu().numpy().flatten()
        metrics = compute_metrics(moisture, preds)
        ttt_results[n_steps] = metrics

        print(f"  TTT steps={n_steps:>3d}: R²={metrics['r2']:.4f}, "
              f"RMSE={metrics['rmse']:.4f}, RPD={metrics['rpd']:.2f}")

        del model_copy

    # ========== Step 3: Classical baselines for comparison ==========
    print("\n--- Step 3: Classical baselines (m5→mp6, 30 transfer samples) ---")

    rng = np.random.RandomState(42)
    idx = rng.permutation(80)
    train_idx, test_idx = idx[:30], idx[30:]

    baseline_results = run_baseline_comparison(
        m5_spectra[train_idx], mp6_spectra[train_idx],
        m5_spectra[test_idx], mp6_spectra[test_idx],
        moisture[train_idx], moisture[test_idx],
    )

    print(f"\n{'Method':<20} {'R²':>8} {'RMSE':>10}")
    print("-" * 40)
    for method, m in baseline_results.items():
        print(f"{method:<20} {m['r2']:>8.4f} {m['rmsep']:>10.4f}")

    # ========== Step 4: Diagnosis ==========
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)

    best_ttt_r2 = max(m["r2"] for m in ttt_results.values())
    ds_r2 = baseline_results.get("DS", {}).get("r2", 0)

    print(f"  Pretrain loss:     {final_loss:.4f}")
    print(f"  Zero-shot TTT R²: {ttt_results[0]['r2']:.4f}")
    print(f"  Best TTT R²:      {best_ttt_r2:.4f} (at {max(ttt_results, key=lambda k: ttt_results[k]['r2'])} steps)")
    print(f"  DS baseline R²:   {ds_r2:.4f} (30 labeled samples)")
    print()

    if best_ttt_r2 > 0.0:
        print("  ✅ PASS: TTT produces positive R² → architecture works!")
        print("  → Proceed to full pretraining (50K steps on full corpus)")
        if best_ttt_r2 > 0.3:
            print("  🎉 EXCELLENT: R²>0.3 at zero-shot → paper-worthy result!")
    else:
        print("  ⚠️  WARNING: TTT R² ≤ 0 → architecture may need debugging")
        print("  Possible issues:")
        print("    1. Pretraining too short (try more steps)")
        print("    2. TTT learning rate too high/low")
        print("    3. Model not learning useful representations")
        print("  → Run with --max-steps 5000 before full pretraining")

    # Save everything
    output = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "pretrain_steps": args.max_steps,
        "pretrain_time_sec": pretrain_time,
        "pretrain_final_loss": float(final_loss),
        "ttt_results": {str(k): v for k, v in ttt_results.items()},
        "baseline_results": baseline_results,
        "diagnosis": {
            "best_ttt_r2": float(best_ttt_r2),
            "ds_r2": float(ds_r2),
            "pass": best_ttt_r2 > 0.0,
        },
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    out_path = experiments_dir / "diagnostic_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nFull results saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="SpectralFM Diagnostic")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_diagnostic(args)


if __name__ == "__main__":
    main()
