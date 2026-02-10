#!/usr/bin/env python3
"""
SpectralFM v2: Run Pretraining

Pretrains the SpectralFM model on the HDF5 corpus using MSRP + multi-loss.

Usage:
    python scripts/run_pretrain.py [--max-steps 5000] [--batch-size 32]

For quick testing:
    python scripts/run_pretrain.py --max-steps 100 --batch-size 16 --light
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime

from src.config import SpectralFMConfig, get_light_config
from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
from src.data.datasets import PretrainHDF5Dataset
from src.training.trainer import PretrainTrainer


def main():
    parser = argparse.ArgumentParser(description="Pretrain SpectralFM")
    parser.add_argument('--max-steps', type=int, default=5000,
                        help='Maximum training steps (default: 5000)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--corpus', type=str, default=None,
                        help='Path to HDF5 corpus (default: data/pretrain/spectral_corpus_v2.h5)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint directory (default: checkpoints/)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--light', action='store_true',
                        help='Use lightweight model config (faster for testing)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit training samples (for quick debugging)')
    parser.add_argument('--log-every', type=int, default=50,
                        help='Log every N steps')
    parser.add_argument('--save-every', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--val-every', type=int, default=500,
                        help='Validate every N steps (default: 500)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='Enable W&B logging (default: on)')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='Disable W&B logging')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity (team/username)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Experiment run name')
    args = parser.parse_args()

    # Set up paths
    project_dir = Path(__file__).parent.parent
    corpus_path = Path(args.corpus) if args.corpus else project_dir / "data" / "pretrain" / "spectral_corpus_v2.h5"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else project_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = project_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Verify corpus exists
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Run 'python scripts/build_corpus_v2.py' first to build the corpus.")
        sys.exit(1)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Configuration
    if args.light:
        print("Using lightweight model configuration")
        cfg = get_light_config()
    else:
        cfg = SpectralFMConfig()

    cfg.device = device
    cfg.pretrain.batch_size = args.batch_size
    cfg.pretrain.lr = args.lr
    cfg.pretrain.max_steps = args.max_steps
    cfg.checkpoint_dir = str(checkpoint_dir)
    cfg.log_dir = str(log_dir)

    print(f"\n{'='*60}")
    print("SpectralFM Pretraining")
    print(f"{'='*60}")
    print(f"Corpus: {corpus_path}")
    print(f"Corpus size: {corpus_path.stat().st_size / 1e6:.1f} MB")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model config: {'light' if args.light else 'full'}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # Build data loaders with 90/10 validation split
    print("\nLoading data...")
    full_dataset = PretrainHDF5Dataset(
        str(corpus_path),
        augment=True,
        max_samples=args.max_samples,
    )
    print(f"Total samples: {len(full_dataset):,}")

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Batches per epoch: {len(train_loader):,}")

    # Create model
    print("\nCreating model...")
    base_model = SpectralFM(cfg)
    model = SpectralFMForPretraining(base_model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    # Create trainer
    trainer = PretrainTrainer(
        model=model,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        use_wandb=args.wandb,
        run_name=args.run_name,
        wandb_entity=args.wandb_entity,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print(f"\n{'='*60}")
    print("Starting pretraining...")
    print(f"{'='*60}\n")

    try:
        history = trainer.train(
            max_steps=args.max_steps,
            log_every=args.log_every,
            val_every=args.val_every,
            save_every=args.save_every,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint(checkpoint_dir / "pretrain_interrupted.pt")
        history = trainer.history

    # Save training log
    log_path = log_dir / f"pretrain_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump({
            "config": {
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "light_model": args.light,
                "corpus": str(corpus_path),
            },
            "history": history,
            "final_step": trainer.step,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nTraining log saved to {log_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Pretraining Complete!")
    print(f"{'='*60}")
    print(f"Final step: {trainer.step}")
    if history:
        final_loss = history[-1].get('total', 'N/A')
        print(f"Final loss: {final_loss}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
