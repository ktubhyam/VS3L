# VS³L

**Toward Standard-Free Calibration Transfer in Vibrational Spectroscopy via Self-Supervised Learning**

> **V**ibrational **S**pectroscopy via **S**elf-**S**upervised **L**earning

A self-supervised foundation model for vibrational spectroscopy (NIR, IR, Raman) that learns to disentangle chemical content from instrument-specific artifacts. VS³L combines a hybrid Mamba-Transformer backbone with Sinkhorn-based optimal transport domain adaptation, a variational information bottleneck, and physics-informed regularization to achieve calibration transfer across spectrometers using 10 or fewer labeled samples — where classical methods require 30-60.

> **Author:** Tubhyam Karthikeyan (ICT Mumbai / InvyrAI)
>
> **Target Journal:** Analytical Chemistry (ACS, IF 7.4)

---

## The Problem

Calibration transfer in spectroscopy has relied on the same approaches for 30+ years: measure 10-60 transfer samples on both instruments, then apply PDS/DS/SBC correction. At $50-200 per reference analysis, this is expensive and must be repeated for every new instrument. After four decades, the field still lacks a scalable solution.

## Our Approach

VS³L proposes a **fifth strategy** for calibration transfer — after instrument matching, global modeling, model updating, and sensor selection (Workman & Mark, 2017): learn instrument-invariant chemical representations from a large pretraining corpus, then adapt with minimal transfer data.

- **Standard-free (TTT):** Test-time training on unlabeled spectra from the new instrument — no paired standards needed
- **Sample-efficient (LoRA):** Fine-tuning with as few as 5-10 transfer samples
- **Target:** 10 transfer samples outperforms classical methods using 50

## Architecture

```
Spectrum (B, 2048)
  -> WaveletEmbedding     DWT multi-scale + Conv1d patching + wavenumber PE + [CLS] + [DOMAIN]
  -> MambaBackbone         4 selective SSM blocks, O(n) complexity
  -> MixtureOfExperts      4 experts, top-2 gating, optional KAN activations
  -> TransformerEncoder    2 blocks, 8 heads, global reasoning
  -> VIBHead               disentangle z_chem (128d) + z_inst (64d)
  -> Heads                 Reconstruction | Regression | FNO Transfer
```

**Design rationale:**
- **Mamba (O(n))** — selective state space model for long-range spectral dependencies without quadratic cost
- **Transformer (O(n^2))** — global self-attention for expressiveness where it matters
- **Wavelet decomposition** — DWT separates sharp absorption peaks (detail coefficients) from baselines (approximation coefficients)
- **VIB disentanglement** — variational information bottleneck splits the latent space into transferable chemistry (z_chem) and discardable instrument signature (z_inst)
- **Optimal transport** — Sinkhorn divergence aligns latent distributions across instruments
- **Physics-informed losses** — Beer-Lambert linearity, spectral smoothness, non-negativity, peak shape constraints

## Project Structure

```
VS3L/
├── run.py                              # Entry point (pretrain / finetune / evaluate / ttt)
├── src/
│   ├── config.py                       # All hyperparameters (dataclass-based)
│   ├── models/
│   │   ├── embedding.py                # WaveletEmbedding + WavenumberPE
│   │   ├── mamba.py                    # Pure PyTorch selective SSM
│   │   ├── moe.py                      # Mixture of Experts + KAN layers
│   │   ├── transformer.py              # Lightweight TransformerEncoder
│   │   ├── heads.py                    # VIB, Reconstruction, Regression, FNO heads
│   │   └── spectral_fm.py              # Full model assembly + TTT
│   ├── losses/
│   │   └── losses.py                   # MSRP, contrastive, physics, OT, VIB, MoE losses
│   ├── training/
│   │   └── trainer.py                  # Pretrain + finetune + TTT training loops
│   ├── evaluation/
│   │   ├── metrics.py                  # R2, RMSEP, RPD, bias, conformal prediction
│   │   └── baselines.py               # PDS, SBC, DS classical baselines
│   └── data/
│       ├── datasets.py                 # Data loading, augmentation, preprocessing
│       ├── build_pretrain_corpus.py     # Download + preprocess pretraining data
│       └── pretraining_pipeline.py      # Pretraining dataset class
├── scripts/
│   ├── run_baselines.py                # Run classical baseline comparison
│   └── run_finetune_test.py            # Fine-tuning validation script
├── data/
│   ├── raw/                            # Original .mat files
│   └── processed/                      # Preprocessed .npy arrays
│       ├── corn/                       # 80 samples x 3 instruments x 700 channels
│       └── tablet/                     # 655 samples x 2 instruments x 650 channels
├── experiments/                        # Experiment results (JSON)
├── checkpoints/                        # Saved model weights
├── figures/                            # Generated plots
├── paper/                              # Research notes and brainstorms
├── requirements.txt
├── PROJECT_STATUS.md                   # Current state and known issues
└── IMPLEMENTATION_PLAN.md              # Detailed task breakdown
```

## Datasets

### Evaluation (preprocessed, included)

| Dataset | Samples | Instruments | Channels | Properties |
|---------|---------|-------------|----------|------------|
| **Corn** | 80 | 3 (m5, mp5, mp6) | 700 | moisture, oil, protein, starch |
| **Tablet** | 655 | 2 | 650 | active ingredient, weight, hardness |

### Pretraining Corpus (to be downloaded)

| Source | Spectra | Modality |
|--------|---------|----------|
| ChEMBL IR-Raman | ~220K | IR, Raman |
| USPTO-Spectra | ~177K | Mixed |
| NIST WebBook | ~5.2K | IR |
| RRUFF | ~8.6K | Raman |
| **Total** | **~411K** | |

## Installation

```bash
git clone https://github.com/ktubhyam/VS3L.git
cd VS3L
pip install -r requirements.txt

# Optional: CUDA Mamba kernels (Linux + CUDA only)
pip install mamba-ssm>=1.2.0
```

**Requirements:** Python 3.10+, PyTorch 2.0+, ~40GB VRAM for full pretraining (A100 recommended). CPU mode available for development.

## Usage

```bash
# Smoke test — verify forward/backward pass
python run.py --mode smoke_test

# Classical baselines on corn dataset
python scripts/run_baselines.py

# Pretrain on spectral corpus
python run.py --mode pretrain

# LoRA fine-tune for calibration transfer
python run.py --mode finetune --checkpoint checkpoints/pretrain_best.pt

# Standard-free transfer via test-time training
python run.py --mode ttt --checkpoint checkpoints/pretrain_best.pt
```

## Baselines

| Method | Type | Description |
|--------|------|-------------|
| PDS | Classical | Piecewise Direct Standardization (Wang, Veltkamp & Kowalski, 1991) |
| DS | Classical | Direct Standardization |
| SBC | Classical | Slope/Bias Correction |
| PLS | Classical | Partial Least Squares regression |
| VS³L (ours) | Foundation model | Self-supervised pretraining + LoRA transfer + TTT |

## Benchmark Targets

| Method | R2 (corn moisture) | Transfer Samples |
|--------|-------------------|-----------------|
| PDS | ~0.55 | 30 |
| DS | ~0.69 | 30 |
| LoRA-CT (literature) | 0.952 | 50 |
| **VS³L (target)** | **>0.96** | **10** |

## Pretraining Objectives

| Loss | Purpose |
|------|---------|
| **MSRP** | Masked Spectrum Reconstruction — contiguous block masking, learn spectral structure |
| **Contrastive** | BYOL-style instrument-invariance between augmented views of same spectrum |
| **Denoising** | Reconstruct clean spectrum from synthetically corrupted input |
| **Physics** | Beer-Lambert linearity, smoothness, non-negativity, peak shape constraints |
| **OT Alignment** | Sinkhorn-based Wasserstein distance across instrument latent distributions |
| **VIB** | Variational Information Bottleneck — disentangle z_chem from z_inst |

## Transfer Methods

| Method | Transfer Samples | Description |
|--------|-----------------|-------------|
| **TTT** | 0 (unlabeled only) | Run K steps of MSRP self-supervision on unlabeled target spectra |
| **LoRA** | 5-10 (labeled) | Low-rank adaptation of transformer attention layers |
| **FNO** | N/A | Fourier Neural Operator head for resolution-independent spectral mapping |

## Current Status

Core architecture implemented (~12K lines across 20 Python modules). Evaluation datasets preprocessed. Classical baselines running (PDS, DS, SBC). Forward pass and training loop under active debugging.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for details.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{karthikeyan2026vs3l,
  title={VS$^3$L: Toward Standard-Free Calibration Transfer in Vibrational Spectroscopy via Self-Supervised Learning},
  author={Karthikeyan, Tubhyam},
  journal={Analytical Chemistry},
  year={2026}
}
```

---

*Under active development. Targeting publication in Analytical Chemistry (ACS).*
