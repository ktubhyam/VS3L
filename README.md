# SSM-OT-CalTransfer

**Bridging State Space Models and Optimal Transport for Zero-to-Few-Shot Spectral Calibration Transfer**

The first self-supervised foundation model for vibrational spectroscopy (NIR, IR, Raman). Combines a hybrid Mamba-Transformer backbone with Sinkhorn-based optimal transport domain adaptation, variational information bottleneck disentanglement, and physics-informed regularization to achieve calibration transfer across spectrometers using 10 or fewer labeled samples.

> **Author:** Tubhyam Karthikeyan (ICT Mumbai / InvyrAI)
>
> **Target Journal:** Analytical Chemistry (ACS, IF 7.4)

---

## The Problem

Calibration transfer in spectroscopy has relied on the same approaches for 30+ years: measure 10-60 transfer samples on both instruments, then apply PDS/DS/SBC correction. This is expensive ($50-200 per reference analysis) and must be repeated for every new instrument.

## Our Approach

SSM-OT-CalTransfer proposes a **fifth strategy** for calibration transfer: learn instrument-invariant chemical representations from a massive pretraining corpus, then adapt with minimal transfer data.

- **Zero-shot:** Test-time training (TTT) on unlabeled spectra from the new instrument
- **Few-shot:** LoRA fine-tuning with as few as 5-10 transfer samples
- **Target:** 10 transfer samples beats classical methods using 50

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

**Key design choices:**
- **Mamba (O(n))** handles long-range spectral dependencies efficiently
- **Transformer (O(n^2))** adds global expressiveness where needed
- **Wavelet decomposition** separates sharp peaks from baselines
- **VIB disentanglement** splits latent space into transferable chemistry vs. discardable instrument signature
- **Optimal transport** aligns latent distributions across instruments via Sinkhorn divergence
- **Physics-informed losses** enforce Beer-Lambert linearity, smoothness, non-negativity, peak shape

## Project Structure

```
SSM-OT-CalTransfer/
├── run.py                         # Entry point (pretrain / finetune / evaluate / ttt)
├── src/
│   ├── config.py                  # All hyperparameters (dataclass-based)
│   ├── models/
│   │   ├── embedding.py           # WaveletEmbedding + WavenumberPE
│   │   ├── mamba.py               # Pure PyTorch selective SSM
│   │   ├── moe.py                 # Mixture of Experts + KAN layers
│   │   ├── transformer.py         # Lightweight TransformerEncoder
│   │   ├── heads.py               # VIB, Reconstruction, Regression, FNO heads
│   │   └── spectral_fm.py         # Full model assembly + TTT
│   ├── losses/
│   │   └── losses.py              # MSRP, contrastive, physics, OT, VIB, MoE losses
│   ├── training/
│   │   └── trainer.py             # Pretrain + finetune + TTT training loops
│   ├── evaluation/
│   │   ├── metrics.py             # R2, RMSEP, RPD, bias, conformal prediction
│   │   └── baselines.py           # PDS, SBC, DS classical baselines
│   └── data/
│       ├── datasets.py            # Data loading, augmentation, wavelet preprocessing
│       ├── build_pretrain_corpus.py    # Download + preprocess pretraining data
│       └── pretraining_pipeline.py     # Pretraining dataset class
├── scripts/
│   ├── run_baselines.py           # Run classical baseline comparison
│   └── run_finetune_test.py       # Fine-tuning validation script
├── data/
│   ├── raw/                       # Original .mat files
│   └── processed/                 # Preprocessed .npy arrays
│       ├── corn/                  # 80 samples x 3 instruments x 700 channels
│       └── tablet/                # 655 samples x 2 instruments x 650 channels
├── experiments/                   # Experiment results (JSON)
├── checkpoints/                   # Saved model weights
├── figures/                       # Generated plots
├── paper/                         # Research notes and brainstorms
├── requirements.txt
├── CLAUDE.md                      # Dev instructions for Claude Code
├── PROJECT_STATUS.md              # Current state and known issues
└── IMPLEMENTATION_PLAN.md         # Detailed task breakdown
```

## Datasets

### Evaluation (included, preprocessed)

| Dataset | Samples | Instruments | Channels | Properties |
|---------|---------|-------------|----------|------------|
| **Corn** | 80 | 3 (m5, mp5, mp6) | 700 | moisture, oil, protein, starch |
| **Tablet** | 655 | 2 | 650 | active ingredient, weight, hardness |

### Pretraining (to be downloaded)

| Source | Spectra | Modality |
|--------|---------|----------|
| ChEMBL IR-Raman | ~220K | IR, Raman |
| USPTO-Spectra | ~177K | Mixed |
| NIST WebBook | ~5.2K | IR |
| RRUFF | ~8.6K | Raman |
| **Total** | **~411K** | |

## Installation

```bash
# Clone
git clone https://github.com/ktubhyam/SSM-OT-CalTransfer.git
cd SSM-OT-CalTransfer

# Install dependencies
pip install -r requirements.txt

# Optional (CUDA Mamba kernels — Linux + CUDA only)
pip install mamba-ssm>=1.2.0
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ (2.2.2 for Intel Mac, 2.4+ for Apple Silicon/CUDA)
- ~40GB VRAM for full pretraining (A100 recommended)
- CPU-only mode available for development and fine-tuning

## Usage

```bash
# Smoke test (verify everything works)
python run.py --mode smoke_test

# Run classical baselines on corn dataset
python scripts/run_baselines.py

# Pretrain on spectral corpus
python run.py --mode pretrain

# Fine-tune for calibration transfer
python run.py --mode finetune --checkpoint checkpoints/pretrain_best.pt

# Zero-shot transfer via test-time training
python run.py --mode ttt --checkpoint checkpoints/pretrain_best.pt
```

## Baselines Implemented

| Method | Type | Description |
|--------|------|-------------|
| PDS | Classical | Piecewise Direct Standardization (Kowalski) |
| DS | Classical | Direct Standardization |
| SBC | Classical | Slope/Bias Correction |
| PLS | Classical | Partial Least Squares regression |
| SSM-OT-CalTransfer (ours) | Foundation model | Few-shot LoRA transfer + TTT |

## Key Benchmark Target

| Method | R2 (corn moisture) | Transfer Samples |
|--------|-------------------|-----------------|
| PDS | ~0.55 | 30 |
| DS | ~0.69 | 30 |
| LoRA-CT (literature) | 0.952 | 50 |
| **SSM-OT-CalTransfer (target)** | **>0.96** | **10** |

## Current Status

**Phase 1 (Make It Run):** Core architecture implemented (~12K lines across 20 Python modules), evaluation datasets preprocessed (Corn: 80 samples x 3 instruments, Tablet: 655 samples x 2 instruments), classical baselines running (PDS, DS, SBC). Forward pass and training loop under active debugging.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed progress tracking.

## Technical Details

### Pretraining Objectives
- **MSRP** — Masked Spectrum Reconstruction Pretraining (contiguous block masking)
- **Contrastive** — BYOL-style instrument-invariance between augmented views
- **Denoising** — Reconstruct clean spectrum from synthetically corrupted input
- **Physics** — Beer-Lambert, smoothness, non-negativity, peak shape constraints
- **OT Alignment** — Sinkhorn-based Wasserstein distance across instrument latents
- **VIB** — Variational Information Bottleneck for z_chem/z_inst disentanglement

### Transfer Methods
- **TTT (zero-shot):** Run K steps of MSRP self-supervision on unlabeled target spectra
- **LoRA (few-shot):** Low-rank adaptation of transformer layers with N labeled pairs
- **FNO head:** Fourier Neural Operator for resolution-independent spectral mapping

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{karthikeyan2026ssmotcaltransfer,
  title={Bridging State Space Models and Optimal Transport for Zero-to-Few-Shot Spectral Calibration Transfer},
  author={Karthikeyan, Tubhyam},
  journal={Analytical Chemistry},
  year={2026}
}
```

---

*Under active development. This is a research project targeting publication in Analytical Chemistry (ACS).*
