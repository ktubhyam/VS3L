# P4: GPU Execution Playbook
# SpectralFM v2 — Lambda Cloud A10 (24GB)
# Estimated total GPU time: 4-8 hours
# Date: 2026-02-10

## Overview

Everything below is ready to copy-paste into a Lambda Cloud terminal.
All CPU-side infrastructure is complete:
- 19/19 smoke tests passing
- 30 baseline experiments (24 corn + 6 tablet) with 5 seeds each
- Scripts: run_diagnostic.py, run_pretrain.py, run_finetune.py, run_ttt.py, run_experiments.py
- Visualization: src/evaluation/visualization.py

## Baseline Targets (from completed CPU runs)

### m5→mp6 (hardest corn pair, our primary benchmark)
| Property | Best Classical | R² | Method |
|----------|---------------|-----|--------|
| moisture | 0.687±0.095 | CCA |
| oil | 0.571±0.049 | CCA |
| protein | 0.875±0.037 | SBC |
| starch | 0.769±0.027 | SBC |

### Success criteria
- **Paper-worthy**: Zero-shot TTT R² > 0.0 on ANY property (novel paradigm)
- **Strong paper**: Beat CCA/SBC at 10 transfer samples
- **Exceptional**: Beat CCA/SBC at 5 samples AND zero-shot R² > 0.25

---

## Phase 1: Instance Setup (~10 min)

```bash
# 1. Clone repo
git clone <REPO_URL> VS3L && cd VS3L

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn h5py wandb matplotlib seaborn pywt
pip install mamba-ssm  # If fails, pure-torch fallback works

# 3. Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

# 4. Verify data
python3 -c "
import h5py, numpy as np
h = h5py.File('data/pretrain/spectral_corpus_v2.h5', 'r')
print(f'Corpus: {len(h[\"spectra\"])} spectra, shape {h[\"spectra\"].shape}')
h.close()
for f in ['m5_spectra','mp5_spectra','mp6_spectra','properties']:
    a = np.load(f'data/processed/corn/{f}.npy')
    print(f'  {f}: {a.shape}')
"

# 5. Verify tests
python3 -m pytest tests/ -x -q --tb=short 2>&1 | tail -5

# 6. W&B login (optional)
wandb login
```

---

## Phase 2: Diagnostic (~20 min) — RUN FIRST

Trains ONLY on corn m5+mp5 (160 spectra), tests TTT on mp6.

```bash
python3 scripts/run_diagnostic.py --device cuda --max-steps 2000
```

### Decision tree:
- R² > 0.3 → EXCELLENT, proceed to full pretraining
- R² > 0.0 → PASS, proceed to full pretraining
- R² ≤ 0.0 → DEBUG: try --max-steps 5000, adapt_layers="all", check loss convergence

---

## Phase 3: Full Pretraining (~2-6 hours)

### 3a. Sanity check (5 min)
```bash
python3 scripts/run_pretrain.py \
    --max-steps 100 --batch-size 64 --num-workers 4 \
    --log-every 10 --save-every 50 --val-every 50 \
    --no-wandb --run-name "sanity_check"
```
Check: loss ~1.0 → ~0.8 in 100 steps.

### 3b. Full run
```bash
python3 scripts/run_pretrain.py \
    --max-steps 50000 --batch-size 64 --lr 3e-4 \
    --num-workers 4 --log-every 100 --save-every 5000 --val-every 1000 \
    --wandb --run-name "pretrain_v2_50k"
```

Expected: loss 1.0 → 0.3-0.5, throughput ~200-500 samp/s, 2-4 hours.

---

## Phase 4: Evaluation (~2-4 hours)

### 4a. Zero-shot TTT — THE KEY EXPERIMENT
```bash
python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --source m5 --target mp6 --property moisture --device cuda
```

### 4b. Full TTT sweep
```bash
python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --sweep --device cuda
```

### 4c. Sample efficiency — THE KEY FIGURE
```bash
python3 scripts/run_finetune.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --all-properties --sweep --device cuda
```

### 4d. Full experiment suite
```bash
python3 scripts/run_experiments.py --all \
    --checkpoint checkpoints/best_pretrain.pt --n-seeds 3
```

---

## Phase 5: Collect Results

```bash
# Generate figures
python3 -c "
from src.evaluation.visualization import generate_all_figures_from_experiments
generate_all_figures_from_experiments('experiments', 'figures')
"

# Tar for download
tar -czf spectral_fm_results.tar.gz \
    experiments/ figures/ checkpoints/best_pretrain.pt logs/

echo "Size: $(du -sh spectral_fm_results.tar.gz | cut -f1)"
```

---

## Contingency

| Problem | Fix |
|---------|-----|
| Loss stuck > 0.8 after 10K steps | `--lr 1e-4` or `--warmup-steps 2000` |
| TTT R² ≤ 0 zero-shot | Shift to few-shot narrative; run `--sweep` |
| OOM on A10 | `--batch-size 32 --lr 2e-4` |
| mamba-ssm install fails | Pure-torch fallback is automatic |

## Cost: ~$3-6 total (A10 @ $0.75/hr × 4-8 hr)
