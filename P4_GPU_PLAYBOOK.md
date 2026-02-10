# P4: GPU Execution Playbook
# SpectralFM v2 — Lambda Cloud A10 (24GB VRAM)

## Overview
Everything is built. This playbook covers setup → pretrain → evaluate → iterate.
Estimated total GPU time: 4-8 hours.

---

## Phase 1: Instance Setup (~10 min)

### 1A. Launch Lambda Cloud
```bash
# After SSH into instance:
nvidia-smi  # Verify A10 24GB
```

### 1B. Clone and Install
```bash
git clone https://github.com/[YOUR-REPO]/VS3L.git
cd VS3L

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn h5py wandb matplotlib
pip install mamba-ssm  # Mamba SSM (requires CUDA)
pip install causal-conv1d  # Mamba dependency

# Optional
pip install pywt  # wavelet transforms (may already be in torch)

# Verify
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python3 -c "from mamba_ssm import Mamba; print('Mamba OK')"
```

### 1C. Upload Data
```bash
# From local machine:
scp data/pretrain/spectral_corpus_v2.h5 lambda:~/VS3L/data/pretrain/
scp -r data/processed/ lambda:~/VS3L/data/processed/

# Or use the existing experiments/baselines_complete.json too
scp experiments/baselines_complete.json lambda:~/VS3L/experiments/
```

### 1D. W&B Login (optional)
```bash
wandb login  # paste API key
```

---

## Phase 2: Diagnostic Test (~20 min) ⚡ MUST DO FIRST

**Purpose**: Validate the architecture works before investing 2-6 hours in full pretraining.

```bash
cd ~/VS3L
python3 scripts/run_diagnostic.py --device cuda --max-steps 2000
```

### Interpreting Results:
| Result | Meaning | Action |
|--------|---------|--------|
| TTT R² > 0.3 | 🎉 EXCELLENT | Proceed immediately to full pretrain |
| TTT R² > 0.0 | ✅ PASS | Proceed to full pretrain |
| TTT R² ≤ 0.0 | ⚠️ DEBUG | Check loss curves, try more steps |

### If FAIL — Debug Steps:
1. Check pretrain loss is decreasing: `cat experiments/diagnostic_results.json | python3 -m json.tool`
2. Try longer pretraining: `python3 scripts/run_diagnostic.py --device cuda --max-steps 5000`
3. Try different TTT learning rate: edit `run_diagnostic.py`, set `ttt_lr=1e-3`
4. Check gradient flow: add `torch.nn.utils.clip_grad_norm_` logging
5. Verify data loading: `python3 -c "from scripts.run_diagnostic import CornPretrainDataset; d=CornPretrainDataset('data'); print(len(d), d[0][0].shape)"`

---

## Phase 3: Full Pretraining (~2-6 hours)

### 3A. Sanity Check (5 min)
```bash
python3 scripts/run_pretrain.py \
    --max-steps 100 --batch-size 64 --num-workers 4 \
    --log-every 10 --save-every 50 --val-every 50 \
    --no-wandb --run-name "sanity_check"
```
**Check**: Loss should decrease from ~1.0 to ~0.8 in 100 steps. Throughput should be >200 samp/s.

### 3B. Full Run
```bash
python3 scripts/run_pretrain.py \
    --max-steps 50000 --batch-size 64 --lr 3e-4 \
    --num-workers 4 --log-every 100 --save-every 5000 --val-every 1000 \
    --wandb --run-name "pretrain_v2_50k"
```

### 3C. Monitor
```bash
# In another terminal:
tail -f logs/pretrain_v2_50k.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if 'val_loss' in d:
        print(f'Step {d[\"step\"]}: val_loss={d[\"val_loss\"]:.4f}')
    elif d.get('step', 0) % 1000 == 0:
        print(f'Step {d[\"step\"]}: loss={d[\"loss\"]:.4f}, lr={d.get(\"lr\",0):.6f}, samp/s={d.get(\"throughput\",0):.0f}')
"
```

### 3D. Expected Behavior
- Steps 0-1000: Loss drops rapidly (1.0 → 0.5)
- Steps 1000-10000: Steady decrease (0.5 → 0.3)
- Steps 10000-50000: Gradual improvement (0.3 → 0.15-0.25)
- Val loss should track train loss with small gap (<0.1)
- If val loss diverges: reduce lr, increase dropout, or stop early

### 3E. Checkpoints
```bash
ls -la checkpoints/
# Should see: best_pretrain.pt, checkpoint_5000.pt, ..., checkpoint_50000.pt
```

---

## Phase 4: Evaluation (~2-4 hours)

### 4A. Zero-Shot TTT (THE MAKE-OR-BREAK EXPERIMENT)
```bash
python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --source m5 --target mp6 --property moisture \
    --device cuda
```

**Critical Result**: Look at `ttt_zeroshot_r2` in output.
- R² > 0.3 at 0 samples → **NEW PARADIGM** (paper lead result)
- R² > 0.0 at 0 samples → Promising, proceed
- R² < 0.0 → TTT doesn't work, pivot to LoRA-only narrative

### 4B. Sample Efficiency Sweep (KEY FIGURE)
```bash
python3 scripts/run_finetune.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --all-properties --sweep \
    --device cuda
```
This runs [1,3,5,10,20,30,50] samples × 5 seeds × 4 properties.

**Target**: Beat CCA (R²=0.687) at ≤10 samples for moisture.

### 4C. TTT + Few-Shot Combined
```bash
python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --sweep --device cuda
```

### 4D. TTT Ablation
```bash
python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --ablation --device cuda
```

### 4E. Full Experiment Suite
```bash
# Run all automated experiments (E1-E5, E10-E12)
python3 scripts/run_experiments.py --all \
    --checkpoint checkpoints/best_pretrain.pt --n-seeds 5
```

---

## Phase 5: Results Analysis

### 5A. Generate Figures
```bash
python3 -c "from src.evaluation.visualization import generate_all_figures_from_experiments; generate_all_figures_from_experiments()"
```

### 5B. Download Results
```bash
# From local machine:
scp -r lambda:~/VS3L/experiments/ ./experiments_gpu/
scp -r lambda:~/VS3L/figures/ ./figures/
scp lambda:~/VS3L/checkpoints/best_pretrain.pt ./checkpoints/
```

---

## Decision Tree After Results

```
Zero-shot TTT R² > 0.3?
├── YES → Lead with "zero-shot calibration transfer" narrative
│         Target: Analytical Chemistry (high confidence)
│         Also run scaling experiment (E7) for Nature MI attempt
│
└── NO → Zero-shot TTT R² > 0.0?
    ├── YES → Lead with "few-shot LoRA transfer with TTT warmup"
    │         Compare TTT+LoRA vs LoRA-only vs baselines
    │         Target: Analytical Chemistry
    │
    └── NO → Lead with "physics-informed pretraining for sample-efficient transfer"
              Focus on LoRA beating baselines at low N
              Target: Analytical Chemistry (moderate confidence)
```

---

## Baseline Targets (from completed CPU evaluation)

### Key Experiment: Corn m5→mp6 Moisture (30 transfer samples)
| Method | R² (mean±std) |
|--------|---------------|
| CCA | 0.687 ± 0.095 |
| DS | 0.666 ± 0.052 |
| SBC | 0.354 ± 0.065 |
| PDS | -1.551 ± 2.007 |
| di-PLS | varies |
| Target Direct | 0.719 ± 0.071 |
| No Transfer | -17.277 ± 2.561 |

**SpectralFM must beat CCA (0.687) at ≤10 transfer samples to claim sample efficiency.**

### Strongest Baseline Per Property (m5→mp6):
- Moisture: CCA R²=0.687
- Oil: CCA R²=0.571
- Protein: SBC R²=0.875 (!)
- Starch: SBC R²=0.769

### Tablet (hardness only transferable):
- CCA: R²=0.851 (spec_1→spec_2), R²=0.887 (spec_2→spec_1)
- Other properties: all methods fail (R² < 0)

---

## Troubleshooting

### CUDA OOM
```bash
# Reduce batch size
python3 scripts/run_pretrain.py --batch-size 32 ...

# Or enable gradient checkpointing (add to SpectralFM.__init__):
# self.mamba.gradient_checkpointing = True
```

### Mamba Installation Fails
```bash
# Try building from source
pip install mamba-ssm --no-build-isolation
# Or use pre-built wheel
pip install mamba-ssm==2.2.2
```

### Loss NaN
```bash
# Reduce learning rate
python3 scripts/run_pretrain.py --lr 1e-4 ...
# Check gradient clipping is active (should be in trainer.py)
```

### W&B Connection Issues
```bash
# Run without W&B, still get JSONL logs
python3 scripts/run_pretrain.py --no-wandb ...
```

---

## Time Budget

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Setup | 10 min | 10 min |
| Diagnostic | 20 min | 30 min |
| Sanity check | 5 min | 35 min |
| Full pretrain | 2-4 hours | 2.5-4.5 hours |
| TTT zero-shot | 15 min | 2.75-4.75 hours |
| Sample efficiency sweep | 1-2 hours | 3.75-6.75 hours |
| Full experiment suite | 1-2 hours | 4.75-8.75 hours |
| Figure generation | 5 min | ~5-9 hours |

**Total estimated: 5-9 hours of GPU time.**
