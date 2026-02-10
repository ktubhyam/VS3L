#!/usr/bin/env python3
"""
SpectralFM v2: Publication-Quality Visualization

Generates all figures for the paper:
- Fig 1: Sample efficiency curves (R² vs N transfer samples)
- Fig 2: t-SNE disentanglement (z_chem vs z_inst)
- Fig 3: Attention heatmaps (wavelength importance)
- Fig 4: Scaling law (log corpus size vs R²)
- Fig 5: Uncertainty calibration plots
- Fig 6: TTT step budget analysis
- Fig 7: Loss ablation bar charts

All figures saved as both PDF (paper) and PNG (preview).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# Style Configuration (publication-ready)
# ============================================================

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
}

# Color palette: accessible, distinguishable, printer-friendly
COLORS = {
    "SpectralFM": "#2166AC",       # blue
    "SpectralFM+TTT": "#053061",   # dark blue
    "DS": "#B2182B",               # red
    "PDS": "#EF8A62",              # salmon
    "SBC": "#67A9CF",              # light blue
    "CCA": "#D6604D",              # coral
    "di-PLS": "#FDDBC7",          # peach
    "No_Transfer": "#999999",      # gray
    "Target_Direct": "#1B7837",    # green
    "LoRA-CT": "#762A83",          # purple
}

MARKERS = {
    "SpectralFM": "o",
    "SpectralFM+TTT": "s",
    "DS": "^",
    "PDS": "v",
    "SBC": "D",
    "CCA": "<",
    "di-PLS": ">",
    "LoRA-CT": "p",
    "Target_Direct": "*",
}


def apply_style():
    if HAS_MPL:
        plt.rcParams.update(STYLE)


def save_fig(fig, name, figures_dir="figures"):
    """Save figure as both PDF and PNG."""
    d = Path(figures_dir)
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", bbox_inches="tight", dpi=300)
    print(f"  Saved: {d / name}.pdf + .png")
    plt.close(fig)


# ============================================================
# Fig 1: Sample Efficiency Curves
# ============================================================

def plot_sample_efficiency(
    spectral_fm_results: Dict[int, Dict],
    baseline_results: Dict[str, Dict],
    ttt_results: Optional[Dict[int, Dict]] = None,
    title: str = "Sample Efficiency: Corn m5→mp6 Moisture",
    figures_dir: str = "figures",
):
    """Plot R² vs number of transfer samples.

    Args:
        spectral_fm_results: {n_samples: {"r2_mean": ..., "r2_std": ...}}
        baseline_results: {method: {"r2": ...}} at full n_transfer
        ttt_results: optional {n_samples: {"r2_mean": ..., "r2_std": ...}} for TTT+LoRA
    """
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    # SpectralFM curve
    ns = sorted(spectral_fm_results.keys())
    means = [spectral_fm_results[n]["r2_mean"] for n in ns]
    stds = [spectral_fm_results[n].get("r2_std", 0) for n in ns]
    ax.errorbar(ns, means, yerr=stds, label="SpectralFM (LoRA)",
                color=COLORS["SpectralFM"], marker=MARKERS["SpectralFM"],
                capsize=3, linewidth=2)

    # TTT curve (if available)
    if ttt_results:
        ns_ttt = sorted(ttt_results.keys())
        means_ttt = [ttt_results[n]["r2_mean"] for n in ns_ttt]
        stds_ttt = [ttt_results[n].get("r2_std", 0) for n in ns_ttt]
        ax.errorbar(ns_ttt, means_ttt, yerr=stds_ttt, label="SpectralFM (TTT+LoRA)",
                    color=COLORS["SpectralFM+TTT"], marker=MARKERS["SpectralFM+TTT"],
                    capsize=3, linewidth=2, linestyle="--")

    # Baseline horizontal lines (at their fixed n_transfer)
    for method in ["DS", "PDS", "SBC", "CCA", "di-PLS"]:
        if method in baseline_results:
            r2 = baseline_results[method].get("r2", baseline_results[method].get("r2_mean"))
            if r2 is not None and not np.isnan(r2):
                ax.axhline(y=r2, color=COLORS.get(method, "#666"),
                          linestyle=":", alpha=0.7, linewidth=1)
                ax.text(max(ns) * 0.95, r2 + 0.01, method,
                       fontsize=8, color=COLORS.get(method, "#666"),
                       ha="right", va="bottom")

    # Target direct upper bound
    if "Target_Direct" in baseline_results:
        td_r2 = baseline_results["Target_Direct"].get("r2")
        if td_r2 is not None:
            ax.axhline(y=td_r2, color=COLORS["Target_Direct"],
                      linestyle="-.", alpha=0.5, linewidth=1)
            ax.text(max(ns) * 0.95, td_r2 + 0.01, "Upper bound",
                   fontsize=8, color=COLORS["Target_Direct"], ha="right")

    ax.set_xlabel("Number of transfer samples")
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(-0.1, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    save_fig(fig, "sample_efficiency", figures_dir)


# ============================================================
# Fig 2: t-SNE Disentanglement
# ============================================================

def plot_tsne_disentanglement(
    z_chem: np.ndarray,  # (N, d_chem)
    z_inst: np.ndarray,  # (N, d_inst)
    labels_chem: np.ndarray,  # (N,) continuous or categorical
    labels_inst: np.ndarray,  # (N,) instrument labels
    chem_name: str = "Moisture (%)",
    figures_dir: str = "figures",
):
    """Plot t-SNE of z_chem and z_inst colored by chemistry and instrument.

    The key insight: z_chem should cluster by chemistry (not instrument),
    z_inst should cluster by instrument (not chemistry).
    """
    if not HAS_MPL or not HAS_SKLEARN:
        print("matplotlib/sklearn not available, skipping t-SNE")
        return

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    perplexity = min(30, len(z_chem) - 1)

    # z_chem t-SNE
    tsne_chem = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    z_chem_2d = tsne_chem.fit_transform(z_chem)

    # z_inst t-SNE
    tsne_inst = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    z_inst_2d = tsne_inst.fit_transform(z_inst)

    # Panel A: z_chem colored by chemistry
    sc = axes[0, 0].scatter(z_chem_2d[:, 0], z_chem_2d[:, 1],
                            c=labels_chem, cmap="viridis", s=20, alpha=0.7)
    plt.colorbar(sc, ax=axes[0, 0], label=chem_name)
    axes[0, 0].set_title("z_chem colored by chemistry")
    axes[0, 0].set_xlabel("t-SNE 1")
    axes[0, 0].set_ylabel("t-SNE 2")

    # Panel B: z_chem colored by instrument
    unique_inst = np.unique(labels_inst)
    inst_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_inst)))
    for i, inst in enumerate(unique_inst):
        mask = labels_inst == inst
        axes[0, 1].scatter(z_chem_2d[mask, 0], z_chem_2d[mask, 1],
                          c=[inst_colors[i]], label=inst, s=20, alpha=0.7)
    axes[0, 1].legend(title="Instrument", fontsize=8)
    axes[0, 1].set_title("z_chem colored by instrument")
    axes[0, 1].set_xlabel("t-SNE 1")
    axes[0, 1].set_ylabel("t-SNE 2")

    # Panel C: z_inst colored by chemistry
    sc = axes[1, 0].scatter(z_inst_2d[:, 0], z_inst_2d[:, 1],
                            c=labels_chem, cmap="viridis", s=20, alpha=0.7)
    plt.colorbar(sc, ax=axes[1, 0], label=chem_name)
    axes[1, 0].set_title("z_inst colored by chemistry")
    axes[1, 0].set_xlabel("t-SNE 1")
    axes[1, 0].set_ylabel("t-SNE 2")

    # Panel D: z_inst colored by instrument
    for i, inst in enumerate(unique_inst):
        mask = labels_inst == inst
        axes[1, 1].scatter(z_inst_2d[mask, 0], z_inst_2d[mask, 1],
                          c=[inst_colors[i]], label=inst, s=20, alpha=0.7)
    axes[1, 1].legend(title="Instrument", fontsize=8)
    axes[1, 1].set_title("z_inst colored by instrument")
    axes[1, 1].set_xlabel("t-SNE 1")
    axes[1, 1].set_ylabel("t-SNE 2")

    fig.suptitle("VIB Disentanglement: Chemistry vs. Instrument", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "tsne_disentanglement", figures_dir)


# ============================================================
# Fig 3: TTT Step Budget
# ============================================================

def plot_ttt_steps(
    ttt_results: Dict[int, Dict],
    baseline_r2: Optional[float] = None,
    title: str = "Zero-Shot TTT: R² vs Adaptation Steps",
    figures_dir: str = "figures",
):
    """Plot R² improvement as a function of TTT gradient steps."""
    if not HAS_MPL:
        return

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    steps = sorted(ttt_results.keys())
    r2s = [ttt_results[s]["r2"] for s in steps]

    ax.plot(steps, r2s, "o-", color=COLORS["SpectralFM+TTT"], linewidth=2, markersize=7)

    if baseline_r2 is not None:
        ax.axhline(y=baseline_r2, color=COLORS["DS"], linestyle=":",
                  label=f"DS (30 samples): R²={baseline_r2:.3f}")

    ax.axhline(y=0, color="#999", linestyle="-", alpha=0.3)
    ax.set_xlabel("TTT gradient steps")
    ax.set_ylabel("R²")
    ax.set_title(title)
    if baseline_r2 is not None:
        ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig(fig, "ttt_steps", figures_dir)


# ============================================================
# Fig 4: Scaling Law
# ============================================================

def plot_scaling_law(
    corpus_sizes: List[int],
    r2_means: List[float],
    r2_stds: Optional[List[float]] = None,
    title: str = "Pretraining Scaling Law",
    figures_dir: str = "figures",
):
    """Plot downstream R² vs log(corpus size)."""
    if not HAS_MPL:
        return

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    x = np.log10(corpus_sizes)
    ax.errorbar(x, r2_means,
                yerr=r2_stds if r2_stds else None,
                fmt="o-", color=COLORS["SpectralFM"], capsize=4,
                linewidth=2, markersize=8)

    # Fit line
    if len(x) >= 3:
        coeffs = np.polyfit(x, r2_means, 1)
        x_fit = np.linspace(x[0] - 0.2, x[-1] + 0.5, 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--",
               color=COLORS["SpectralFM"], alpha=0.4,
               label=f"slope={coeffs[0]:.3f}")
        ax.legend()

    ax.set_xlabel("log₁₀(Pretraining corpus size)")
    ax.set_ylabel("Downstream Transfer R²")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Nice tick labels
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in corpus_sizes], rotation=30, fontsize=8)

    save_fig(fig, "scaling_law", figures_dir)


# ============================================================
# Fig 5: Ablation Bar Chart
# ============================================================

def plot_ablation(
    ablation_results: Dict[str, float],
    title: str = "Loss Component Ablation",
    figures_dir: str = "figures",
):
    """Bar chart showing R² with/without each loss component."""
    if not HAS_MPL:
        return

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    names = list(ablation_results.keys())
    values = list(ablation_results.values())

    # Color: full model green, ablations red-ish
    colors = [COLORS["Target_Direct"] if i == 0 else COLORS["DS"]
              for i in range(len(names))]

    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8, height=0.6)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("R²")
    ax.set_title(title)
    ax.invert_yaxis()

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f"{val:.3f}", va="center", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "ablation", figures_dir)


# ============================================================
# Fig 6: Uncertainty Calibration
# ============================================================

def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    title: str = "Prediction Calibration",
    figures_dir: str = "figures",
):
    """Calibration plot: predicted vs actual + uncertainty coverage."""
    if not HAS_MPL:
        return

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Predicted vs Actual
    ax1.scatter(y_true, y_pred, c=uncertainties, cmap="YlOrRd",
               s=30, alpha=0.7, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax1.set_xlabel("True value")
    ax1.set_ylabel("Predicted value")
    ax1.set_title("Predicted vs. Actual")
    cb = plt.colorbar(ax1.collections[0], ax=ax1)
    cb.set_label("Uncertainty (σ)")

    # Panel B: Coverage plot
    # For each confidence level, compute actual coverage
    confidence_levels = np.linspace(0.5, 0.99, 20)
    from scipy import stats
    actual_coverages = []
    for cl in confidence_levels:
        z = stats.norm.ppf((1 + cl) / 2)
        lower = y_pred - z * uncertainties
        upper = y_pred + z * uncertainties
        covered = ((y_true >= lower) & (y_true <= upper)).mean()
        actual_coverages.append(covered)

    ax2.plot(confidence_levels, actual_coverages, "o-",
            color=COLORS["SpectralFM"], linewidth=2, markersize=4)
    ax2.plot([0.5, 1], [0.5, 1], "k--", alpha=0.5, linewidth=1,
            label="Perfect calibration")
    ax2.set_xlabel("Expected coverage")
    ax2.set_ylabel("Actual coverage")
    ax2.set_title("Uncertainty Calibration")
    ax2.legend()
    ax2.set_xlim(0.48, 1.02)
    ax2.set_ylim(0.48, 1.02)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "calibration", figures_dir)


# ============================================================
# Table Generator (LaTeX)
# ============================================================

def generate_latex_table(
    results: Dict[str, Dict[str, Dict]],
    methods: List[str],
    metric: str = "r2",
    caption: str = "Calibration transfer results",
    label: str = "tab:results",
) -> str:
    """Generate LaTeX table from experiment results.

    Args:
        results: {experiment_key: {method: {metric_mean: ..., metric_std: ...}}}
        methods: ordered list of method names
        metric: which metric to show (r2, rmsep, rpd)
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    n_cols = len(methods) + 1
    lines.append(r"\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append(r"\toprule")

    # Header
    header = "Experiment & " + " & ".join(methods) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for exp_key, exp_results in results.items():
        row_parts = [exp_key.replace("_", r"\_")]
        best_val = -float("inf")
        best_method = None

        # Find best for bolding
        for m in methods:
            r = exp_results.get(m, {})
            val = r.get(f"{metric}_mean", r.get(metric, float("nan")))
            if not np.isnan(val) and val > best_val:
                best_val = val
                best_method = m

        for m in methods:
            r = exp_results.get(m, {})
            mean = r.get(f"{metric}_mean", r.get(metric, float("nan")))
            std = r.get(f"{metric}_std", 0)

            if np.isnan(mean):
                row_parts.append("---")
            elif std > 0:
                cell = f"{mean:.3f}$\\pm${std:.3f}"
                if m == best_method:
                    cell = r"\textbf{" + cell + "}"
                row_parts.append(cell)
            else:
                cell = f"{mean:.3f}"
                if m == best_method:
                    cell = r"\textbf{" + cell + "}"
                row_parts.append(cell)

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================
# Batch Figure Generator
# ============================================================

def generate_all_figures_from_experiments(experiments_dir="experiments",
                                          figures_dir="figures"):
    """Load all experiment JSONs and generate every figure."""
    exp_dir = Path(experiments_dir)
    fig_dir = Path(figures_dir)

    # 1. Sample efficiency
    for f in exp_dir.glob("finetune_*.json"):
        with open(f) as fp:
            data = json.load(fp)
        for prop, res in data.get("results", {}).items():
            if "sweep" in res:
                sweep = {int(k): v for k, v in res["sweep"].items()}
                baselines = res.get("baselines", {})
                plot_sample_efficiency(sweep, baselines,
                                      title=f"Sample Efficiency: {prop}",
                                      figures_dir=str(fig_dir))
                print(f"  Generated sample_efficiency for {prop}")

    # 2. TTT steps
    for f in exp_dir.glob("ttt_zeroshot_*.json"):
        with open(f) as fp:
            data = json.load(fp)
        results = {int(k): v for k, v in data.get("results", {}).items()}
        plot_ttt_steps(results, figures_dir=str(fig_dir))
        print("  Generated ttt_steps")

    # 3. Diagnostic
    diag_path = exp_dir / "diagnostic_results.json"
    if diag_path.exists():
        with open(diag_path) as fp:
            data = json.load(fp)
        ttt_r = {int(k): v for k, v in data.get("ttt_results", {}).items()}
        ds_r2 = data.get("baseline_results", {}).get("DS", {}).get("r2")
        plot_ttt_steps(ttt_r, baseline_r2=ds_r2,
                      title="Diagnostic: Zero-Shot TTT on Corn mp6",
                      figures_dir=str(fig_dir))
        print("  Generated diagnostic ttt plot")

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    generate_all_figures_from_experiments()
