#!/usr/bin/env python3
"""
============================
    ||W_mlp||_2 / ||W_attn||_2 ≈ 1  →  Stable (no collapse)
    ||W_mlp||_2 / ||W_attn||_2 >> 1  →  Collapse
    ||W_mlp||_2 / ||W_attn||_2 << 1  →  Delayed collapse

Usage:
    python analyze_spectral_balance.py --results results_with_attention --output plots/
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_results(results_dir):
    """Load all JSON spectra files."""
    data = {}
    for json_file in Path(results_dir).glob("*_spectra.json"):
        with open(json_file) as f:
            d = json.load(f)
        key = json_file.stem.replace("_spectra", "")
        data[key] = d
    return data


def extract_metrics(data):
    """Extract per-layer metrics and compute ratios."""
    processed = {}
    for key, d in data.items():
        metrics = d.get("layer_metrics", [])
        if not metrics:
            continue

        mlp_sn = [m.get("mlp_spectral_norm", 0) for m in metrics]
        attn_sn = [m.get("attn_spectral_norm", 0) for m in metrics]
        A_sn = [m.get("A_spectral_norm", 0) for m in metrics]
        erank = d.get("effective_ranks", [])
        cond = d.get("condition_numbers", [])

        ratio = [m / max(a, 1e-6) for m, a in zip(mlp_sn, attn_sn)]

        processed[key] = {
            "model_id": d["model_id"],
            "hidden_dim": d["hidden_dim"],
            "layers": d["num_layers"],
            "mlp_sn": mlp_sn,
            "attn_sn": attn_sn,
            "A_sn": A_sn,
            "ratio": ratio,
            "erank": erank,
            "cond": cond,
            "min_erank": min(erank) if erank else 0,
            "mean_erank": np.mean(erank) if erank else 0,
            "max_cond": max(cond) if cond else 0,
            "mean_ratio": np.mean(ratio) if ratio else 0,
            "stable": min(erank) > 10 if erank else False,
            "collapse_layer": next((i for i, e in enumerate(erank) if e < 2), len(erank)) if erank else 0,
        }
    return processed


def plot_spectral_norms(processed, output_dir):
    """Figure 1: Spectral norms across layers."""
    n_models = len(processed)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols  # ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    fig.suptitle("Spectral Pathway Norms Across Layers", fontsize=15, fontweight='bold', y=1.02)

    items = list(processed.items())
    for idx, (key, d) in enumerate(items):
        ax = axes[idx // n_cols, idx % n_cols]
        layers = np.arange(d["layers"])

        ax.semilogy(layers, d["mlp_sn"], 'o-', color='#e74c3c', label='||W_mlp||₂', markersize=4, linewidth=1.5)
        ax.semilogy(layers, d["attn_sn"], 's-', color='#3498db', label='||W_attn||₂', markersize=4, linewidth=1.5)

        if not d["stable"]:
            ax.axvline(x=d["collapse_layer"], color='gray', linestyle='--', alpha=0.5)

        ax.set_title(d["model_id"].split("/")[-1], fontweight='bold')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Spectral Norm")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        ratio_text = f"Avg ratio: {d['mean_ratio']:.2f}"
        ax.text(0.02, 0.98, ratio_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Turn off unused subplots
    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_spectral_norms.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/fig1_spectral_norms.png")


def plot_ratio_trajectories(processed, output_dir):
    """Figure 2: Ratio across layers."""
    fig, axes = plt.subplots(1, len(processed), figsize=(3.5 * len(processed), 4.5))
    fig.suptitle("Spectral Balance Ratio: ||W_mlp||₂ / ||W_attn||₂", fontsize=14, fontweight='bold', y=1.05)

    if len(processed) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(processed)))

    for idx, ((key, d), color) in enumerate(zip(processed.items(), colors)):
        ax = axes[idx]
        layers = np.arange(d["layers"])

        ax.semilogy(layers, d["ratio"], 'o-', color=color, markersize=5, linewidth=2)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhspan(0.5, 2.0, alpha=0.15, color='green')

        if not d["stable"]:
            ax.axvline(x=d["collapse_layer"], color='gray', linestyle=':', linewidth=2, alpha=0.6)
        else:
            ax.text(0.5, 0.95, '✅ STABLE', transform=ax.transAxes, fontsize=10,
                    fontweight='bold', color='green', ha='center', va='top')

        ax.set_title(d["model_id"].split("/")[-1], fontweight='bold', fontsize=11)
        ax.set_xlabel("Layer", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Ratio (log scale)", fontsize=10)
        ax.set_ylim([0.05, 500])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_ratio_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/fig2_ratio_trajectories.png")


def plot_phase_diagram(processed, output_dir):
    """Figure 3: Phase diagram and stability map."""
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.25)

    colors = plt.cm.tab10(np.linspace(0, 1, len(processed)))

    # Left: Layer-level scatter
    ax1 = fig.add_subplot(gs[0])
    for (key, d), color in zip(processed.items(), colors):
        marker = 'o' if d['stable'] else 'x'
        ax1.scatter(d["ratio"], d["erank"], c=[color], s=60, alpha=0.6,
                   edgecolors='black', linewidth=0.5, label=d["model_id"].split("/")[-1], marker=marker, zorder=3)

    ax1.axvline(x=0.5, color='#f39c12', linestyle='--', alpha=0.5)
    ax1.axvline(x=2.0, color='#27ae60', linestyle='--', alpha=0.5)
    ax1.axvline(x=5.0, color='#e74c3c', linestyle='--', alpha=0.5)

    ax1.text(0.15, 1200, 'Attention\nDominates', fontsize=8, ha='center', color='#f39c12', style='italic')
    ax1.text(1.0, 1200, 'STABLE\nZONE', fontsize=10, ha='center', color='#27ae60', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.6))
    ax1.text(15, 1200, 'MLP Dominates\n→ Collapse', fontsize=8, ha='center', color='#e74c3c', style='italic')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$||W_{\mathrm{mlp}}||_2 \,/\, ||W_{\mathrm{attn}}||_2$ (log scale)', fontsize=12)
    ax1.set_ylabel('Effective Rank (log scale)', fontsize=12)
    ax1.set_title('(a) Layer-Level Phase Diagram', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.03, 50])
    ax1.set_ylim([0.8, 2000])

    # Right: Model-level map
    ax2 = fig.add_subplot(gs[1])
    for (key, d), color in zip(processed.items(), colors):
        marker = 'o' if d['stable'] else 'X'
        size = 200 if d['stable'] else 150
        ax2.scatter(d['mean_ratio'], d['min_erank'], c=[color], s=size,
                   marker=marker, edgecolors='black', linewidth=1.5, zorder=5)

        offset = (0.15, 15) if d['mean_ratio'] < 1 else (0.4, -8)
        status = "STABLE" if d['stable'] else f"Collapse L{d['collapse_layer']}"
        ax2.annotate(f"{d['model_id'].split('/')[-1]}\n{status}",
                    (d['mean_ratio'], d['min_erank']),
                    xytext=offset, textcoords='offset points',
                    fontsize=9, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax2.axvline(x=0.5, color='#f39c12', linestyle='--', alpha=0.5)
    ax2.axvline(x=2.0, color='#27ae60', linestyle='--', alpha=0.5)
    ax2.axvline(x=5.0, color='#e74c3c', linestyle='--', alpha=0.5)
    ax2.axvspan(0.5, 2.0, alpha=0.1, color='green')
    ax2.axvspan(5, 30, alpha=0.08, color='red')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\mathrm{Avg}\,||W_{\mathrm{mlp}}||_2 \,/\, ||W_{\mathrm{attn}}||_2$ (log scale)', fontsize=12)
    ax2.set_ylabel('Minimum Effective Rank (log scale)', fontsize=12)
    ax2.set_title('(b) Model-Level Stability Map', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.05, 30])
    ax2.set_ylim([0.5, 2000])

    plt.suptitle(r'Spectral Balance Hypothesis: $\frac{||W_{\mathrm{mlp}}||_2}{||W_{\mathrm{attn}}||_2} \approx 1$ Prevents Lyapunov Collapse',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_phase_diagram.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/fig3_phase_diagram.png")


def plot_summary_dashboard(processed, output_dir):
    """Figure 4: Comprehensive dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.25)

    colors = plt.cm.tab10(np.linspace(0, 1, len(processed)))

    # (a) Effective rank
    ax1 = fig.add_subplot(gs[0, 0])
    for (key, d), color in zip(processed.items(), colors):
        layers = np.arange(d["layers"])
        ax1.semilogy(layers, d["erank"], 'o-', color=color, label=d["model_id"].split("/")[-1], markersize=3, linewidth=1.5)
    ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Effective Rank (log scale)")
    ax1.set_title("(a) Effective Rank Trajectories", fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # (b) Condition number
    ax2 = fig.add_subplot(gs[0, 1])
    for (key, d), color in zip(processed.items(), colors):
        layers = np.arange(d["layers"])
        ax2.semilogy(layers, d["cond"], 's-', color=color, label=d["model_id"].split("/")[-1], markersize=3, linewidth=1.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Condition Number (log scale)")
    ax2.set_title("(b) Condition Number Trajectories", fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # (c) Ratio trajectories
    ax3 = fig.add_subplot(gs[1, 0])
    for (key, d), color in zip(processed.items(), colors):
        layers = np.arange(d["layers"])
        ax3.semilogy(layers, d["ratio"], 'D-', color=color, label=d["model_id"].split("/")[-1], markersize=3, linewidth=1.5)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhspan(0.5, 2.0, alpha=0.15, color='green')
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("||W_mlp||₂ / ||W_attn||₂ (log scale)")
    ax3.set_title("(c) Spectral Balance Ratio Trajectories", fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.03, 50])

    # (d) Bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    summary = sorted([(d['mean_ratio'], d['min_erank'], d['stable'], d['model_id'].split("/")[-1], c)
                      for (_, d), c in zip(processed.items(), colors)], key=lambda x: x[0])

    labels = [s[3] for s in summary]
    ratios = [s[0] for s in summary]
    bar_colors = [s[4] for s in summary]

    bars = ax4.barh(labels, ratios, color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    for bar, s in zip(bars, summary):
        width = bar.get_width()
        ax4.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
        icon = '✅' if s[2] else '❌'
        ax4.text(0.1, bar.get_y() + bar.get_height()/2, icon, ha='left', va='center', fontsize=12)

    ax4.axvline(x=0.5, color='#f39c12', linestyle='--', alpha=0.7)
    ax4.axvline(x=2.0, color='#27ae60', linestyle='--', alpha=0.7)
    ax4.axvline(x=5.0, color='#e74c3c', linestyle='--', alpha=0.7)
    ax4.set_xlabel("Avg ||W_mlp||₂ / ||W_attn||₂")
    ax4.set_title("(d) Spectral Balance by Model", fontweight='bold')
    ax4.set_xlim([0, 20])
    ax4.grid(True, alpha=0.3, axis='x')

    plt.suptitle("Spectral Balance Controls Residual Stream Geometry in Transformers",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(f"{output_dir}/fig4_summary_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/fig4_summary_dashboard.png")


def print_statistics(processed):
    """Print correlation statistics."""
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)

    all_ratios = []
    all_eranks = []
    for d in processed.values():
        all_ratios.extend(d["ratio"])
        all_eranks.extend(d["erank"])

    all_ratios = np.array(all_ratios)
    all_eranks = np.array(all_eranks)

    # Log-log correlation
    valid = (all_ratios > 0) & (all_eranks > 0)
    log_ratio = np.log10(all_ratios[valid])
    log_erank = np.log10(all_eranks[valid])

    corr = np.corrcoef(log_ratio, log_erank)[0, 1]
    print(f"\nLog-log correlation (ratio vs erank): r = {corr:.3f}")

    # Model-level correlation
    mean_ratios = [d["mean_ratio"] for d in processed.values()]
    min_eranks = [d["min_erank"] for d in processed.values()]
    corr_model = np.corrcoef(np.log10(mean_ratios), np.log10(min_eranks))[0, 1]
    print(f"Model-level correlation: r = {corr_model:.3f}")

    print("\n" + "-"*70)
    print(f"{'Model':<30s} {'Ratio':<10s} {'Min ERank':<12s} {'Status':<15s}")
    print("-"*70)
    for d in processed.values():
        status = "STABLE" if d["stable"] else f"Collapse L{d['collapse_layer']}"
        print(f"{d['model_id']:<30s} {d['mean_ratio']:<10.2f} {d['min_erank']:<12.1f} {status:<15s}")


def main():
    parser = argparse.ArgumentParser(description="Analyze spectral balance in transformer residual streams")
    parser.add_argument("--results", type=str, default="results_with_attention", help="Directory with JSON spectra")
    parser.add_argument("--output", type=str, default="plots", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading results from: {args.results}")
    data = load_results(args.results)
    print(f"Loaded {len(data)} models")

    processed = extract_metrics(data)

    print("\nGenerating plots...")
    plot_spectral_norms(processed, args.output)
    plot_ratio_trajectories(processed, args.output)
    plot_phase_diagram(processed, args.output)
    plot_summary_dashboard(processed, args.output)

    print_statistics(processed)

    print(f"\n✅ All plots saved to: {args.output}/")


if __name__ == "__main__":
    main()
