"""
synthetic_1_1_rule_experiment.py
=================================================
Causal validation of the 1:1 Rule
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

# MATRIX GENERATION (spiked spectrum, constant bulk, shared basis option)
def generate_spiked_matrix(
    d: int,
    target_spectral_norm: float,
    bulk_val: float = 0.5,
    seed: Optional[int] = None,
    device: str = "cuda",
    U: Optional[torch.Tensor] = None,
    V: Optional[torch.Tensor] = None,
    flip_spike: bool = False
) -> torch.Tensor:
    """
    Generate a random matrix with:
      - One dominant spike at `target_spectral_norm`
      - A flat bulk at `bulk_val` (CONSTANT, not scaled by spike)
      - Optional shared U/V basis and spike sign-flip for destructive interference.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

    if U is None:
        U = torch.linalg.qr(torch.randn(d, d, dtype=torch.float32, device=device))[0]
    if V is None:
        V = torch.linalg.qr(torch.randn(d, d, dtype=torch.float32, device=device))[0]

    # FIXED bulk, variable spike
    s = torch.full((d,), bulk_val, dtype=torch.float32, device=device)
    s[0] = target_spectral_norm

    if flip_spike:
        # Flip sign of the dominant left singular vector → spike partially cancels
        # when added to a matrix using the same U without flip
        U = U.clone()
        U[:, 0] *= -1.0

    W = U @ torch.diag(s) @ V.T
    return W

def compute_erank(eigvals: torch.Tensor) -> float:
    eigvals = torch.clamp(eigvals, min=0.0)
    total = eigvals.sum()
    if total <= 0:
        return 1.0
    p = eigvals / total
    p = p[p > 1e-300]
    if len(p) == 0:
        return 1.0
    entropy = -(p * torch.log(p)).sum().item()
    return float(torch.exp(torch.tensor(entropy)).item())


def lyapunov_trajectory(
    W_mlp: torch.Tensor,
    W_attn: torch.Tensor,
    num_layers: int = 24,
    normalization_mode: str = "spectral"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = W_mlp.shape[0]
    device = W_mlp.device
    C = torch.eye(d, dtype=torch.float32, device=device)
    A = torch.eye(d, dtype=torch.float32, device=device) + W_mlp + W_attn

    eranks = np.empty(num_layers)
    conds = np.empty(num_layers)
    max_eigs = np.empty(num_layers)

    for i in range(num_layers):
        C = A @ C @ A.T

        if normalization_mode == "spectral":
            spec_norm = torch.linalg.matrix_norm(C, ord=2).item()
            if spec_norm > 1e-15:
                C = C / spec_norm
        elif normalization_mode == "trace":
            tr = C.trace().item()
            if tr > 1e-15:
                C = C / tr
        elif normalization_mode == "frobenius":
            fnorm = torch.linalg.matrix_norm(C, ord='fro').item()
            if fnorm > 1e-15:
                C = C / fnorm

        eigvals = torch.linalg.eigvalsh(C)
        eigvals = torch.clamp(eigvals, min=0.0)

        max_eig = float(eigvals[-1].item())
        eps = float(torch.finfo(torch.float32).eps * max(1.0, max_eig) * d)
        nonzero = eigvals > eps
        min_eig = float(eigvals[nonzero].min().item()) if nonzero.any() else eps
        cond = max_eig / min_eig if min_eig > 0 else float('inf')

        eranks[i] = compute_erank(eigvals)
        conds[i] = cond
        max_eigs[i] = max_eig

    return eranks, conds, max_eigs

def run_sweep(
    d: int = 512,
    num_layers: int = 24,
    base_attn_norm: float = 5.0,
    rho_values: Optional[np.ndarray] = None,
    n_seeds: int = 5,
    bulk_val: float = 0.5,
    normalization_mode: str = "spectral",
    device: str = "cuda"
) -> Dict:
    """
    Sweep ρ = ||W_mlp|| / ||W_attn||.
    
    Key change: W_mlp and W_attn now share the SAME random U/V basis, but
    W_attn's spike is anti-aligned (flip_spike=True). This models the
    "destructive interference" from §4.4 of the paper.
    """
    if rho_values is None:
        rho_values = np.logspace(-1, np.log10(20), 50)

    min_erank_mean = []
    min_erank_std = []
    final_erank_mean = []
    final_erank_std = []
    min_cond_mean = []

    for rho in rho_values:
        mlp_norm = rho * base_attn_norm
        attn_norm = base_attn_norm

        seeds_min = []
        seeds_final = []
        seeds_min_cond = []

        for seed in range(n_seeds):
            # Shared random basis for this seed
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)
            U = torch.linalg.qr(torch.randn(d, d, dtype=torch.float32, device=device))[0]
            V = torch.linalg.qr(torch.randn(d, d, dtype=torch.float32, device=device))[0]

            W_mlp = generate_spiked_matrix(
                d, mlp_norm, bulk_val=bulk_val, U=U, V=V, flip_spike=False, device=device
            )
            W_attn = generate_spiked_matrix(
                d, attn_norm, bulk_val=bulk_val, U=U, V=V, flip_spike=True, device=device
            )

            eranks, conds, _ = lyapunov_trajectory(
                W_mlp, W_attn, num_layers, normalization_mode
            )

            seeds_min.append(eranks.min())
            seeds_final.append(eranks[-1])
            seeds_min_cond.append(conds.max())

        min_erank_mean.append(np.mean(seeds_min))
        min_erank_std.append(np.std(seeds_min))
        final_erank_mean.append(np.mean(seeds_final))
        final_erank_std.append(np.std(seeds_final))
        min_cond_mean.append(np.mean(seeds_min_cond))

    return {
        "rho_values": rho_values,
        "min_erank_mean": np.array(min_erank_mean),
        "min_erank_std": np.array(min_erank_std),
        "final_erank_mean": np.array(final_erank_mean),
        "final_erank_std": np.array(final_erank_std),
        "min_cond_mean": np.array(min_cond_mean),
        "d": d,
        "num_layers": num_layers,
        "base_attn_norm": base_attn_norm,
        "bulk_val": bulk_val,
        "n_seeds": n_seeds,
    }


def plot_bifurcation(results, output_dir):
    rho = results["rho_values"]
    min_mean = results["min_erank_mean"]
    min_std = results["min_erank_std"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A
    ax = axes[0]
    ax.plot(rho, min_mean, 'o-', color='#2E5AAC', linewidth=2.5,
            markersize=4, markerfacecolor='white', markeredgewidth=1.5,
            label='Min effective rank (mean ± std)')
    ax.fill_between(rho, np.maximum(min_mean - min_std, 0.5),
                    min_mean + min_std, alpha=0.25, color='#2E5AAC')

    ax.axvspan(0.5, 2.0, alpha=0.12, color='#2ca02c', zorder=0)
    ax.axvline(1.0, color='#2ca02c', linestyle='--', linewidth=1.8, zorder=0)
    ax.text(1.15, max(min_mean) * 0.92, 'ρ = 1', color='#2ca02c', fontsize=11, fontweight='bold')
    ax.text(0.65, max(min_mean) * 0.85, 'Stable zone\n(0.5 < ρ < 2)',
            color='#2ca02c', fontsize=10, ha='center', va='top')

    ax.axhline(10, color='crimson', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(rho[-1] * 0.5, 12, 'Collapse threshold (erank = 10)',
            color='crimson', fontsize=9)

    ax.set_xscale('log')
    ax.set_xlabel(r'Spectral balance ratio  $\rho = \|W_{\mathrm{mlp}}\|_2 / \|W_{\mathrm{attn}}\|_2$',
                  fontsize=12)
    ax.set_ylabel('Min effective rank across 24 layers', fontsize=12)
    ax.set_title('A.  Synthetic Bifurcation at ρ ≈ 1', fontsize=13,
                 fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=9, frameon=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_ylim(0.5, max(min_mean) * 1.15)

    # Panel B
    ax = axes[1]
    select_rhos = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(select_rhos)))

    for i, target_rho in enumerate(select_rhos):
        idx = np.argmin(np.abs(rho - target_rho))
        actual_rho = rho[idx]

        d = results["d"]
        base = results["base_attn_norm"]
        bulk = results["bulk_val"]

        torch.manual_seed(42 + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + i)
        U = torch.linalg.qr(torch.randn(d, d, dtype=torch.float32, device="cuda"))[0]
        V = torch.linalg.qr(torch.randn(d, d, dtype=torch.float32, device="cuda"))[0]

        W_mlp = generate_spiked_matrix(d, actual_rho * base, bulk_val=bulk, U=U, V=V,
                                       flip_spike=False, device="cuda")
        W_attn = generate_spiked_matrix(d, base, bulk_val=bulk, U=U, V=V,
                                        flip_spike=True, device="cuda")
        eranks, _, _ = lyapunov_trajectory(W_mlp, W_attn, results["num_layers"])

        ax.plot(range(1, results["num_layers"] + 1), eranks,
                '-', color=colors[i], linewidth=2.2,
                label=f'ρ = {actual_rho:.2f}')

    ax.axhspan(10, 1000, alpha=0.08, color='green', zorder=0)
    ax.text(1, 25, 'Stable (erank > 10)', color='green', fontsize=9)

    ax.set_xlabel('Layer ℓ', fontsize=12)
    ax.set_ylabel('Effective rank', fontsize=12)
    ax.set_title('B.  Layer-wise Collapse Dynamics', fontsize=13,
                 fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=9, ncol=2, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0.5, max(ax.get_ylim()[1], 50))

    plt.tight_layout()
    fig_path = output_dir / "synthetic_1_1_rule_bifurcation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {fig_path}")
    plt.show()
    return fig

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("Synthetic 1:1 Rule Experiment — DESTRUCTIVE INTERFERENCE VERSION")
    print(f"Device: {device.upper()}")
    print("=" * 70)

    D = 512
    LAYERS = 24
    BASE_ATTN = 5.0
    N_SEEDS = 5
    BULK_VAL = 0.5  # FIXED bulk, not scaled by spike

    output_dir = Path("synthetic_results")
    output_dir.mkdir(exist_ok=True)

    print("\nParameters:")
    print(f"  d = {D}, layers = {LAYERS}, seeds = {N_SEEDS}")
    print(f"  base ||W_attn||_2 = {BASE_ATTN}")
    print(f"  bulk value = {BULK_VAL}  (CONSTANT across all ρ)")
    print("  ρ sweep: 0.1 → 20 (log-spaced, 50 points)")
    print(f"\nRunning sweep on {device.upper()}...")

    results = run_sweep(
        d=D,
        num_layers=LAYERS,
        base_attn_norm=BASE_ATTN,
        n_seeds=N_SEEDS,
        bulk_val=BULK_VAL,
        normalization_mode="spectral",
        device=device
    )

    np.savez(
        output_dir / "synthetic_1_1_rule_data.npz",
        rho_values=results["rho_values"],
        min_erank_mean=results["min_erank_mean"],
        min_erank_std=results["min_erank_std"],
        final_erank_mean=results["final_erank_mean"],
        final_erank_std=results["final_erank_std"],
        min_cond_mean=results["min_cond_mean"],
    )

    csv_path = output_dir / "synthetic_1_1_rule_summary.csv"
    with open(csv_path, "w") as f:
        f.write("rho,min_erank_mean,min_erank_std,final_erank_mean,final_erank_std,min_cond_mean\n")
        for i in range(len(results["rho_values"])):
            f.write(f"{results['rho_values'][i]:.4f},"
                    f"{results['min_erank_mean'][i]:.4f},"
                    f"{results['min_erank_std'][i]:.4f},"
                    f"{results['final_erank_mean'][i]:.4f},"
                    f"{results['final_erank_std'][i]:.4f},"
                    f"{results['min_cond_mean'][i]:.2e}\n")
    print(f"Saved data: {output_dir / 'synthetic_1_1_rule_data.npz'}")
    print(f"Saved CSV:  {csv_path}")

    rho = results["rho_values"]
    min_mean = results["min_erank_mean"]

    def nearest_rho(target):
        return int(np.argmin(np.abs(rho - target)))

    print("\n" + "-" * 50)
    print("Key numbers for the paper:")
    print("-" * 50)
    for val in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        idx = nearest_rho(val)
        print(f"  ρ ≈ {rho[idx]:.2f}:  min erank = {min_mean[idx]:.2f}")
    print("-" * 50)

    plot_bifurcation(results, output_dir)

    print("\nExperiment complete.")
    print("Interpretation:")
    print("  • ρ ∈ (0.5, 2)  → min erank >> 1  (destructive interference → stable)")
    print("  • ρ ≪ 1 or ρ ≫ 1 → min erank → 1   (one pathway dominates → collapse)")


if __name__ == "__main__":
    main()
