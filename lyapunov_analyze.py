
"""
Lyapunov Residual Stream Analyzer For Attention & MLP
"""

import torch
import torch.nn as nn
import json
import csv
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODELS = [
    "google/gemma-3-270m",
    "google/gemma-2-2b",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "microsoft/phi-2",
    "HuggingFaceTB/SmolLM2-360M",
]


# =============================================================================
# CONFIGURATION
# =============================================================================

class AnalysisConfig:
    NORMALIZATION_MODE = "spectral"
    USE_DOUBLE_PRECISION = True
    DIAGNOSE_SPECTRA = True
    INCLUDE_ATTENTION = True


# =============================================================================
# ARCHITECTURE DISCOVERY
# =============================================================================

def discover_blocks(model) -> List[nn.Module]:
    """Auto-discover transformer blocks."""
    root = model.module if hasattr(model, "module") else model

    paths = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("transformer", "layers"),
        ("gpt_neox", "layers"),
    ]

    for path in paths:
        obj = root
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            return list(obj)

    # Fallback
    for name, module in root.named_modules():
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) > 0:
            first = module[0]
            has_attn = any('attn' in n or 'attention' in n for n, _ in first.named_modules())
            has_mlp = any('mlp' in n or 'ffn' in n or 'feed' in n for n, _ in first.named_modules())
            if has_attn or has_mlp:
                return list(module)

    raise ValueError("Cannot discover transformer blocks.")


def discover_mlp_weights(block: nn.Module) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:
    """Auto-discover MLP up/down weights. Returns (W_up, W_down, hidden_dim)."""
    mlp_up = None
    mlp_down = None
    hidden_dim = None

    up_patterns = ["up_proj", "fc1", "gate_up_proj", "c_fc", "dense_h_to_4h", "w1", "wi"]
    down_patterns = ["down_proj", "fc2", "dense_4h_to_h", "c_proj", "w2", "wo"]
    gate_up_patterns = ["gate_up_proj"]

    for name, module in block.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        name_lower = name.lower()
        weight = module.weight.data

        is_gate_up = any(p in name_lower for p in gate_up_patterns)
        is_up = any(p in name_lower for p in up_patterns) or is_gate_up
        is_down = any(p in name_lower for p in down_patterns)

        if is_up:
            if is_gate_up and weight.shape[0] % 2 == 0:
                mid = weight.shape[0] // 2
                mlp_up = weight[mid:, :].cpu().float()
            else:
                mlp_up = weight.cpu().float()
            if hidden_dim is None:
                hidden_dim = weight.shape[1]

        if is_down:
            mlp_down = weight.cpu().float()

    return mlp_up, mlp_down, hidden_dim


def discover_attention_output_proj(block: nn.Module, hidden_dim: int) -> Optional[torch.Tensor]:
    """
    For the Lyapunov analysis, we need W_o @ W_o^T which maps hidden_dim -> hidden_dim,
    or we can use W_o directly if it\'s square. For non-square W_o, we use W_o @ W_o^T.
    """
    attn_out_patterns = [
        "o_proj", "out_proj", "dense", "attn.out", "self_attn.o_proj",
        "attention.dense", "attn.c_proj"
    ]

    for name, module in block.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        name_lower = name.lower()
        weight = module.weight.data

        is_output = any(p in name_lower for p in attn_out_patterns)

        if is_output:
            w = weight.cpu().float()

            # W_o typically has shape [hidden_dim, num_heads * head_dim]
            # For the residual stream: x + Attn(x) = x + softmax(QK^T)V @ W_o
            # The linearized effect is x @ W_o^T (if V = x @ W_v)
            # But W_o may not be square. We compute the effective contribution.

            if w.shape[0] == hidden_dim:
                # Standard case: W_o maps from attention space to hidden_dim
                # For covariance propagation, we need W_o @ W_o^T if non-square
                # or W_o directly if square
                if w.shape[1] == hidden_dim:
                    return w  # Square, use directly
                else:
                    # Non-square (GQA): compute W_o @ W_o^T to get [hidden_dim, hidden_dim]
                    return w @ w.T
            elif w.shape[1] == hidden_dim:
                # Transposed orientation
                w_t = w.T
                if w_t.shape[0] == w_t.shape[1]:
                    return w_t
                else:
                    return w_t @ w_t.T

    return None


def discover_all_block_weights(block: nn.Module, b_idx: int) -> Dict:
    """Discover all relevant weight matrices from a transformer block."""
    mlp_up, mlp_down, hidden_dim = discover_mlp_weights(block)
    attn_out = None

    if hidden_dim is not None:
        attn_out = discover_attention_output_proj(block, hidden_dim)

    return {
        "mlp_up": mlp_up,
        "mlp_down": mlp_down,
        "attn_out": attn_out,
        "hidden_dim": hidden_dim,
    }


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def diagnose_spectrum(eigvals: torch.Tensor, layer_idx: int, hidden_dim: int, model_short: str = "") -> Dict:
    """Analyze spectral structure."""
    eigvals_np = eigvals.cpu().numpy()
    eigvals_sorted = np.sort(eigvals_np)[::-1]

    max_eig = float(eigvals_sorted[0])
    p = eigvals_sorted / eigvals_sorted.sum()
    pr = float(1.0 / (p ** 2).sum())
    erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))

    noise_floor = float(np.finfo(np.float64 if eigvals.dtype == torch.float64 else np.float32).eps * max_eig * hidden_dim)
    bulk = eigvals_sorted[1:]
    num_above_noise = int((bulk > noise_floor).sum())
    spike_ratio = float(max_eig / eigvals_sorted[min(9, len(eigvals_sorted)-1)]) if len(eigvals_sorted) > 1 else float('inf')

    if num_above_noise < 5 and max_eig > 0.99:
        classification = "COLLAPSE"
    elif spike_ratio > 1e3 and num_above_noise > 50:
        classification = "SPIKE"
    elif num_above_noise > hidden_dim * 0.5:
        classification = "BULK"
    else:
        classification = "TRANSITION"

    print(f"\n  [DIAG] {model_short} L{layer_idx}: {classification}")
    print(f"    max_eig={max_eig:.4f}, spike_ratio={spike_ratio:.2e}, above_noise={num_above_noise}/{len(bulk)}")
    print(f"    PR={pr:.2f}, erank={erank:.2f}")

    return {
        "classification": classification,
        "max_eig": max_eig,
        "spike_ratio": spike_ratio,
        "num_above_noise": num_above_noise,
        "participation_ratio": pr,
        "effective_rank": erank,
    }


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def compute_lyapunov_spectrum(
    C_prev: torch.Tensor,
    W_up: Optional[torch.Tensor],
    W_down: Optional[torch.Tensor],
    W_attn_out: Optional[torch.Tensor],
    hidden_dim: int,
    config: AnalysisConfig = None
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float, Optional[Dict], Dict]:
    """
    Compute Lyapunov covariance with full residual stream.

    A = I + W_mlp_eff + W_attn_eff
    where W_mlp_eff = W_down @ W_up
    and W_attn_eff = W_attn_out (or W_o @ W_o^T for GQA)
    """
    if config is None:
        config = AnalysisConfig()

    dtype = torch.float64 if config.USE_DOUBLE_PRECISION else torch.float32
    C_prev = C_prev.to(dtype)
    device = C_prev.device

    # Build MLP contribution
    W_mlp_eff = None
    if W_up is not None and W_down is not None:
        W_up = W_up.to(dtype)
        W_down = W_down.to(dtype)
        W_mlp_eff = W_down @ W_up

    # Build attention contribution
    W_attn_eff = None
    if config.INCLUDE_ATTENTION and W_attn_out is not None:
        W_attn_eff = W_attn_out.to(dtype)
        # Ensure shape is [hidden_dim, hidden_dim]
        if W_attn_eff.shape != (hidden_dim, hidden_dim):
            print(f"    WARNING: W_attn_eff shape {W_attn_eff.shape} != ({hidden_dim}, {hidden_dim}), skipping attention")
            W_attn_eff = None

    # Build transition matrix
    A = torch.eye(hidden_dim, dtype=dtype, device=device)

    if W_mlp_eff is not None:
        A = A + W_mlp_eff

    if W_attn_eff is not None:
        A = A + W_attn_eff

    # Compute spectral norms
    metrics = {}
    if W_mlp_eff is not None:
        metrics["mlp_spectral_norm"] = float(torch.linalg.matrix_norm(W_mlp_eff, ord=2).item())
        metrics["mlp_frobenius_norm"] = float(torch.linalg.matrix_norm(W_mlp_eff, ord='fro').item())
    else:
        metrics["mlp_spectral_norm"] = 0.0
        metrics["mlp_frobenius_norm"] = 0.0

    if W_attn_eff is not None:
        metrics["attn_spectral_norm"] = float(torch.linalg.matrix_norm(W_attn_eff, ord=2).item())
        metrics["attn_frobenius_norm"] = float(torch.linalg.matrix_norm(W_attn_eff, ord='fro').item())
    else:
        metrics["attn_spectral_norm"] = 0.0
        metrics["attn_frobenius_norm"] = 0.0

    metrics["A_spectral_norm"] = float(torch.linalg.matrix_norm(A, ord=2).item())

    # Propagate covariance
    C_next = A @ C_prev @ A.T

    # Normalization
    if config.NORMALIZATION_MODE == "trace":
        trace = C_next.trace()
        if trace > 1e-15:
            C_next = C_next / trace
    elif config.NORMALIZATION_MODE == "spectral":
        try:
            temp_eigvals = torch.linalg.eigvalsh(C_next)
            spec_norm = temp_eigvals[-1].item()
        except RuntimeError:
            spec_norm = torch.linalg.matrix_norm(C_next, ord=2).item()
        if spec_norm > 1e-15:
            C_next = C_next / spec_norm
    elif config.NORMALIZATION_MODE == "frobenius":
        fnorm = torch.linalg.matrix_norm(C_next, ord='fro')
        if fnorm > 1e-15:
            C_next = C_next / fnorm

    # Eigendecomposition
    try:
        eigvals = torch.linalg.eigvalsh(C_next)
    except RuntimeError:
        _, s, _ = torch.linalg.svd(C_next)
        eigvals = s ** 2
        eigvals, _ = torch.sort(eigvals)

    eigvals = torch.clamp(eigvals, min=0.0)

    max_eig = float(eigvals[-1].item())
    eps = float(torch.finfo(dtype).eps * max(1.0, max_eig) * hidden_dim)
    nonzero_mask = eigvals > eps
    if nonzero_mask.any():
        min_eig = float(eigvals[nonzero_mask].min().item())
    else:
        min_eig = eps

    cond = max_eig / min_eig if min_eig > 0 else float('inf')

    eigvals_sum = eigvals.sum().item()
    if eigvals_sum > 0:
        p = eigvals / eigvals_sum
        p_nonzero = p[p > 1e-300]
        if len(p_nonzero) > 0:
            entropy = float(-(p_nonzero * torch.log(p_nonzero)).sum().item())
            erank = float(torch.exp(torch.tensor(entropy)).item())
        else:
            erank = 1.0
    else:
        erank = 1.0

    diagnostic = None
    return C_next.float(), eigvals.float(), max_eig, min_eig, cond, erank, diagnostic, metrics


# =============================================================================
# MODEL ANALYSIS
# =============================================================================

def analyze_model(
    model_id: str,
    device: str = "cuda:0",
    dtype=torch.float32,
    token: Optional[str] = None,
    config: AnalysisConfig = None,
    output_dir: Optional[str] = None
) -> Dict:
    """Compute Lyapunov spectra."""
    if config is None:
        config = AnalysisConfig()

    print(f"\n{'='*60}")
    print(f"Analyzing: {model_id}")
    print(f"  Norm: {config.NORMALIZATION_MODE}, Double: {config.USE_DOUBLE_PRECISION}")
    print(f"  Include Attention: {config.INCLUDE_ATTENTION}")
    print(f"{'='*60}")

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": device,
        "trust_remote_code": True,
    }
    if token:
        load_kwargs["token"] = token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (ValueError, AttributeError, TypeError) as e:
        err_str = str(e).lower()
        if "pad_token_id" in err_str or "padding_idx" in err_str:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=token)
            cfg.pad_token_id = pad_id
            load_kwargs["config"] = cfg
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        else:
            raise

    model.eval()
    blocks = discover_blocks(model)
    num_layers = len(blocks)

    block_weights = {}
    hidden_dim = None

    for b_idx, block in enumerate(blocks):
        weights = discover_all_block_weights(block, b_idx)
        if weights["hidden_dim"] is not None:
            block_weights[b_idx] = weights
            if hidden_dim is None:
                hidden_dim = weights["hidden_dim"]

    if hidden_dim is None:
        raise ValueError(f"Could not determine hidden dimension for {model_id}")

    print(f"  Hidden dim: {hidden_dim}, Layers: {num_layers}")
    print(f"  Blocks with MLP: {sum(1 for w in block_weights.values() if w['mlp_up'] is not None)}/{num_layers}")
    print(f"  Blocks with Attn: {sum(1 for w in block_weights.values() if w['attn_out'] is not None)}/{num_layers}")

    device_for_cov = "cpu"
    C = [torch.eye(hidden_dim, device=device_for_cov, dtype=torch.float64 if config.USE_DOUBLE_PRECISION else torch.float32)]

    condition_numbers = []
    max_eigenvalues = []
    min_eigenvalues = []
    effective_ranks = []
    all_eigvals = []
    diagnostics = []
    layer_metrics = []

    for b_idx in range(num_layers):
        weights = block_weights.get(b_idx, {})
        W_up = weights.get("mlp_up")
        W_down = weights.get("mlp_down")
        W_attn_out = weights.get("attn_out")

        if W_up is not None:
            W_up = W_up.to(device_for_cov)
        if W_down is not None:
            W_down = W_down.to(device_for_cov)
        if W_attn_out is not None:
            W_attn_out = W_attn_out.to(device_for_cov)

        C_next, eigvals, max_eig, min_eig, cond, erank, _, metrics = compute_lyapunov_spectrum(
            C[-1], W_up, W_down, W_attn_out, hidden_dim, config
        )

        if config.DIAGNOSE_SPECTRA and (erank < 10 or cond > 1e10 or max_eig > 0.99):
            diag = diagnose_spectrum(eigvals, b_idx, hidden_dim, model_id.split("/")[-1])
            diagnostics.append({"layer": b_idx, **diag})

        C.append(C_next)
        all_eigvals.append([float(v) for v in eigvals.cpu().numpy().tolist()])

        condition_numbers.append(cond)
        max_eigenvalues.append(max_eig)
        min_eigenvalues.append(min_eig)
        effective_ranks.append(erank)
        layer_metrics.append(metrics)

        attn_str = f"attn_sn={metrics['attn_spectral_norm']:.3f}" if metrics['attn_spectral_norm'] > 0 else "attn=NA"
        mlp_str = f"mlp_sn={metrics['mlp_spectral_norm']:.3f}" if metrics['mlp_spectral_norm'] > 0 else "mlp=NA"
        print(f"  L{b_idx:2d}: cond={cond:.2e}, erank={erank:.2f}, A_sn={metrics['A_spectral_norm']:.3f}, {mlp_str}, {attn_str}")

    results = {
        "model_id": model_id,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "normalization_mode": config.NORMALIZATION_MODE,
        "include_attention": config.INCLUDE_ATTENTION,
        "condition_numbers": condition_numbers,
        "max_eigenvalues": max_eigenvalues,
        "min_eigenvalues": min_eigenvalues,
        "effective_ranks": effective_ranks,
        "layer_metrics": layer_metrics,
        "all_eigenvalues": all_eigvals,
        "diagnostics": diagnostics,
    }

    if output_dir:
        save_results(results, output_dir)

    return results


def save_results(results: Dict, output_dir: str):
    """Save results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    model_safe = results["model_id"].replace("/", "_")

    json_path = os.path.join(output_dir, f"{model_safe}_spectra.json")
    json_results = {
        "model_id": results["model_id"],
        "num_layers": results["num_layers"],
        "hidden_dim": results["hidden_dim"],
        "normalization_mode": results.get("normalization_mode", "trace"),
        "include_attention": results.get("include_attention", False),
        "condition_numbers": [float(v) for v in results["condition_numbers"]],
        "max_eigenvalues": [float(v) for v in results["max_eigenvalues"]],
        "min_eigenvalues": [float(v) for v in results["min_eigenvalues"]],
        "effective_ranks": [float(v) for v in results["effective_ranks"]],
        "layer_metrics": results.get("layer_metrics", []),
        "diagnostics": results.get("diagnostics", []),
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    csv_path = os.path.join(output_dir, f"{model_safe}_layers.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "condition_number", "max_eigenvalue", "min_eigenvalue", 
            "effective_rank", "A_spectral_norm", "mlp_spectral_norm", "attn_spectral_norm"
        ])
        for i in range(results["num_layers"]):
            metrics = results.get("layer_metrics", [{}])[i] if i < len(results.get("layer_metrics", [])) else {}
            writer.writerow([
                i, results["condition_numbers"][i], results["max_eigenvalues"][i],
                results["min_eigenvalues"][i], results["effective_ranks"][i],
                metrics.get("A_spectral_norm", 0),
                metrics.get("mlp_spectral_norm", 0),
                metrics.get("attn_spectral_norm", 0),
            ])
    print(f"  Saved CSV: {csv_path}")


def print_summary(results: Dict):
    """Print formatted summary."""
    print(f"\n  SUMMARY for {results['model_id']}:")
    print(f"    Max condition number: {max(results['condition_numbers']):.2e}")
    print(f"    Mean condition number: {sum(results['condition_numbers'])/len(results['condition_numbers']):.2e}")
    print(f"    Mean effective rank: {sum(results['effective_ranks'])/len(results['effective_ranks']):.2f}")
    print(f"    Max effective rank: {max(results['effective_ranks']):.2f}")
    print(f"    Min effective rank: {min(results['effective_ranks']):.2f}")

    metrics = results.get("layer_metrics", [])
    if metrics:
        avg_mlp_sn = sum(m.get("mlp_spectral_norm", 0) for m in metrics) / len(metrics)
        avg_attn_sn = sum(m.get("attn_spectral_norm", 0) for m in metrics) / len(metrics)
        avg_A_sn = sum(m.get("A_spectral_norm", 0) for m in metrics) / len(metrics)
        print(f"    Avg MLP spectral norm: {avg_mlp_sn:.3f}")
        print(f"    Avg Attn spectral norm: {avg_attn_sn:.3f}")
        print(f"    Avg A spectral norm: {avg_A_sn:.3f}")

    if results.get("diagnostics"):
        print(f"    Flagged layers: {len(results['diagnostics'])}")


def cross_model_comparison(all_results: Dict[str, Dict]):
    """Print cross-model comparison."""
    print("\n" + "="*90)
    print("CROSS-MODEL COMPARISON")
    print("="*90)
    print(f"{'Model':<45s} {'Max Cond':<14s} {'Mean ERank':<12s} {'Min ERank':<12s} {'Avg ||A||₂':<12s}")
    print("-"*90)
    for model_id, results in all_results.items():
        max_cond = max(results['condition_numbers'])
        mean_erank = sum(results['effective_ranks']) / len(results['effective_ranks'])
        min_erank = min(results['effective_ranks'])
        metrics = results.get("layer_metrics", [])
        avg_A_sn = sum(m.get("A_spectral_norm", 0) for m in metrics) / len(metrics) if metrics else 0
        print(f"{model_id:<45s} {max_cond:<14.2e} {mean_erank:<12.2f} {min_erank:<12.2f} {avg_A_sn:<12.3f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    TOKEN = os.environ.get("HF_TOKEN", "TOKEN_HERE")

    config = AnalysisConfig()
    config.NORMALIZATION_MODE = "spectral"
    config.DIAGNOSE_SPECTRA = True
    config.INCLUDE_ATTENTION = True

    all_results = {}
    output_base = "results_with_attention"

    for model_id in MODELS:
        try:
            results = analyze_model(model_id, token=TOKEN, config=config, output_dir=output_base)
            all_results[model_id] = results
            print_summary(results)
        except Exception as e:
            print(f"\n  FAILED for {model_id}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        cross_model_comparison(all_results)
