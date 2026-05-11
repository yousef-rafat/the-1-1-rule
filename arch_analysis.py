# extract architecture metadata from any HuggingFace model without loading weights.

import json
from transformers import AutoConfig
from typing import Dict, Optional
from lyapunov_analyze import MODELS


def inspect_architecture(model_id: str, token: Optional[str] = None) -> Dict:
    """
    Extract architecture metadata from model config.
    doesn't download weights just config.json
    """
    try:
        config = AutoConfig.from_pretrained(
            model_id, 
            trust_remote_code=True,
            token=token
        )
    except Exception as e:
        return {"model_id": model_id, "error": str(e)}
    
    # Activation function detection
    hidden_act = getattr(config, "hidden_act", None)
    if hidden_act is None:
        hidden_act = getattr(config, "activation_function", "unknown")
    
    # MLP type inference
    intermediate_size = getattr(config, "intermediate_size", None)
    hidden_size = getattr(config, "hidden_size", None)
    
    mlp_type = "unknown"
    if intermediate_size and hidden_size:
        ratio = intermediate_size / hidden_size
        if hasattr(config, "num_local_experts") and config.num_local_experts > 1:
            mlp_type = "MoE"
        elif ratio >= 3.5:
            if "swiglu" in str(hidden_act).lower() or "silu" in str(hidden_act).lower():
                mlp_type = "SwiGLU"
            elif "gelu" in str(hidden_act).lower():
                mlp_type = "GELU-FFN"
            elif "relu" in str(hidden_act).lower():
                mlp_type = "ReLU-FFN"
            else:
                mlp_type = f"FFN({hidden_act})"
        else:
            mlp_type = "Dense"
    
    # Norm type
    norm_type = getattr(config, "norm_type", None)
    if norm_type is None:
        # Check common attributes
        if hasattr(config, "rms_norm_eps"):
            norm_type = "RMSNorm"
        elif hasattr(config, "layer_norm_eps") or hasattr(config, "layer_norm_epsilon"):
            norm_type = "LayerNorm"
        else:
            norm_type = "unknown"
    
    # Position embedding
    pos_emb = "unknown"
    if hasattr(config, "rope_theta"):
        pos_emb = f"RoPE(theta={config.rope_theta})"
    elif hasattr(config, "max_position_embeddings") and not hasattr(config, "rope_theta"):
        pos_emb = "Learned"
    
    # Attention type
    attn_type = "standard"
    if getattr(config, "num_key_value_heads", None) and config.num_key_value_heads != config.num_attention_heads:
        attn_type = f"GQA(kv_heads={config.num_key_value_heads})"
    elif getattr(config, "num_attention_heads", None):
        attn_type = "MHA"
    
    # Head dimension
    num_heads = getattr(config, "num_attention_heads", None)
    head_dim = None
    if hidden_size and num_heads:
        head_dim = hidden_size // num_heads
    
    metadata = {
        "model_id": model_id,
        "hidden_size": hidden_size,
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "mlp_ratio": round(intermediate_size / hidden_size, 2) if intermediate_size and hidden_size else None,
        "activation": str(hidden_act),
        "mlp_type": mlp_type,
        "norm_type": norm_type,
        "position_embedding": pos_emb,
        "attention_type": attn_type,
        "vocab_size": getattr(config, "vocab_size", None),
        "total_params_estimate_M": estimate_params(config),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
        "rope_theta": getattr(config, "rope_theta", None),
        "sliding_window": getattr(config, "sliding_window", None),
    }
    
    return metadata


def estimate_params(config) -> Optional[float]:
    """Rough parameter count estimate in millions."""
    try:
        h = config.hidden_size
        v = config.vocab_size
        l = config.num_hidden_layers
        i = getattr(config, "intermediate_size", 4 * h)
        nh = config.num_attention_heads
        nkv = getattr(config, "num_key_value_heads", nh)
        
        # Embedding
        embed = v * h
        if getattr(config, "tie_word_embeddings", False):
            embed += 0  # shared
        else:
            embed += v * h
        
        # Attention per layer
        # Q: h * h, K: h * (nkv/nh * h), V: same, O: h * h
        head_dim = h // nh
        q_params = h * h
        kv_params = 2 * h * (nkv * head_dim)
        o_params = h * h
        attn_per_layer = q_params + kv_params + o_params
        
        # MLP per layer
        if "swiglu" in str(getattr(config, "hidden_act", "")).lower():
            # gate + up + down
            mlp_per_layer = 3 * h * i
        else:
            # up + down
            mlp_per_layer = 2 * h * i
        
        # Norms (negligible)
        norms_per_layer = 2 * h  # 2 RMSNorm/LayerNorm per layer
        
        total = embed + l * (attn_per_layer + mlp_per_layer + norms_per_layer)
        return round(total / 1e6, 1)
    except Exception:
        return None


def print_architecture_table(models: list, token: Optional[str] = None):
    """Print a formatted comparison table."""
    results = []
    for model_id in models:
        meta = inspect_architecture(model_id, token)
        results.append(meta)
    
    print("\n" + "="*100)
    print("ARCHITECTURE COMPARISON")
    print("="*100)
    print(f"{'Model':<35s} {'d_model':<8s} {'Layers':<7s} {'Act':<10s} {'MLP':<10s} {'Norm':<10s} {'Attn':<12s} {'Est.Params':<10s}")
    print("-"*100)
    
    for r in results:
        if "error" in r:
            print(f"{r['model_id']:<35s} ERROR: {r['error']}")
            continue
        print(f"{r['model_id']:<35s} {r['hidden_size'] or '?'!s:<8s} {r['num_hidden_layers'] or '?'!s:<7s} "
              f"{r['activation']:<10s} {r['mlp_type']:<10s} {r['norm_type']:<10s} "
              f"{r['attention_type']:<12s} {r['total_params_estimate_M'] or '?'!s:<10s}M")
    
    return results


if __name__ == "__main__":
    import os
    TOKEN = os.environ.get("HF_TOKEN", None)
    
    metadata = print_architecture_table(MODELS, token=TOKEN)
    
    with open("architecture_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\nSaved metadata to architecture_metadata.json")
