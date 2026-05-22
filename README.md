
# The 1:1 Rule: Lyapunov Analysis of Spectral Balance in Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- **The 1:1 Rule:** For geometrically stable residual streams, the spectral norms of the MLP and attention pathways appear to require matching to within a factor of two.
-  When the spectral balance ratio ρ = ‖W_mlp‖₂ / ‖W_attn‖₂ drifts outside [0.5, 2], the residual stream tends to collapse to effective rank ≈ 1.

---

## Overview

This repository contains the official implementation for analyzing the spectral geometry of transformer residual streams via a Lyapunov covariance propagation framework. We analyze pretrained decoder-only language models and demonstrate that the **spectral balance ratio** $\\rho$ between the MLP effective matrix and the attention output projection is a strong predictor of geometric stability.  

<br>
 
<img width="4010" height="2800" alt="fig4_new" src="https://github.com/user-attachments/assets/328b43f1-a921-4540-849b-0f512848baef" />

### Key Features

- **Zero-data analysis:** Inspect weight matrices directly without forward passes on data.
- **Lyapunov iterator:** Propagate covariance through the residual stream layer-by-layer.
- **Synthetic causal validation:** Controlled random-matrix experiments to test the 1:1 Rule independently of training data or architecture.
- **Multi-architecture support:** Tested on Gemma, Qwen, Llama, Phi, Pythia, and SmolLM families (MHA, GQA, GELU, GeGLU, SwiGLU, RMSNorm, LayerNorm).
- **Reproducible figures:** Scripts to regenerate all paper figures and tables.

---

## Installation

```bash
git clone https://anonymous.4open.science/r/the-1-1-rule-FD86
cd the-1-1-rule
pip install -e .
```

---

## Models

The following models are analyzed in the paper:

| Model | Params | Hidden Dim | Layers | Activation | Norm | Attention |
|---|---|---|---|---|---|---|
| Gemma-3-270M | 270M | 640 | 18 | SwiGLU | RMSNorm | GQA |
| Gemma-2-2B | 2.2B | 2304 | 26 | GeGLU | RMSNorm | GQA |
| Qwen2.5-0.5B | 500M | 896 | 24 | SwiGLU | RMSNorm | GQA |
| Llama-3.2-1B | 1.1B | 2048 | 16 | SwiGLU | RMSNorm | GQA |
| Phi-2 | 2.7B | 2560 | 32 | GELU | LayerNorm | MHA |
| Pythia-1.4B | 1.4B | 2048 | 24 | GELU | LayerNorm | MHA |
| SmolLM2-360M | 360M | 960 | 32 | SwiGLU | RMSNorm | GQA |


---
<img width="7316" height="1432" alt="fig2_new" src="https://github.com/user-attachments/assets/27f741df-4a81-44d6-a190-54c62ffe5000" />  

## Methodology

### 1. Weight Extraction

For each layer $\\ell$:

- **MLP:** Compute the effective matrix $W_{\\mathrm{mlp}}^{(\\ell)} = W_{\\mathrm{down}}^{(\\ell)} W_{\\mathrm{up}}^{(\\ell)}$ (incorporates SwiGLU gating where applicable).
- **Attention:** Use the output projection $W_{\\mathrm{o}}^{(\\ell)}$ (for GQA, we compute the covariance contribution $W_{\\mathrm{o}} W_{\\mathrm{o}}^{\\top}$).

### 2. Lyapunov Propagation

Starting from $C_0 = \\mathbf{I}_d$, we iterate:

$$C_{\\ell+1} = \\frac{A_\\ell C_\\ell A_\\ell^{\\top}}{\\|A_\\ell C_\\ell A_\\ell^{\\top}\\|_2}, \\quad A_\\ell = \\mathbf{I} + W_{\\mathrm{mlp}}^{(\\ell)} + W_{\\mathrm{attn}}^{(\\ell)}$$

Normalization by the spectral norm preserves the condition number and effective rank while preventing numerical overflow.

### 3. Spectral Statistics

- **Effective Rank:** $\\mathrm{erank}(C) = \\exp\\left(-\\sum_i p_i \\log p_i\\right)$ where $p_i = \\lambda_i / \\sum_j \\lambda_j$.
- **Condition Number:** $\\kappa(C) = \\lambda_1 / \\lambda_d$.
- **Spectral Balance Ratio:** $\rho_\ell = \lVert W_{\mathrm{mlp}}^{(\ell)} \rVert_2 \/\ \lVert W_{\mathrm{attn}}^{(\ell)} \rVert_2$

---

## Citation

Citation information will be added upon publication.

## License

This project is licensed under the MIT License.
