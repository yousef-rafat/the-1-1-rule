# The 1:1 Rule: Lyapunov Analysis of Spectral Balance in Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-DOI-orange)]()

- **The 1:1 Rule:** For geometrically stable residual streams, the spectral norms of the MLP and attention pathways appear to require matching to within a factor of two.
- When the spectral balance ratio `ρ = ||W_mlp||₂ / ||W_attn||₂` drifts outside `[0.5, 2]`, the residual stream tends to collapse to effective rank `≈ 1`.
---

## Overview

This repository contains the official implementation for analyzing the spectral geometry of transformer residual streams via a Lyapunov covariance propagation framework. We analyze pretrained decoder-only language models and demonstrate that the **spectral balance ratio** $\rho$ between the MLP effective matrix and the attention output projection is a strong predictor of geometric stability.

### Key Features

- **Zero-data analysis:** Inspect weight matrices directly without forward passes on data.
- **Lyapunov iterator:** Propagate covariance through the residual stream layer-by-layer.
- **Multi-architecture support:** Tested on Gemma, Qwen, Phi, and SmolLM families (MHA, GQA, GELU, SwiGLU, RMSNorm, LayerNorm).
- **Reproducible figures:** Scripts to regenerate all paper figures and tables.

---

## Installation

```bash
git clone https://github.com/yousef-rafat/the-1-1-rule.git
cd the-1-1-rule
pip install -e .
```
---

## Methodology

### 1. Weight Extraction

For each layer $\ell$:

- **MLP:** Compute the effective matrix $W_{\\mathrm{mlp}}^{(\\ell)} = W_{\\mathrm{down}}^{(\\ell)} W_{\\mathrm{up}}^{(\\ell)}$ (incorporates SwiGLU gating where applicable).
- **Attention:** Use the output projection $W_{\\mathrm{o}}^{(\\ell)}$ (for GQA, we compute the covariance contribution $W_{\\mathrm{o}} W_{\\mathrm{o}}^{\\top}$).

### 2. Lyapunov Propagation

Starting from $C_0 = \\mathbf{I}_d$, we iterate:

$$C_{\\ell+1} = \\frac{A_\\ell C_\\ell A_\\ell^{\\top}}{\\|A_\\ell C_\\ell A_\\ell^{\\top}\\|_2}, \\quad A_\\ell = \\mathbf{I} + W_{\\mathrm{mlp}}^{(\\ell)} + W_{\\mathrm{attn}}^{(\\ell)}$$

Normalization by the spectral norm preserves the condition number and effective rank while preventing numerical overflow.

### 3. Spectral Statistics

- **Effective Rank:** $\mathrm{erank}(C) = \exp\left(-\sum_i p_i \log p_i\right)$ where $p_i = \lambda_i / \sum_j \lambda_j$.
- **Condition Number:** $\kappa(C) = \lambda_1 / \lambda_d$.
- **Spectral Balance Ratio:** $\rho_\ell = \lVert W_{\mathrm{mlp}}^{(\ell)} \rVert_2 \/\ \lVert W_{\mathrm{attn}}^{(\ell)} \rVert_2$
---

## Citation

If you use this code or find the 1:1 Rule useful, please cite:

```bibtex
@article{gamaleldin2025oneone,
  title={The 1:1 Rule: Lyapunov Analysis of Spectral Balance in Transformers},
  author={Gamaleldin, Yousef},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License.
