
# The 1:1 Rule: Lyapunov Analysis of Spectral Balance in Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

-  Main observation: In the models we tested, the residual stream tends to remain stable when the spectral norms of the MLP and attention pathways are of similar magnitude.
-  In our experiments, the residual stream often becomes increasingly concentrated in a single direction when this ratio moves outside roughly 0.5–2.

---

## Overview

This repository contains the code for our experiments on the spectral geometry of transformer residual streams. We analyze pretrained decoder-only language models and find that the **spectral balance ratio** $\\rho$ between the MLP effective matrix and the attention output projection is a strong predictor of geometric stability.  

<br>
 
<img width="4010" height="2800" alt="fig4_new" src="https://github.com/user-attachments/assets/328b43f1-a921-4540-849b-0f512848baef" />

### Key Features

- **Weight-only analysis:** The main analysis works directly from the model weights and does not require a dataset.
- **Covariance propagation:** We propagate the covariance through the residual stream one layer at a time.
- **Synthetic experiments:** Controlled random-matrix experiments let us test the proposed mechanism separately from the pretrained models.
- **Multi-architecture:** Tested on Gemma, Qwen, Llama, Phi, Pythia, and SmolLM families.
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

## Synthetic validation

We also test the effect with synthetic matrices. We construct spiked MLP and attention matrices with correlated singular-vector bases, while making their dominant directions anti-aligned.

We sweep the spectral ratio

$\rho_\ell = \lVert W_{\mathrm{mlp}} \rVert_2 \/\ \lVert W_{\mathrm{attn}} \rVert_2$

and apply the same Lyapunov propagation for 24 layers.

Under anti-alignment, the system reproduces the observed bifurcation: ratios within $0.5 < \rho < 2$ maintain high effective rank, while strongly imbalanced ratios produce rank-1 collapse.

As a control, we repeat the experiment with independent singular-vector bases instead of the anti-aligned construction. In this case, the sharp transition largely disappears. This suggests that the relative orientation of the two pathways matters in addition to their spectral norms.

<br>

<img width="1389" height="540" alt="sync_results" src="https://github.com/user-attachments/assets/6399ff7c-8fd5-4b71-b5cc-f0128ee87f9f" />

## Citation

Citation information will be added upon publication.

## License

This project is licensed under the MIT License.
