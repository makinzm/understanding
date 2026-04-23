# Meta Information

- URL: [Understanding the Difficulty of Training Transformers](https://arxiv.org/abs/2004.08249)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Liu, L., Liu, X., Gao, J., Chen, W., & Han, J. (2020). Understanding the Difficulty of Training Transformers. *Proceedings of EMNLP 2020*.

# Understanding the Difficulty of Training Transformers

## Overview

This paper investigates why Transformer training requires specialized optimizers (e.g., Adam) and warmup learning rate schedules, which plain SGD cannot replicate. The central finding **challenges the conventional wisdom** that unbalanced gradients (gradient vanishing in Post-LN Transformers) are the root cause of training instability. Instead, the authors identify an **amplification effect** caused by heavy dependency on residual branches, and propose **Admin (Adaptive Model Initialization)** to control this dependency.

**Applicability:** Engineers and researchers training deep Transformer-based models (translation, language modeling, etc.) who encounter training instability or divergence with Post-LN architectures.

## Background: Post-LN vs Pre-LN

Two common Transformer residual configurations:

| Configuration | Formula | Behavior |
|---|---|---|
| Post-LN | $x_i = \text{LayerNorm}(x_{i-1} + f_i(x_{i-1}))$ | More expressive, often diverges with deep models |
| Pre-LN | $x_i = x_{i-1} + f_i(\text{LayerNorm}(x_{i-1}))$ | Stabler training, slightly lower peak performance |

Here $f_i$ is the sub-layer function (self-attention or FFN), and $x_i \in \mathbb{R}^{d}$ is the layer output. LayerNorm normalizes its input to zero mean and unit variance.

## Section 3: Gradient Analysis (Why It Is NOT the Root Cause)

### Theorem 1: Gradient Behavior

The authors prove that Post-LN **encoders** do not suffer from gradient vanishing at initialization — only Post-LN **decoders** do (due to the cross-attention mechanism). The gradient of the loss $\mathcal{L}$ with respect to parameters $\theta_i$ in layer $i$ satisfies:

$$\frac{\partial \mathcal{L}}{\partial \theta_i} = \frac{\partial \mathcal{L}}{\partial x_N} \cdot \prod_{k=i}^{N} \frac{\partial x_k}{\partial x_{k-1}} \cdot \frac{\partial x_i}{\partial \theta_i}$$

- **Pre-LN encoders:** Gradient norms remain $O(1)$ across all layers due to identity shortcut.
- **Post-LN encoders:** Gradient norms are also $O(1)$ at initialization because LayerNorm re-normalizes outputs.
- **Post-LN decoders:** Gradient norms decay toward 0 for earlier layers.

### Why Fixing Gradient Vanishing Alone Is Insufficient

The authors construct a hybrid architecture (Post-LN encoder + Pre-LN decoder) that eliminates gradient vanishing for the encoder. Despite this fix, training still diverges, demonstrating that gradient imbalance is a symptom, not the cause.

> [!NOTE]
> "Fixing gradient vanishing alone cannot stabilize model training successfully." — Section 3.4

### Adaptive Optimizers vs SGD

Adam handles non-uniform gradient distributions via per-parameter adaptive learning rates. This is why Adam can train Transformers where SGD cannot — not because gradients are balanced, but because Adam implicitly normalizes gradient magnitudes.

## Section 4: The Amplification Effect (Root Cause)

### Layer Dependency Metric

Define the **residual dependency** of layer $i$ on branch $j$ as the coefficient $\beta_{i,j}$ such that:

$$\hat{x}_i = \sum_{j \leq i} \beta_{i,j} \cdot \hat{a}_j$$

where $\hat{x}_i$ is the normalized output of layer $i$, and $\hat{a}_j$ is the normalized output of residual branch $j$. Each $\beta_{i,j}$ measures the proportion of branch $j$'s contribution to layer $i$'s output.

### Theorem 2: Output Change Under Perturbation

When model parameters are perturbed by $\epsilon$ during a gradient update, the change in layer $N$'s output satisfies:

$$\Delta_N \approx \sum_{i=1}^{N} \beta_{N,i}^2 \cdot C_i$$

where $C_i$ is a constant related to the derivative of $f_i$. This yields two corollaries:

- **Corollary 1 (Pre-LN):** $\beta_{N,i}^2 \sim O(1/N)$, so $\Delta_N = O(\log N)$
- **Corollary 2 (Post-LN):** $\beta_{N,i}^2 \sim O(1)$ for large $i$, so $\Delta_N = O(N)$

**Consequence:** Post-LN's output disturbance scales **linearly** with depth $N$, while Pre-LN's scales only **logarithmically**. A 72-layer Post-LN model experiences 12× more output amplification than a Pre-LN model.

> [!IMPORTANT]
> The amplification effect explains why warmup helps but does not fully stabilize Post-LN: warmup reduces the effective learning rate early in training (shrinking $\epsilon$), but does not change the $O(N)$ vs $O(\log N)$ scaling of $\beta_{N,i}^2$.

### Why Post-LN Has Larger β

At initialization in Post-LN, LayerNorm is applied **after** addition, so the residual branch output dominates (has larger variance than the skip connection). In Pre-LN, LayerNorm is applied **before**, partially suppressing the residual branch contribution and distributing dependencies more evenly across layers.

## Section 4.3: Admin — Adaptive Model Initialization

Admin introduces learnable scalar parameters $\omega_i \geq 0$ for each residual branch, modifying the Post-LN update rule to:

$$x_i = \text{LayerNorm}(x_{i-1} + \omega_i \cdot f_i(x_{i-1}))$$

### Algorithm: Admin Initialization

**Input:** Network with $N$ residual branches $f_1, \ldots, f_N$; first training batch $\mathcal{B}$

**Phase 1 — Profiling:**
```
Initialize all ω_i = 1 (standard Post-LN behavior)
Forward pass with batch B through the network
For each layer i, record: V_i = Var[f_i(x_{i-1})]
```

**Phase 2 — Re-initialization:**
```
For each layer i from 1 to N:
    ω_i = sqrt(V_1 + V_2 + ... + V_{i-1})
    # This sets ω_i² ≈ Σ_{j<i} Var[f_j], restricting self-dependency β_{i,i} ≈ 1/i
Reset all other parameters to standard initialization
```

**Training:** Proceed with standard Adam optimizer and no warmup required.

**Post-training Reparameterization:**
```
# Absorb ω_i into the adjacent weight matrices and LayerNorm parameters
# No change to inference computation; ω_i parameters are eliminated
```

> [!NOTE]
> "Profiling uses less than 8192 tokens, taking only a few seconds even for large models." — Appendix

### Why Admin Works

By setting $\omega_i^2 \approx \sum_{j < i} \text{Var}[f_j(x_{j-1})]$, Admin ensures that at initialization:

$$\beta_{N,i} \approx \frac{1}{\sqrt{i}}$$

This gives $\sum_i \beta_{N,i}^2 \approx \sum_i \frac{1}{i} = O(\log N)$ — matching Pre-LN's stability while preserving the Post-LN structure's expressive capacity after training.

## Comparison with Similar Approaches

| Method | Normalization | Extra Params | Post-Train Overhead | Max Stable Depth |
|---|---|---|---|---|
| Post-LN (baseline) | Post | None | None | ~12L |
| Pre-LN | Pre | None | None | 60L+ |
| **Admin** | Post | $\omega_i$ per layer | None (reparameterized) | 72L+ |
| ReZero | None | scalar per layer | Persists | ~20L |
| FixUp | None | scalars | Persists | ~30L |

> [!TIP]
> Pre-LN was popularized in GPT-2 and many modern LLMs. Admin enables training Post-LN models as deep as Pre-LN while recovering Post-LN's stronger final performance.

## Experiments

- **Dataset:** IWSLT'14 De-En (small-scale); WMT'14 En-De (medium); WMT'14 En-Fr (large-scale)
- **Hardware:** Not specified in detail
- **Optimizer:** Adam (β₁=0.9, β₂=0.98, ε=10⁻⁸); no warmup needed for Admin
- **Metric:** BLEU score (tokenized)

### Key Results

| Configuration | Post-LN | Pre-LN | Admin |
|---|---|---|---|
| IWSLT De-En (6L enc–6L dec) | 35.64 | 35.50 | **35.67** |
| WMT En-Fr (6L–6L) | 41.29 | 40.74 | **41.47** |
| WMT En-De (18L–18L) | **diverged** | 28.38 | **29.03** |
| WMT En-Fr (60L enc–12L dec) | **diverged** | 43.10 | **43.80** |

- Admin achieves **43.80 BLEU** on WMT'14 En-Fr (state-of-the-art without back-translation at the time).
- Admin trains a **72-layer Transformer** successfully where Post-LN diverges and Pre-LN is the maximum viable baseline.
- Admin shows less sensitivity to learning rate choices in grid search experiments.

## Summary of Findings

1. **Gradient vanishing in Post-LN is not the primary instability cause** — only decoders suffer, and fixing it alone does not help.
2. **Amplification effect is the root cause**: Post-LN output disturbance scales $O(N)$ vs Pre-LN's $O(\log N)$ under gradient updates.
3. **Admin resolves the amplification effect** with a two-phase initialization using per-layer scalar $\omega_i$, with zero inference overhead after reparameterization.
4. **Warmup schedules reduce but do not eliminate the amplification effect**, explaining their partial effectiveness.
