# Meta Information

- URL: [[1810.10183] Multi-Head Attention with Disagreement Regularization](https://arxiv.org/abs/1810.10183)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, J., Tu, Z., Yang, B., Lyu, M. R., & Zhang, T. (2018). Multi-Head Attention with Disagreement Regularization. Proceedings of EMNLP 2018, pp. 2897–2903.

# Introduction

Multi-head attention (Vaswani et al., 2017) allows models to attend to information from multiple representation subspaces at different positions simultaneously. However, the original Transformer architecture contains **no mechanism to guarantee that different attention heads capture distinct features**. In practice, heads can collapse into learning nearly identical attention patterns, wasting model capacity.

This paper proposes **disagreement regularization**: auxiliary loss terms added during training to explicitly encourage diversity across attention heads. The approach is applicable to any practitioner training Transformer-based models where head redundancy is a concern—particularly in neural machine translation (NMT)—without adding any new parameters.

> [!NOTE]
> "There is no mechanism to guarantee that different attention heads indeed capture distinct features."

---

# Background: Multi-Head Attention

Standard multi-head attention projects queries $Q \in \mathbb{R}^{n \times d_{\text{model}}}$, keys $K \in \mathbb{R}^{m \times d_{\text{model}}}$, and values $V \in \mathbb{R}^{m \times d_{\text{model}}}$ into $H$ subspaces using learned projection matrices $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d_{\text{model}} \times d_k}$ for each head $i$:

$$\hat{Q}^i = Q W_Q^i, \quad \hat{K}^i = K W_K^i, \quad \hat{V}^i = V W_V^i$$

where $d_k = d_{\text{model}} / H$.

Each head computes scaled dot-product attention:

$$A^i = \text{softmax}\!\left(\frac{\hat{Q}^i (\hat{K}^i)^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times m}$$

$$O^i = A^i \hat{V}^i \in \mathbb{R}^{n \times d_k}$$

The outputs of all heads are concatenated and projected:

$$\text{MultiHead}(Q, K, V) = [O^1; O^2; \ldots; O^H] W_O$$

where $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

---

# Disagreement Regularization

## Training Objective

The standard NMT objective maximizes log-likelihood $\mathcal{L}(\theta)$. The augmented objective adds a disagreement term:

$$\mathcal{L}'(\theta) = \mathcal{L}(\theta) + \lambda \cdot D(\cdot)$$

where $\lambda = 1.0$ (set empirically) and $D(\cdot)$ is one of three regularization types defined below. The regularization is **added to encoder self-attention, decoder self-attention, and encoder-decoder attention** independently.

## Three Regularization Variants

### 1. Disagreement on Subspaces (Sub.)

Maximizes the dissimilarity between the projected value representations $\hat{V}^i$ across different heads using negative cosine similarity:

$$D_{\text{sub}} = -\frac{1}{H^2} \sum_{i=1}^{H} \sum_{j=1}^{H} \frac{\hat{V}^i \cdot \hat{V}^j}{\|\hat{V}^i\| \|\hat{V}^j\|}$$

- Input to regularizer: projected values $\hat{V}^i \in \mathbb{R}^{n \times d_k}$ for each head $i$
- Encourages heads to project the original values into distinct subspaces

### 2. Disagreement on Attended Positions (Pos.)

Penalizes overlap in the attention distributions $A^i$ across heads using element-wise multiplication:

$$D_{\text{pos}} = -\frac{1}{H^2} \sum_{i=1}^{H} \sum_{j=1}^{H} \left| A^i \odot A^j \right|$$

- Input to regularizer: attention weight matrices $A^i \in \mathbb{R}^{n \times m}$ for each head $i$
- Encourages heads to attend to different source positions

### 3. Disagreement on Outputs (Out.)

Maximizes dissimilarity between the output representations $O^i$ of each head using negative cosine similarity:

$$D_{\text{out}} = -\frac{1}{H^2} \sum_{i=1}^{H} \sum_{j=1}^{H} \frac{O^i \cdot O^j}{\|O^i\| \|O^j\|}$$

- Input to regularizer: head output vectors $O^i \in \mathbb{R}^{n \times d_k}$ for each head $i$
- Encourages heads to produce distinct output representations regardless of how they attend

## Pseudocode

```
# Forward pass with disagreement regularization
for each attention layer:
    for i in 1..H:
        V_hat[i] = V @ W_V[i]          # (n, d_k)
        K_hat[i] = K @ W_K[i]
        Q_hat[i] = Q @ W_Q[i]
        A[i] = softmax(Q_hat[i] @ K_hat[i].T / sqrt(d_k))  # (n, m)
        O[i] = A[i] @ V_hat[i]         # (n, d_k)

    # Compute disagreement losses (choose one or combine)
    D_sub = -mean over i≠j of cosine_sim(V_hat[i], V_hat[j])
    D_pos = -mean over i≠j of |A[i] ⊙ A[j]|
    D_out = -mean over i≠j of cosine_sim(O[i], O[j])

# Total loss
loss = cross_entropy_loss + λ * D_out  # Out. is best single term
```

---

# Comparison with Related Methods

| Method | Mechanism | Direction | Parameters Added |
|--------|-----------|-----------|-----------------|
| Standard multi-head attention | No head diversity constraint | N/A | 0 |
| Agreement regularization (prior work) | Encourages consensus across models | Agreement | 0 |
| **Disagreement regularization (this work)** | Penalizes redundancy among heads | Disagreement | 0 |
| Attention head pruning | Removes redundant heads post-training | N/A | 0 (removes) |

> [!IMPORTANT]
> This work inverts the direction of prior "agreement" regularization techniques (used in multi-model ensembles) and applies it within a single model's attention heads to enforce diversity rather than consensus.

---

# Experiments

- **Datasets:**
  - WMT17 Chinese→English (Zh→En): 20M sentence pairs
  - WMT14 English→German (En→De): 4M sentence pairs
- **Baseline model:** Transformer-Base and Transformer-Big (Vaswani et al., 2017) with 8 attention heads ($H = 8$)
- **Evaluation metric:** BLEU score with sign-test statistical significance testing
- **Regularization hyperparameter:** $\lambda = 1.0$

## Key Results

**Effect of individual regularization types (Zh→En, Transformer-Base):**

| Regularization | BLEU | Δ |
|----------------|------|---|
| Baseline | 24.13 | — |
| Sub. only | 24.59 | +0.46 |
| Pos. only | 24.42 | +0.29 |
| Out. only | 24.78 | +0.65 |
| All three | 24.57 | +0.44 |

Output-level regularization is the strongest single term. Combining all three does **not** yield additive improvements, suggesting overlapping guidance between terms.

**Applying to different attention types (Zh→En, Transformer-Base):**

| Applied to | BLEU |
|------------|------|
| Encoder self-attention only | 24.61 |
| Decoder self-attention only | 24.56 |
| Encoder-decoder attention only | 24.50 |
| All three attention types | **24.85** |

**Main results (BLEU):**

| Model | Zh→En | En→De |
|-------|-------|-------|
| Transformer-Base | 24.13 | 27.64 |
| Transformer-Base + Disagreement | **24.85** | **28.51** |
| Transformer-Big | 24.56 | 28.58 |
| Transformer-Big + Disagreement | **25.08** | **29.28** |

> [!NOTE]
> "Transformer-Base with disagreement regularization achieves comparable performance with Transformer-Big, while the training speed is nearly twice faster."

## Quantitative Analysis of Diversity

The paper measures actual disagreement using a normalized metric $\exp(D)$:

| Metric | Baseline | + Sub. | + Pos. | + Out. |
|--------|----------|--------|--------|--------|
| Subspace disagreement | 0.917 | **0.935** | 0.927 | 0.935 |
| Position disagreement | 0.007 | 0.012 | **0.219** | 0.012 |
| Output disagreement | 0.921 | 0.950 | 0.927 | **0.997** |

The extremely low baseline position disagreement (0.007) reveals that **attention heads naturally tend to attend to the same positions**, meaning positional diversity is not the primary strength of multi-head attention. Output-level diversity (0.921 baseline) is already higher but can be pushed to near-maximal (0.997) with the output regularization.

---

# Conclusion

Disagreement regularization is a lightweight, parameter-free technique that improves multi-head attention by explicitly penalizing redundancy among heads. The output-level variant ($D_{\text{out}}$) applied across all three attention types (encoder self, decoder self, encoder-decoder) yields consistent improvements of +0.5–0.9 BLEU on Chinese→English and English→German translation. The approach reduces the performance gap between Transformer-Base and Transformer-Big models while maintaining nearly 2× training speed advantage.
