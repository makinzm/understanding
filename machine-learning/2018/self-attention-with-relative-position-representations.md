# Meta Information

- URL: [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-Attention with Relative Position Representations. NAACL-HLT 2018.

# Self-Attention with Relative Position Representations

## Overview

This paper extends the Transformer's self-attention mechanism to incorporate **relative position representations** instead of (or in addition to) absolute sinusoidal position encodings. By treating pairs of input elements as edges in a labeled graph and learning relative position embeddings, the approach improves machine translation quality without significant computational overhead.

The method is applicable whenever sequence order matters but absolute positions are less informative than pairwise distances—for example, in translation, where structural patterns (subject-verb-object) recur at various absolute positions but always involve adjacent or nearby tokens.

## Background

### Standard Self-Attention

Given a sequence of input vectors $x = (x_1, \ldots, x_n)$ where $x_i \in \mathbb{R}^{d_x}$, standard self-attention computes output $z_i \in \mathbb{R}^{d_z}$ as:

$$e_{ij} = \frac{(x_i W^Q)(x_j W^K)^\top}{\sqrt{d_z}}$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})}$$

$$z_i = \sum_{j=1}^{n} \alpha_{ij}(x_j W^V)$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d_x \times d_z}$ are learned projection matrices.

**Limitation**: This formulation has no inherent notion of position. The original Transformer addresses this by adding sinusoidal **absolute position encodings** to the input embeddings before self-attention. However, absolute positions do not directly capture the *relative* distance between tokens, which is often more linguistically meaningful.

### Difference from RNNs and CNNs

- **RNNs**: Position is implicitly encoded by sequential processing; each hidden state integrates information from all previous tokens in order.
- **CNNs**: Local position is captured by the kernel's receptive field; long-range dependencies require stacking many layers.
- **Transformers (original)**: All pairs of tokens attend directly in $O(1)$ layers, but positional information must be injected explicitly.

## Proposed Method: Relation-aware Self-Attention

### Graph-based Interpretation

The authors model the input sequence as a fully connected directed graph, where each pair $(x_i, x_j)$ is connected by a labeled edge. Two types of edge labels are learned:

- $a^K_{ij} \in \mathbb{R}^{d_z}$: edge label used in computing attention *scores*
- $a^V_{ij} \in \mathbb{R}^{d_z}$: edge label used in computing attention *outputs*

### Modified Attention Equations

**Attention score** (incorporating relative position into key):

$$e_{ij} = \frac{x_i W^Q (x_j W^K + a^K_{ij})^\top}{\sqrt{d_z}}$$

**Output** (incorporating relative position into value):

$$z_i = \sum_{j=1}^{n} \alpha_{ij}(x_j W^V + a^V_{ij})$$

Compared to the standard formulas, both the key term and value term receive an additive position-dependent correction.

### Clipped Relative Position Encoding

The edge labels are determined by the clipped relative distance between positions $i$ and $j$:

$$a^K_{ij} = w^K_{\text{clip}(j-i,\, k)}, \quad a^V_{ij} = w^V_{\text{clip}(j-i,\, k)}$$

where the clipping function is:

$$\text{clip}(x, k) = \max(-k,\, \min(k,\, x))$$

This results in $2k + 1$ distinct learnable vectors for each of $w^K$ and $w^V$, covering relative distances $\{-k, \ldots, 0, \ldots, k\}$.

**Design rationale**: Clipping at distance $k$ reflects the assumption that precise relative position beyond $k$ steps is less important. It also allows the model to generalize to sequences longer than those seen during training, since no new position indices are introduced.

> [!NOTE]
> The paper finds $k = 16$ optimal for the base model and $k = 8$ for the big model on WMT tasks.

### Efficient Implementation

The naive computation of all $a^K_{ij}$ terms would require $O(n^2 d_z)$ space and $O(n^2 d_z h)$ parameters (for $h$ heads). The authors reduce this by:

1. **Sharing representations across heads**: All attention heads use the same $w^K$ and $w^V$ vectors, reducing parameter count from $O(h \cdot (2k+1) \cdot d_z)$ to $O((2k+1) \cdot d_z)$.
2. **Decomposing the score computation**: The modified score $e_{ij} = \frac{x_i W^Q (x_j W^K)^\top + x_i W^Q (a^K_{ij})^\top}{\sqrt{d_z}}$ is computed as two separate batched matrix multiplications and summed, avoiding materializing the full $n \times n \times d_z$ tensor.

**Training speed impact**: Only a 7% slowdown on P100 GPUs compared to the baseline Transformer.

### Algorithm (Pseudocode)

```
Input: sequence x ∈ R^{n × d_x}, max relative distance k
Parameters: W^Q, W^K, W^V ∈ R^{d_x × d_z}
            w^K ∈ R^{(2k+1) × d_z}, w^V ∈ R^{(2k+1) × d_z}

For each position i in 1..n:
    For each position j in 1..n:
        rel = clip(j - i, k)          # integer in [-k, k]
        a^K_ij = w^K[rel + k]         # look up relative key embedding
        a^V_ij = w^V[rel + k]         # look up relative value embedding

# Attention scores (two terms summed)
Q = x W^Q                             # (n, d_z)
K = x W^K                             # (n, d_z)
scores_abs = Q K^T / sqrt(d_z)        # (n, n)
scores_rel = einsum("id,ijd->ij", Q, A^K) / sqrt(d_z)  # (n, n)
e = scores_abs + scores_rel

# Softmax and output
α = softmax(e, dim=-1)                # (n, n)
V = x W^V                             # (n, d_z)
z = α (V + A^V)                       # (n, d_z), A^V stacked along axis 1
```

## Ablation Studies

| Configuration | EN-DE BLEU (Base) |
|---|---|
| Absolute position only (baseline) | 26.5 |
| Relative ($a^K$ only) | 25.8 |
| Relative ($a^V$ only) | 25.3 |
| Relative ($a^K$ and $a^V$) | 26.8 |
| No position | 12.5 |

> [!IMPORTANT]
> Relative position in attention *scoring* ($a^K$) contributes more than relative position in *values* ($a^V$). Using $a^K$ alone still outperforms no position encoding; $a^V$ alone performs worst among configured variants (but still far above no-position baseline). The full combination of both is best.

**Effect of clipping distance $k$** (base model, EN-DE):

| $k$ | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| BLEU | 25.5 | 25.9 | 25.9 | 25.9 | 25.8 | 25.9 | 25.9 | 25.9 |

Performance is robust for $k \geq 2$; very short relative distances ($k=0$) hurt moderately.

## Experiments

- **Datasets**:
  - WMT 2014 English-German: ~4.5M sentence pairs; newstest2014 as test set
  - WMT 2014 English-French: ~36M sentence pairs; newstest2014 as test set
- **Optimizer**: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$) with warmup over 4,000 steps
- **Regularization**: Dropout (residual, attention, embeddings), label smoothing $\epsilon_{ls} = 0.1$
- **Inference**: Beam search with beam size 4, length penalty $\alpha = 0.6$
- **Hardware**:
  - Base model: 8× K40 GPUs, 100,000 steps
  - Big model: 8× P100 GPUs, 300,000 steps
- **Results**:

| Model | EN-DE BLEU | EN-FR BLEU |
|---|---|---|
| Transformer Base (absolute) | 26.5 | 38.2 |
| Transformer Base (relative) | **26.8** | **38.7** |
| Transformer Big (absolute) | 27.9 | 41.2 |
| Transformer Big (relative) | **29.2** | **41.5** |

Relative position representations improve BLEU by +1.3 on EN-DE and +0.3 on EN-FR for the big model.

## Comparison with Related Methods

| Method | Position Type | Mechanism |
|---|---|---|
| Original Transformer (Vaswani et al., 2017) | Absolute sinusoidal | Added to input embeddings before attention |
| This work | Relative (clipped) | Added to key and value inside each attention head |
| RNN (e.g., LSTM) | Implicit sequential | Hidden state integrates order through recurrence |
| CNN | Local relative | Kernel covers fixed-size window; stacking for long range |
| RPE (prior, e.g., Dai et al.) | Relative | Similar idea explored in memory-augmented networks |

**Key difference from absolute encoding**: Absolute encodings tell each token "I am at position 5"; relative encodings tell pairs of tokens "you are 3 steps ahead of me." The latter is invariant to the absolute starting position of a phrase, which is useful for capturing recurring syntactic patterns.

> [!TIP]
> Subsequent work (e.g., Transformer-XL, T5) adopted and extended relative position encoding, often replacing the additive edge embedding with a bias term in the attention logit. This paper's formulation is the direct ancestor of those approaches.
