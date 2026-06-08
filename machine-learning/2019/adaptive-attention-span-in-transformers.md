# Meta Information

- URL: [Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Sukhbaatar, S., Grave, E., Bojanowski, P., & Joulin, A. (2019). Adaptive Attention Span in Transformers. ACL 2019.

# Adaptive Attention Span in Transformers

Standard Transformers fix the same attention span (context window size) for all attention heads across all layers. This approach is computationally wasteful because empirical analysis shows that different heads develop fundamentally different needs: some heads only attend to immediately preceding tokens, while others track long-range dependencies across hundreds of tokens. This paper introduces a learnable, per-head masking mechanism that allows each attention head to independently determine its own optimal attention span, reducing memory and compute costs while enabling longer effective context lengths.

**Who benefits:** Practitioners training character-level language models or any sequence model where different positions/layers have heterogeneous dependency ranges. Particularly useful when scaling to long sequences (>1000 tokens) where the standard quadratic cost of full attention is prohibitive.

## Background: Standard Self-Attention

Given a sequence of $T$ tokens, a single attention head computes:

$$s_{tr} = \frac{x_t \cdot x_r}{\sqrt{d}}, \quad a_{tr} = \frac{\exp(s_{tr})}{\sum_{q=1}^{T} \exp(s_{tq})}, \quad y_t = \sum_{r=1}^{T} a_{tr} x_r$$

where:
- $x_t \in \mathbb{R}^{d}$ is the hidden state at position $t$
- $s_{tr} \in \mathbb{R}$ is the raw attention score between positions $t$ and $r$
- $a_{tr} \in \mathbb{R}$ is the normalized attention weight
- $y_t \in \mathbb{R}^{d}$ is the attended output at position $t$
- $d$ is the head dimension

The maximum attention span $S$ (the number of past tokens each position can attend to) is fixed identically for all heads.

## Adaptive Attention Span

### Soft Masking Function

Each attention head $i$ is assigned a learnable scalar parameter $z_i \in [0, S]$ representing its current attention span. A differentiable soft masking function $m_{z_i} : \mathbb{R} \to [0, 1]$ gates how much weight is placed on tokens at distance $x$ from the current position:

$$m_z(x) = \min\!\left(\max\!\left(\frac{1}{R}(R + z - x),\ 0\right),\ 1\right)$$

where:
- $x \in \mathbb{R}^+$ is the distance from the current token (i.e., $x = t - r$)
- $z \in [0, S]$ is the learnable span parameter for this head (initialized to 0)
- $R$ is a fixed hyperparameter (set to 32 in experiments) that controls the softness of the transition from 0 to 1
- $m_z(x) = 1$ for $x \leq z$ (tokens within the span are fully attended)
- $m_z(x) = 0$ for $x \geq z + R$ (tokens beyond the span are masked out)
- The region $[z, z+R]$ provides a smooth gradient-friendly ramp

> [!NOTE]
> The value of $z$ is constrained to $[0, S]$ using a clamp/sigmoid transformation during optimization, allowing gradient-based learning while respecting boundary constraints.

### Modified Attention Weights

The masking function is applied before softmax normalization so that masked positions have zero weight:

$$a_{tr} = \frac{m_z(t - r) \exp(s_{tr})}{\sum_{q} m_z(t - q) \exp(s_{tq})}, \quad y_t = \sum_{r} a_{tr} x_r$$

Tokens at distance $> z + R$ from position $t$ receive zero weight from head $i$, effectively reducing the span of computation for that head.

### Span Regularization

To encourage heads to learn minimal sufficient spans, an L1 penalty on all $M$ span parameters across all layers is added to the language modeling loss:

$$\mathcal{L} = -\log P(w_1, \ldots, w_T) + \frac{\lambda}{M} \sum_{i=1}^{M} z_i$$

where $\lambda$ controls the trade-off between perplexity and span efficiency. In experiments, $\lambda = 2 \times 10^{-6}$ is used.

### Dynamic Attention Span

A more powerful extension makes the span depend on the current input rather than being a fixed per-head scalar. Each head computes:

$$z_t = S \cdot \sigma(v^\top x_t + b)$$

where:
- $x_t \in \mathbb{R}^d$ is the current token's hidden state
- $v \in \mathbb{R}^d$ and $b \in \mathbb{R}$ are learnable parameters
- $\sigma$ is the sigmoid function
- $z_t \in (0, S)$ is the token-specific span for this head at position $t$

This allows the span to expand at semantic boundaries (e.g., start of a new word in character-level modeling) and contract within repetitive local patterns.

## Algorithm: Forward Pass with Adaptive Span

```
Input: tokens x_1, ..., x_T, each x_t ∈ R^d
       learnable span z_i ∈ [0, S] for each head i
       hyperparameters R (ramp width), S (max span)

For each attention head i:
  1. Compute raw scores: s_tr = (x_t · x_r) / sqrt(d)  for all t, r
  2. Compute mask:       m_z(t - r) for each (t, r) pair
        where m_z(x) = clamp((R + z - x) / R, 0, 1)
  3. Zero masked logits: s_tr ← s_tr where m_z(t-r) > 0, else -∞
  4. Normalize:          a_tr = softmax(s_tr) * m_z(t - r)
  5. Attend:             y_t = sum_r a_tr * x_r

Loss:
  L = NLL(predictions) + (λ / M) * sum_i z_i
```

> [!IMPORTANT]
> At inference time, the actual computation can be truncated to only attend to $\lceil z_i + R \rceil$ past tokens for head $i$, achieving real FLOPS savings. Without this truncation, the masking reduces memory access patterns but not raw multiply-add count.

## Comparison with Related Methods

| Method | Attention Span | Span Type | Overhead |
|---|---|---|---|
| Vanilla Transformer | Fixed $S$ for all heads | Global constant | None |
| Transformer-XL | Fixed segment + recurrence cache | Global constant | Recurrence cache |
| **Adaptive Span (this work)** | Learnable per head | Head-specific scalar | $M$ extra params ($z_i$) |
| **Dynamic Span (this work)** | Input-dependent per head | Per-token per-head | $M$ extra linear projections |
| Sparse Transformers | Fixed sparse patterns (strided/local) | Global pattern | None |

**Key differences from Transformer-XL:** Transformer-XL extends context via segment recurrence and relative position encodings but still uses the same span for all heads. Adaptive Span learns that most heads need short spans, achieving comparable perplexity with far fewer FLOPS.

**Key differences from Sparse Attention:** Sparse Transformers (Child et al., 2019) use hand-designed fixed patterns, while adaptive span learns patterns end-to-end. Adaptive span can discover that early layers need only local context while late layers need global context.

# Experiments

- **Datasets:**
  - `text8`: 100M characters of Wikipedia text (90M train / 5M val / 5M test), evaluated in bits-per-character (bpc)
  - `enwik8`: 100M characters of raw Wikipedia XML (90M train / 5M val / 5M test), evaluated in bits-per-character (bpc)
  - Character-level prediction task; no tokenization applied
- **Hardware:** 8 × NVIDIA V100 GPUs
- **Optimizer:** Adagrad with learning rate 0.07, gradient clipping at 0.03
- **Training steps:** 600K–900K steps
- **Batch size:** 64 sequences of 512 tokens each
- **Model architecture:**
  - Small model: 12 layers, 8 heads, $d=512$, FFN width 2048, 38M parameters
  - Large model: 24 layers, 8 heads, $d=512$, FFN width 4096, 209M parameters
  - Max span $S \in \{512, 1024, 2048, 4096, 8192\}$
- **Results (bpc on test set):**
  - Small adaptive-span model: **1.11 bpc** on text8 (vs. 1.18 for baseline T12 with fixed span)
  - Large adaptive-span model: **1.07 bpc** on text8 (vs. 1.08 for Transformer-XL with 277M params)
  - Large adaptive-span model on enwik8: **1.02 bpc** (state-of-the-art at time of publication)
  - FLOPS: Large adaptive model uses **~70% fewer FLOPS** than Transformer-XL at comparable perplexity
- **Span analysis:** With $S=4096$, the average learned span across all heads is ~314 tokens; lower layers learn spans of ~100 tokens while higher layers learn spans up to ~4000 tokens. Dynamic span extends average spans to ~8000 characters with $S=8192$.
