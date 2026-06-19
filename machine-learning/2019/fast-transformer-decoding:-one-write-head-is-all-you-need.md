# Meta Information

- URL: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. arXiv preprint arXiv:1911.02150.

# Overview

Multi-query attention is a structural modification to the standard Transformer's multi-head attention mechanism. Instead of having each attention head maintain its own separate key and value projections, all heads share a single set of keys and values while keeping independent query projections per head. This reduces the memory footprint of the key-value cache by a factor of $h$ (the number of heads) during autoregressive inference, yielding order-of-magnitude speedups in decoding with negligible quality loss.

**Who benefits**: Engineers deploying Transformer-based language models or sequence-to-sequence models in production, where inference latency is critical and decoder speed is bottlenecked by memory bandwidth rather than compute.

**When to apply**: During autoregressive (incremental) decoding, where key-value tensors for all previous timesteps must be loaded from memory at every decoding step.

# Background: Multi-Head Attention

## Notation

Let:
- $n$ = sequence length (number of tokens generated so far)
- $d$ = model dimension
- $h$ = number of attention heads
- $k$ = key/value dimension per head (often $k = d/h$)
- $b$ = batch size

## Standard Dot-Product Attention

Given a query matrix $Q \in \mathbb{R}^{m \times k}$, key matrix $K \in \mathbb{R}^{n \times k}$, and value matrix $V \in \mathbb{R}^{n \times v}$, attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{k}}\right) V$$

Output shape: $\mathbb{R}^{m \times v}$.

## Standard Multi-Head Attention (MHA)

MHA learns $h$ independent sets of projection matrices:
- Query projections: $W^Q_i \in \mathbb{R}^{d \times k}$ for $i = 1, \ldots, h$
- Key projections: $W^K_i \in \mathbb{R}^{d \times k}$ for $i = 1, \ldots, h$
- Value projections: $W^V_i \in \mathbb{R}^{d \times v}$ for $i = 1, \ldots, h$
- Output projection: $W^O \in \mathbb{R}^{hv \times d}$

For input $x \in \mathbb{R}^{n \times d}$ and query input $x' \in \mathbb{R}^{m \times d}$:

$$q_i = x' W^Q_i, \quad k_i = x W^K_i, \quad v_i = x W^V_i$$
$$\text{head}_i = \text{Attention}(q_i, k_i, v_i) \in \mathbb{R}^{m \times v}$$
$$\text{MultiHead}(x', x) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O \in \mathbb{R}^{m \times d}$$

Key-value tensors $k_i, v_i$ are of size $\mathbb{R}^{n \times k}$ each, giving a total KV-cache size of $2hk n$ scalars.

## Memory Bandwidth Bottleneck During Incremental Decoding

During autoregressive decoding (generating one token at a time, $m=1$), the query $x' \in \mathbb{R}^{1 \times d}$ is a single vector. Each decoding step must:
1. Compute one new key/value pair per head (cheap, proportional to $d$).
2. Load all previously cached keys/values: $\Theta(hnk)$ scalars from memory.
3. Compute attention over $n$ positions per head: $\Theta(hnk)$ FLOPs.

The memory-to-compute ratio is $\Theta(n/d + 1/b)$ (memory reads per FLOP). When $n$ is large (long sequences), loading the KV-cache dominates and the decoder is memory-bandwidth bound, not compute-bound.

# Multi-Query Attention (MQA)

## Core Idea

Multi-query attention keeps $h$ independent query projection matrices but reduces to a **single** shared key projection $W^K \in \mathbb{R}^{d \times k}$ and a **single** shared value projection $W^V \in \mathbb{R}^{d \times v}$.

$$q_i = x' W^Q_i \quad (i = 1, \ldots, h)$$
$$k = x W^K \in \mathbb{R}^{n \times k}$$
$$v = x W^V \in \mathbb{R}^{n \times v}$$
$$\text{head}_i = \text{Attention}(q_i, k, v)$$
$$\text{MultiQueryAttn}(x', x) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

> [!NOTE]
> "We propose multi-query attention, where the keys and values are shared across all of the different attention heads. The different queries can still be thought of as attending to different aspects of the representation, but they all inspect the same set of values at the same positions." — Shazeer (2019)

## Pseudocode for Multi-Query Attention (Single Decoding Step)

```
# Inputs at decoding step t:
#   q_input: current token embedding, shape (1, d)
#   KV_cache: accumulated keys and values, shapes (t, k) and (t, v)

function MultiQueryAttentionDecode(q_input, KV_cache_K, KV_cache_V):
    # Compute queries for all heads (independent projections)
    for i in 1..h:
        q_i = q_input @ W_Q[i]      # shape (1, k)

    # Single shared key/value for new token
    k_new = q_input @ W_K           # shape (1, k)
    v_new = q_input @ W_V           # shape (1, v)

    # Append to cache
    K = concat(KV_cache_K, k_new)   # shape (t+1, k)
    V = concat(KV_cache_V, v_new)   # shape (t+1, v)

    # Attend: each head uses the same K, V
    for i in 1..h:
        scores_i = (q_i @ K.T) / sqrt(k)   # shape (1, t+1)
        weights_i = softmax(scores_i)        # shape (1, t+1)
        head_i = weights_i @ V               # shape (1, v)

    # Concatenate and project
    output = concat(head_1, ..., head_h) @ W_O   # shape (1, d)
    return output, K, V
```

## Memory and Compute Analysis

| Method | KV-cache size | Memory per step | Memory-to-Compute Ratio |
|---|---|---|---|
| Multi-Head Attention | $2hkn$ | $\Theta(hn)$ | $\Theta(n/d + 1/b)$ |
| Multi-Query Attention | $2kn$ | $\Theta(n)$ | $\Theta(1/d + n/(dh) + 1/b)$ |

The $n/d$ term (dominant when $n$ is large) is reduced by a factor of $h$, making decoding memory-bandwidth efficient.

## Difference from Multi-Head Attention

| Aspect | Multi-Head Attention | Multi-Query Attention |
|---|---|---|
| Query projections | $h$ separate $W^Q_i$ | $h$ separate $W^Q_i$ |
| Key projections | $h$ separate $W^K_i$ | 1 shared $W^K$ |
| Value projections | $h$ separate $W^V_i$ | 1 shared $W^V$ |
| KV-cache size | $2hkn$ | $2kn$ (1/h of MHA) |
| Decoder speed | Baseline | ~3.8–12x faster |
| Training speed | Baseline | ~Unchanged |
| Quality | Baseline | Minor degradation |

> [!IMPORTANT]
> Encoder attention (cross-attention in seq2seq) benefits from MQA because the encoder KV-cache is computed once and reused across all decoding steps. Self-attention in the decoder benefits because the cache grows with each generated token. Both can be replaced with multi-query attention.

# Architecture Details

The paper applies MQA to both self-attention and encoder-decoder attention in a 6-layer Transformer:
- $h = 8$ heads, $d = 1024$, $k = v = 128$ (so standard head dim = $d/h = 128$)
- ~211M parameters (same as baseline due to minor parameter count reduction in KV projections being offset by other components)
- The output projection $W^O$ shape is unchanged at $\mathbb{R}^{hv \times d}$

# Experiments

## Datasets

| Task | Dataset | Description |
|---|---|---|
| Translation | WMT 2014 English-German | Standard MT benchmark; training/dev/test splits |
| Language Modeling | One Billion Word (Chelba et al., 2013) | Large-scale LM benchmark; word-level perplexity |

## Training Setup

- Hardware: 32-core TPUv3
- Steps: 100,000 training steps
- Batch size and optimizer: Standard Transformer hyperparameters (Adam)
- Baseline: 6-layer Transformer with 8-head MHA

## Translation Quality (WMT EN-DE)

| Model | Dev BLEU | Test BLEU (greedy) | Test BLEU (beam-4) |
|---|---|---|---|
| Multi-Head Attention | 26.7 | 27.4 | 28.4 |
| Multi-Query Attention | 26.5 | 27.0 | 28.5 |

Quality loss is minimal (−0.2 to +0.1 BLEU points).

## Language Modeling (One Billion Word)

| Model | Perplexity |
|---|---|
| Multi-Head Attention | 29.9 |
| Multi-Query Attention | 30.2 |

Perplexity increase of 0.3 points — negligible for practical purposes.

## Inference Speed (WMT EN-DE Translation)

| Decode Mode | Multi-Head (μs/token) | Multi-Query (μs/token) | Speedup |
|---|---|---|---|
| Encoder | 136 | 118 | 1.15x |
| Decoder (greedy) | 46 | 3.8 | **12x** |
| Decoder (beam-4) | 203 | 32 | **6.3x** |
| Training step | 13.2 | 13.0 | ~1x |

> [!NOTE]
> Training speed is essentially unchanged because during training, all tokens are processed in parallel ($m = n$), making the computation compute-bound rather than memory-bandwidth-bound.

# Comparison with Related Methods

| Method | Approach | Trade-off |
|---|---|---|
| Standard MHA | Full separate KV per head | Highest quality, slowest decode |
| Multi-Query Attention (this paper) | Shared KV across heads | Minimal quality loss, 6–12x decode speedup |
| Sparse Attention (Liu et al., 2018) | Attend to subset of positions | Reduces compute but not memory bandwidth for cached positions |
| Linear Attention (Performer, etc.) | Approximate kernelized attention | Avoids quadratic memory but changes computation fundamentally |
| Grouped-Query Attention (GQA, 2023) | Groups of heads share KV | Interpolates between MHA and MQA |

> [!TIP]
> Grouped-Query Attention (Ainslie et al., 2023, arXiv:2305.13245) extends this idea by having $g$ groups of heads share keys and values, providing a spectrum between full MHA ($g=h$) and full MQA ($g=1$).

> [!CAUTION]
> The paper was written before the large-scale adoption of MQA in models like PaLM, Falcon, and Mistral. Later empirical work (GQA paper) shows that models trained from scratch with MQA perform better than models with MHA converted to MQA post-hoc.
