# Meta Information

- URL: [Parallel Context-of-Experts Decoding for Retrieval Augmented Generation](https://arxiv.org/abs/2601.08670)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Corallo, G., & Papotti, P. (2026). Parallel Context-of-Experts Decoding for Retrieval Augmented Generation. arXiv:2601.08670.

# Overview

## Problem Statement

Retrieval Augmented Generation (RAG) faces a fundamental latency-accuracy trade-off:

| Strategy | Approach | Problem |
|---|---|---|
| Full concatenation | All retrieved documents in one context | Prefill bottleneck: quadratic attention cost scales with total context length |
| Separate KV caches | Encode each document independently | No cross-document interaction: the model cannot reason across multiple sources |

**Pced** (Parallel Context-of-Experts Decoding) resolves this tension without training by shifting evidence aggregation from the attention phase to the decoding phase.

> [!NOTE]
> "We propose Pced, a training-free framework that shifts evidence aggregation from the attention mechanism to the decoding phase."

## Applicability

- **Who**: Practitioners deploying open-weight LLMs in RAG pipelines that require multi-document reasoning (e.g., open-domain QA, multi-hop QA).
- **When**: Inference time, on corpora with precomputed KV caches.
- **Where**: Systems that can access raw output logits (closed-source/API-only models are excluded).
- **Not applicable**: Closed-source models where logit access is restricted.

# Architecture

## Input / Output

| Component | Input | Output |
|---|---|---|
| Retriever | Query $q$ | Top-$N$ document chunks $\{d_1, \ldots, d_N\}$ with retrieval scores $\{r_1^{\text{ret}}, \ldots, r_N^{\text{ret}}\}$ |
| Cross-encoder reranker | $(q, d_k)$ pair | Reranking score $r_k^{\text{rank}}$ |
| Relevance fusion | $r_k^{\text{ret}}, r_k^{\text{rank}}$ | Fused relevance $r_k$ via harmonic mean |
| KV cache lookup | $d_k$ | Precomputed key-value pairs for each layer |
| Pced decoder | $N+1$ parallel logit vectors at each step | Next token $\hat{t}$ |

## Offline KV Cache Preparation

For each document $d_k$ in the corpus, the model encodes $d_k$ independently and stores its layer-wise key-value tensors:

$$\text{KV}_k^{(l)} \in \mathbb{R}^{|d_k| \times 2 \times H \times d_h}$$

where $l$ is the layer index, $H$ is the number of attention heads, and $d_h$ is the per-head dimension. At query time, the top-$N$ cached KV tensors are loaded without re-encoding, yielding $O(1)$ prefill cost per document (amortized).

## Relevance Score Fusion

Retrieval score and reranking score are combined via harmonic mean to prevent either signal from dominating:

$$r_k = \frac{2 \cdot r_k^{\text{ret}} \cdot r_k^{\text{rank}}}{r_k^{\text{ret}} + r_k^{\text{rank}}}$$

## Parallel Context-of-Experts Architecture

At each decode step, $N+1$ parallel forward passes run in a single batched call:

- **Amateur expert** (index 0): uses an empty KV cache, representing the model's prior distribution $p_0(t \mid q)$.
- **Contextual experts** (indices $1, \ldots, N$): each uses the KV cache of a single retrieved document $d_k$, producing $p_k(t \mid q, d_k)$.

The batched input has shape $[N+1, 1, d_{\text{model}}]$ at each decoding step, reusing the $N+1$ KV caches without cross-document attention.

# Retrieval-Aware Contrastive Decoding

## Calibration Formula

For each expert $k \in \{1, \ldots, N\}$, the adjusted logit score at token $t$ is:

$$\hat{s}_k(t) = (1 + \beta_0)\, s_k(t) - \beta_0\, s_0(t) + \gamma \log r_k$$

| Symbol | Meaning |
|---|---|
| $s_k(t) = \log p_k(t \mid q, d_k)$ | Log-probability of token $t$ under expert $k$ |
| $s_0(t) = \log p_0(t \mid q)$ | Log-probability under the amateur expert (prior) |
| $\beta_0 \geq 0$ | Contrastive strength: amplifies document-conditioned signal relative to prior |
| $\gamma \geq 0$ | Retrieval gating weight: up-weights more relevant documents |
| $r_k \in (0, 1]$ | Fused relevance score of document $k$ |

The $\gamma \log r_k$ term reduces the influence of low-relevance documents, making irrelevant experts contribute near-zero signal even if their raw logits are high.

## Token Selection (Max Aggregation)

The next token is chosen as:

$$\hat{t} = \arg\max_{t,\, k} \hat{s}_k(t)$$

This is a **token-level expert switching** strategy: at each generation step the model dynamically selects which expert contributes, enabling cross-document reasoning without shared attention.

## Dynamic $\beta$ Selection

Instead of fixing $\beta_0$, Pced selects $\beta_0$ dynamically per step from a small candidate set (e.g., $\{0.1, 0.5, 1.0\}$) by picking the value that maximises the entropy-weighted calibrated score, avoiding per-task tuning.

> [!IMPORTANT]
> Dynamic $\beta$ selection is critical for generalization: fixed $\beta$ values can degrade performance on tasks where contrastive decoding is unhelpful (e.g., tasks where the model prior already dominates).

# Algorithm: Pced Decoding Loop

```
Input:  query q, retrieved docs {d_1,...,d_N}, relevance scores {r_1,...,r_N},
        precomputed KV caches {KV_1,...,KV_N}, model M, hyperparameters γ
Output: generated response T

1. Build amateur KV cache KV_0 ← empty (model prior, no document context)
2. FOR step = 1, 2, ..., max_len:
   a. Forward pass (batched):
      [s_0(t), s_1(t), ..., s_N(t)] ← M([KV_0, KV_1, ..., KV_N], q, T_prev)
   b. Select β_0 dynamically from candidate set
   c. FOR each expert k in {1,...,N}:
         ŝ_k(t) = (1+β_0)·s_k(t) - β_0·s_0(t) + γ·log(r_k)   for all t
   d. ŝ*(t) = max_k ŝ_k(t)   (max aggregation over experts)
   e. t̂ = argmax_t ŝ*(t)
   f. Append t̂ to T; update all KV caches with new token
3. RETURN T
```

# Comparison with Related Methods

| Method | Cross-doc Reasoning | Prefill Cost | Training Required | Logit Access Needed |
|---|---|---|---|---|
| Full concatenation | Yes (shared attention) | $O((\sum_k |d_k|)^2)$ | No | No |
| Separate KV caches (no fusion) | No | $O(|d_k|^2)$ per doc | No | No |
| KV cache merging (avg/concat) | Partial (static merge) | $O(N \cdot |d_k|^2)$ once | No | No |
| CAD / Contrastive Decoding | Partial (single doc) | $O(|d_k|^2)$ | No | Yes |
| **Pced (this work)** | **Yes (decode-time)** | **$O(|d_k|^2)$ per doc** | **No** | **Yes** |

> [!TIP]
> Contrastive Decoding (CD) was introduced for single-model expert/amateur contrast. Pced extends this to multi-document RAG by treating each document as an independent expert and injecting retrieval relevance scores.

# Experiments

## Datasets

| Dataset | Task | Split Used |
|---|---|---|
| Natural Questions (NQ) | Open-domain QA (single-hop) | LOFT benchmark subset |
| HotpotQA | Multi-hop QA (cross-document) | LOFT benchmark subset |
| MuSiQue | Multi-hop QA | LOFT benchmark subset |
| KV Retrieval | Synthetic key-value ICL | LOFT benchmark subset |
| LongBench QA tasks | Query-focused QA | Standard LongBench split |
| LongBench summarization | Summarization | Standard LongBench split |
| LongBench code | Code completion | Standard LongBench split |

## Hardware

Not explicitly stated in the fetched content.

## Models Evaluated

- Mistral-Nemo-13B-Instruct
- Llama-3.1-8B-Instruct
- Qwen3-8B

## Key Results

**Latency (65k token context, Llama-3.1-8B):**
- Time-to-first-token: 0.14 s (Pced) vs. 25.50 s (full concatenation) — **>180× speedup**
- End-to-end latency: ~1.7× reduction

**Multi-hop QA (Llama-3.1-8B, HotpotQA):**
- Pced: 64 accuracy
- KV merging baseline: 16 accuracy
- Full concatenation: comparable to Pced despite not sharing attention at prefill

**Single-hop QA (NQ):**
- Pced: 85 accuracy
- Single-document baseline: 58
- Full concatenation: 79

**Scalability:** Performance is stable as the number of candidate documents increases from 8 to 128, demonstrating effective suppression of irrelevant experts via the $\gamma \log r_k$ term.

# Limitations

1. **Logit access required**: The contrastive decoding formula requires raw output logits, which are unavailable for closed-source or API-only LLMs (e.g., GPT-4, Claude).
2. **Storage overhead**: Storing precomputed KV caches proportional to corpus size is costly. For HotpotQA with Llama-3.1-8B, the KV store requires approximately 11.04 GB.
3. **Retrieval quality dependency**: If the retriever fails to surface relevant documents, Pced has no mechanism to recover; it only aggregates what is retrieved.
