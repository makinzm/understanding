# Meta Information

- URL: [H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., et al. (2023). H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. arXiv:2306.14048.

# H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

## Overview

H₂O is a KV-cache eviction policy for LLM inference that reduces memory consumption by retaining only a small budget of $k$ cached key-value pairs per decoding step, selected by combining **heavy-hitter tokens** (those with highest accumulated attention scores) and **recent tokens** (most recent $k/2$ tokens). The method exploits empirical observation that attention matrices in pre-trained LLMs are over 95% sparse at inference time, meaning that fewer than 5% of KV entries dominate the output for any given decoding step.

**Applicability:** Engineers and researchers deploying large-scale LLMs (OPT, LLaMA, GPT-NeoX) who face GPU memory bottlenecks due to KV-cache growth. Especially relevant for long-sequence inference, large batch sizes, and latency-sensitive production scenarios.

## Background: KV Cache Memory Problem

Autoregressive generation maintains a KV cache of all past key-value pairs for each attention head. For a model with $L$ layers, $H$ heads, head dimension $d$, batch size $B$, and sequence length $n$, the cache size is:

$$\text{KV cache size} = 2 \cdot L \cdot H \cdot d \cdot B \cdot n \cdot \text{sizeof}(\text{dtype})$$

For a 30B-parameter model with batch size 128 and sequence length 1024, this reaches **180 GB**, far exceeding GPU VRAM capacity. This bottleneck forces practitioners to use small batch sizes, reducing throughput.

## Key Empirical Observations

### Attention Sparsity

Attention weight matrices $A \in \mathbb{R}^{n \times n}$ computed during decoding show over 95% sparsity: at each generation step, only about 5% of the $n$ cached KV positions receive non-negligible attention weight. This implies that retaining a small subset of KV pairs is sufficient for accurate decoding.

### Heavy-Hitter Phenomenon

The accumulated attention score for token $j$ up to step $i$ is defined as:

$$o_j = \sum_{t=1}^{i} a_{t,j}, \quad a_{t,j} = \text{softmax}\!\left(\frac{Q_t K_j^\top}{\sqrt{d}}\right)$$

These accumulated scores follow a **power-law distribution** across positions: a small set of tokens (heavy-hitters, H²) consistently accumulate high scores and are attended to by many subsequent query tokens. Retaining these heavy-hitters is both necessary and sufficient to approximate full-cache attention.

### Greedy Local Selection Sufficiency

Heavy-hitters identified using only local statistics (attention scores from the current and preceding tokens) match those identified using global future-token statistics. This means the eviction decision can be made online, without look-ahead.

## H₂O Algorithm

### Problem Formulation

Let $[n] = \{1, \ldots, n\}$ be the token indices processed so far. An **eviction policy** maps cache state $S_{i-1} \subseteq [i-1]$ to updated state $S_i \subseteq [i]$ subject to:

- $|S_i| = k$ (fixed cache budget)
- $|S_i \setminus S_{i-1}| \leq 1$ (at most one eviction per step)

The score function for a candidate set $T \subseteq [i]$ is:

$$F_{\text{score}}(T) = \sum_{j \in T} o_j$$

where $o_j$ is the normalized accumulated attention score for token $j$. H₂O greedily maximizes $F_{\text{score}}$ at each step.

### Pseudocode

```
Input:  query Q ∈ ℝ^{n×d}, key K ∈ ℝ^{n×d}, cache budget k
Output: indices S_n ⊆ [n] of retained KV pairs

S_0 = {}
o = zeros(n)          # accumulated attention scores

for i = 1 to n:
    if i <= k:
        S_i = S_{i-1} ∪ {i}       # fill cache during prefill
    else:
        # Compute normalized attention for step i over current cache + new token
        candidates = S_{i-1} ∪ {i}
        a_i = softmax(Q_i · K_{candidates}^T / sqrt(d))   # shape: (|candidates|,)
        o[candidates] += a_i                               # update accumulated scores

        # Evict the token with minimum accumulated score
        u = argmin_{j ∈ candidates} o[j]
        S_i = candidates \ {u}                             # retain top-k by score

return S_n
```

> [!NOTE]
> The cache naturally splits into two roles: **heavy-hitters** (tokens with high $o_j$, retained across many steps) and **recent tokens** (tokens newly added, not yet evicted). Ablation studies confirm both components are necessary — using either alone causes 3%–23% accuracy degradation.

### Dimensions

- $Q_i \in \mathbb{R}^{1 \times d}$: query vector at decoding step $i$
- $K_{S_{i-1},*} \in \mathbb{R}^{k \times d}$: keys for cached positions
- Attention output $a_i \in \mathbb{R}^{k}$: normalized attention weights over cached entries
- Cache indices $S_i \subseteq [i]$: set of $k$ retained token positions

## Theoretical Guarantee

**Theorem (informal):** Under mild submodularity assumptions on the attention function, the greedy H₂O eviction policy achieves:

$$F_{\text{score}}(\tilde{S}_i) \geq (1 - \alpha)(1 - 1/e) \cdot \max_{|S|=k} F_{\text{score}}(S) - \beta$$

where $\tilde{S}_i$ is the greedy solution, $(1 - 1/e) \approx 0.632$ is the standard submodular greedy ratio, and $\alpha, \beta > 0$ are small approximation parameters. This establishes that the greedy policy is near-optimal even though it cannot see future queries.

## Comparison with Related Methods

| Method | Strategy | Long-context | Memory Reduction | Notes |
|---|---|---|---|---|
| Full KV Cache | Retain all tokens | Limited by memory | 0× | Baseline |
| Local Window | Retain only last $w$ tokens | No heavy-hitter retention | High | Misses early influential tokens |
| StreamingLLM | Attention sinks + recent window | Yes, fixed size | High | No per-head accumulation |
| **H₂O (ours)** | Heavy-hitters + recent tokens | Yes, up to 4M tokens | 5–10× | Per-head scoring, dynamic update |

> [!IMPORTANT]
> StreamingLLM retains a fixed set of "attention sink" tokens (typically the first few tokens) plus a sliding window. H₂O dynamically identifies heavy-hitters per head using accumulated scores, achieving lower perplexity on long sequences (PG-19 dataset) at equivalent cache budgets.

## Implementation

H₂O uses a **circular queue** per attention head to manage cache slots efficiently:
- Slot 0 to $k/2 - 1$: heavy-hitter positions (updated when a heavy-hitter is evicted and replaced)
- Slot $k/2$ to $k - 1$: recent-token circular buffer
- No memory swapping; eviction is an in-place pointer update

The system is built on top of FlexGen and supports OPT, LLaMA, and GPT-NeoX architectures. Quantization (INT4/INT8) is compatible and can be combined with H₂O for additional compression.

## Experiments

- **Datasets:**
  - XSUM (abstractive summarization, 1000 samples)
  - CNN/DailyMail (summarization, 1000 samples)
  - COPA (causal reasoning)
  - MathQA (math word problems)
  - OpenBookQA (science QA)
  - PiQA (physical intuition QA)
  - RTE (textual entailment)
  - Winogrande (commonsense reasoning)
  - AlpacaEval (instruction-following generation)
  - MT-Bench (multi-turn conversation quality)
  - PG-19 (long-document language modeling, up to 4M tokens)

- **Models tested:** OPT-6.7B, OPT-30B, LLaMA-7B, GPT-NeoX-20B

- **Hardware:** NVIDIA A100 GPUs

- **Results:**
  - With 20% cache budget ($k = 0.2n$): matches full KV-cache accuracy on all benchmarks
  - Throughput: 29× over DeepSpeed Zero-Inference, 29× over HuggingFace Accelerate, 3× over FlexGen (OPT-30B)
  - Latency: up to 1.9× reduction at identical batch size on A100
  - Long-context: lower perplexity than StreamingLLM on PG-19 up to 4M token sequences
  - Ablation: heavy-hitters alone cause 2.85%–22.75% accuracy drop; recent tokens alone cause 5.5%–16% drop; combined H₂O is within <1% of baseline across zero-shot to ten-shot settings
