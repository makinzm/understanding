# Meta Information

- URL: [[2306.14048] H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Ré, C., Barrett, C., Wang, Z., & Chen, B. (2023). H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. NeurIPS 2023.

> [!CAUTION]
> NOTE comments are my personal understanding and may contain errors.

# 1. Motivation

During autoregressive generation, a Transformer-based LLM caches the key and value vectors of every previously generated token so it does not need to recompute them at each decoding step (the "KV cache"). This cache grows linearly with sequence length and batch size, and it quickly becomes the dominant memory consumer, more than the model weights themselves, when serving long contexts or large batches. For example, a 30B-parameter OPT model with batch size 128 and sequence length 1024 requires around 180GB of KV cache, far exceeding the model's own weight size.

This document is applicable to practitioners who serve autoregressive LLMs (e.g., OPT, LLaMA, GPT-NeoX family models) for long-context or high-throughput generation on memory-constrained GPUs, and who want to reduce KV cache memory without retraining the model or modifying its architecture.

The authors observe that attention matrices computed during inference are highly sparse (over 95% sparsity in some layers), meaning most cached key/value pairs contribute almost nothing to the output at a given decoding step. This suggests that most of the cache can be evicted without materially changing the model's output, if the right tokens are kept.

# 2. Heavy Hitters (H$_2$)

## a. Definition

```math
\begin{align}
n & \in \mathbb{Z}^{+} : \text{number of tokens generated so far} \\
d & \in \mathbb{Z}^{+} : \text{head dimension} \\
Q & \in \mathbb{R}^{n \times d} : \text{query matrix} \\
K & \in \mathbb{R}^{n \times d} : \text{key matrix} \\
S_i & \subseteq [n] : \text{set of token indices retained in the KV cache at decoding step } i
\end{align}
```

For a token index set $S_i \subset [n]$ evaluated at generation step $i$, the (normalized, causally-masked) attention output restricted to the cache is:

```math
\begin{align}
o_i := D_i^{-1} \exp\left(Q_{i,*} K_{S_i,*}^{T}\right), \quad D_i := \left(\exp\left(Q_{i,*} K_{S_i,*}^{T}\right)\right) \cdot \mathbf{1}_{|S_i|}
\end{align}
```

where $Q_{i,*} \in \mathbb{R}^{1 \times d}$ is the query of the current token, $K_{S_i,*} \in \mathbb{R}^{|S_i| \times d}$ is the subset of cached keys indexed by $S_i$, and $D_i \in \mathbb{R}$ is the softmax normalizer over the cached keys.

Empirically, the accumulated attention score received by a token across generation steps follows a power-law distribution: a small number of tokens ("Heavy Hitters", H$_2$) account for the majority of the total attention mass. The paper reports that the identities of these heavy hitters are strongly correlated with word co-occurrence statistics in the training corpus (i.e., frequently co-occurring tokens tend to become heavy hitters regardless of their position), and that removing them from the cache causes a sharp increase in perplexity, much larger than removing an equal number of randomly chosen tokens.

> [!NOTE]
> The paper states: "The emergence of H$_2$ is natural and strongly correlates with the frequent co-occurrence of tokens in the text, and removing them results in significant performance degradation."

## b. Why local (greedy) statistics suffice

Identifying heavy hitters in principle requires knowing the *full* future attention pattern of a token (i.e., how much attention it will receive from all tokens generated after it), which is not available at cache-eviction time. The authors show empirically that using only the *accumulated attention score received so far* (a purely local/causal, greedy statistic) is nearly as effective at identifying heavy hitters as using the full (oracle) future attention pattern. This observation is what makes an online, step-by-step eviction policy practical.

# 3. H$_2$O Eviction Policy

## a. Cache budget

At each decoding step the cache is constrained to hold at most $k$ tokens, split between:
- Heavy hitters: tokens with the largest accumulated attention score so far.
- Recent tokens: the most recently generated tokens (a local, recency-based window).

This combination matters because heavy hitters alone are insufficient: recent tokens, even if they have not yet accumulated much attention, are often needed to correctly attend to the immediate local context (e.g., for tasks requiring nearby-token dependencies), while heavy hitters alone would otherwise be evicted before they show their importance if only recency were used.

## b. Formulation as dynamic submodular maximization

**Definition (Submodularity, adapted from the paper's Definition 4.1).** For $Z \subset [n]$, a set function $f(\cdot) := F(Z, \cdot): 2^{[n]} \to \mathbb{R}$ is submodular if for all $X, Y$ with $Z \subseteq X \subseteq Y \subset [n]$ and all $x \in [n] \setminus Y$:

```math
\begin{align}
f(X \cup \{x\}) - f(X) \geq f(Y \cup \{x\}) - f(Y)
\end{align}
```

i.e., adding a token to a smaller set yields at least as much marginal gain as adding it to a larger set (diminishing returns). The authors define the eviction-score function $F_{\text{score}}(T) := \sum_{s \in T} o_s$, the sum of attention outputs contributed by the tokens in cache subset $T$, and show that under the observed attention-sparsity structure this function is submodular.

**Greedy eviction rule (Definition 4.3).** At decoding step $i$, given the previous cache $S_{i-1}$ (of size $k$) and the newly generated token $i$, the new cache is:

```math
\begin{align}
S_i \leftarrow (S_{i-1} \cup \{i\}) \setminus \{u\}, \qquad u \leftarrow \operatorname*{argmin}_{v \in S_{i-1} \cup \{i\}} F_{\text{score}}\left((S_{i-1} \cup \{i\}) \setminus \{v\}\right)
\end{align}
```

In words: add the new token to the cache, then evict whichever single token's removal costs the *least* total attention score, i.e., the token contributing the least accumulated attention. This is a greedy, dynamic (streaming) submodular maximization procedure, performed once per decoding step, and it only requires $O(1)$ additional bookkeeping (a running attention-score tally per cached token) beyond standard attention computation.

## c. Algorithm (pseudocode)

```
Input: cache budget k = k_heavy + k_recent, generated tokens x_1..x_n
Initialize S_0 = {}, score[j] = 0 for all j
for i = 1 to n:
    compute attention output o_i using cached keys/values indexed by S_{i-1} ∪ {i}
    for each cached token j in S_{i-1} ∪ {i}:
        score[j] += attention_weight(i, j)      # accumulate attention mass received
    candidates = S_{i-1} ∪ {i}
    recent_set  = k_recent most recent indices in candidates
    heavy_pool  = candidates \ recent_set
    S_i = recent_set ∪ (top k_heavy indices of heavy_pool by score[j])
    evict all tokens in candidates \ S_i from the KV cache
Output: final cache trajectory S_1, ..., S_n
```

## d. Theoretical guarantee (Theorem 4.4)

Under the assumption that the attention scheme is submodular (and mild regularity conditions), the greedy policy applied at every step achieves a near-optimal cache with respect to the best possible fixed-size cache in hindsight:

```math
\begin{align}
f(\tilde{S}_i) \geq (1 - \alpha)\left(1 - \frac{1}{e}\right) \max_{|S| = k} f(S) - \beta
\end{align}
```

where $\tilde{S}_i$ is the cache produced by the greedy policy at step $i$, $\alpha, \beta > 0$ are small approximation-error terms that depend on how closely the local (causal) statistics approximate the true submodular oracle, and $(1 - 1/e) \approx 0.632$ is the classical greedy approximation ratio for monotone submodular maximization under a cardinality constraint.

> [!IMPORTANT]
> The guarantee only holds under the submodularity assumption on the attention scheme; the authors note this may not hold exactly for every layer/head of every architecture, so the bound is best understood as a theoretical justification for the greedy heuristic rather than a strict, universally verified guarantee.

# 4. Differences from Related Methods

| Method | Mechanism | Requires retraining? | Handles heavy hitters explicitly? |
|---|---|---|---|
| StreamingLLM | Keeps a fixed "attention sink" (first few tokens) + a sliding recent window | No | No — uses a fixed sink position, not attention-score-based selection |
| Sparse Transformer (strided/fixed) | Fixed, predefined sparsity pattern applied without fine-tuning | No (as tested here) | No — causes high miss rate and accuracy loss when applied directly to a dense-trained model |
| SpAtten | Accumulates attention scores to prune tokens/heads | No | Partially — does not account for the variance of importance across different heads and layers |
| Multi-query / grouped-query attention | Shares key/value heads across query heads to shrink cache size | Yes (needs retraining or fine-tuning) | No — reduces cache size structurally, independent of token importance |
| Quantization (e.g., 4-bit KV cache) | Reduces per-element storage size of cached K/V | No | No — orthogonal to which tokens are kept |
| H$_2$O (this paper) | Dynamic greedy eviction combining heavy hitters (by accumulated attention score) + recent window | No | Yes — this is the central mechanism |

The paper reports that H$_2$O evaluated against StreamingLLM-style caching achieves lower perplexity at equal cache size across sequence lengths up to four million tokens, and that combining H$_2$O with quantization ("almost always achieves better accuracy than H$_2$O or quantization alone") confirms the two techniques are complementary rather than redundant, since one reduces *which* tokens are stored and the other reduces the storage cost *per* stored token.

# 5. Experiments

- Datasets/Benchmarks:
  - HELM framework: XSUM, CNN/Daily Mail (summarization)
  - lm-eval-harness: COPA, MathQA, OpenBookQA, PiQA, RTE, Winogrande (multiple-choice / classification-style reasoning tasks)
  - Generation quality: AlpacaEval, MT-Bench
- Models tested: OPT (6.7B, 30B, up to 175B in some experiments), LLaMA (7B, 13B), GPT-NeoX-20B
- Hardware: NVIDIA A100 (80GB) and NVIDIA T4 (16GB) GPUs
- Cache budgets evaluated: 4%, 10%, 20%, and 60% of the full prompt length, with 20% as the headline operating point
- Results:
  - At a 20% KV cache budget, H$_2$O matches the full-cache (100%) accuracy across the tested OPT, LLaMA, and GPT-NeoX models on the downstream tasks above, whereas naively applying fixed sparse-attention patterns at the same budget loses up to 35% accuracy.
  - Throughput (T4 GPU, OPT-6.7B, 512+32 sequence length): H$_2$O reaches 35.1 tokens/s vs. 20.2 (FlexGen, 1.74×), 10.2 (DeepSpeed, 3.44×), and 20.4 (Hugging Face Accelerate, 1.72×) tokens/s.
  - Throughput (T4 GPU, OPT-30B, 512+1024 sequence length): H$_2$O reaches 13.82 tokens/s vs. 7.1 (FlexGen, 1.95×) and 0.6 (DeepSpeed, 23×) tokens/s.
  - Latency (A100, OPT-30B, 7000+1024 sequence length): H$_2$O reaches 50.4s vs. 57.0s for FlexGen (1.13× faster).
  - Throughput (A100, OPT-6.7B, 2048+2048 sequence length, batch 64): H$_2$O reaches 1161.0 tokens/s while FlexGen runs out of memory (OOM) at this batch size.
  - On the real-world XSUM summarization dataset, H$_2$O reaches 30.40 tokens/s (OPT-6.7B) and 6.70 tokens/s (OPT-30B), versus FlexGen's 10.80 and 3.29 tokens/s respectively (2.81× and 2.04× speedups).
  - Overall, across the reported hardware/model/sequence-length configurations, H$_2$O achieves up to 29× throughput improvement over DeepSpeed and Hugging Face Accelerate, and up to 3× over FlexGen, together with up to 1.9× latency reduction on A100 GPUs, while also enabling roughly 5× reduction in KV cache memory at equal accuracy.

# 6. Limitations

- The theoretical near-optimality guarantee (Theorem 4.4) depends on an assumption that the attention scheme is submodular; the paper does not claim this holds exactly for every layer and attention head of every architecture, only that it is a reasonable approximation supported by empirical sparsity observations.
- The tested models were originally trained with a 2K-token context length; while H$_2$O is evaluated on sequences far longer than this (up to millions of tokens in the StreamingLLM-style comparison), extrapolation this far beyond the training distribution may behave differently for other model families not tested.
- The paper focuses on decoder-only autoregressive models (OPT, LLaMA, GPT-NeoX); applicability to encoder-decoder or non-causal architectures is not evaluated.
- SpAtten-style variance in token importance across heads and layers is noted as a phenomenon, but H$_2$O's own per-head/per-layer behavior under this variance is not exhaustively characterized for all tested architectures.

> [!TIP]
> Reference implementation (built on top of FlexGen): https://github.com/FMInference/H2O
