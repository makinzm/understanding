# Meta Information

- URL: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. Google Research.

# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

## Overview

GQA addresses the memory bandwidth bottleneck during autoregressive transformer inference. The core problem is that multi-head attention (MHA) must load all key-value (KV) heads for every decoding step, which becomes a bandwidth-limited operation when batch sizes are small. Multi-query attention (MQA) solved this by collapsing all KV heads into one, but this can degrade model quality. GQA introduces a configurable middle ground and a cost-effective way to convert existing MHA checkpoints without full retraining.

**Who benefits**: Engineers deploying large transformer-based language models (e.g., T5, LLaMA) in low-latency serving environments, where memory bandwidth is the inference bottleneck.

## Attention Variants

| Method | # KV Heads | Memory Bandwidth | Quality |
|--------|-----------|-----------------|---------|
| Multi-Head Attention (MHA) | $H$ (same as query heads) | High | Best |
| Multi-Query Attention (MQA) | $1$ | Lowest | Degraded |
| Grouped-Query Attention (GQA-G) | $G$ ($1 \le G \le H$) | Intermediate | Near-MHA |

**Input/Output for attention**:
- Query: $Q \in \mathbb{R}^{B \times T \times H \times d_k}$
- Key/Value (MHA): $K, V \in \mathbb{R}^{B \times T \times H \times d_k}$
- Key/Value (MQA): $K, V \in \mathbb{R}^{B \times T \times 1 \times d_k}$
- Key/Value (GQA-G): $K, V \in \mathbb{R}^{B \times T \times G \times d_k}$
- Output: $O \in \mathbb{R}^{B \times T \times H \times d_k}$

Where $B$ = batch size, $T$ = sequence length, $H$ = number of heads, $G$ = number of groups, $d_k$ = head dimension.

## Method

### 2.1 Uptraining: Converting MHA Checkpoints to MQA/GQA

Instead of training a new model from scratch, the authors propose "uptraining" an existing MHA checkpoint. The conversion procedure is:

1. **Mean-pool** the $H$ KV projection weight matrices into $G$ matrices (one per group):
   $$W_K^{(g)} = \frac{1}{|S_g|} \sum_{h \in S_g} W_K^{(h)}, \quad W_V^{(g)} = \frac{1}{|S_g|} \sum_{h \in S_g} W_V^{(h)}$$
   where $S_g$ is the set of original heads assigned to group $g$.

2. **Continue pre-training** for $\alpha \times T_{\text{original}}$ steps using the same data and hyperparameters, where $\alpha = 0.05$ (i.e., 5% of the original pre-training budget).

> [!NOTE]
> The paper evaluated three initialization strategies: (1) mean pooling, (2) selecting a single head, (3) random initialization. Mean pooling consistently outperformed the others because it preserves the most information from the pre-trained checkpoint.

### 2.2 Grouped-Query Attention (GQA)

GQA partitions the $H$ query heads into $G$ groups of equal size $H/G$. All query heads within group $g$ share the same key and value heads $K^{(g)}$ and $V^{(g)}$.

**Pseudocode for GQA forward pass**:
```
Input: Q ∈ R^{B×T×H×d_k}, K_grouped ∈ R^{B×T×G×d_k}, V_grouped ∈ R^{B×T×G×d_k}
For each query head h in [1..H]:
    g = ceil(h * G / H)           # assign head h to group g
    A[h] = softmax(Q[h] @ K[g]^T / sqrt(d_k))  # attention scores
    O[h] = A[h] @ V[g]            # weighted sum
Output: concat(O[1], ..., O[H]) projected back to d_model
```

**Special cases**:
- $G = 1$: equivalent to MQA (all queries share one KV head)
- $G = H$: equivalent to MHA (each query has its own KV head)
- $G = 8$: the default recommended setting (empirically validated)

> [!IMPORTANT]
> GQA provides an additional practical benefit during model sharding: with MQA on tensor-parallel systems, the single KV head must be replicated across devices (wasting memory). GQA with $G \ge$ number of devices avoids this replication. MHA reduces bandwidth proportionally only for single-device deployments.

## Experiments

- **Datasets**:
  - Summarization: CNN/Daily Mail, arXiv, PubMed, MediaSum, Multi-News
  - Translation: WMT 2014 English-to-German
  - Question Answering: TriviaQA
- **Base models**: T5 Large and T5 XXL (encoder-decoder architecture)
- **Implementation**: JAX/Flax
- **Hardware**: TPUv3 chips; uptraining at $\alpha=0.05$ cost ~600 TPUv3 chip-days for T5-XXL
- **Optimizer**: Adafactor (same hyperparameters as original T5 training)

### Key Results

- **Inference latency** (T5-XXL, batch=1, sequence=512 tokens):
  - MHA-XXL: ~1.51 s/step
  - MQA-XXL: ~0.24 s/step (6.3× faster)
  - GQA-8-XXL: ~0.28 s/step (5.4× faster, near-MQA speed)
- **Quality**: GQA-8-XXL matches MHA-XXL quality within noise on summarization/QA/translation benchmarks, while MQA shows measurable degradation on some tasks.

### Ablation: Number of Groups

Increasing $G$ from 1 to 8 adds only ~0.5 ms inference overhead per step but yields substantial quality gains. Beyond 8, overhead increases more steeply as the configuration approaches MHA.

### Ablation: Uptraining Proportion ($\alpha$)

Both MQA and GQA see most of their recovery from $\alpha=0$ to $\alpha=0.05$, with diminishing returns beyond $\alpha=0.10$. GQA reaches acceptable quality with less uptraining than MQA, suggesting it is a more efficient conversion target.

## Comparison with Related Methods

| Method | Approach | Key Difference from GQA |
|--------|----------|------------------------|
| MQA (Shazeer 2019) | Single KV head from scratch | Requires full retraining; no uptraining recipe |
| Flash Attention | Tiled SRAM computation | Reduces compute cost, not bandwidth cost |
| Speculative Sampling | Draft model for parallel decoding | Orthogonal; complementary to GQA |
| Multi-head Latent Attention (MLA, DeepSeek) | Low-rank KV compression | Compresses KV further via projection; not uptraining-based |

> [!TIP]
> GQA has been adopted in production models including LLaMA-2 (70B), Mistral 7B, and Gemma. In these decoder-only models the authors expect GQA's advantage to be even larger because encoder-decoder models already distribute compute across the full sequence at encoding time.

## Limitations

- Quality measured primarily with ROUGE scores, which are imperfect for summarization.
- Evaluation limited to encoder-decoder (T5); stronger GQA advantages are expected in decoder-only architectures but not directly measured here.
- No direct comparison of uptrained GQA versus training GQA from scratch.
