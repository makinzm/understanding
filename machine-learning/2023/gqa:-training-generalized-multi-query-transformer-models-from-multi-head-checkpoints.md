# Meta Information

- URL: [[2305.13245] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- LICENSE: [Deed - Attribution 4.0 International - Creative Commons](https://creativecommons.org/licenses/by/4.0/)
- Reference: Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv:2305.13245.

> [!CAUTION]
> NOTE comments is my personal understanding and may contain errors.

# 1. Introduction

Autoregressive decoder inference in Transformers is a severe bottleneck for large language models, because at each decoding step the model must reload the entire key-value (KV) cache from memory, and this memory-bandwidth cost dominates the arithmetic cost when the batch is small (e.g. incremental decoding of a single sequence at a time).

Multi-query attention (MQA), proposed prior to this paper, reduces the KV cache from $H$ key/value heads to a single shared key/value head while keeping $H$ separate query heads. This shrinks the amount of data reloaded per step by a factor of $H$ and greatly speeds up decoding, but it can degrade model quality and training stability because it removes almost all of the head-level diversity in the key/value representations.

This paper addresses two practical problems with MQA:

1. Existing large multi-head attention (MHA) checkpoints (e.g. public T5 checkpoints) cannot use MQA's speed benefits without being retrained from scratch, which is expensive.
2. MQA alone can be too aggressive a reduction, hurting quality and training stability relative to MHA.

The authors propose two contributions to solve these problems:

1. **Uptraining**: a recipe to convert an existing MHA checkpoint into an MQA (or GQA) checkpoint using only a small additional amount of pretraining compute (as low as 5% of the original pretraining compute), instead of retraining from scratch.
2. **Grouped-query attention (GQA)**: a generalization that interpolates between MHA and MQA by sharing key/value heads across groups of query heads rather than collapsing them to one shared head, giving a tunable quality/speed tradeoff.

> [!NOTE]
> Applicability: this technique is aimed at practitioners who already have a large pretrained MHA Transformer (encoder-decoder or decoder-only) and want cheaper, faster incremental-decoding inference (e.g. for summarization, translation, or QA serving) without paying the cost of pretraining a new MQA model from scratch. It is most valuable for large models served at scale, where inference cost dominates over one-time uptraining cost.

# 2. Background: Why Multi-Query Attention Speeds Up Inference

```math
\begin{align}
h \in \mathbb{Z}^{+} & : \text{number of attention heads in MHA} \\
d \in \mathbb{Z}^{+} & : \text{dimension per head} \\
n \in \mathbb{Z}^{+} & : \text{current sequence length (context so far)}
\end{align}
```

During autoregressive decoding, at every generated token the model must read the cached keys and values for all previous positions to compute attention. In standard multi-head attention (MHA), each of the $h$ heads has its own key and value cache, so per step the amount of data loaded from memory scales with $h$. In multi-query attention (MQA), all $h$ query heads share a single key head and a single value head, so the KV cache that must be loaded per step is $h$ times smaller.

> [!NOTE]
> Direct reasoning from the paper: "Multi-query attention reduces the memory bandwidth overhead from loading the keys and values by sharing a single key and value head across all query heads... the memory bandwidth needed to load the keys and values every decoding step is reduced by a factor of $h$."

The paper also observes that this bottleneck matters less as models scale up, because the KV cache size scales with the model's hidden dimension (linearly), while the number of FLOPs and parameters scale roughly with the square of the hidden dimension: "larger models suffer relatively less from memory bandwidth overhead from attention, as the KV-cache scales with model dimension while model FLOPs and parameters scale with the square of model dimension." This is one reason the authors introduce GQA — for very large models the aggressive reduction of MQA may be unnecessary, and a milder reduction (GQA) can preserve more quality while still removing most of the bandwidth bottleneck.

A further practical constraint: on multi-accelerator (model-parallel) deployments, a single shared key/value head (MQA) must be replicated across every partition/shard, which itself adds overhead — "standard sharding for large models replicates the single key and value head by the number of model partitions." GQA, by keeping multiple (but fewer than $h$) key/value heads, allows a natural way to keep at least one key/value head per shard.

# 3. Method

## 3.1. Uptraining: Converting MHA Checkpoints to Multi-Query Checkpoints

Instead of training a multi-query (or grouped-query) model from random initialization, the authors convert an existing pretrained MHA checkpoint in two steps:

**Step 1 — Checkpoint conversion.** For each target key/value head, the corresponding original per-head key (or value) projection matrices are combined into a single projection matrix by mean-pooling:

```math
\begin{align}
W_K^{(i)} \in \mathbb{R}^{d_{model} \times d} & : \text{key projection matrix of original head } i, \; i = 1, \dots, h \\
\bar{W}_K \in \mathbb{R}^{d_{model} \times d} & : \text{pooled key projection matrix for the new shared head} \\
\bar{W}_K &= \frac{1}{h}\sum_{i=1}^{h} W_K^{(i)}
\end{align}
```

The same mean-pooling is applied to the value projection matrices. The paper reports that mean-pooling the original heads' projection matrices worked better than two alternative conversion strategies: (a) selecting just one of the original heads' projection matrix and discarding the rest, or (b) initializing the new projection matrix randomly. Mean-pooling is preferred because it retains information from all of the original heads instead of discarding most of the pretrained representation.

**Step 2 — Additional pretraining.** After conversion, the model is pretrained further for a small proportion $\alpha$ of the original pretraining steps, using the same pretraining objective, data, and optimizer as the original model. In the experiments, $\alpha = 0.05$ (5% of original pretraining steps), and the total uptraining compute for the largest model was about 600 TPUv3 chip-days — small relative to pretraining a comparable model from scratch.

> [!IMPORTANT]
> Uptraining is applied to the decoder self-attention and the encoder-decoder cross-attention of the T5 models used in the experiments, but **not** to the encoder self-attention. This is because the encoder is processed in parallel (not autoregressively) during inference, so its KV cache is not the memory-bandwidth bottleneck that uptraining is meant to address.

## 3.2. Grouped-Query Attention (GQA)

```math
\begin{align}
h \in \mathbb{Z}^{+} & : \text{number of query heads (unchanged from the original MHA model)} \\
g \in \mathbb{Z}^{+} & : \text{number of groups}, \; 1 \le g \le h, \; h \bmod g = 0 \\
d \in \mathbb{Z}^{+} & : \text{dimension per head} \\
x \in \mathbb{R}^{n \times d_{model}} & : \text{input sequence of length } n
\end{align}
```

GQA divides the $h$ query heads into $g$ groups. All query heads within the same group share one key head and one value head, so the model has $h$ query projections but only $g$ key projections and $g$ value projections:

```math
\begin{align}
Q_i &= x W_Q^{(i)} \in \mathbb{R}^{n \times d}, & i = 1, \dots, h \\
K_j &= x \bar{W}_K^{(j)} \in \mathbb{R}^{n \times d}, & j = 1, \dots, g \\
V_j &= x \bar{W}_V^{(j)} \in \mathbb{R}^{n \times d}, & j = 1, \dots, g \\
\text{head}_i &= \text{softmax}\!\left(\frac{Q_i K_{g(i)}^\top}{\sqrt{d}}\right) V_{g(i)} \in \mathbb{R}^{n \times d}
\end{align}
```

where $g(i) = \lceil i \cdot g / h \rceil$ maps query head $i$ to its assigned group's shared key/value head, and each group's $\bar{W}_K^{(j)}, \bar{W}_V^{(j)}$ is produced by mean-pooling the original heads assigned to that group during checkpoint conversion (Section 3.1).

The number of groups $g$ is a tunable interpolation parameter between the two prior extremes:

| Setting | Equivalent to | KV cache size relative to MHA |
|---|---|---|
| $g = 1$ | Multi-query attention (MQA) | $1/h$ |
| $g = h$ | Multi-head attention (MHA) | $1$ (no reduction) |
| $1 < g < h$ | Grouped-query attention (GQA), e.g. GQA-8 means $g = 8$ | $g/h$ |

> [!NOTE]
> "GQA divides query heads into G groups, each of which shares a single key head and value head." GQA-1 is equivalent to MQA, and GQA-H (where G equals the original number of heads H) is equivalent to MHA.

This design lets a practitioner pick $g$ to trade off inference speed against quality: smaller $g$ gives cache savings and speed closer to MQA, larger $g$ gives quality closer to MHA. In the experiments the authors use $g = 8$ (denoted GQA-8) as their main configuration.

## Algorithm: Uptraining Pipeline (Pseudocode)

```
Input: pretrained MHA checkpoint θ_MHA with h heads, target group count g (g = 1 for MQA), uptraining proportion α
Output: uptrained checkpoint θ_GQA with g key/value heads

1. Partition the h query heads into g contiguous groups of size h/g
2. For each group j = 1..g:
     W_K_pooled[j] = mean( W_K[i] for i in group j )
     W_V_pooled[j] = mean( V_K[i] for i in group j )   # analogous mean-pooling of value projections
3. Initialize θ_GQA by copying all parameters from θ_MHA except replacing the per-head
   key/value projections with the pooled projections from step 2
4. Continue pretraining θ_GQA on the original pretraining data/objective/optimizer
   for α * (original total pretraining steps) additional steps
5. Return θ_GQA
```

# 4. Differences from Related Methods

| Method | Query heads | Key/Value heads | KV cache vs. MHA | Quality | Trained from scratch? |
|---|---|---|---|---|---|
| MHA (baseline) | $h$ | $h$ | $1\times$ | Highest | Yes |
| MQA | $h$ | $1$ | $1/h$ | Can degrade, less training-stable | Original MQA papers train from scratch |
| GQA (this paper) | $h$ | $g$, $1 < g < h$ | $g/h$ | Close to MHA | No — uptrained from an existing MHA checkpoint |

Compared to training an MQA model from scratch (the prior approach), this paper's uptraining recipe reuses an existing MHA checkpoint and needs only a small fraction of additional pretraining compute, which is the key practical difference. Compared to plain MQA, GQA keeps more than one key/value head, which the experiments show is both more accurate and more stable to train — the paper reports that GQA is more training-stable than MQA variants, whose training the authors observed could be less stable.

# 5. Experiments

- Dataset: CNN/Daily Mail and MultiNews (summarization of short/multi-document news), arXiv and PubMed (summarization of long scientific documents), MediaSum (summarization of interview transcripts), WMT 2014 English-to-German (translation), TriviaQA (question answering).
- Hardware: TPUv4 chips for inference timing measurements (8 TPUs per run, largest batch size that fits up to 32 per TPU); uptraining compute measured in TPUv3 chip-days (~600 chip-days for the largest model).
- Optimizer: Adafactor, with the same hyperparameters and learning rate schedule as the original T5 pretraining, for both original pretraining and the uptraining phase. Fine-tuning on downstream tasks used a constant learning rate of 0.001, batch size 128, and dropout rate 0.1, with input lengths ranging 512–2048 and output lengths ranging 32–512 depending on the task.
- Base models: T5.1.1 Large and XXL encoder-decoder checkpoints, implemented in JAX/Flax/Flaxformer, uptrained into MQA-XXL and GQA-8-XXL variants.
- Results: uptrained GQA-8-XXL achieves quality close to the original MHA-XXL model (e.g. ROUGE-1 on CNN/DM: 47.1 for GQA-8-XXL vs. 47.2 for MHA-XXL vs. 46.6 for MQA-XXL) while inference time per sample is close to MQA-XXL (0.28 ms for GQA-8-XXL vs. 0.24 ms for MQA-XXL vs. 1.51 ms for MHA-XXL, all much faster than MHA-Large's 0.37 ms despite MHA-Large being a smaller model). This pattern — GQA nearly matching MHA quality at nearly MQA speed — held consistently across the summarization, translation, and QA tasks evaluated.

> [!IMPORTANT]
> The paper notes an evaluation limitation: they did not train a comparable GQA/MQA XXL model completely from scratch to compare against the uptrained version, so the results demonstrate that uptraining is an effective and cheap conversion recipe relative to MHA and to from-scratch MQA baselines reported in prior work, but do not fully isolate the effect of uptraining versus training a grouped/multi-query model from scratch at that scale.
