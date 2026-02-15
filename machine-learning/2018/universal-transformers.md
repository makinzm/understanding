# Meta Information

- URL: [[1807.03819] Universal Transformers](https://arxiv.org/abs/1807.03819)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2019). Universal Transformers. *International Conference on Learning Representations (ICLR 2019)*.

---

# Introduction

The Universal Transformer (UT) addresses a fundamental limitation of the standard Transformer (Vaswani et al., 2017): its fixed, non-recurrent depth means each token receives the same number of computation steps regardless of the complexity of the task at that position. UTs introduce **recurrence over the depth dimension** — the same transition function is applied repeatedly across $T$ timesteps, rather than using $T$ different feed-forward layers with independent weights.

> [!NOTE]
> "UTs combine the parallelizability and global receptive field of feed-forward sequence models like the Transformer with the recurrent inductive bias of RNNs."

**Who should use this**: Researchers and practitioners working on tasks that require reasoning, length generalization, or variable-depth computation — including algorithmic tasks, question answering, language modeling, and machine translation.

> [!TIP]
> The standard Transformer it builds upon: [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)

---

# Architecture

## Comparison with Standard Transformer

| Property | Transformer | Universal Transformer |
|----------|-------------|-----------------------|
| Depth | Fixed $L$ distinct layers | $T$ recurrent steps with shared weights |
| Inductive bias | Position-wise only | Recurrent (depth-wise) + position-wise |
| Computation per token | Fixed | Dynamic (with ACT) |
| Turing-complete | No | Yes (with sufficient steps) |
| Parameters | $O(L \cdot d^2)$ | $O(d^2)$ (shared across steps) |

## Encoder

**Input**: A sequence of $n$ tokens, embedded into $H^0 \in \mathbb{R}^{n \times d}$.

At each recurrent step $t \in \{1, \ldots, T\}$, the encoder updates its hidden state $H^t \in \mathbb{R}^{n \times d}$ as follows:

**Step 1 — Add position and timestep embeddings**:

$$\tilde{H}^{t-1} = H^{t-1} + P^t$$

where $P^t \in \mathbb{R}^{n \times d}$ encodes both token position $i$ and recurrent timestep $t$ (see Position Encoding section).

**Step 2 — Multi-head self-attention with residual and LayerNorm**:

$$A^t = \text{LayerNorm}\!\left(\tilde{H}^{t-1} + \text{MultiHeadSelfAttention}(\tilde{H}^{t-1})\right)$$

**Step 3 — Transition function with residual and LayerNorm**:

$$H^t = \text{LayerNorm}\!\left(A^t + \text{Transition}(A^t)\right)$$

where $\text{Transition}(\cdot)$ is either a position-wise feed-forward network or a depth-wise separable convolution.

**Output**: $H^T \in \mathbb{R}^{n \times d}$ after $T$ steps.

## Decoder

The decoder follows the same recurrent structure but adds a cross-attention sublayer between the self-attention and transition steps:

$$A^t_{\text{dec}} = \text{LayerNorm}\!\left(\tilde{H}^{t-1}_{\text{dec}} + \text{MultiHeadSelfAttention}(\tilde{H}^{t-1}_{\text{dec}})\right)$$

$$C^t = \text{LayerNorm}\!\left(A^t_{\text{dec}} + \text{MultiHeadAttention}(A^t_{\text{dec}},\ H^T_{\text{enc}},\ H^T_{\text{enc}})\right)$$

$$H^t_{\text{dec}} = \text{LayerNorm}\!\left(C^t + \text{Transition}(C^t)\right)$$

Decoder self-attention is causally masked to prevent attending to future positions.

## Multi-Head Self-Attention

Scaled dot-product attention for a single head given queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{n \times d_k}$, values $V \in \mathbb{R}^{n \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \in \mathbb{R}^{n \times d_v}$$

Multiple heads ($k$ total) are projected and concatenated:

$$\text{head}_i = \text{Attention}(HW_i^Q,\ HW_i^K,\ HW_i^V)$$

$$\text{MultiHeadSelfAttention}(H) = \text{Concat}(\text{head}_1, \ldots, \text{head}_k)\,W^O \in \mathbb{R}^{n \times d}$$

where $W_i^Q, W_i^K \in \mathbb{R}^{d \times d_k}$, $W_i^V \in \mathbb{R}^{d \times d_v}$, $W^O \in \mathbb{R}^{kd_v \times d}$.

## Position and Timestep Encoding

Both token position $i \in \{1, \ldots, n\}$ and recurrent timestep $t \in \{1, \ldots, T\}$ are encoded using sinusoidal embeddings that are **summed** together:

$$P^t_{i,2j} = \sin\!\left(\frac{i}{10000^{2j/d}}\right) + \sin\!\left(\frac{t}{10000^{2j/d}}\right)$$

$$P^t_{i,2j+1} = \cos\!\left(\frac{i}{10000^{2j/d}}\right) + \cos\!\left(\frac{t}{10000^{2j/d}}\right)$$

This 2D encoding lets the model distinguish both *which position* in the sequence and *which recurrent step* it is currently processing.

> [!IMPORTANT]
> The addition of a **timestep** dimension to position encoding is a key design choice distinguishing UTs from standard Transformers, which only encode position. Without it, the recurrent steps would be indistinguishable from each other.

---

# Adaptive Computation Time (ACT)

## Motivation

Different tokens may require different amounts of processing. A common word in a simple sentence needs fewer refinement steps than an ambiguous word in a complex inference chain. The UT can incorporate **Adaptive Computation Time (ACT)** (Graves, 2016) to allow each position to halt early when it has converged.

## Algorithm

For each position $i$ independently:

1. At each step $t$, compute a halting probability: $p_i^t = \sigma(W_h \cdot h_i^t + b_h) \in [0, 1]$
2. Accumulate halting scores: $s_i^t = \sum_{\tau=1}^{t} p_i^\tau$
3. Halt position $i$ at the first step $T_i$ where $s_i^{T_i} \geq 1 - \epsilon$ (typically $\epsilon = 0.01$)
4. Set the *remainder* $r_i = 1 - s_i^{T_i - 1}$ to ensure probabilities sum to 1
5. Output a weighted combination:

$$\hat{h}_i = \sum_{t=1}^{T_i} p_i^t \cdot h_i^t \quad \text{(with } p_i^{T_i} \text{ replaced by } r_i\text{)}$$

A **ponder cost penalty** $\tau \cdot \sum_i (T_i + R_i)$ is added to the loss, where $\tau$ is a hyperparameter controlling the efficiency-accuracy tradeoff.

> [!NOTE]
> ACT introduces a soft, differentiable halting mechanism. Positions that halt early do not receive further gradient updates from self-attention with other positions, effectively acting as a learned early-exit mechanism.

---

# Theoretical Properties

## Turing Completeness

A standard Transformer is not Turing-complete because it applies a fixed number of operations. A Universal Transformer, when paired with ACT and unbounded recurrent steps, **can simulate any Turing machine** in the following sense:

- By setting $T = n$ (input length) and using convolutional transitions, a UT reduces to a **Neural GPU** — a known Turing-complete model.
- More generally, UTs can represent any function computable by an RNN, since they apply the same transition function recurrently.

This theoretical property motivates UT's better generalization on algorithmic tasks compared to standard Transformers.

---

# Experiments

## Datasets

| Dataset | Task | Notes |
|---------|------|-------|
| bAbI | 20 synthetic reasoning tasks | 1K and 10K training variants |
| Subject-Verb Agreement | Syntactic agreement prediction | Linzen et al. (2016) |
| LAMBADA | Discourse-level language modeling | ~10K passages, ~80K train sentences |
| Algorithmic Tasks | Copy, Reverse, Addition on decimal strings | Generalization: train length 40, test length 400 |
| Learning to Execute (LTE) | Program evaluation (memorization + program tasks) | Zaremba & Sutskever (2014) |
| WMT 2014 En-De | Machine translation | 4.5M sentence pairs |

## Key Results

### bAbI (Reasoning)

UT with dynamic halting (ACT) achieved **0.21% mean error** on the 10K-example setting, compared to 15.2% for the standard Transformer. The model solved all 20 tasks. The average ponder time correlated with task difficulty: tasks requiring 1 supporting fact averaged 2.3±0.8 steps, while 3-fact tasks averaged 3.8±2.2 steps — showing ACT learns to allocate more computation where needed.

### Algorithmic Tasks (Length Generalization)

Trained on length 40, tested on length 400:

| Model | Copy (acc) | Reverse (acc) | Addition (acc) |
|-------|-----------|---------------|----------------|
| LSTM | 45% | 66% | 8% |
| Transformer | 53% | 13% | 7% |
| Universal Transformer | **91%** | **96%** | **34%** |

UTs generalize far better to unseen lengths — a direct consequence of the recurrent inductive bias that makes depth-wise computation sequence-length independent.

### Subject-Verb Agreement

UT with ACT reached **99.2% accuracy** on predicting verb number. With 5+ intervening attractors (the hardest setting), UT scored 90.7% vs. the previous best of 84.2%.

### LAMBADA (Language Modeling)

| Model | Perplexity (test) | Reading Comprehension |
|-------|-------------------|-----------------------|
| Transformer | 7321 | — |
| Universal Transformer | **142** | 56.25% |
| Previous best | — | 55.69% |

UT's average ponder time was 8.2±2.1 steps; a fixed 9-step model without ACT failed to match this, suggesting that the **irregularity** of computation introduced by ACT acts as a regularizer.

### Machine Translation (WMT14 En-De)

| Model | BLEU |
|-------|------|
| Transformer (base) | 28.0 |
| Weighted Transformer (base) | 28.4 |
| **Universal Transformer (base)** | **28.9** |

UT outperforms the standard Transformer by +0.9 BLEU at the base model scale, demonstrating that recurrent inductive bias also benefits large-scale sequence-to-sequence tasks.

---

# Implementation Details

- **Optimizer**: Adam with learning rate warmup (identical schedule to Vaswani et al., 2017)
- **Regularization**: Dropout, label smoothing, layer normalization (applied before each sublayer, as in pre-norm variant)
- **Transition function**: Separable depth-wise convolution for some tasks; position-wise FFN for others
- **ACT threshold**: $\epsilon = 0.01$; ponder cost $\tau$ tuned per task
- **Weight sharing**: All recurrent steps share the same parameters — the encoder has one set of attention and transition weights, and the decoder has one set for each of its three sublayers

---

# Differences from Similar Models

| Model | Recurrence | Parallelism | Dynamic Depth | Turing-complete |
|-------|-----------|-------------|---------------|-----------------|
| LSTM / GRU | Over time steps | No (sequential) | No | Yes |
| Transformer | No | Yes | No | No |
| Universal Transformer | Over depth | Yes (within step) | Yes (with ACT) | Yes |
| Neural GPU | Over time steps | Yes | No | Yes |
| Neural Turing Machine (NTM) | Over time steps | No | No | Yes |

The UT uniquely combines (1) full parallelism across sequence positions within each step, (2) recurrence across depth, and (3) optional dynamic halting — giving it properties of both RNNs and standard Transformers while being strictly more expressive.

---

# Limitations

- **Training cost**: Recurrence across depth means $T$ forward passes per training step rather than a single pass, increasing wall-clock time.
- **Hyperparameter sensitivity**: The ACT ponder cost $\tau$ requires per-task tuning.
- **Translation gains are modest**: On WMT14, the improvement (+0.9 BLEU) is meaningful but not dramatic, suggesting the recurrent bias is less critical when training data is abundant.
