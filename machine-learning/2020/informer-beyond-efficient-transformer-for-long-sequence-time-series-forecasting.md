# Meta Information

- URL: [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI 2021.

---

# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

## Problem Statement

Long Sequence Time-Series Forecasting (LSTF) requires predicting far into the future (e.g., 48 hours ahead for electricity consumption or 2 weeks ahead for wind speed). Vanilla Transformers have three critical bottlenecks that prevent their direct application at these scales:

1. **Quadratic complexity**: Standard self-attention requires $O(L^2)$ time and memory in sequence length $L$, making long input/output sequences infeasible.
2. **Memory explosion with deep stacking**: Multiple encoder/decoder layers rapidly exhaust GPU memory.
3. **Step-by-step decoding**: Autoregressive decoding requires $L$ sequential forward passes, making inference $O(L)$ in the prediction horizon.

The Informer addresses all three bottlenecks simultaneously while maintaining competitive forecasting accuracy.

## Input / Output Specification

- **Input**: $\mathcal{X}^t = \{x_1^t, x_2^t, \ldots, x_{L_x}^t \mid x_i^t \in \mathbb{R}^{d_{\text{model}}}\}$ — a context window of $L_x$ timesteps with $d_{\text{model}}$-dimensional feature vectors.
- **Output**: $\mathcal{Y}^t = \{y_1^t, y_2^t, \ldots, y_{L_y}^t \mid y_i^t \in \mathbb{R}^{d_y}\}$ — predicted future values for $L_y$ timesteps ($d_y$ target dimensions).

The encoder-decoder architecture processes inputs from the lookback window and generates predictions in a single forward pass (no autoregression during inference).

---

## Core Components

### 1. Input Representation

Each scalar input is lifted into a $d_{\text{model}}$-dimensional space through three additive embedding components:

$$\mathbf{X}_{\text{feed}}^t[i] = \alpha \cdot \mathbf{u}_i^t + \text{PE}(L_x \cdot (t-1) + i) + \sum_p \text{SE}_p(L_x \cdot (t-1) + i)$$

| Component | Description |
|---|---|
| $\alpha \cdot \mathbf{u}_i^t$ | Scalar projection of raw value to $\mathbb{R}^{d_{\text{model}}}$, scaled by learnable $\alpha$ |
| $\text{PE}(\cdot)$ | Fixed sinusoidal positional encoding capturing local sequence position |
| $\text{SE}_p(\cdot)$ | Learnable global time-stamp embedding for each time attribute $p$ (minute, hour, day of week, month, holiday) |

The local positional embedding captures relative order within a window; the global stamp embedding injects seasonality and periodicity information.

---

### 2. ProbSparse Self-Attention

**Motivation**: In canonical self-attention, the score matrix $\mathbf{A} = \text{Softmax}(QK^\top / \sqrt{d})$ follows a long-tail distribution—a small fraction of query–key pairs produce dominant attention weights, while the rest approximate uniform distributions. Computing all $L_Q \times L_K$ pairs wastes computation.

**Query Sparsity Measurement**: For query $q_i \in \mathbb{R}^d$ and key set $K \in \mathbb{R}^{L_K \times d}$, define:

$$M(q_i, K) = \max_j \left\{ \frac{q_i k_j^\top}{\sqrt{d}} \right\} - \frac{1}{L_K} \sum_j \frac{q_i k_j^\top}{\sqrt{d}}$$

This measures the gap between the maximum dot-product (the dominant pair) and the mean dot-product (a proxy for the uniform baseline). A large $M$ signals that $q_i$ has a "peaky" attention distribution with genuine dominant dependencies; a small $M$ suggests near-uniform (uninformative) attention.

**Algorithm (ProbSparse Attention)**:

```
Input: Q ∈ R^{L_Q × d}, K ∈ R^{L_K × d}, V ∈ R^{L_K × d}
       u = c · ln(L_Q)  # number of selected top queries

1. Sample Ū = u randomly sampled keys from K  # O(ln L_K) keys per query
2. For each query q_i, compute M̄(q_i, K̄) using sampled keys
3. Select Top-u queries by M̄ → Q̄ ∈ R^{u × d}
4. Compute sparse attention: Ā = Softmax(Q̄K^⊤ / √d) · V  # only u × L_K operations
5. Fill remaining (L_Q - u) queries with mean(V) as default output
Output: Attention(Q, K, V) ∈ R^{L_Q × d}
```

**Complexity**: $O(L_K \ln L_Q)$ rather than $O(L_Q L_K)$; for $L_Q = L_K = L$, this reduces to $O(L \ln L)$.

> [!NOTE]
> The authors prove (Lemma 1) that $M(q_i, K) \geq 0$ always holds and that the max-mean measurement is upper-bounded by $\max_j \{q_i k_j^\top / \sqrt{d}\} + \ln L_K$, ensuring numerical stability without computing log-sum-exp over all keys.

---

### 3. Self-Attention Distilling (Encoder)

The encoder applies a "distilling" operation between successive attention layers to progressively reduce the sequence length:

$$\mathbf{X}_{j+1}^t = \text{MaxPool}\!\left(\text{ELU}\!\left(\text{Conv1d}\!\left([\mathbf{X}_j^t]_{\text{AB}}\right)\right)\right)$$

Where:
- $[\cdot]_{\text{AB}}$ denotes the output of the multi-head attention block at layer $j$.
- Conv1d applies a 1-D convolution with kernel size 3 along the time axis.
- ELU is the Exponential Linear Unit activation.
- MaxPool uses stride 2, halving the sequence length.

**Effect**: After each distilling layer, the sequence length is halved: $L_x \to \lfloor L_x / 2 \rfloor \to \cdots$. A stack of $J$ encoder layers reduces the input to $\lfloor L_x / 2^J \rfloor$ timesteps.

**Robustness**: To protect against distilling failures, the authors train multiple encoder "replicas" with half-length inputs (i.e., one replica starts from layer 2, another from layer 3, etc.) and concatenate their outputs.

**Memory complexity**: Distilling reduces total memory from $O(J \cdot L^2)$ to $O((2-\epsilon)L \log L)$.

---

### 4. Generative-Style Decoder

The decoder avoids autoregressive (step-by-step) prediction by accepting a compound input:

$$\mathbf{X}_{\text{de}}^t = \text{Concat}(\mathbf{X}_{\text{token}}^t, \mathbf{X}_0^t)$$

Where:
- $\mathbf{X}_{\text{token}}^t \in \mathbb{R}^{L_{\text{token}} \times d_{\text{model}}}$ — a "start token" drawn from a recent history segment (the last $L_{\text{token}}$ observed timesteps before the prediction window, analogous to BERT's [CLS] token).
- $\mathbf{X}_0^t \in \mathbb{R}^{L_y \times d_{\text{model}}}$ — zero-padded placeholder slots for the target sequence.

The decoder processes this combined input through two transformer layers:
1. Masked multi-head ProbSparse attention (causal masking over $\mathbf{X}_{\text{token}} \| \mathbf{X}_0$)
2. Cross-attention with the encoded memory

A final linear projection maps $d_{\text{model}} \to d_y$ to produce scalar predictions.

> [!IMPORTANT]
> By filling the target slots with zeros and passing them all at once, the entire $L_y$-step prediction is generated in **one forward pass**, reducing inference from $O(L_y)$ to $O(1)$ sequential steps.

---

## Architecture Overview

```
Input (L_x timesteps)
    │
    ▼
Input Embedding (scalar projection + position + timestamp)
    │
    ▼
Encoder Layer 1 ─── ProbSparse MHA ─── Distilling (halve length)
    │
    ▼
Encoder Layer 2 ─── ProbSparse MHA ─── Distilling (halve length)
    │
    ▼
Encoder Layer 3 ─── ProbSparse MHA ─────────────────────────────────────────
    │                                                                       │
    └─────────────── Encoder Memory (concatenated multi-scale outputs) ────┘
                                            │
                                            ▼
Decoder Input: [X_token (L_token steps) ‖ X_0 (L_y zeros)]
    │
    ▼
Decoder Layer 1 ─── Masked ProbSparse MHA ─── Cross-Attention with encoder
    │
    ▼
Decoder Layer 2 ─── Masked ProbSparse MHA ─── Cross-Attention with encoder
    │
    ▼
Linear Projection ─── Output: Ŷ ∈ R^{L_y × d_y}
```

**Hyperparameters (reported in paper)**:
- Embedding dimension: $d_{\text{model}} = 512$
- Encoder attention heads: 8; Decoder attention heads: 8
- Feed-forward layer dimension: 2048
- Encoder layers: 3 (main stack) + replicas at half/quarter input lengths
- Decoder layers: 2
- Optimizer: Adam with linear warmup
- Batch size: 32; maximum training epochs: 8 with early stopping (patience = 3)

---

## Comparison with Similar Methods

| Method | Attention Complexity | Decoder | Memory |
|---|---|---|---|
| Transformer (vanilla) | $O(L^2)$ | Autoregressive ($O(L)$ steps) | $O(L^2)$ |
| LogSparse Transformer | $O(L \log L)$ | Autoregressive ($O(L)$ steps) | $O(L \log^2 L)$ |
| Reformer (LSH) | $O(L \log L)$ | Autoregressive ($O(L)$ steps) | $O(L \log L)$ |
| **Informer** | $O(L \log L)$ | **Generative ($O(1)$ steps)** | $O(L \log L)$ |

> [!NOTE]
> Reformer and LogSparse Transformer achieve sub-quadratic attention but still decode autoregressively, making inference slow for long predictions. Informer's generative decoder is its key differentiator at test time.

**Key differences from standard Transformer**:
- ProbSparse attention selects a subset of queries, not all pairs — unlike LogSparse which uses fixed sparse patterns.
- Distilling creates a hierarchy of scales within a single encoder, similar in spirit to U-Net but applied temporally.
- The generative decoder is analogous to sequence-to-sequence models with a "first-token" prompt, but with full target length provided upfront.

---

# Experiments

- **Datasets**:
  - **ETTh1, ETTh2** (Electricity Transformer Temperature): 2 years of hourly transformer oil temperature + 6 power load features; ~17,420 training samples each.
  - **ETTm1**: Same stations at 15-minute granularity; ~69,680 training samples.
  - **ECL** (Electricity Consuming Load): 321 clients × 2 years hourly electricity consumption.
  - **Weather**: 1,600 U.S. stations × 4 years of hourly weather readings.

- **Evaluation**: Mean Squared Error (MSE) and Mean Absolute Error (MAE); prediction horizons: 24, 48, 168, 336, 720 timesteps.

- **Hardware**: Not explicitly stated; all experiments fit within single-GPU memory due to distilling.

- **Optimizer**: Adam with early stopping (patience = 3), max 8 epochs.

- **Results**:
  - Informer achieves 61% lower MSE than LSTM on ETTh1 at 720-step prediction (MSE 0.269 vs. 0.683).
  - Outperforms LogSparse Transformer and Reformer at most long-horizon settings.
  - Univariate forecasting shows larger gains than multivariate (attributed to heterogeneity in feature predictability).

- **Ablation findings**:
  - Removing ProbSparse (using canonical attention): marginal accuracy gain but $O(L^2)$ memory — infeasible at $L > 720$.
  - Removing distilling: out-of-memory at $L > 720$ timesteps.
  - Removing generative decoder (reverting to dynamic decoding): performance degrades significantly at long prediction horizons due to error accumulation.
