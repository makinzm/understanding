# Meta Information

- URL: [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., & Salakhutdinov, R. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. ACL 2019.

# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

## Background and Motivation

Standard Transformer-based language models process text in fixed-length segments (e.g., 512 tokens) with no mechanism to share information across segment boundaries. This creates two problems:

1. **Context fragmentation**: The model cannot condition on tokens outside the current segment, making it unable to capture long-range dependencies that span segment boundaries.
2. **Context boundary problem**: When sentences are split at segment boundaries, the model sees semantically incomplete contexts at the start of each segment, degrading performance.

Recurrent neural networks (RNNs/LSTMs) handle sequential input but suffer from vanishing gradients and cannot parallelize computation. Transformer-XL resolves both issues by combining the parallelism of Transformers with a recurrence mechanism that allows information to flow across segments.

> [!NOTE]
> "Vanilla Transformer: evaluation is performed using a context of the same length as training, but the segment right boundary is shifted by one position at each step." This is computationally expensive (O(length) forward passes per step) compared to the cached evaluation in Transformer-XL.

## Architecture

Transformer-XL introduces two key technical contributions: **segment-level recurrence** and **relative positional encoding**. These work together to extend the effective context length beyond any single segment.

### Segment-Level Recurrence Mechanism

During training of segment $\tau+1$, the model **caches and reuses hidden states from segment $\tau$** as additional context. The stop-gradient operator prevents gradients from flowing across segment boundaries (keeping training tractable), while information still flows forward through the cached states.

**N-layer forward computation** for segment $\tau$, layer $n = 1, \ldots, N$:

Define the extended context by concatenating cached states from the previous segment with current hidden states:

$$\tilde{h}_{\tau+1}^{n-1} = \left[\text{SG}(m_\tau^{n-1}) \circ h_{\tau+1}^{n-1}\right]$$

where $\text{SG}(\cdot)$ is stop-gradient, $\circ$ is concatenation along the sequence axis, and $m_\tau^{n-1}$ is the cached memory from segment $\tau$ at layer $n-1$.

Compute queries, keys, and values:

$$q_{\tau+1}^n = h_{\tau+1}^{n-1} W_q^{n\top}, \quad k_{\tau+1}^n = \tilde{h}_{\tau+1}^{n-1} W_{k,E}^{n\top}, \quad v_{\tau+1}^n = \tilde{h}_{\tau+1}^{n-1} W_v^{n\top}$$

Note that queries come only from the current segment, but keys and values come from the **extended context** (both cached and current). This asymmetry is key: current tokens attend to all context but do not modify the cached states.

Compute attention, add residual, and apply FFN:

$$A_{\tau,i,j}^n = \text{(attention score between position } i \text{ in current and } j \text{ in extended context)}$$

$$a_\tau^n = \text{Masked-Softmax}(A_\tau^n / \sqrt{d}) \cdot v_\tau^n$$

$$o_\tau^n = \text{LayerNorm}(\text{Linear}(a_\tau^n) + h_\tau^{n-1})$$

$$h_\tau^n = \text{PositionwiseFeedForward}(o_\tau^n)$$

The base case is $h_\tau^0 := E_{s_\tau}$ (word embeddings). With $N$ layers and segment length $L$, the **maximum dependency length** is $O(N \times L)$ — linear in both depth and segment length.

> [!IMPORTANT]
> During evaluation, the model can reuse the entire cached history segment-by-segment, requiring only **one forward pass per token** instead of one per context position. This yields up to **1,800× speedup** over vanilla Transformers using sliding-window evaluation.

**Difference from vanilla Transformer**: Vanilla Transformers process each segment independently with no cross-segment information flow. Transformer-XL's recurrence connects segments through the layer-wise cache, extending context length to $O(N \times L)$ rather than $O(L)$.

**Difference from RNNs**: RNNs compress all history into a fixed-size hidden state vector, losing information about distant context. Transformer-XL caches full hidden-state sequences from previous segments, preserving detailed information without compression.

### Relative Positional Encoding

Reusing absolute positional embeddings across segments causes **temporal confusion** — the model cannot distinguish whether a position index refers to the current or the previous segment. Transformer-XL replaces absolute positional embeddings with **relative positional encodings** that encode the distance between positions.

**Standard absolute attention score** between query position $i$ and key position $j$:

$$A_{i,j}^{\text{abs}} = \underbrace{E_{x_i}^\top W_q^\top W_{k,E} E_{x_j}}_{(a)} + \underbrace{E_{x_i}^\top W_q^\top W_{k,R} U_j}_{(b)} + \underbrace{u^\top W_{k,E} E_{x_j}}_{(c)} + \underbrace{u^\top W_{k,R} U_j}_{(d)}$$

**Proposed relative attention score**:

$$A_{i,j}^{\text{rel}} = \underbrace{E_{x_i}^\top W_q^\top W_{k,E} E_{x_j}}_{(a)\text{ content-content}} + \underbrace{E_{x_i}^\top W_q^\top W_{k,R} R_{i-j}}_{(b)\text{ content-position}} + \underbrace{u^\top W_{k,E} E_{x_j}}_{(c)\text{ global content bias}} + \underbrace{v^\top W_{k,R} R_{i-j}}_{(d)\text{ global position bias}}$$

Key changes from absolute to relative:
- $U_j$ (absolute position embedding at key position) → $R_{i-j}$ (relative position encoding, a function of distance $i-j$)
- Query position embedding $U_i$ is replaced by two **trainable scalar vectors** $u, v \in \mathbb{R}^d$ (shared across all layers)
- $R_{i-j}$ uses a sinusoidal encoding (not learned), similar to the original Transformer but indexed by relative distance

> [!TIP]
> Shaw et al. (2018) proposed a simpler relative encoding with only two terms. Transformer-XL's four-term decomposition separates content-based and position-based biases more explicitly, yielding better performance (−0.64 perplexity on enwik8 vs. Shaw et al.).

### Efficient Linear-Time Relative Attention

Computing $W_{k,R} R_{i-j}$ for all pairs $(i,j)$ naïvely costs $O(L^2)$. The paper derives an $O(L)$ algorithm:

1. Precompute all needed relative embeddings:
   $$Q := \left[R_0^\top, R_1^\top, \ldots, R_{M+L-1}^\top\right] W_{k,R}^\top \in \mathbb{R}^{(M+L) \times d}$$
   where $M$ is the memory length and $L$ is the current segment length.

2. Compute $B = q Q^\top \in \mathbb{R}^{L \times (M+L)}$ — this handles term (b) for all query-key pairs simultaneously.

3. Extract the relevant sub-matrix by a **left-shift operation** to align relative distances correctly.

4. Term (d) uses the same trick with vector $v$ instead of query embeddings.

This reduces the complexity of relative attention from $O((M+L)^2 \cdot d)$ (naïve) to $O((M+L) \cdot d)$ (precomputed) per layer.

## Relative Effective Context Length (RECL)

Standard metrics like perplexity are influenced by model capacity. The paper introduces **RECL** to fairly measure how well a model exploits long-range context:

1. Train models with varying context lengths $c$.
2. Record the perplexity gain $\Delta_c$ from increasing context from 0 to $c$.
3. Normalize by the gain achievable with an oracle (very long context).
4. RECL is the context length at which the normalized gain plateaus.

Results: Transformer-XL achieves RECL of ~900 tokens, compared to ~110 for RNNs and ~200 for vanilla Transformers — approximately **80% longer than RNNs** and **450% longer than vanilla Transformers**.

## Experiments

- **Dataset** (word-level):
  - WikiText-103: 103M training tokens, 28K Wikipedia articles, average article length ~3,600 tokens — designed to test long-range dependencies
  - One Billion Word: 1B training tokens, sentence-shuffled (no long-term dependencies) — tests short-range modeling
  - Penn Treebank (PTB): ~1M tokens, small-scale standard benchmark
- **Dataset** (character-level):
  - enwik8: 100M bytes of raw Wikipedia text — rich in long-range structure
  - text8: 100M characters of lowercased, punctuation-stripped Wikipedia text
- **Hardware**: Not explicitly stated for all experiments; large models trained on TPUs
- **Optimizer**: Adam; learning rate scheduling with linear warmup and cosine decay
- **Results**:
  - WikiText-103: **18.3 perplexity** (previous SoTA: 20.5, a 10.7% relative improvement)
  - enwik8: **0.99 bits-per-character** (previous SoTA: 1.06)
  - text8: **1.08 bits-per-character** (previous SoTA: 1.13)
  - One Billion Word: **21.8 perplexity** (previous SoTA: 23.7)
  - Penn Treebank: **54.5 perplexity** (previous SoTA: 55.3, without finetune)

## Ablation Studies

| Variant | enwik8 bpc |
|---------|-----------|
| Full Transformer-XL | **0.99** |
| Without recurrence (vanilla Transformer with absolute pos.) | +2.7 |
| With Shaw et al. (2018) relative encoding (no recurrence) | +1.5 |
| With recurrence + Shaw et al. encoding | +0.64 |

Both the recurrence mechanism and the relative positional encoding contribute independently. On One Billion Word (where long-term dependencies don't exist), recurrence alone still improves by **+1.9 perplexity** by resolving context fragmentation at segment boundaries.

## Applicability

Transformer-XL is applicable when:
- **Long-range dependencies** are present in the data (e.g., document-level language modeling, code, long-form generation)
- **Efficient inference** is required: the cached state allows single-pass token generation at evaluation time
- **Fixed-length context** of standard Transformers is insufficient

It is less beneficial when:
- Training data has sentence-level shuffling (like One Billion Word), removing long-range structure
- Memory overhead of caching large hidden-state sequences is prohibitive (memory scales as $O(N \times M \times d)$)

> [!CAUTION]
> The paper caches hidden states from only the **immediately preceding segment** during training (single-step recurrence). Multiple previous segments can be cached during evaluation, but the training objective does not directly optimize for multi-step recurrence. The effective gain from multi-step caching at evaluation time may be limited compared to a model trained with multi-step recurrence.
