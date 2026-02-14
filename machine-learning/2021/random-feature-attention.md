# Meta Information

- URL: [Random Feature Attention](https://arxiv.org/abs/2103.02143)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Peng, H., Pappas, N., Yogatama, D., Schwartz, R., Smith, N. A., & Kong, L. (2021). Random Feature Attention. ICLR 2021.

# Introduction

Standard softmax attention has quadratic time and space complexity $O(N^2)$ in sequence length $N$, making it impractical for long sequences. Random Feature Attention (RFA) reduces this to linear complexity $O(N)$ by approximating the softmax kernel using random Fourier features. This makes RFA a drop-in replacement for standard attention, requiring no architectural changes beyond swapping the attention kernel.

RFA is applicable to any practitioner who needs to process long sequences efficiently, including tasks like language modeling, machine translation, and long text classification. The key insight is that the softmax function can be decomposed as an inner product of feature maps $\phi(\mathbf{q}) \cdot \phi(\mathbf{k})$, enabling efficient aggregation of key-value pairs without materializing the full attention matrix.

# Background: Random Feature Methods

RFA builds on the classical random Fourier features framework (Rahimi & Recht, 2007), which approximates shift-invariant kernels $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{x} - \mathbf{y})$ using Monte Carlo sampling of random projections.

For the Gaussian kernel $k(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x} \cdot \mathbf{y} / \sigma^2)$, the feature map is:

$$\phi(\mathbf{x}) = \frac{1}{\sqrt{D}} \left[ \sin(\mathbf{w}_1 \cdot \mathbf{x}), \ldots, \sin(\mathbf{w}_D \cdot \mathbf{x}), \cos(\mathbf{w}_1 \cdot \mathbf{x}), \ldots, \cos(\mathbf{w}_D \cdot \mathbf{x}) \right]^\top$$

where $\mathbf{w}_i \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_d)$ are sampled i.i.d. and $D$ is the number of random features. The approximation holds as $D \to \infty$:

$$\exp(\mathbf{x} \cdot \mathbf{y} / \sigma^2) \approx \exp\!\left(\frac{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2}{2\sigma^2}\right) \cdot \phi(\mathbf{x}) \cdot \phi(\mathbf{y})$$

> [!NOTE]
> The paper's implementation learns the scale $\sigma^2$ per head via element-wise scaling rather than using the isotropic $\sigma^2 \mathbf{I}$ prescribed by theory. This is a practical departure that works better empirically.

# Random Feature Attention (RFA)

## Input and Output

- **Input**: Query $\mathbf{q}_t \in \mathbb{R}^d$, Keys $\{\mathbf{k}_i\}_{i=1}^N \subset \mathbb{R}^d$, Values $\{\mathbf{v}_i\}_{i=1}^N \subset \mathbb{R}^d$
- **Feature map output**: $\phi(\mathbf{x}) \in \mathbb{R}^{2D}$ (sine and cosine components concatenated)
- **Attention output**: $\mathbf{h}_t \in \mathbb{R}^d$

## Non-Causal (Cross) Attention

For encoder self-attention or cross-attention, the full context is available. The standard softmax attention:

$$\text{Attention}(\mathbf{q}_t, K, V) = \frac{\sum_i \exp(\mathbf{q}_t \cdot \mathbf{k}_i) \mathbf{v}_i}{\sum_j \exp(\mathbf{q}_t \cdot \mathbf{k}_j)}$$

is replaced by RFA, factoring out the exponential norms (which cancel in the ratio):

$$\text{RFA}(\mathbf{q}_t, K, V) = \frac{\phi(\mathbf{q}_t)^\top \left(\sum_i \phi(\mathbf{k}_i) \otimes \mathbf{v}_i\right)}{\phi(\mathbf{q}_t) \cdot \sum_j \phi(\mathbf{k}_j)}$$

where $\otimes$ is the outer product and $\phi(\mathbf{k}_i) \otimes \mathbf{v}_i \in \mathbb{R}^{2D \times d}$.

The key efficiency gain: $\mathbf{S} = \sum_i \phi(\mathbf{k}_i) \otimes \mathbf{v}_i \in \mathbb{R}^{2D \times d}$ and $\mathbf{z} = \sum_j \phi(\mathbf{k}_j) \in \mathbb{R}^{2D}$ are computed once and reused for all queries, reducing complexity from $O(N^2 d)$ to $O(N D d)$.

## Causal Attention (Recurrent Form)

For decoder self-attention, where position $t$ attends only to positions $\leq t$, the sums are accumulated incrementally. This gives a recurrent computation with hidden states $\mathbf{S}_t \in \mathbb{R}^{2D \times d}$ and $\mathbf{z}_t \in \mathbb{R}^{2D}$:

**Algorithm: Causal RFA**

```
Initialize: S ← 0 ∈ ℝ^{2D×d},  z ← 0 ∈ ℝ^{2D}
For t = 1, 2, ..., N:
    S ← S + φ(k_t) ⊗ v_t        # outer product accumulation
    z ← z + φ(k_t)               # normalization term accumulation
    h_t ← φ(q_t)ᵀ S / (φ(q_t) · z)   # attention output, shape: d
Return {h_1, ..., h_N}
```

This replaces the $O(N^2)$ attention matrix with $O(N)$ recurrent updates, enabling autoregressive decoding in $O(1)$ time per step (the hidden states $\mathbf{S}_t$ and $\mathbf{z}_t$ summarize all past context).

## Gated Variant (RFA-Gate)

RFA-Gate introduces a learned scalar gate $g_t \in (0, 1)$ to model recency bias—the idea that recent tokens are often more relevant than distant ones in language:

$$g_t = \sigma(\mathbf{w}_g \cdot \mathbf{x}_t + b_g)$$

$$\mathbf{S}_t = g_t \mathbf{S}_{t-1} + (1 - g_t) \phi(\mathbf{k}_t) \otimes \mathbf{v}_t$$

$$\mathbf{z}_t = g_t \mathbf{z}_{t-1} + (1 - g_t) \phi(\mathbf{k}_t)$$

When $g_t \approx 1$, history is preserved; when $g_t \approx 0$, the model resets to focus on the current token. This mirrors an exponential moving average over the key-value context, without increasing time complexity.

> [!IMPORTANT]
> The gating mechanism is critical for language modeling performance. Ablation shows RFA-Gate consistently outperforms ungated RFA on WikiText-103, gaining 1.2–1.5 perplexity points on large models. Without the gate, RFA underperforms vanilla softmax attention on perplexity.

# Comparison with Related Methods

| Method | Kernel / Approximation | Complexity | Notes |
|---|---|---|---|
| Softmax Transformer | $\exp(\mathbf{q} \cdot \mathbf{k})$ | $O(N^2)$ | Exact, expressive |
| **RFA (Gaussian)** | Random Fourier (Gaussian) | $O(N)$ | Proposed; best performer |
| φ_ELU (Katharopoulos et al. 2020) | $\text{ELU}(\cdot) + 1$ | $O(N)$ | Concurrent; inferior in practice |
| Performer (Choromanski et al. 2020) | Positive random features | $O(N)$ | Concurrent; different kernel |
| Linformer | Low-rank projection | $O(N)$ | Non-causal only |
| Longformer | Sparse local + global | $O(N)$ | Window-based, not general |

> [!NOTE]
> RFA substantially outperforms φ_ELU on WikiText-103 (lower perplexity) and machine translation (higher BLEU), demonstrating that the choice of feature map—not just the linearization idea—is critical.

> [!TIP]
> Performer uses "positive" random features to avoid negative values in the approximation, trading approximation accuracy for numerical stability. RFA uses the full sine+cosine (complex exponential) feature map which is theoretically cleaner.

# Complexity Analysis

| Setting | Time (Training) | Time (Decoding per step) | Space |
|---|---|---|---|
| Softmax Transformer | $O(N^2 d)$ | $O(MN)$ | $O(MN)$ |
| RFA | $O(N D d)$ | $O(M + N)$ | $O((M + N) D d)$ |

Here $M$ = source length, $N$ = target length, $D$ = number of random features. Since $D \ll N$ typically (the paper uses $D = d/2$ or $D = d$), RFA achieves dramatic savings for long sequences.

At sequence length 2,048: RFA achieves **12× decoding speedup** over softmax attention.

# Experiments

- **Datasets**:
  - Language Modeling: WikiText-103 (103M tokens, word-level)
  - Machine Translation: WMT14 EN→DE (~4.5M sentence pairs), WMT14 EN→FR (~36M sentence pairs), IWSLT14 DE→EN (~160K sentence pairs)
  - Long Text Classification: Long Range Arena benchmark — ListOps, IMDb reviews, AAN (ACL Anthology Network) citation prediction
- **Hardware**: Not explicitly specified
- **Optimizer**: Adam (following baseline configurations)
- **Key Results**:
  - WikiText-103: RFA-Gate-Gaussian matches or surpasses softmax transformer baseline (up to 1.2–1.5 perplexity point improvement on large models with cross-segment state)
  - WMT14 EN→DE/EN→FR: RFA achieves comparable BLEU to softmax baseline with **1.8–1.9× decoding speedup**
  - Long Range Arena: RFA averages **0.3% accuracy improvement** over vanilla transformer with **1.1–5.3× speedup** depending on sequence length
  - Feature map size: $D = d$ or $D = d/2$ is sufficient; Gaussian kernel outperforms arc-cosine kernel

> [!NOTE]
> No training speedup was observed for moderate sequence lengths (512 tokens), because the $O(N^2)$ attention matrix computation is GPU-optimized and the $2D \times d$ outer product in RFA is memory-bandwidth bound at small $N$. The benefit emerges at inference time with longer sequences.
