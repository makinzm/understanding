# Meta Information

- URL: [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Wang, S., Li, B. Z., Khabsa, M., Fang, H., & Ma, H. (2020). Linformer: Self-Attention with Linear Complexity. arXiv:2006.04768.

# Linformer: Self-Attention with Linear Complexity

## Background and Motivation

Standard Transformer self-attention has $O(n^2)$ time and space complexity with respect to sequence length $n$, making it prohibitively expensive for long sequences. The key insight of Linformer is that the self-attention matrix $P = \text{softmax}(QK^\top / \sqrt{d})$ is empirically low-rank: a spectrum analysis of pretrained RoBERTa-base models reveals a clear long-tail distribution of singular values, with higher layers showing more skewed distributions where information concentrates in a small number of dominant singular values.

> [!NOTE]
> From the paper: "We observe that there is a clear long-tail spectrum distribution, indicating that most of the information of the self-attention matrix can be recovered from the first few largest singular vectors."

## Theoretical Foundation

### Theorem 1: Self-Attention Is Low-Rank

For any query matrix $Q \in \mathbb{R}^{n \times d_k}$, key matrix $K \in \mathbb{R}^{n \times d_k}$, and value matrix $V \in \mathbb{R}^{n \times d_v}$, there exists a low-rank matrix $\hat{P}$ of rank $\Theta(\log n)$ that approximates the attention output $P \cdot V$ with high probability, where $P = \text{softmax}(QK^\top / \sqrt{d_k})$.

The proof leverages the **Johnson-Lindenstrauss (JL) lemma**: for any $\epsilon > 0$, $\delta > 0$, and $m$ points in $\mathbb{R}^d$, a random projection to $k = O(\epsilon^{-2} \log(m/\delta))$ dimensions preserves pairwise distances up to factor $(1 \pm \epsilon)$ with probability at least $1 - \delta$.

### Theorem 2: Linear Self-Attention

By introducing projection matrices $E_i, F_i \in \mathbb{R}^{n \times k}$ for each head $i$, the standard multi-head attention can be approximated using:

$$\bar{P}_i = \text{softmax}\left(\frac{Q_i (E_i K_i)^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times k}$$

$$\text{head}_i = \bar{P}_i \cdot (F_i V_i W_i^V) \in \mathbb{R}^{n \times d_v}$$

When the projected dimension satisfies $k = O(d / \epsilon^2)$, this achieves $\epsilon$-error approximation **independent of sequence length $n$**, reducing complexity from $O(n^2)$ to $O(nk)$.

## Linformer Architecture

### Input/Output

- **Input**: Token sequence $X \in \mathbb{R}^{n \times d_{\text{model}}}$, where $n$ is sequence length and $d_{\text{model}}$ is embedding dimension.
- **Output**: Context-enriched representation $\mathbb{R}^{n \times d_{\text{model}}}$, same shape as input.

### Algorithm: Linear Multi-Head Self-Attention

```
Input: X ∈ R^(n × d_model), projection matrices E_i, F_i ∈ R^(n × k)
Hyperparameters: num_heads h, head_dim d_k = d_model / h, projected_dim k

For each head i = 1, ..., h:
  1. Project to queries, keys, values:
       Q_i = X · W_i^Q  ∈ R^(n × d_k)
       K_i = X · W_i^K  ∈ R^(n × d_k)
       V_i = X · W_i^V  ∈ R^(n × d_v)

  2. Apply linear projections to keys and values:
       K̄_i = E_i · K_i  ∈ R^(k × d_k)   [n→k projection]
       V̄_i = F_i · V_i  ∈ R^(k × d_v)   [n→k projection]

  3. Compute low-rank attention:
       Ā_i = Q_i · K̄_i^T / sqrt(d_k)  ∈ R^(n × k)
       P̄_i = softmax(Ā_i)             ∈ R^(n × k)

  4. Weighted sum:
       head_i = P̄_i · V̄_i            ∈ R^(n × d_v)

Output: Concat(head_1, ..., head_h) · W^O  ∈ R^(n × d_model)
```

**Space complexity**: The attention matrix $\bar{P}_i \in \mathbb{R}^{n \times k}$ instead of $\mathbb{R}^{n \times n}$, saving $O(n^2 - nk)$ memory.

## Parameter Sharing Strategies

Three sharing strategies reduce the number of extra projection parameters:

| Strategy | Description | # Projection Matrices |
|---|---|---|
| Headwise sharing | One $E$, $F$ per layer, shared across all heads | $2L$ |
| Key-value sharing | Single projection $E = F$ per head | $hL$ |
| Layerwise sharing | Single $E$, $F$ for entire model | $2$ |

The authors found that **layerwise sharing** (just 2 matrices for the full 12-layer, 12-head model) achieved the best downstream performance, despite using the fewest parameters.

## Alternative Projection Methods

Beyond learned linear projections $E, F$, the authors test:

- **Mean pooling**: Average consecutive tokens to reduce length $n \to k$
- **Max pooling**: Take maximum over a window of size $n/k$
- **Convolution**: Depthwise conv with kernel size $n/k$

Among these, **mean pooling** achieves comparable performance to learned projections with no additional parameters.

## Complexity Comparison

| Architecture | Time Complexity | Space Complexity | Notes |
|---|---|---|---|
| Transformer | $O(n^2 d)$ | $O(n^2)$ | Baseline |
| Sparse Transformer | $O(n \sqrt{n} d)$ | $O(n \sqrt{n})$ | 2% accuracy drop |
| Reformer (LSH) | $O(n \log n \cdot d)$ | $O(n \log n)$ | Only efficient for $n > 2048$ |
| Longformer | $O(n \cdot w \cdot d)$ | $O(n \cdot w)$ | $w$ = window size |
| **Linformer** | $\mathbf{O(nk \cdot d)}$ | $\mathbf{O(nk)}$ | $k \ll n$, fully linear |

> [!IMPORTANT]
> Linformer achieves $O(1)$ sequential operations (like Transformer), unlike Reformer which requires $O(\log n)$ sequential operations due to hash-based bucketing.

## Experiments

- **Dataset**: BookCorpus + English Wikipedia (3,300M words total) for pretraining; GLUE benchmark (SST-2, IMDB, QNLI, QQP) for fine-tuning evaluation
- **Hardware**: 64 Tesla V100 GPUs (16GB each)
- **Optimizer**: Adam with warmup (250,000 pretraining steps)
- **Baseline**: RoBERTa-base (12 layers, 12 heads, $d_{\text{model}}=768$)
- **Sequence lengths tested**: $n \in \{512, 1024\}$, projected dimension $k \in \{64, 128, 256\}$

### Key Quantitative Results

- At $k = 128$ (for $n = 512$) and $k = 256$ (for $n = 1024$), Linformer perplexity on masked language modeling matches RoBERTa-base within 0.1 points.
- GLUE average: Linformer (layerwise, $k=256$) scores 92.30 vs. RoBERTa-base 92.25.
- **Inference speedup** (16GB V100, batch=256):
  - $n=512$: 1.5× faster, 1.7× less memory
  - $n=8192$: 5.5× faster, 28× less memory
  - $n=65536$: 20× faster, 60× less memory

> [!NOTE]
> Memory gains scale approximately as $n/k$, which is a direct consequence of replacing the $n \times n$ attention matrix with an $n \times k$ matrix.

## Applicability

**Who**: NLP practitioners working with Transformer-based language models who need to process long sequences (documents, legal text, scientific papers, code files) efficiently.

**When**: Useful when (1) sequence length $n > 512$; (2) hardware memory is constrained; (3) inference latency at long sequences is a bottleneck; (4) approximate attention is acceptable.

**Where**: Drop-in replacement for standard self-attention in any Transformer encoder architecture. The projection matrices $E_i, F_i$ are added per layer (or shared across layers) and trained end-to-end.

> [!CAUTION]
> Linformer's approximation quality depends on the choice of $k$; too small a $k$ degrades quality. Also, the projection matrices $E, F$ require the sequence length $n$ to be fixed at training time, limiting applicability to variable-length inputs without padding.
