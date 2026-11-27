# Meta Information

- URL: [BP-Transformer: Modelling Long-Range Context via Binary Partitioning](https://arxiv.org/abs/1911.04070)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Ye, Z., Guo, Q., Gan, Q., Qiu, X., & Zhang, Z. (2019). BP-Transformer: Modelling Long-Range Context via Binary Partitioning. arXiv preprint arXiv:1911.04070.

# BP-Transformer: Modelling Long-Range Context via Binary Partitioning

## Problem: Quadratic Complexity of Self-Attention

Standard Transformer self-attention computes attention between every pair of tokens, yielding $O(n^2)$ time and memory complexity for a sequence of length $n$. This becomes a critical bottleneck when processing long documents such as books, legal texts, or scientific articles, where $n$ can reach thousands of tokens.

BP-Transformer (BPT) addresses this by replacing the dense all-pairs attention with a sparse, hierarchical attention pattern constructed via binary partitioning of the token sequence.

## Binary Partitioning Tree

The core idea is to organize the input token sequence into a balanced binary tree where:

- Leaves correspond to individual tokens (or token spans).
- Internal nodes represent merged spans at increasing granularity.

Given a sequence $X = (x_1, x_2, \ldots, x_n)$, the binary partitioning recursively splits the sequence into two halves until reaching individual tokens:

```
Level 0 (root):  [x_1, ..., x_n]
Level 1:         [x_1, ..., x_{n/2}]  |  [x_{n/2+1}, ..., x_n]
...
Level log(n):    [x_1]  [x_2]  ...  [x_n]  (leaves)
```

Each internal node stores an aggregated representation of the span it covers, typically computed as the mean or attention-pooled vector of the tokens in that span.

> [!NOTE]
> The binary partitioning is fixed and positional (not content-adaptive). This keeps the construction deterministic and efficient, unlike learned clustering methods.

## Fine-to-Coarse Attention Mechanism

For a given token $x_i$ (a query), BPT defines which keys it attends to using a **fine-to-coarse** strategy:

- Tokens **close** to $x_i$: attend to individual token representations (fine-grained, local context).
- Tokens **far** from $x_i$: attend to coarser span representations from higher levels of the binary tree.

More precisely, given a parameter $k$ that controls the number of neighbors attended to at each level, token $x_i$ attends to:

- Its $k$ nearest neighbors at the leaf level (full token resolution).
- Representatives of $k$ contiguous spans at level 1 (spans of size 2).
- Representatives of $k$ contiguous spans at level 2 (spans of size 4).
- ... and so on up to the root level.

This results in $O(k \log n)$ attention connections per token, giving total complexity:

$$\text{Complexity} = O(k \cdot n \cdot \log(n/k))$$

where $k$ is a hyperparameter controlling the trade-off between efficiency and expressiveness.

> [!IMPORTANT]
> When $k = n$, BPT degenerates to standard full self-attention ($O(n^2)$). When $k = 1$, BPT achieves near-linear complexity $O(n \log n)$. In practice, $k$ is set to a small constant (e.g., 16 or 32).

## Algorithm: BPT Attention

**Input:** Token sequence $X \in \mathbb{R}^{n \times d}$ where $n$ is sequence length and $d$ is hidden dimension.
**Output:** Context-enriched representations $Z \in \mathbb{R}^{n \times d}$.

```
1. Build binary partitioning tree over positions 1..n
2. For each internal node v at depth ℓ covering span S_v:
     rep(v) ← AveragePool(X[S_v])  # or learned pooling
3. For each token x_i (i = 1..n):
     neighbors ← {}
     For ℓ = 0 to log(n):
         span_size ← 2^ℓ
         k_neighbors ← k closest spans of size span_size around i
         neighbors ← neighbors ∪ {rep(node) for node in k_neighbors}
     Compute attention: z_i ← Attention(q_i, K_neighbors, V_neighbors)
4. Return Z = [z_1, z_2, ..., z_n]
```

The attention computation at each token uses the standard scaled dot-product:

$$z_i = \text{softmax}\left(\frac{Q_i K_{\text{neighbors}}^\top}{\sqrt{d_k}}\right) V_{\text{neighbors}}$$

where $Q_i \in \mathbb{R}^{1 \times d_k}$, $K_{\text{neighbors}} \in \mathbb{R}^{|N_i| \times d_k}$, $V_{\text{neighbors}} \in \mathbb{R}^{|N_i| \times d}$, and $|N_i| = O(k \log n)$.

## Comparison with Related Methods

| Method | Complexity | Attention Pattern | Notes |
|---|---|---|---|
| Full Transformer | $O(n^2)$ | Dense all-pairs | Baseline; exact but expensive |
| Longformer | $O(n \cdot w)$ | Sliding window + global tokens | Local + task-specific global attention |
| BigBird | $O(n)$ | Random + window + global | Theoretically approximates full attention |
| Sparse Transformer | $O(n \sqrt{n})$ | Strided or fixed patterns | Fixed sparse patterns |
| Reformer | $O(n \log n)$ | LSH-based | Approximate; content-adaptive |
| **BP-Transformer** | $O(k \cdot n \cdot \log(n/k))$ | Hierarchical fine-to-coarse | Deterministic tree structure |

> [!TIP]
> BPT is related to hierarchical attention in that it uses multi-scale representations, but differs from models like HIBERT or HANs because BPT applies within a single sequence rather than across document sections.

## Applicability

BPT is designed for:

- **Who**: NLP researchers and practitioners working on tasks with long input sequences.
- **When**: When sequences exceed the practical length limit of full self-attention (e.g., $n > 512$).
- **Where**: Text classification, language modeling, and machine translation over long documents; can replace standard self-attention layers in any Transformer architecture.

## Implementation Details

The authors implemented BPT in PyTorch with custom CUDA kernels for sparse attention operations to achieve practical speedups on GPU hardware. The sparse attention pattern induced by the binary tree structure is not natively supported by standard dense matrix multiplication, requiring specialized implementations.

# Experiments

- **Datasets**:
  - **Text Classification**: Long document classification benchmarks (e.g., IMDB, Hyperpartisan news detection).
  - **Machine Translation**: WMT translation benchmarks (standard seq2seq settings).
  - **Language Modeling**: Long-range language modeling benchmarks (e.g., WikiText-103, enwik8 character-level).
- **Hardware**: GPU cluster with CUDA support for sparse kernel operations.
- **Optimizer**: Adam optimizer (standard for Transformer training).
- **Results**:
  - BPT outperforms full self-attention models on long-document tasks where quadratic complexity is prohibitive.
  - BPT achieves competitive perplexity on language modeling with significantly reduced memory and computation compared to full attention.
  - On machine translation, BPT matches or exceeds standard Transformer performance while scaling to longer sequences.
  - The fine-to-coarse attention pattern proves particularly effective at capturing both local syntactic structure and long-range discourse relationships.

> [!CAUTION]
> The exact numerical results (BLEU scores, perplexity values, accuracy figures) were not directly extractable from the available sources. The above is a qualitative summary based on the abstract and descriptions; consult the full paper for precise experimental numbers.
