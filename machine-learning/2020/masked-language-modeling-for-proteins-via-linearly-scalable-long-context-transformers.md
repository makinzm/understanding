# Meta Information

- URL: [Masked Language Modeling for Proteins via Linearly Scalable Long-Context Transformers](https://arxiv.org/abs/2006.03555)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Belanger, D., Colwell, L., & Weller, A. (2020). Masked Language Modeling for Proteins via Linearly Scalable Long-Context Transformers. arXiv preprint arXiv:2006.03555.

# Masked Language Modeling for Proteins via Linearly Scalable Long-Context Transformers (Performer)

This paper introduces the **Performer**, a Transformer architecture that replaces standard softmax self-attention with FAVOR (Fast Attention Via Orthogonal Random features). By approximating the attention matrix using random feature maps, Performer reduces the time and space complexity from quadratic $O(L^2 d)$ to linear $O(L d^2 \log d)$ in sequence length $L$, enabling masked language modeling on protein sequences of up to $L = 8192$ tokens — lengths that make standard Transformers infeasible.

Performers are applicable to any practitioner or researcher working with long biological sequences (e.g., genomics, proteomics), long documents, or other domains where sequence length is the primary bottleneck.

## Background: Standard Attention

Standard bidirectional dot-product attention computes:

$$\text{Att}_{\leftrightarrow}(Q, K, V) = D^{-1} A V$$

where:
- $Q, K, V \in \mathbb{R}^{L \times d}$ are query, key, and value matrices ($L$ = sequence length, $d$ = head dimension)
- $A = \exp\!\left(\tfrac{QK^\top}{\sqrt{d}}\right) \in \mathbb{R}^{L \times L}$ is the attention matrix
- $D = \text{diag}(A \mathbf{1}_L)$ is a diagonal normalization matrix

Computing $A$ explicitly requires $O(L^2 d)$ time and $O(L^2)$ memory — prohibitive for long sequences.

## Generalized Attention (GA)

The paper introduces a **Generalized Attention** framework to unify softmax and other attention variants:

$$A^{g,K,h}_{ij} = g(Q_i^\top) \, K(Q_i^\top, K_j^\top) \, h(K_j^\top)$$

where $g$ and $h$ are element-wise functions and $K$ is a kernel function. Standard softmax attention is recovered when $K(x, y) = \exp(x^\top y / \sqrt{d})$, $g = h = 1$.

## FAVOR: Fast Attention Via Orthogonal Random Features

### Core Idea

FAVOR approximates the kernel $K(x, y)$ using a **random feature map** $\phi: \mathbb{R}^d \to \mathbb{R}^M$ such that:

$$K(x, y) = \mathbb{E}[\phi(x)^\top \phi(y)]$$

By substituting $\phi(Q_i)$ and $\phi(K_j)$ for query and key representations, the attention matrix factorizes as:

$$A \approx \hat{Q} \hat{K}^\top, \quad \hat{Q} = \phi(Q) \in \mathbb{R}^{L \times M}, \; \hat{K} = \phi(K) \in \mathbb{R}^{L \times M}$$

This allows computation as $(\hat{Q} \hat{K}^\top) V = \hat{Q} (\hat{K}^\top V)$ using matrix associativity — the inner product $\hat{K}^\top V \in \mathbb{R}^{M \times d}$ is computed first, reducing complexity to $O(LMd)$.

### Attention Decomposition for Softmax

For softmax attention, the attention weights involve an exponential kernel. The paper rewrites:

$$A_{ij} = \exp\!\left(\tfrac{Q_i^\top K_j}{\sqrt{d}}\right) = \exp\!\left(-\tfrac{\|Q_i - K_j\|^2}{2\sqrt{d}}\right) \exp\!\left(\tfrac{\|Q_i\|^2 + \|K_j\|^2}{2\sqrt{d}}\right)$$

This decomposes into $A = D_e B D_k$, where:
- $B_{ij} = \exp\!\bigl(-\|Q_i - K_j\|^2 / r\bigr)$ is a Gaussian (RBF) kernel
- $D_e$ and $D_k$ are diagonal matrices from per-row and per-column normalization

### Random Feature Map $\phi$

For the RBF kernel $B$, classical random Fourier features give an unbiased estimator using $M$ random samples $\omega_m \sim \mathcal{N}(0, I_d)$:

$$\phi(x) = \frac{1}{\sqrt{M}} \bigl(\cos(\omega_1^\top x + b_1), \ldots, \cos(\omega_M^\top x + b_M)\bigr)^\top$$

where $b_m \sim \text{Uniform}[0, 2\pi]$.

### Orthogonal Random Features (ORF)

To reduce variance in the approximation, the paper uses **orthogonal random features**: instead of sampling $\omega_1, \ldots, \omega_M$ independently, they are drawn from an isotropic distribution subject to mutual orthogonality via Gram-Schmidt or Givens rotations.

Three ORF variants are evaluated:
| Variant | Space Complexity | Computation |
|---|---|---|
| Regular ORFs | $O(Md)$ | $O(Md^2)$ preprocessing |
| Hadamard ORFs | $O(M + Ld + ML)$ | $O(M \log d)$ |
| Givens ORFs | $O(M \log d)$ | $O(M \log d)$ |

### FAVOR Algorithm (Bidirectional and Unidirectional)

**Algorithm: FAVOR Attention**

```
Input: Q, K, V ∈ R^{L×d}, M random features
Output: Normalized attention output O ∈ R^{L×d}

1. Compute diagonal matrices D_e, D_k from per-row norms of Q, K
2. Sample M orthogonal random vectors ω_1,...,ω_M
3. Compute Q̂ = D_e · φ(Q/r^{1/2}) ∈ R^{L×M}   # random feature queries
4. Compute K̂ = D_k · φ(K/r^{1/2}) ∈ R^{L×M}   # random feature keys

--- Bidirectional (Full Sequence) ---
5. Buf₁ = K̂ᵀ · V ∈ R^{M×d}          # aggregate keys × values
6. Buf₂ = Q̂ · Buf₁ ∈ R^{L×d}         # query-weighted sum
7. norm = Q̂ · (K̂ᵀ · 1_L) ∈ R^L       # normalization
8. Return Buf₂ / norm (row-wise)

--- Unidirectional (Causal/Autoregressive) ---
5. Compute prefix-sum tensor G^PS via equation (14):
      G^PS_i = Σ_{j≤i} K̂_j^T · V_j ∈ R^{M×d}
6. Buf = Q̂_i · G^PS_i for each i
7. Normalize by prefix-sum of K̂ᵀ 1
```

**Complexity comparison:**

| Method | Time | Space |
|---|---|---|
| Standard Attention | $O(L^2 d)$ | $O(L^2)$ |
| Reformer (LSH) | $O(L d^2 \log L)$ | $O(L d \log L)$ |
| **Performer (FAVOR)** | $O(L M d)$ with $M = O(d \log d)$ | $O(Ld + ML)$ |

With $M = \Theta(d \log d)$, FAVOR achieves **$O(Ld^2 \log d)$** time, linear in $L$.

## Theoretical Guarantees

**Theorem 1 (Uniform Convergence):** For RBF kernels with queries and keys bounded by norm $R$, to achieve approximation error $\|\hat{A} - A\|_1 \leq \varepsilon$, it suffices to use:

$$M = \Omega\!\left(\frac{d}{\delta^2} \log \frac{4\sigma R}{\delta d^{1/4}}\right) \text{ random features}$$

where $\delta = \varepsilon / (g^* h^*)$. Critically, $M$ is **independent of sequence length $L$**, confirming the approximation quality does not degrade with longer sequences. The optimal number of features is:

$$M_{\text{opt}} = \Theta(d \log d)$$

## Backward Compatibility with Pretrained Transformers

A key practical result: pretrained Transformer weights can be fine-tuned to work with FAVOR. Starting from a BERT checkpoint with FAVOR substituted in (initial accuracy ~0.07), the model recovers baseline masked language modeling accuracy within a fraction of original training steps. This makes Performer an efficient drop-in replacement for deployed Transformers.

## Experiments

- **Dataset**: TrEMBL protein sequence database
  - Train: 104,863,744 sequences (mean length 353.09 amino acids)
  - Validation: 102,400 sequences
  - Test: 1,033,216 sequences
  - Out-of-distribution test: 29,696 sequences
  - Concatenated variant: 4,532,224 training samples at $L = 8{,}192$ (for long-context experiments)
- **Dataset**: ImageNet64 ($L = 12{,}288$) for image pixel modeling benchmarks
- **Hardware**: TPU pods (not specified further)
- **Optimizer**: Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$; gradient clipping = 0.5, weight decay = 0.1, dropout = 0.1
- **Architecture**: 36-layer models; 256 random features (default)

### Key Results

| Model | Test Accuracy | Test Perplexity |
|---|---|---|
| Standard Transformer | 30.80% | 9.37 |
| Reformer (LSH) | — | worse |
| **Performer (Generalized, ReLU)** | **31.58%** | **9.17** |

Performer with generalized (ReLU-based) attention outperforms the standard Transformer on protein masked language modeling, while scaling to sequence lengths ($L = 8{,}192$) where the standard Transformer runs out of memory. Reformer achieves $O(L d^2 \log L)$ — still super-linear in $L$ — and underperforms at these lengths.

## Differences from Similar Methods

| Method | Complexity | Key Mechanism | Limitation |
|---|---|---|---|
| Standard Transformer | $O(L^2 d)$ | Full softmax attention | Memory-prohibitive for $L > 4096$ |
| Reformer | $O(L d^2 \log L)$ | Locality-Sensitive Hashing | Super-linear; relies on input structure |
| Sparse Transformer | $O(L^{3/2} d)$ | Fixed sparse patterns | Approximation error; no guarantees |
| **Performer (FAVOR)** | $O(L d^2 \log d)$ | Random feature kernel approximation | Approximation; requires fine-tuning for pretrained models |

> [!NOTE]
> Unlike sparse attention methods (Reformer, Sparse Transformer) that restrict which tokens can attend to each other, FAVOR approximates the **full** attention matrix. This preserves the theoretical expressiveness of dense attention while achieving linear complexity.

> [!IMPORTANT]
> The paper's title emphasizes protein modeling, but FAVOR is a general-purpose Transformer modification. The protein domain was chosen because protein sequences are naturally long (hundreds to thousands of amino acids) and benefit most from linear-complexity attention.

> [!TIP]
> The follow-up paper "Rethinking Attention with Performers" (Choromanski et al., 2021, ICLR) extends this work with improved theoretical analysis and experiments on more benchmarks.
