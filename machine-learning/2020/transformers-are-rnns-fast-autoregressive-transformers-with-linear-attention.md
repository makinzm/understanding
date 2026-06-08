# Meta Information

- URL: [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*.

# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

## Overview

This paper reformulates standard softmax self-attention as a **linear dot-product of kernel feature maps**, exploiting the associativity of matrix multiplication to reduce the per-token complexity from $O(N^2)$ to $O(N)$ in both time and memory. The resulting **linear transformer** can also be rewritten as an RNN, enabling $O(1)$ per-step autoregressive inference. This benefits researchers and engineers building long-sequence autoregressive models (language models, image generation, speech recognition) where the quadratic cost of standard attention becomes prohibitive.

**Applicability:** Most beneficial when sequence length $N$ exceeds the feature-map dimension $D^2$ (i.e., the cross-over point where linear complexity beats quadratic). At inference time on long sequences (e.g., CIFAR-10, $N=3072$), the speedup reaches over 4000×.

## Background: Standard Softmax Attention

Given input $X \in \mathbb{R}^{N \times D}$, projections produce:

```math
\begin{align}
  Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad (W \in \mathbb{R}^{D \times D})
\end{align}
```

The generalized attention output $V'_i \in \mathbb{R}^M$ for query $i$ is:

```math
\begin{align}
  V'_i = \frac{\sum_{j=1}^{N} \text{sim}(Q_i, K_j)\, V_j}{\sum_{j=1}^{N} \text{sim}(Q_i, K_j)}
\end{align}
```

Softmax attention uses $\text{sim}(Q_i, K_j) = \exp(Q_i^\top K_j / \sqrt{D})$, which requires forming the $N \times N$ attention matrix — the source of $O(N^2)$ complexity.

## Linear Attention via Kernel Feature Maps

### Key Idea

Replace the similarity function with a kernel decomposition:

```math
\begin{align}
  \text{sim}(Q_i, K_j) = \varphi(Q_i)^\top \varphi(K_j)
\end{align}
```

where $\varphi: \mathbb{R}^D \to \mathbb{R}^C$ is a feature map satisfying $\text{sim}(x, y) \geq 0$. Substituting:

```math
\begin{align}
  V'_i = \frac{\sum_{j=1}^{N} \varphi(Q_i)^\top \varphi(K_j)\, V_j}{\sum_{j=1}^{N} \varphi(Q_i)^\top \varphi(K_j)}
\end{align}
```

### Associativity Trick — $O(N)$ Computation

Since $\varphi(Q_i)^\top$ is independent of $j$, it can be pulled out:

```math
\begin{align}
  V'_i = \frac{\varphi(Q_i)^\top \left(\sum_{j=1}^{N} \varphi(K_j) V_j^\top\right)}{\varphi(Q_i)^\top \left(\sum_{j=1}^{N} \varphi(K_j)\right)}
\end{align}
```

The matrix $\sum_{j=1}^{N} \varphi(K_j) V_j^\top \in \mathbb{R}^{C \times M}$ is computed once and reused for all $N$ queries, eliminating the $N \times N$ intermediate matrix entirely. Total complexity is $O(NCM)$ where typically $C = D$.

### Feature Map Choice

The authors use $\varphi(x) = \text{elu}(x) + 1$, applied elementwise:

```math
\begin{align}
  \varphi(x)_k = \begin{cases} x_k + 1 & x_k \geq 0 \\ e^{x_k} & x_k < 0 \end{cases}
\end{align}
```

This guarantees strictly positive values (required so that $\text{sim}(Q_i, K_j) > 0$ always), while avoiding zero gradients that ReLU would produce for negative inputs. Softmax cannot be used because its feature map is infinite-dimensional.

## Causal (Autoregressive) Linear Attention as an RNN

For decoder-style autoregressive generation, the causal mask restricts each position $i$ to attend only to $j \leq i$. The cumulative sums become recurrences:

```math
\begin{align}
  S_i &= S_{i-1} + \varphi(K_i) V_i^\top \in \mathbb{R}^{C \times M} \\
  z_i &= z_{i-1} + \varphi(K_i) \in \mathbb{R}^{C}
\end{align}
```

The output at position $i$ is then:

```math
\begin{align}
  y_i = f_l\!\left(\frac{\varphi(Q_i)^\top S_i}{\varphi(Q_i)^\top z_i} + x_i\right)
\end{align}
```

where $f_l$ is the feedforward sub-network. This is exactly an **RNN** with matrix-valued hidden state $(S_i, z_i)$ updated in $O(CM)$ per step, enabling $O(1)$-per-step inference regardless of sequence length.

> [!IMPORTANT]
> At training time, the full bidirectional version (without causal mask) can still be parallelized over the entire sequence using prefix-sum operations, preserving GPU efficiency.

### Algorithm: Training Forward Pass (Causal Linear Attention)

```
Input: Q, K, V ∈ R^{N×D}; feature map φ
S ← 0_{C×M}  # attention memory
z ← 0_C       # normalization memory
for i = 1 to N:
    S ← S + φ(K_i) · V_i^T      # outer product, shape C×M
    z ← z + φ(K_i)              # shape C
    V'_i ← φ(Q_i)^T S / (φ(Q_i)^T z)   # shape M
Output: V' ∈ R^{N×M}
```

This runs in $O(NCM)$ time and $O(CM)$ memory (constant in $N$), vs. $O(N^2 M)$ time and $O(N^2)$ memory for standard causal attention.

### Gradient Computation in Constant Memory

Gradients with respect to $Q$, $K$, $V$ can be computed via reverse-mode cumulative sums, requiring $O(CM)$ memory — independent of $N$. This is implemented in approximately 200 lines of CUDA code, enabling training on very long sequences.

## Complexity Comparison

| Method | Time Complexity | Memory |
|--------|----------------|--------|
| Softmax attention | $O(N^2 \max(D, M))$ | $O(N^2)$ |
| Linear attention (training) | $O(NCM)$ | $O(N \cdot CM)$ |
| Linear attention (inference) | $O(CM)$ per step | $O(CM)$ |
| Reformer (LSH) | $O(N \log N)$ | $O(N \log N)$ |

Linear attention is favorable over softmax when $N > D$ (approximately).

## Comparison with Related Methods

| Method | Complexity | Approximates Softmax? | Causal Inference Cost |
|--------|-----------|----------------------|----------------------|
| **Linear Transformer (this work)** | $O(N)$ | No (different kernel) | $O(1)$ per step |
| Sparse Attention (Child et al. 2019) | $O(N\sqrt{N})$ | Partial | $O(N\sqrt{N})$ |
| Reformer (Kitaev et al. 2020) | $O(N \log N)$ | Approximate | $O(N \log N)$ |
| Performer (Choromanski et al. 2020) | $O(N)$ | Yes (random features) | $O(1)$ per step |

Key differences from Performer: Linear Transformer uses a deterministic ELU+1 feature map, does not approximate softmax, and was developed independently at roughly the same time.

# Experiments

## Datasets

- **Synthetic sequence duplication**: 1000 sequences of length 64, vocabulary size 10; tests ability to copy content
- **MNIST sequential image generation**: $28 \times 28 = 784$ token sequence, 256 pixel values; bits/dim metric
- **CIFAR-10 sequential image generation**: $32 \times 32 \times 3 = 3072$ tokens; bits/dim metric
- **Wall Street Journal (WSJ) speech recognition**: Standard train/dev/test splits; phoneme error rate (PER) metric

## Key Results

- **Synthetic task**: Linear transformer converges as fast as softmax; Reformer fails to converge
- **MNIST**: Linear achieves 0.644 bits/dim vs. softmax 0.621 (slightly worse), but runs **317× faster** at inference (142.8 vs. 0.45 images/sec)
- **CIFAR-10**: Linear achieves 3.40 bits/dim vs. softmax 3.47 (better quality), at **4,462× faster** inference (17.85 vs. 0.004 images/sec)
- **WSJ speech**: Linear achieves 8.08 PER vs. softmax 5.12 PER — a quality gap remains, but linear outperforms LSTM (10.94 PER) and Reformer (9.33 PER)

> [!NOTE]
> The CIFAR-10 speedup of 4,462× arises because the sequence length is $N=3072$, making the $O(N^2)$ softmax overhead extremely expensive while linear attention costs $O(N)$.

## Hardware

GPU (specific model not reported); custom CUDA kernel (~200 lines) implements the linear-time gradient.
