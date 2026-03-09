# Meta Information

- URL: [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*.

# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

## Overview

This paper reformulates the standard softmax self-attention as a **linear dot-product of kernel feature maps**, exploiting the associativity of matrix multiplication to reduce the complexity from $O(N^2)$ to $O(N)$ in both time and memory. The key insight is that this linearized causal attention is mathematically equivalent to a Recurrent Neural Network (RNN) with a matrix-valued hidden state, establishing a formal connection between Transformers and RNNs.

**Applicability:** Researchers and practitioners building autoregressive sequence models (language models, image generation, speech recognition) who need $O(N^2)$ softmax attention replaced by a computationally efficient alternative — particularly beneficial for long sequences where quadratic cost is prohibitive. The method applies wherever causal (decoder-side) or full self-attention is used.

## Background: Standard Softmax Attention

A Transformer layer computes:

$$T_l(x) = f_l(A_l(x) + x)$$

where $f_l$ is a feedforward sub-network and $A_l$ is self-attention. Given input $x \in \mathbb{R}^{N \times D}$:

$$Q = xW_Q, \quad K = xW_K, \quad V = xW_V \quad (W \in \mathbb{R}^{D \times D})$$

$$A_l(x) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{D}}\right)V$$

The generalized attention with similarity function $\text{sim}(Q_i, K_j) \geq 0$ is:

$$V'_i = \frac{\sum_{j=1}^{N} \text{sim}(Q_i, K_j)\, V_j}{\sum_{j=1}^{N} \text{sim}(Q_i, K_j)}$$

The softmax uses $\text{sim}(Q_i, K_j) = \exp(Q_i^\top K_j / \sqrt{D})$, which requires computing an $N \times N$ attention matrix — the source of $O(N^2)$ complexity.

## Linear Attention via Kernel Feature Maps

### Key Idea

Replace the similarity function with a kernel decomposition:

$$\text{sim}(Q_i, K_j) = \varphi(Q_i)^\top \varphi(K_j)$$

where $\varphi: \mathbb{R}^D \to \mathbb{R}^C$ is a feature map. This yields:

$$V'_i = \frac{\sum_{j=1}^{N} \varphi(Q_i)^\top \varphi(K_j)\, V_j}{\sum_{j=1}^{N} \varphi(Q_i)^\top \varphi(K_j)}$$

### Associativity Trick (O(N) Computation)

By pulling $\varphi(Q_i)^\top$ out of the summation and applying matrix associativity:

$$V'_i = \frac{\varphi(Q_i)^\top \sum_{j=1}^{N} \varphi(K_j) V_j^\top}{\varphi(Q_i)^\top \sum_{j=1}^{N} \varphi(K_j)}$$

In vectorized form:

$$(\varphi(Q)\,\varphi(K)^\top)\,V = \varphi(Q)\,(\varphi(K)^\top V)$$

The right-hand-side computes $\varphi(K)^\top V \in \mathbb{R}^{C \times M}$ once ($O(NCM)$) and reuses it for all queries, eliminating the $N \times N$ matrix entirely.

### Feature Map Choice

The authors propose:

$$\varphi(x) = \text{elu}(x) + 1$$

where $\text{elu}(x) = x$ for $x > 0$ and $e^x - 1$ for $x \leq 0$. The $+1$ offset ensures strictly positive similarity scores ($\varphi(x) > 0$ component-wise), a necessary condition for attention normalization to be well-defined.

> [!NOTE]
> Exact softmax attention cannot be linearized because $\exp(Q^\top K)$ requires an infinite-dimensional feature map. The feature map $\varphi(x) = \text{elu}(x) + 1$ is an approximation that trades exact softmax behavior for linear-time computation.

## Causal (Autoregressive) Linear Attention

### Masked Attention

For autoregressive generation, causal masking restricts each position to attend only to previous positions:

$$V'_i = \frac{\sum_{j=1}^{i} \varphi(Q_i)^\top \varphi(K_j)\, V_j}{\sum_{j=1}^{i} \varphi(Q_i)^\top \varphi(K_j)}$$

### Hidden State Formulation

Define cumulative states:

$$S_i = \sum_{j=1}^{i} \varphi(K_j)\, V_j^\top \in \mathbb{R}^{C \times M}, \qquad Z_i = \sum_{j=1}^{i} \varphi(K_j) \in \mathbb{R}^{C}$$

Then:

$$V'_i = \frac{\varphi(Q_i)^\top S_i}{\varphi(Q_i)^\top Z_i}$$

### Recurrent Update (RNN Equivalence)

With initial states $S_0 = 0, Z_0 = 0$:

$$S_i = S_{i-1} + \varphi(x_i W_K)\,(x_i W_V)^\top$$

$$Z_i = Z_{i-1} + \varphi(x_i W_K)$$

$$y_i = f_l\!\left(\frac{\varphi(x_i W_Q)^\top S_i}{\varphi(x_i W_Q)^\top Z_i} + x_i\right)$$

This is precisely an RNN with matrix-valued hidden state $(S_i, Z_i)$: the model processes tokens sequentially, updating the hidden state via a fixed recurrence and reading the output from it. At inference, only the current hidden state needs to be stored — $O(C \times M)$ memory independent of sequence length.

> [!IMPORTANT]
> The RNN equivalence applies only with **causal masking**. Bidirectional (full) linear attention does not have a recurrent formulation in the same sense; it still achieves $O(N)$ cost via the associativity trick but requires the full sequence context.

## Algorithms

### Forward Pass (Causal Linear Attention)

```
function CausalLinearAttention(φ(Q), φ(K), V):
    # φ(Q), φ(K) ∈ R^{N×C}, V ∈ R^{N×M}
    S ← 0  # R^{C×M}
    Z ← 0  # R^{C}
    for i = 1 to N:
        S ← S + φ(K_i) ⊗ V_i  # outer product, R^{C×M}
        Z ← Z + φ(K_i)         # R^{C}
        V̄_i ← (φ(Q_i)^T S) / (φ(Q_i)^T Z)  # R^{M}
    return V̄  # R^{N×M}
```

Complexity: $O(N \cdot C \cdot M)$ time, $O(C \cdot M)$ additional memory (hidden state only).

### Backward Pass (Gradient with Reverse Cumulative Sums)

The gradient computation uses **reverse cumulative sums** to maintain constant memory during backpropagation:

```
function Backward(φ(Q), φ(K), V, G):
    # Forward scan for query gradients
    S ← 0
    for i = 1 to N:
        S ← S + φ(K_i) ⊗ V_i
        ∇_{φ(Q_i)}L ← G_i · S^T          # R^{C}

    # Reverse scan for key and value gradients
    S ← 0
    for i = N down to 1:
        S ← S + φ(Q_i) ⊗ G_i             # R^{C×M}
        ∇_{V_i}L    ← S^T · φ(K_i)       # R^{M}
        ∇_{φ(K_i)}L ← S · V_i            # R^{C}

    return ∇_{φ(Q)}L, ∇_{φ(K)}L, ∇_V L
```

This avoids storing the full $N \times C \times M$ computation graph, keeping backward pass memory at $O(C \cdot M)$.

## Complexity Comparison

| Method | Time | Memory |
|---|---|---|
| Softmax Attention | $O(N^2 D)$ | $O(N^2 + ND)$ |
| Linear Attention (this work) | $O(N C M)$ | $O(CM)$ at inference |
| Polynomial Kernel (degree 2) | $O(N D^2 M)$ | $O(D^2 M)$ |

Linear attention is faster than softmax when $CM \ll N \cdot D$, which holds for long sequences. For typical configurations ($C \approx D$, $M \approx D$), linear attention is $O(ND^2)$ vs. softmax $O(N^2 D)$, giving $O(N/D)$ speedup — crucial when $N \gg D$.

## Comparison with Related Methods

| Method | Complexity | Exact Attention? | Causal Support | Gradient Memory |
|---|---|---|---|---|
| Softmax Transformer | $O(N^2)$ | Yes | Yes (masking) | $O(N^2)$ |
| Linear Transformer (this) | $O(N)$ | No (approx.) | Yes (RNN form) | $O(CM)$ |
| Reformer | $O(N \log N)$ | Approx. (LSH) | Yes | $O(N \log N)$ |
| Longformer | $O(N)$ | Local window only | Yes | $O(N)$ |
| Performer | $O(N)$ | Random features approx. | Yes | $O(CM)$ |

> [!TIP]
> The Performer (Choromanski et al., 2020) uses random Fourier features to approximate softmax attention, whereas this work uses a deterministic ELU-based feature map. Both achieve $O(N)$ but with different approximation guarantees and empirical trade-offs.

## Experiments

- **Datasets:**
  - **Synthetic copy task:** Sequences up to length $2^{16}$ with vocabulary of 10 symbols, trained to duplicate input
  - **MNIST** ($28 \times 28 = 784$ pixels, treated as a pixel-level sequence): 60,000 training / 10,000 test images
  - **CIFAR-10** ($32 \times 32 \times 3 = 3{,}072$ tokens per image): standard split (50,000 / 10,000)
  - **WSJ (Wall Street Journal) Speech Recognition:** ~80 hours total audio; 40-dimensional mel-scale filterbank features; average sequence length 800 frames, max 2,400 frames

- **Hardware:** Single GPU (exact GPU model not specified)

- **Optimizers:**
  - MNIST/CIFAR-10: RAdam, learning rate $10^{-4}$
  - WSJ: Adam for LSTM baseline ($lr = 10^{-3}$), RAdam for transformers ($lr = 10^{-4}$)

- **Key Results:**
  - **MNIST** — Linear: 0.644 bits/dim at 142.8 images/sec; Softmax: 0.621 bits/dim at 0.45 images/sec → **317× faster inference** with marginal quality loss
  - **CIFAR-10** — Linear: 3.40 bits/dim at 17.85 images/sec; Softmax: 3.47 bits/dim at 0.004 images/sec → **4,462× faster inference**, with linear achieving better perplexity given the same wall-clock budget due to speed advantages
  - **WSJ ASR** — Linear: 8.08 PER at 824 sec/epoch; Softmax: 5.12 PER at 2,711 sec/epoch; Bi-LSTM: 10.94 PER → **3.3× faster training** per epoch; quality gap vs. softmax remains (8.08 vs. 5.12 PER)

> [!CAUTION]
> The dramatic speedup numbers (317× and 4,462×) reflect autoregressive **inference** where softmax must recompute all attention for each new token (sequential, no parallelism), while linear attention uses its RNN form with $O(CM)$ per step. During parallel training, the speedup is smaller (roughly linear in $N/D$).
