# Meta Information

- URL: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv preprint arXiv:2205.14135.

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## Background: GPU Memory Hierarchy and Standard Attention

Modern GPUs (e.g., NVIDIA A100) have a two-level memory hierarchy:

- **HBM (High-Bandwidth Memory)**: 40–80 GB capacity, ~1.5–2.0 TB/s bandwidth
- **SRAM (on-chip cache)**: ~20 MB total (192 KB per streaming multiprocessor), ~19 TB/s bandwidth

The standard attention computation for a single head is:

$$S = QK^\top \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = PV \in \mathbb{R}^{N \times d}$$

where $Q, K, V \in \mathbb{R}^{N \times d}$ are the query, key, and value matrices, $N$ is the sequence length, and $d$ is the head dimension.

**Problem**: The intermediate matrices $S$ and $P$ each require $O(N^2)$ memory, causing quadratic growth in both memory and HBM read/write operations. For $N = 4096$, $d = 64$, storing $S$ and $P$ requires ~2 GB for float32. Standard attention requires $\Theta(Nd + N^2)$ HBM accesses.

## Core Insight: IO-Awareness

FlashAttention's key insight is that modern deep learning is **memory-bandwidth bound**, not compute bound. The number of floating-point operations is less important than the number of HBM reads/writes (I/O operations). Standard attention implementations materialize the $N \times N$ attention matrix in HBM even when it does not fit in SRAM, causing a bottleneck.

FlashAttention avoids materializing the full attention matrix by:
1. **Tiling**: Breaking the computation into blocks that fit in SRAM
2. **Recomputation**: Recomputing attention weights during the backward pass instead of storing them

## Softmax Decomposition for Block-wise Computation

To compute softmax block-by-block without access to global statistics, FlashAttention maintains the following statistics:

$$m(x) := \max_i x_i, \quad f(x) := \left[e^{x_1 - m(x)}, \ldots, e^{x_B - m(x)}\right], \quad \ell(x) := \sum_i f(x)_i$$

$$\text{softmax}(x) := \frac{f(x)}{\ell(x)}$$

For two concatenated vectors $x = [x^{(1)}, x^{(2)}]$, these statistics compose as:

$$m(x) = \max(m(x^{(1)}), m(x^{(2)}))$$
$$\ell(x) = e^{m(x^{(1)}) - m(x)} \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)} \ell(x^{(2)})$$

This allows correct softmax computation across blocks by carrying forward running statistics $(m, \ell)$.

## FlashAttention Forward Pass Algorithm

**Block sizes** chosen to fit in SRAM of size $M$:

$$B_c = \left\lceil \frac{M}{4d} \right\rceil, \quad B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil, d\right)$$

**Algorithm (Forward Pass)**:

```
Input: Q, K, V ∈ ℝ^{N×d} in HBM, SRAM size M
Output: O ∈ ℝ^{N×d}

1. Divide Q into T_r = ⌈N/B_r⌉ blocks Q_1,...,Q_{T_r} of size B_r × d
2. Divide K, V into T_c = ⌈N/B_c⌉ blocks K_1,...,K_{T_c}, V_1,...,V_{T_c} of size B_c × d
3. Initialize O = zeros(N, d), ℓ = zeros(N), m = -inf(N) in HBM

4. For j = 1 to T_c:                         // outer loop over K, V blocks
   a. Load K_j, V_j from HBM to SRAM

   5. For i = 1 to T_r:                      // inner loop over Q blocks
      a. Load Q_i, O_i, ℓ_i, m_i from HBM to SRAM
      b. S_ij = Q_i K_j^T ∈ ℝ^{B_r × B_c}   // on-chip dot products
      c. m̃_ij = rowmax(S_ij) ∈ ℝ^{B_r}
      d. P̃_ij = exp(S_ij - m̃_ij) ∈ ℝ^{B_r × B_c}  // numerically stable exp
      e. ℓ̃_ij = rowsum(P̃_ij) ∈ ℝ^{B_r}
      f. m_i^new = max(m_i, m̃_ij)
      g. ℓ_i^new = exp(m_i - m_i^new)·ℓ_i + exp(m̃_ij - m_i^new)·ℓ̃_ij
      h. O_i ← diag(ℓ_i^new)^{-1} (diag(ℓ_i)·exp(m_i - m_i^new)·O_i + P̃_ij·V_j)
      i. Write O_i, ℓ_i^new, m_i^new back to HBM

6. Return O
```

> [!NOTE]
> The outer loop iterates over key/value blocks while the inner loop iterates over query blocks. This ordering ensures each K_j, V_j pair is loaded once from HBM, reducing I/O operations.

## FlashAttention Backward Pass

Instead of storing the $O(N^2)$ attention matrices $S$ and $P$, FlashAttention stores only:
- Output $O \in \mathbb{R}^{N \times d}$
- Normalization statistics $(\ell, m) \in \mathbb{R}^N$ per head

During the backward pass, the attention weights are **recomputed on-chip** from the saved $Q, K$ blocks. The three gradient components are:

$$dV = P^\top dO, \quad dQ = P \cdot dS_{\text{rows}}, \quad dK = P^\top \cdot dS_{\text{cols}}$$

where $dS$ is the gradient through the softmax, computed as:

$$dS_{ij} = P_{ij} \left(dP_{ij} - \sum_k dP_{ik} P_{ik}\right)$$

> [!IMPORTANT]
> The backward pass performs strictly more FLOPs than the standard approach (due to recomputation), yet runs faster because HBM accesses dominate total runtime on modern hardware.

## IO Complexity Analysis

| Method | HBM Accesses |
|--------|-------------|
| Standard attention | $\Theta(Nd + N^2)$ |
| FlashAttention | $\Theta(N^2 d^2 M^{-1})$ |

For $d = 64$, $M = 100\,\text{KB}$ (in elements), the reduction factor is approximately $M / (4d) \approx 9\times$.

**Proposition**: For $d \leq M \leq Nd$, FlashAttention's HBM access complexity is $\Theta(N^2 d^2 M^{-1})$.

**Lower bound (Proposition 3)**: For any exact attention algorithm, $\Omega(N^2 d^2 M^{-1})$ HBM accesses are required. FlashAttention is asymptotically optimal.

## Block-Sparse FlashAttention

For approximate attention with a sparsity mask $\tilde{M} \in \{0,1\}^{N/B_r \times N/B_c}$ with fraction $s$ of nonzero blocks:

$$\text{HBM accesses} = \Theta\left(Nd + N^2 d^2 M^{-1} s\right)$$

The algorithm skips loading blocks that are masked out, achieving proportional speedup to sparsity level while maintaining exact arithmetic on retained blocks.

> [!TIP]
> Block-sparse patterns from methods like BigBird or Longformer (local + strided + global) can be directly combined with FlashAttention for long-sequence transformers.

## Comparison with Related Methods

| Method | Exact? | Memory | HBM Accesses | Notes |
|--------|--------|--------|--------------|-------|
| Standard attention | Yes | $O(N^2)$ | $\Theta(Nd + N^2)$ | Baseline |
| Linformer | No | $O(N)$ | $\Theta(Nk)$ | Approximates K, V via projections |
| Performer | No | $O(N)$ | $\Theta(Nd)$ | Random feature kernel approximation |
| Reformer | No | $O(N\log N)$ | $\Theta(N\log N \cdot d)$ | LSH-based |
| FlashAttention | **Yes** | $O(N)$ | $\Theta(N^2 d^2 M^{-1})$ | IO-optimal exact |
| Block-Sparse FlashAttention | Approx. | $O(N)$ | $\Theta(N^2 d^2 M^{-1} s)$ | Sparse + IO-optimal |

> [!IMPORTANT]
> Unlike approximate methods, FlashAttention computes **exact** attention (same output as standard attention, no approximation error). The speedup comes purely from optimizing memory access patterns.

# Experiments

- **Datasets**:
  - BERT-large pretraining: English Wikipedia corpus, sequence length 512
  - GPT-2 (small & medium) language modeling: OpenWebText, sequence lengths 1K–4K
  - Long-Range Arena (LRA): 5 classification tasks, sequence lengths 1K–4K (ListOps, Text, Retrieval, Image, Pathfinder)
  - Path-X: Pathfinding task on $128 \times 128$ images, effective sequence length 16,384
  - Path-256: Pathfinding on $256 \times 256$ images, effective sequence length 65,536
  - MIMIC-III clinical notes: avg. 2,395 tokens, max 14,562 tokens; ICU mortality prediction
  - ECtHR legal documents: avg. 2,197 tokens, max 49,392 tokens; multilabel classification

- **Hardware**: NVIDIA A100 GPU (40 GB HBM, 192 KB on-chip SRAM per SM), A100 PCIe for some experiments

- **Optimizer**: Adam for GPT-2; AdamW for BERT; task-specific for LRA

- **Key Results**:
  - BERT-large pretraining: 15% faster than MLPerf 1.1 record (17.4 min vs. 20.0 min to 72% dev accuracy)
  - GPT-2-small: 3.5× speedup over HuggingFace (2.7 days vs. 9.5 days for 117M parameters on OpenWebText)
  - GPT-2-medium: 3.0× speedup (6.9 days vs. 21.0 days for 345M parameters)
  - GPT-2 at 4K context: 0.7 perplexity improvement over 1K baseline, 30% faster training
  - LRA: 2.4× overall speedup; FlashAttention beats all approximate attention methods in accuracy
  - Path-X (16K): 61.4% accuracy — first transformer to exceed random (50%) on this benchmark
  - Path-256 (64K): 63.1% accuracy — enabled by FlashAttention's linear memory scaling
  - MIMIC-III (16K context): +4.3 F1 points vs. 512-token baseline
  - ECtHR (8K context): +8.5 F1 points vs. 512-token baseline
  - Memory: linear growth in sequence length; up to 20× more memory-efficient than exact baselines

> [!CAUTION]
> The FLOP count for FlashAttention is higher than standard attention (due to backward-pass recomputation), but wall-clock time is lower because the algorithm is memory-bandwidth bound rather than compute bound on current GPU hardware.
