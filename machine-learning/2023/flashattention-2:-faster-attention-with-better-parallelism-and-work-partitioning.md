# Meta Information

- URL: [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv preprint arXiv:2307.08691.

# Overview

FlashAttention-2 is an exact attention algorithm designed to compute multi-head self-attention without materializing the $N \times N$ attention matrix, achieving 2× speedup over FlashAttention (2022) through three targeted GPU optimizations: (1) reducing non-matrix-multiplication floating-point operations, (2) parallelizing across the sequence length dimension, and (3) redistributing work among GPU warps to minimize synchronization overhead.

**Applicability**: Any practitioner training or serving Transformer models (LLMs, vision transformers) who wants to extend context length or improve throughput on modern NVIDIA GPUs (A100, H100). The algorithm is an IO-aware drop-in replacement for standard scaled dot-product attention.

## Background: Standard Attention and IO Bottleneck

Standard attention for queries $Q \in \mathbb{R}^{N \times d}$, keys $K \in \mathbb{R}^{N \times d}$, values $V \in \mathbb{R}^{N \times d}$ is:

```math
\begin{align}
  S &= Q K^\top \in \mathbb{R}^{N \times N} \\
  P &= \text{softmax}(S) \in \mathbb{R}^{N \times N} \\
  O &= P V \in \mathbb{R}^{N \times d}
\end{align}
```

The bottleneck is writing and reading the $N \times N$ matrices $S$ and $P$ to/from GPU HBM (high-bandwidth memory). For $N = 16{,}384$ and $d = 128$, this matrix alone requires ~8 GB (in float16), far exceeding the ~20 MB on-chip SRAM.

**Memory bandwidth figures (A100 80GB):**
| Memory tier | Capacity | Bandwidth |
|---|---|---|
| HBM | 80 GB | ~2.0 TB/s |
| On-chip SRAM per SM | 192 KB | ~19 TB/s |

IO-aware algorithms tile computation to fit in SRAM, avoiding repeated round-trips to HBM.

## FlashAttention (v1) Recap

FlashAttention (Dao et al., 2022) introduced tiled forward and backward passes with online softmax so that intermediate $N \times N$ matrices never leave SRAM. It split $Q$ and $K, V$ into blocks, computed partial softmax statistics $(m_i, \ell_i)$ incrementally, and rescaled outputs at each tile.

**Key shortcoming**: the forward pass parallelized only over batch size $B$ and number of heads $h$, leaving sequence parallelism unexploited. Additionally, each warp held a separate copy of key-value blocks and had to synchronize ("split-K") to combine partial outputs, wasting registers and HBM bandwidth.

## FlashAttention-2 Optimizations

### 3.1 Reducing Non-MatMul FLOPs

Modern GPU Tensor Cores execute matrix multiplications at 16× higher FLOP/s than scalar operations. The original FlashAttention rescaled the running output $O_i$ after every inner loop iteration, introducing extra multiplications. FlashAttention-2 keeps the output **unscaled** throughout and applies a single final rescaling:

```math
\begin{align}
  \tilde{O}_i &\leftarrow \tilde{O}_i + e^{s_{ij} - m_i^{\text{new}}} V_j \quad \text{(no per-block rescale)}
\end{align}
```

It also stores only the **logsumexp** $L_i = m_i + \log \ell_i$ rather than separate $m_i$ and $\ell_i$, halving the scalar bookkeeping operations.

### 3.2 Parallelism Over Sequence Length

The original implementation launched one thread block per $(batch, head)$ pair, so when $B \times h$ is small (e.g., long-context inference with batch=1) GPUs were under-utilized.

FlashAttention-2 additionally tiles $Q$ **across sequence length** and launches one thread block per $(batch, head, \text{query\_tile})$ triple. Each thread block independently processes its tile of queries against all of $K, V$, producing a tile of the output $O$.

> [!IMPORTANT]
> Choosing which matrix ($Q$ or $K,V$) to tile across the outer loop determines whether thread blocks can be scheduled independently. Tiling over $Q$ gives independent blocks; tiling over $K,V$ would require accumulation across blocks and introduce synchronization. FlashAttention-2 moves $Q$ to the outer loop compared to v1.

### 3.3 Warp-Level Work Partitioning

Within each thread block, work is distributed among 4 warps. FlashAttention-1 split both $Q$ and $K,V$ across warps (each warp computed a partial output), requiring an all-reduce synchronization inside the block after every tile.

FlashAttention-2 assigns different $Q$ rows to different warps while **sharing** the same $K, V$ blocks across warps. This eliminates the intra-block synchronization and reduces shared-memory read/write traffic.

| | FlashAttention-1 | FlashAttention-2 |
|---|---|---|
| $Q$ split across warps | No | Yes (different rows) |
| $K,V$ split across warps | Yes | No (shared by all warps) |
| Intra-block synchronization | Required | Eliminated |

## Forward Pass Algorithm (Pseudocode)

**Input:** $Q, K, V \in \mathbb{R}^{N \times d}$, tile sizes $B_r, B_c$  
**Output:** $O \in \mathbb{R}^{N \times d}$, logsumexp $L \in \mathbb{R}^N$

```
1. Split Q into tiles Q_1, …, Q_{T_r} of shape B_r × d
   Split K, V into tiles K_1, …, K_{T_c} and V_1, …, V_{T_c} of shape B_c × d
2. For each query tile i = 1 to T_r (parallelized across thread blocks):
     Initialize O_i = 0 ∈ R^{B_r × d}, ℓ_i = 0 ∈ R^{B_r}, m_i = -∞ ∈ R^{B_r}
     For each key-value tile j = 1 to T_c:
       Load K_j, V_j from HBM to SRAM
       Compute S_ij = Q_i K_j^T ∈ R^{B_r × B_c}    (on-chip matmul)
       m_i^new = max(m_i, rowmax(S_ij))
       P_ij = exp(S_ij - m_i^new)                   (pointwise, on-chip)
       ℓ_i^new = exp(m_i - m_i^new) * ℓ_i + rowsum(P_ij)
       O_i = diag(exp(m_i - m_i^new)) * O_i + P_ij V_j   (rescale + accumulate)
       m_i = m_i^new, ℓ_i = ℓ_i^new
     O_i = diag(1/ℓ_i) * O_i                        (final normalization)
     L_i = m_i + log(ℓ_i)                           (logsumexp for backward)
     Write O_i, L_i to HBM
```

> [!NOTE]
> The single rescaling `diag(1/ℓ_i)` applied at the end (rather than at every inner iteration as in v1) is the key change reducing non-matmul FLOPs.

## Backward Pass Algorithm

The backward pass recomputes attention weights on-chip from stored $Q, K, V, O, L$ rather than re-reading $N \times N$ attention maps. This keeps peak memory at $O(N)$ instead of $O(N^2)$.

**Key equations** for gradient computation (per tile):

```math
\begin{align}
  dV_j &= P_{ij}^\top dO_i \\
  dP_{ij} &= dO_i V_j^\top \\
  dS_{ij} &= P_{ij} \odot (dP_{ij} - D_i) \quad \text{where } D_i = \text{rowsum}(dO_i \odot O_i) \\
  dQ_i &\mathrel{+}= dS_{ij} K_j, \quad dK_j \mathrel{+}= dS_{ij}^\top Q_i
\end{align}
```

The scalar $D_i \in \mathbb{R}^{B_r}$ is loaded from HBM once per query tile and reused across all key-value tiles, replacing the per-tile softmax denominator computation needed in v1.

## Causal Masking

For autoregressive (causal) attention, blocks where all queries precede all keys are zeroed. Block-structured tiling allows these blocks to be **skipped entirely**, not merely masked. Since roughly half the blocks lie below the diagonal, causal masking yields a 1.7–1.8× speedup over causal-masked standard attention and ~1.1× over non-causal FlashAttention.

## Multi-Query and Grouped-Query Attention

FlashAttention-2 supports MQA (one $K,V$ head shared by all $Q$ heads) and GQA (one $K,V$ head shared by a group of $Q$ heads) by loading a single $K,V$ tile once and looping over the corresponding $Q$ tiles in the same thread block, amortizing HBM reads.

# Experiments

- **Dataset / Benchmark**: Synthetic attention benchmarks (forward and backward) sweeping sequence length from 512 to 16,384 on A100 SXM4 80GB PCIe; end-to-end GPT-3 style language model training on 8× A100
- **Hardware**: A100 80GB (primary), H100 SXM5 80GB (preliminary)
- **Precision**: FP16
- **Optimizer**: not specified for attention benchmarks; standard AdamW for end-to-end training

**Attention benchmark results (A100):**
| Metric | FlashAttention-1 | FlashAttention-2 | Standard Attention |
|---|---|---|---|
| Forward TFLOP/s | ~27 | ~50–73% of peak | Baseline |
| Backward TFLOP/s | ~20 | ~63% of peak | Baseline |
| Speedup vs FA-1 | 1× | 1.7–3.0× | — |
| Speedup vs standard | — | 3–10× | 1× |

**End-to-end GPT training throughput (TFLOPs/s per A100):**
| Model | Context | No FA | FA-1 | FA-2 |
|---|---|---|---|---|
| GPT-3 1.3B | 2k | 142 | 189 | 196 |
| GPT-3 1.3B | 8k | 72 | 170 | 220 |
| GPT-3 2.7B | 2k | 149 | 189 | 205 |
| GPT-3 2.7B | 8k | 80 | 175 | 225 |

225 TFLOPs/s represents 72% model FLOPs utilization (MFU), approaching what was previously considered the ceiling for GPU training efficiency.

# Comparison with Related Methods

| Method | Exactness | Memory | Sequence limit | Key mechanism |
|---|---|---|---|---|
| Standard Attention | Exact | $O(N^2)$ | ~4k (memory) | Dense matmul |
| FlashAttention-1 | Exact | $O(N)$ | ~16k | IO-aware tiling, online softmax |
| **FlashAttention-2** | **Exact** | **$O(N)$** | **~100k+** | FA-1 + better parallelism + warp partitioning |
| Sparse Attention (BigBird, Longformer) | Approximate | $O(N)$ | 16k–128k | Structured sparsity |
| Linear Attention (Performer, Linformer) | Approximate | $O(N)$ | Unbounded | Kernel/low-rank approximation |

FlashAttention-2 is the only exact method achieving competitive throughput at extended context lengths, making it preferable for tasks where approximation errors degrade quality (e.g., code generation, long-document understanding).

> [!TIP]
> The official implementation and benchmarking code is available at [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). The library also includes fused MLP kernels and other memory-efficient primitives for Transformer training.
