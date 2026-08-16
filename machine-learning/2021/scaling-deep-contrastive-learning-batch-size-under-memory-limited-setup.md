# Meta Information

- URL: [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Gao, L., Zhang, Y., Han, J., & Callan, J. (2021). Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup. Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-2021).

# Overview

Contrastive learning requires large batches to obtain high-quality representations because each example's loss depends on all other examples in the batch as in-batch negatives. Fitting large batches into GPU memory is the primary bottleneck: standard gradient accumulation fails because splitting a batch into smaller sub-batches reduces the number of in-batch negatives per step, which does not emulate large-batch training.

This paper introduces **Gradient Cache**, a technique that decouples gradient computation into two independent components to enable arbitrarily large effective batch sizes on memory-constrained hardware. A single consumer GPU (RTX 2080ti) can match or exceed results previously requiring 8 V100 GPUs.

**Who, when, where:**
- Researchers with limited GPU resources who need large-batch contrastive learning (e.g., dense retrieval, sentence representation learning)
- Any setting where in-batch negatives are used and batch size is constrained by GPU memory

# Background: In-Batch Negatives and the Batch-Size Problem

Contrastive learning with in-batch negatives defines a loss over pairs $(s_i, t_i)$ from a batch of size $N$. For a query $s_i \in \mathbb{R}^d$ and a positive passage $t_i \in \mathbb{R}^d$, the InfoNCE-style loss is:

```math
\begin{align}
  \mathcal{L} = -\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(f(s_i), g(t_i)))}{\sum_{j=1}^{N} \exp(\text{sim}(f(s_i), g(t_j)))}
\end{align}
```

where $f, g$ are encoders (e.g., BERT) and $\text{sim}(\cdot, \cdot)$ is dot-product similarity. Because the denominator sums over all $j \in \{1, \ldots, N\}$, the loss for any example $i$ depends on representations of all other examples. Backpropagation therefore requires all representations and their encoder computation graphs to be held in GPU memory simultaneously, making memory cost linear in $N$.

Simple gradient accumulation over sub-batches of size $M < N$ does not help: each step only sees $M$ negatives, not $N$, so the gradients are not equivalent.

# Gradient Cache: Core Idea

## Mathematical Decomposition (Section 3.2)

The chain rule decomposes the gradient of the loss with respect to encoder parameters $\Theta$ as:

```math
\begin{align}
  \frac{\partial \mathcal{L}}{\partial \Theta} = \sum_{i=1}^{N} \frac{\partial \mathcal{L}}{\partial f(s_i)} \cdot \frac{\partial f(s_i)}{\partial \Theta}
\end{align}
```

**Key observation:** the two factors have different dependencies:

| Factor | Depends on | Memory implication |
|---|---|---|
| $\mathbf{u}_i = \partial \mathcal{L} / \partial f(s_i) \in \mathbb{R}^d$ | All representations (not encoder weights) | Computed once, stored as small vectors |
| $\partial f(s_i) / \partial \Theta$ | Only example $s_i$ and $\Theta$ (not the batch) | Computed per sub-batch without storing others |

This means: once $\mathbf{u}_i$ is known, the encoder gradient for example $i$ can be computed in isolation, sub-batch by sub-batch. The encoder does not need to be in a computation graph during the full-batch forward pass.

## Four-Step Algorithm

**Input:** Batch $\mathcal{S} = \{s_i\}_{i=1}^N$, $\mathcal{T} = \{t_i\}_{i=1}^N$; sub-batch size $M$; encoders $f, g$ with parameters $\Theta$

```
Step 1 — Graph-less forward pass:
  For all i in [1..N]:
    h_i = f(s_i)     # encode without building computation graph (no_grad)
    v_i = g(t_i)     # same
  Cache H = {h_i}, V = {v_i}

Step 2 — Compute representation gradients:
  Compute full-batch contrastive loss L using H and V (no encoder in graph)
  Backpropagate L through similarity computation:
    u_i = ∂L/∂h_i  for all i      # shape: R^d
    w_i = ∂L/∂v_i  for all i      # shape: R^d
  Store gradient cache U = {u_i}, W = {w_i}

Step 3 — Sub-batch gradient accumulation:
  For k = 0, M, 2M, ..., N-M:
    sub_S = S[k : k+M]
    For i in sub_S:
      h_i = f(s_i)                 # with computation graph this time
      backprop: h_i * u_i → Θ     # use cached u_i as upstream gradient
      accumulate ∂Θ
    (same for T sub-batches using w_i)

Step 4 — Optimizer step:
  Apply accumulated gradients to Θ
```

**Memory cost:** The gradient cache stores $2Nd$ floats (representations) plus $2Nd$ floats (their gradients), totaling $4Nd$ values. For $N=512, d=768$: ~1.6M floats ≈ 6 MB, negligible compared to model parameters (~440 MB for BERT-base). Encoder memory during Step 3 depends only on $M$, not $N$.

> [!IMPORTANT]
> The gradient update produced by Gradient Cache is mathematically **identical** to training with the full batch $N$ in a single forward/backward pass. This is not an approximation—it is exact.

## Multi-GPU Extension (Section 3.4)

After Step 1, an `all-gather` operation synchronizes all representations across GPUs. Each GPU then computes representation gradients using the global batch but only accumulates encoder gradients for its local examples in Step 3. No additional communication is needed during Step 3; standard gradient reduction during optimization handles parameter synchronization.

## Extension to Deep Distance Functions (Section 5)

When similarity is parameterized as $\Phi(f(s_i), g(t_j))$ (e.g., a learned cross-attention function), an additional **Distance Gradient Cache** is introduced. It stores $\partial \mathcal{L} / \partial d_{ij}$ for all pairs $(i,j)$, allowing $\Phi$'s parameters to be updated in a third pass without loading all representations simultaneously. The decomposition extends to three stages: encoder → distance function → loss.

# Comparison with Related Methods

| Method | Memory for encoder | Negatives per step | Gradient equivalence |
|---|---|---|---|
| Full-batch | $O(N)$ per step | $N$ | Exact |
| Gradient accumulation | $O(M)$ per step | $M \ll N$ | No (different loss) |
| Gradient checkpointing | $O(\sqrt{N})$ per step | $N$ | Exact, but slow |
| **Gradient Cache (ours)** | $O(M)$ per step | $N$ (global cache) | **Exact** |

Gradient checkpointing recomputes activations to save memory but does not reduce memory proportional to batch size for the full-batch contrastive loss. In experiments, checkpointing scaled only to batch size 64 with 2× overhead versus accumulation.

# Experiments

- **Dataset:** Natural Questions (NQ) open-domain QA dataset, used for dense passage retrieval evaluation
- **Model:** Dense Passage Retriever (DPR) — dual-encoder with BERT-base encoders for queries and passages
- **Evaluation metric:** Top-$k$ retrieval recall ($k \in \{5, 20, 100\}$)
- **Hardware:** Single NVIDIA RTX 2080ti (11 GB) for single-GPU experiments; reference DPR used 8 V100 GPUs
- **Batch configuration:** Effective batch size 128, sub-batch sizes 16 (queries) / 8 (passages)

**Key results:**

| Method | GPUs | Batch size | Top-5 | Top-20 | Top-100 |
|---|---|---|---|---|---|
| DPR (original) | 8 × V100 | 128 | — | 78.4 | 85.4 |
| Sequential (max fit) | 1 × 2080ti | small | 59.3 | 71.9 | 80.9 |
| Gradient accumulation | 1 × 2080ti | 128 (M-sized steps) | 64.3 | 77.2 | 84.9 |
| **Gradient Cache** | 1 × 2080ti | 128 | **68.6** | **79.3** | **86.0** |
| Gradient Cache | 1 × 2080ti | 512 | 68.3 | 79.9 | 86.6 |

Gradient Cache on a single consumer GPU **exceeds** the original DPR numbers from 8 V100s. Larger batches ($N=512$) further improve recall, demonstrating that the method successfully scales batch size beyond hardware limits.

**Training speed:** The method adds ~20% overhead relative to standard training (for the graph-less pre-computation pass). Training time on one RTX 2080ti: ~31 hours, versus ~24 hours on 8 V100s for the original DPR setup. The method maintains near-linear throughput scaling with batch size, whereas gradient accumulation plateaus due to sub-batch negative limitations.

> [!NOTE]
> "The gradient cache technique produces exact same gradient update as training with large batch." — Section 3.3

# Implementation

- GradCache library: [github.com/luyug/GradCache](https://github.com/luyug/GradCache)
- DPR-specific code: [github.com/luyug/GC-DPR](https://github.com/luyug/GC-DPR)

The library is designed to wrap existing PyTorch dual-encoder training loops with minimal code changes.
