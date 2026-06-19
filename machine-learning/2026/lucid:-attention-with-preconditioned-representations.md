# Meta Information

- URL: [LUCID: Attention with Preconditioned Representations](https://arxiv.org/abs/2602.10410)
- LICENSE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) / [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Duvvuri, S. S., Patel, N., Gupta, N., & Dhillon, I. S. (2026). LUCID: Attention with Preconditioned Representations. arXiv:2602.10410.

# LUCID: Attention with Preconditioned Representations

## Overview

LUCID (**L**earning with **U**n**c**orrelated Key **I**nnovations and **D**ecorelation) is a modification to standard softmax attention that addresses two fundamental limitations in Transformers for long-context scenarios:

1. **Attention noise**: Softmax attention diffuses probability mass to irrelevant tokens as context length grows, reducing retrieval precision.
2. **Learnability degradation**: Sharpening attention via temperature reduction causes vanishing gradients (the Jacobian collapses to zero), impairing learning.

LUCID resolves both problems by introducing a **preconditioner** derived from key-key similarities in a Reproducing Kernel Hilbert Space (RKHS), decoupling precision from temperature.

> [!IMPORTANT]
> LUCID is targeted at practitioners training decoder-only language models with long context windows (32K–128K tokens). The overhead is minimal (0–5.5% training, ~1.3% inference latency), making it a near drop-in replacement for softmax attention.

## Problem: Two Failure Modes of Softmax Attention

### Attention Noise in Long Contexts

Given queries $Q \in \mathbb{R}^{n \times d}$, keys $K \in \mathbb{R}^{n \times d}$, and values $V \in \mathbb{R}^{n \times d}$, standard softmax attention is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

As context length $n$ grows, correlated keys cause the softmax distribution to spread probability mass across many tokens rather than concentrating on the relevant one. This is the **attention noise** problem.

### Learnability vs. Precision Tradeoff

To sharpen attention, one can reduce the temperature $\tau$:

$$\text{softmax}\!\left(\frac{QK^\top}{\tau\sqrt{d}}\right)$$

However, lower $\tau$ causes the softmax Jacobian $\frac{\partial \text{softmax}}{\partial z}$ to approach zero, creating vanishing gradients. Conversely, raising $\tau$ preserves gradients but loses precision. Standard attention cannot satisfy both requirements simultaneously.

## LUCID Attention: Formulation

### Input/Output

- **Input**: $Q \in \mathbb{R}^{n \times d}$, $K \in \mathbb{R}^{n \times d}$, $V \in \mathbb{R}^{n \times d}$
- **Output**: $O \in \mathbb{R}^{n \times d}$, same shape as standard attention output

### Key Normalization (RMS Normalization)

Keys are RMS-normalized before constructing the preconditioner:

$$k_{i,\text{RN}} \leftarrow \sqrt{d} \cdot \frac{k_i}{\|k_i\|_2}$$

This ensures the unit diagonal of the preconditioner matrix and controls condition numbers for stable training at 128K tokens.

### Preconditioner Construction

The preconditioner $P \in \mathbb{R}^{n \times n}$ is computed from key-key similarities:

$$P = \left(M \odot \exp\!\left(\frac{K_{\text{RN}} K_{\text{RN}}^\top}{\sqrt{d}} - \sqrt{d}\right)\right)^{-1}$$

where $M$ is the causal (lower-triangular) mask and $\odot$ is element-wise multiplication.

> [!NOTE]
> The exponentiated key-key similarity matrix $\exp(K_{\text{RN}}K_{\text{RN}}^\top/\sqrt{d})$ corresponds to the kernel Gram matrix in RKHS with the exponential kernel. The subtraction of $\sqrt{d}$ ensures the diagonal entries are 1 after normalization.

### Full LUCID Attention

$$\text{LUCID}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + \hat{M}\right) \cdot P \cdot V$$

where $\hat{M}$ is the additive causal mask (large negative values for future positions).

**Computation order:**
1. Compute $K_{\text{RN}}$ (RMS-normalized keys)
2. Compute key-key similarity matrix $K_{\text{RN}} K_{\text{RN}}^\top / \sqrt{d}$
3. Construct lower-triangular preconditioner $P$ via exponentiation and inversion
4. Compute standard attention weights $A = \text{softmax}(QK^\top/\sqrt{d} + \hat{M})$
5. Apply preconditioner and values: $O = A \cdot P \cdot V$

## Theoretical Foundation: RKHS Objective

### Derivation from Gradient Descent

LUCID is derived from minimizing a quadratic objective in RKHS feature space. For each position $t$, define:

$$f_t(S) = \frac{1}{2} \|S\phi(k_t) - v_t\|^2$$

where $S$ is a linear map in feature space and $\phi(k_t)$ is the RKHS feature map of key $k_t$.

Gradient descent on this objective yields the **delta rule in infinite dimensions**:

$$S_{t} = S_{t-1} - \eta \frac{\partial f_t}{\partial S} = S_{t-1} - \eta (S_{t-1}\phi(k_t) - v_t)\phi(k_t)^\top$$

The recurrent update only fires when the current prediction $S_{t-1}\phi(k_t)$ differs from the target $v_t$, providing self-regulation.

> [!NOTE]
> Standard softmax attention is derived from a **linear** objective $f_t(S) = -\langle S\phi(k_t), v_t\rangle$, which always updates regardless of prediction quality. The quadratic objective in LUCID introduces a "correction" term that inherently decorrelates keys.

### Connection to DeltaNet

LUCID generalizes DeltaNet from finite-dimensional feature spaces to infinite-dimensional RKHS. In closed form over all context positions, the RKHS solution yields the preconditioned attention formula above.

## Learnability Theorem

**Theorem 1**: Under standard operating conditions (non-degenerate queries/keys), the gradient $\frac{\partial O}{\partial Q}$ for LUCID attention is non-zero:

$$\frac{\partial O}{\partial Q} = \frac{K^\top}{\sqrt{d}}\left(\text{diag}(a) - aa^\top\right)\left(M \odot \exp\!\left(\frac{K_{\text{RN}}K_{\text{RN}}^\top}{\sqrt{d}} - \sqrt{d}\right)\right)^{-1}$$

The preconditioner $P$ is invertible with non-trivial null space, so the Jacobian remains non-zero even when softmax saturates. This decoupling is the core innovation:

> [!IMPORTANT]
> "LUCID decouples these concerns: the preconditioner provides precision while standard temperature preserves gradient flow."

## Efficient Implementation

### Block-wise Memory-Efficient Algorithms

Because $P$ is a lower-triangular matrix, solving linear systems $PX = B$ can be done via **forward substitution** using the cuBLAS TRSM kernel, avoiding explicit matrix inversion.

Three algorithms provide memory-efficient forward and backward passes:

| Algorithm | Purpose |
|-----------|---------|
| Algorithm 2 | Block-wise forward pass (chunked over sequence length) |
| Algorithm 3 | Block-wise backward pass for $Q$, $K$, $V$ gradients |
| Algorithm 4 | Memory-efficient preconditioner backward |
| Algorithm 5 | LUCID-PaTH: Integration with PaTH positional encoding |

### Causal Structure Exploitation

The lower-triangular structure of $P$ (due to causal masking) allows:
- Forward substitution $O(n^2 d)$ per head instead of inversion $O(n^3)$
- Batched TRSM operations via cuBLAS for GPU efficiency

### Training and Inference Overhead

| Model size | Training overhead | Inference overhead (32K ctx) |
|------------|------------------|-----------------------------|
| Small | ~5.5% | ~1.3% |
| Large (~1B) | ~0% | ~1.3% |

## LUCID-PaTH Variant

LUCID-PaTH combines LUCID's key decorrelation with **PaTH** (Positional Attention with Tiled Hadamard) positional encoding for length extrapolation beyond training context. This variant maintains 0.21–0.25 accuracy on long-context retrieval tasks across 32K–128K contexts, where standard attention drops near zero.

## Experiments

### Datasets

- **BABILong**: Multi-needle retrieval benchmark, contexts up to 128K tokens
- **RULER**: Multi-needle retrieval benchmark for long-context LLMs
- **LongBench**: Long-context understanding benchmark (HotpotQA, QMSum, MultiFieldQA-en, PassageRetrieval)
- **SCROLLS**: Long document summarization and QA benchmark (GovReport, SummScreenFD, QMSum, Qasper, NarrativeQA, QuALITY, ContractNLI)
- **Synthetic two-phase task**: Self-retrieval then cumulative averaging, used to isolate learnability vs. retrieval tradeoffs

### Baselines

- Standard softmax attention (same architecture, compute-matched)
- DeltaNet (linear attention variant)
- GLA (Gated Linear Attention)
- GSA (Gated Slot Attention)
- PaTH Attention (preconditioned positional encoding only)

### Key Results

**Long-context retrieval (BABILong, RULER):**
- BABILong multi-needle: up to **18% improvement** over standard attention
- RULER benchmarks: up to **14% improvement**

**Attention noise metric (hitrate):**
- Standard attention hitrate: 0.1817
- LUCID hitrate: 0.2845 (**56.6% relative improvement**)

**Long-context understanding:**
- LongBench HotpotQA F1: 0.086 (best among compared methods)
- LongBench QMSum ROUGE-L: 12.60 (best among compared methods)

**Synthetic learnability test:**
- Phase 1 (self-retrieval): Standard attention reduces Jacobian magnitude ~1000× to achieve sharpness; LUCID achieves sharpness without gradient collapse
- Phase 2 (cumulative averaging): Standard attention fails to adapt due to vanishing gradients; LUCID rapidly learns new patterns

### Hardware

- Approximately 1 billion parameter language models trained up to 128K token contexts
- GPU acceleration with cuBLAS TRSM for triangular solves

## Comparison with Similar Algorithms

| Method | Complexity | Key Mechanism | Long-context quality |
|--------|-----------|---------------|---------------------|
| Standard Softmax | $O(n^2 d)$ | Temperature scaling | Degrades (noise + vanishing gradients) |
| DeltaNet | $O(nd^2)$ | Finite-dim delta rule | Better than softmax linear, limited capacity |
| GLA / GSA | $O(nd^2)$ | Gated linear recurrence | Competitive but lower than quadratic |
| PaTH Attention | $O(n^2 d)$ | Positional preconditioner | Extrapolation, but not key decorrelation |
| **LUCID** | $O(n^2 d)$ | RKHS key decorrelation | Best on retrieval, maintains learnability |
| LUCID-PaTH | $O(n^2 d)$ | RKHS + positional precond. | Best on length extrapolation tasks |

> [!NOTE]
> LUCID maintains $O(n^2 d)$ complexity (same as standard attention), unlike linear attention variants that trade quality for $O(nd^2)$ complexity. The preconditioner computation is the same order as the standard attention kernel.

## Limitations

- **Bidirectional settings**: The causal lower-triangular structure of $P$ is lost in encoder-style (bidirectional) models (e.g., diffusion transformers). Without triangularity, solving $Px = b$ is $O(n^3)$, making the approach expensive. The authors identify this as an important direction for future work.
- **Condition number growth**: Empirical measurements show that the condition number $\kappa$ of $P$ grows with sequence length $n$, which could cause numerical instability at extreme lengths. RMS normalization mitigates this but does not eliminate the growth.

## Applicability

- **Who**: ML practitioners and researchers building or fine-tuning decoder-only Transformers (language models, code models).
- **When**: When deploying models at long context lengths (16K–128K tokens) where attention noise and gradient degradation become problematic.
- **Where**: Autoregressive generation tasks (language modeling, retrieval-augmented generation, long-document QA) where causal attention is used. Not directly applicable to bidirectional encoders without additional research.
