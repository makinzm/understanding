# Meta Information

- URL: [FlashOptim: Optimizers for Memory Efficient Training](https://arxiv.org/abs/2602.23349)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Gonzalez Ortiz, J. J., Gupta, A., Renard, C., & Blalock, D. (2026). FlashOptim: Optimizers for Memory Efficient Training. arXiv:2602.23349.

# FlashOptim: Optimizers for Memory Efficient Training

FlashOptim is a suite of memory-reduction techniques targeting the optimizer state, master weights, and gradient buffers used during large neural network training. The primary motivation is that standard AdamW training requires 16 bytes per parameter (4 bytes each for master weights, gradients, momentum, and variance), whereas FlashOptim reduces this to 7 bytes—a >50% reduction—without degrading model quality. This makes FlashOptim relevant for practitioners fine-tuning or pre-training large language models on memory-constrained GPU clusters.

## Memory Layout in Mixed-Precision Training

During standard BF16 mixed-precision training with AdamW, per-parameter memory is allocated as follows:

| Component | Dtype | Bytes |
|---|---|---|
| Master weights | FP32 | 4 |
| Gradients | BF16 | 2 |
| Optimizer momentum ($m_t$) | FP32 | 4 |
| Optimizer variance ($v_t$) | FP32 | 4 |
| **Total (reference)** | | **14–16** |

FlashOptim targets each of these components individually:

| Component | FlashOptim | Bytes |
|---|---|---|
| Master weights | BF16 + INT8 split | 3 |
| Gradients | BF16 | 2 |
| Optimizer momentum | INT8 quantized | 1 |
| Optimizer variance | UINT8 quantized | 1 |
| **Total (FlashOptim)** | | **7** |

## Technique 1: ULP-Normalized Weight Splitting

### Background

Full-precision (FP32) master weights are stored to retain gradient accumulation fidelity in BF16 training. The naive approach—storing a full FP32 copy—costs 4 bytes/parameter. A common prior technique is to split FP32 into BF16 (upper bits) + BF16 (lower bits), requiring 4 bytes but with suboptimal error structure.

### ULP-Based Error Correction

FlashOptim exploits the structure of floating-point rounding. For any FP32 value $\theta$ and its BF16 approximation $\hat{\theta}$, the rounding error $e = \theta - \hat{\theta}$ satisfies:

```math
\begin{align}
  e \in \left[-\frac{u}{2},\ \frac{u}{2}\right], \quad u = \text{ULP}(\hat{\theta})
\end{align}
```

where $\text{ULP}(\hat{\theta})$ is the unit in the last place of the BF16 value. Because $u$ is known from $\hat{\theta}$, only the normalized error $e / (u/2) \in [-1, 1]$ needs to be stored. This is quantized to a $b$-bit integer:

```math
\begin{align}
  q = \text{clip}\!\left(\text{round}\!\left(\frac{e}{u/2} \cdot (2^{b-1} - 1)\right),\, -2^{b-1},\, 2^{b-1}-1\right)
\end{align}
```

With $b = 8$ (INT8), the scheme stores BF16 + INT8 = 3 bytes per parameter, providing ~24-bit effective precision. On an exhaustive sweep of all finite FP32 values, bitwise perfect reconstruction occurs for 99.92% of values, with a worst-case relative error below $10^{-6}$.

> [!NOTE]
> The key insight is that normalizing the residual by the local ULP magnitude eliminates the wide dynamic range of raw residuals, making 8-bit quantization sufficient.

## Technique 2: Companded Optimizer State Quantization

### Motivation

Momentum $m_t$ and variance $v_t$ in AdamW have non-uniform distributions that are poorly matched to linear quantization at 8 bits. Linear quantization of $v_t$ causes training to diverge entirely because the heavy-tailed distribution wastes most quantization bins on large values.

### Companding Functions

Companding applies a nonlinear transform before quantization and its inverse after dequantization, redistributing probability mass for better utilization of quantization bins.

**Momentum** ($m_t \in \mathbb{R}$, unbounded, symmetric): A softsign-like function compresses the unbounded range to $(-1, 1)$:

```math
\begin{align}
  \varphi_m(x) = \frac{2x}{1 + |x|}
\end{align}
```

The normalized value is then linearly quantized to INT8 in $[-1, 1]$.

**Variance** ($v_t \geq 0$, heavy-tailed): A square-root transform reduces skewness before normalization:

```math
\begin{align}
  \varphi_v(x) = \sqrt{x}
\end{align}
```

The transformed values are normalized by the group maximum and quantized to UINT8 in $[0, 1]$.

### Group-Wise Quantization

Both states use group-wise quantization with group size $G = 32$, storing one FP16 scale per group. This keeps the scale overhead to $16 / 32 = 0.5$ bits per parameter.

**Momentum quantization algorithm:**

```
Input:  momentum tensor m ∈ ℝ^N, group size G=32
Output: quantized INT8 tensor q_m, scale s_m ∈ ℝ^(N/G)

for each group g of size G:
    m̃[g] = φ_m(m[g])              # apply softsign compander
    s_m[g] = absmax(m̃[g])
    q_m[g] = round(m̃[g] / s_m[g] * 127)   # quantize to INT8
```

**Variance quantization algorithm:**

```
Input:  variance tensor v ∈ ℝ^N (v ≥ 0), group size G=32
Output: quantized UINT8 tensor q_v, scale s_v ∈ ℝ^(N/G)

for each group g of size G:
    ṽ[g] = φ_v(v[g])              # apply sqrt compander
    s_v[g] = max(ṽ[g])
    q_v[g] = round(ṽ[g] / s_v[g] * 255)    # quantize to UINT8
```

**Dequantization** inverts each step: undo the INT→float scaling, undo the compander via $\varphi_m^{-1}(y) = y / (1 - |y|)$ and $\varphi_v^{-1}(y) = y^2$, then restore the group scale.

## Technique 3: Fused Triton Kernels

All compression and quantization operations are fused into single Triton CUDA kernels to avoid materializing intermediate tensors. The kernels implement the compress-update-decompress cycle for each optimizer step and incur negligible throughput overhead versus a standard BF16 optimizer step.

## Supported Optimizers

FlashOptim provides three drop-in optimizer replacements:

| Optimizer | Reference | FlashOptim variant |
|---|---|---|
| SGD with momentum | `torch.optim.SGD` | FlashSGD |
| AdamW | `torch.optim.AdamW` | FlashAdamW |
| Lion | — | FlashLion |

All variants are compatible with PyTorch FSDP sharding, activation checkpointing, and per-layer gradient release (when gradient accumulation is disabled).

## Comparison with Related Methods

| Method | Momentum | Variance | Master Wt | Total (bytes/param) |
|---|---|---|---|---|
| BF16 + FP32 Adam | FP32 (4) | FP32 (4) | FP32 (4) | 16 |
| BitsAndBytes (8-bit Adam) | INT8 | INT8 | FP32 (4) | 10 |
| Modular adaptive optimizer | FP32 | FP32 | BF16×2 (4) | 14 |
| **FlashAdamW** | INT8 + scale | UINT8 + scale | BF16+INT8 | **7** |

The key differences from prior 8-bit optimizers (e.g., BitsAndBytes) are:
1. ULP-normalized weight splitting saves 1 byte on master weights that prior methods leave at FP32.
2. Companding with domain-specific transforms (softsign for momentum, sqrt for variance) directly models the statistical structure of each state, unlike dynamic exponent or block-wise methods.

# Experiments

- **Dataset (image classification)**: ImageNet-1K; ResNet-50 trained to convergence
- **Dataset (LLM pre-training)**: C4; GPT-2 (124M parameters) trained from scratch
- **Dataset (LLM fine-tuning)**: GSM8K math reasoning benchmark; Llama-3.1-8B (8B parameters) fine-tuned
- **Hardware**: NVIDIA H100 GPUs, PyTorch 2.8, CUDA 12.8
- **Optimizer**: Identical hyperparameters for reference and FlashOptim variants; reference optimizers reimplemented with matching Triton kernels for fair comparison; 3 random seeds each

**Key results:**
- FlashAdamW matches reference AdamW loss trajectories on GPT-2 124M pre-training with no measurable degradation.
- On Llama-3.1-8B fine-tuning, FlashAdamW achieves identical GSM8K accuracy as reference AdamW.
- Peak GPU memory for Llama-3.1-8B fine-tuning reduces from 175 GiB to 113 GiB (36% reduction).
- Optimizer step throughput is unchanged compared to a Triton-kernel reference implementation.
- Checkpoint size for a 7B-class model drops from ~84 GB (FP32 Adam states + weights) to ~35 GB.
