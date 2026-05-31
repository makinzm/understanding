# Meta Information

- URL: [DiffusionBlocks: Blockwise Training for Generative Models via Score-Based Diffusion](https://arxiv.org/abs/2506.14202)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Shing, M., & Akiba, T. (2025). DiffusionBlocks: Blockwise Training for Generative Models via Score-Based Diffusion. arXiv:2506.14202. (Submitted to ICML)

# DiffusionBlocks: Blockwise Training for Generative Models via Score-Based Diffusion

DiffusionBlocks is a memory-efficient training framework for neural networks that interprets each transformer or ResNet block as performing a denoising step within a continuous-time score-based diffusion process. By decomposing the network into $B$ independently trainable blocks—each responsible for a specific noise-level band—gradient computation is localized per block, yielding a $B$-fold reduction in activation memory relative to end-to-end backpropagation. The framework is applicable to any architecture with explicit residual connections (ResNets, U-Nets, Transformers) and is validated on both image generation (CIFAR-10, ImageNet-256) and language modeling (LM1B).

## Background: Score-Based Diffusion

Score-based generative models learn to reverse a forward noising process. The forward process adds Gaussian noise to data $y \in \mathbb{R}^d$ to create a noisy sample $z_\sigma = y + \sigma\epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ and $\sigma$ is the noise level. The reverse process is governed by the probability flow ODE:

```math
\begin{align}
  \frac{dz_\sigma}{d\sigma} = -\sigma \nabla_z \log p_\sigma(z_\sigma)
\end{align}
```

The score function $\nabla_z \log p_\sigma(z_\sigma)$ is approximated by a denoiser network $D_\theta$:

```math
\begin{align}
  \nabla_z \log p_\sigma(z_\sigma) \approx \frac{D_\theta(z_\sigma, \sigma) - z_\sigma}{\sigma^2}
\end{align}
```

The denoiser is trained via the denoising score matching objective:

```math
\begin{align}
  \mathcal{L}(\theta) = \mathbb{E}\left[w(\sigma)\|D_\theta(z_\sigma, \sigma, x) - y\|_2^2\right]
\end{align}
```

where $w(\sigma)$ is a noise-level weighting function.

## Residual Networks as Euler Discretization of the ODE

The core theoretical insight is that residual connections implement Euler steps of the probability flow ODE. A single residual block computes:

```math
\begin{align}
  z_{\sigma_l} = z_{\sigma_{l-1}} + g_\theta(z_{\sigma_{l-1}})
\end{align}
```

This matches the Euler discretization:

```math
\begin{align}
  z_{\sigma_l} = z_{\sigma_{l-1}} + \frac{\Delta\sigma_l}{\sigma_{l-1}}\left(z_{\sigma_{l-1}} - D_\theta(z_{\sigma_{l-1}}, \sigma_{l-1})\right)
\end{align}
```

where $\Delta\sigma_l = \sigma_l - \sigma_{l-1}$ is the step size. This interpretation means each layer naturally corresponds to denoising at a particular noise level, making it possible to partition the network by noise level and train each partition independently.

## Block Partitioning

Given a network with $L$ layers split into $B$ blocks, each block $i$ is assigned a noise-level interval $[\sigma_i, \sigma_{i+1}]$ and trained with its own loss $\mathcal{L}(\theta_i)$, receiving no gradients from adjacent blocks.

### Equi-Probability Partitioning

A naïve uniform split of the noise range performs poorly because the log-normal noise distribution $p(\sigma)$ assigns unequal probability mass to equal-width intervals. DiffusionBlocks uses **equi-probability partitioning**: block boundaries are placed such that each block covers the same cumulative probability mass $1/B$ under $p(\sigma)$.

Block boundaries $\sigma_i$ are computed as:

```math
\begin{align}
  \sigma_i = \exp\!\left(P_\text{mean} + P_\text{std} \cdot \Phi^{-1}(p_i)\right)
\end{align}
```

where $\Phi^{-1}$ is the inverse standard normal CDF and the probability thresholds are:

```math
\begin{align}
  p_i = p_\text{min} + \frac{i}{B}(p_\text{max} - p_\text{min})
\end{align}
```

Here $p_\text{min} = \Phi\!\left(\frac{\log\sigma_\text{min} - P_\text{mean}}{P_\text{std}}\right)$ and similarly for $p_\text{max}$.

> [!IMPORTANT]
> This partitioning allocates more parameters to intermediate noise levels, where score estimation is hardest. Ablation shows equi-probability (FID 45.50) substantially outperforms uniform partitioning (FID 68.06) on CIFAR-10.

### Controlled Overlap

Without overlap, independent block training can create discontinuities at block boundaries during inference. DiffusionBlocks introduces a controlled overlap by expanding each block's training range by an overlap coefficient $\gamma$:

```math
\begin{align}
  [\sigma_i / \alpha,\ \sigma_{i+1} \cdot \alpha], \quad \alpha = \left(\frac{\sigma_{i+1}}{\sigma_i}\right)^\gamma
\end{align}
```

The optimal value found empirically is $\gamma = 0.1$. Larger $\gamma$ reduces independence between blocks (defeating the memory benefit) while smaller $\gamma$ reintroduces boundary artifacts.

## Algorithms

**Training (per iteration):**

1. Sample block index $i \sim \text{Uniform}(0, B-1)$
2. Sample noise $\sigma$ from block $i$'s expanded range $[\sigma_i/\alpha, \sigma_{i+1}\cdot\alpha]$
3. Construct noisy input: $z_\sigma = y + \sigma\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
4. Forward pass through block $i$ only: compute $D_{\theta_i}(z_\sigma, \sigma, x)$
5. Compute loss: $\mathcal{L}(\theta_i) = w(\sigma)\|D_{\theta_i}(z_\sigma, \sigma, x) - y\|_2^2$ (images) or cross-entropy (language modeling)
6. Backpropagate and update only $\theta_i$

**Inference:**

1. Start from $z_{\sigma_\text{max}} \sim \mathcal{N}(0, \sigma_\text{max}^2 I)$
2. Discretize noise schedule into $M$ steps
3. For each step $t$: identify block $i$ such that $\sigma_t \in [\sigma_i, \sigma_{i+1}]$; apply ODE solver using $D_{\theta_i}$
4. Return $z_{\sigma_\text{min}}$ as the generated sample

> [!NOTE]
> Language modeling inference requires $O(KM)$ forward passes for $K$ tokens and $M$ diffusion steps, creating multiplicative overhead compared to standard autoregressive generation.

## Input/Output Specifications

| Component | Input | Output |
|---|---|---|
| Block $i$ (image) | $z_\sigma \in \mathbb{R}^{H \times W \times C}$, $\sigma \in \mathbb{R}$ | denoised $\hat{y} \in \mathbb{R}^{H \times W \times C}$ |
| Block $i$ (language) | token embeddings $\in \mathbb{R}^{T \times d}$, $\sigma \in \mathbb{R}$ | token logits $\in \mathbb{R}^{T \times V}$ |
| Full network (inference) | $z_{\sigma_\text{max}} \in \mathbb{R}^{H \times W \times C}$ | generated image $\in \mathbb{R}^{H \times W \times C}$ |

where $H, W$ are spatial dimensions, $C$ channels, $T$ sequence length, $d$ hidden dimension, $V$ vocabulary size.

## Comparison with Related Methods

| Method | Memory Reduction | Gradient Communication | Architecture Constraint |
|---|---|---|---|
| End-to-End Backprop | $1\times$ (baseline) | Full network | None |
| Gradient Checkpointing | $\sim\sqrt{L}\times$ | Full network | None |
| Pipeline Parallelism | Linear in stages | Cross-stage | None |
| **DiffusionBlocks** | $B\times$ (exact) | None (per block) | Residual connections required |

Unlike gradient checkpointing, which trades memory for recomputation, DiffusionBlocks eliminates inter-block gradient flow entirely. Unlike pipeline parallelism, it requires no inter-device communication during training. The key constraint is that the architecture must support the ODE interpretation, i.e., it must have explicit residual connections.

> [!TIP]
> The framework is most beneficial when the number of blocks $B$ is 4–6 for a 12-layer network. Beyond this, performance degrades because each block has too few layers to learn its noise-level subproblem.

# Experiments

- **Datasets**: CIFAR-10 (32×32 images), ImageNet-256 (256×256 images), One Billion Words / LM1B (text)
- **Image architecture**: Diffusion Transformer (DiT-S/2 with 12 layers; DiT-L/2 with 24 layers), partitioned into 4 blocks
- **Language architecture**: Llama-style transformer, 12 layers, hidden dim $d=768$, 12 attention heads, context length 256 tokens, partitioned into 4 blocks
- **Optimizer**: AdamW, learning rate $1\times10^{-4}$ (images) / $3\times10^{-4}$ (language)
- **Batch size**: 512 (CIFAR-10), 1024 (ImageNet), 256 (LM1B)
- **Training duration**: 100 epochs (CIFAR-10/ImageNet), 10 epochs (LM1B)
- **Hardware**: Not specified
- **Evaluation metrics**: FID (image quality, lower is better), MAUVE score (language quality, higher is better)
- **Key results**:
  - CIFAR-10: DiffusionBlocks FID 41.39 vs End-to-End FID 41.87 (comparable)
  - ImageNet-256: DiffusionBlocks FID 15.55 vs End-to-End FID 16.62 (DiffusionBlocks better)
  - LM1B: DiffusionBlocks MAUVE 0.779 vs End-to-End MAUVE 0.595 (DiffusionBlocks substantially better)
  - Memory: 4× reduction with 4 blocks; inference: 3–4× faster generation due to parallel block evaluation
