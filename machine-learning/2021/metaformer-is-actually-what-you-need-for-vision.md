# Meta Information

- URL: [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yu, W., Luo, M., Zhou, P., Si, C., Zhou, Y., Wang, X., Feng, J., & Yan, S. (2021). MetaFormer Is Actually What You Need for Vision. arXiv:2111.11418.

> [!CAUTION]
> NOTE comments are my personal understanding and may contain errors.

# 1. Introduction

This paper challenges the common assumption that attention mechanisms are the primary driver of Transformer success in computer vision. The authors propose that **the general Transformer architecture (MetaFormer) — not the specific token mixer — is the critical factor**. To support this, they introduce **PoolFormer**, which replaces self-attention with simple average pooling and still achieves competitive accuracy on ImageNet-1K while using fewer parameters than state-of-the-art vision Transformers and MLP-based models.

> [!NOTE]
> "Based on this observation, we hypothesize that the general architecture of the Transformers, instead of the specific token mixer module, is more essential to the model's performance."

This insight redirects research from designing increasingly complex attention mechanisms to understanding and improving the fundamental architectural components: residual connections, normalization, and feedforward projections.

# 2. MetaFormer Framework

## 2.1 Abstraction

MetaFormer is an abstraction of the Transformer architecture where the token mixer is treated as a **plug-in component** that can be any operation mixing information across tokens. Formally, given an input sequence of tokens $X \in \mathbb{R}^{n \times d}$ (where $n$ = number of tokens, $d$ = channel dimension), one MetaFormer block computes:

$$Y = \text{TokenMixer}(\text{Norm}(X)) + X$$
$$Z = \sigma(Y W_1) W_2 + Y$$

where:
- $\text{Norm}(\cdot)$: normalization layer (LayerNorm or modified form)
- $\text{TokenMixer}(\cdot)$: any operation that mixes information across the $n$ token dimension
- $W_1 \in \mathbb{R}^{d \times r}$, $W_2 \in \mathbb{R}^{r \times d}$: weight matrices of the Channel MLP with expansion ratio $r$
- $\sigma(\cdot)$: GELU activation

The four key components are:
| Component | Role |
|---|---|
| Input Embedding | Converts raw input (image patches) to token representations |
| Token Mixer | Mixes information spatially across tokens (flexible/swappable) |
| Channel MLP | Two-layer FFN with GELU; applies per-token transformation |
| Normalization | LayerNorm (or Modified LayerNorm) applied before each sub-block |

## 2.2 Supported Token Mixer Variants

The framework supports any token mixer, including:
- **Self-Attention** (original Transformer): computes pairwise dot-product attention, $O(n^2 d)$ complexity
- **Spatial MLP** (MLP-Mixer style): learns a weight matrix $W \in \mathbb{R}^{n \times n}$ over tokens, $O(n^2)$ learnable parameters
- **Average Pooling** (PoolFormer): computes the average over a local neighborhood, $O(1)$ learnable parameters
- **Identity Mapping**: no mixing at all (used in ablation studies)

# 3. PoolFormer Architecture

## 3.1 Pooling Token Mixer

The token mixer in PoolFormer is non-parametric **average pooling with subtraction of the center token** to avoid information leakage:

$$\text{TokenMixer}(T)[:, i, j] = \frac{1}{K^2} \sum_{p=-\lfloor K/2 \rfloor}^{\lfloor K/2 \rfloor} \sum_{q=-\lfloor K/2 \rfloor}^{\lfloor K/2 \rfloor} T[:, i+p, j+q] - T[:, i, j]$$

where:
- $T \in \mathbb{R}^{C \times H \times W}$: input feature map (channels × height × width)
- $K$: pooling kernel size (default $K = 3$)

This operation requires **zero learnable parameters** for the token mixing step itself and runs in $O(C \cdot H \cdot W)$ time.

> [!NOTE]
> The subtraction of the center token $T[:, i, j]$ is critical to avoid the model simply copying the input; it forces the mixer to encode *relative* neighborhood context.

## 3.2 Hierarchical Design

PoolFormer follows the hierarchical 4-stage design common to modern vision backbones (e.g., PVT, Swin Transformer):

| Stage | Output Resolution | Embedding Dim (S/M) |
|---|---|---|
| Stage 1 | $H/4 \times W/4$ | 64 / 96 |
| Stage 2 | $H/8 \times W/8$ | 128 / 192 |
| Stage 3 | $H/16 \times H/16$ | 320 / 384 |
| Stage 4 | $H/32 \times W/32$ | 512 / 768 |

Block distribution across stages: $[L/6, L/6, L/2, L/6]$ blocks, where $L$ is the total number of blocks. Patch embedding uses a stride-2 convolution between stages for downsampling.

## 3.3 Model Variants

| Model | Blocks $L$ | #Params | MACs |
|---|---|---|---|
| PoolFormer-S12 | 12 | 11.9M | 1.8G |
| PoolFormer-S24 | 24 | 21.4M | 3.4G |
| PoolFormer-S36 | 36 | 30.9M | 5.0G |
| PoolFormer-M36 | 36 | 56.9M | 8.8G |
| PoolFormer-M48 | 48 | 73.5M | 11.6G |

Input is a standard $224 \times 224$ RGB image; output is a class logit vector $\in \mathbb{R}^{1000}$ for ImageNet classification.

## 3.4 Modified LayerNorm (MLN)

Standard LayerNorm normalizes over the channel dimension. For 2D feature maps $X \in \mathbb{R}^{C \times H \times W}$, the authors use **Modified LayerNorm** that normalizes over both channels and spatial dimensions simultaneously:

$$\text{MLN}(X)_{c,h,w} = \frac{X_{c,h,w} - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_c + \beta_c$$

where $\mu$ and $\sigma^2$ are computed over the $(C, H, W)$ dimensions jointly, and $\gamma, \beta \in \mathbb{R}^C$ are learnable per-channel scale/shift parameters.

> [!NOTE]
> This modification avoids computing separate statistics for every spatial position, making it more efficient for image feature maps.

# 4. Differences from Similar Methods

| Method | Token Mixer | Complexity | Params in Mixer | Architecture |
|---|---|---|---|---|
| ViT (DeiT) | Self-Attention | $O(n^2 d)$ | $4d^2$ | Isotropic (single scale) |
| MLP-Mixer | Spatial MLP | $O(n^2)$ | $n^2$ | Isotropic |
| ResMLP | Channel MLP only | $O(nd)$ | $n \cdot d$ | Isotropic |
| Swin Transformer | Window Attention | $O(n \cdot w^2 d)$ | $4d^2$ | Hierarchical |
| PoolFormer (ours) | Avg Pooling | $O(n \cdot K^2)$ | 0 | Hierarchical |

Key distinctions:
- **vs. MLP-Mixer / ResMLP**: PoolFormer uses a hierarchical 4-stage structure (vs. flat isotropic), enabling better multi-scale feature extraction for dense prediction tasks.
- **vs. Swin Transformer**: Swin uses shifted window attention (requires explicit window partitioning logic); PoolFormer uses a single uniform pooling operation with no learnable parameters.
- **vs. DeiT/ViT**: DeiT uses quadratic attention ($O(n^2)$); PoolFormer uses $O(n)$ pooling without positional embeddings.

# 5. Experiments

## 5.1 ImageNet-1K Classification

- **Dataset**: ImageNet-1K — 1.28M training images, 50K validation images, 1,000 classes
- **Training**: 300 epochs, AdamW optimizer, batch size 4096, peak learning rate $4 \times 10^{-3}$ (linear scaling)
- **Augmentation**: RandAugment, MixUp, CutMix, CutOut, random erasing
- **Regularization**: Stochastic depth, LayerScale, label smoothing 0.1
- **Hardware**: Not explicitly stated (standard GPU cluster setup)

Key results (top-1 accuracy on ImageNet-1K val):
- PoolFormer-S12: **77.2%** (11.9M params, 1.8G MACs) — matches ResNet-50 (79.8%) with 40% fewer params
- PoolFormer-M36: **82.1%** — outperforms DeiT-B (81.8%, +0.3%) with 35% fewer params and 50% fewer MACs
- PoolFormer-M36: **82.1%** — outperforms ResMLP-B24 (81.0%, +1.1%) with 52% fewer params and 62% fewer MACs

## 5.2 Object Detection (COCO)

- **Dataset**: MS COCO 2017 — 118K train, 5K val images
- **Frameworks**: RetinaNet (one-stage) and Mask R-CNN (two-stage, includes instance segmentation)
- **Training**: 12-epoch "1×" schedule, AdamW

PoolFormer-S12 achieves 36.2 box AP with RetinaNet vs. ResNet-18's 31.8 AP, a +4.4 AP gain despite similar parameter counts.

## 5.3 Semantic Segmentation (ADE20K)

- **Dataset**: ADE20K — 20K training, 2K validation images, 150 semantic categories
- **Framework**: Semantic FPN head
- **Training**: 40K iterations, AdamW

PoolFormer-S12: 37.2 mIoU (vs. PVT-Tiny: 35.7 mIoU); PoolFormer-M48: 42.7 mIoU.

# 6. Ablation Studies

## 6.1 Token Mixer Impact

Replacing pooling with different mixers on the same MetaFormer architecture:

| Token Mixer | Top-1 Accuracy | Learnable Params |
|---|---|---|
| Identity (no mixing) | 74.3% | 0 |
| Global Random Matrix | 75.8% | $n^2$ (frozen) |
| Average Pooling | 78.1% | 0 |
| Depthwise Conv | 78.1% | $K^2 \cdot C$ |
| Self-Attention | ~81%+ | $4d^2$ |

Even identity mapping (no information mixing at all) achieves 74.3%, far higher than random baselines in prior vision MLP works. This supports the hypothesis that MetaFormer's structural components — not the mixer — drive performance.

## 6.2 Architectural Component Ablation

| Modification | Top-1 Accuracy |
|---|---|
| Baseline (PoolFormer-S12) | 77.2% |
| Remove residual connections | 0.1% (complete collapse) |
| Remove channel MLP | 5.7% |
| Replace MLN with standard LN | −0.3% |
| Replace GELU with ReLU | −0.1% |
| Change pooling size $K$ (3→7) | −0.1% |

Residual connections are **critical** — removing them causes complete training failure. The channel MLP is the second most important component.

## 6.3 Hybrid MetaFormer

Combining pooling in lower stages (1 and 2) with attention in higher stages (3 and 4):

- Hybrid model: **81.0%** top-1 accuracy, 16.5M params, 2.5G MACs
- Matches ResMLP-B24 efficiency with substantially fewer parameters and computation

> [!IMPORTANT]
> The hybrid result (81.0% at 16.5M params) demonstrates that the MetaFormer insight can be used **as a design principle**: use cheap token mixers in early stages (where spatial resolution is large and attention would be expensive), and expressive mixers in later stages.

# 7. Applicability

**Who**: Computer vision researchers designing backbone architectures for classification, detection, and segmentation.

**When**: When seeking to understand the source of Transformer-style model performance, or when designing efficient vision backbones.

**Where**: Image-level tasks (ImageNet), dense prediction tasks (COCO, ADE20K), and any domain where hierarchical feature extraction is beneficial.

The MetaFormer framework is particularly useful as a **baseline** for ablation studies — by swapping only the token mixer while keeping all other components fixed, researchers can fairly attribute performance differences to the mixer alone.
