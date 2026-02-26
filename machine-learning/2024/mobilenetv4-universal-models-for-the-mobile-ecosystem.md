# Meta Information

- URL: [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Qin, D., Leichner, C., Delakis, M., Fornoni, M., Luo, S., Yang, F., Wang, W., Banbury, C., Ye, C., Akin, B., Aggarwal, V., Zhu, T., Moro, D., & Howard, A. (2024). MobileNetV4 -- Universal Models for the Mobile Ecosystem. arXiv:2404.10518.

# MobileNetV4 -- Universal Models for the Mobile Ecosystem

MobileNetV4 (MNv4) is a family of neural network architectures designed to achieve Pareto-optimal efficiency on diverse mobile hardware (CPUs, EdgeTPUs, DSPs, GPUs, Apple Neural Engine) without platform-specific tuning. The core insight is that a single model design can simultaneously minimize latency across heterogeneous accelerators by balancing compute and memory bandwidth using the Roofline Model.

**Applicability**: Researchers and practitioners deploying computer vision models on mobile or edge devices (smartphones, IoT sensors, embedded systems) who need accuracy/latency tradeoffs that generalize across hardware vendors.

## Background: Roofline Model for Hardware-Independent Efficiency

The Roofline Model characterizes the computational bottleneck of each layer as either compute-bound or memory-bandwidth-bound. For layer $i$:

```math
\begin{align}
  \text{LayerTime}_i = \max\left(\frac{\text{MACs}_i}{\text{PeakMACs}}, \frac{\text{WeightBytes}_i + \text{ActivationBytes}_i}{\text{PeakMemBW}}\right)
\end{align}
```

The **ridge point** $r$ (in MACs/byte) is the minimum operational intensity at which peak compute is achievable. Hardware with $r \approx 0$ is purely memory-bound; hardware with $r \approx 500$ is heavily compute-bound. The total model latency is:

```math
\begin{align}
  \text{ModelTime} = \sum_i \text{LayerTime}_i
\end{align}
```

By sweeping ridge points from 0 to 500 MACs/byte, the authors simulate the full spectrum of mobile hardware characteristics without needing to measure on every device during design. This yields a **hardware-independent Pareto efficiency** criterion.

## Universal Inverted Bottleneck (UIB)

The UIB block is the primary building block of MNv4. It unifies four existing block designs by introducing two optional depthwise convolutions (DW1 and DW2) that can each be independently present or absent:

| Configuration | DW1 | DW2 | Description |
|---------------|-----|-----|-------------|
| Inverted Bottleneck (IB) | absent | present | Spatial mixing on expanded features (MobileNetV2 style) |
| ConvNext | present | absent | Spatial mixing with larger kernel before expansion |
| ExtraDW | present | present | Both depthwise ops; cheap depth and receptive field increase (new) |
| FFN | absent | absent | Two 1×1 pointwise convolutions only |

**Input/Output**: UIB takes a feature map $x \in \mathbb{R}^{H \times W \times C_{in}}$ and outputs $y \in \mathbb{R}^{H' \times W' \times C_{out}}$, where $H', W'$ depend on stride settings (1 or 2).

**Computation order** for the ExtraDW variant (most general form):
1. Optional DW1: $z_1 = \text{DWConv}(x)$, $z_1 \in \mathbb{R}^{H \times W \times C_{in}}$
2. Pointwise expansion: $z_2 = \text{PW}(z_1)$, $z_2 \in \mathbb{R}^{H \times W \times C_{expand}}$ where $C_{expand} = 4 \cdot C_{in}$
3. Optional DW2: $z_3 = \text{DWConv}(z_2)$, $z_3 \in \mathbb{R}^{H' \times W' \times C_{expand}}$
4. Pointwise projection: $y = \text{PW}(z_3)$, $y \in \mathbb{R}^{H' \times W' \times C_{out}}$

A residual connection is added when $H = H'$ and $C_{in} = C_{out}$.

> [!NOTE]
> "The UIB block's design allows the NAS to select the most efficient spatial mixing strategy per layer, rather than committing to a single block type for the entire network."

## Mobile MQA (Multi-Query Attention)

Standard Multi-Head Self-Attention (MHSA) with $h$ heads projects queries, keys, and values independently:
- $Q \in \mathbb{R}^{N \times h \times d_k}$, $K \in \mathbb{R}^{N \times h \times d_k}$, $V \in \mathbb{R}^{N \times h \times d_v}$

**Multi-Query Attention (MQA)** shares a single key and value across all heads:
- $Q \in \mathbb{R}^{N \times h \times d_k}$, $K \in \mathbb{R}^{N \times 1 \times d_k}$, $V \in \mathbb{R}^{N \times 1 \times d_v}$

This reduces KV memory by a factor of $h$ (number of heads), which is the dominant bottleneck on memory-bandwidth-limited accelerators (e.g., EdgeTPUs).

**Asymmetric spatial downsampling**: Before computing attention, keys and values are spatially downsampled using a stride-2 depthwise convolution, reducing the sequence length from $N$ to $N/4$ (for 2D spatial downsampling). This adds a further 20% efficiency gain with only −0.06% accuracy drop.

**Mobile MQA efficiency vs. MHSA** (measured on EdgeTPU):

| Metric | MHSA | Mobile MQA | Change |
|--------|------|------------|--------|
| Latency | baseline | −39% | |
| MACs | baseline | −28.6% | |
| Top-1 Accuracy | baseline | −0.03% | |

> [!TIP]
> Multi-Query Attention was originally proposed by Shazeer (2019) for NLP; MNv4 adapts it for spatial feature maps in vision by treating spatial positions as tokens.

## Refined Neural Architecture Search (NAS)

### Two-Stage Search Strategy

**Stage 1 – Coarse-grained search**: Fix depthwise kernel sizes to 3×3 and expansion factor to 4 in all UIB blocks. Search only over filter sizes per stage. This reduces the search space while finding the dominant architectural hyperparameters.

**Stage 2 – Fine-grained search**: Fix filter sizes from Stage 1. Search over the UIB variant per layer (IB, ConvNext, ExtraDW, FFN) by searching the presence and kernel size (3×3 or 5×5) of DW1 and DW2 independently.

**Improvement from two-stage vs. one-stage search**:
- +0.22% ImageNet validation accuracy
- −4.68% Pixel 6 EdgeTPU latency

### Offline Distillation During Search

Rather than training candidate models from scratch, the NAS uses a pre-trained JFT-300M teacher to distill into each candidate. This reduces sensitivity to training hyperparameters and produces more reliable latency/accuracy estimates during search.

## Model Variants

MNv4 is released in two series — Conv (convolutions only) and Hybrid (convolutions + Mobile MQA) — across three sizes:

| Model | Top-1 (ImageNet-1K) | MACs | Params | Pixel 6 CPU Latency |
|-------|---------------------|------|--------|---------------------|
| MNv4-Conv-S | 73.8% | 0.2G | 3.8M | 2.4ms |
| MNv4-Conv-M | 79.9% | 1.0G | 9.2M | 11.4ms |
| MNv4-Conv-L | 82.9% | 5.9G | 31M | 59.9ms |
| MNv4-Hybrid-M | 80.7% | 1.2G | 10.5M | 14.3ms |
| MNv4-Hybrid-L | 83.4% | 7.2G | 35.9M | 87.6ms |

**Network structure overview** (shared across variants):
1. Conv2D stem (3×3, stride 2) — fixed
2. FusedIB blocks — fixed (stages 1–2)
3. UIB blocks with NAS-selected configurations — searched (stages 3–6)
4. Mobile MQA blocks in Hybrid variants — inserted in later stages
5. Pointwise head with global average pooling — fixed

## Enhanced Knowledge Distillation Recipe

For high-accuracy training, MNv4 uses a three-dataset distillation mixture with dynamically sampled batches:

| Dataset | Augmentation | Size | Purpose |
|---------|-------------|------|---------|
| $\mathcal{D}_1$ | Inception Crop + RandAugment | 500× ImageNet-1K replicas | Standard supervised diversity |
| $\mathcal{D}_2$ | Inception Crop + extreme Mixup ($\alpha=1$) | 1000× ImageNet-1K replicas | Label smoothing via interpolation |
| $\mathcal{D}_3$ | Standard crop + class-balanced sampling | 130M JFT-300M images (10× replicated) | Long-tail generalization |

**Teacher model**: EfficientNet-L2 (pre-trained on JFT-300M), achieving 90.4% ImageNet top-1 accuracy.

**Training**: 2000 epochs with cosine decay; soft cross-entropy loss against teacher logits.

**Results**:
- MNv4-Conv-L student: 85.9% accuracy — only 1.6% below EfficientNet-L2 teacher, despite using 15× fewer parameters and 48× fewer MACs
- MNv4-Hybrid-L with JFT pre-training: 87.0% top-1 accuracy at 3.8ms on Pixel 8 EdgeTPU

> [!IMPORTANT]
> The JFT-300M dataset used in distillation is a proprietary Google dataset and is not publicly available. The trained MNv4 models themselves are open-sourced.

## Comparison with Similar Architectures

| Model | Key Design | Pareto Universality | Mobile MQA | UIB |
|-------|-----------|---------------------|------------|-----|
| MobileNetV1 | Depthwise separable convolutions | No (CPU-optimized) | No | No |
| MobileNetV2 | Inverted bottleneck + linear bottleneck | No | No | No |
| MobileNetV3 | SE blocks + h-swish activation | No (EdgeTPU-friendly) | No | No |
| EfficientNet-Lite | Compound scaling; SE removed for mobile | Partial | No | No |
| MobileViT | ViT blocks + convolutions | No (GPU-biased) | No | No |
| **MobileNetV4** | UIB + Mobile MQA + Roofline NAS | **Yes** | **Yes** | **Yes** |

**Key differences from MobileNetV3**:
- MobileNetV3 uses SE (Squeeze-and-Excite) blocks, which are efficient on some hardware but inefficient on EdgeTPUs due to global average pooling overhead. MNv4 replaces SE with Mobile MQA in Hybrid variants.
- MobileNetV3 NAS uses a single-stage search; MNv4 uses a two-stage search with distillation, yielding better efficiency.
- MobileNetV3 achieves ~2× lower speed than MNv4 at equivalent ImageNet accuracy on both CPUs and EdgeTPUs.

# Experiments

- **Dataset (Classification)**: ImageNet-1K (1.28M training images, 50K validation, 1000 classes); ImageNet-21K (used for pre-training in some baselines); JFT-300M (used for teacher and distillation, proprietary)
- **Dataset (Detection)**: MS-COCO 2017 (118K train, 5K val, 80 object categories)
- **Hardware benchmarked**: ARM Cortex-A55 CPU (Pixel 6), ARM Cortex-A78 CPU (Samsung S23), Qualcomm Hexagon DSP (Pixel 4), ARM Mali-G78 GPU (Pixel 7), Apple Neural Engine (iPhone 15), Google EdgeTPU (Pixel 6, Pixel 8)
- **Optimizer**: SGD with momentum (NAS phase); Adam with cosine decay (distillation phase)
- **Key results**:
  - MNv4-Conv-M achieves 79.9% ImageNet-1K top-1 at 11.4ms on Pixel 6 CPU — approximately 2× faster than MobileNetV3 at similar accuracy
  - MNv4-Hybrid-M achieves 34.0% COCO AP with RetinaNet + FPN decoder
  - MNv4-Hybrid-L achieves 87.0% ImageNet-1K top-1 with JFT pre-training at 3.8ms on Pixel 8 EdgeTPU
