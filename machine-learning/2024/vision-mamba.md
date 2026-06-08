# Meta Information

- URL: [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., & Wang, X. (2024). Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model. arXiv:2401.09417.

# Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model

Vision Mamba (Vim) proposes a pure state space model (SSM)-based vision backbone that eliminates the quadratic-complexity attention mechanism of Transformers. It introduces bidirectional Mamba blocks with position embeddings to capture global visual context while maintaining linear memory complexity. The target users are computer vision researchers and practitioners who need efficient processing of high-resolution images (e.g., medical imaging, remote sensing), where Transformer-based architectures become prohibitively slow or memory-intensive.

## Background: State Space Models

Structured State Space Models (S4) model sequences through a continuous linear dynamical system:

$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)$$
$$y(t) = \mathbf{C}h(t)$$

where $x(t) \in \mathbb{R}$ is the input signal, $h(t) \in \mathbb{R}^{N}$ is the hidden state, $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the state transition matrix, $\mathbf{B} \in \mathbb{R}^{N \times 1}$ is the input projection, and $\mathbf{C} \in \mathbb{R}^{1 \times N}$ is the output projection.

For neural network training, the continuous system is discretized using the **Zero-Order Hold (ZOH)** rule with a timescale parameter $\Delta$:

$$\bar{\mathbf{A}} = \exp(\Delta \mathbf{A})$$
$$\bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$

The discrete recurrence becomes:

$$h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t$$
$$y_t = \mathbf{C} h_t$$

**Mamba** extends S4 by making $\Delta$, $\mathbf{B}$, and $\mathbf{C}$ input-dependent (selective SSM), enabling the model to selectively retain or ignore information from the input sequence.

> [!NOTE]
> The key distinction from Transformers is that SSMs have $O(M)$ memory complexity in sequence length $M$, whereas self-attention is $O(M^2)$.

## Architecture

### Image Tokenization

An input image $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ is split into $M$ non-overlapping patches of size $P \times P$ (default $P = 16$), giving $M = \frac{HW}{P^2}$ patch tokens. Each patch is linearly projected to a $D$-dimensional embedding and a classification token [CLS] is prepended:

$$\mathbf{T}_0 = [\mathbf{t}_{cls}; \mathbf{t}_1^0 \mathbf{W}; \mathbf{t}_2^0 \mathbf{W}; \ldots; \mathbf{t}_M^0 \mathbf{W}] + \mathbf{E}_{pos}$$

where $\mathbf{W} \in \mathbb{R}^{P^2 C \times D}$ is the patch projection weight, $\mathbf{E}_{pos} \in \mathbb{R}^{(M+1) \times D}$ is the learnable position embedding.

### Vision Mamba Block

Each Vim block processes tokens bidirectionally. The computation for a single block with input $\mathbf{T}_l \in \mathbb{R}^{(M+1) \times D}$:

**Step 1: Layer Normalization and projection**
$$\tilde{\mathbf{T}} = \text{Norm}(\mathbf{T}_l)$$
$$\mathbf{x} = \text{Linear}_x(\tilde{\mathbf{T}}) \in \mathbb{R}^{(M+1) \times ED}$$
$$\mathbf{z} = \text{Linear}_z(\tilde{\mathbf{T}}) \in \mathbb{R}^{(M+1) \times ED}$$

where $E$ is the expand factor (default $E = 2$).

**Step 2: Convolution preprocessing (both directions share)**
$$\mathbf{x}' = \text{SiLU}(\text{Conv1d}(\mathbf{x}))$$

**Step 3: Direction-specific SSM projections**

Forward direction ($\rightarrow$):
$$\mathbf{B}_f = \text{Linear}_B(\mathbf{x}'), \quad \mathbf{C}_f = \text{Linear}_C(\mathbf{x}'), \quad \Delta_f = \text{Softplus}(\text{Linear}_\Delta(\mathbf{x}'))$$

Backward direction ($\leftarrow$): same projections applied to the reversed sequence $\text{flip}(\mathbf{x}')$.

**Step 4: SSM computation**

$$y_f = \text{SSM}(\bar{\mathbf{A}}_f, \bar{\mathbf{B}}_f, \mathbf{C}_f)(\mathbf{x}') \in \mathbb{R}^{(M+1) \times ED}$$
$$y_b = \text{flip}(\text{SSM}(\bar{\mathbf{A}}_b, \bar{\mathbf{B}}_b, \mathbf{C}_b)(\text{flip}(\mathbf{x}'))) \in \mathbb{R}^{(M+1) \times ED}$$

**Step 5: Gating and output**
$$\mathbf{T}_{l+1} = \text{Linear}_{out}((y_f + y_b) \odot \text{SiLU}(\mathbf{z})) + \mathbf{T}_l$$

> [!IMPORTANT]
> The bidirectional design is critical because SSMs are inherently causal (left-to-right). For vision, both left-to-right and right-to-left context is needed to understand spatial relationships. Processing both directions and summing their outputs gives the model full sequence context at each position.

### Model Configurations

| Config | Layers ($L$) | Hidden dim ($D$) | Expand ($E$) | SSM dim ($N$) | Params |
|--------|-------------|-----------------|--------------|---------------|--------|
| Vim-Ti | 24 | 192 | 2 | 16 | 7M |
| Vim-S  | 24 | 384 | 2 | 16 | 26M |

## Efficiency Analysis

### Computational Complexity Comparison

For a sequence of length $M$ and hidden dimension $D$:

- **Self-Attention (Transformer):** $\Omega_{\text{attn}} = 4MD^2 + 2M^2D$ — **quadratic** in $M$
- **SSM (Vision Mamba):** $\Omega_{\text{SSM}} = 6MDEN$ — **linear** in $M$ (with fixed $N=16$)

At high resolutions (e.g., $1248 \times 1248$), this translates to:
- **2.8× faster inference** than DeiT
- **86.8% GPU memory savings** compared to DeiT

### IO-Awareness

Memory bandwidth is optimized by keeping intermediate SSM states in fast SRAM rather than writing to slow HBM. The memory footprint reduces from $O(BMEN)$ to $O(BME + EN)$ where $B$ is batch size.

### Memory During Backpropagation

Intermediate states are recomputed during the backward pass (gradient checkpointing), trading compute for memory.

## Differences from Related Methods

| Method | Architecture Type | Bidirectional | 2D Spatial Prior | Complexity |
|--------|------------------|---------------|-----------------|------------|
| ViT / DeiT | Pure Transformer | Yes (attention) | No | $O(M^2)$ |
| Swin Transformer | Windowed Transformer | Local window | Yes (shifted window) | $O(M)$ |
| S4ND-ViT | Hybrid SSM+Transformer | Partial | Yes (2D SSM) | $O(M)$ |
| ConvSSM | SSM+Convolution | Partial | Yes (conv) | $O(M)$ |
| **Vision Mamba (Vim)** | **Pure SSM** | **Yes (bidirectional)** | **No (sequence only)** | **$O(M)$** |

> [!NOTE]
> Prior SSM-based vision methods either combined SSMs with convolutions or attention to compensate for causal limitations. Vim is the first **pure-SSM** architecture that handles 2D images as 1D token sequences without 2D spatial priors, relying solely on position embeddings for spatial awareness.

## Classification Token Positioning

A key design choice is where to insert the [CLS] token in the 1D token sequence. Three strategies were evaluated:

- **Head:** Place [CLS] at position 0 → sees all context in the forward pass
- **Tail:** Place [CLS] at position $M$ → sees all context in the backward pass
- **Middle:** Place [CLS] at position $M/2$ → equidistant from all tokens in both directions

> [!IMPORTANT]
> Experiments show **middle positioning achieves the best performance** (76.1% vs. 75.2% for head). This is because the SSM recurrence gives the middle token equal maximum path length to all other tokens in both forward and backward directions, effectively making it the most "central" position in the recurrent graph.

## Experiments

- **Datasets:**
  - ImageNet-1K (1.28M training images, 50K validation, 1000 classes) for image classification
  - ADE20K (20K training, 2K validation images, 150 semantic categories) for semantic segmentation
  - COCO 2017 (118K training, 5K validation images) for object detection and instance segmentation

- **Hardware:** 8 A800 GPUs for ImageNet training

- **Optimizer:** AdamW, weight decay 0.05, cosine learning rate schedule

- **Training:** 300 epochs on ImageNet, batch size 1024, learning rate 1e-3, DeiT-style augmentation (RandAugment, Mixup, CutMix, random erasing)

- **Key Results:**
  - Vim-Ti (7M params): 76.1% Top-1 on ImageNet vs. DeiT-Ti 72.2% (+3.9%)
  - Vim-S (26M params): 80.5% Top-1 on ImageNet vs. DeiT-S 79.8% (+0.7%)
  - Vim-Ti on ADE20K: mIoU 41.0 vs. DeiT-Ti 39.2 (+1.8)
  - Vim-Ti on COCO: +1.3 box AP, +1.1 mask AP over DeiT-Ti

## Ablation: Bidirectional Strategy

| Strategy | ImageNet Top-1 | ADE20K mIoU |
|----------|----------------|-------------|
| Unidirectional SSM | 73.2% | 32.3 |
| Bidirectional SSM only | 73.6% | 34.8 |
| Bidirectional SSM + Conv1d | **73.9%** | **35.9%** |

The combination of bidirectional scanning and 1D convolution preprocessing provides the best results.

> [!TIP]
> The original Mamba paper: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
> The ViT paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
