# Meta Information

- URL: [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J., Lucic, M., & Dosovitskiy, A. (2021). MLP-Mixer: An all-MLP Architecture for Vision. NeurIPS 2021.

# MLP-Mixer: An all-MLP Architecture for Vision

## Overview

MLP-Mixer demonstrates that neither convolutions nor self-attention are necessary to achieve competitive performance on image recognition benchmarks. The architecture uses only multi-layer perceptrons (MLPs) applied alternately across two axes of a patch-feature table: one MLP mixes spatial (token) information, and another mixes per-location channel (feature) information. Given large-scale pre-training data and modern regularization, Mixer matches or approaches state-of-the-art convolutional networks and Vision Transformers (ViTs), while being simpler and faster.

**Applicability:** MLP-Mixer is suited for researchers studying inductive biases in vision architectures, practitioners who require high-throughput inference at large model scales (pre-trained on ImageNet-21k or JFT-300M/JFT-3B), and anyone exploring architectures without explicit locality or permutation-equivariance assumptions.

## 1. Input Representation

Given an image $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$, it is divided into non-overlapping patches of size $P \times P$. This yields $S = \frac{HW}{P^2}$ patches. Each patch is projected linearly to a $C_{hidden}$-dimensional vector, producing a table:

$$\mathbf{X} \in \mathbb{R}^{S \times C_{hidden}}$$

- $S$: number of patches (tokens), e.g., $14 \times 14 = 196$ for a $224 \times 224$ image with $P=16$
- $C_{hidden}$: hidden (channel) dimension, e.g., 768 for Mixer-B/16

No positional embeddings are added; token-mixing MLPs implicitly learn positional structure.

## 2. Architecture

### 2.1 Mixer Layer

Each Mixer layer consists of two MLP blocks applied in sequence, each preceded by layer normalization and followed by a skip connection:

**Token-mixing (spatial mixing):**

$$\mathbf{U}_{*,i} = \mathbf{X}_{*,i} + W_2 \, \sigma\!\left(W_1 \, \mathrm{LN}(\mathbf{X})_{*,i}\right), \quad i = 1, \ldots, C_{hidden}$$

- Input: $\mathbf{X} \in \mathbb{R}^{S \times C_{hidden}}$; the MLP is applied to each column independently
- $W_1 \in \mathbb{R}^{D_S \times S}$, $W_2 \in \mathbb{R}^{S \times D_S}$, $D_S$: token-mixing MLP hidden dimension
- The same $W_1, W_2$ are shared across all $C_{hidden}$ columns (parameter sharing)
- Output: $\mathbf{U} \in \mathbb{R}^{S \times C_{hidden}}$

**Channel-mixing (per-location feature mixing):**

$$\mathbf{Y}_{j,*} = \mathbf{U}_{j,*} + W_4 \, \sigma\!\left(W_3 \, \mathrm{LN}(\mathbf{U})_{j,*}\right), \quad j = 1, \ldots, S$$

- Input: $\mathbf{U} \in \mathbb{R}^{S \times C_{hidden}}$; the MLP is applied to each row independently
- $W_3 \in \mathbb{R}^{D_C \times C_{hidden}}$, $W_4 \in \mathbb{R}^{C_{hidden} \times D_C}$, $D_C$: channel-mixing MLP hidden dimension
- The same $W_3, W_4$ are shared across all $S$ rows
- Output: $\mathbf{Y} \in \mathbb{R}^{S \times C_{hidden}}$

Here $\sigma$ denotes the GELU nonlinearity. $\mathrm{LN}$ is layer normalization.

### 2.2 Full Architecture

```
Input image  →  Patch embedding (linear projection per patch)
              →  [Mixer Layer] × L
              →  Global average pooling over S tokens  →  Linear classifier
```

**Pseudocode for a single Mixer Layer:**

```
function MixerLayer(X):  # X: [S, C_hidden]
    # Token-mixing
    Y = LayerNorm(X)         # [S, C_hidden]
    Y = Y.T                  # [C_hidden, S]
    Y = MLP_token(Y)         # [C_hidden, S] -> [C_hidden, S]
    Y = Y.T                  # [S, C_hidden]
    X = X + Y                # residual

    # Channel-mixing
    Z = LayerNorm(X)         # [S, C_hidden]
    Z = MLP_channel(Z)       # [S, C_hidden] -> [S, C_hidden]
    X = X + Z                # residual
    return X
```

After $L$ Mixer layers, global average pooling over the $S$ dimension produces a $C_{hidden}$-dimensional vector fed to a linear classifier head.

## 3. Model Variants

| Model    | Layers ($L$) | Patch size ($P$) | $C_{hidden}$ | $D_S$ | $D_C$ | Parameters |
|----------|-------------|------------------|--------------|-------|-------|-----------|
| Mixer-S/32 | 8         | 32×32            | 512          | 256   | 2048  | —         |
| Mixer-S/16 | 8         | 16×16            | 512          | 256   | 2048  | 18M       |
| Mixer-B/32 | 12        | 32×32            | 768          | 384   | 3072  | —         |
| Mixer-B/16 | 12        | 16×16            | 768          | 384   | 3072  | 59M       |
| Mixer-L/16 | 24        | 16×16            | 1024         | 512   | 4096  | 207M      |
| Mixer-H/14 | 32        | 14×14            | 1280         | 640   | 5120  | 431M      |

Naming follows ViT conventions: B = Base, L = Large, H = Huge; the number after `/` is the patch size in pixels.

## 4. Relationship to CNNs and ViTs

> [!NOTE]
> The authors show MLP-Mixer is "most related to the extreme [CNN] where each feature map is given a separate 1×1 conv, and a single depthwise conv of full receptive field replaces each N×N conv."

| Property | CNN | ViT | MLP-Mixer |
|----------|-----|-----|-----------|
| Spatial mixing | Local convolutions (inductive bias) | Global self-attention (quadratic cost) | Global MLP (linear cost, parameter-shared) |
| Channel mixing | Pointwise conv / FC | Feedforward layer per token | Channel-mixing MLP (shared across tokens) |
| Positional encoding | Implicit (conv) | Explicit (learned) | None (order sensitivity via token MLP) |
| Computational complexity in $S$ | $O(S)$ | $O(S^2)$ | $O(S)$ |
| Inductive locality bias | Strong | Weak | Weakest |

The token-mixing MLP differs from self-attention in that its weights do not depend on the input values (no query-key-value computation), yet it still allows all tokens to interact globally.

## 5. Training Details

- **Optimizer:** Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$), with linear warmup and linear decay (pre-training)
- **Regularization:** RandAugment, mixup, dropout, stochastic depth
- **Fine-tuning:** Momentum SGD, cosine learning rate decay, higher resolution inputs
- **Higher resolution:** If resolution increases from $224 \to 448$, $S$ quadruples ($196 \to 784$). Token-mixing weight matrices are resized (e.g., bilinear interpolation of weight rows treated as $\sqrt{S} \times \sqrt{S}$ grids).

## 6. Experiments

### Datasets

| Dataset | Role | Size |
|---------|------|------|
| ImageNet (ILSVRC-2012) | Pre-train / fine-tune eval | 1.28M train, 50K val |
| ImageNet-21k | Pre-train | ~21M images, 21k classes |
| JFT-300M | Pre-train | ~300M images, ~30k classes |
| JFT-3B | Pre-train (scaling) | ~3B images |
| CIFAR-10 / CIFAR-100 | Downstream eval | 50K train, 10K test |
| Oxford Pets / Flowers | Downstream eval | Thousands |
| VTAB-1k benchmark | Downstream eval (19 tasks) | 1000 labeled examples/task |

- **Hardware:** TPU-v3 accelerators
- **Key metric:** Top-1 accuracy on ImageNet validation set

### Key Results

- **Mixer-H/14 (JFT-300M):** 87.94% top-1 on ImageNet — only 0.5% behind ViT-H/14, but 2.5× faster throughput (540 img/s/core vs. ~215 img/s/core)
- **Mixer-L/16 (JFT-300M):** 87.25% top-1; comparable to EfficientNet-L2 (88.4%) and ViT-L/16 (87.76%)
- **Mixer-B/16 (ImageNet only):** 76.44% top-1, approximately 3% behind ViT-B/16 (79.67%), showing Mixer's greater data hunger

### Data Scaling Behavior

When trained on progressively larger subsets of JFT-300M (3%, 10%, 30%, 100%), Mixer's performance grows faster than ResNets. At the 3% subset, Mixer substantially underperforms ResNets, but at 100% it approaches or matches them. This demonstrates Mixer has weaker inductive biases (locality, translation equivariance) but compensates with large-scale data.

### Permutation Sensitivity

Testing with globally shuffled pixel order (random permutation applied consistently to all images):

- ResNet-50: drops ~75% → ~36% accuracy (loses some spatial bias but CNNs partially recover via global pooling)
- MLP-Mixer: drops ~87% → ~42% accuracy (larger drop, weaker locality bias than ResNet)
- Shuffled patches (permuted patch order, pixels intact): Mixer accuracy drops ~5% (tokens can be reordered with modest impact)

This confirms Mixer has weaker built-in locality bias than CNNs, which is consistent with its absence of convolutional structure.

## 7. Ablation Insights (Appendix)

- **Untying token-mixing weights:** Allowing separate weights per column in token-mixing (removing the shared-weight assumption) does not improve accuracy and increases parameters.
- **Channel grouping:** Grouping channels for token-mixing gave <2% improvement for small models but hurt scalability.
- **Pyramidal structure:** Reduces training time but improvements do not transfer to downstream tasks after fine-tuning.
- **Patch size effect:** Smaller patches increase $S$ but improve performance; there is a compute-accuracy tradeoff.

## 8. Embedding Layer Visualization

The learned patch-embedding filters vary by patch resolution:
- **32×32 patches:** Gabor-like edge and texture filters (resembling CNN filters)
- **16×16 patches:** High-frequency, less structured responses

Token-mixing MLPs develop a mix of local and global feature detectors, with some units showing oppositely-phased filter pairs similar to CNN quadrature pairs. This emergent structure arises without any spatial inductive bias built into the architecture.

> [!TIP]
> A compact 43-line JAX/Flax implementation is provided in the paper, illustrating that Mixer's simplicity is genuine — the core computation is a transposition, MLP application, and transposition back.
