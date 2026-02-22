# Meta Information

- URL: [Image Transformer](https://arxiv.org/abs/1802.05751)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, Ł., Shazeer, N., Ku, A., & Tran, D. (2018). Image Transformer. Proceedings of the 35th International Conference on Machine Learning (ICML 2018).

# Image Transformer

## Overview

The Image Transformer applies the self-attention mechanism from NLP Transformers to image generation, replacing convolutional and recurrent architectures. The key challenge it addresses is the quadratic memory complexity of full self-attention over all image pixels, which is intractable for images of even moderate resolution.

The model generates images autoregressively: each pixel is conditioned on all previously generated pixels (in raster-scan order). For a $h \times w$ image with 3 color channels, the model factorizes the joint distribution as:

$$\log p(x) = \sum_{t=1}^{h \cdot w \cdot 3} \log p(x_t \mid x_{<t})$$

> [!NOTE]
> "We combine local self-attention to attend to local neighborhoods with much larger receptive fields than CNNs, with a model size that remains manageable even for larger images."

**Applicability**: This method is suited for likelihood-based image generation, class-conditional synthesis, and super-resolution. It is most effective when training stability matters (unlike GANs) and when sharp, mode-covering generation is required.

## Architecture

### Pixel Representation

Each pixel intensity (0–255 per channel) is embedded as a $d$-dimensional vector:
- **Categorical**: 256 learned embedding vectors per channel, resulting in a $d$-dimensional input per pixel channel.
- **Continuous (DMOL)**: A $1 \times 3$ strided convolution combines all three channels, yielding one $d$-dimensional vector per pixel position.

A positional encoding is added to each pixel representation. Two variants are evaluated:
1. **Sinusoidal**: $d/2$ dimensions encode the row coordinate, $d/2$ encode the column and channel using sine/cosine functions of different frequencies.
2. **Learned**: Trainable embeddings for (row, column, channel) position.

### Local Self-Attention

Full self-attention over all $n = h \cdot w \cdot 3$ positions has $O(n^2)$ complexity. The Image Transformer restricts each query position to attend only to a local memory block $M$, reducing complexity to $O(h \cdot w \cdot l_m \cdot d)$ where $l_m$ is the memory block size.

**Self-Attention Computation** (per query position $q \in \mathbb{R}^d$ with memory $M \in \mathbb{R}^{l_m \times d}$):

$$q_a = \text{LayerNorm}\!\left(q + \text{dropout}\!\left(\text{softmax}\!\left(\frac{W_q q \cdot (M W_k)^\top}{\sqrt{d}}\right) M W_v\right)\right)$$

$$q' = \text{LayerNorm}\!\left(q_a + \text{dropout}(W_1 \cdot \text{ReLU}(W_2 q_a))\right)$$

where $W_q, W_k, W_v \in \mathbb{R}^{d \times d}$ are learned projection matrices, and the feed-forward sublayer applies position-wise with shared weights.

### 1D Local Attention

Images are flattened in raster-scan order. Non-overlapping query blocks of length $l_q = 256$ are formed. Each query block's memory block includes the same $l_q$ positions plus $l_m - l_q$ preceding generated pixels, giving a total memory size of $l_m = 512$.

```
For each non-overlapping query block Q of length l_q:
  M = [l_m - l_q preceding positions] ++ Q
  For each q in Q:
    Attend to all positions in M (causally masked future positions)
```

### 2D Local Attention

Images are partitioned into rectangular query blocks that are contiguous in 2D image space (not the raster-scan linearization). This gives more balanced access to horizontally and vertically adjacent pixels.

```
Partition image into non-overlapping query blocks of size w_q × h_q:
  For each query block Q:
    Memory block M extends Q by:
      h_m rows above
      w_m pixels left and right
    For each q in Q:
      Attend to all M positions generated before q (causal masking)
```

In experiments: $l_q = 8 \times 32$, memory extends $h_m = 8$ rows above and $w_m = 16$ pixels left/right.

> [!IMPORTANT]
> 2D local attention provides more balanced vertical and horizontal context than 1D, since 1D locality in raster-scan order is dominated by horizontally adjacent pixels with limited vertical context.

### Encoder–Decoder for Super-Resolution

For conditional image synthesis (e.g., super-resolution from $8 \times 8$ to $32 \times 32$):

**Encoder** (processes low-resolution input, $h_{lr} \times w_{lr} \times 3$):
- Embed each pixel with intensity embeddings and 2D positional encodings.
- Apply $N_{enc}$ transformer layers with **unmasked** (bidirectional) self-attention.
- Output: contextualized representation $E \in \mathbb{R}^{(h_{lr} \cdot w_{lr} \cdot 3) \times d}$.

**Decoder** (generates high-resolution output autoregressively):
- Three sub-layers per block:
  1. **Local self-attention** (causally masked) over previously generated pixels.
  2. **Cross-attention** to encoder output $E$ (decoder attends to full encoder context).
  3. **Position-wise feed-forward** network.

Typical configuration: $N_{enc} = 4$, $N_{dec} = 12$, $d = 512$, 8 attention heads, feed-forward dim = 2048, dropout = 0.1.

## Output Distributions

| Distribution | Parameters/pixel | Description |
|---|---|---|
| **Categorical (cat)** | $256 \times 3 = 768$ | One 256-way softmax per channel, channels factorized |
| **Discretized Mixture of Logistics (DMOL)** | 100 (10 components) | Captures ordinal pixel structure; each component has 1 mixture weight, 3 means, 3 log-scales, 3 dependency coefficients |

DMOL reduces output dimensionality by 7× vs categorical for $32 \times 32$ images (102,400 vs 786,432).

## Differences from Similar Methods

| Method | Attention Scope | Complexity | Notable Feature |
|---|---|---|---|
| **Image Transformer** | Local ($l_m$ pixels) | $O(n \cdot l_m \cdot d)$ | Self-attention on raster-scan or 2D blocks |
| **PixelCNN / PixelCNN++** | Convolutional receptive field | $O(n \cdot k^2 \cdot d)$ | Masked convolutions, limited receptive field |
| **Gated PixelCNN** | 25-pixel receptive field | $O(n \cdot d)$ | Row + column stacks for 2D context |
| **PixelRNN** | Full recurrent context | $O(n^2)$ | Sequential, captures long-range but slow |
| **Original Transformer** | Full self-attention | $O(n^2 \cdot d)$ | No locality restriction; infeasible for images |

> [!TIP]
> The [PixelCNN++ paper](https://arxiv.org/abs/1701.05517) (Salimans et al., 2017) introduced the DMOL output distribution used here.

## Experiments

### Datasets

| Dataset | Task | Resolution | Size |
|---|---|---|---|
| **CIFAR-10** | Unconditional / class-conditional generation, super-resolution | $32 \times 32$ | 50,000 train / 10,000 test |
| **ImageNet** | Unconditional generation | $64 \times 64$ | ~1.2M train |
| **CelebA** | Super-resolution | $8 \times 8 \to 32 \times 32$ | ~200K images |

### Hardware and Training

- Hardware: TPUs (training), P100 and K40 GPUs (evaluation)
- Optimizer: Adam with learning rate warm-up and decay
- Sampling temperature: $\tau = 0.8$–$1.0$ (tempered softmax)

### Key Results

**Unconditional Generation (bits/dim, lower is better):**

| Model | CIFAR-10 | ImageNet 64×64 |
|---|---|---|
| Gated PixelCNN | 3.03 | 3.83 |
| PixelCNN++ | 2.92 | — |
| **Image Transformer (1D, DMOL)** | **2.90** | **3.77** |

**Super-Resolution (CelebA, NLL bits/dim):**

| Attention Type | NLL |
|---|---|
| 1D Local | 2.68 |
| 2D Local | **2.61** |

Human evaluation on CelebA super-resolution: 36.11% of generated $32 \times 32$ images fooled evaluators (vs. ~11% for prior work).

Experiments also show that larger local attention windows consistently improve performance, with significant gains up to 256-pixel effective receptive fields. Beyond this scale, marginal improvements diminish.

## Summary

The Image Transformer demonstrates that Transformer self-attention can be adapted to image generation by restricting attention to local neighborhoods (1D or 2D partitions). This achieves larger effective receptive fields than CNNs (up to 256 pixels vs. 25 in Gated PixelCNN), better capturing global image structure, while keeping memory complexity tractable. The DMOL output distribution reduces output dimensionality by 7× without accuracy loss. The method is well-suited for tasks requiring sharp, mode-covering image generation without the training instability of GANs.
