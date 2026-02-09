# Meta Information

- URL: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:2010.11929.

# Abstract

This paper demonstrates that pure Transformer architectures, originally designed for natural language processing, can be applied directly to image classification tasks without requiring convolutional layers. The Vision Transformer (ViT) splits images into fixed-size patches, treats them as sequence tokens, and processes them through a standard Transformer encoder. When pre-trained on large datasets (14M-300M images) and fine-tuned on downstream tasks, ViT achieves state-of-the-art performance on ImageNet (88.55%), CIFAR-100 (94.55%), and the VTAB benchmark (77.63%), while requiring 2-4× less computational resources than comparable ResNet models during pre-training.

> [!NOTE]
> The key insight is that "large scale training trumps inductive bias" - while CNNs have built-in assumptions about locality and translation equivariance, Transformers can learn these patterns from data when given sufficient training examples.

# Introduction

Transformers have become the standard architecture for natural language processing due to their scalability and computational efficiency. However, in computer vision, convolutional neural networks (CNNs) remain dominant. Previous attempts to apply self-attention to images either combined attention with CNNs or required specialized hardware accelerators.

Vision Transformer (ViT) applies the standard Transformer architecture directly to images with minimal modifications. An image is divided into patches (typically 16×16 pixels), each patch is linearly embedded, and the resulting sequence is processed by a Transformer encoder. A classification head is attached to the output corresponding to a learnable class token.

**Key differences from CNNs:**
- CNNs have translation equivariance and locality built into their architecture through convolutional kernels
- ViT has minimal image-specific inductive bias, learning spatial relationships purely from data
- ViT requires large-scale pre-training to match or exceed CNN performance

**Applicability conditions:**
- Most effective when pre-trained on large datasets (>14M images)
- On small datasets (ImageNet-1k alone), ViT underperforms ResNets due to lack of inductive bias
- Best suited for scenarios where computational efficiency during pre-training is important

# Method

## Vision Transformer (ViT)

### Input Representation

An input image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of flattened 2D patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where:
- $(H, W)$ is the original image resolution
- $C$ is the number of channels (typically 3 for RGB)
- $P$ is the patch size (typically 16)
- $N = HW/P^2$ is the number of patches (sequence length)

For example, with a 224×224 image and patch size 16:
- $N = 224 \times 224 / (16 \times 16) = 196$ patches
- Each patch is $16 \times 16 \times 3 = 768$ dimensional after flattening

### Patch Embedding

Each patch is linearly projected to dimension $D$ (the Transformer's hidden size):

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1\mathbf{E}; \mathbf{x}_p^2\mathbf{E}; \cdots; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{\text{pos}}$$

where:
- $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the trainable patch embedding projection matrix
- $\mathbf{x}_{\text{class}} \in \mathbb{R}^D$ is a learnable class token prepended to the sequence (similar to BERT's [CLS] token)
- $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ is the learnable position embedding
- $\mathbf{z}_0 \in \mathbb{R}^{(N+1) \times D}$ is the input to the Transformer encoder

> [!NOTE]
> The position embeddings are 1D learnable parameters, not hand-crafted sinusoidal encodings. The model learns to encode 2D spatial structure despite receiving only 1D positional information.

### Transformer Encoder

The Transformer encoder consists of $L$ layers, each containing:

1. **Layer Normalization (LN)**: Applied before each block
2. **Multi-Head Self-Attention (MSA)**: Computes attention across all patches
3. **MLP Block**: Two-layer feedforward network with GELU activation

The computation for layer $\ell$ is:

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$

$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

where:
- $\mathbf{z}_\ell \in \mathbb{R}^{(N+1) \times D}$ is the output of layer $\ell$
- MSA has $k$ attention heads
- MLP consists of two linear layers: $D \to 4D \to D$ (hidden dimension is 4× the embedding dimension)

> [!IMPORTANT]
> ViT uses pre-normalization (Layer Norm before attention/MLP) rather than post-normalization (after), which improves training stability for large models.

### Classification Head

After the final Transformer layer, the representation of the class token is extracted:

$$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$

where $\mathbf{z}_L^0 \in \mathbb{R}^D$ is the first token of the final layer output.

During pre-training, a MLP head is attached:

$$\hat{y} = \text{MLP}_{\text{head}}(\mathbf{y})$$

During fine-tuning, this is replaced with a single linear layer:

$$\hat{y} = \mathbf{W}_{\text{head}} \mathbf{y}$$

where $\mathbf{W}_{\text{head}} \in \mathbb{R}^{K \times D}$ and $K$ is the number of classes in the target dataset.

### Model Variants

| Model | Layers ($L$) | Hidden Size ($D$) | MLP Size | Heads ($k$) | Parameters |
|-------|--------------|-------------------|----------|-------------|------------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

All variants use patch size $P = 16$ by default. Notation like "ViT-L/16" indicates ViT-Large with 16×16 patches.

## Hybrid Architecture

The paper also explores a hybrid model where patch embeddings are extracted from a ResNet's intermediate feature maps instead of raw pixels. Patches of size 1×1 are extracted from the ResNet's feature map, meaning the patch embedding projection is applied to each spatial location of the CNN features.

## Fine-Tuning and Higher Resolution

When fine-tuning on higher resolution images than used during pre-training:
1. Keep the patch size $P$ the same
2. This increases the sequence length $N$
3. Pre-trained position embeddings are 2D-interpolated to match the new sequence length

> [!NOTE]
> The paper states: "We find that feeding higher resolution images during fine-tuning improves performance, even though the model was pre-trained at lower resolution."

# Experiments

## Experimental Setup

### Pre-training Datasets
- **ImageNet**: 1.3M images, 1000 classes
- **ImageNet-21k**: 14M images, 21,000 classes
- **JFT-300M**: 303M high-resolution images, 18,000 classes (proprietary Google dataset)

### Pre-training Configuration
- **Optimizer**: Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$
- **Batch Size**: 4096
- **Weight Decay**: 0.1 (high regularization)
- **Learning Rate Schedule**: Linear warmup followed by linear decay
- **Resolution**: 224×224 for ImageNet/ImageNet-21k, 384×384 for some JFT models

### Fine-tuning Configuration
- **Optimizer**: SGD with momentum 0.9
- **Batch Size**: 512
- **Learning Rate Schedule**: Cosine learning rate decay
- **Resolution**: Often higher than pre-training (e.g., 384×384 or 512×512)
- **Regularization**: No weight decay, lighter than pre-training

### Evaluation Benchmarks
- **ImageNet**: 1000-class classification, 50k validation images
- **ImageNet ReaL**: Re-annotated ImageNet validation labels (more accurate)
- **CIFAR-10/100**: Small-scale datasets with 50k/10k training/test images
- **Oxford-IIIT Pets**: 37 pet breeds, ~7k images
- **Oxford Flowers-102**: 102 flower categories, ~8k images
- **VTAB**: 19 diverse visual tasks across natural, specialized, and structured domains
- **ObjectNet**: Distribution shift test with 50k images

## Main Results

### Performance on ImageNet

| Model | Pre-training | ImageNet Top-1 | ImageNet ReaL | Parameters |
|-------|--------------|----------------|---------------|------------|
| ViT-H/14 | JFT-300M | **88.55%** | **90.72%** | 632M |
| ViT-L/16 | JFT-300M | 87.76% | 90.54% | 307M |
| BiT-L (ResNet-152x4) | JFT-300M | 87.54% | 90.54% | 928M |
| Noisy Student (EfficientNet-L2) | JFT-300M + unlabeled | 88.4% | - | 480M |
| ViT-L/16 | ImageNet-21k | 85.30% | 88.62% | 307M |

**Key findings:**
- ViT-H/14 achieves state-of-the-art 88.55% on ImageNet when pre-trained on JFT-300M
- ViT requires fewer parameters than BiT (632M vs 928M) for comparable performance
- Without large-scale pre-training (JFT), ViT underperforms ResNets

### Performance on Small Datasets

| Model | CIFAR-10 | CIFAR-100 | Pets | Flowers-102 |
|-------|----------|-----------|------|-------------|
| ViT-H/14 (JFT) | 99.50% | 94.55% | 97.56% | 99.68% |
| ViT-L/16 (JFT) | 99.42% | 93.90% | 97.32% | 99.74% |
| BiT-L (JFT) | 99.37% | 93.51% | 96.62% | 99.63% |

ViT achieves state-of-the-art or near state-of-the-art on these downstream tasks after JFT pre-training.

### VTAB Benchmark

VTAB evaluates transfer learning across 19 tasks in three groups:
- **Natural**: Tasks similar to ImageNet (7 tasks)
- **Specialized**: Medical, satellite imagery, etc. (4 tasks)
- **Structured**: Geometric understanding tasks (8 tasks)

**Results (average across groups):**
- ViT-H/14 (JFT): 77.63% average
- BiT-R152x4 (JFT): 76.29% average

ViT outperforms ResNets on the VTAB benchmark, suggesting better transfer learning capabilities.

## Pre-training Data Requirements

A critical experiment compares ViT and ResNet performance across different pre-training dataset sizes:

**Key observations:**
1. **Small data (ImageNet only)**: BiT outperforms ViT by ~3% on ImageNet validation
2. **Medium data (ImageNet-21k)**: Performance gap narrows
3. **Large data (JFT-300M)**: ViT overtakes ResNets

> [!IMPORTANT]
> This demonstrates that the inductive bias of CNNs (locality, translation equivariance) provides an advantage on smaller datasets, but Transformers can learn these patterns from data when given sufficient training examples.

## Computational Cost

ViT is more computationally efficient than ResNets during pre-training:
- ViT-L/16 requires ~2× less compute than BiT-L to reach the same ImageNet performance
- ViT-H/14 achieves higher accuracy than BiT-L with ~2.5× less compute

The paper measures cost in exaFLOPs (total FLOPs across all training examples during pre-training).

## Scaling Analysis

The paper evaluates how performance scales with:
1. **Model size**: Larger models (Base → Large → Huge) consistently improve performance
2. **Data size**: More pre-training data yields better results
3. **Compute**: Performance improves log-linearly with compute budget

**Finding:** ViT-L/16 has not saturated performance even at JFT-300M scale, suggesting further improvements with more data.

# Inspecting Vision Transformer

## Visualization of Learned Representations

### Position Embeddings

The learned 1D position embeddings encode 2D spatial structure:
- Embeddings show row-column patterns when visualized as similarity matrices
- Nearby patches have similar position embeddings
- The model learns to encode 2D image topology without explicit 2D positional encoding

### Attention Distance

**Attention distance** is defined as the average distance (in image space) between a query patch and the patches it attends to, weighted by attention weights.

**Observations:**
- Some attention heads attend primarily to nearby patches (CNN-like local attention)
- Other heads attend globally across the entire image from the lowest layers
- Average attention distance increases with network depth
- This is analogous to how CNN receptive fields grow with depth, but ViT can attend globally from layer 1

### Attention Maps

Analysis of attention from the class token to image patches reveals:
- Attention patterns resemble semantic segmentation
- The model attends to semantically relevant regions for classification
- Different heads specialize in different spatial scales and semantic patterns

> [!CAUTION]
> While attention maps provide interpretability, they should not be over-interpreted as the model's complete "reasoning" process, as attention is only one component of the full computation.

## Self-Supervision

The paper explores masked patch prediction (similar to BERT's masked language modeling) as a self-supervised pre-training objective:

**Setup:**
- Mask 50% of patches randomly
- Predict mean 3-channel color of masked patches (regression task)

**Results:**
- ViT-B/16 achieves 79.9% ImageNet accuracy with self-supervised pre-training
- This is 4% lower than supervised pre-training (84.0%)
- Shows promise but supervised pre-training remains superior

> [!NOTE]
> Subsequent work (MAE, 2021) improved self-supervised ViT training significantly using higher masking ratios (75%) and predicting raw pixels.

# Comparison with CNNs

## Inductive Bias

| Aspect | CNNs | Vision Transformer |
|--------|------|-------------------|
| **Translation Equivariance** | Built-in through weight sharing in convolutional kernels | Must be learned from data |
| **Locality** | Strong prior: each neuron only sees local receptive field | Global self-attention from layer 1 |
| **Parameter Efficiency** | High due to weight sharing | Lower, but compensated by global receptive field |
| **Data Requirements** | Work well on small datasets | Require large-scale pre-training |
| **Computational Efficiency** | Lower during inference (fewer parameters) | Lower during training (more parallelizable) |

## When to Use ViT vs CNNs

**Use ViT when:**
- Large-scale pre-training data is available (>10M images)
- Computational resources for pre-training are available
- Global reasoning is important for the task
- Transfer learning to diverse downstream tasks is needed

**Use CNNs when:**
- Training from scratch on small datasets
- Inference efficiency (speed/memory) is critical
- Strong translation equivariance is beneficial
- Hardware/infrastructure favors convolutional operations

# Conclusion

Vision Transformer demonstrates that Transformer architectures can match or exceed CNN performance on image classification when pre-trained at sufficient scale. The key insight is that **large-scale training trumps inductive bias**: while CNNs have architectural priors suited for images, Transformers can learn these patterns from data given enough examples.

**Main contributions:**
1. Direct application of standard Transformer architecture to images via patch sequences
2. State-of-the-art results on ImageNet (88.55%) and VTAB (77.63%) with lower pre-training cost
3. Evidence that vision-specific inductive biases are not necessary when training at scale
4. Analysis showing ViT learns both local and global attention patterns

**Limitations identified by authors:**
- Requires large-scale pre-training datasets
- Self-supervised pre-training lags behind supervised
- Extension to detection and segmentation requires further work (addressed by subsequent papers like ViTDet)

> [!TIP]
> For implementation details and pre-trained models, see the official repository: https://github.com/google-research/vision_transformer
