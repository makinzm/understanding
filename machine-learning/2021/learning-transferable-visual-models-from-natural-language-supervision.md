# Meta Information

- URL: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.

# CLIP: Learning Transferable Visual Models From Natural Language Supervision

## Overview

CLIP (Contrastive Language-Image Pre-training) is a visual representation learning method that jointly trains an image encoder and a text encoder on 400 million (image, text) pairs collected from the internet. The key insight is that natural language provides a much richer and more scalable source of supervision than fixed label sets: instead of predicting a discrete class index, CLIP learns to align image and text embeddings in a shared latent space using a contrastive objective.

**Who uses this**: Researchers and practitioners who need general-purpose visual features transferable to diverse downstream tasks without per-task retraining. Applicable to zero-shot classification, image-text retrieval, multimodal search, and as a backbone for vision-language systems (e.g., DALL-E, Stable Diffusion).

**When**: CLIP is effective when labeled training data for the target task is scarce or unavailable, or when the target label vocabulary was not seen during supervised training. It excels at tasks well-described by natural language but struggles with highly specialized tasks requiring fine-grained domain knowledge (e.g., satellite imagery, pathology).

## Motivation: Limitations of Standard Vision Supervision

Standard image classifiers are trained on fixed, manually annotated datasets (e.g., ImageNet with 1,000 classes). This approach has three fundamental weaknesses:

1. **Label bottleneck**: Expensive human annotation limits scale. ImageNet took years to curate, while the internet contains billions of image-text pairs available for free.
2. **Narrow generalization**: Models trained to predict a fixed label set cannot recognize novel categories without retraining.
3. **Poor distribution shift robustness**: ImageNet-trained models degrade sharply when images differ in style, domain, or rendering from the training distribution.

CLIP addresses all three by replacing discrete label prediction with a contrastive objective over natural language descriptions.

## Model Architecture

CLIP consists of two independently parameterized encoders that map inputs into a shared $d$-dimensional embedding space.

### Image Encoder

Two families of architectures were explored:

| Architecture | Description |
|---|---|
| ResNet variants | Modified ResNet-50/101 with ResNet-D improvements, antialiased blur pooling, and attention pooling replacing global average pooling |
| Vision Transformers (ViT) | ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px with standard patch embedding |

The attention pooling layer is a single multi-head QKV attention layer that takes a learned `[CLS]`-like query token attending over spatial patch features, producing a fixed-size embedding.

Scale variants (RN50x4, RN50x16, RN50x64) allocate compute equally across width, depth, and resolution, following EfficientNet-style scaling.

### Text Encoder

- Architecture: Transformer with causal masking
- Parameters: 63 million (base model)
- Depth: 12 layers, width: 512, attention heads: 8
- Tokenization: Byte-pair encoding (BPE) with vocabulary size 49,152
- Max sequence length: 76 tokens
- Special tokens: `[SOS]` prepended, `[EOS]` appended
- Feature extraction: The activations at the `[EOS]` token position after layer normalization serve as the text embedding $t \in \mathbb{R}^d$

### Shared Embedding Space

Both encoders project their outputs through a linear layer to a common $d$-dimensional space. The embeddings are L2-normalized before computing similarities:

```math
\begin{align}
I_i &= \text{normalize}(W_I \cdot \text{ImageEncoder}(x_i^{\text{img}})) \in \mathbb{R}^d \\
T_j &= \text{normalize}(W_T \cdot \text{TextEncoder}(x_j^{\text{txt}})) \in \mathbb{R}^d
\end{align}
```

where $W_I \in \mathbb{R}^{d \times d_I}$ and $W_T \in \mathbb{R}^{d \times d_T}$ are learned projection matrices.

## Contrastive Pre-Training Algorithm

The training objective is to predict which of $N \times N$ possible image-text pairings actually occurred in a batch of $N$ pairs. The similarity matrix $S \in \mathbb{R}^{N \times N}$ has entry:

```math
\begin{align}
S_{ij} = \frac{I_i \cdot T_j}{\tau}
\end{align}
```

where $\tau$ is a learnable log-parameterized temperature scalar initialized to 0.07 and clipped to prevent log($\tau$) from exceeding 100.

The loss is symmetric cross-entropy applied along both the image axis (each image matched to its text) and the text axis (each text matched to its image):

```math
\begin{align}
\mathcal{L} = \frac{1}{2}\left(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}}\right)
\end{align}
```

where $\mathcal{L}_{\text{image}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{k=1}^{N} \exp(S_{ik})}$ and $\mathcal{L}_{\text{text}}$ is the analogous column-wise loss.

**Pseudocode** (following Figure 3 in the paper):

```
# Shapes: image_features [N, d_I], text_features [N, d_T]
image_embeddings = normalize(image_encoder(images) @ W_I)   # [N, d]
text_embeddings  = normalize(text_encoder(texts)  @ W_T)    # [N, d]

logits = image_embeddings @ text_embeddings.T / tau          # [N, N]

# Ground truth: diagonal entries are positive pairs
labels = arange(N)                                           # [N]

loss_i = cross_entropy(logits,   labels)  # image -> text
loss_t = cross_entropy(logits.T, labels)  # text  -> image
loss   = (loss_i + loss_t) / 2
```

> [!NOTE]
> The contrastive objective is equivalent to a multinomial logistic regression classifier with L2-normalized inputs, L2-normalized weights, no bias, and temperature scaling. Positive pairs are on the diagonal; all off-diagonal entries are negatives within the same batch.

> [!IMPORTANT]
> An early experiment with a generative objective (predicting exact word tokens from images, similar to VirTex) was 3× slower to reach the same zero-shot performance as the contrastive objective, motivating the switch to contrastive learning.

## Comparison with Prior Work

| Method | Supervision | Scale | Zero-Shot Transfer |
|---|---|---|---|
| ImageNet supervised (ResNet-50) | 1.28M labeled images | Fixed 1,000 classes | No |
| VirTex | Captions (COCO scale) | Generative text prediction | Limited |
| ConVIRT | Medical image-report pairs | Contrastive | Domain-specific |
| ALIGN (concurrent) | 1.8B noisy image-text pairs | Contrastive | Yes |
| **CLIP** | **400M web image-text pairs** | **Contrastive** | **Yes, 27+ datasets** |

CLIP differs from ConVIRT (its closest predecessor) primarily in scale and the use of a large curated web dataset (WIT: WebImageText) versus medical image-report pairs.

## Zero-Shot Classification Procedure

CLIP repurposes the text encoder as a hypernetwork that generates a linear classifier from class name embeddings — no gradient updates are needed for a new task.

**Step-by-step**:

1. For each of $K$ target classes with label $c_k$, construct a text prompt: `"A photo of a {c_k}."`
2. Compute text embedding: $t_k = \text{normalize}(W_T \cdot \text{TextEncoder}(\text{prompt}_k)) \in \mathbb{R}^d$
3. Stack classifier weights: $W_{\text{cls}} = [t_1, t_2, \ldots, t_K]^\top \in \mathbb{R}^{K \times d}$
4. For a test image $x$, compute: $I = \text{normalize}(W_I \cdot \text{ImageEncoder}(x)) \in \mathbb{R}^d$
5. Predict: $\hat{y} = \arg\max_k (W_{\text{cls}} I)_k / \tau$

This is equivalent to cosine-similarity nearest-neighbor search in the shared embedding space.

## Prompt Engineering and Ensembling

Using `"{label}"` as the text directly performs poorly because single words are ambiguous and differ in distribution from the descriptive captions seen during training. Wrapping labels in context prompts reduces this distribution gap.

**ImageNet prompt**: `"A photo of a {label}."` alone gives +1.3% over bare label names.

**Task-specific templates**:

| Dataset | Prompt template |
|---|---|
| Oxford Pets | `"A photo of a {label}, a type of pet."` |
| Food101 | `"A photo of a {label}, a type of food."` |
| EuroSAT | `"A centered satellite photo of {label}."` |
| OCR tasks | `"The number {label} in the center of the image."` |

**Ensemble**: Averaging embeddings from 80 different prompt templates (e.g., `"A photo of a big {label}"`, `"A photo of a small {label}"`, etc.) improves zero-shot ImageNet accuracy by an additional 3.5% over a single prompt.

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 32,768 |
| Training epochs | 32 |
| Optimizer | Adam with decoupled weight decay |
| Learning rate schedule | Cosine decay with warmup |
| Temperature $\tau$ | Learned, log-parameterized |
| Mixed precision | Yes |
| Largest model (ViT-L/14@336) | 18 days on 592 V100 GPUs |

## Experiments

- **Datasets (evaluation)**: 27 diverse datasets spanning object recognition, action recognition, geo-localization, OCR, texture, scene, satellite imagery, and medical imaging:
  - ImageNet, ImageNet-V2, ImageNet-Sketch, ImageNet-A, ImageNet-R, ObjectNet
  - CIFAR-10, CIFAR-100, STL-10, SVHN
  - Oxford-IIIT Pets, Food101, Flowers102, Stanford Cars, FGVC Aircraft, DTD, EuroSAT, RESISC45, Country211
  - SUN397, UCF101, Kinetics-700, Kinetics-400
  - PatchCamelyon (histopathology), CLEVRCounts, HatefulMemes, SST2, MNIST
- **Distribution shift robustness** (7 datasets): ImageNet-V2, ImageNet-Sketch, ImageNet-A, ImageNet-R, ObjectNet, YouTube-BB, ImageNet-Vid
- **Hardware**: Up to 592 V100 GPUs for the largest model
- **Optimizer**: Adam with cosine decay
- **Key results**:
  - 76.2% zero-shot accuracy on ImageNet (matches supervised ResNet-50)
  - Outperforms supervised baselines on 16 of 27 datasets in zero-shot setting
  - Reduces effective robustness gap on distribution-shifted ImageNet variants by up to 75% relative to a standard ImageNet model
  - Linear probe with CLIP features achieves 85.4% on ImageNet (ViT-L/14@336), surpassing BiT-M and EfficientNet-L2

## Limitations

- **Compute**: Reaching state-of-the-art performance requires approximately 1,000× more compute than this training run.
- **Fine-grained classification**: Accuracy on FGVC Aircraft and fine-grained texture recognition (DTD) remains below supervised methods.
- **Abstract reasoning**: CLEVRCounts (counting objects) and arithmetic tasks show near-random performance.
- **Out-of-distribution generalization**: On MNIST (handwritten digits), zero-shot CLIP scores only 76% despite near-perfect performance on printed digit datasets, indicating sensitivity to rendering style.
- **Social bias**: The model inherits stereotypical associations from internet text, including gender and racial biases in classification under certain prompts.
- **Task-description gap**: Performance is sensitive to prompt phrasing; the best prompt requires manual engineering per dataset.
