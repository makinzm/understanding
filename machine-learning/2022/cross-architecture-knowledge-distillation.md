# Meta Information

- URL: [Cross-Architecture Knowledge Distillation](https://arxiv.org/abs/2207.05273)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yufan Liu, Jiajiong Cao, Bing Li, Weiming Hu, Jingting Ding, Liang Li (2022). Cross-Architecture Knowledge Distillation. arXiv preprint arXiv:2207.05273.

---

# Cross-Architecture Knowledge Distillation

## Overview

**Knowledge distillation (KD)** is a model compression technique where a smaller *student* model is trained to mimic a larger *teacher* model, transferring the teacher's "dark knowledge" (soft label distributions, intermediate features) into the compact student. Classical KD assumes teacher and student share the same architecture family (CNN-to-CNN, Transformer-to-Transformer). This paper addresses the harder **cross-architecture** setting—specifically, transferring knowledge from a **Transformer teacher** (e.g., ViT, Swin) to a **CNN student** (e.g., ResNet, MobileNet, EfficientNet)—where the feature representations differ fundamentally in spatial structure, attention mechanisms, and token organisation.

**Who would use this:** Practitioners who want to deploy efficient CNN inference models while benefiting from large Transformer-based teachers trained on large datasets. Applicable in image classification, object detection, instance segmentation, and biometric tasks.

## Problem: Why Cross-Architecture KD Is Hard

| Dimension | CNN features | Transformer features |
|---|---|---|
| Spatial layout | Grid of local receptive fields | Sequence of patch tokens |
| Global context | Limited (requires many layers) | Built-in via self-attention |
| Feature channel structure | Channel-major | Token-major |

Directly minimising the $\ell_2$ distance between CNN and Transformer intermediate features fails because the two representations live in incompatible spaces. Prior projection-based methods (e.g., FitNet, CRD) project one space into the other with a single adapter, which cannot simultaneously capture both attention semantics and pixel-level feature geometry.

## Method: CAKD (Cross-Architecture Knowledge Distillation)

The method introduces two complementary projectors plus a multi-view robust training scheme.

### 1. Partially Cross Attention (PCA) Projector

**Goal:** Map CNN student features into the Transformer's *attention* space so the student learns to model global token relationships.

**Input/Output:**

- Input: CNN feature map $f_S \in \mathbb{R}^{B \times C_S \times H \times W}$, Teacher attention map $\text{Attn}_T \in \mathbb{R}^{B \times h \times N \times N}$ and value matrix $V_T \in \mathbb{R}^{B \times h \times N \times d}$ (where $h$ = attention heads, $N = HW/p^2$ patches, $d$ = head dimension).
- Output: Projected attention map $\text{PCAttn}_S \in \mathbb{R}^{B \times h \times N \times N}$ and projected value $V_S \in \mathbb{R}^{B \times h \times N \times d}$.

**Architecture:** Three $3 \times 3$ convolutional layers produce $Q_S, K_S, V_S$ matrices from the CNN feature map. A *probabilistic teacher component mixing* with parameter $p \sim \mathcal{U}(0,1)$ stochastically blends teacher and student query/key matrices during training to stabilise early optimisation.

**Loss (Equation 4):**

$$\mathcal{L}_{\text{proj1}} = \|\text{Attn}_T - \text{PCAttn}_S\|_2^2 + \left\|\frac{V_T V_T^\top}{\sqrt{d}} - \frac{V_S V_S^\top}{\sqrt{d}}\right\|_2^2$$

The first term aligns attention probability distributions; the second aligns value correlation matrices, enforcing that the student learns similar token co-activation patterns.

### 2. Group-wise Linear (GL) Projector

**Goal:** Map CNN student features into the Transformer's *feature* space at pixel level (not just attention level).

**Input/Output:**

- Input: CNN feature map $f_S \in \mathbb{R}^{B \times C_S \times H \times W}$.
- Output: Projected feature $h'_S \in \mathbb{R}^{B \times N \times d_T}$ aligned with teacher hidden state $h_T \in \mathbb{R}^{B \times N \times d_T}$.

**Architecture:** Rather than applying one FC layer per token (which would require $N = HW/p^2$ separate weight matrices), neighbouring $4 \times 4$ patches are *grouped*, reducing the number of FC weight matrices from 196 to 16. Each group shares weights across its member patches. Dropout regularisation prevents overfitting.

**Loss (Equation 6):**

$$\mathcal{L}_{\text{proj2}} = \|h_T - h'_S\|_2^2$$

### 3. Multi-View Robust Training Scheme

**Goal:** Improve student robustness by adversarially training the projectors against transformed (corrupted) views of student features.

**Components:**

- **Multi-view generator $G$**: Applies random augmentations (colour jittering, random crops, patch masking, rotation) with probability $p \geq 0.5$ to produce $m$ transformed student feature views $\{h'^{(k)}_S\}_{k=1}^m$.
- **Adversarial discriminator $D$**: A three-layer MLP that distinguishes teacher features $h_T$ from transformed student features $h'^{(k)}_S$. Updated every 5 epochs.

**Generator Loss (Equation 9):**

$$\mathcal{L}_{\text{MVG}} = \frac{1}{m} \sum_{k=1}^{m} \log\left(1 - D(h'^{(k)}_S)\right)$$

The student is trained to maximise $\mathcal{L}_{\text{MVG}}$ (confuse the discriminator), forcing the projected features to remain teacher-like even under diverse transformations.

### Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{proj1}} + \mathcal{L}_{\text{proj2}} + \lambda \cdot \mathcal{L}_{\text{MVG}}$$

where $\lambda$ is a weighting hyperparameter balancing the adversarial component.

### Algorithm Summary

```
Input: Teacher model T (Transformer), Student model S (CNN)
       Training set D, hyperparameter λ, epochs E

Initialize: PCA projector P1, GL projector P2, discriminator D_adv

For each epoch e in 1..E:
  For each batch (x, y) in D:
    # Forward pass
    h_T, Attn_T, V_T = T(x)          # Teacher intermediate features
    f_S = S_backbone(x)               # Student CNN feature map

    # PCA Projector
    PCAttn_S, V_S = P1(f_S, Attn_T, V_T)   # probabilistic mixing
    L_proj1 = ||Attn_T - PCAttn_S||^2 + ||V_T V_T^T/√d - V_S V_S^T/√d||^2

    # GL Projector
    h'_S = P2(f_S)
    L_proj2 = ||h_T - h'_S||^2

    # Multi-view adversarial training
    {h'^(k)_S} = G(h'_S)              # random augmentations
    L_MVG = (1/m) Σ log(1 - D_adv(h'^(k)_S))

    # Update student + projectors
    L_total = L_proj1 + L_proj2 + λ * L_MVG
    Update S, P1, P2 by ∇L_total

    # Update discriminator every 5 epochs
    if e % 5 == 0:
      L_D = -Σ[log D_adv(h_T) + log(1 - D_adv(h'^(k)_S))]
      Update D_adv by ∇L_D
```

## Experiments

- **Datasets:**
  - **CIFAR-100**: Small-scale image classification, 100 classes. Standard train/test split (50,000 / 10,000).
  - **ImageNet (ILSVRC 2012)**: Large-scale image classification, 1,000 classes (1.28M train / 50K val).
  - **COCO**: Object detection and instance segmentation evaluation.
  - **CelebA-Spoof**: Face anti-spoofing benchmark.
- **Hardware:** 8 NVIDIA Tesla GPUs.
- **Optimizer:** SGD, momentum 0.9, weight decay $10^{-4}$.
- **Training duration:** 200 epochs (CIFAR-100), 120 epochs (ImageNet). All experiments repeated 5 times with different random seeds.
- **Results (selected):**
  - CIFAR-100, Transformer→CNN pairs: average +2.7% Top-1 accuracy over 14 baselines.
  - CIFAR-100, CNN→CNN pairs: up to +2.16% Top-1 accuracy.
  - ImageNet, ResNet50x2 with Transformer teacher: 80.72% Top-1 accuracy, exceeding ViT-B/32 baseline by 2.43%.
  - COCO object detection: +0.3–1.1 AP improvement over baseline student.
  - COCO instance segmentation: +0.3–1.0 AP improvement.
  - CelebA-Spoof face anti-spoofing: −0.3–0.4 EER improvement.

## Comparison with Similar Methods

| Method | Architecture constraint | Feature alignment | Adversarial training |
|---|---|---|---|
| FitNet | Homologous (same family) | Single linear projector | No |
| CRD (Contrastive KD) | Homologous | Contrastive loss | No |
| ReviewKD | Homologous | Multi-level residual | No |
| DeiT | Transformer→Transformer | Distillation token | No |
| MiniLM | Transformer→Transformer | Attention/value relation | No |
| **CAKD (ours)** | **Any architecture** | **Dual projectors (PCA + GL)** | **Yes (MVG)** |

> [!NOTE]
> Unlike MiniLM, which aligns value relation matrices within the Transformer family, CAKD's PCA projector first *translates* CNN feature maps into the Transformer's query/key/value space before applying the value-correlation alignment, enabling cross-architecture transfer.

> [!IMPORTANT]
> The GL projector's group-wise parameter sharing is critical for efficiency: using one FC layer per token would require 196 weight matrices ($14 \times 14$ patches) for $224 \times 224$ ImageNet images, whereas grouping into $4 \times 4$ neighborhoods reduces this to 16 matrices—a $12\times$ parameter reduction with comparable accuracy.

> [!CAUTION]
> The paper reports results on specific teacher–student pairs. Generalisation to very large Transformer teachers (e.g., ViT-L) or very small CNN students (e.g., MobileNetV2 0.35×) was not evaluated. Hyperparameter $\lambda$ and augmentation probabilities may require tuning for new domains.
