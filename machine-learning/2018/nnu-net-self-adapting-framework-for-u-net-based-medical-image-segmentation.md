# Meta Information

- URL: [nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/abs/1809.10486)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Isensee, F., Petersen, J., Klein, A., Zimmerer, D., Jaeger, P. F., Kohl, S., Wasserthal, J., Kohler, G., Norajitra, T., Wirkert, S., & Maier-Hein, K. H. (2018). nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation. arXiv:1809.10486.

# nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation

nnU-Net ("no-new-Net") is a self-configuring medical image segmentation framework that automatically adapts preprocessing, training, and inference pipelines to arbitrary segmentation datasets without manual tuning. The central claim is that non-architectural aspects of a segmentation system (resampling strategy, normalization, patch size, data augmentation, loss function, ensemble strategy) have far greater impact on performance than novel architectural designs.

The framework was evaluated on the **Medical Segmentation Decathlon** challenge, which requires a single algorithm to generalize across ten distinct segmentation tasks (different anatomical structures, imaging modalities, dataset sizes) without task-specific adjustments.

> [!NOTE]
> "The contribution of non-architectural aspects in segmentation methods is much more impactful, but at the same time also severely underestimated." — Isensee et al.

## Network Architectures

nnU-Net uses three vanilla U-Net variants. All share two modifications from the original U-Net: **Leaky ReLU** activations (negative slope $10^{-2}$) instead of standard ReLU, and **instance normalization** instead of batch normalization.

| Variant | Input | Use case |
|---|---|---|
| 2D U-Net | 2D slices from 3D volume; $x \in \mathbb{R}^{B \times C \times H \times W}$ | Highly anisotropic datasets (large in-plane vs. through-plane spacing) |
| 3D U-Net | 3D volumetric patches; $x \in \mathbb{R}^{B \times C \times D \times H \times W}$ | Isotropic or near-isotropic volumetric data |
| U-Net Cascade | Two-stage: 3D U-Net on downsampled → 3D U-Net on full resolution | Datasets whose median shape exceeds $4\times$ the voxel budget of the 3D U-Net |

In the **U-Net Cascade**, Stage 1 processes a low-resolution version of the full image. Stage 2 receives the full-resolution image concatenated with the Stage 1 class probability maps as additional input channels, refining the coarse prediction. This cascade is triggered automatically; datasets like Heart, Liver, Lung, and Pancreas activate it due to their large volumes.

### Dynamic Topology Adaptation

Network depth and pooling layers are determined from the dataset's spatial characteristics:

- Pooling continues along each axis until the corresponding spatial dimension of the feature map falls below 8 voxels.
- The 2D U-Net uses 256×256 patches with batch size 42 and 30 base feature maps by default; these scale with the median in-plane image size.
- The 3D U-Net uses patches up to $128^3$ voxels with batch size 2; aspect ratios are matched to the dataset's geometry.
- Total voxels processed per optimizer step are constrained to 5% of total dataset volume, with batch size increased for small datasets.

> [!IMPORTANT]
> Network topology (number of pooling stages, feature map counts, patch size) varies dramatically across tasks. For example, Hippocampus (small structures, MRI) and Liver (large organ, CT) end up with fundamentally different network depths and patch sizes — all determined automatically.

## Preprocessing

### Cropping

Background voxels (zero-valued borders) are removed to reduce computation. Bounding boxes are computed per patient and applied uniformly.

### Resampling

All images are resampled to the **median voxel spacing** of the training set:
- Image data: third-order spline interpolation.
- Segmentation labels: nearest-neighbor interpolation.

For U-Net Cascade Stage 1, datasets are additionally downsampled by iteratively doubling voxel spacing until the volume fits within the 3D U-Net's voxel budget.

### Normalization

- **CT images**: Intensities are clipped to the $[0.5, 99.5]$ percentile range of the dataset's foreground voxels, then z-score normalized using the dataset-wide mean and standard deviation.
- **MRI and other modalities**: Per-patient z-score normalization (mean and std computed per scan, not dataset-wide).
- If cropping reduces average patient volume by $\geq 25\%$, normalization statistics are computed only within non-zero mask regions.

## Training Procedure

All models are trained from scratch using **5-fold cross-validation**. Each fold produces a separate model; the five models are later ensembled for inference.

### Loss Function

A combined Dice and cross-entropy loss is used:

```math
\begin{align}
  \mathcal{L} = \mathcal{L}_{dc} + \mathcal{L}_{CE}
\end{align}
```

The Dice loss is:

```math
\begin{align}
  \mathcal{L}_{dc} = -\frac{2}{|K|} \sum_{k \in K} \frac{\sum_{i \in I} u_i^k v_i^k}{\sum_{i \in I} u_i^k + \sum_{i \in I} v_i^k}
\end{align}
```

where $u \in \mathbb{R}^{|I| \times |K|}$ is the softmax output of the network, $v \in \{0,1\}^{|I| \times |K|}$ is the one-hot ground truth, $i$ indexes voxels, and $k$ indexes classes.

> [!NOTE]
> Cross-entropy alone is sensitive to class imbalance (rare foreground structures); Dice loss alone can be unstable when foreground is absent. The sum of both losses empirically stabilizes training across varied tasks.

### Optimizer and Schedule

- **Optimizer**: Adam, initial learning rate $3 \times 10^{-4}$.
- **Epoch**: 250 mini-batches.
- **Learning rate decay**: Reduced by factor 5 when the exponential moving average of training loss fails to improve by $\geq 5 \times 10^{-3}$ over 30 epochs.
- **Early stopping**: Training ends when validation loss fails to improve by $\geq 5 \times 10^{-3}$ over 60 epochs, or when learning rate drops below $10^{-6}$.

### Data Augmentation (On-the-fly)

The same augmentation parameters are applied across all datasets per network type:
- Random rotations (full 3D for isotropic data; 2D slice-wise for anisotropic data where the longest patch axis exceeds the shortest by $> 2\times$)
- Random scaling
- Random elastic deformations
- Gamma correction (contrast adjustment)
- Random mirroring

For the **U-Net Cascade Stage 2**, random morphological operations (erosion, dilation, opening, closing) and random connected component removal are applied to the Stage 1 predictions before passing them as input channels, preventing the stage 2 network from co-adapting to perfect stage 1 outputs.

### Patch Sampling

At least 33% of voxels in each training batch must contain at least one foreground class voxel, selected randomly from available foreground classes. This ensures rare structures (e.g., small tumors) are sufficiently represented in training.

## Inference

1. **Patch-based prediction**: Patches are extracted with 50% overlap.
2. **Gaussian weighting**: Predictions near patch borders (lower reliability) are down-weighted using a Gaussian centered on the patch center.
3. **Test-time augmentation (TTA)**: Predictions are aggregated over all valid mirror augmentations (up to $2^3 = 8$ for 3D), yielding up to 64 predictions per central voxel when combined across the 5-fold ensemble.
4. **Ensemble**: The 5 cross-validation models are averaged.

## Postprocessing

Connected component analysis is performed on cross-validation predictions to determine, per class, whether the structure always appears as a single connected component in the training data. For such classes, only the largest connected component is retained in test predictions, removing spurious small detections.

## Model Selection

All pairwise ensembles of the three architectures (2D, 3D, Cascade) are evaluated on cross-validation. The best-performing single model or ensemble is selected automatically for submission, without requiring separate test set evaluations.

## Comparison with Related Methods

| Aspect | Original U-Net (2015) | Attention U-Net | Dense U-Net | nnU-Net |
|---|---|---|---|---|
| Normalization | Batch normalization | Batch normalization | Batch normalization | Instance normalization |
| Activation | ReLU | ReLU | ReLU | Leaky ReLU |
| Architecture change | Baseline encoder-decoder | Adds attention gates | Adds dense connections | No structural change |
| Configuration | Manual per task | Manual per task | Manual per task | Fully automatic |
| Dimensionality | 2D (original) | 2D/3D | 2D/3D | 2D + 3D + Cascade auto-selected |
| Loss | Cross-entropy | Dice | Dice | Dice + Cross-entropy |

> [!TIP]
> The original U-Net paper: Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. https://arxiv.org/abs/1505.04597

> [!TIP]
> V-Net (Milletari et al., 2016) introduced the volumetric 3D U-Net with Dice loss: https://arxiv.org/abs/1606.04797

## Applicability

nnU-Net is best suited for:
- **Who**: Researchers and practitioners applying deep learning to medical image segmentation without domain-specific architecture expertise.
- **When**: When a new segmentation dataset arrives and the user wants a strong baseline without manual hyperparameter search.
- **Where**: Any voxel-wise segmentation task on 2D or 3D medical images (CT, MRI, ultrasound, etc.).

The framework is less suitable when inference latency is critical (TTA with 64 predictions per voxel is expensive) or when training compute is severely limited (three networks + 5-fold CV requires significant resources).

# Experiments

- **Datasets**: All seven Medical Segmentation Decathlon phase 1 tasks:
  - Brain Tumour (MRI, multi-channel, from BRATS challenge)
  - Heart (MRI, large volume — triggers cascade)
  - Liver (CT, large volume — triggers cascade)
  - Hippocampus (MRI, small structure, no cascade)
  - Prostate (MRI, multi-channel, no cascade)
  - Lung (CT, large volume — triggers cascade)
  - Pancreas (CT, large volume — triggers cascade)
  - Plus three undisclosed phase 2 datasets (evaluated without algorithm changes)
- **Hardware**: Not specified in the paper.
- **Optimizer**: Adam, initial LR $3 \times 10^{-4}$, with decay schedule and early stopping.
- **Results**:
  - At submission time, nnU-Net ranked first in mean Dice score on all Decathlon tasks except Brain Tumour class 1 (where a known distribution shift between BRATS challenge years degraded performance).
  - 3D U-Net Cascade outperformed 2D approaches on large-volume datasets (Heart, Liver, Lung, Pancreas).
  - 3D U-Net without cascade was best for smaller datasets (Hippocampus, Prostate).
  - Ensembling across architectures further improved robustness on cross-validation.

> [!CAUTION]
> The paper does not provide a controlled ablation study quantifying the individual contribution of each design choice (resampling, normalization, loss, TTA, etc.). The authors acknowledge this as future work.
