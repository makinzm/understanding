# Meta Information

- URL: [[2308.10832] EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition](https://arxiv.org/abs/2308.10832)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Berton, G., Trivigno, G., Caputo, B., & Masone, C. (2023). EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition. ICCV 2023.

---

# Abstract

Visual Place Recognition (VPR) determines the geographic location of a query photograph by retrieving the most visually similar image from a geotagged database. Existing methods are vulnerable to large viewpoint shifts between query and database images because their training data only pairs images captured from identical or near-identical orientations. EigenPlaces is a training protocol that overcomes this by using Singular Value Decomposition on geographic coordinates to identify building facades, then explicitly grouping training images that observe the same facade from spatially diverse lateral positions. This forces the backbone network to embed identical places into nearby descriptor vectors regardless of horizontal viewing angle. EigenPlaces achieves state-of-the-art on most of the 16 evaluated VPR benchmarks while requiring 50% smaller descriptors and 60% less GPU memory than the best competing method (MixVPR).

> [!NOTE]
> `descriptor` means the fixed-length vector output by the network that represents an image for retrieval; `backbone` refers to the convolutional neural network (e.g., ResNet-50) that extracts features from the input image before pooling and projection.
>
> `facade` refers to the front of a building that faces the street; `lateral position` means the position of the camera along the road, which affects the angle at which it views the facade.

---

# 1. Introduction

**Problem.** Given a query image $q$ and a geo-tagged database $\mathcal{D} = \{(i_k, p_k)\}$ of $|\mathcal{D}|$ images with known UTM positions $p_k$, VPR retrieves the database image whose descriptor is nearest to the query descriptor and reports its position as the estimated location. The core challenge is that images of the same place captured from different horizontal angles (viewpoint shift) can have very different pixel-level appearances.

**Why existing training fails.** Prior classification-based methods such as [CosPlace](https://github.com/gmberton/CosPlace) assign each training class a single orientation angle: all images in a class face the same direction, so the model never sees the same building from two different sides. Metric-learning methods such as [NetVLAD](https://github.com/Relja/netvlad) mine positives from GPS proximity, which also tends to select same-viewpoint pairs. Neither strategy explicitly teaches viewpoint invariance.

**EigenPlaces solution.** By computing SVD (singular value decomposition) on the 2-D UTM coordinates of images within a small geographic cell, the method identifies the dominant road axis ($V_0$) and the perpendicular direction toward building facades ($V_1$). A focal point $c_i$ is placed along $V_1$ at distance $D$ from the cell centre. Images are then grouped by which focal point they face, so one class contains images shooting the same facade from left, centre, and right. Minimising CosFace loss on such classes teaches the network to produce identical descriptors across these viewpoints.

---

# 2. Related Work

| Method | Training Strategy | Viewpoint Diversity in Classes |
|--------|-------------------|-------------------------------|
| NetVLAD | Weakly supervised metric learning (GPS triplets) | None (GPS positives tend to share viewpoints) |
| CosPlace | Classification with orientation-grouped cells | None (all images face same direction per class) |
| [GSV-Cities](https://github.com/amaralibey/gsv-cities) / Conv-AP | Classification with pre-defined classes | Low (pre-defined clusters at similar angles) |
| [MixVPR](https://github.com/amaralibey/MixVPR) | Classification + feature mixing augmentation | Low (same orientation-based grouping) |
| **EigenPlaces** | Classification with SVD-based focal-point classes | **Explicit** (lateral spread along road) |

Post-processing methods ([SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork?tab=readme-ov-file), [LoFTR](https://github.com/zju3dv/LoFTR), [Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD)) operate on a retrieved shortlist and are complementary: they refine ranking but cannot help if the correct match was never retrieved. EigenPlaces targets the retrieval stage.

---

# 3. Method

## 3.1 Map Partition

**Input.** All training images with UTM positions $(e_j, n_j)$ (east, north).

**Output.** A sequence of training batches, each containing a non-overlapping subset of images grouped into classes by their angle toward a focal point.

**Procedure.**
1. Overlay a regular grid of side $M = 15$ metres on the map.
2. Select every $N$-th cell in both latitude and longitude ($N = 3$), yielding a non-overlapping subset of $\frac{1}{N^2}$ of all cells per epoch.
3. The subset shifts by one cell each epoch, so all cells are visited over $N^2 = 9$ epochs without ever placing spatially adjacent cells in the same training batch epoch.

The $M=15$ m scale is chosen so that a single cell spans roughly one building face, ensuring that images in the same cell all observe a common facade.

> [!NOTE]
> This is how to create training classes without any manual labelling: each cell is a class, and all images in that cell belong to the same class. The SVD-based focal point construction (next section) then ensures that images in the same class have diverse viewpoints toward the same facade.

## 3.2 EigenPlaces Class Construction via SVD

**Input.** A cell $i$ containing $p$ images at UTM positions $X_i \in \mathbb{R}^{p \times 2}$.

> [!NOTE]
> Any cell includes only images that are geographically close (within 15 m), so they all observe the same building facade but from different angles along the road.

**Output.** Two classes of images in cell $i$: one for lateral viewpoint diversity, one for frontal viewpoint diversity, which is used for training labels in the CosFace loss.

**Algorithm.**

```
Procedure EigenPlaces_ClassConstruct(X_i, D):
  # 1. Centre the coordinates
  X̂_i = X_i - E[X_i]          # E[X_i] is the centroid, shape (1,2)

  # 2. Singular Value Decomposition
  U, Σ, V^T = SVD(X̂_i)        # V ∈ R^{2×2}, columns are eigenvectors

  V_0 = V[:, 0]               # 1st principal component ≈ road direction
  V_1 = V[:, 1]               # 2nd principal component ⊥ road, toward facades

  # 3. Lateral focal point (for multi-view datasets)
  c_lat = E[X_i] + D * V_1    # (Eq. 1)

  # 4. Frontal focal point (for front-facing camera datasets)
  c_front = E[X_i] + D * V_0

  # 5. For each focal point c ∈ {c_lat, c_front}:
  For each image j in cell i:
    Δe_j = c[east]  - X_i[j, east]    # (Eq. 2)
    Δn_j = c[north] - X_i[j, north]   # (Eq. 3)
    α_j  = arctan(Δe_j / Δn_j)        # (Eq. 4)
    Assign image j to the class whose mean angle α is closest to α_j.

  Return two classes per cell (lateral class, frontal class)
```

**Geometric interpretation.** Because each image $j$ is at a different position relative to the focal point $c$, its facing angle $\alpha_j$ differs from its neighbours. Images on the left side of the road face the facade diagonally from the left; images on the right face it from the right. Both end up in the same class, creating the desired viewpoint diversity.

**Effect of focal distance $D$.**
- $D \to \infty$: all images share the same angle $\alpha_j \approx \text{const}$, which degenerates to orientation-grouped classes identical to CosPlace.
- $D = 0$: focal point is the cell centroid; images are grouped by whether they look inward or outward, with no road-direction information.
- Empirically, $D \in [10, 20]$ m is optimal; $D = 20$ m is used for final models.

## 3.3 Training

**Architecture.**
- **Backbone $f$:** [VGG-16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html) or [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html), pretrained on ImageNet.
- **Pooling:** Generalized Mean (GeM) aggregates spatial feature maps into a fixed-length vector.
- **Projection head:** A fully-connected layer maps the pooled features to a $d$-dimensional descriptor ($d \in \{128, 512, 2048\}$).
- **Input:** RGB image of any resolution (resized during training).
- **Output:** $\ell_2$-normalised descriptor $x \in \mathbb{R}^d$.

**CosFace loss ([Large Margin Cosine Loss](https://arxiv.org/abs/1801.09414)).**

For each loss component (lateral or frontal), the normalised class weight matrix $W \in \mathbb{R}^{d \times C}$ (one column per training class) is maintained. For a sample $x_i$ with ground-truth class $y_i$:

$$\cos\theta_j = W_j^\top x_i, \quad W = \frac{W^*}{\|W^*\|}, \quad x = \frac{x^*}{\|x^*\|}$$

$$\mathcal{L}_\text{lat} = \frac{1}{N} \sum_i -\log \frac{e^{s(\cos\theta_{y_i} - m)}}{e^{s(\cos\theta_{y_i} - m)} + \sum_{j \neq y_i} e^{s\cos\theta_j}}$$

where $s$ is a scale factor and $m > 0$ is a cosine margin that enforces inter-class separation.

**Combined loss.** Each cell produces two classes; two CosFace classifiers ($W_\text{lat}$, $W_\text{front}$) are trained jointly:

$$\mathcal{L} = \mathcal{L}_\text{lat}(f, W_\text{lat}) + \mathcal{L}_\text{front}(f, W_\text{front})$$

**Training hyperparameters.**

| Hyperparameter | Value |
|---|---|
| Training dataset | SF-XL (360° panoramas cropped by orientation) |
| Iterations | 200,000 |
| Batch size | 128 (64 images per loss component) |
| Optimiser | Adam |
| Learning rate | $10^{-5}$ |
| Cell side $M$ | 15 m |
| Partition stride $N$ | 3 |
| Focal distance $D$ | 10 m (default), 20 m (best) |
| Data augmentation | Colour jitter, random crop |

**Comparison with CosPlace.** [CosPlace](https://github.com/gmberton/CosPlace) also uses geographic cells and CosFace, but groups images by their recorded GPS heading angle, so all images in a class face the same absolute direction. EigenPlaces ignores GPS heading and instead uses the SVD-inferred facade direction to build classes that span a range of viewing angles toward the same facade.

---

# 4. Experiments

## Datasets

**Multi-view** (queries may approach from any direction):

| Dataset | Queries | Database | Domain Shift |
|---------|---------|----------|--------------|
| Pitts30k | 6,816 | 10,000 | None |
| Pitts250k | 8,280 | 83,952 | None |
| Tokyo 24/7 | 315 | 75,984 | Day/Night |
| San Francisco Landmark | 598 | 1,040,000 | Viewpoint, distance |
| SF-XL test v1 | 1,000 | 2,800,000 | Night, viewpoint |
| SF-XL test v2 | 598 | 2,800,000 | Viewpoint |
| AmsterTime | 1,231 | 1,231 | Long-term (historical photos) |
| Eynsham | 24,000 | 24,000 | None |

**Frontal-view** (queries captured by forward-facing dashcam):

| Dataset | Queries | Database | Domain Shift |
|---------|---------|----------|--------------|
| MSLS Val | 740 | 18,871 | Day/Night |
| Nordland | 27,592 | 27,592 | Summer/Winter |
| St Lucia | 1,464 | 1,549 | None |
| SVOX Night/Overcast/Rain/Snow/Sun | ~870 each | 17,000 | Weather/Lighting |

**Evaluation metric.** Recall@1: the fraction of queries for which the top-1 retrieved database image is within 25 m of ground truth (10 frames for Nordland, image-pair accuracy for AmsterTime).

## Hardware

- Training: single NVIDIA RTX 3090 (24 GB), ~24 hours for full training
- Peak GPU memory: < 7 GB for ResNet-50 with 2048-D descriptors

## Key Results

EigenPlaces (ResNet-50, 512-D) achieves Recall@1 of **91.9** on Pitts30k, **89.8** on Tokyo 24/7, and **89.5** on MSLS Val — setting the best or joint-best on 12 of 16 datasets at comparable descriptor size.

MixVPR with 4096-D descriptors outperforms EigenPlaces (2048-D) on Nordland (76.2 vs 71.2) and some SVOX conditions, but requires 2× larger descriptors and >18 GB GPU memory vs. <7 GB for EigenPlaces.

## Ablation: Loss Components

Architecture: ResNet-18, 512-D, evaluated on Recall@1.

| $\mathcal{L}_\text{lat}$ | $\mathcal{L}_\text{front}$ | Pitts30k | Tokyo 24/7 | MSLS Val | St Lucia | Average |
|:---:|:---:|---:|---:|---:|---:|---:|
| ✓ | ✗ | 90.2 | 80.0 | 83.1 | 97.3 | 87.6 |
| ✗ | ✓ | 89.5 | 78.1 | 85.8 | 99.3 | 88.2 |
| ✓ | ✓ | **90.5** | **82.2** | **86.2** | **99.0** | **89.5** |

The lateral loss drives viewpoint robustness on multi-view datasets (Tokyo +1.9 pp); the frontal loss improves sequence-based datasets (MSLS +2.7 pp, St Lucia +2.0 pp); combining both gives the best balanced performance.

## Ablation: Focal Distance $D$

Architecture: ResNet-18, 512-D.

| $D$ (m) | Pitts30k | Tokyo 24/7 | MSLS Val | St Lucia | Average |
|---:|---:|---:|---:|---:|---:|
| 0 | 89.4 | 74.0 | 82.6 | 98.4 | 86.1 |
| 10 | 90.5 | 82.2 | 86.2 | 99.0 | 89.5 |
| **20** | **90.3** | **84.4** | **86.1** | **99.5** | **90.1** |
| 30 | 90.3 | 82.9 | 85.0 | 99.5 | 89.4 |
| 50 | 90.4 | 83.8 | 85.9 | 99.5 | 89.9 |

$D = 0$ collapses performance (Tokyo drops 8 pp), confirming that road-perpendicular focal displacement is essential. Performance stabilises for $D \in [10, 50]$ m, making the hyperparameter easy to set in practice.

---

# 5. Conclusions

EigenPlaces improves viewpoint robustness in VPR by reformulating the class-construction step of classification-based training. Instead of grouping images by their GPS heading (CosPlace) or by pre-defined clusters (MixVPR, Conv-AP), it uses SVD on within-cell geographic coordinates to locate building facades and then assigns training classes so that images viewing the same facade from different lateral positions are always co-classified. This geometric insight requires no additional labels beyond GPS position. The resulting models deliver state-of-the-art Recall@1 on the majority of 16 benchmarks covering multi-view, weather, and seasonal domain shifts, while using smaller descriptors and less GPU memory than competing methods.

**Applicability.** EigenPlaces is most beneficial when deploying VPR in environments where the query camera may approach a landmark from a different angle than the database images (e.g., urban robot navigation, AR localisation). It is less necessary when the deployment camera is always front-facing and viewpoint shift is minimal.
