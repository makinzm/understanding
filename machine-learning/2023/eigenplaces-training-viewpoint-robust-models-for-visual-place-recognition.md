# Meta Information

- URL: [[2308.10832] EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition](https://arxiv.org/abs/2308.10832)
- ar5iv: [https://ar5iv.labs.arxiv.org/html/2308.10832](https://ar5iv.labs.arxiv.org/html/2308.10832)
- LICENSE: [Deed - Attribution 4.0 International - Creative Commons](https://creativecommons.org/licenses/by/4.0/)
- Code: [https://github.com/gmberton/EigenPlaces](https://github.com/gmberton/EigenPlaces)
- Benchmark: [https://github.com/gmberton/auto_VPR](https://github.com/gmberton/auto_VPR)

> [!CAUTION]
> NOTE comments are personal understanding and may contain errors.

---

# Title and Authors

**Title:** EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition

**Authors:**
- Gabriele Berton (Politecnico di Torino)
- Gabriele Trivigno (Politecnico di Torino)
- Barbara Caputo (Politecnico di Torino)
- Carlo Masone (Politecnico di Torino)

Contact: {gabriele.berton, gabriele.trivigno, barbara.caputo, carlo.masone}@polito.it

**Venue:** ICCV 2023

---

# Abstract

Visual Place Recognition (VPR) aims to predict an image's location using visual features via image retrieval, matching queries against geotagged photo databases using learned global descriptors. The challenge of recognizing places from different viewpoints is addressed through a novel training method called EigenPlaces that clusters training data to explicitly present multiple viewpoints of identical points of interest without requiring extra supervision. The method involves selecting points of interest based on geographical data distribution using Singular Value Decomposition (SVD). Results demonstrate that EigenPlaces outperforms previous state-of-the-art on multiple datasets while requiring 60% less GPU memory for training and using 50% smaller descriptors.

---

# 1. Introduction

Visual Place Recognition (VPR) predicts photograph locations using visual features through image retrieval approaches. Deep neural networks extract global descriptors from query and database images, followed by nearest neighbor search. While existing approaches address scalability and illumination challenges, recognizing images with heavy viewpoint shifts remains an open problem. Post-processing methods using spatial verification or local feature matching require successful initial retrieval.

The paper argues that improving the robustness to viewpoint shifts already at the retrieval stage requires training networks to extract descriptors more invariant to perspective changes. EigenPlaces accomplishes this by clustering training data so each class contains multiple viewpoints depicting the same scene, forcing models to learn viewpoint-robust global descriptors.

**Key Contributions:**
1. Novel training protocol rendering models robust to viewpoint changes
2. Rigorous VPR benchmark on comprehensive dataset collection (16 datasets)
3. State-of-the-art results on numerous datasets with 60% less GPU memory to train and 50% more compact descriptors compared to previous best methods

---

# 2. Related Work

## 2.1 Visual Place Recognition

Early VPR work focused on local features (SIFT, SURF, RootSIFT) before deep learning adoption. CNN-based approaches emerged with classification-trained features for landmark retrieval. Key works:

- **NetVLAD** [5]: Introduced specialized pooling for urban VPR using most-similar database images (likely same viewpoint)
- **CosPlace** [8]: Classification-based training inspired by face recognition; creates classes with identical orientation
- **Conv-AP** [?]: Large-scale GSV-cities training
- **MixVPR** [2]: Uses pre-defined classes with similar viewpoints, large descriptor size (4096-D)

Prior work has not explicitly addressed viewpoint-invariance for VPR at the retrieval stage.

## 2.2 Viewpoint Invariant Matching

Post-processing methods refine retrieval results using local features, assuming shortlists contain at least one positive match:
- **SuperGlue** [44]: Learning feature matching with graph neural networks
- **DELG** [12]: Unifying deep local and global features
- **LoFTR** [46]: Detector-free local feature matching with transformers
- **Patch-NetVLAD** [22]: Multi-scale fusion of locally-global descriptors
- **GeoWarp** [9]: Viewpoint invariant dense matching

These methods are complementary to EigenPlaces, which improves robustness at the retrieval stage rather than re-ranking.

---

# 3. Method

## 3.1 Map Partition

Maps are divided into cells of size M×M where M = 15 meters. Cells are grouped into non-overlapping subsets ensuring no visual overlap between cells in the same subset. This prevents placing images of identical locations in different classes.

The method takes one cell every N cells in latitude and longitude directions (N = 3), evaluating 1/N² of cells per epoch with the partition shifting after each epoch. This provides training diversity while managing computational cost.

Unlike CosPlace, EigenPlaces relies solely on image positions without using orientation for class construction.

## 3.2 EigenPlaces Class Construction via SVD

Given a cell with p images, the UTM coordinates (east, north) are organized in matrix:

```math
X_i \in \mathbb{R}^{p \times 2}
```

Singular Value Decomposition (SVD) is computed on the centered coordinate matrix. The SVD gives two principal components:
- **V₀** (first principal component): estimates the direction of the road
- **V₁** (second principal component): perpendicular to road, points toward points of interest (e.g., building facades)

**Focal Point Definition (Equation 1):**

```math
c_i = \mathbb{E}[X_i] + D \times V_1
```

Where:
- $\mathbb{E}[X_i]$ = center of mass of the cell's image positions
- $V_1$ = second principal component from SVD (perpendicular to road direction)
- $D$ = focal distance parameter

Images facing the focal point $c_i$ are grouped into a single class.

**Interpretation of focal distance D:**
- $D \to \infty$: selects images with identical geographical orientation (similar to CosPlace)
- $D \to 0$: selects images facing opposite directions toward the mean position

**Frontal-view datasets:** For datasets with only front-facing cameras, a second focal point is generated along the first principal component $V_0$ (aligned with road direction) to create frontal-view classes.

**Assumption:** The method assumes images align along roads. At crossroads or non-linear distributions, eigenvectors may not perfectly align with actual roads, but the approach still provides training samples viewing the same point from different perspectives.

## 3.3 Training

For each image $j$ with UTM coordinates $x_j = (e_j, n_j)$, the orientation angle toward the focal point $c_i = (e_c, n_c)$ is computed:

**Equations 2-4 (Angle Calculation):**

```math
\Delta e_j = e_c - e_j
```

```math
\Delta n_j = n_c - n_j
```

```math
\alpha_j = \arctan\!\left(\frac{\Delta e_j}{\Delta n_j}\right)
```

Images with orientation closest to $\alpha_j$ are selected. Since $\alpha_j$ varies across images within the class (because each image is at a different position relative to the focal point), this enables depicting the same place from different viewpoints.

### Loss Function: CosFace (Large Margin Cosine Loss)

**Equation 3 - Lateral Loss:**

```math
\mathcal{L}_{\text{lat}} = \frac{1}{N} \sum_i -\log \frac{e^{s(\cos\theta_{y_i} - m)}}{e^{s(\cos\theta_{y_i} - m)} + \sum_{j \neq i} e^{s \cos\theta_j}}
```

**Equation 4 - Cosine similarity constraints:**

```math
\cos\theta_j = W_{\text{lat},j}^{\top} x_i
```

```math
W_{\text{lat}} = \frac{W^*}{\|W^*\|}
```

```math
x = \frac{x^*}{\|x^*\|}
```

Where:
- $s$ = scale parameter
- $m$ = margin parameter
- $W_{\text{lat}}$ = normalized class weight vectors
- $x$ = normalized feature descriptor
- $\theta_{y_i}$ = angle between descriptor and its ground-truth class weight
- $\theta_j$ = angle between descriptor and $j$-th class weight

### Two Loss Components

Each cell generates two classes and two corresponding losses:

1. **Lateral Loss ($\mathcal{L}_{\text{lat}}$):** Focal point placed on second principal component ($V_1$, perpendicular to road). Handles multi-view datasets by grouping images facing the same facade from different lateral positions.

2. **Frontal Loss ($\mathcal{L}_{\text{front}}$):** Focal point placed on first principal component ($V_0$, along road direction). Handles front-facing datasets.

**Equation 5 - Final Combined Loss:**

```math
\mathcal{L} = \mathcal{L}_{\text{lat}}(f, W_{\text{lat}}) + \mathcal{L}_{\text{front}}(f, W_{\text{front}})
```

Where $f$ is the backbone feature extractor.

---

# 4. Experiments

## 4.1 Datasets

### Multi-view Datasets

| Dataset | Queries | Database | Orientation | Scenery | Domain Shift |
|---------|---------|----------|-------------|---------|--------------|
| AmsterTime | 1,231 | 1,231 | multi-view | urban | long-term |
| Eynsham | 24k | 24k | multi-view | urban & country | none |
| Pitts30k | 6.8k | 10k | multi-view | urban | none |
| Pitts250k | 8.3k | 84k | multi-view | urban | none |
| Tokyo 24/7 | 315 | 76k | multi-view | urban | day/night |
| San Francisco Landmark | 598 | 1.04M | multi-view | urban | viewpoint/distance |
| SF-XL test v1 | 1,000 | 2.8M | multi-view | mostly urban | viewpoint, night |
| SF-XL test v2 | 598 | 2.8M | multi-view | mostly urban | viewpoint |

### Frontal-view Datasets

| Dataset | Queries | Database | Orientation | Scenery | Domain Shift |
|---------|---------|----------|-------------|---------|--------------|
| MSLS Val | 740 | 18.9k | frontal-view | mostly urban | day/night |
| Nordland | 27,592 | 27,592 | frontal-view | country | summer/winter |
| St Lucia | 1,464 | 1,549 | frontal-view | suburb | none |
| SVOX Night | 823 | 17k | frontal-view | urban | day/night |
| SVOX Overcast | 872 | 17k | frontal-view | urban | weather |
| SVOX Rain | 937 | 17k | frontal-view | urban | weather |
| SVOX Snow | 870 | 17k | frontal-view | urban | weather |
| SVOX Sun | 854 | 17k | frontal-view | urban | weather |

## 4.2 Implementation Details

### Architecture
- **Backbone:** VGG-16 or ResNet-50
- **Pooling:** Generalized Mean (GeM)
- **Output:** Global descriptors via fully connected layer (variable dimensionality: 128, 512, 2048)

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Iterations | 200,000 |
| Batch size | 128 images (64 per loss component) |
| Learning rate | 1e-5 |
| Optimizer | Adam [28] |
| Cell size M | 15 meters |
| Partition parameter N | 3 |
| Focal distance D | 10m (default), 20m (optimal) |
| Training dataset | SF-XL (360° panoramas with orientation crops) |

### Data Augmentation
- Color jittering
- Random cropping

### Evaluation Metric

**Recall@N** = percentage of queries for which at least one of the top-N predicted database images is within the threshold distance from the ground-truth location.

- Default threshold: 25 meters
- Nordland: 10 frames distance
- AmsterTime: Image pair matching

## 4.3 Comparison with Previous Work

### Multi-view Datasets - Recall@1 (Table 3)

| Method | Backbone | Dim | AmsterTime | Eynsham | Pitts30k | Pitts250k | Tokyo 24/7 | SF Landmark | SF-XL v1 | SF-XL v2 |
|--------|----------|-----|-----------|---------|----------|-----------|-----------|------------|---------|---------|
| NetVLAD | VGG-16 | 4096 | 16.3 | 77.7 | 85.0 | 85.9 | 69.8 | 79.1 | 40.0 | 76.9 |
| SFRS | VGG-16 | 4096 | 29.7 | 72.3 | 89.1 | 90.4 | 80.3 | 83.1 | 50.3 | 83.8 |
| CosPlace | VGG-16 | 512 | 38.7 | 88.3 | 88.4 | 89.7 | 81.9 | 80.8 | 65.9 | 83.1 |
| **EigenPlaces** | VGG-16 | 512 | **38.0** | **89.4** | **89.7** | **91.2** | **82.2** | **83.8** | **69.4** | **86.3** |
| CosPlace | ResNet-50 | 128 | 39.9 | 88.6 | 89.0 | 89.6 | 81.0 | 82.9 | 69.1 | 86.5 |
| MixVPR | ResNet-50 | 128 | 23.1 | 84.8 | 87.7 | 88.7 | 56.8 | 66.9 | 36.7 | 68.4 |
| **EigenPlaces** | ResNet-50 | 128 | **37.9** | **89.1** | **89.6** | **90.2** | **79.4** | **85.5** | **72.4** | **86.6** |
| CosPlace | ResNet-50 | 512 | 46.4 | 89.9 | 90.2 | 91.7 | 89.5 | 85.6 | 76.7 | 89.0 |
| Conv-AP | ResNet-50 | 512 | 28.4 | 86.2 | 89.1 | 90.4 | 61.3 | 68.4 | 41.8 | 64.0 |
| MixVPR | ResNet-50 | 512 | 35.8 | 87.6 | 90.4 | 93.0 | 78.4 | 79.4 | 57.7 | 84.3 |
| **EigenPlaces** | ResNet-50 | 512 | **45.7** | **90.5** | **91.9** | **93.5** | **89.8** | **89.5** | **82.6** | **90.6** |
| CosPlace | ResNet-50 | 2048 | 47.7 | 90.0 | 90.9 | 92.3 | 87.3 | 87.1 | 76.4 | 88.8 |
| Conv-AP | ResNet-50 | 2048 | 31.3 | 86.6 | 90.4 | 92.3 | 71.1 | 71.7 | 47.8 | 68.1 |
| **EigenPlaces** | ResNet-50 | 2048 | **48.9** | **90.7** | **92.5** | **94.1** | **93.0** | **89.6** | **84.1** | **90.8** |
| Conv-AP | ResNet-50 | 4096 | 33.9 | 87.5 | 90.5 | 92.3 | 76.2 | 73.7 | 47.5 | 74.4 |
| MixVPR | ResNet-50 | 4096 | 40.2 | 89.4 | 91.5 | 94.1 | 85.1 | 83.8 | 71.1 | 88.5 |
| Conv-AP | ResNet-50 | 8192 | 35.0 | 87.6 | 90.5 | 92.6 | 72.1 | 74.4 | 49.3 | 75.8 |

### Frontal-view Datasets - Recall@1 (Table 4)

| Method | Backbone | Dim | MSLS Val | Nordland | St Lucia | SVOX Night | SVOX Overcast | SVOX Rain | SVOX Snow | SVOX Sun |
|--------|----------|-----|----------|----------|----------|-----------|--------------|----------|----------|---------|
| NetVLAD | VGG-16 | 4096 | 58.9 | 13.1 | 64.6 | 8.0 | 66.4 | 51.5 | 54.4 | 35.4 |
| SFRS | VGG-16 | 4096 | 70.0 | 16.0 | 75.9 | 28.6 | 81.1 | 69.7 | 76.0 | 54.8 |
| CosPlace | VGG-16 | 512 | 82.6 | 58.5 | 95.3 | 44.8 | 88.5 | 85.2 | 89.0 | 67.3 |
| **EigenPlaces** | VGG-16 | 512 | **84.2** | 54.5 | **95.4** | 42.3 | **89.4** | 83.5 | **89.2** | **69.7** |
| CosPlace | ResNet-50 | 128 | 85.5 | 54.7 | 98.7 | 35.4 | 88.5 | 80.4 | 86.6 | 65.2 |
| MixVPR | ResNet-50 | 128 | 79.1 | 47.8 | 99.0 | 25.9 | 92.3 | 80.9 | 87.7 | 73.5 |
| **EigenPlaces** | ResNet-50 | 128 | 83.4 | 50.5 | 98.8 | 29.0 | 90.9 | 83.8 | 91.1 | 68.5 |
| CosPlace | ResNet-50 | 512 | 86.9 | 66.5 | 99.1 | 51.6 | 90.0 | 87.3 | 89.5 | 75.9 |
| Conv-AP | ResNet-50 | 512 | 82.3 | 59.2 | 99.2 | 36.0 | 90.5 | 80.3 | 86.4 | 75.3 |
| MixVPR | ResNet-50 | 512 | 83.6 | 67.2 | 99.2 | 44.8 | 93.9 | 86.4 | 93.9 | 78.7 |
| **EigenPlaces** | ResNet-50 | 512 | **89.5** | **67.9** | **99.5** | 51.5 | 92.8 | **89.0** | 92.0 | **83.1** |
| CosPlace | ResNet-50 | 2048 | 87.4 | 71.9 | 99.6 | 50.7 | 92.2 | 87.0 | 92.0 | 78.5 |
| Conv-AP | ResNet-50 | 2048 | 81.2 | 62.3 | 99.3 | 37.9 | 92.0 | 83.7 | 90.2 | 80.3 |
| **EigenPlaces** | ResNet-50 | 2048 | **89.1** | 71.2 | **99.6** | **58.9** | **93.1** | **90.0** | **93.1** | **86.4** |
| Conv-AP | ResNet-50 | 4096 | 82.8 | 59.6 | 99.6 | 41.9 | 91.2 | 81.9 | 87.9 | 82.0 |
| MixVPR | ResNet-50 | 4096 | 87.2 | 76.2 | 99.6 | 64.4 | 96.2 | 91.5 | 96.8 | 84.8 |
| Conv-AP | ResNet-50 | 8192 | 82.4 | 62.9 | 99.7 | 43.4 | 91.9 | 82.8 | 91.0 | 80.4 |

### Key Findings from Comparisons

1. Older methods (NetVLAD 2016, SFRS 2020) underperform despite using larger 4096-D descriptors.
2. No single model achieves state-of-the-art on all datasets.
3. EigenPlaces achieves best overall results, especially on multi-view datasets.
4. MixVPR outperforms EigenPlaces on some frontal-view datasets but requires 2× larger descriptors (4096-D vs 2048-D) for best performance.
5. Both methods handle grayscale datasets (Nordland, Eynsham) without specific training.
6. Low-dimensionality descriptors (128-D) work well on minimal-shift datasets but struggle on cross-domain challenges.

## 4.4 Resource Analysis

### GPU Memory Footprint

| Method | GPU Memory | Descriptor Dim |
|--------|-----------|----------------|
| EigenPlaces (ResNet-50, 2048-D) | < 7 GB | 2048 |
| MixVPR | > 18 GB | 4096 (for best results) |

EigenPlaces trains using less than 7 GB GPU memory (ResNet-50, 2048-D descriptors), compared to MixVPR's more than 18 GB requirement with 480-image batches. This represents approximately 60% less GPU memory.

### Training Time

- ~24 hours on a single NVIDIA RTX 3090 GPU for best architecture
- Similar training time to SFRS, CosPlace, MixVPR
- Descriptor dimensionality has negligible impact on training time

### Inference Performance

- EigenPlaces achieves 2× faster matching speed than MixVPR with half the memory requirements (smaller descriptors)
- Descriptor extraction time becomes negligible on large-scale databases

## 4.5 Ablation Studies

### Ablation on Loss Components (Table 5)

Architecture: ResNet-18, 512-D output dimensionality

| Lateral Loss | Frontal Loss | Pitts30k | Tokyo 24/7 | MSLS Val | St Lucia | Average |
|:---:|:---:|----------|-----------|----------|----------|---------|
| Yes | No | 90.2 | 80.0 | 83.1 | 97.3 | 87.6 |
| No | Yes | 89.5 | 78.1 | 85.8 | 99.3 | 88.2 |
| Yes | Yes | **90.5** | **82.2** | **86.2** | **99.0** | **89.5** |

**Observations:**
- Lateral loss excels on multi-view datasets (Tokyo 24/7: 80.0 vs 78.1)
- Frontal loss excels on frontal-view datasets (MSLS Val: 85.8 vs 83.1, St Lucia: 99.3 vs 97.3)
- Combined approach achieves the best balanced performance across all dataset types

### Ablation on Focal Distance D (Table 6)

Architecture: ResNet-18, 512-D output dimensionality

| Focal Distance (meters) | Pitts30k | Tokyo 24/7 | MSLS Val | St Lucia | Average |
|-------------------------|----------|-----------|----------|----------|---------|
| 0 | 89.4 | 74.0 | 82.6 | 98.4 | 86.1 |
| 10 | 90.5 | 82.2 | 86.2 | 99.0 | 89.5 |
| **20** | **90.3** | **84.4** | **86.1** | **99.5** | **90.1** |
| 30 | 90.3 | 82.9 | 85.0 | 99.5 | 89.4 |
| 50 | 90.4 | 83.8 | 85.9 | 99.5 | 89.9 |

**Observations:**
- D = 0 degrades performance significantly (especially Tokyo 24/7: 74.0)
- Optimal performance at D = 20 meters (average 90.1)
- Higher focal distances improve frontal-view results by aligning with road-parallel image orientations
- Performance is relatively stable for D ∈ [10, 50] meters

### Embedding Invariance Analysis

Cosine similarity matrices of 40 images from identical locations with varying viewpoints demonstrate that EigenPlaces maintains higher correlation across distant viewpoint indices compared to CosPlace and MixVPR, confirming effective viewpoint robustness is embedded in the learned descriptors.

---

# 5. Conclusions

EigenPlaces introduces a novel training algorithm addressing perspective shifts in VPR by inferring points of interest from within-cell geographical distributions using SVD, creating classes with maximal viewpoint variation. Loss minimization teaches networks to recognize identical points from diverse perspectives.

Extensive experiments across 16 diverse datasets demonstrate state-of-the-art results on the majority of tests using lighter descriptors than competing methods. The method is particularly effective on multi-view datasets while maintaining competitive performance on frontal-view datasets.

---

# Appendix A: Training Data Visualization

Visualization comparing training data selection across:
- **NetVLAD:** Uses most-similar database images (likely same viewpoint)
- **CosPlace:** Creates classes with identical orientation (no viewpoint diversity within class)
- **GSV-Cities (Conv-AP/MixVPR):** Uses pre-defined classes with similar viewpoints
- **EigenPlaces:** Explicitly creates classes with large viewpoint shifts (different lateral positions facing same facade)

# Appendix B: Dataset Descriptions

Detailed dataset descriptions with sample images for all 16 evaluation datasets including multi-view and frontal-view categories.

# Appendix C: Extended Results

**Table 7:** Multi-view dataset recalls with Recall@1/5/10/20 across all datasets for all backbone/descriptor combinations.

**Table 8:** Frontal-view dataset recalls with Recall@1/5/10/20 across all datasets for all backbone/descriptor combinations.

Both tables include comprehensive comparisons across: NetVLAD, SFRS, CosPlace, Conv-AP, MixVPR, and EigenPlaces. Qualitative results showing EigenPlaces' superior handling of challenging viewpoints are also included.

---

# References

[1] Ali-bey, A., Chaib-draa, B., & Giguère, P. (2022). "GSV-cities: Toward appropriate supervised visual place recognition." *Neurocomputing*, 513, 194–203.

[2] Ali-bey, A., Chaib-draa, B., & Giguère, P. (2023). "MixVPR: Feature mixing for visual place recognition." *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 2998–3007.

[3] Anoosheh, A., Sattler, T., Timofte, R., Pollefeys, M., & Van Gool, L. (2019). "Night-to-day image translation for retrieval-based localization." *2019 International Conference on Robotics and Automation (ICRA)*, 5958–5964.

[4] Arandjelović, R., & Zisserman, A. (2012). "Three things everyone should know to improve object retrieval." 2911–2918.

[5] Arandjelović, R., Gronat, P., Torii, A., Pajdla, T., & Sivic, J. (2018). "NetVLAD: CNN architecture for weakly supervised place recognition." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(6), 1437–1451.

[6] Babenko, A., Slesarev, A., Chigorin, A., & Lempitsky, V. (2014). "Neural codes for image retrieval." *ArXiv*, abs/1404.1777.

[7] Bay, H., Ess, A., Tuytelaars, T., & Van Gool, L. (2008). "Speeded-up robust features (SURF)." *Computer Vision and Understanding*, 110, 346–359.

[8] Berton, G., Masone, C., & Caputo, B. (2022). "Rethinking visual geo-localization for large-scale applications." *CVPR*, June.

[9] Berton, G., Masone, C., Paolicelli, V., & Caputo, B. (2021). "Viewpoint invariant dense matching for visual geolocalization." *IEEE International Conference on Computer Vision*, 12169–12178.

[10] Berton, G., Mereu, R., Trivigno, G., Masone, C., Csurka, G., Sattler, T., & Caputo, B. (2022). "Deep visual geo-localization benchmark." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.

[11] Berton, G., Paolicelli, V., Masone, C., & Caputo, B. (2021). "Adaptive-attentive geolocalization from few queries: A hybrid approach." *IEEE Winter Conference on Applications of Computer Vision*, 2918–2927.

[12] Cao, B., Araujo, A., & Sim, J. (2020). "Unifying deep local and global features for image search." *European Conference on Computer Vision*, 726–743.

[13] Chen, D. M., Baatz, G., Köser, K., Tsai, S. S., Vedantham, R., Pylvänäinen, T., Roimela, K., Chen, X., Bach, J., Pollefeys, M., Girod, B., & Grzeszczuk, R. (2011). "City-scale landmark identification on mobile devices." *IEEE Conference on Computer Vision and Pattern Recognition*, 737–744.

[14] Chen, Z., Jacobson, A., Sunderhauf, N., Upcroft, B., Liu, L., Shen, C., Reid, I., & Milford, M. (2017). "Deep learning features at scale for visual place recognition." *2017 IEEE International Conference on Robotics and Automation*, 3223–3230.

[15] Chen, Z., Liu, L., Sa, I., Ge, Z., & Chli, M. (2018). "Learning context flexible attention model for long-term visual place recognition." *IEEE Robotics and Automation Letters*, 3(4), 4015–4022.

[16] Chen, Z., Maffra, F., Sa, I., & Chli, M. (2017). "Only look once, mining distinctive landmarks from ConvNet for visual place recognition." *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 9–16.

[17] Cummins, M., & Newman, P. (2009). "Highly scalable appearance-only SLAM - FAB-MAP 2.0." *Robotics: Science and Systems*.

[18] Deng, J., Guo, J., & Zafeiriou, S. (2019). "ArcFace: Additive angular margin loss for deep face recognition." *IEEE Conference on Computer Vision and Pattern Recognition*, 4685–4694.

[19] Doan, A., Latif, Y., Chin, T., Liu, Y., Do, T., & Reid, I. (2019). "Scalable place recognition under appearance change for autonomous driving." *IEEE International Conference on Computer Vision*, 9319–9328.

[20] Garg, S., Suenderhauf, N., & Milford, M. (2019). "Semantic–geometric visual place recognition: a new perspective for reconciling opposing views." *The International Journal of Robotics Research*.

[21] Ge, Y., Wang, H., Zhu, F., Zhao, R., & Li, H. (2020). "Self-supervising fine-grained region similarities for large-scale image localization." *Computer Vision – ECCV 2020*, 369–386.

[22] Hausler, S., Garg, S., Xu, M., Milford, M., & Fischer, T. (2021). "Patch-NetVLAD: Multi-scale fusion of locally-global descriptors for place recognition." *IEEE Conference on Computer Vision and Pattern Recognition*, 14141–14152.

[23] Hausler, S., Jacobson, A., & Milford, M. (2019). "Multi-process fusion: Visual place recognition using multiple image processing methods." *IEEE Robotics and Automation Letters*, 4(2), 1924–1931.

[24] He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition." *IEEE Conference on Computer Vision and Pattern Recognition*, 770–778.

[25] Ibrahimi, S., van Noord, N., Alpherts, T., & Worring, M. (2021). "Inside out visual place recognition." *British Machine Vision Conference*.

[26] Khaliq, A., Ehsan, S., Chen, Z., Milford, M., & McDonald-Maier, K. (2020). "A holistic visual place recognition approach using lightweight CNNs for significant viewpoint and appearance changes." *IEEE Transactions on Robotics*, 36(2), 561–569.

[27] Kim, H. J., Dunn, E., & Frahm, J. (2017). "Learned contextual feature reweighting for image geo-localization." *IEEE Conference on Computer Vision and Pattern Recognition*, 3251–3260.

[28] Kingma, D., & Ba, J. (2014). "Adam: A method for stochastic optimization." *International Conference on Learning Representations*, December.

[29] Leyva-Vallina, M., Strisciuglio, N., & Petkov, N. (2023). "Data-efficient large scale place recognition with graded similarity supervision." *CVPR*.

[30] Liu, D., Cui, Y., Yan, L., Mousas, C., Yang, B., & Chen, Y. (2021). "DenseNet: Weakly supervised visual localization using multi-scale feature aggregation." *Proceedings of the AAAI Conference on Artificial Intelligence*, 6101–6109.

[31] Liu, L., Li, H., & Dai, Y. (2019). "Stochastic Attraction-Repulsion Embedding for Large Scale Image Localization." *IEEE International Conference on Computer Vision*.

[32] Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., & Song, L. (2017). "SphereFace: Deep hypersphere embedding for face recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 212–220.

[33] Lowe, D. G. (2004). "Distinctive image features from scale-invariant keypoints." *International Journal of Computer Vision*, 60(2), 91–110.

[34] Maddern, W., Pascoe, G., Linegar, C., & Newman, P. (2017). "1 Year, 1000km: The Oxford RobotCar Dataset." *The International Journal of Robotics Research*.

[35] Masone, C., & Caputo, B. (2021). "A survey on deep visual place recognition." *IEEE Access*, 9, 19516–19547.

[36] Mereu, R., Trivigno, G., Berton, G., Masone, C., & Caputo, B. (2022). "Learning sequential descriptors for sequence-based visual place recognition." *IEEE Robotics and Automation Letters*, 7(4), 10383–10390.

[37] Milford, M., & Wyeth, G. (2008). "Mapping a suburb with a single camera using a biologically inspired SLAM system." *IEEE Transactions on Robotics*, 24, 1038–1053.

[38] Oliva, A., & Torralba, A. (2006). "Building the gist of a scene: the role of global image features in recognition." *Progress in Brain Research*, 155, 23–36.

[39] Peng, G., Yue, Y., Zhang, J., Wu, Z., Tang, X., & Wang, D. (2021). "Semantic reinforced attention learning for visual place recognition." *IEEE International Conference on Robotics and Automation*, 13415–13422.

[40] Peng, G., Zhang, J., Li, H., & Wang, D. (2021). "Attentional pyramid pooling of salient visual residuals for place recognition." *IEEE International Conference on Computer Vision*, 885–894.

[41] Porav, H., Maddern, W., & Newman, P. (2018). "Adversarial training for adverse conditions: Robust metric localisation using appearance transfer." *2018 IEEE International Conference on Robotics and Automation (ICRA)*, 1011–1018.

[42] Radenović, F., Tolias, G., & Chum, O. (2018). "Fine-tuning CNN Image Retrieval with No Human Annotation." *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

[43] Razavian, A., Sullivan, J., Maki, A., & Carlsson, S. (2015). "Visual Instance Retrieval with Deep Convolutional Networks." *CoRR*, abs/1412.6574.

[44] Sarlin, P., DeTone, D., Malisiewicz, T., & Sarpali, P. (2020). "SuperGlue: Learning feature matching with graph neural networks." *CVPR*.

[45] Simonyan, K., & Zisserman, A. (2015). "Very deep convolutional networks for large-scale image recognition." *International Conference on Learning Representations*.

[46] Sun, J., Shen, Z., Wang, Y., Bao, H., & Zhou, X. (2021). "LoFTR: Detector-free local feature matching with transformers." *CVPR*.

[47] Sünderhauf, N., Neubert, P., & Protzel, P. (2013). "Are we there yet? Challenging SeqSLAM on a 3000 km journey across all four seasons." *Proc. of Workshop on Long-Term Autonomy, IEEE International Conference on Robotics and Automation*.

[48] Tan, F., Yuan, J., & Ordonez, V. (2021). "Instance-level image retrieval using reranking transformers." *IEEE International Conference on Computer Vision*.

[49] Tolias, G., Sicre, R., & Jégou, H. (2016). "Particular object retrieval with integral max-pooling of CNN activations." *CoRR*, abs/1511.05879.

[50] Torii, A., Arandjelović, R., Sivic, J., Okutomi, M., & Pajdla, T. (2018). "24/7 place recognition by view synthesis." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(2), 257–271.

[51] Torii, A., Sivic, J., Okutomi, M., & Pajdla, T. (2015). "Visual place recognition with repetitive structures." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 37(11), 2346–2359.

[52] Torii, A., Sivic, J., Okutomi, M., & Pajdla, T. (2015). "Visual place recognition with repetitive structures." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 37(11), 2346–2359.

[53] Torii, A., Taira, H., Sivic, J., Pollefeys, M., Okutomi, M., Pajdla, T., & Sattler, T. (2021). "Are large-scale 3D models really necessary for accurate visual localization?" *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43, 814–829.

[54] Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., Li, Z., & Liu, W. (2018). "CosFace: Large margin cosine loss for deep face recognition." *IEEE Conference on Computer Vision and Pattern Recognition*, 5265–5274.

[55] Wang, R., Shen, Y., Zuo, W., Zhou, S., & Zheng, N. (2022). "TransVPR: Transformer-based place recognition with multi-level attention aggregation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 13648–13657.

[56] Wang, X., Han, X., Huang, W., Dong, D., & Scott, M. R. (2019). "Multi-similarity loss with general pair weighting for deep metric learning." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 5022–5030.

[57] Warburg, F., Hauberg, S., Lopez-Antequera, M., Gargallo, P., Kuang, Y., & Civera, J. (2020). "Mapillary street-level sequences: A dataset for lifelong place recognition." *IEEE Conference on Computer Vision and Pattern Recognition*, June.

[58] Yildiz, B., Khademi, S., Siebes, R., & Van Gemert, J. (2022). "AmsterTime: A visual place recognition benchmark dataset for severe domain shift." *2022 26th International Conference on Pattern Recognition (ICPR)*, 2749–2755.

[59] Yu, J., Zhu, C., Zhang, J., Huang, Q., & Tao, D. (2020). "Spatial pyramid-enhanced NetVLAD with weighted triplet loss for place recognition." *IEEE Transactions on Neural Networks and Learning Systems*, 31(2), 661–674.

[60] Zaffar, M., Garg, S., Milford, M., Kooij, J., Flynn, D., McDonald-Maier, K., & Ehsan, S. (2021). "VPR-Bench: An open-source visual place recognition evaluation framework with quantifiable viewpoint and appearance change." *International Journal of Computer Vision*, 129(7), 2136–2174.

[61] Zhang, J., Cao, Y., & Wu, Q. (2021). "Vector of locally and adaptively aggregated descriptors for image feature representation." *Pattern Recognition*, 116, 107952.

[62] Zhu, Y., Wang, J., Xie, L., & Zheng, L. (2018). "Attention-based pyramid aggregation network for visual place recognition." *2018 ACM Multimedia Conference on Multimedia Conference*, 99–107.
