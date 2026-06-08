# Meta Information

- URL: [A Survey on 3D Gaussian Splatting](https://arxiv.org/abs/2401.03890)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Chen, G., & Wang, W. (2024). A Survey on 3D Gaussian Splatting. arXiv:2401.03890.

# A Survey on 3D Gaussian Splatting

## Overview

3D Gaussian Splatting (3D GS) is an explicit radiance field technique that represents scenes as millions of learnable 3D Gaussian primitives, enabling real-time novel-view synthesis with high visual quality. Unlike Neural Radiance Fields (NeRF), which use implicit neural network representations requiring costly ray marching, 3D GS uses tile-based rasterization with GPU parallelization to achieve interactive rendering speeds (30+ FPS).

**Who uses it**: Computer vision researchers and developers working on novel-view synthesis, real-time rendering, SLAM, scene editing, text-to-3D generation, and autonomous driving simulation.

**When to use**: When real-time rendering performance is required (e.g., VR/AR, autonomous driving simulation, interactive editing) and when scene editability is prioritized over compact representation.

## 3D Gaussian Representation

Each 3D Gaussian is characterized by four learnable attributes:

| Attribute | Symbol | Description |
|-----------|--------|-------------|
| Position (mean) | $\boldsymbol{\mu} \in \mathbb{R}^3$ | Center location in world space |
| Opacity | $\alpha \in [0,1]$ | Transparency of the Gaussian |
| Covariance matrix | $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$ | Shape and orientation (symmetric positive semi-definite) |
| Color | $c$ | View-dependent appearance via spherical harmonics coefficients |

### Covariance Matrix Parameterization

Direct optimization of $\boldsymbol{\Sigma}$ risks producing non-positive semi-definite matrices. Instead, it is factored as:

$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$$

where $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ is a rotation matrix derived from a unit quaternion $\mathbf{q} \in \mathbb{R}^4$, and $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$ is a diagonal scale matrix with $\mathbf{s} \in \mathbb{R}^3$.

### 2D Projection (Splatting)

To render, each 3D Gaussian is projected into 2D image space. The projected covariance is:

$$\boldsymbol{\Sigma}' = \mathbf{J}\mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^T\mathbf{J}^T$$

where $\mathbf{J} \in \mathbb{R}^{2 \times 3}$ is the Jacobian of the affine approximation of the projective transformation, and $\mathbf{W} \in \mathbb{R}^{3 \times 3}$ is the camera viewing transformation.

## Rendering Algorithm

Pixel colors are computed via alpha compositing of depth-sorted Gaussians:

$$\mathbf{C} = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha'_i \prod_{j=1}^{i-1}(1 - \alpha'_j)$$

where the effective per-pixel opacity of Gaussian $i$ is:

$$\alpha'_i = \alpha_i \cdot \exp\!\left(-\tfrac{1}{2}(\mathbf{x}' - \boldsymbol{\mu}'_i)^T {\boldsymbol{\Sigma}'_i}^{-1}(\mathbf{x}' - \boldsymbol{\mu}'_i)\right)$$

with $\mathbf{x}' \in \mathbb{R}^2$ the pixel coordinate and $\boldsymbol{\mu}'_i \in \mathbb{R}^2$ the projected Gaussian center.

**Step-by-step rendering pipeline:**

1. **Frustum culling** – Discard 3D Gaussians outside the camera's view frustum.
2. **Splatting** – Project each remaining 3D ellipsoid to a 2D ellipse using the projection above.
3. **Tile division** – Partition the image into $16 \times 16$ pixel tiles (each tile maps to one CUDA block).
4. **Gaussian replication** – Assign each Gaussian a key encoding its tile ID and depth value; Gaussians spanning multiple tiles are duplicated.
5. **Depth sorting** – Sort all replicated Gaussians by the combined tile-depth key using a GPU radix sort.
6. **Parallel alpha compositing** – Each tile processes its sorted Gaussians front-to-back in parallel using shared memory, accumulating color and transparency.

> [!NOTE]
> "Tiles and pixels correspond to CUDA blocks and threads" — enabling massive GPU parallelization that makes real-time rendering feasible.

## Optimization

### Loss Function

The model is trained by minimizing a combined photometric loss:

$$\mathcal{L} = (1 - \lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$$

where $\mathcal{L}_1$ is the mean absolute error between rendered and ground-truth images, $\mathcal{L}_{\text{D-SSIM}}$ is the structural dissimilarity, and $\lambda$ controls their balance.

### Density Control (Adaptive Densification and Pruning)

The Gaussian set is dynamically adjusted during training:

**Densification** (targets Gaussians with large view-space positional gradients):
- *Clone*: Duplicate small Gaussians in under-reconstructed (missing geometry) regions.
- *Split*: Divide large Gaussians into two smaller ones in over-reconstructed (high-variance) regions.

**Pruning**:
- Remove Gaussians with opacity $\alpha$ below a threshold (near-transparent).
- Remove Gaussians that grow excessively large in world space or view space.
- Reset $\alpha$ to near zero for Gaussians close to camera positions after certain iterations (to avoid floaters).

> [!IMPORTANT]
> Quaternion and scale vector gradients are computed analytically rather than via automatic differentiation to reduce memory and compute cost during optimization.

## Comparison with NeRF

| Aspect | NeRF | 3D Gaussian Splatting |
|--------|------|----------------------|
| Representation | Implicit MLP | Explicit Gaussian primitives |
| Rendering | Volumetric ray marching (slow) | Tile-based rasterization (real-time) |
| Rendering speed | Minutes per image | 30+ FPS |
| Scene editability | Difficult (latent space) | Direct manipulation of primitives |
| Memory | Compact (MLP weights) | High (millions of Gaussians) |
| Training | Hours | Minutes to hours |
| View-dependent effects | MLP-encoded | Spherical harmonics per Gaussian |

## Application Areas

### SLAM

3D GS-based SLAM systems (e.g., SplaTAM) use differentiable rendering to jointly optimize camera poses and scene geometry. SplaTAM achieves trajectory estimation error of **0.36 cm** on the Replica dataset, compared to **0.52 cm** for Point-SLAM. Gaussian-SLAM achieves PSNR 38.90 dB with ~578× speedup over Point-SLAM.

### Dynamic Scene Modeling

4D Gaussian Splatting extends the representation temporally, associating each Gaussian with a time-varying deformation field to model moving objects. Methods like CoGS achieve PSNR 37.90 dB on the D-NeRF dataset, surpassing NeRF-based approaches by over 5 dB.

### AI-Generated Content (AIGC)

3D GS supports text-to-3D generation and human avatar modeling with real-time rendering. GART achieves PSNR 32.22 dB on ZJU-MoCap; Human101 renders at 104 FPS with competitive quality.

### Autonomous Driving

Methods like DrivingGaussian reconstruct dynamic driving scenes from multi-sensor data, achieving PSNR 28.74 dB on nuScenes, surpassing S-NeRF by 3.31 dB.

# Experiments

- **Dataset (SLAM/static)**: Replica (18 indoor scenes with dense meshes and HDR textures)
- **Dataset (dynamic)**: D-NeRF (synthetic animated objects, 50–200 frames per sequence)
- **Dataset (driving)**: nuScenes (1000 driving scenes, multi-sensor: cameras + LiDAR + RADAR)
- **Dataset (human avatars)**: ZJU-MoCap (23 synchronized cameras at $1024 \times 1024$ resolution)
- **Hardware**: Not specified (GPU-based; performance reported in FPS implying consumer/server GPUs)
- **Metrics**: PSNR (dB), SSIM, LPIPS (image quality); ATE (trajectory error in cm for SLAM); FPS (rendering speed)
- **Results**:
  - SLAM: SplaTAM achieves 0.36 cm ATE vs. 0.52 cm for Point-SLAM on Replica
  - Dynamic: CoGS achieves PSNR 37.90 dB on D-NeRF, >5.22 dB gain over NeRF baselines
  - Driving: DrivingGaussian-L achieves PSNR 28.74 dB on nuScenes, +3.31 dB over S-NeRF
  - Avatar: Human101 renders at 104 FPS on ZJU-MoCap

## Open Challenges

1. **Data efficiency**: Handling sparse inputs and reducing artifacts from limited viewpoints.
2. **Memory scalability**: Millions of Gaussians make large-scale scene modeling memory-intensive.
3. **Advanced rendering**: Incorporating physically based light transport (reflections, refractions) beyond view-dependent color.
4. **Optimization robustness**: Reducing sensitivity to hyperparameters and initialization quality.
5. **Mesh extraction**: Bridging the gap between volumetric Gaussian representation and surface meshes required for downstream tasks.
6. **Extended attributes**: Endowing Gaussians with semantic, linguistic, or physical properties for reasoning and simulation.
