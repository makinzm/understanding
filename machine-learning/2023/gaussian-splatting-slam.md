# Meta Information

- URL: [Gaussian Splatting SLAM](https://arxiv.org/abs/2312.06741)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Matsuki, H., Murai, R., Kelly, P. H. J., & Davison, A. J. (2024). Gaussian Splatting SLAM. CVPR 2024.

# Gaussian Splatting SLAM

## Overview

Gaussian Splatting SLAM is the first real-time monocular and RGB-D SLAM system that uses 3D Gaussian Splatting (3DGS) as its sole scene representation. Developed at the Dyson Robotics Laboratory and Software Performance Optimisation Group at Imperial College London, the system achieves near-photorealistic rendering fidelity while tracking camera poses at approximately 3 fps on a commodity GPU (RTX 4090). It is applicable in robotics, augmented reality, and any scenario requiring simultaneous dense map construction and real-time camera localization.

## Background: 3D Gaussian Splatting

3DGS represents a scene as a set of anisotropic 3D Gaussian primitives $\mathcal{G} = \{\mathcal{G}^i\}$. Each Gaussian $\mathcal{G}^i$ is parameterized by:

- **Mean** $\mu^i_W \in \mathbb{R}^3$: world-frame position
- **Covariance** $\Sigma^i_W \in \mathbb{R}^{3 \times 3}$: anisotropic shape/orientation (parameterized as $\Sigma = RSS^TR^T$ with rotation $R$ and scale diagonal $S$)
- **Color** $c^i \in \mathbb{R}^3$: RGB appearance (spherical harmonics omitted for efficiency)
- **Opacity** $\alpha^i \in [0, 1]$: transparency

### Rendering Pipeline

Gaussians are projected to 2D image space and rendered via alpha-blending. For a camera pose $T_{CW} \in SE(3)$:

$$\mu^I = \pi(T_{CW} \cdot \mu_W), \quad \Sigma^I = JW\Sigma_W W^T J^T$$

where $\pi$ is the perspective projection, $J$ is the Jacobian of the projection, and $W$ is the rotational component of $T_{CW}$.

The pixel color $\mathcal{C}_p$ and depth $\mathcal{D}_p$ are synthesized by alpha-compositing over the $\mathcal{N}$ Gaussians sorted by depth:

$$\mathcal{C}_p = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

$$\mathcal{D}_p = \sum_{i \in \mathcal{N}} z_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

where $z_i$ is the depth of the $i$-th Gaussian's center in camera coordinates.

## System Architecture

The system alternates between two phases in a single-threaded loop: **tracking** (camera pose estimation given fixed Gaussians) and **mapping** (joint optimization of Gaussian parameters and keyframe poses).

### Tracking

Given a new frame with observed image $\bar{I}$ (and depth $\bar{D}$ for RGB-D), the camera pose $T_{CW}$ is optimized by minimizing photometric and geometric residuals:

**Photometric loss:**
$$E_{pho} = \| I(\mathcal{G}, T_{CW}) - \bar{I} \|_1$$

**Geometric loss (RGB-D only):**
$$E_{geo} = \| D(\mathcal{G}, T_{CW}) - \bar{D} \|_1$$

**Combined tracking objective (RGB-D):**
$$\min_{T_{CW}} \; \lambda_{pho} E_{pho} + (1 - \lambda_{pho}) E_{geo}, \quad \lambda_{pho} = 0.9$$

Optimization runs for 100 iterations per frame using the Adam optimizer. The critical innovation is the derivation of **analytical Jacobians** for $T_{CW} \in SE(3)$ using Lie algebra, enabling direct backpropagation through the rasterizer without automatic differentiation overhead.

#### SE(3) Jacobian Derivation

The derivative on the SE(3) manifold is defined via the left-perturbation model:

$$\frac{\mathcal{D}f(T)}{\mathcal{D}T} \triangleq \lim_{\tau \to 0} \frac{\log(f(\text{Exp}(\tau) \circ T) \circ f(T)^{-1})}{\tau}$$

The key derived Jacobians (output $\in \mathbb{R}^{6}$) are:

$$\frac{\mathcal{D}\mu^C}{\mathcal{D}T_{CW}} = [I \; | \; -\mu^C_\times], \quad \frac{\mathcal{D}W}{\mathcal{D}T_{CW}} = \begin{bmatrix} 0 & -W_{:,1\times} \\ 0 & -W_{:,2\times} \\ 0 & -W_{:,3\times} \end{bmatrix}$$

where $\mu^C_\times$ denotes the skew-symmetric matrix of $\mu^C$, and $W_{:,i}$ denotes the $i$-th column of $W$.

> [!IMPORTANT]
> These closed-form Jacobians are critical to achieving 3 fps tracking speed. Automatic differentiation through the full rasterizer would be significantly slower.

### Mapping

Mapping jointly optimizes Gaussian parameters and keyframe poses over an active window $\mathcal{W} = \mathcal{W}_k \cup \mathcal{W}_r$, where $\mathcal{W}_k$ is the current keyframe window (size 10 for Replica, 8 for TUM) and $\mathcal{W}_r$ is a random set of historical keyframes to prevent catastrophic forgetting.

**Mapping objective:**

$$\min_{\mathcal{G}, \{T_{CW}^k\}_{k \in \mathcal{W}}} \sum_{k \in \mathcal{W}} E_{pho}^k + \lambda_{iso} E_{iso}$$

with 150 iterations per mapping step.

### Isotropic Regularization

A novel regularization term encourages Gaussians to remain roughly spherical, preventing elongated needle-like artifacts that degrade tracking:

$$E_{iso} = \sum_{i=1}^{|\mathcal{G}|} \| s^i - \tilde{s}^i \cdot \mathbf{1} \|_1$$

where $s^i \in \mathbb{R}^3$ are the three scaling parameters of the $i$-th Gaussian, $\tilde{s}^i = \text{mean}(s^i)$ is their mean, and $\mathbf{1}$ is the all-ones vector. The weight $\lambda_{iso} = 10$.

> [!NOTE]
> "We observe that without this term, Gaussians tend to be stretched along the viewing direction, which hinders the tracking performance."

### Keyframe Selection

A new keyframe is inserted when either:
1. **Covisibility drops**: $\text{IOU}_{cov}(i,j) < \tau_{iou}$ (0.95 for Replica, 0.90 for TUM)
2. **Translation threshold exceeded**: $t_{ij} > kf_m \cdot \hat{D}_i$ ($kf_m=0.04$ for Replica, $0.08$ for TUM)

Covisibility metrics over visible Gaussian sets $\mathcal{G}^v_i$:

$$\text{IOU}_{cov}(i,j) = \frac{|\mathcal{G}^v_i \cap \mathcal{G}^v_j|}{|\mathcal{G}^v_i \cup \mathcal{G}^v_j|}, \quad \text{OC}_{cov}(i,j) = \frac{|\mathcal{G}^v_i \cap \mathcal{G}^v_j|}{\min(|\mathcal{G}^v_i|, |\mathcal{G}^v_j|)}$$

A Gaussian is considered visible if the accumulated opacity $\alpha$ along its viewing ray has not yet reached 0.5.

### Gaussian Insertion and Pruning

**Insertion (monocular):** New Gaussians are added at pixel locations where the depth error exceeds a threshold. Their 3D positions are sampled from:
- Observed pixels: $\mathcal{N}(\mathcal{D}_p, 0.2 \sigma_D)$
- Unobserved regions: $\mathcal{N}(\hat{D}, 0.5 \sigma_D)$

**Pruning:** Gaussians are removed when:
- Opacity $\alpha < 0.7$ (transparent/invisible)
- Inserted within the last 3 keyframes but not observed by $\geq 3$ other frames (spurious)

Pruning only occurs once the keyframe window $\mathcal{W}_k$ is full.

## Pseudocode: Main Loop

```
Initialize: insert Gaussians from first frame
for each new frame f:
    # Tracking
    T_CW = optimize_pose(G, I_f, D_f, T_prev)  # 100 Adam iters

    if is_keyframe(f, T_CW, W_k):
        W_k.add(f)
        W_r = sample_random_keyframes()
        W = W_k ∪ W_r

        # Gaussian insertion
        new_gaussians = densify(f, G, D_f)
        G = G ∪ new_gaussians

        # Mapping
        for 150 iterations:
            G, {T_CW^k} = optimize(G, {T_CW^k}, W, E_pho + λ_iso * E_iso)

        # Pruning
        if |W_k| == max_window_size:
            G = prune(G, W_k)
```

## Comparison with Similar Methods

| Method | Representation | Tracking | Rendering Speed | Memory |
|---|---|---|---|---|
| iMAP | MLP | Photometric | ~0.1 fps | 101.6 MB |
| NICE-SLAM | Hierarchical Grid | Photometric | 0.54 fps | — |
| Co-SLAM | Hash Grid + MLP | Photometric | — | 1.6 MB |
| Point-SLAM | Neural Point Cloud | Photometric | 1.33 fps | — |
| **Gaussian Splatting SLAM** | **3D Gaussians** | **Photometric+Geometric** | **769 fps** | **2.6–4.0 MB** |

> [!TIP]
> The key advantage over neural-implicit methods (NICE-SLAM, Co-SLAM) is that Gaussians are an **explicit** representation: no neural network inference is needed for rendering, enabling the 769 fps speed advantage.

**Convergence basin analysis:** The Gaussian representation achieves 79–82% pose recovery success rate from large perturbations, compared to ~30–50% for Hash Grid SDF and MLP-based methods, demonstrating greater robustness for camera localization initialization.

## Experiments

- **Datasets:**
  - TUM RGB-D: 3 sequences (`fr1/desk`, `fr2/xyz`, `fr3/office`) from an indoor dataset with ground-truth poses from a motion capture system
  - Replica: 8 sequences (simulated indoor rooms: `room0`–`room2`, `office0`–`office3`) with perfect ground-truth depth and poses
- **Hardware:** Intel Core i9 12900K, NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Optimizer:** Adam
- **Metrics:** ATE RMSE (cm) for tracking; PSNR/SSIM/LPIPS for rendering quality

**Key results:**

- **TUM RGB-D (monocular):** ATE = 4.44 cm average; outperforms DSO (7.94 cm), DROID-VO (8.57 cm), and DepthCov
- **TUM RGB-D (RGB-D):** ATE = 1.58 cm average; competitive with ORB-SLAM2 (1.32 cm)
- **Replica (RGB-D):** ATE = 0.79 cm average; best tracking on 4 of 8 sequences
- **Rendering (Replica):** PSNR = 37.50 dB, SSIM = 0.960, LPIPS = 0.070 — superior to all neural-implicit baselines
- **Rendering speed:** 769 fps at 1200×680 vs. Point-SLAM at 1.33 fps (~578× speedup)
- **Memory:** 2.6–4.0 MB map size vs. 101.6 MB for iMAP

## Limitations

- **Scale**: Tested only on room-scale environments; trajectory drift accumulates in larger scenes due to the absence of loop closure detection and correction
- **Speed**: 2.5–3.2 fps tracking does not yet meet the 30 fps standard for real-time robotics; the bottleneck is Gaussian optimization, not rendering
- **Geometry**: No explicit surface normal extraction; the system does not recover clean meshes, only volume-rendered images
- **Spherical harmonics omitted**: Color representation is simplified to a single RGB value per Gaussian (no view-dependent effects)
