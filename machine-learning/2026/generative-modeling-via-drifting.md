# Meta Information

- URL: [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770)
- LICENSE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Deng, M., Li, H., Li, T., Du, Y., & He, K. (2026). Generative Modeling via Drifting. arXiv:2602.04770.

---

# Generative Modeling via Drifting

## Overview

This paper introduces **Drifting Models**, a generative modeling paradigm that reconceptualizes the generation process as a training-time distribution evolution rather than an inference-time iterative refinement. Instead of running a stochastic differential equation (SDE) or ordinary differential equation (ODE) at inference time (as in diffusion or flow models), a Drifting Model trains a generator whose output distribution is gradually pushed toward the data distribution through a **drifting field**. At convergence the generator naturally produces samples in one single forward pass — no multi-step sampling is required.

**Applicability**: Researchers and practitioners who need high-quality image generation or decision-making policies with strict latency constraints (robotics, real-time rendering) benefit most. The method requires a pre-trained self-supervised feature encoder and a sufficient compute budget (≈ 1280 training epochs on ImageNet).

---

## Problem Setting

| Term | Definition |
|------|------------|
| $p$ | Target data distribution |
| $q_\theta$ | Generator-induced distribution: $q_\theta = f_\theta \# p_\varepsilon$ where $p_\varepsilon = \mathcal{N}(0, I)$ |
| $f_\theta$ | Neural generator mapping noise $\varepsilon \in \mathbb{R}^D$ to samples $x \in \mathbb{R}^D$ |
| $\varepsilon$ | Gaussian noise input, $\varepsilon \sim \mathcal{N}(0, I)$ |

**Goal**: Find $\theta^*$ such that $q_{\theta^*} = p$.

---

## Drifting Field

### Intuition

The drifting field $\mathbf{V}_{p,q}(\mathbf{x})$ defines, for every point $\mathbf{x}$ in sample space, a velocity that moves $\mathbf{x}$ from the current generated distribution $q$ toward the target data distribution $p$. Applying this field to generated samples and treating the result as training targets causes the generator to chase an ever-improving target.

### Attraction–Repulsion Decomposition

The field is split into an attraction component (pulls toward real data) and a repulsion component (pushes away from generated samples), ensuring that when $q = p$ the two components cancel exactly:

$$\mathbf{V}^+_p(\mathbf{x}) = \frac{1}{Z_p(\mathbf{x})} \mathbb{E}_{y^+ \sim p}\bigl[k(\mathbf{x}, y^+)(y^+ - \mathbf{x})\bigr]$$

$$\mathbf{V}^-_q(\mathbf{x}) = \frac{1}{Z_q(\mathbf{x})} \mathbb{E}_{y^- \sim q}\bigl[k(\mathbf{x}, y^-)(y^- - \mathbf{x})\bigr]$$

$$\mathbf{V}_{p,q}(\mathbf{x}) = \mathbf{V}^+_p(\mathbf{x}) - \mathbf{V}^-_q(\mathbf{x})$$

with normalization factors:

$$Z_p(\mathbf{x}) = \mathbb{E}_{y^+ \sim p}[k(\mathbf{x}, y^+)], \quad Z_q(\mathbf{x}) = \mathbb{E}_{y^- \sim q}[k(\mathbf{x}, y^-)]$$

The kernel function is an exponential distance kernel with temperature $\tau$:

$$k(\mathbf{x}, \mathbf{y}) = \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{y}\|}{\tau}\right)$$

### Anti-Symmetry Property

> [!IMPORTANT]
> The field must satisfy $\mathbf{V}_{p,q}(\mathbf{x}) = -\mathbf{V}_{q,p}(\mathbf{x})$ for all $\mathbf{x}$. This ensures the equilibrium condition: when $q = p$, the drifting field becomes zero everywhere, and training has no gradient. Ablations show that breaking this symmetry (e.g., scaling attraction or repulsion by different constants) leads to catastrophic failure (FID rises from 8.46 to over 112).

### Kernelized Form

Substituting the normalization into the combined field yields the kernelized expression (Eq. 11 in the paper):

$$\mathbf{V}_{p,q}(\mathbf{x}) = \frac{1}{Z_p Z_q}\mathbb{E}_{p,q}\bigl[k(\mathbf{x}, y^+) k(\mathbf{x}, y^-)(y^+ - y^-)\bigr]$$

---

## Training Objective

The generator is trained to regress toward the drifted version of its own output. Using a stop-gradient on the target to avoid degenerate solutions:

$$\mathcal{L}(\theta) = \mathbb{E}_\varepsilon\bigl[\|f_\theta(\varepsilon) - \mathrm{sg}\bigl(f_\theta(\varepsilon) + \mathbf{V}_{p,\,q_\theta}(f_\theta(\varepsilon))\bigr)\|^2\bigr]$$

Because the stop-gradient freezes the target, the loss equals $\mathbb{E}[\|\mathbf{V}\|^2]$ when differentiated through $f_\theta$, meaning the generator is penalized for producing samples that are far from the drifted position.

### Feature-Space Extension

Direct pixel-space drifting degrades quality. Instead, a pre-trained feature extractor $\phi: \mathbb{R}^D \to \mathbb{R}^F$ maps samples to a feature space where the drifting field is computed:

$$\mathcal{L}_\phi(\theta) = \mathbb{E}_\varepsilon\bigl[\|\phi(f_\theta(\varepsilon)) - \mathrm{sg}\bigl(\phi(f_\theta(\varepsilon)) + \mathbf{V}_{p,q_\theta}(\phi(f_\theta(\varepsilon)))\bigr)\|^2\bigr]$$

Multi-scale features from multiple ResNet stages are concatenated to capture both local and global structure.

---

## Training Algorithm

```
Algorithm: Drifting Model Training (one iteration)
─────────────────────────────────────────────────────────────────
Input:
  f_θ        — generator network
  φ          — frozen feature extractor (pre-trained MAE)
  D_pos      — batch of N_pos real data samples
  N          — number of noise samples per batch

1. ε  ← randn([N, C])                  # sample Gaussian noise
2. x  ← f_θ(ε)                         # generate: x ∈ ℝ^{N×D}
3. y⁺ ← sample(D_pos, N_pos)           # positive samples ∈ ℝ^{N_pos×D}
4. y⁻ ← x (re-use generated batch)    # negative samples ∈ ℝ^{N×D}
5. h  ← φ(x)                           # feature: h ∈ ℝ^{N×F}
6. h⁺ ← φ(y⁺)                          # feature: h⁺ ∈ ℝ^{N_pos×F}
7. h⁻ ← φ(y⁻)                          # feature: h⁻ ∈ ℝ^{N×F}
8. Compute kernel weights (softmax over exponential distances)
9. V  ← attraction(h, h⁺) − repulsion(h, h⁻)  # drifting field in feature space
10. target ← stopgrad(h + V)           # frozen regression target
11. loss   ← MSE(h, target)            # = ||V||² after stop-gradient
12. θ ← θ − lr · ∇_θ loss
─────────────────────────────────────────────────────────────────
```

### Classifier-Free Guidance (CFG)

For class-conditional generation, the negative sample distribution is augmented with unconditional (null-class) samples to implement CFG without modifying inference:

$$\tilde{q}(\cdot \mid c) = (1 - \gamma)\,q_\theta(\cdot \mid c) + \gamma\,p_{\mathrm{data}}(\cdot \mid \emptyset)$$

where $\gamma$ is the guidance weight. At inference time the same $\alpha$ scale used during training is applied, requiring no extra forward passes.

---

## Model Architecture

### Generator $f_\theta$

- **Type**: DiT-style transformer (Diffusion Transformer architecture)
- **Input**: Gaussian noise $\varepsilon \in \mathbb{R}^{32 \times 32 \times 4}$ (latent space) or $\varepsilon \in \mathbb{R}^{256 \times 256 \times 3}$ (pixel space)
- **Output**: Generated sample $x$ with same shape as input
- **Patch size**: 2 (DiT/2 configuration for latent; /16 for pixel)
- **Conditioning**: adaLN-zero for class labels

| Model Variant | Parameters | Space |
|---------------|-----------|-------|
| B/2 | ~133M | Latent |
| **L/2** | **~463M** | **Latent (best)** |
| B/16 | ~133M | Pixel |
| L/16 | ~464M | Pixel |

### Feature Extractor $\phi$

- **Architecture**: ResNet-style MAE (Masked Autoencoder) pre-trained on the same latent space
- **Input**: Sample $x \in \mathbb{R}^{32 \times 32 \times 4}$
- **Output**: Multi-scale feature vector $h \in \mathbb{R}^F$ (width 256–640 depending on variant)
- **Pre-training**: Self-supervised MAE on ImageNet latents, optionally fine-tuned with classification head

> [!NOTE]
> The quality of $\phi$ strongly determines generation quality. A weak encoder (SimCLR) achieves FID 11.05 while a strong latent-MAE (640-dim, 1280 epochs) achieves FID 4.28. Fine-tuning $\phi$ with a classification objective further improves to FID 3.36.

### VAE Decoder

- **Type**: SD-VAE (Stable Diffusion VAE)
- **Mapping**: $\mathbb{R}^{32 \times 32 \times 4} \to \mathbb{R}^{256 \times 256 \times 3}$
- **Parameters**: ~49M (frozen during generator training)

---

## Experiments

### Datasets

| Dataset | Size | Task |
|---------|------|------|
| ImageNet (256×256) | 1.28M training images, 50K evaluation samples | Class-conditional image generation |
| Robotic Control (Diffusion Policy benchmark) | Multiple tasks | Visuomotor policy learning |

**Robotic tasks**: Lift, Can, ToolHang, PushT (single-stage); BlockPush, Kitchen (multi-stage), with both state-based and visual observations.

### Evaluation Metrics

- **FID** (Fréchet Inception Distance): lower is better; computed over 50K samples
- **IS** (Inception Score): higher is better
- **NFE** (Number of Function Evaluations): number of network forward passes at inference; 1 = one-step

### Main Results — ImageNet 256×256

**Latent Space (Table 5):**

| Method | NFE | FID ↓ | IS ↑ | Params |
|--------|-----|-------|------|--------|
| DiT-XL/2 | 250×2 | 2.27 | 278.2 | 675M |
| SiT-XL/2 | 250×2 | 2.06 | 270.3 | 675M |
| iMeanFlow-XL/2 | 1 | 1.72 | 282.0 | 610M |
| **Drifting L/2** | **1** | **1.54** | 258.9 | 463M |

**Pixel Space (Table 6):**

| Method | NFE | FID ↓ | IS ↑ | Params |
|--------|-----|-------|------|--------|
| SiD2, UViT/1 | 512×2 | 1.38 | — | — |
| PixelDiT/16 | 200×2 | 1.61 | 292.7 | 797M |
| **Drifting L/16** | **1** | **1.61** | **307.5** | 464M |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Total batch size | 4096 |
| Positive samples per class ($N_{\mathrm{pos}}$) | 64 |
| Negative samples per class ($N_{\mathrm{neg}}$) | 64 |
| Training epochs (final) | 1280 |
| Training epochs (ablations) | 100 |
| CFG weight ($\gamma$) | 1.0 (default) |

---

## Comparison with Related Methods

| Method | Distribution Evolution | Inference Steps | Adversarial Training | SDE/ODE |
|--------|----------------------|-----------------|---------------------|---------|
| Diffusion Models (DDPM, DiT) | At inference (SDE) | 250–1000 | No | Yes |
| Flow Matching (SiT) | At inference (ODE) | 250 | No | Yes |
| GANs | At training | 1 | Yes | No |
| Consistency Models | Approximate SDE trajectory at training | 1–4 | No | Implicit |
| **Drifting Models** | **At training (drifting field)** | **1** | **No** | **No** |

**vs. Diffusion/Flow Matching**: Both use iterative inference steps guided by score functions or flow fields. Drifting replaces inference-time iteration with training-time distribution evolution; the generator at test time is simply a one-step mapping.

**vs. GANs**: Like GANs, Drifting uses a single-pass generator, but replaces adversarial min-max optimization with a stable regression objective (MSE on drifted targets). This avoids mode collapse and training instability.

**vs. Consistency Models**: Consistency models distill an SDE/ODE into fewer steps by enforcing self-consistency along trajectories. Drifting requires no pre-trained diffusion model and does not rely on any SDE/ODE formulation.

**vs. Moment Matching / MMD**: Drifting is related to kernel-based distribution matching, but explicitly constructs a directional velocity field rather than minimizing a scalar distance metric.

---

## Ablation Studies

**Anti-symmetry (Table 1)**: Breaking the attraction/repulsion balance causes catastrophic failure — scaling attraction by 1.5× raises FID from 8.46 to 41.05; scaling repulsion by 2.0× raises FID to 112.84.

**Sample allocation**: Increasing $N_{\mathrm{pos}}$ and $N_{\mathrm{neg}}$ per batch monotonically improves FID by providing better estimates of the drifting field.

**Feature encoder quality (Table 3)**:

| Encoder | FID ↓ |
|---------|-------|
| SimCLR | 11.05 |
| MoCo-v2 | 8.41 |
| Latent-MAE (256-dim) | 6.12 |
| Latent-MAE (640-dim, 1280 epochs) | 4.28 |
| Latent-MAE + classification fine-tuning | 3.36 |

---

## Limitations

1. **Feature encoder dependency**: The method requires a high-quality self-supervised encoder; the reason why direct pixel-space drifting fails is not fully explained theoretically.
2. **Theoretical identifiability**: Sufficient conditions for identifying $p$ from the equilibrium of $\mathbf{V}_{p,q}=0$ are established only in appendices and are not tight.
3. **Kernel design**: Only the exponential distance kernel is explored; alternative kernels may offer different quality/speed trade-offs.
