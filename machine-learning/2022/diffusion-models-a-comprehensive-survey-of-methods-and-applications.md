# Meta Information

- URL: [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/abs/2209.00796)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., & Yang, M.-H. (2023). Diffusion Models: A Comprehensive Survey of Methods and Applications. *ACM Computing Surveys*.

# Diffusion Models: A Comprehensive Survey of Methods and Applications

This survey organizes diffusion model research (as of 2022) into three algorithmic improvement directions and seven application domains. The core contribution is demonstrating that DDPM, SGM, and Score SDE—the three dominant formulations—are mathematically equivalent under specific parameterizations, and then surveying methods that improve upon each.

**Applicability**: Researchers and practitioners who need to choose among diffusion model variants, understand efficiency trade-offs, or apply diffusion models to specialized domains (e.g., molecular generation, medical imaging, NLP).

---

## 1. Three Foundational Formulations

### 1.1 Denoising Diffusion Probabilistic Models (DDPM)

[[2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

DDPM defines a **forward Markov chain** that progressively adds Gaussian noise over $T$ steps, and learns a **reverse denoising process** parameterized by a neural network.

**Forward Process** — Input: clean data $x_0 \in \mathbb{R}^d$; Output: noisy sequence $x_1, \ldots, x_T \in \mathbb{R}^d$

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t \mathbf{I}\right)$$

where $\beta_t \in (0,1)$ is a pre-defined noise schedule. Via reparameterization, any intermediate step can be sampled directly:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t) \mathbf{I}\right)$$

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,\mathbf{I})$$

where $\alpha_t := 1 - \beta_t$ and $\bar\alpha_t := \prod_{s=0}^{t} \alpha_s$.

**Reverse Process** — Input: noise sample $x_T \sim \mathcal{N}(0,\mathbf{I})$; Output: approximate clean data $x_0$

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\right)$$

The following inequality holds, so maximizing the evidence lower bound (ELBO) is equivalent to minimizing the KL divergence between the forward and reverse processes:

```math
\begin{aligned}
\mathbf{KL} &= \mathbb{E}_{q(x_{0:T})}\!\left[\log \frac{q(x_{1:T} \mid x_0)}{p_\theta(x_{0:T})}\right] \\
&= \int q(x_0, ..., x_T) \log \frac{q(x_1 \mid x_0) q(x_2 \mid x_1) \cdots q(x_T \mid x_{T-1})}{p_\theta(x_T) p_\theta(x_{T-1} \mid x_T) \cdots p_\theta(x_0 \mid x_1)} dx_0 \cdots dx_T \\
&= \int  q(x_0, ..., x_T) \log \Pi_{t=1}^T q(x_t \mid x_{t-1}) dx_0 \cdots dx_T - \int  q(x_0, ..., x_T) \log \Pi_{t=1}^T p_\theta(x_{t-1} \mid x_t) dx_0 \cdots dx_T \\
&= \sum_{t=1}^T \text{Entropy}(q(x_t \mid x_{t-1})) - \text{[ELBO\ a.k.a.\ VLB]}(p_\theta) \\
&\geq -\text{ELBO}(p_\theta)
\end{aligned}
```

**Simplified Training Objective** (noise prediction):

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon}\!\left[\lambda(t)\, \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]$$

where $\varepsilon_\theta$ is the U-Net denoising network and $\lambda(t)$ is a time-dependent weighting.

> [!NOTE]
> Ho et al. (2020) found that setting $\lambda(t) = 1$ (the "simplified" objective) performs better than the full VLB weighting in practice.

### 1.2 Score-Based Generative Models (SGM)

[[2101.09258] Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258)

SGM estimates the **score function** $\nabla_x \log p(x)$ (gradient of the log-density) at multiple noise levels, then generates samples via **annealed Langevin dynamics**.

**Denoising Score Matching Objective**:
```math
\begin{aligned}
\mathbb{E}_{t, x_0, \varepsilon}\!\left[\lambda(t)\, \|\nabla_{x_t} \log q(x_t) - s_\theta(x_t, t)\|^2\right] &=
\mathbb{E}_{t, x_0, \varepsilon}\!\left[\lambda(t)\, \|\varepsilon + \sigma_t\, s_\theta(x_t, t)\|^2\right] + \text{const}
\end{aligned}
```

where $s_\theta(x_t, t) \approx \nabla_{x_t} \log q(x_t)$ is the score network.

**Annealed Langevin Dynamics Sampling**:

$$x_t^{(i+1)} \leftarrow x_t^{(i)} + \frac{s_t}{2}\, s_\theta(x_t^{(i)}, t) + \sqrt{s_t}\, \varepsilon^{(i)}, \quad \varepsilon^{(i)} \sim \mathcal{N}(0,\mathbf{I})$$

As step size $s_t \to 0$ and iterations $N \to \infty$, the samples converge to the true distribution.

### 1.3 Score SDEs (Continuous-Time Generalization)

[[2011.13456] Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

Score SDEs unify DDPM and SGM under a single continuous-time framework using stochastic differential equations.

**General Forward SDE**:

$$dx = f(x,t)\, dt + g(t)\, dw$$

where $f$ is the drift, $g$ is the diffusion coefficient, and $w$ is a standard Wiener process.

- **VP SDE** (DDPM continuous limit): $dx = -\frac{1}{2}\beta(t)x\, dt + \sqrt{\beta(t)}\, dw$
- **VE SDE** (SGM continuous limit): $dx = \sqrt{\tfrac{d[\sigma(t)^2]}{dt}}\, dw$

**Reverse-Time SDE** (Anderson, 1982):

$$dx = \left[f(x,t) - g(t)^2 \nabla_x \log q_t(x)\right] dt + g(t)\, d\bar{w}$$

where $\bar{w}$ is a backward Wiener process.

**Probability Flow ODE** (deterministic equivalent):

$$dx = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log q_t(x)\right] dt$$

This ODE has the same marginal distributions $q_t(x)$ as the reverse SDE, enabling exact likelihood computation via change-of-variables.

> [!IMPORTANT]
> DDPM, SGM, and Score SDEs are mathematically equivalent: they share the same training objective (scaled noise prediction = scaled score matching) and the same marginal distributions. The Score SDE framework is strictly more general, encompassing the other two as special cases.

**Equivalence summary**:

| Formulation | Process type | Training loss reduces to | Sampling |
|---|---|---|---|
| DDPM | Discrete Markov chain | $\|\varepsilon - \varepsilon_\theta\|^2$ | Ancestral sampling |
| SGM | Multi-scale score matching | $\|\varepsilon + \sigma_t s_\theta\|^2$ | Annealed Langevin |
| Score SDE | Continuous SDE | Generalized score matching | SDE/ODE solvers |

---

## 2. Algorithmic Improvements

### 2.1 Efficient Sampling

The primary bottleneck of diffusion models is slow inference (hundreds to thousands of NFE — neural function evaluations). Methods divide into **learning-free** and **learning-based** approaches.

#### Learning-Free Solvers

**DDIM (Denoising Diffusion Implicit Models)**: [[2010.02502] Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

Extends DDPM to non-Markovian forward processes with $\sigma_t = 0$, yielding a deterministic reverse mapping that enables **subsampling** of timesteps (e.g., 50 instead of 1000):

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \frac{x_t - \sqrt{1-\bar\alpha_t}\, \varepsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}} + \sqrt{1-\bar\alpha_{t-1}}\, \varepsilon_\theta(x_t,t)$$

> [!NOTE]
> DDIM samples are deterministic given the same initial noise, enabling meaningful latent space interpolation—unlike DDPM stochastic sampling.

**DPM-Solver**: [[2206.00927] DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)

Exploits the semi-linear structure of the probability flow ODE to derive a tailored solver. Unlike generic ODE solvers (Euler, Heun), DPM-Solver computes an analytical linear part and numerically integrates only the nonlinear score network contribution, producing high-quality samples in **10–20 NFE** versus hundreds for DDPM.

#### Learning-Based Methods

**Progressive Distillation**: [[2202.00512] Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)

Trains a student model to match two-step DDIM output using a single step, then repeats iteratively. Halves required steps each round, enabling 4-step generation without quality loss.

**Consistency Models**: [[2303.01469] Consistency Models](https://arxiv.org/abs/2303.01469)

Directly learn a mapping from any noisy $x_t$ on the same ODE trajectory back to $x_0$, by enforcing **self-consistency**: $f_\theta(x_t, t) = f_\theta(x_{t'}, t')$ for any two points on the same trajectory.

### 2.2 Improved Likelihood Estimation

**Cosine Noise Schedule (iDDPM)**: [[2102.09672] Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

Replaces linear $\beta_t$ schedule with:

$$\bar\alpha_t = \frac{h(t)}{h(0)}, \quad h(t) = \cos^2\!\left(\frac{t/T + m}{1+m} \cdot \frac{\pi}{2}\right)$$

This avoids sudden large noise additions at the start/end of the chain that hurt likelihood.

**Reverse Variance Learning (iDDPM)**: [[2201.06503] Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models](https://arxiv.org/abs/2201.06503)

Rather than fixing $\Sigma_\theta = \beta_t \mathbf{I}$ or $\Sigma_\theta = \tilde\beta_t \mathbf{I}$, learns an interpolation:

$$\Sigma_\theta(x_t,t) = \exp\!\left(v \cdot \log \beta_t + (1-v) \cdot \log \tilde\beta_t\right)$$

where $v$ is network-predicted and $\tilde\beta_t := \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} \cdot \beta_t$ is the lower bound variance.

**Exact Likelihood via ODE**: [[2206.08265] Maximum Likelihood Training for Score-Based Diffusion ODEs by High-Order Denoising Score Matching](https://arxiv.org/abs/2206.08265)

The probability flow ODE enables exact log-likelihood via the instantaneous change-of-variables formula:

$$\log p_\theta^{\text{ODE}}(x_0) = \log p_T(x_T) + \int_0^T \nabla \cdot \tilde{f}_\theta(x_t, t)\, dt$$

where the trace of the Jacobian is estimated efficiently via the Skilling–Hutchinson estimator.

### 2.3 Special Data Structures

#### Discrete Data (NLP)

Standard Gaussian diffusion requires continuous inputs. For discrete tokens:

**Multinomial Diffusion / VQ-Diffusion** uses a categorical transition matrix:

$$q(x_t \mid x_{t-1}) = v^\top(x_t)\, Q_t\, v(x_{t-1})$$

where $v(x)$ is the one-hot encoding and $Q_t$ is a "mask-and-absorb" or "uniform" transition matrix.

- [[2111.14822] Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/abs/2111.14822)

**Diffusion-LM** operates in an embedding space, adding a rounding projection step to map continuous samples back to vocabulary.

- [[2205.14217] Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)

#### Invariant Structures (Molecules)

For 3D molecular generation, the score network must be **equivariant** to roto-translations (SE(3)-equivariant). E(n)-equivariant GNN architectures (e.g., EGNN) are used as the score network backbone.

#### Manifold-Constrained Data

For data lying on Riemannian manifolds (e.g., spheres, hyperbolic spaces), the Gaussian noise process is replaced with **heat kernel diffusion** on the manifold, and Riemannian Langevin dynamics for sampling.

- [[2208.07949] Riemannian Diffusion Models](https://arxiv.org/abs/2208.07949)

---

## 3. Connections to Other Generative Models

| Generative Model | Relationship to Diffusion |
|---|---|
| VAE | Score SDE encoder ≈ VAE encoder; latent diffusion decouples perception and generation |
| GAN | Diffusion outperforms GAN on FID in unconditional image generation; GAN faster at inference |
| Normalizing Flows | Probability flow ODE is a continuous normalizing flow with fixed architecture |
| Energy-Based Models | Score function = gradient of energy; EBM MCMC sampling ≈ Langevin dynamics |
| Autoregressive Models | Both exact; AR scales poorly with sequence length; diffusion allows parallel sampling |

---

## 4. Applications

### 4.1 Computer Vision

- **Super-resolution**: SR3 and Palette use DDPM conditioned on low-res images; CDM (Cascaded Diffusion Models) achieves ImageNet FID 4.88 with cascaded 64→256→1024 generation
- **Inpainting**: RePaint masks pixels and iteratively re-noises/denoises for coherent completion
- **Video generation**: Video Diffusion Models extend U-Net with 3D convolutions and attention over space-time
- **Point cloud generation**: Applying score SDE on 3D point sets with SE(3)-equivariant networks

### 4.2 Natural Language Processing

Diffusion in text is challenging due to discrete tokens. Approaches include diffusion on embeddings (Diffusion-LM) and masked/absorbing multinomial diffusion (MaskGIT-style).

### 4.3 Multimodal Generation

- **Text-to-image**: DALL-E 2 uses CLIP embeddings + diffusion prior; Stable Diffusion uses Latent Diffusion Model (LDM) — diffusion in a compressed VAE latent space $z \in \mathbb{R}^{h \times w \times c}$ rather than pixel space, reducing computation by $\sim8\times$
- **Text-to-3D**: DreamFusion distills 2D diffusion model knowledge via Score Distillation Sampling (SDS)
- **Text-to-audio**: AudioLM and related models apply diffusion on audio codec tokens

### 4.4 Temporal Data

- **Time-series imputation**: CSDI conditions diffusion on observed dimensions for multivariate time series
- **Forecasting**: TimeGrad applies recurrent score network over autoregressive prediction steps

### 4.5 Interdisciplinary Applications

- **Drug design / molecular generation**: GEOM dataset (GEOMol); DiffSBDD and TargetDiff generate molecules conditioned on protein binding sites
- **Medical imaging**: Score-SDE inversion for MRI/CT reconstruction; diffusion for MRI super-resolution and anomaly detection
- **Robust learning**: Diffusion purification uses forward-reverse process to remove adversarial perturbations before classification

---

## 5. Comparison with Similar Approaches

### Diffusion vs. GAN

GANs generate samples in a single forward pass (fast inference) but suffer from mode collapse and training instability. Diffusion models require many NFE at inference but are more stable to train and achieve better sample diversity and likelihood.

### Diffusion vs. Normalizing Flows

Normalizing flows require bijective architectures with tractable Jacobians, heavily constraining network design. The probability flow ODE formulation of score SDEs is a continuous normalizing flow (CNF) but trained via score matching rather than maximum likelihood, avoiding the Jacobian bottleneck.

### DDPM vs. Score SDE (Discrete vs. Continuous)

DDPM uses $T=1000$ discrete steps with fixed schedule; Score SDEs can use adaptive ODE solvers, enable flexible step count at inference, and provide exact likelihood without approximation.

---

# Experiments

- **Datasets**: CIFAR-10 ($32\times32$ images), ImageNet ($64\times64$, $256\times256$, $1024\times1024$), LSUN (church, bedroom), CelebA-HQ, FFHQ (face generation); GEOM dataset for molecular generation; various medical imaging datasets (MRI, CT)
- **Hardware**: Not consolidated in survey (varies per cited paper)
- **Key Quantitative Results**:
  - CDM achieves FID 4.88 on ImageNet $256\times256$ (surpassing BigGAN-deep: 6.95)
  - DPM-Solver reaches comparable quality to DDIM-1000 in 10–20 NFE
  - Stable Diffusion (LDM) reduces inference memory vs. pixel-space diffusion by $\sim8\times$ while maintaining sample quality
  - Score SDE achieves FID 2.20 on CIFAR-10 (state-of-the-art at submission time)
