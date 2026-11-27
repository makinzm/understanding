# Meta Information

- URL: [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021.

# Score-Based Generative Modeling through Stochastic Differential Equations

## Overview

This paper presents a unified framework for score-based generative modeling by casting both the forward noising process and the reverse generative process as **Stochastic Differential Equations (SDEs)**. The core principle is: transforming data into noise is straightforward via a forward SDE; reversing this SDE yields a generative model that can synthesize data from noise.

The framework subsumes existing methods—Score Matching with Langevin Dynamics (SMLD) and Denoising Diffusion Probabilistic Models (DDPM)—as special discretizations of particular SDE families, and introduces new capabilities: exact likelihood computation, flexible sampling strategies, and controllable generation via inverse problem solving.

> [!NOTE]
> "Creating noise from data is easy; creating data from noise is generative modeling." — a central design principle of the paper.

**Who uses this**: Researchers and practitioners building generative image models, density estimators, or solving inverse problems (inpainting, colorization, super-resolution) with a single pretrained model.

---

## Background: Score Matching and Langevin Dynamics

A **score function** is the gradient of the log probability density: $\nabla_x \log p(x)$.

Prior work (SMLD) trained a neural network $s_\theta(x, \sigma)$ to approximate $\nabla_x \log p_\sigma(x)$ for a sequence of noise levels $\sigma_1 < \sigma_2 < \cdots < \sigma_N$, then sampled via annealed Langevin dynamics. DDPM similarly defined a discrete Markov chain of noising steps and reversed it. Both are now interpreted as special cases of the SDE framework.

---

## Forward SDE: Diffusing Data to Noise

**Input**: Data sample $x(0) \sim p_0(x)$ where $p_0$ is the (unknown) data distribution.
**Output**: $x(T) \sim p_T(x)$, a tractable prior (typically $\mathcal{N}(0, I)$).

The continuous forward process is defined by the Itô SDE:

$$dx = f(x, t)\,dt + g(t)\,dw, \quad t \in [0, T]$$

where:
- $f(x, t) \in \mathbb{R}^d$ is the **drift coefficient** (vector-valued)
- $g(t) \in \mathbb{R}$ is the **diffusion coefficient** (scalar)
- $w \in \mathbb{R}^d$ is a standard Wiener process (Brownian motion)
- $p_t(x)$ denotes the marginal distribution of $x(t)$

### Three SDE Families

| Family | Drift $f(x,t)$ | Diffusion $g(t)$ | Relation to prior work |
|--------|---------------|-----------------|------------------------|
| **VE (Variance Exploding)** | $0$ | $\sqrt{d[\sigma^2(t)]/dt}$ | Generalizes SMLD |
| **VP (Variance Preserving)** | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)}$ | Generalizes DDPM |
| **sub-VP** | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)ds})}$ | Novel; tighter variance bound |

For VE-SDE, variance grows without bound: $\sigma(t) \to \infty$ as noise increases. For VP-SDE, the marginal variance is bounded to $[0, 1]$. Sub-VP has variance strictly smaller than VP at all $t$, which the authors find beneficial for likelihood estimation.

---

## Reverse-Time SDE: Generating Data from Noise

**Input**: $x(T) \sim p_T$ (noise sample).
**Output**: $x(0) \sim p_0$ (data sample).

By Anderson (1982), the time-reversal of the forward SDE is:

$$dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar{w}$$

where $\bar{w}$ is a reverse-time Wiener process and $dt$ is an infinitesimal negative timestep.

The unknown term $\nabla_x \log p_t(x)$ — the **time-dependent score function** — is approximated by a neural network $s_\theta(x, t) \approx \nabla_x \log p_t(x)$.

---

## Training Objective

**Input**: Training pairs $(x(0), x(t))$ where $x(t)$ is obtained by forward diffusion.
**Output**: Score network parameters $\theta^*$.

The continuous-time denoising score matching objective is:

$$\theta^* = \arg\min_\theta \mathbb{E}_t \left\{ \lambda(t) \mathbb{E}_{x(0)} \mathbb{E}_{x(t)|x(0)} \left[ \| s_\theta(x(t), t) - \nabla_{x(t)} \log p_{0t}(x(t)|x(0)) \|_2^2 \right] \right\}$$

where:
- $t$ is sampled uniformly from $[0, T]$
- $\lambda(t) > 0$ is a positive weighting function
- $p_{0t}(x(t)|x(0))$ is the transition kernel of the forward SDE (often Gaussian in closed form)

The target $\nabla_{x(t)} \log p_{0t}(x(t)|x(0))$ is analytically tractable because $p_{0t}$ is Gaussian for linear SDEs. For VE-SDE: $\nabla_x \log p_{0t}(x(t)|x(0)) = (x(t) - x(0))/\sigma^2(t)$.

> [!IMPORTANT]
> This objective unifies SMLD and DDPM training as special cases. SMLD uses discrete noise levels and denoising score matching; DDPM uses discrete steps and $\epsilon$-prediction. Both are recovered by choosing appropriate $f$, $g$, and $\lambda$.

---

## Sampling Methods

### Predictor-Corrector (PC) Samplers

**Algorithm**:
```
Given: score network s_θ, T timesteps, Corrector steps M
Initialize x(T) ~ p_T

For t from T to 0:
  # Predictor step (numerical SDE solver)
  x(t-Δt) = NumericalSDE(x(t), s_θ, Δt)

  # Corrector step (score-based MCMC)
  For m in 1..M:
    z ~ N(0, I)
    ε = 2α ||z||₂² / ||s_θ(x, t)||₂²   # step size
    x(t-Δt) = x(t-Δt) + ε·s_θ(x(t-Δt), t-Δt) + √(2ε)·z

Return x(0)
```

- **Predictors**: Euler-Maruyama, Reverse Diffusion, Ancestral Sampling
- **Correctors**: Annealed Langevin Dynamics, HMC (score-based MCMC)

PC samplers outperform pure Euler-Maruyama discretization by correcting accumulated errors at each step.

### Probability Flow ODE

An alternative **deterministic** sampling path corresponding to the same marginal distributions $\{p_t\}$:

$$dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right] dt$$

**Input**: $x(T) \sim p_T$.
**Output**: $x(0) \approx p_0$, computed deterministically.

Key properties:
- Can use any black-box ODE solver (e.g., RK45 with adaptive stepsize), reducing NFE by >90%
- Enables **exact log-likelihood computation** via the instantaneous change-of-variables formula
- Provides unique **latent codes** (encodings) for each data point, enabling interpolation and temperature scaling

> [!TIP]
> The probability flow ODE is equivalent to a **Neural ODE** (Chen et al., 2018). See [Neural ODEs](https://arxiv.org/abs/1806.07366) for background on continuous-depth models.

---

## Exact Likelihood Computation

Using the probability flow ODE, the log-likelihood of data $x(0)$ is:

$$\log p_0(x(0)) = \log p_T(x(T)) + \int_0^T \nabla \cdot \tilde{f}_\theta(x(t), t)\, dt$$

where $\tilde{f}_\theta(x, t) = f(x,t) - \frac{1}{2}g(t)^2 s_\theta(x, t)$ and the divergence $\nabla \cdot \tilde{f}_\theta$ is estimated via the Skilling-Hutchinson trace estimator:

$$\nabla \cdot \tilde{f}_\theta(x, t) \approx \epsilon^\top \frac{\partial \tilde{f}_\theta(x, t)}{\partial x} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This makes score-based diffusion models the first to support exact (up to ODE solver tolerance) likelihood evaluation, matching or surpassing normalizing flows in bits/dim.

---

## Controllable Generation (Solving Inverse Problems)

For a downstream task with observed $y$ (e.g., masked image for inpainting), the conditional score is:

$$\nabla_x \log p_t(x|y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y|x)$$

**Conditional reverse SDE**:

$$dx = \left[f(x,t) - g(t)^2 \left(\nabla_x \log p_t(x) + \nabla_x \log p_t(y|x)\right)\right] dt + g(t)\,d\bar{w}$$

The unconditional score $\nabla_x \log p_t(x)$ is provided by the pretrained $s_\theta$; the likelihood score $\nabla_x \log p_t(y|x)$ is computed analytically or approximated for each task.

**Applications**:
- **Class-conditional generation**: $p(y|x)$ from a Gaussian-noise-corrupted classifier
- **Image inpainting**: $p(y|x)$ is uniform on known pixels, imposes constraints
- **Image colorization**: Condition on grayscale channel

> [!IMPORTANT]
> No retraining or fine-tuning is required. The same unconditional score model serves all inverse problems by plugging in the appropriate $\nabla_x \log p_t(y|x)$.

---

## Architecture

Two families of U-Net-based architectures are introduced:

| Model | Target SDE | Features |
|-------|-----------|----------|
| **NCSN++** | VE-SDE | 4× depth increase vs. original NCSN, continuous time conditioning |
| **DDPM++** | VP / sub-VP SDE | Adapted from DDPM architecture, continuous time embedding |

Continuous-time conditioning replaces the discrete noise-level index with a scalar $t$, allowing evaluation at arbitrary noise levels during PC sampling.

---

## Comparison with Similar Algorithms

| Aspect | SMLD (NCSN) | DDPM | **This Work (SDE framework)** |
|--------|------------|------|-------------------------------|
| Process type | Discrete noise sequence | Discrete Markov chain | Continuous SDE |
| Sampling | Annealed Langevin | Ancestral sampling | PC sampler or ODE solver |
| Likelihood | Not available | Variational lower bound | **Exact** (via ODE) |
| Latent codes | Not available | Not available | **Unique** (ODE invertible) |
| Conditional gen. | Not supported | Not supported | **Yes** (no retraining) |
| Generation quality (FID) | ~5 | ~3.17 | **~2.20** |

---

## Experiments

- **Dataset**: CIFAR-10 (50k train / 10k test, 32×32 images), CelebA-HQ (256×256 and 1024×1024)
- **Hardware**: Not explicitly stated; implied TPU/GPU clusters for training
- **Optimizer**: Adam with learning rate decay
- **Evaluation metrics**: FID (Frechet Inception Distance), Inception Score (IS), NLL (bits/dim)

**Key Results**:
- NCSN++ (VE-SDE) achieves FID **2.20** on CIFAR-10, outperforming DDPM (3.17) and StyleGAN2-ADA (2.92)
- DDPM++ (sub-VP SDE) achieves **2.99 bits/dim** on CIFAR-10, surpassing all prior generative models including normalizing flows (Flow++: 3.29)
- First score-based model capable of generating high-resolution (1024×1024) CelebA-HQ images
- PC sampler with 2000 NFE surpasses ancestral sampling with the same budget; ODE solver achieves competitive quality with <100 NFE

> [!NOTE]
> Using the probability flow ODE with adaptive step-size solvers (e.g., RK45) reduces neural function evaluations by more than 90% compared to fixed-step Euler-Maruyama with 2000 steps.

---

## Limitations

- **Sampling speed**: Even with ODE solvers, generation is slower than GANs, which require a single forward pass.
- **Hyperparameter sensitivity**: The PC sampler introduces predictor/corrector combinations and Langevin step sizes requiring careful tuning; no universal guidelines are provided.
- **Conditional likelihood approximation**: The $\nabla_x \log p_t(y|x)$ term for complex observation models may be hard to compute analytically, limiting inverse problem generality.
