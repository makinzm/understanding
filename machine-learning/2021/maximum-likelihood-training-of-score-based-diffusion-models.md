# Meta Information

- URL: [Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Song, Y., Durkan, C., Murray, I., & Ermon, S. (2021). Maximum Likelihood Training of Score-Based Diffusion Models. arXiv preprint arXiv:2101.09258.

# Maximum Likelihood Training of Score-Based Diffusion Models

Score-based diffusion models (e.g., DDPM, NCSN) generate high-quality samples by reversing a forward diffusion process that gradually corrupts data into noise. Training typically minimizes a **weighted score matching** loss, which does not directly optimize log-likelihood. This paper introduces **likelihood weighting** — a specific choice of the weighting function $\lambda(t) = g(t)^2$ — that turns the score matching objective into an upper bound on the negative log-likelihood (NLL). Combined with importance sampling and variational dequantization, the resulting model, **ScoreFlow**, achieves NLLs of 2.83 bits/dim on CIFAR-10 and 3.76 bits/dim on ImageNet 32×32, matching state-of-the-art autoregressive models.

**Who benefits**: Practitioners and researchers working on generative modeling who need principled density estimation from diffusion models, not just sample quality.

## Background: Score-Based Diffusion via SDEs

### Forward Process (Diffusing Data to Noise)

Given data $\mathbf{x}(0) \sim p(\mathbf{x})$, the forward SDE diffuses data toward a tractable prior $\pi(\mathbf{x})$ (e.g., Gaussian):

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}, \quad t \in [0, T]$$

- $\mathbf{f}(\mathbf{x}, t)$: drift coefficient (vector-valued function)
- $g(t)$: scalar diffusion coefficient
- $\mathbf{w}$: standard Wiener process

The marginal distribution at time $t$ is denoted $p_t(\mathbf{x})$, with $p_0 = p$ (data) and $p_T \approx \pi$ (noise prior).

**Two concrete SDE types used in this paper:**

| SDE type | Drift $\mathbf{f}(\mathbf{x}, t)$ | Diffusion $g(t)$ |
|---|---|---|
| Variance Preserving (VP) | $-\frac{1}{2}\beta(t)\mathbf{x}$ | $\sqrt{\beta(t)}$ |
| Sub-VP | $-\frac{1}{2}\beta(t)\mathbf{x}$ | $\sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)ds})}$ |

### Reverse Process and Score Matching

The reverse-time SDE allows generation by running diffusion backward in time:

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})\right]dt + g(t)\,d\bar{\mathbf{w}}$$

The **score function** $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ is intractable, so a neural network $\mathbf{s}_\theta(\mathbf{x}, t)$ approximates it by minimizing the **score matching objective**:

$$\mathcal{J}_{\text{SM}}(\theta; \lambda(\cdot)) \triangleq \frac{1}{2}\int_0^T \mathbb{E}\left[\lambda(t)\|\nabla_\mathbf{x} \log p_t(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}, t)\|_2^2\right] dt$$

where $\lambda(t) > 0$ is a weighting function. Prior work used heuristic weightings (e.g., $\lambda(t) = 1$ or $\lambda(t)$ chosen for signal-to-noise balance); this paper shows the **theoretically motivated** choice.

### Likelihood via Continuous Normalizing Flows (CNFs)

The log-likelihood of a data point $\mathbf{x}(0)$ under the model can be computed using the **probability flow ODE** — an ODE equivalent to the reverse SDE that generates the same marginals:

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})\right]dt$$

By the instantaneous change-of-variables formula for CNFs:

$$\log p_\theta^{\text{ODE}}(\mathbf{x}(0)) = \log \pi(\mathbf{x}(T)) + \int_0^T \nabla_\mathbf{x} \cdot \left[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \mathbf{s}_\theta(\mathbf{x}, t)\right] dt$$

Computing this requires solving the ODE and estimating trace terms — tractable but expensive. The key insight: **training directly on this objective is computationally prohibitive** during optimization, motivating the likelihood weighting bound.

## Likelihood Weighting (Core Contribution)

### Theorem 1: KL Divergence Upper Bound

Let $p(\mathbf{x})$ be the data distribution, $\pi(\mathbf{x})$ the prior, and $p_\theta^{\text{SDE}}$ the distribution induced by the reverse SDE. Under standard regularity conditions:

$$D_{\text{KL}}(p \,\|\, p_\theta^{\text{SDE}}) \leq \mathcal{J}_{\text{SM}}(\theta;\, g(\cdot)^2) + D_{\text{KL}}(p_T \,\|\, \pi)$$

> [!IMPORTANT]
> The critical result: choosing $\lambda(t) = g(t)^2$ makes $\mathcal{J}_{\text{SM}}$ an upper bound on the KL divergence (equivalently, a bound on the expected NLL up to a constant). This is the **likelihood weighting** scheme.

The term $D_{\text{KL}}(p_T \,\|\, \pi)$ is negligible when $T$ is large and the SDE sufficiently mixes.

### Theorem 2: Tightness

The bound is tight if and only if $\mathbf{s}_\theta(\mathbf{x}, t) \equiv \nabla_\mathbf{x} \log p_t(\mathbf{x})$ (perfect score matching) and $p_T = \pi$. In that case:

$$p_\theta^{\text{SDE}} = p_\theta^{\text{ODE}} = p \quad \text{(the true data distribution)}$$

This means the SDE and ODE formulations coincide at optimum.

### Theorem 3: Per-Datapoint NLL Bound

For any individual data point $\mathbf{x}$, the following bound holds:

$$-\log p_\theta^{\text{SDE}}(\mathbf{x}) \leq \mathcal{L}_\theta^{\text{DSM}}(\mathbf{x})$$

where $\mathcal{L}_\theta^{\text{DSM}}(\mathbf{x})$ is the **denoising score matching** upper bound, computable as:

$$\mathcal{L}_\theta^{\text{DSM}}(\mathbf{x}) \triangleq \frac{1}{2}\int_0^T g(t)^2 \mathbb{E}_{\mathbf{x}(t) \sim p_{0t}(\cdot|\mathbf{x})}\left[\|\mathbf{s}_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t)|\mathbf{x})\|_2^2\right] dt + C(\mathbf{x})$$

where $C(\mathbf{x})$ is a constant independent of $\theta$, and $p_{0t}(\mathbf{x}(t)|\mathbf{x})$ is the Gaussian transition kernel of the forward SDE (known analytically).

> [!NOTE]
> This per-datapoint bound enables **variational dequantization** (treating discrete pixel values as continuous by adding noise) in combination with the diffusion model likelihood bound.

## Variance Reduction via Importance Sampling

The score matching objective with likelihood weighting becomes:

$$\mathcal{J}_{\text{SM}}(\theta;\, g(\cdot)^2) = \frac{1}{2}\int_0^T g(t)^2 \,\mathbb{E}\left[\|\nabla_\mathbf{x} \log p_t(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}, t)\|_2^2\right] dt$$

The integrand's magnitude can vary greatly over $t$. To reduce variance in Monte Carlo estimation, importance sampling replaces uniform sampling of $t \sim \text{Uniform}[0, T]$ with a **proposal distribution** $p_{\text{IS}}(t)$ proportional to $g(t)^2$:

$$\int_0^T g(t)^2 h(t)\,dt = Z \int_0^T p_{\text{IS}}(t)\,\alpha(t)^2 h(t)\,dt$$

where $Z = \int_0^T g(t)^2\,dt$ is the normalizing constant and $\alpha(t) = g(t)/\sqrt{p_{\text{IS}}(t)}$ are the IS correction weights. Sampling $\tilde{t} \sim p_{\text{IS}}$ and reweighting by $\alpha(\tilde{t})^2$ gives a lower-variance estimate of the objective.

## Training Algorithm

**Input**: Dataset $\mathcal{D}$, score network $\mathbf{s}_\theta$, SDE parameters $(f, g)$, importance sampling distribution $p_{\text{IS}}$

**Training procedure** (one iteration):
1. Sample data point $\mathbf{x}(0) \sim \mathcal{D}$
2. Sample time $t \sim p_{\text{IS}}$ (proposal distribution based on $g(t)^2$)
3. Sample noisy $\mathbf{x}(t) \sim p_{0t}(\cdot|\mathbf{x}(0))$ using the known Gaussian transition kernel
4. Compute the denoising score matching loss with likelihood weighting:
   $$\ell = \frac{1}{2} \alpha(t)^2 \cdot \|\mathbf{s}_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))\|_2^2$$
5. Update $\theta$ via gradient descent on $\ell$

**Evaluation** (likelihood computation):
- Solve the probability flow ODE to compute $\log p_\theta^{\text{ODE}}(\mathbf{x}(0))$ using a numerical ODE solver
- Alternatively, use $\mathcal{L}_\theta^{\text{DSM}}(\mathbf{x})$ as an upper bound (cheaper)

> [!NOTE]
> Likelihood-weighted training does not require solving the ODE at training time — only at evaluation. This keeps training cost comparable to standard score matching.

## Variational Dequantization

Real images have discrete pixel values (e.g., integers in $[0, 255]$). To model discrete data with a continuous density model, **dequantization** adds uniform noise: $\tilde{\mathbf{x}} = \mathbf{x} + \mathbf{u}$, where $\mathbf{u} \sim \text{Uniform}[0, 1]^D$.

**Variational dequantization** (Ho et al., 2019) improves this by learning the noise distribution $q_\phi(\mathbf{u}|\mathbf{x})$:

$$\log p_\theta^{\text{dequant}}(\mathbf{x}) \geq \mathbb{E}_{\mathbf{u} \sim q_\phi(\cdot|\mathbf{x})}\left[\log p_\theta(\mathbf{x} + \mathbf{u}) - \log q_\phi(\mathbf{u}|\mathbf{x})\right]$$

Substituting the per-datapoint bound from Theorem 3 for $\log p_\theta(\mathbf{x} + \mathbf{u})$ gives a jointly trainable ELBO over both $\theta$ (score network) and $\phi$ (dequantization flow).

## Comparison with Related Methods

| Method | Likelihood Weighting | Direct NLL Optimization | Sample Quality (FID) | NLL Training Cost |
|---|---|---|---|---|
| DDPM (Ho et al., 2020) | Heuristic ($\lambda=1$) | No | Excellent | Cheap |
| Song et al. (2021) SMLD/DDPM | Various heuristics | No | Excellent | Cheap |
| Continuous NF (Grathwohl et al.) | N/A | Yes (direct) | Moderate | Expensive (ODE at train time) |
| **This work (ScoreFlow)** | $\lambda(t) = g(t)^2$ (principled) | No (bound) | Excellent | Cheap (ODE only at eval) |

> [!IMPORTANT]
> Key differentiator from standard diffusion training: prior work uses heuristic weighting designed for sample quality. This work derives the **theoretically correct** weighting for likelihood, without requiring expensive ODE integration during training.

> [!TIP]
> See Song et al. (2021) "Score-Based Generative Modeling through Stochastic Differential Equations" ([arXiv:2011.13456](https://arxiv.org/abs/2011.13456)) for the SDE framework that this paper builds upon.

## Experiments

- **Datasets**:
  - CIFAR-10: 50,000 training images, 10,000 test images, $32 \times 32$ RGB
  - ImageNet 32×32: Downsampled ImageNet, $32 \times 32$ RGB
- **Hardware**: Google Cloud TPU v3 (via TPU Research Cloud)
- **Optimizer**: Adam
- **Architecture**: DDPM++ (standard and deep variants)
- **SDE variants tested**: VP-SDE, subVP-SDE

**Key quantitative results**:

| Model | CIFAR-10 NLL (bits/dim) | ImageNet NLL (bits/dim) | FID (CIFAR-10) |
|---|---|---|---|
| Baseline (VP, heuristic weighting) | 3.04 | 3.96 | 3.98 |
| + Likelihood weighting only | 2.94 | 3.92 | 5.18 |
| + Likelihood weighting + IS | 2.83 | 3.80 | 6.03 |
| ScoreFlow (+ variational dequant) | 2.83 | **3.76** | — |

ScoreFlow matches PixelSNAIL (2.85 bits/dim) and $\delta$-VAE (2.83 bits/dim) on CIFAR-10 NLL, while maintaining good sample quality (FID improves slightly with dequantization compared to non-dequantized variants).

> [!CAUTION]
> Adding likelihood weighting slightly increases FID compared to heuristic weighting (5.18 vs. 3.98). There is a trade-off: optimizing for likelihood does not necessarily maximize sample quality as measured by FID, since FID measures distribution distance in feature space rather than exact density.
