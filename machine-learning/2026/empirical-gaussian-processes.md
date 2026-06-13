# Meta Information

- URL: [Empirical Gaussian Processes](https://arxiv.org/abs/2602.12082)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Jihao Andreas Lin, Sebastian Ament, Louis C. Tiao, David Eriksson, Maximilian Balandat, Eytan Bakshy (2026). Empirical Gaussian Processes. arXiv:2602.12082.

# Empirical Gaussian Processes

## Overview

Empirical Gaussian Processes (EGPs) address a fundamental limitation in classical GP modeling: the reliance on handcrafted parametric kernel functions. Standard kernels (e.g., RBF, Matérn) require domain expertise to choose and impose restrictive assumptions on the prior covariance structure. EGPs instead **learn the GP prior empirically from historical datasets**, allowing the prior to reflect rich, non-trivial covariance structures present in real data without manual specification.

**Target users**: Machine learning practitioners and researchers who have access to historical task-related datasets (e.g., hyperparameter optimization logs, time series from similar domains) and want to adapt GP priors to their problem structure rather than hand-designing kernels.

**Five main contributions:**
1. A framework for learning non-parametric GP priors from $S$ independent datasets via maximum likelihood estimation.
2. Theoretical convergence proof: EGPs weakly converge (in KL-divergence sense) to the best Gaussian approximation of the true data-generating process.
3. Closed-form Expectation-Maximization (EM) algorithm with exact updates, extended from discrete grids to continuous heterogeneous observation domains.
4. Residual interpolation technique to prevent variance collapse when extrapolating beyond historical data ranges.
5. Demonstrated competitive performance on GIFT-Eval time series benchmark and LCBench learning curve extrapolation.

## Background: Gaussian Processes

A Gaussian Process $f \sim \mathcal{GP}(\mu, k)$ is fully specified by its mean function $\mu: \mathcal{X} \to \mathbb{R}$ and kernel (covariance) function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$. Given noisy observations $\mathbf{y} = f(\mathbf{X}) + \boldsymbol{\varepsilon}$ where $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$, the posterior is also Gaussian with analytically tractable mean and covariance.

The core challenge: choosing $\mu$ and $k$ appropriately. Handcrafted kernels encode human assumptions; the EGP approach replaces this with data-driven estimation.

## Empirical Statistics

Given $S$ historical function samples $\{f_i\}_{i=1}^{S}$ (from related tasks), the **empirical mean** and **empirical covariance** are:

$$\hat{\mu}(\mathbf{x}) = \frac{1}{S} \sum_{i=1}^{S} f_i(\mathbf{x})$$

$$\hat{k}(\mathbf{x}, \mathbf{x}') = \frac{1}{S-1} \sum_{i=1}^{S} \bigl(f_i(\mathbf{x}) - \hat{\mu}(\mathbf{x})\bigr)\bigl(f_i(\mathbf{x}') - \hat{\mu}(\mathbf{x}')\bigr)$$

These empirical statistics define the EGP prior: $f \sim \mathcal{GP}(\hat{\mu}, \hat{k})$.

**Theoretical guarantee**: For almost every sequence of sample paths, the EGP weakly converges to the GP that minimizes KL-divergence from the true data-generating process, as $S \to \infty$.

## Setting 1: Discrete Observations (Dense Grids)

**Input**: $S$ datasets $\{(\mathbf{X}_i, \mathbf{y}_i)\}_{i=1}^S$ where observations are on a shared dense grid of size $N$.

**Key operation**: Compute empirical mean vector $\hat{\boldsymbol{\mu}} \in \mathbb{R}^N$ and empirical covariance matrix $\hat{\mathbf{K}} \in \mathbb{R}^{N \times N}$ at the grid points. $\hat{\mathbf{K}}$ is positive semi-definite by construction.

**Extension to new inputs** $\mathbf{x}^*$ not on the grid: Use a base kernel $k_\theta$ for cross-covariance interpolation:

$$k_{\text{EGP}}(\mathbf{x}^*, \mathbf{X}) = k_\theta(\mathbf{x}^*, \mathbf{X}) \mathbf{K}_\theta(\mathbf{X}, \mathbf{X})^{-1} \hat{\mathbf{K}}(\mathbf{X}, \mathbf{X})$$

**SVD acceleration**: Replace full datasets with $M \ll S$ eigen-observations derived via singular value decomposition of the centered data matrix, reducing complexity from $O(S)$ to $O(M)$ per prediction.

## Setting 2: Continuous-Domain (Sparse/Irregular Observations)

**Input**: $S$ datasets $\{(\mathbf{X}_i, \mathbf{y}_i)\}_{i=1}^S$ where $\mathbf{X}_i \in \mathbb{R}^{N_i \times d}$ are spatially heterogeneous (different tasks observe at different locations, $N_i$ varies).

This setting requires a principled probabilistic model since direct empirical statistics are not computable across misaligned grids.

### Latent Variable Model

Introduce $M$ reference (inducing) locations $\mathbf{Z} \in \mathbb{R}^{M \times d}$ with latent variables $\mathbf{u}_i \in \mathbb{R}^M$ for each task $i$:

$$f_i(\mathbf{x}) = \mathbf{w}(\mathbf{x})^\top \mathbf{u}_i + \varepsilon(\mathbf{x})$$

where $\mathbf{w}(\mathbf{x}) = \mathbf{K}_\theta(\mathbf{Z}, \mathbf{Z})^{-1} \mathbf{k}_\theta(\mathbf{Z}, \mathbf{x})$ are kernel interpolation weights from the base kernel $k_\theta$.

The weight matrix for task $i$: $\mathbf{W}_i = \mathbf{K}_\theta(\mathbf{X}_i, \mathbf{Z}) \mathbf{K}_\theta(\mathbf{Z}, \mathbf{Z})^{-1} \in \mathbb{R}^{N_i \times M}$.

Prior on latent variables: $\mathbf{u}_i \sim \mathcal{N}(\boldsymbol{\mu}_u, \mathbf{\Sigma}_u)$ where $\boldsymbol{\mu}_u \in \mathbb{R}^M$ and $\mathbf{\Sigma}_u \in \mathbb{R}^{M \times M}$ are parameters to learn.

### EM Algorithm (Closed-Form)

**Parameters to learn**: $\boldsymbol{\theta} = \{\boldsymbol{\mu}_u, \mathbf{\Sigma}_u, \sigma^2\}$

**E-step** (compute posterior of $\mathbf{u}_i$ given observations):

$$\mathbf{\Lambda}_i = \mathbf{\Sigma}_u^{-1} + \frac{1}{\sigma^2} \mathbf{W}_i^\top \mathbf{W}_i$$

$$\boldsymbol{\mu}_{u|i} = \boldsymbol{\mu}_u + \mathbf{\Lambda}_i^{-1} \mathbf{W}_i^\top (\mathbf{y}_i - \mathbf{W}_i \boldsymbol{\mu}_u) / \sigma^2$$

$$\mathbf{\Sigma}_{u|i} = \mathbf{\Lambda}_i^{-1}$$

**M-step** (update parameters using sufficient statistics aggregated over all $S$ tasks):

$$\hat{\boldsymbol{\mu}}_u = \frac{1}{S} \sum_{i=1}^S \boldsymbol{\mu}_{u|i}$$

$$\hat{\mathbf{\Sigma}}_u = \frac{1}{S} \sum_{i=1}^S \Bigl(\mathbf{\Sigma}_{u|i} + (\boldsymbol{\mu}_{u|i} - \hat{\boldsymbol{\mu}}_u)(\boldsymbol{\mu}_{u|i} - \hat{\boldsymbol{\mu}}_u)^\top\Bigr)$$

$$\hat{\sigma}^2 = \frac{1}{S} \sum_{i=1}^S \frac{\|\mathbf{y}_i - \mathbf{W}_i \boldsymbol{\mu}_{u|i}\|^2 + \text{tr}(\mathbf{W}_i \mathbf{\Sigma}_{u|i} \mathbf{W}_i^\top)}{N_i}$$

**Computational complexity**: $O(S(N_i M^2 + M^3))$ dominated by the $M$-dimensional linear solves.

### EGP Kernel (Continuous Setting)

After EM, the learned prior induces a kernel:

$$k_{\text{EGP}}(\mathbf{x}, \mathbf{x}') = \mathbf{w}(\mathbf{x})^\top \hat{\mathbf{\Sigma}}_u \mathbf{w}(\mathbf{x}') + k_\theta(\mathbf{x}, \mathbf{x}') - \mathbf{w}(\mathbf{x})^\top \mathbf{K}_\theta(\mathbf{Z}, \mathbf{Z}) \mathbf{w}(\mathbf{x}')$$

This decomposes into a structured learned component (via $\hat{\mathbf{\Sigma}}_u$) and a residual base kernel contribution.

## Residual Interpolation for Extrapolation

**Problem**: Outside the range of historical data (extrapolation regime), the EGP reverts to the base kernel $k_\theta$ because the interpolation weights $\mathbf{w}(\mathbf{x}) \to 0$. This causes **variance starvation** — the prior collapses to trivial behavior where it should express uncertainty.

**Solution**: Learn residual functions $\delta_\mu$ and $\delta_\Sigma$ that capture the discrepancy between the empirical statistics and base parametric model. Interpolate these residuals smoothly using a secondary GP.

The corrected kernel for test points $\mathbf{x}^*$:

$$k_{\text{corrected}}(\mathbf{x}^*, \mathbf{x}') = k_{\text{EGP}}(\mathbf{x}^*, \mathbf{x}') + A(\mathbf{x}^*, \mathbf{x}') + B(\mathbf{x}^*, \mathbf{x}')$$

where $A$ is the Schur complement of the base kernel (ensuring $A \succeq 0$) and $B$ is a congruence transformation of the learned covariance (ensuring $B \succeq 0$). Their sum remains positive semi-definite.

> [!IMPORTANT]
> Residual interpolation is essential for time series forecasting tasks where future prediction points lie outside the historical range. Without it, the EGP cannot propagate learned structure into extrapolation regions.

## Comparison with Related Methods

| Method | Prior Learning | Optimization | Handles Heterogeneous Inputs | Extrapolation |
|---|---|---|---|---|
| Standard GP (RBF, Matérn) | Manual kernel design | MLE on single task | Yes (kernel-based) | Poor (high uncertainty) |
| Deep Kernel Learning | Parametric (neural net) | Gradient-based | Yes | Risk of instability |
| Schwaighofer et al. (2004) | Discrete grid empirical | EM | No (fixed grid) | No |
| **Empirical GP (this work)** | **Non-parametric via EM** | **Closed-form EM** | **Yes** | **Via residual interpolation** |

**Differences from meta-learning**: Methods like MAML or Neural Processes optimize within parametric families using gradient descent. EGPs estimate non-parametric statistics without gradient-based optimization, avoiding local minima and instability issues.

**Differences from Bayesian optimization with warm-starting**: Transfer BO methods typically use parametric kernel transfer. EGPs directly encode multi-task covariance structure without committing to a parametric family.

# Experiments

## Dataset: Kernel Recovery

- Handcrafted data: S&P 500 financial time series (used as proxy for geometric Brownian motion) and atmospheric CO₂ measurements (Mauna Loa).
- Task: Recover the covariance structure of a known kernel without providing it explicitly.
- Result: EGP achieved **21% lower RMSE** than an expert-designed composite kernel on CO₂ data; successfully identified long-range dependencies without explicit periodic kernel specification.

## Dataset: GIFT-Eval Time Series Benchmark

- **97 datasets** spanning 7 domains: energy, weather, economics, transportation, retail, cloud ops, and web traffic.
- **144,000 total time series** with varying lengths and frequencies.
- Task: Multi-step ahead forecasting.
- Metric: Weighted quantile loss (WQL), mean absolute scaled error (MASE).
- **Hardware**: Intel Cooper Lake CPUs; total compute < 3,000 CPU hours.
- **Results**: EGP achieved highest rank among all statistical models and outperformed 4 out of 8 deep learning baselines (including N-HiTS, PatchTST, and TimesFM on several domains). Performance improved log-linearly with increased context length.

## Dataset: LCBench Learning Curve Extrapolation

- **35 OpenML classification datasets**.
- **2,000 hyperparameter configurations** each with partial learning curves (epochs 1–51).
- Task: Predict final performance from partial observations (10–40% of epochs seen).
- Baselines: LC-PFN (a pre-trained foundation model for learning curves), power law extrapolation.
- **Results**: EGP consistently outperformed LC-PFN and power law, with the gap widening when fewer epochs are observed (10–20%). Performance improved log-linearly as more historical curves $S$ were added.

> [!NOTE]
> "Empirical GPs occupy a middle ground between the scarce-data and abundant-data regimes. When historical datasets are insufficient, stronger parametric assumptions may be needed. When massive pre-training corpora are available, foundation models may achieve superior expressiveness but risk degradation under distribution shift."

## Implementation

- Built in **BoTorch** (PyTorch-based GP library).
- Code to be released upon publication.
- Optimizer: EM algorithm (no gradient-based optimizer needed for prior learning); standard GP posterior uses Cholesky decomposition.
