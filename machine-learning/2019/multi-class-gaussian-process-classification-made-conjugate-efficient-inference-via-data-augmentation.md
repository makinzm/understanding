# Meta Information

- URL: [Multi-Class Gaussian Process Classification Made Conjugate: Efficient Inference via Data Augmentation](https://arxiv.org/abs/1905.09670)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Galy-Fajou, T., Wenzel, F., Donner, C., & Opper, M. (2019). Multi-Class Gaussian Process Classification Made Conjugate: Efficient Inference via Data Augmentation. UAI 2019.

# Overview

Multi-class Gaussian Process (GP) classification has historically been difficult because the standard softmax likelihood is non-conjugate to the GP prior, forcing expensive approximate inference (EP or non-conjugate VI with numerical quadrature). This paper introduces the **logistic-softmax likelihood** and a three-step data augmentation scheme that together render the model conditionally conjugate, enabling closed-form block coordinate ascent variational inference (CAVI) without any numerical integration or sampling.

**Who benefits**: Practitioners needing calibrated probabilistic multi-class classifiers at scale — e.g., on datasets with 10k–500k points and up to 10 classes — who want both accurate predictions and well-calibrated uncertainty estimates faster than EP-based methods.

**When applicable**: Classification tasks where the number of classes $C$ is moderate (2–10), and where inducing-point sparse GP approximations are acceptable.

# Background: Gaussian Process Classification

A GP prior is placed independently on each class-specific latent function $f^c$:

$$f^c \sim \mathcal{GP}(0, k^c(\mathbf{x}, \mathbf{x}'))$$

where $k^c$ is a covariance kernel (e.g., RBF). For $N$ data points and $C$ classes, the latent matrix is $\mathbf{F} \in \mathbb{R}^{N \times C}$, with row $\mathbf{f}_i = [f_i^1, \dots, f_i^C]^\top \in \mathbb{R}^C$ for observation $i$.

## Standard Likelihood Choices and Their Problems

| Likelihood | Form | Problem |
|---|---|---|
| Softmax (multinomial probit) | $\text{softmax}(\mathbf{f}_i)_k$ | Non-conjugate; requires quadrature or sampling |
| Robust-max | $\approx$ hard max | Sharp, poorly calibrated uncertainty |
| Heaviside (via EP) | Pairwise binary GPs | $O(C^2)$ GPs; very slow at scale |

The proposed **logistic-softmax** likelihood fixes these issues.

# Logistic-Softmax Likelihood

## Definition

$$p(y_i = k \mid \mathbf{f}_i) = \frac{\sigma(f_i^k)}{\sum_{c=1}^{C} \sigma(f_i^c)}$$

where $\sigma(z) = (1 + e^{-z})^{-1}$ is the logistic function. This is equivalent to $\text{softmax}(\log \sigma(\mathbf{f}_i))_k$, i.e., applying softmax to the log-sigmoid of the latent values.

> [!NOTE]
> Unlike the standard softmax $e^{f_i^k}/\sum_c e^{f_i^c}$, replacing exponentials with logistic sigmoids removes the non-integrability bottleneck. The denominator $\sum_c \sigma(f_i^c)$ is a sum of bounded functions, which the augmentation can decouple.

**Input**: latent vector $\mathbf{f}_i \in \mathbb{R}^C$
**Output**: probability vector in $\Delta^{C-1}$ (probability simplex)

# Three-Step Data Augmentation

The key insight is to introduce three layers of auxiliary variables to progressively "conjugatize" the likelihood. After augmentation, the joint density factors in a Gaussian way with respect to each $f^c$.

## Augmentation 1: Gamma Variables $\lambda_i$

**Goal**: Decouple the normalizer $\sum_c \sigma(f_i^c)$ using the integral identity:

$$\frac{1}{z} = \int_0^\infty e^{-\lambda z} d\lambda$$

**Augmented likelihood** (introducing $\lambda_i \in \mathbb{R}_{>0}$ with improper prior $p(\lambda_i) \propto \mathbf{1}_{(0,\infty)}(\lambda_i)$):

$$p(y_i = k \mid \mathbf{f}_i, \lambda_i) = \sigma(f_i^k) \prod_{c=1}^C \exp(-\lambda_i \sigma(f_i^c))$$

**Complete conditional**: $p(\lambda_i \mid \mathbf{n}_i) = \text{Gamma}(1 + \sum_c n_i^c,\ C)$ (after Augmentation 2).

## Augmentation 2: Poisson Variables $n_i^c$

**Goal**: Rewrite each exponential-sigmoid factor using the Poisson moment generating function:

$$e^{-\lambda_i \sigma(f_i^c)} = \sum_{n=0}^\infty \text{Po}(n \mid \lambda_i \sigma(f_i^c)) \cdot (\sigma(-f_i^c))^n$$

**Augmented likelihood** (introducing $n_i^c \in \mathbb{Z}_{\geq 0}$ with prior $p(n_i^c \mid \lambda_i) = \text{Po}(n_i^c \mid \lambda_i)$):

$$p(y_i = k \mid \mathbf{f}_i, \lambda_i, \mathbf{n}_i) = \sigma(f_i^k) \prod_{c=1}^C \sigma(-f_i^c)^{n_i^c}$$

**Complete conditional**: $p(n_i^c \mid f_i^c, \lambda_i) = \text{Po}(n_i^c \mid \lambda_i \sigma(f_i^c))$.

## Augmentation 3: Pólya-Gamma Variables $\omega_i^c$

**Goal**: Remove the remaining sigmoid factors using the Pólya-Gamma (PG) identity:

$$\sigma(z)^a \propto \int_0^\infty e^{az/2 - z^2 \omega/2} \text{PG}(\omega \mid a, 0)\, d\omega$$

Define $y_i^{\prime c} = \mathbf{1}[y_i = c]$ (one-hot). After applying PG to the factor for class $c$:

**Augmented likelihood** (introducing $\omega_i^c \in \mathbb{R}_{>0}$ per class):

$$p(y_i=k \mid \mathbf{f}_i, \lambda_i, \mathbf{n}_i, \boldsymbol{\omega}_i) \propto \prod_{c=1}^C \exp\!\left(\frac{y_i^{\prime c} - n_i^c}{2} f_i^c - \frac{(f_i^c)^2 \omega_i^c}{2}\right)$$

This is **Gaussian in each** $f_i^c$, achieving conjugacy.

**Complete conditional**: $p(\omega_i^c \mid n_i^c, f_i^c, y_i) = \text{PG}(\omega_i^c \mid y_i^{\prime c} + n_i^c,\ |f_i^c|)$.

# Variational Inference

## Sparse GP Approximation

To scale to large $N$, $M \ll N$ inducing points $\mathbf{Z}^c \in \mathbb{R}^{M \times d}$ are introduced per class. The inducing variables $\mathbf{u}^c = f^c(\mathbf{Z}^c) \in \mathbb{R}^M$ give the sparse approximation via the Titsias framework.

The variational distribution over inducing variables is:

$$q(\mathbf{u}^c) = \mathcal{N}(\mathbf{u}^c \mid \boldsymbol{\mu}^c, \boldsymbol{\Sigma}^c), \quad \boldsymbol{\mu}^c \in \mathbb{R}^M, \boldsymbol{\Sigma}^c \in \mathbb{R}^{M \times M}$$

## Full Variational Posterior

$$q(\mathbf{u}, \boldsymbol{\lambda}, \mathbf{n}, \boldsymbol{\omega}) = \prod_c q(\mathbf{u}^c) \cdot \prod_i q(\lambda_i) \cdot \prod_{i,c} q(n_i^c) q(\omega_i^c)$$

Each factor has a known parametric form (Gamma, Poisson, PG, Gaussian), enabling closed-form CAVI updates.

## CAVI Update Equations

Let $\boldsymbol{\kappa}^c = \mathbf{K}_{nm}^c (\mathbf{K}_{mm}^c)^{-1} \in \mathbb{R}^{N \times M}$ be the inter-domain covariance ratio.

**For $q(\mathbf{u}^c)$** — Gaussian update:

$$\boldsymbol{\Sigma}^c = \left(\boldsymbol{\kappa}^{c\top} \mathrm{diag}(\boldsymbol{\theta}^c) \boldsymbol{\kappa}^c + (\mathbf{K}_{mm}^c)^{-1}\right)^{-1}$$

$$\boldsymbol{\mu}^c = \frac{1}{2} \boldsymbol{\Sigma}^c \boldsymbol{\kappa}^{c\top}(\mathbf{y}^{\prime c} - \boldsymbol{\gamma}^c)$$

where $\theta_i^c = \mathbb{E}[\omega_i^c]$ (PG mean) and $\gamma_i^c = \mathbb{E}[n_i^c]$ (Poisson mean).

**For $q(\lambda_i)$** — Gamma update:

$$q(\lambda_i) = \text{Gamma}\!\left(\alpha_i,\ C\right), \quad \alpha_i = 1 + \sum_c \gamma_i^c$$

**For $q(n_i^c)$** — Poisson update (mean $\gamma_i^c$):

$$\gamma_i^c = \mathbb{E}[\lambda_i] \cdot \sigma(-\bar{f}_i^c) \cdot \frac{\exp(-\kappa_i^{c\top} \boldsymbol{\mu}^c / 2)}{\cosh(\bar{f}_i^c / 2)}$$

**For $q(\omega_i^c)$** — PG mean:

$$\mathbb{E}[\omega_i^c] = \frac{y_i^{\prime c} + \gamma_i^c}{2 \bar{f}_i^c} \tanh\!\left(\frac{\bar{f}_i^c}{2}\right)$$

> [!IMPORTANT]
> The negative ELBO is **convex** in the global parameters $(\boldsymbol{\mu}^c, \boldsymbol{\Sigma}^c)$ for fixed auxiliary variables. This guarantees that CAVI converges to a global optimum (of the ELBO) with respect to variational Gaussian parameters — unlike general VI with non-conjugate likelihoods.

## Stochastic Variational Inference (SVI) Algorithm

```
Input:  Data (X, y), minibatch size |S|, inducing points Z^c (M per class),
        step-size schedule {ρ_t}
Output: Variational parameters (μ^c, Σ^c) for c = 1, ..., C

Initialize μ^c ← 0, Σ^c ← K_mm^c  for all c
for t = 1, 2, ... do
  Sample minibatch S ⊂ {1, ..., N}
  for each i ∈ S do
    Compute f̄_i^c = κ_i^c ⊤ μ^c  (predictive mean per class)
    Update γ_i^c ← E[n_i^c]   (Poisson means via PG moments)
    Update θ_i^c ← E[ω_i^c]   (PG means)
  end for
  for each class c do
    Compute natural gradient estimates μ̂^c, Σ̂^c using minibatch S
    μ^c ← (1 - ρ_t) μ^c + ρ_t μ̂^c
    Σ^c ← (1 - ρ_t) Σ^c + ρ_t Σ̂^c
  end for
  Update kernel hyperparameters h via gradient ascent on ELBO
end for
```

**Complexity per iteration**: $O(C M^3)$ for the inducing point covariance, $O(N M)$ for the cross-covariance products (or $O(|S| M)$ per minibatch step).

# Comparison with Related Methods

| Method | Likelihood | Inference | Conjugate? | Complexity | Calibration |
|---|---|---|---|---|---|
| EP (Softmax) | Softmax | EP | No | $O(CN^3)$ | Good |
| EP (Heaviside, pairwise) | Heaviside | EP | Approx. | $O(C^2 M^3)$ | Good |
| SVI (Robust-max) | Robust-max (approx. softmax) | Natural gradient VI | No | $O(CM^3)$ | Poor |
| **This work** (Logistic-Softmax) | Logistic-softmax | CAVI/SVI | **Yes** | $O(CM^3)$ | **Good** |

> [!NOTE]
> "Conjugate" here means that after augmentation, all full conditionals are in the exponential family with closed-form updates — no numerical integration required.

The robust-max likelihood replaces the softmax denominator with the maximum of the remaining classes, producing overconfident (near-0/1) predictions on overlapping classes. The logistic-softmax avoids this, yielding calibrated predictive entropy similar to softmax-based methods but with 10× faster inference than natural gradient SVI for robust-max.

# Experiments

- **Datasets**:
  - *Combined* (Binary): 98,528 points, 50 features, 3 classes
  - *Shuttle*: 58,000 points, 9 features, 7 classes
  - *CovType*: 581,012 points, 54 features, 7 classes
  - *Fashion-MNIST*: 70,000 points (60k train / 10k test), 784 features, 10 classes
  - *MNIST*: 70,000 points (60k train / 10k test), 784 features, 10 classes

- **Inducing points**: $M = 100$ or $M = 200$ depending on dataset

- **Baselines**:
  - EP with pairwise Heaviside likelihood (GPy)
  - Natural gradient SVI with robust-max likelihood

- **Results** (key findings):
  - 1–2 orders of magnitude faster wall-clock convergence than EP-based methods
  - ~10× faster than natural gradient SVI (robust-max) while achieving better-calibrated uncertainty
  - Competitive or superior test accuracy; markedly better negative log-likelihood (NLL) vs. robust-max on overlapping-class datasets

- **Hardware**: Not explicitly stated; experiments run on standard workstation

# Implementation

- Julia package: [AugmentedGaussianProcesses.jl](https://github.com/theogf/AugmentedGaussianProcesses.jl)
- The inference loop requires only matrix operations available in standard linear algebra libraries; no specialized quadrature routines needed.
