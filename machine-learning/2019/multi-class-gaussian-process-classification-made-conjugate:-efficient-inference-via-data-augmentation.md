# Meta Information

- URL: [Multi-Class Gaussian Process Classification Made Conjugate](https://arxiv.org/abs/1905.09670)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Galy-Fajou, T., Wenzel, F., Donner, C., & Opper, M. (2019). Multi-Class Gaussian Process Classification Made Conjugate. *Proceedings of the 35th Conference on Uncertainty in Artificial Intelligence (UAI 2019)*.
- Code: [theogf/AugmentedGaussianProcesses.jl](https://github.com/theogf/AugmentedGaussianProcesses.jl)

---

# Overview

Multi-class Gaussian Process (GP) classification has historically relied on the softmax likelihood, which couples the $C$ per-class latent functions inside a normalizer that has no closed-form expectation under a Gaussian posterior. Existing approaches either fall back on numerical quadrature/sampling (non-conjugate variational inference) or on expectation propagation (EP) with $O(C^2)$ pairwise binary GPs, both of which are slow at the data sizes (tens of thousands to hundreds of thousands of points) typical of modern classification tasks.

This paper proposes the **logistic-softmax likelihood**, a drop-in replacement for the softmax that substitutes the logistic (sigmoid) function for the exponential inside the normalizer. Combined with a **three-step data augmentation** (Gamma, Poisson, Pólya-Gamma), the resulting joint distribution becomes conditionally conjugate everywhere, which permits a closed-form block coordinate ascent variational inference (CAVI) scheme with no quadrature, no sampling of the ELBO gradient, and a provably convex negative ELBO.

**Who benefits**: Practitioners who need calibrated probabilistic multi-class classifiers on datasets ranging from ~50K to ~600K points with up to ~10 classes, and who want both competitive accuracy and well-calibrated uncertainty estimates at a fraction of the training time of EP-based methods.

**When applicable**: Classification problems where the number of classes $C$ is moderate (the standard algorithm scales as $O(CM^3)$ in the number of inducing points $M$), and where sparse GP approximations via inducing points are an acceptable trade-off for scalability. An "extreme classification" variant that subsamples classes per iteration removes the $C$ dependence from the per-step cost, extending applicability to larger label spaces.

---

# Background: Gaussian Process Classification

A GP prior is placed independently on each of $C$ class-specific latent functions:

```math
\begin{align}
  f^c \sim \mathcal{GP}(0, k^c(\mathbf{x}, \mathbf{x}'))
\end{align}
```

where $k^c$ is a covariance kernel (e.g., RBF) for class $c \in \{1, \dots, C\}$. For $N$ observations, the latent matrix is $\mathbf{F} \in \mathbb{R}^{N \times C}$, with row $\mathbf{f}_i = [f_i^1, \dots, f_i^C]^\top \in \mathbb{R}^C$ holding the $C$ latent values for observation $i$, and label $y_i \in \{1, \dots, C\}$.

## Why the Standard Softmax Is Hard

The multinomial (softmax) likelihood is:

```math
\begin{align}
  p(y_i = k \mid \mathbf{f}_i) = \frac{\exp(f_i^k)}{\sum_{c=1}^{C} \exp(f_i^c)}
\end{align}
```

Because the exponential normalizer couples all $C$ latents multiplicatively, $\mathbb{E}_{q(\mathbf{f}_i)}[\log p(y_i \mid \mathbf{f}_i)]$ has no closed form under a Gaussian $q$, forcing either numerical integration (expensive per point) or sampling (high-variance gradients). Two existing alternatives and their drawbacks:

| Likelihood | Form | Problem |
|---|---|---|
| Softmax (multinomial) | $\mathrm{softmax}(\mathbf{f}_i)_k$ | Non-conjugate; needs quadrature/sampling |
| Robust-max | Near-hard-max with uniform noise | Poorly calibrated, overconfident uncertainty |
| Heaviside (via EP) | Pairwise binary comparisons | $O(C^2)$ GPs; very slow at scale |

---

# The Logistic-Softmax Likelihood

The paper's central idea is to replace the exponential in the softmax with the logistic function $\sigma(z) = (1 + \exp(-z))^{-1}$:

```math
\begin{align}
  p(y_i = k \mid \mathbf{f}_i) = \frac{\sigma(f_i^k)}{\sum_{c=1}^{C} \sigma(f_i^c)}
\end{align}
```

This preserves the qualitative behavior of softmax (larger $f_i^k$ relative to competitors increases $p(y_i=k)$) while making each factor a bounded logistic term amenable to the Pólya-Gamma augmentation scheme used widely for logistic (binary) GP classification. Unlike robust-max, which saturates to an (almost) deterministic decision and is over-confident, the logistic-softmax retains graded, well-calibrated probability mass between classes.

---

# Three-Step Data Augmentation

The goal is to introduce auxiliary latent variables so that, conditional on them, every remaining factor is Gaussian-conjugate in $\mathbf{f}_i$. This is done in three successive augmentation steps.

## Step 1: Gamma Augmentation (removes the sum-of-terms normalizer)

Using the integral identity $\frac{1}{z} = \int_0^\infty \exp(-\lambda z) \, d\lambda$ applied to $z = \sum_c \sigma(f_i^c)$, an auxiliary variable $\lambda_i > 0$ is introduced:

```math
\begin{align}
  p(y_i = k \mid \mathbf{f}_i, \lambda_i) = \sigma(f_i^k) \prod_{c=1}^{C} \exp\bigl(-\lambda_i \, \sigma(f_i^c)\bigr)
\end{align}
```

Marginalizing $\lambda_i$ (with an implicit improper prior consistent with the integral identity) recovers the original likelihood. This turns the *ratio* into a *product* of per-class factors, at the cost of one new scalar variable per data point.

## Step 2: Poisson Augmentation (linearizes the exponential-of-sigmoid factors)

Each factor $\exp(-\lambda_i \sigma(f_i^c))$ is itself still non-conjugate in $f_i^c$. Using the Poisson probability generating function, a count variable $n_{ic} \in \mathbb{N}_0$ is introduced per (data point, class) pair such that:

```math
\begin{align}
  p(y_i = k \mid \mathbf{f}_i, \lambda_i, \mathbf{n}_i) = \sigma(f_i^k) \prod_{c=1}^{C} \bigl(\sigma(-f_i^c)\bigr)^{n_{ic}}, \qquad
  p(n_{ic} \mid \lambda_i) = \mathrm{Po}(n_{ic} \mid \lambda_i)
\end{align}
```

Summing over $n_{ic} = 0, 1, 2, \dots$ recovers the Step-1 expression $\exp(-\lambda_i \sigma(f_i^c))$. Now every factor is a *power* of a logistic function, which is the canonical form the Pólya-Gamma trick applies to.

## Step 3: Pólya-Gamma Augmentation (makes the model Gaussian-conjugate)

Using the Pólya-Gamma integral identity for powers of the logistic function,

```math
\begin{align}
  \sigma(z)^n = 2^{-n} \exp\!\left(\frac{nz}{2} - \frac{z^2 \omega}{2}\right) \mathrm{PG}(\omega \mid n, 0), \qquad \text{integrated over } \omega > 0,
\end{align}
```

an auxiliary Pólya-Gamma variable $\omega_{ic}$ is introduced for every (data point, class) pair, yielding a likelihood that is **linear-quadratic in $f_i^c$**, i.e. of the same functional form as a Gaussian:

```math
\begin{align}
  p(y_i = k \mid \mathbf{f}_i, \lambda_i, \mathbf{n}_i, \boldsymbol{\omega}_i) = \prod_{c=1}^{C} 2^{-(y'_{ic} + n_{ic})} \exp\!\left(\frac{(y'_{ic} - n_{ic}) f_i^c}{2} - \frac{(f_i^c)^2 \, \omega_{ic}}{2}\right)
\end{align}
```

where $y'_{ic} = \mathbb{1}[y_i = c]$ is the one-hot indicator of the true class. Because the exponent is quadratic in $f_i^c$ with a Gaussian-compatible sign, the complete conditional of $\mathbf{f}^c$ given all auxiliary variables and data is Gaussian — the model is now **conditionally conjugate**.

> [!IMPORTANT]
> The three augmentations are applied in sequence specifically because each removes one obstacle to conjugacy: Gamma removes the shared-normalizer coupling across classes, Poisson turns an exponential-of-sigmoid into a power-of-sigmoid, and Pólya-Gamma turns that power-of-sigmoid into a Gaussian-form exponential quadratic in $f_i^c$. Skipping any step leaves a non-conjugate factor.

---

# Complete Conditional Distributions

After augmentation, every variable in the model has a closed-form complete conditional, enabling exact Gibbs sampling or coordinate ascent variational inference:

```math
\begin{align}
  p(\mathbf{f}^c \mid \mathbf{y}, \boldsymbol{\omega}^c, \mathbf{n}^c) &= \mathcal{N}\!\left(\mathbf{f}^c \,\middle|\, \tfrac{1}{2} A^c (\mathbf{y}'^c - \mathbf{n}^c),\; A^c\right), \qquad A^c = \bigl(\mathrm{diag}(\boldsymbol{\omega}^c) + (K^c)^{-1}\bigr)^{-1} \\
  p(\lambda_i \mid \mathbf{n}_i) &= \mathrm{Ga}\!\left(\lambda_i \,\middle|\, 1 + \sum_{c} n_{ic},\; C\right) \\
  p(n_{ic} \mid f_i^c, \lambda_i) &= \mathrm{Po}\bigl(n_{ic} \mid \lambda_i \, \sigma(f_i^c)\bigr) \\
  p(\omega_{ic} \mid n_{ic}, f_i^c, y_i) &= \mathrm{PG}\bigl(\omega_{ic} \mid y'_{ic} + n_{ic},\; |f_i^c|\bigr)
\end{align}
```

Here $\mathbf{y}'^c \in \{0,1\}^N$ is the one-hot indicator vector for class $c$ across all $N$ points, $\mathbf{n}^c \in \mathbb{N}_0^N$ collects the Poisson counts for class $c$, and $K^c \in \mathbb{R}^{N \times N}$ is the GP prior covariance (kernel Gram matrix) for class $c$.

---

# Scalable Variational Inference

## Sparse GP Approximation

To scale beyond a handful of thousand points, each class-specific GP is approximated with $M \ll N$ inducing points $\mathbf{Z}^c$ and inducing values $\mathbf{u}^c \in \mathbb{R}^M$, with $K_{mm}^c \in \mathbb{R}^{M \times M}$ the inducing-point kernel matrix and $\kappa^c \in \mathbb{R}^{N \times M}$ the cross-covariance (data-to-inducing) matrix scaled by $(K_{mm}^c)^{-1}$.

## Structured Mean-Field Variational Family

The variational posterior factorizes as:

```math
\begin{align}
  q(\mathbf{u}, \boldsymbol{\lambda}, \mathbf{n}, \boldsymbol{\omega}) = q(\mathbf{u}) \, q(\boldsymbol{\lambda}) \, q(\boldsymbol{\omega}, \mathbf{n})
\end{align}
```

with the per-factor forms:

```math
\begin{align}
  q(\mathbf{u}) &= \prod_{c=1}^{C} \mathcal{N}(\mathbf{u}^c \mid \boldsymbol{\mu}^c, \Sigma^c) \\
  q(\boldsymbol{\lambda}) &= \prod_{i=1}^{N} \mathrm{Ga}(\lambda_i \mid \alpha_i, \beta_i) \\
  q(\boldsymbol{\omega}, \mathbf{n}) &= \prod_{i=1}^{N} \prod_{c=1}^{C} \mathrm{PG}(\omega_{ic} \mid y'_{ic} + n_{ic}, b_{ic}) \, \mathrm{Po}(n_{ic} \mid \gamma_{ic})
\end{align}
```

where $\boldsymbol{\mu}^c \in \mathbb{R}^M$, $\Sigma^c \in \mathbb{R}^{M \times M}$ parameterize the inducing-point posterior for class $c$, and $\alpha_i, \beta_i, b_{ic}, \gamma_{ic}$ are the local (per-data-point) variational parameters.

## Closed-Form Coordinate Ascent Updates

Because the augmented model is fully conjugate, each factor's optimal update (maximizing the ELBO holding the others fixed) has a closed form, computed in the following order per iteration:

```math
\begin{align}
  \gamma_{ic} &= \frac{\exp(\psi(\alpha_i)) \exp(-\kappa_{i}^c {}^\top \boldsymbol{\mu}^c / 2)}{\beta_i \cosh(\bar f_i^c / 2)} \\
  \alpha_i &= 1 + \sum_{c} \gamma_{ic}, \qquad \beta_i = C \\
  \theta_{ic} &= \mathbb{E}[\omega_{ic}] = \frac{y'_{ic} + \gamma_{ic}}{2 b_{ic}} \tanh(b_{ic} / 2) \\
  \Sigma^c &= \Bigl( (\kappa^c)^\top \mathrm{diag}(\boldsymbol{\theta}^c) \, \kappa^c + (K_{mm}^c)^{-1} \Bigr)^{-1} \\
  \boldsymbol{\mu}^c &= \tfrac{1}{2} \Sigma^c (\kappa^c)^\top (\mathbf{y}'^c - \boldsymbol{\gamma}^c)
\end{align}
```

where $\psi(\cdot)$ is the digamma function and $\bar f_i^c$ is the posterior mean of $f_i^c$ under the current $q(\mathbf{u}^c)$. Every update is a closed-form expression — no sampling or quadrature is used anywhere in the loop, which is the paper's key practical advantage over EP (Heaviside) and non-conjugate VI (robust-max, standard softmax).

> [!IMPORTANT]
> The negative ELBO of this augmented model is **convex** in the variational parameters, so coordinate ascent is guaranteed to converge to the global optimum of the (augmented) variational objective — unlike generic non-conjugate VI, which can get stuck in local optima.

## Training Algorithm

**Input**: Training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$, number of inducing points $M$, minibatch size $|\mathcal{S}|$

**Output**: Variational parameters $\{\boldsymbol{\mu}^c, \Sigma^c\}_{c=1}^C$, kernel hyperparameters $h$

```
1. Initialize inducing locations Z^c (e.g., k-means on inputs), variational parameters
2. Repeat until convergence:
   a. Sample a minibatch S of data points
   b. For each i in S: update local parameters alpha_i, gamma_{ic}, b_{ic} (closed form)
   c. Update global parameters mu^c, Sigma^c for each class c (closed form, using minibatch-scaled sufficient statistics)
   d. Take a natural-gradient / gradient step on kernel hyperparameters h to further increase the ELBO
3. Return {mu^c, Sigma^c}, h
```

## Computational Complexity

Each iteration requires inverting or factorizing $M \times M$ matrices per class: $O(CM^3)$ overall (plus $O(NM^2)$ for the cross terms in a minibatch of size $N$). For **extreme classification** (large $C$), the paper proposes subsampling a subset of classes per iteration in addition to subsampling data points, reducing the per-iteration cost to $O(M^3)$, independent of $C$.

## Prediction

For a test input $\mathbf{x}_\star$, the predictive distribution marginalizes the approximate posterior over each class's latent function:

```math
\begin{align}
  p(y = k \mid \mathbf{x}_\star, \mathbf{y}) \approx \int p(y = k \mid \mathbf{f}_\star) \prod_{c=1}^{C} q(f_\star^c \mid \mathbf{x}_\star) \, d\mathbf{f}_\star
\end{align}
```

which has no closed form (due to the logistic-softmax likelihood in the integrand) and is instead approximated via Monte Carlo sampling of the $C$ independent Gaussian posteriors $q(f_\star^c \mid \mathbf{x}_\star)$.

---

# Experiments

- **Datasets**:
  - Seven synthetic toy sets (500 points, 3 classes, Gaussian-mixture generated with noise variance $\sigma^2 \in [0, 1]$) — used to compare calibration of logistic-softmax vs. softmax, Heaviside, and robust-max.
  - Wine (3 classes, no inducing points — full GP) — used to validate the augmented VI posterior against exact Gibbs sampling.
  - Shuttle (58,000 points, 9 features, 7 classes) — used to study the effect of the number of inducing points and hyperparameter optimization.
  - Combined (98,528 points, 50 features, 3 classes), CovType (581,000 points, 54 features, 7 classes), Fashion-MNIST (70,000 points, 784 features, 10 classes), MNIST (70,000 points, 784 features, 10 classes) — main benchmark suite, all from the LIBSVM dataset repository, features normalized to zero mean and unit variance.
- **Baselines**: Robust-max (Hensman & Matthews, 2015; GPFlow implementation with natural-gradient optimization), Heaviside/EP (Villacampa-Calvo & Hernández-Lobato, 2017).
- **Metrics**: Test classification error, negative log-likelihood (NLL), calibration (reliability-diagram-based expected calibration error), and wall-clock training time.
- **Results**:
  - On the toy datasets, robust-max shows visibly poor calibration and inflated NLL as class overlap increases ($\sigma^2 > 0.5$); logistic-softmax tracks the well-calibrated softmax and Heaviside baselines closely.
  - On Wine, the augmented VI posterior closely matches exact Gibbs-sampling posteriors (final accuracy 0.96 vs. 0.98 for Gibbs), with a slight underestimation of posterior variance.
  - On the main benchmark suite, logistic-softmax reaches comparable or better NLL than the baselines while converging **one to two orders of magnitude faster** than the Heaviside/EP model and **about 10× faster** than robust-max.

---

# Comparison with Related Methods

| Method | Likelihood coupling | Inference | Conjugate? | Per-iteration cost | Calibration |
|---|---|---|---|---|---|
| Softmax + non-conjugate VI | Exponential, all classes | Quadrature/sampling | No | High (per-sample quadrature) | Good |
| Robust-max | Near-deterministic argmax + noise | Non-conjugate VI | No | Moderate | Poor (overconfident) |
| Heaviside (EP) | Pairwise binary comparisons | Expectation propagation | Partially | $O(C^2)$ pairwise GPs | Good |
| **Logistic-softmax (this paper)** | Logistic, all classes | Closed-form CAVI | Yes (after augmentation) | $O(CM^3)$, or $O(M^3)$ with class subsampling | Good |

**Key differences**: Unlike robust-max, the logistic-softmax retains graded probability mass between competing classes, avoiding overconfident predictions. Unlike Heaviside/EP, it avoids the $O(C^2)$ blow-up from pairwise class comparisons, scaling linearly in $C$ (or independent of $C$ with class subsampling). Unlike standard softmax with non-conjugate VI, every update in the augmented model is closed-form, removing the need for quadrature or high-variance sampling-based gradient estimates and yielding the empirically observed order-of-magnitude speedups.

> [!CAUTION]
> The augmentation introduces $O(NC)$ auxiliary local variables ($\lambda_i$, $n_{ic}$, $\omega_{ic}$), so while each update is closed-form and matrix-inversion-free beyond the $M \times M$ inducing covariance, the bookkeeping overhead scales with both $N$ and $C$; the paper's speed comparisons are against Heaviside/EP and robust-max specifically, not against all conceivable inference schemes.

> [!TIP]
> The Pólya-Gamma augmentation technique for logistic likelihoods was originally introduced in Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference for logistic models using Pólya-Gamma latent variables. *Journal of the American Statistical Association*, 108(504), 1339-1349 — useful background for understanding Step 3 above.
