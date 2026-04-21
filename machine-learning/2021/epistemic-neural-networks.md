# Meta Information

- URL: [Epistemic Neural Networks](https://arxiv.org/abs/2107.08924)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Osband, I., Wen, Z., Asghari, S. M., Dwaracherla, V., Ibrahimi, M., Lu, X., & Van Roy, B. (2021). Epistemic Neural Networks. arXiv:2107.08924.

# Epistemic Neural Networks

## Overview

Epistemic Neural Networks (ENNs) are a framework for neural networks that produce **joint predictions** — probability distributions over sequences of outputs that correctly capture dependencies across multiple inputs. This is critical for distinguishing **epistemic uncertainty** (reducible via more data) from **aleatoric uncertainty** (irreducible noise). The paper also introduces the **epinet**, a lightweight add-on module that endows any pretrained network with high-quality joint predictions at roughly 2× the cost of a single forward pass, dramatically outperforming deep ensembles of 100 particles.

**Who uses this**: ML practitioners and researchers working on active learning, reinforcement learning, Bayesian optimization, and any decision-making system where knowing *why* a model is uncertain (data limitation vs. inherent noise) improves downstream performance.

**When**: ENNs are applicable whenever a model must make sequential decisions under uncertainty, or whenever confidence calibration across correlated inputs matters — e.g., exploration in RL, recommender systems, or scientific experiment design.

## Problem Statement: Why Marginal Predictions Are Insufficient

A conventional neural network $f_\theta: \mathcal{X} \to \mathbb{R}^C$ produces a marginal class distribution for each input independently. Given a sequence of inputs $x_{1:\tau} = (x_1, \ldots, x_\tau)$ and labels $y_{1:\tau}$, the factored (marginal) prediction is:

```math
\begin{align}
\hat{P}^{1:\tau}_{\text{NN}}(y_{1:\tau}) = \prod_{t=1}^{\tau} \text{softmax}(f_\theta(x_t))_{y_t}
\end{align}
```

This treats each output as conditionally independent given the parameters. Consider a binary classifier outputting 0.5 for two different inputs: one may reflect aleatoric noise (the label is genuinely random), while the other may reflect epistemic uncertainty (the model lacks training data near that region). The marginal prediction 0.5 is identical in both cases and cannot distinguish them.

> [!NOTE]
> **Theorem 1 (informal):** There exist learning problems where minimizing marginal log-loss leads to perfect marginal predictions but decision-making performance is worse than random chance. Marginal accuracy alone is not sufficient to guarantee effective sequential decision-making.

## Framework: Epistemic Neural Networks

### Definition

An ENN is a pair $(f, P_Z)$ where:

- $f_\theta: \mathcal{X} \times \mathcal{Z} \to \mathbb{R}^C$ is a parameterized function mapping input $x$ and epistemic index $z$ to logits
- $P_Z$ is a reference distribution over the epistemic index space $\mathcal{Z}$

The joint prediction over a sequence of $\tau$ inputs is formed by marginalizing over $z$:

```math
\begin{align}
\hat{P}^{1:\tau}_{\text{ENN}}(y_{1:\tau}) = \int_z P_Z(dz) \prod_{t=1}^{\tau} \text{softmax}(f_\theta(x_t, z))_{y_t}
\end{align}
```

The key insight: sampling a single $z$ and computing predictions for all inputs in the sequence creates **statistical dependencies** across outputs. If the model is uncertain in a given region of input space, the same $z$ will cause correlated errors across nearby inputs — precisely the behavior of epistemic uncertainty.

> [!IMPORTANT]
> **Theorem 2 (informal):** If an ENN achieves small expected joint log-loss, then its actions attain near-optimal expected reward for any decision problem whose quality is bounded by KL-divergence from the true posterior. Small joint log-loss is both necessary and sufficient for effective decision-making under uncertainty.

### Relation to Bayesian Neural Networks

> [!NOTE]
> **Theorem 3:** All Bayesian Neural Networks (BNNs) can be expressed as ENNs, but not all ENNs can be expressed as BNNs. The ENN framework strictly generalizes Bayesian inference — it provides a computationally tractable path to Bayesian-quality uncertainty without requiring a proper posterior.

## Epinet Architecture

The epinet is a lightweight network added on top of a base network that implements the ENN framework efficiently.

### Components

**Full ENN output** (input $x \in \mathbb{R}^d$, epistemic index $z \in \mathbb{R}^{D_Z}$, output $\in \mathbb{R}^C$):

```math
\begin{align}
f_\theta(x, z) = \mu_\zeta(x) + \sigma_\eta(\text{sg}[\phi_\zeta(x)],\, z)
\end{align}
```

| Symbol | Meaning |
|--------|---------|
| $\mu_\zeta(x) \in \mathbb{R}^C$ | Base network output (standard class logits) |
| $\phi_\zeta(x) \in \mathbb{R}^{d'}$ | Intermediate features from base network (e.g., last hidden layer) |
| $\text{sg}[\cdot]$ | Stop-gradient: gradients do not flow into the base network via this path |
| $\sigma_\eta(\cdot, \cdot) \in \mathbb{R}^C$ | Epinet output (epistemic correction) |
| $z \sim \mathcal{N}(0, I_{D_Z})$ | Epistemic index; $D_Z = 8$ (testbed), $30$ (ImageNet) |

**Epinet decomposition** into learnable and fixed prior components:

```math
\begin{align}
\sigma_\eta(\tilde{x}, z) = \sigma_\eta^L(\tilde{x}, z) + \sigma^P(\tilde{x}, z)
\end{align}
```

- $\sigma_\eta^L$: learnable component trained via SGD
- $\sigma^P$: prior network with fixed random weights (provides exploration diversity)

**Learnable epinet** (output $\in \mathbb{R}^C$):

```math
\begin{align}
\sigma_\eta^L(\tilde{x}, z) := \text{mlp}_\eta([\tilde{x}; z])^\top z
\end{align}
```

where $\text{mlp}_\eta$ maps $[\tilde{x}; z] \in \mathbb{R}^{d' + D_Z}$ to $\mathbb{R}^{D_Z \times C}$, and the inner product with $z \in \mathbb{R}^{D_Z}$ yields output in $\mathbb{R}^C$.

### Computational Cost Comparison

| Method | Parameter count (relative) | Joint log-loss quality |
|--------|--------------------------|----------------------|
| Deep Ensemble (100 particles) | 100× single model | Good |
| Epinet | < 2× single model | Better than 100-particle ensemble |
| Dropout / SNGP / MIMO | 1–2× single model | Worse than ensemble |

## Training Algorithm

**Algorithm 1: ENN Training via SGD**

```
Input: Dataset D = {(x_i, y_i)}_{i=1}^N, ENN (f, P_Z), initial params θ_0, steps T
Output: Trained parameters θ_T

for t = 0, ..., T-1:
    Sample data minibatch Ĩ ~ Uniform({1,...,N})
    Sample epistemic indices Z̃ ~ P_Z  (|Z̃| = D_Z)
    Compute gradient:
        grad ← ∇_θ|_{θ=θ_t} Σ_{z ∈ Z̃} Σ_{i ∈ Ĩ} ℓ(θ, z, x_i, y_i)
    Update: θ_{t+1} ← Adam(θ_t, grad)
```

### Loss Functions

**Standard cross-entropy with L2 regularization** ($\lambda > 0$):

```math
\begin{align}
\ell_\lambda^{\text{XENT}}(\theta, z, x_i, y_i) := -\ln\bigl(\text{softmax}(f_\theta(x_i, z))_{y_i}\bigr) + \lambda \|\theta\|_2^2
\end{align}
```

**Bootstrap-perturbed variant** (controls signal-to-noise sensitivity, $p \in (0,1)$):

```math
\begin{align}
\ell_{p,\lambda}^{\text{XENT}}(\theta, z, x_i, y_i) :=
\begin{cases}
\ell_\lambda^{\text{XENT}}(\theta, z, x_i, y_i) & \text{if } c_i^\top z > \Phi^{-1}(p) \\
0 & \text{otherwise}
\end{cases}
\end{align}
```

where $c_i \in \mathbb{R}^{D_Z}$ is sampled uniformly from the unit sphere per data point, and $\Phi^{-1}(p)$ is the $p$-th quantile of the standard normal. This randomly masks training signal per $(z, i)$ pair, mimicking Bayesian bootstrap resampling.

### Convergence Guarantee

> [!NOTE]
> **Theorem 4:** For linear-Gaussian regression with an epinet trained using Gaussian-bootstrapped loss, the predictive distribution converges in distribution to the exact Bayesian posterior as $D_Z \to \infty$. This validates the epinet's statistical soundness on the class of problems where Bayesian inference is tractable.

## Comparison with Similar Approaches

| Method | Joint predictions | Computational cost | Notes |
|--------|-----------------|-------------------|-------|
| Deep Ensembles | ✓ (via ensemble spread) | $K \times$ base model | $K = 100$ needed for good joints |
| Bayesian NNs (VI) | ✓ (via weight posterior) | 2–3× base model | Approximate; often underestimates uncertainty |
| MC Dropout | Partial (marginals only effectively) | 1× base model | Poor calibration on joint predictions |
| SNGP | Marginals only | 1× base model | Gaussian process output layer, no epistemic index |
| MIMO | Partial | 1× base model | Multiple heads, but no explicit epistemic index |
| **Epinet (ENN)** | ✓ (explicit joint via $z$) | < 2× base model | Asymptotically Bayes-optimal for linear-Gaussian |

The fundamental distinction between ENNs and all prior methods (except ensembles): the epistemic index $z$ creates **explicit stochastic dependence** across predictions for different inputs, which is the mechanism required for joint predictions to detect epistemic uncertainty.

## Experiments

- **Datasets**:
  - *Neural Testbed*: Synthetic binary classification problems generated from random 2-layer MLPs (width 50). Input dimension $D \in \{2, 10, 100\}$; training samples $T = \lambda D$ where $\lambda \in \{1, 10, 100, 1000\}$; temperature $\rho \in \{0.01, 0.1, 0.5\}$; 5 random seeds per setting.
  - *ImageNet*: Standard 1000-class image classification. Evaluated on ResNet-50/101/152/200 with frozen pretrained weights; epinet attached to last hidden layer features ($d' = 2048$ for ResNet-50).
  - *CIFAR-10 / CIFAR-100*: Image classification benchmarks (results in appendices; qualitatively similar to ImageNet).
- **Hardware**: JAX / Jaxline framework on TPUs (unspecified configuration)
- **Optimizer**: Adam, learning rate $10^{-3}$, L2 decay $\lambda$ tuned per setting
- **Evaluation**: Joint log-loss via **dyadic sampling** — draw $\tau = 10$ points from pairs of anchor inputs; compute $-\ln \hat{P}^{1:\tau}(y_{1:\tau})$ averaged over seeds
- **Results**:
  - On the Neural Testbed, the epinet achieves lower joint log-loss than a 100-particle deep ensemble across all $(D, \lambda, \rho)$ settings, while using fewer parameters than 2 base particles.
  - On ImageNet (ResNet-50), the epinet reduces joint log-loss by a large margin compared to the 100-particle ensemble, with negligible change to marginal log-loss.
  - Baselines (Dropout, SNGP, MIMO, BBB, Hypermodel) all produce substantially worse joint log-loss than ensembles; epinet is the only lightweight method to exceed ensemble quality.
