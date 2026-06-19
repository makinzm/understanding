# Meta Information

- URL: [Classification Imbalance as Transfer Learning](https://arxiv.org/abs/2601.10630)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Xia, E., & Klusowski, J. M. (2026). Classification Imbalance as Transfer Learning. arXiv:2601.10630.

# Classification Imbalance as Transfer Learning

## Overview

Classification imbalance occurs when one class (minority) is far rarer than another (majority) in training data, causing standard ERM to produce classifiers biased toward the majority. This paper reframes imbalance as a **transfer learning problem under label shift**: the learner observes an imbalanced source distribution $\mathbb{P}$ (where $\pi_1 = \mathbb{P}(Y=1) \ll \pi_0 = \mathbb{P}(Y=0)$), but wishes to minimize risk under a balanced target distribution $\mathbb{Q}$ where $Y \sim \text{Ber}(1/2)$, with identical conditional distributions $\mathbb{P}(X|Y=y) = \mathbb{Q}(X|Y=y)$.

This framework unifies analysis of rebalancing methods—bootstrapping, SMOTE, kernel density estimation, diffusion-based sampling, and plug-in approaches—under a single excess-risk decomposition.

**Applicability**: Machine learning practitioners and theorists working on imbalanced binary classification problems (e.g., fraud detection, medical diagnosis, rare event prediction) who wish to choose among oversampling strategies with theoretical guarantees.

## Problem Formulation

**Setting**:
- Source distribution: $\mathbb{P}$ with imbalance ratio $\pi_0 \gg \pi_1$ (e.g., $\pi_0 = 0.9$)
- Target distribution: $\mathbb{Q}$ with equal class probabilities ($Y \sim \text{Ber}(1/2)$)
- Both distributions share conditional distributions: $\mathbb{P}(X|Y=y) = \mathbb{Q}(X|Y=y)$
- Training data: $\mathcal{D} = \{(X_i, Y_i)\}_{i=1}^N$ from $\mathbb{P}$, where $N_1 = |\{i: Y_i=1\}|$ minority samples and $N_0 = N - N_1$ majority samples

**Goal**: Find $\hat{f} \in \mathcal{F}$ minimizing excess risk under $\mathbb{Q}$:

$$\text{Excess Risk} = \mathbb{E}_{\mathbb{Q}}[L(Y, \hat{f}(X)) - L(Y, f^*(X))]$$

where $f^* = \arg\min_{f \in \mathcal{F}} \mathbb{E}_{\mathbb{Q}}[L(Y, f(X))]$.

## Algorithm: Synthetic Rebalancing (Algorithm 1)

The general rebalancing procedure is:

```
Input: Dataset D = {(X_i, Y_i)}_{i=1}^N, synthetic count J, procedure P
1. Construct P̂_{X|Y=1} via procedure P applied to minority samples
2. Generate J synthetic minority samples X̃_j ~ P̂_{X|Y=1}
3. Return:
   f̂ ∈ argmin_{f ∈ F} { (1/(N+J)) Σ_{i=1}^N L(Y_i, f(X_i))
                         + (1/(N+J)) Σ_{j=1}^J L(1, f(X̃_j)) }
```

The effective training set mixes the original $N$ samples with $J$ synthetic minority samples. The choice of $J$ controls the degree of rebalancing; setting $J = N_0 - N_1$ achieves numerical balance.

## Key Theorems

### Theorem 1: Excess Risk Decomposition (Slow Rates)

For Algorithm 1 with a $b$-uniformly bounded loss, with probability at least $1 - \delta$:

$$\mathbb{E}_{\mathbb{Q}}[L(Y, \hat{f}(X)) - L(Y, f^*(X))] \leq 12 R_M(\mathbb{Q}) + 4b \cdot d_{\text{TV}}(\hat{\mathbb{P}}_{X|Y=1}, \mathbb{P}_{X|Y=1}) + 28b \left|\frac{N\pi_0}{N+J} - \frac{1}{2}\right| + t_M(\delta)$$

where:
- $R_M(\mathbb{Q})$ = Rademacher complexity of $\mathcal{F}$ under $\mathbb{Q}$ with $M = N + J$ samples (**oracle term**)
- $d_{\text{TV}}(\hat{\mathbb{P}}_{X|Y=1}, \mathbb{P}_{X|Y=1})$ = total variation distance between estimated and true minority distributions (**cost of transfer**)
- $\left|\frac{N\pi_0}{N+J} - \frac{1}{2}\right|$ = class imbalance in the augmented dataset (**mixture bias**)
- $t_M(\delta)$ = $O(b\sqrt{\log(1/\delta)/M})$ tail term

> [!IMPORTANT]
> The decomposition shows that excess risk = (oracle risk with balanced data) + (cost of imperfect synthesis). This cleanly separates the statistical complexity of learning from the quality of the oversampling procedure.

### Theorem 2: Localization / Fast Rates

Under Lipschitz and strongly convex loss assumptions, with probability at least $1 - \delta$:

$$\|\hat{f} - f^*\|_{\mathbb{Q}_X} \leq C_1 \cdot (8\omega_M(\mathbb{Q}_X) + \omega_M(\hat{\mathbb{P}}_{X|Y=1})) + C_2 \cdot \sqrt{\chi^2(\hat{\mathbb{P}}_{X|Y=1}; \mathbb{P}_{X|Y=1})} + s_M(\delta)$$

where:
- $\omega_M(\cdot)$ = critical radius of the function class $\mathcal{F}$ under the given distribution
- $\chi^2(\hat{\mathbb{P}}_{X|Y=1}; \mathbb{P}_{X|Y=1})$ = chi-squared divergence (**transfer cost**, tighter than TV distance under smoothness)
- $C_1, C_2$ = constants depending on strong convexity and Lipschitz parameters

> [!NOTE]
> Fast rates replace total variation distance (Theorem 1) with chi-squared divergence, which can be substantially smaller under regularity conditions. This matches known fast-rate phenomena in empirical risk minimization.

## Concrete Method Comparisons

### Bootstrapping (Random Oversampling)

$\hat{\mathbb{P}}_{X|Y=1}$ = empirical distribution of observed minority samples. Excess risk is bounded by:

$$\left(1 + \sqrt{\frac{N_1}{J} \log(2N_1)}\right)^2 \cdot R_{2N_1}(\mathbb{Q})$$

Effective sample size: $2N_1$ minority samples. **No curse of dimensionality.**

### SMOTE (Synthetic Minority Oversampling Technique)

For each minority sample, SMOTE generates synthetic points by interpolating between a sample and one of its $k$ nearest neighbors in feature space. The additional cost of transfer term is:

$$O\left(\frac{LD\left(6\left(\frac{k}{N_1}\right)^{1/d} + k \cdot 5^d + 1\right)}{\sqrt{N_1}}\right)$$

where $d$ = feature dimension, $k$ = number of nearest neighbors, $L$ = Lipschitz constant of loss, $D$ = bound on $\|X\|_2$.

> [!IMPORTANT]
> The term $(k/N_1)^{1/d}$ introduces a **curse of dimensionality**: as $d$ increases, SMOTE's error grows faster than bootstrapping's. The ratio of SMOTE vs. bootstrapping excess risk scales approximately as $N_1^{1/2 - 1/d}$, confirming SMOTE's advantage only in very low dimensions.

**Requirement**: SMOTE analysis requires globally Lipschitz loss, which excludes tree-based models and some neural networks.

### Kernel Density Estimation (KDE)

For a $\beta$-smooth minority density, KDE achieves:

$$\mathbb{E}[d_{\text{TV}}(\hat{\mathbb{P}}_{X|Y=1}, \mathbb{P}_{X|Y=1})] \lesssim N_1^{-\beta/(2\beta+d)}$$

Minimax-optimal TV rate; still suffers from curse of dimensionality for large $d$.

### Diffusion-Based Sampling

Using score-based generative models to estimate $\mathbb{P}_{X|Y=1}$:

$$d_{\text{TV}}(\hat{\mathbb{P}}_{X|Y=1}, \mathbb{P}_{X|Y=1}) \lesssim \frac{d \log^3 T}{T} + \Delta_{\text{score}} \cdot \sqrt{\log T}$$

where $T$ = number of diffusion steps and $\Delta_{\text{score}}$ = score function estimation error. Effective when score estimates are accurate and the minority class has rich structure.

### Comparison Table

| Method | Curse of Dimensionality | Key Requirement | Notes |
|---|---|---|---|
| Bootstrapping | None | None | Optimal in moderate-high $d$ |
| SMOTE | $(k/N_1)^{1/d}$ | Lipschitz loss | Only competitive for small $d$ |
| KDE | $N_1^{-\beta/(2\beta+d)}$ | Smoothness of density | Minimax-optimal but slow |
| Diffusion | $\Delta_{\text{score}} \sqrt{\log T}$ | Accurate score estimation | Competitive if score is good |
| Plug-in | None (direct) | Consistent $\hat{g}$ | Computationally simplest |

## Plug-In Estimator (Proposition 1)

An alternative to oversampling: directly estimate the conditional probability $g^*(x) = \mathbb{P}(Y=1|X=x)$ from imbalanced data, then reweight:

$$\hat{f}_{\text{PLUG}}(x) = h\left(\frac{\pi_1 \hat{g}(x)}{\pi_1 \hat{g}(x) + \pi_0(1-\hat{g}(x))}\right)$$

Error bound:

$$\|\hat{f}_{\text{PLUG}} - f^*\|_{\mathbb{Q}_X} \leq \frac{\sqrt{\pi_0/2}}{\pi_1} \cdot \|\hat{g} - g^*\|_{\mathbb{P}_X}$$

> [!NOTE]
> "The plug-in approach has the benefit of being computationally very simple." It avoids generating synthetic samples entirely but pays an extra $\sqrt{\pi_0}/\pi_1$ amplification factor from the imbalance ratio.

## Experiments

- **Dataset**: Synthetic Gaussian mixture model
  - $\mathbb{P}_{X|Y=0} = \mathcal{N}(\mathbf{0}_d, I_d)$
  - $\mathbb{P}_{X|Y=1} = \mathcal{N}(\mathbf{1}_d / \sqrt{d}, I_d)$ (mean vector with unit $\ell_2$ norm)
  - $\pi_0 = 0.9$, varying $d$ and $N$
- **Model**: Logistic regression (Lipschitz, strongly convex loss)
- **Metric**: Excess risk ratio of SMOTE vs. bootstrapping
- **Hardware**: Not specified
- **Result**: The ratio scales approximately as $N_1^{1/2 - 1/d}$, confirming that SMOTE's relative performance degrades as $d$ increases. Bootstrapping is strictly preferable in moderately high dimensions.

## Key Assumptions

1. **Loss Boundedness**: $\sup_{f \in \mathcal{F}} |L(Y, f(X))| \leq b$ (Theorem 1)
2. **Realizability**: $f^* \in \mathcal{F}$ with proper loss (Theorem 2)
3. **For SMOTE**: Global Lipschitz continuity of loss $L$ and bounded covariates $\|X\|_2 \leq D$
4. **For fast rates**: Lipschitz + strongly convex loss (e.g., logistic regression)

## Differences from Related Work

**vs. Ahmad et al. (2025)**:
- This work removes the density lower-bound assumption (which excluded many minority distributions)
- Provides fast rates via localization (Ahmad et al. achieve only slow rates)
- Weaker Lipschitz assumptions on covariates

**vs. Lyu et al. (2025)**:
- Avoids asymptotically non-negligible bias terms present in Lyu et al.
- Provides explicit complexity relationships tied to the target distribution $\mathbb{Q}$

**vs. Standard Transfer Learning**:
- In standard transfer learning, source and target feature distributions differ; here, $\mathbb{P}(X|Y) = \mathbb{Q}(X|Y)$ by construction (only label marginals differ), making the transfer problem more structured.

## Practical Recommendations

1. **High dimensions ($d \geq 10$)**: Prefer bootstrapping over SMOTE; SMOTE's synthesis error dominates.
2. **Low dimensions ($d \leq 3$)**: SMOTE can outperform bootstrapping if $k$ is chosen appropriately.
3. **No synthetic generation needed**: Use the plug-in approach; simpler and avoids dimension-dependent errors.
4. **Rich minority structure + good score model**: Diffusion-based sampling is viable.
5. **Tree-based models**: Avoid SMOTE (requires Lipschitz loss); bootstrapping or plug-in preferred.
