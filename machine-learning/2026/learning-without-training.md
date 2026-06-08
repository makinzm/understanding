# Meta Information

- URL: [Learning Without Training](https://arxiv.org/abs/2602.17985)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: O'Dowd, R. (2026). Learning Without Training. PhD Dissertation, Claremont Graduate University.

---

This dissertation presents three machine learning projects grounded in harmonic analysis and approximation theory. The unifying theme is **direct constructive approximation**: building estimators analytically from data without any gradient-based optimization. The three projects address (1) supervised learning on unknown manifolds, (2) transfer learning as function lifting between manifolds, and (3) active classification via signal separation. All three avoid neural network pitfalls such as local minima, initialization sensitivity, and convergence to misleading local optima.

**Applicability**: Researchers and practitioners who work with high-dimensional data lying on low-dimensional manifolds (e.g., hyperspectral imaging, medical imaging, text classification) and who want theoretically guaranteed convergence rates without tuning an optimizer.

---

# Background and Motivation

## Supervised Learning Paradigm Shortcomings

The dominant paradigm minimizes empirical risk:

```math
\begin{align}
  \hat{f} = \arg\min_{f \in \mathcal{H}} \frac{1}{M} \sum_{j=1}^{M} \ell(f(y_j), z_j)
\end{align}
```

where $\mathcal{H}$ is a hypothesis class (e.g., neural networks), $(y_j, z_j)$ are input-output pairs, and $\ell$ is a loss function. This approach suffers from:

- **Local minima**: Gradient descent may converge to suboptimal solutions.
- **Initialization sensitivity**: Results vary significantly with random seed.
- **Implicit bias**: The function minimizing empirical risk may not generalize; the approximation degree depends on the optimizer's behavior, not just the architecture.

## Proposed Alternative

Instead of optimization, the dissertation builds approximators as **explicit finite sums** over kernel evaluations. The kernel is derived from harmonic analysis and respects the underlying geometry of the data manifold. This yields:

- Deterministic convergence guarantees (up to probability over the data draw).
- Sample complexity depending on **intrinsic manifold dimension** $q$, not ambient dimension $Q$.
- No training loop, no backpropagation, no hyperparameter search beyond manifold dimension.

---

# Chapter 2: Approximation on Manifolds

## Problem Setup

**Input**: Scattered data $\mathcal{D} = \{(y_j, z_j)\}_{j=1}^{M}$ where $y_j \in \mathbb{R}^Q$ lie on an unknown compact submanifold $\mathcal{X} \subset \mathbb{S}^Q$ (mapped to the ambient unit sphere) of intrinsic dimension $q \ll Q$, and $z_j = f(y_j) + \varepsilon_j$ are noisy function evaluations.

**Output**: An approximation $F_n(\mathcal{D}; x)$ of $f$ at any new point $x \in \mathcal{X}$, including out-of-sample points.

## Kernel Construction

The approximator uses a **localized spherical polynomial kernel** of degree less than $n$:

```math
\begin{align}
  F_n(\mathcal{D}; x) := \frac{1}{M} \sum_{j=1}^{M} z_j \, \Phi_{n,q}(x \cdot y_j)
\end{align}
```

where $x \cdot y_j$ is the inner product (cosine similarity) on $\mathbb{S}^Q$. The kernel $\Phi_{n,q}$ is constructed from orthonormalized Hermite functions $h_k$ and a localization window $H: [0, \infty) \to [0, 1]$:

```math
\begin{align}
  \tilde{\Phi}_{n,q}(x) = \sum_{k=0}^{\lfloor n^2/2 \rfloor} H\!\left(\frac{\sqrt{2k}}{n}\right) \mathcal{P}_{k,q}(x)
\end{align}
```

where $\mathcal{P}_{k,q}$ are zonal spherical harmonics of degree $k$ on a $q$-dimensional sphere, and $H$ is a smooth cutoff function. The localization ensures that $\Phi_{n,q}(x \cdot y_j)$ is large only when $x$ and $y_j$ are close on the manifold.

> [!NOTE]
> "Direct construction does not require prior assumptions." The kernel is universal: the same formula applies regardless of the specific manifold, as long as the manifold dimension $q$ is known (or estimated).

## Main Convergence Theorem (Theorem 2.5.1, informal)

For $f \in W^\gamma(\mathcal{X})$ (a Sobolev-like smoothness class with parameter $\gamma > 0$), with high probability over the random draw of $M$ data points:

```math
\begin{align}
  \|F_n(\mathcal{D}; \cdot) - f\|_{\mathcal{X}} \leq c \left(\|z\|_\infty + \|f\|_{W^\gamma}\right) \left(\frac{\log M}{M}\right)^{\gamma/(q + 2\gamma)}
\end{align}
```

**Variables**:
- $c$ — a constant depending on the manifold geometry.
- $q$ — intrinsic manifold dimension.
- $\gamma$ — smoothness order; larger $\gamma$ yields faster convergence.
- $M$ — number of data points.

**Required sample size**: To achieve rate $n^{-\alpha\gamma}$, one needs $M \geq n^{q(2-\alpha) + 2\alpha\gamma} \log(n/\delta)$ with confidence $1-\delta$, for $\alpha \in (0, 4/(2+\gamma)]$.

> [!IMPORTANT]
> The rate $({\log M}/{M})^{\gamma/(q+2\gamma)}$ scales with intrinsic dimension $q$, not ambient dimension $Q$. Classical methods that ignore manifold structure suffer the curse of dimensionality at rate $(\log M/M)^{\gamma/(Q+2\gamma)}$, which is exponentially slower in $Q - q$.

## Comparison with Similar Methods

| Method | Requires Manifold Learning? | Optimization? | Out-of-sample? | Dimension |
|---|---|---|---|---|
| **This work** | No | No | Yes (via kernel) | Intrinsic $q$ |
| Nadaraya-Watson | No | No | Yes | Ambient $Q$ |
| RBF Networks | No | Yes (training) | Yes | Ambient $Q$ |
| Manifold + RKHS | Yes (eigendecomp.) | Yes | Hard | Intrinsic $q$ |

**Numerical validation (Example 2.2.1)**: On a synthetic function with a singularity, this method concentrates approximation error near the singularity, while Nadaraya-Watson and RBF networks scatter the error across the domain. 60% of test points achieve strictly lower error than competing methods.

---

# Chapter 3: Local Transfer Learning

## Problem Formulation

**Input**: Two manifolds $\mathcal{X}$ (source) and $\mathcal{Y}$ (target); a function $f$ learned on $\mathcal{X}$; data from $\mathcal{Y}$ available only on a subset $\mathcal{Y}' \subset \mathcal{Y}$.

**Output**: An approximation of a lifted function $g \approx T[f]$ on $\mathcal{Y}$, where $T$ is a transfer operator (lifting map) relating the two domains.

## Key Contributions

The chapter characterizes which subsets $\mathcal{Y}'$ of the target space allow well-defined liftings. The theory studies how **local smoothness properties** of $f$ on $\mathcal{X}$ translate to local smoothness properties of $g$ on $\mathcal{Y}$, providing conditions under which the lifted function inherits bounded approximation error.

**Connections to inverse problems**: The framework unifies transfer learning with classical inverse problems such as the inverse Radon transform, where a function on a 2D projection space must be lifted to a 3D volume.

The construction uses a **joint data space** $\mathcal{X} \times \mathcal{Y}$ to simultaneously approximate functions on both manifolds, enabling controlled error propagation during the lifting step.

---

# Chapter 4: Classification via Signal Separation (MASC)

## Reformulation

Classical classification seeks a decision boundary. MASC reframes classification as **point-source signal separation**: each class $c$ contributes a distribution $\mu_c$ supported on a subset $\mathcal{S}_c \subset \mathbb{R}^Q$, and the task is to estimate each $\mathcal{S}_c$ separately.

**Input**: Labeled query points $\{(y_j, c_j)\}$ where $c_j \in \{1, \ldots, C\}$ indicates class membership; active learning oracle for additional queries.

**Output**: Estimated supports $\hat{\mathcal{S}}_c$ for each class; classification of new points by nearest-support assignment.

## MASC Algorithm (Measure-based Active Support Classification)

**Step 1 — Signal separation**: Represent the total data distribution as a mixture:

```math
\begin{align}
  \mu = \sum_{c=1}^{C} w_c \, \mu_c, \quad \text{supp}(\mu_c) = \mathcal{S}_c
\end{align}
```

**Step 2 — Support estimation**: Apply a kernel-based reconstruction to estimate $\mathcal{S}_c$ from labeled samples, using localized kernels analogous to Chapter 2.

**Step 3 — Active query selection**: Use the oracle to query points in regions of high uncertainty (near estimated support boundaries), exploiting the active learning setting to reduce label cost.

**Step 4 — Classification**: Assign a new point $x$ to class $\hat{c} = \arg\min_c d(x, \hat{\mathcal{S}}_c)$, where $d(\cdot, \hat{\mathcal{S}}_c)$ is distance to the estimated support.

## Evaluation Metric

MASC is evaluated using the **F-score** combining precision and recall on estimated support regions:

```math
\begin{align}
  F = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\end{align}
```

where Precision = $|\hat{\mathcal{S}}_c \cap \mathcal{S}_c| / |\hat{\mathcal{S}}_c|$ and Recall = $|\hat{\mathcal{S}}_c \cap \mathcal{S}_c| / |\mathcal{S}_c|$.

---

# Experiments

- **Dataset**: Synthetic Circle-on-Ellipse (binary classification, well-separated classes)
- **Dataset**: Document text classification benchmark
- **Dataset**: Salinas Hyperspectral Image — 111 spectral bands, 16 land-cover classes, 3,727 labeled samples
- **Dataset**: Indian Pines Hyperspectral Image — 200 spectral bands, 16 land-cover classes, 10,366 labeled samples
- **Hardware**: Not specified
- **Optimizer**: None (no gradient-based training)
- **Comparisons**: LAND, LEND (active learning baselines); active vs. semi-supervised learning settings
- **Results**: MASC achieves accuracy competitive with LAND and LEND while providing "results much faster" due to elimination of the optimization loop. Manifold approximation (Chapter 2) outperforms Nadaraya-Watson and RBF networks on functions with localized singularities.

---

# Appendices and Mathematical Background

| Appendix | Topic |
|---|---|
| A | Tauberian Theorem — used for kernel decay analysis |
| B | Differential geometry of manifolds — curvature, tangent spaces |
| C | Orthogonal polynomials — Hermite and Legendre bases |
| D | Neural network representation equivalences |
| E | Coefficient encoding for hardware implementation |

> [!TIP]
> The Tauberian theorem (Appendix A) controls how quickly the localized kernel $\Phi_{n,q}$ decays away from its center, which is the key technical ingredient for proving the convergence bound in Theorem 2.5.1.

> [!CAUTION]
> The dissertation does not appear to compare against modern deep learning baselines (e.g., ResNets or transformers) on standard benchmarks, so the practical performance gap relative to state-of-the-art neural networks on large-scale tasks is unclear. The theoretical guarantees are strongest for smooth functions on low-dimensional manifolds with moderate $M$.
