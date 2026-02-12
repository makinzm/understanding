# Meta Information

- URL: [CRAIG: Coresets for Data-efficient Training of Machine Learning Models](https://arxiv.org/abs/1906.01827)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Mirzasoleiman, B., Bilmes, J., & Leskovec, J. (2020). Coresets for Data-efficient Training of Machine Learning Models. *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*.

---

# CRAIG: Coresets for Data-efficient Training of Machine Learning Models

## Overview

CRAIG (Coresets for Accelerating Incremental Gradient descent) is a data subset selection method that constructs a weighted **coreset** — a small, representative subset of training data — to accelerate gradient-based optimization. Rather than training on the full dataset at each step, the model trains on the coreset, achieving up to 6x speedup for convex problems and up to 3x for deep neural networks, with provable convergence guarantees.

**Applicability**: Useful for practitioners dealing with large-scale datasets where training is bottlenecked by data volume rather than model complexity. Works across logistic regression, SVM, and deep neural networks. Particularly effective when data is redundant (many near-duplicate gradients), which is common in natural image datasets.

---

## Problem Formulation

The standard empirical risk minimization objective over a dataset $V = \{(x_i, y_i)\}_{i=1}^n$ is:

$$\min_{w \in \mathcal{W}} f(w) = \frac{1}{n}\sum_{i \in V} f_i(w) + r(w)$$

where $f_i(w) = \ell(w, (x_i, y_i))$ is the per-sample loss, $r(w)$ is a regularizer, and $\mathcal{W} \subseteq \mathbb{R}^d$ is a convex parameter set.

**Goal**: Find the smallest subset $S \subseteq V$ and per-element stepsizes $\gamma_j > 0$ such that the weighted gradient over $S$ approximates the full gradient with error at most $\epsilon$ for all $w \in \mathcal{W}$:

$$S^* = \arg\min |S|, \quad \text{subject to} \quad \max_{w \in \mathcal{W}} \left\|\sum_{i \in V} \nabla f_i(w) - \sum_{j \in S} \gamma_j \nabla f_j(w)\right\| \leq \epsilon$$

> [!NOTE]
> The constraint must hold for **all** $w \in \mathcal{W}$, not just the current iterate. This makes the coreset valid throughout the entire optimization trajectory.

---

## Key Insight: Gradient Distance via Feature Distance

For many common loss functions (e.g., logistic loss, squared loss), the maximum gradient difference between two data points over the parameter space is bounded by their feature-space distance:

$$\max_{w \in \mathcal{W}} \|\nabla f_i(w) - \nabla f_j(w)\| \leq C \|x_i - x_j\|$$

where $C$ is a constant depending on the loss and the bound on $\|w\|$. This key observation means that:
1. Coreset selection can be precomputed **before** training starts.
2. Pairwise distances $d_{ij} = C\|x_i - x_j\|$ in feature space serve as proxies for gradient similarity.

For deep networks, this bound is adapted to use the gradient of the loss with respect to the **last layer's input**, computed cheaply via a forward pass.

---

## Facility Location Formulation

The coreset selection problem is transformed into a **submodular facility location maximization** problem. Define the total "coverage loss" of a subset $S$:

$$L(S) = \sum_{i \in V} \min_{j \in S} d_{ij}$$

where $d_{ij}$ is the gradient distance between points $i$ and $j$ (approximated via feature distances). Adding a sentinel element $s_0$ with $d_{i,s_0} = \max_{i,j} d_{ij}$ ensures $L(\{s_0\}) < \infty$.

The facility location function to maximize is:

$$F(S) = L(\{s_0\}) - L(S \cup \{s_0\})$$

$F(S)$ is **monotone submodular**: each additional element in $S$ provides diminishing returns in reducing coverage error. This allows greedy maximization with approximation guarantees.

---

## CRAIG Algorithm (Greedy Subset Selection)

**Input**: Dataset $V$, gradient error bound $\epsilon$, pairwise distances $\{d_{ij}\}$

**Output**: Weighted coreset $(S, \{\gamma_j\}_{j \in S})$

```
Algorithm: CRAIG
1. Initialize S ← ∅, sentinel s₀ with d_{i,s₀} = max_{i,j} d_{ij} for all i
2. While L(S ∪ {s₀}) > ε:
    a. Find j* = argmax_{j ∈ V\S} F(j | S)  // marginal gain
    b. S ← S ∪ {j*}
3. For each j ∈ S:
    γ_j ← |{ i ∈ V : j = argmin_{s ∈ S} d_{is} }|  // Voronoi cell size
4. Return (S, {γ_j})
```

**Per-element weights $\gamma_j$**: Each coreset element $j$ is assigned a weight equal to the number of training points for which $j$ is the nearest neighbor in gradient space. This forms a **Voronoi partition** of the dataset — $j$ represents its entire cluster in gradient updates.

**Greedy approximation guarantee**: The greedy algorithm finds a subset of size at most $(1 + \ln(\max_e F(e|\emptyset))) \cdot |S^*|$, where $|S^*|$ is the optimal (minimum) coreset size.

---

## Convergence Theorems

### Theorem 1: Strongly Convex Case

For strongly convex $f$ with convexity constant $\mu > 0$, if CRAIG produces subset $S$ with gradient error $\leq \epsilon$, then Incremental Gradient (IG) with stepsize $\alpha_k = \alpha / k^\tau$ ($\tau \in (0,1]$) converges at rate:

$$\|w_k - w^*\|^2 \leq \frac{2\epsilon R}{\mu^2} \quad \text{(in expectation)}$$

with convergence rate $O(1/\sqrt{k})$, identical to IG on the full dataset. The speedup factor is $|V|/|S|$.

### Theorem 2: Smooth + Strongly Convex Case

For smooth component functions with Lipschitz-continuous gradients (constant $L$), IG on the CRAIG coreset converges to a $2\epsilon/\mu$-neighborhood of $w^*$ at rate $O(1/k^\tau)$.

> [!IMPORTANT]
> Both theorems confirm that training on the coreset achieves the **same asymptotic convergence rate** as training on the full dataset, with speedup proportional to the compression ratio $|V|/|S|$.

---

## Dynamic CRAIG for Deep Neural Networks

For deep networks, gradient distances change as weights evolve, so the coreset must be **periodically refreshed**. The dynamic variant:

1. Approximate $d_{ij}$ using the gradient of the loss w.r.t. the last layer's input (computed cheaply during forward pass, without full backpropagation).
2. Recompute the coreset $S$ every fixed number of parameter updates (e.g., every 5 epochs for ResNet-20 on CIFAR-10).
3. Train on the refreshed coreset between updates.

This captures the key insight that last-layer gradients dominate the gradient direction in deep networks, while avoiding the full $O(n^2 d)$ pairwise gradient computation.

---

## Experiments

- **Datasets**:
  - Covtype.binary: 581,012 training samples, 54 features (convex experiments)
  - Ijcnn1: 49,990 training / 91,701 test samples, 22 features (convex experiments)
  - MNIST: 60,000 training / 10,000 test images (non-convex, 2-layer MLP)
  - CIFAR-10: 50,000 training / 10,000 test images (non-convex, ResNet-20)
- **Hardware**: Not explicitly specified
- **Optimizer**: SGD, SVRG, SAGA (convex); SGD with momentum (deep networks)
- **Results**:
  - Logistic regression on 10% Covtype coreset: 2.75x (SGD), 4.5x (SVRG), 2.5x (SAGA) speedup
  - Logistic regression on 30% Ijcnn1 coreset: 5.6x speedup
  - MNIST 2-layer MLP: 2–3x speedup with slightly improved test accuracy
  - CIFAR-10 ResNet-20: Up to 3x speedup, comparable accuracy to full-data training

---

## Comparison with Related Methods

| Method | Gradient Guarantee | Problem-Specific | Applicable to NNs | Complementary to VR |
|---|---|---|---|---|
| **CRAIG** | Yes ($\epsilon$-bound for all $w$) | No (general loss) | Yes (dynamic) | Yes |
| Random Sampling | No | No | Yes | Yes |
| Importance Sampling | Partial (variance reduction) | No | Limited | Yes |
| Problem-specific Coresets (k-means, SVM) | Yes | Yes | No | Varies |
| SVRG / SAGA | No (variance, not subset) | No | Yes | — |

**Key differences**:
- Unlike **random sampling**, CRAIG provably bounds gradient approximation error for all model weights, not just in expectation.
- Unlike **importance sampling** methods, CRAIG uses gradient-space geometry (Voronoi clustering) rather than individual point sensitivities.
- Unlike **problem-specific coresets** (e.g., for k-means or logistic regression), CRAIG applies the same framework across loss functions.
- CRAIG is **complementary to variance reduction** (SVRG, SAGA): combining CRAIG subset selection with SVRG achieves the best of both — reduced data and reduced variance.

> [!CAUTION]
> The convergence analysis assumes bounded parameter space $\mathcal{W}$ and bounded gradients. The extension to unbounded settings or non-Lipschitz losses requires further work.

> [!TIP]
> The facility location submodularity framework (and lazy greedy optimization) is discussed in depth in: Minoux, M. (1978). Accelerated greedy algorithms for maximizing submodular set functions. *Optimization Techniques*.
