# Meta Information

- URL: [Solving the Cold Start Problem on One's Own as an End User via Preference Transfer](https://arxiv.org/abs/2502.12398)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Sato, R. (2025). Solving the Cold Start Problem on One's Own as an End User via Preference Transfer. arXiv:2502.12398.

# Overview

Cold start is the problem where a recommender system cannot produce meaningful recommendations for a new user who has no interaction history. Classic solutions require modifications on the service provider's side (e.g., asking new users for ratings, cross-domain transfer via server-side model changes). This paper proposes **Pretender**, a user-side algorithm that lets the end user independently solve their own cold start problem by transferring their existing preferences from one service to another, with no server-side modifications required.

**Applicability:** End users who already have a preference history in one recommender service (source) and wish to bootstrap their profile in a new service (target). The user must have access to item feature vectors in both services (e.g., genre tags, metadata).

# Problem Formulation

Let:
- $I_S$ = set of source service items, $I_T$ = set of target service items
- Each item $i$ has feature vector $\mathbf{x}_i \in \mathbb{R}^d$
- $D_S = \{(i, y_i)\} \subseteq I_S \times \{0,1\}$ = user's labeled preference history on source (positive/negative)
- Goal: construct $D_T = \{(i, y_i)\} \subseteq I_T \times \{0,1\}$ of exactly $K$ labeled items for the target service

The empirical distributions are:

$$\mu_S = \frac{1}{|D_S|} \sum_{(i,y_i) \in D_S} \delta_{(\mathbf{x}_i, y_i)}, \quad \mu_T^{(D_T)} = \frac{1}{K} \sum_{(i,y_i) \in D_T} \delta_{(\mathbf{x}_i, y_i)}$$

The objective is to minimize the distributional distance $D(\mu_T^{(D_T)}, \mu_S)$.

# Algorithm: Pretender

The algorithm operates in four stages:

**Stage 1 — Item Preparation:**
Construct the labeled candidate set $J_T = \{(i, y) \mid i \in I_T, y \in \{0,1\}\}$ with $|J_T| = 2|I_T| = 2m$. Each candidate is a pair of an item and a hypothetical label.

**Stage 2 — Continuous Optimization (Frank-Wolfe):**
Solve the relaxed (continuous weight) version:

$$\min_{\mathbf{w} \in \mathbb{R}^{2m}} D\!\left(\sum_{j=1}^{2m} w_j \delta_{(\mathbf{x}_{i_j}, y_j)},\ \mu_S\right)$$

subject to $\sum_j w_j = 1$ and $0 \le w_j \le 1/K$.

Use Frank-Wolfe with step size $\eta_t = 2/(t+2)$ for iterations $t = 0, 1, \ldots, L-1$.

```
Input: candidate set J_T, source distribution μ_S, budget K, steps L
Initialize: w_0 = uniform over J_T
For t = 0 to L-1:
    Compute gradient ∇D(w_t) w.r.t. each w_j
    Find j* = argmin_j [∇D(w_t)]_j   (linear minimization oracle)
    Set s_t = e_{j*} / K              (sparse update step)
    Update w_{t+1} = w_t + (2/(t+2)) * (s_t - w_t)
Return w_L
```

Convergence rate is $O(L^{-1})$.

**Stage 3 — Randomized Rounding:**
Sample each candidate independently: $I_j \sim \text{Bernoulli}(K \cdot w_j)$. Set $\hat{D}_T = \{(i_j, y_j) \mid I_j = 1\}$.

**Stage 4 — Postprocessing to Exact Size $K$:**
- If $|\hat{D}_T| < K$: greedily insert candidates from $J_T \setminus \hat{D}_T$ that most reduce $D$.
- If $|\hat{D}_T| > K$: greedily remove candidates from $\hat{D}_T$ that most reduce $D$.
- Output: $D_T$ of exactly $K$ items.

# Distance Objectives

## Maximum Mean Discrepancy (MMD)

$$\text{MMD}(\mu, \nu)^2 = \mathbb{E}_{\mathbf{x},\mathbf{x}' \sim \mu}[k(\mathbf{x},\mathbf{x}')] - 2\mathbb{E}_{\mathbf{x}\sim\mu, \mathbf{x}'\sim\nu}[k(\mathbf{x},\mathbf{x}')] + \mathbb{E}_{\mathbf{x},\mathbf{x}' \sim\nu}[k(\mathbf{x},\mathbf{x}')]$$

where $k$ is a positive definite kernel (e.g., RBF). This is differentiable in the weights and admits efficient gradient computation.

## Wasserstein Distance ($W_1$)

$$W_1(\mu^w, \nu) = \inf_{\gamma \in \Pi(\mu^w, \nu)} \sum_{j,j'} \gamma_{jj'} \|\mathbf{x}_j - \mathbf{x}_{j'}\|$$

The Wasserstein objective captures geometric structure in feature space but suffers from the curse of dimensionality in convergence.

# Theoretical Guarantees

## MMD Guarantee (Corollary 3.6)

$$\text{MMD}(\mu_T^{(D_T^*)}, \mu_S) \le \text{OPT}^{\text{combinatorial}} + O(K^{-1/2})$$

The total error decomposes (via triangle inequality) into three parts:
1. Frank-Wolfe optimization error: $O(L^{-1/2})$ (Proposition 3.1)
2. Randomized rounding error: $O(K^{-1/2} / \sqrt{\delta})$ w.p. $1-\delta$ (Proposition 3.3)
3. Postprocessing error: $O(K^{-1/2} / \sqrt{\delta})$ w.p. $1-\delta$ (Proposition 3.4)

## Coverage Assumption (Theorem 3.8)

Under the assumption that the density ratio $r^* = \sup_\mathbf{x} P(\mathbf{x})/Q(\mathbf{x})$ is bounded, the target service contains approximately $|J_T|/r^*$ effective samples from the source distribution, which bounds the combinatorial optimum.

## Model Generalization (Corollary 3.10)

If the target model's loss function $\ell_T$ is bounded in RKHS norm by $R_\ell$:

$$\mathbb{E}_P[\ell_T] \le \text{training loss on } D_T + R_\ell \cdot \text{MMD}(P, D_T)$$

This means minimizing MMD between $D_T$ and $\mu_S$ directly bounds the model's generalization error on the true target distribution $P$.

## Wasserstein Guarantee (Theorem 3.15)

$$\text{Error} \le \text{OPT}^{\text{combinatorial}} + O(K^{-1/(d+2)})$$

where $d$ is the feature dimension. The curse of dimensionality makes this bound weaker than MMD in high dimensions.

> [!IMPORTANT]
> $\text{OPT}^{\text{combinatorial}}$ is **not monotone** in $K$: adding more items does not necessarily reduce the minimum distributional distance (proven via counterexample). Practitioners should run Pretender for all values $K' = 1, 2, \ldots, K$ in parallel and select the solution with the smallest distance.

# Comparison with Related Methods

| Aspect | Pretender | Server-side cold start | Kernel thinning / Herding |
|---|---|---|---|
| Who acts | End user only | Service provider | Coreset designer (offline) |
| Modification required | None | Server model changes | Access to distribution |
| Error decay (MMD) | $O(K^{-1/2})$ | Varies | $O(K^{-1/2})$ (kernel thinning) |
| Submodular greedy bound | Achieves vanishing error | N/A | $(1 - 1/e)$-approximation only |
| User privacy | High (no data sent) | Low | N/A |

> [!NOTE]
> The paper states: "Pretender can be applied directly by an end user without requiring any modifications to the service itself."

> [!TIP]
> The Frank-Wolfe algorithm used here is the same as conditional gradient methods. See [Jaggi 2013] for a general treatment.

# Limitations

**Inaccessible Implicit Features (Section 4.3):**
Services using collaborative filtering may rely on learned latent embeddings unavailable to users. The authors argue that explicit features (genres, metadata) serve as reasonable surrogates and that users can approximate implicit features from publicly visible recommendation graphs.

**Non-Monotone Optimum (Section 4.2):**
$\text{OPT}^{\text{combinatorial}}$ is not monotone in $K$: increasing the budget can increase the minimum achievable MMD. This is unintuitive and requires running the algorithm multiple times to select the optimal $K$.

# Experiments

- **Datasets:**
  - MovieLens 100K: movie ratings; ratings $\ge 4$ = positive; 90-dimensional features (genres + year)
  - Last.fm: music listening; weighted artist plays as positive; PCA-reduced to 50 dimensions
  - Amazon-Home-Kitchen: product reviews; ratings $\ge 4$ = positive; bag-of-words features (50 dimensions)
- **Settings:** Intersection (source and target share items) and no-intersection (disjoint item sets)
- **Baselines:**
  - Random selection: uniformly sample $K$ items
  - Greedy selection: pick items minimizing $\min_{\mathbf{x}' \in D_S} \|\mathbf{x} - \mathbf{x}'\|_2$
  - Continuous optimal: lower bound from Frank-Wolfe relaxation (not achievable in practice)
- **Metric:** MMD between selected target set and source preference history (lower is better)
- **Results:** Pretender substantially outperforms both Random and Greedy across all three datasets and both settings, closely approaching the continuous optimal lower bound. Notably, Greedy underperforms Random, suggesting naive nearest-neighbor selection is not a good heuristic.
