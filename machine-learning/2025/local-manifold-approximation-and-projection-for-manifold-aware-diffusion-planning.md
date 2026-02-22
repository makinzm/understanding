# Meta Information

- URL: [Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning](https://arxiv.org/abs/2506.00867)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Lee, K. & Choi, J. (2025). Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning. ICML 2025.

# Overview

Diffusion-based trajectory planners (e.g., Diffuser) learn to generate action sequences for long-horizon offline RL by modeling the trajectory distribution. During inference, a guidance function (e.g., cumulative return) steers the diffusion sampling toward high-reward trajectories. However, the standard MSE-trained guidance underestimates the true gradient, causing **manifold deviation**: generated trajectories drift off the feasible trajectory manifold and become physically infeasible.

**LoMAP (Local Manifold Approximation and Projection)** is a training-free, plug-and-play module applied at test time. At each reverse diffusion step, it retrieves k-nearest offline neighbors of the current denoised estimate, forward-diffuses them to the same noise level, runs PCA to construct a local low-rank subspace approximating the trajectory manifold, and projects the current sample onto that subspace. This forces the sample to remain near feasible trajectories throughout the denoising process.

**Applicability:** Offline RL practitioners using diffusion-based planners (Diffuser, Hierarchical Diffuser) on continuous-control tasks (locomotion, maze navigation) where trajectory feasibility is critical. Requires access to the offline dataset at inference time for neighbor retrieval.

# Problem Formulation

Consider an offline RL setting with a Markov Decision Process $(S, A, P, R, \gamma)$. A trajectory is $\tau^0 = (s_0, a_0, s_1, a_1, \ldots, s_H, a_H) \in \mathbb{R}^d$ where $d = H \cdot (|S| + |A|)$.

**Forward diffusion process** (DDPM):

$$q(\tau^i \mid \tau^0) = \mathcal{N}(\tau^i;\ \sqrt{\alpha_i}\,\tau^0,\ (1-\alpha_i)I)$$

where $\alpha_i = \prod_{s=1}^{i}(1 - \beta_s)$ and $\{\beta_i\}$ is the variance schedule.

**Reverse process** (learned denoiser):

$$p_\theta(\tau^{i-1} \mid \tau^i) = \mathcal{N}(\tau^{i-1} \mid \mu_\theta(\tau^i),\ \Sigma^i)$$

**Guided reverse process** (classifier guidance):

$$p_\phi(\tau^{i-1} \mid \tau^i) \propto p_\theta(\tau^{i-1} \mid \tau^i) \cdot \exp\!\bigl(\mathcal{J}_\phi(\tau^{i-1})\bigr)$$

The noise predictor $\varepsilon_\theta : \mathbb{R}^d \to \mathbb{R}^d$ is trained by minimizing:

$$\mathcal{L}(\theta) = \mathbb{E}_{i, \varepsilon, \tau^0}\!\left[\|\varepsilon - \varepsilon_\theta(\tau^i)\|^2\right], \quad \tau^i = \sqrt{\alpha_i}\,\tau^0 + \sqrt{1-\alpha_i}\,\varepsilon$$

# The Guidance Gap Problem

## True vs. MSE Guidance

The **true guidance** at noise level $i$ is:

$$\mathcal{J}_t(\tau^i) = \log \mathbb{E}_{q(\tau^0 \mid \tau^i)}\!\left[\exp\!\bigl(\mathcal{J}(\tau^0)\bigr)\right]$$

The **MSE-optimal guidance** satisfies:

$$\mathcal{J}_\phi^{\text{MSE}}(\tau^i) = \mathbb{E}_{q(\tau^0 \mid \tau^i)}\!\left[\mathcal{J}(\tau^0)\right] \leq \mathcal{J}_t(\tau^i)$$

by Jensen's inequality (since $\exp$ is convex). This underestimation is the root cause of guidance error.

## Proposition 3.2 — Guidance Gap Scales as $O(\sqrt{d})$

For a return function with bounded variance, the gradient gap satisfies:

$$\|\nabla_{\tau^i}\mathcal{J}_t(\tau^i) - \nabla_{\tau^i}\mathcal{J}_\phi^{\text{MSE}}(\tau^i)\|_2 \geq \frac{c}{\sqrt{1-\alpha_i}}\sqrt{d}$$

where $c > 0$ is a dimension-independent constant. This means the guidance error grows with the state-action space dimensionality $d$, making the problem **increasingly severe in high-dimensional environments**.

> [!IMPORTANT]
> The bound $O(\sqrt{d})$ implies that simply improving the guidance neural network does not resolve the issue — the underestimation is inherent to the MSE objective, not a capacity problem. LoMAP addresses this by constraining samples to feasible regions rather than correcting the gradient.

# LoMAP: Local Manifold Approximation and Projection

## Key Idea

Rather than fixing the guidance gradient, LoMAP constrains the sampling trajectory to remain on the **feasible manifold** $\mathcal{M}$ — the set of physically realizable state-action sequences. It approximates $\mathcal{M}$ locally using a low-rank subspace derived from offline neighbors.

## Algorithm

**Input:** Current noisy sample $\tau^{i-1} \in \mathbb{R}^d$, noise predictor $\varepsilon_\theta$, offline dataset $\mathcal{D}$, neighbors $k$, variance threshold $\lambda$

**Output:** Projected sample $\tau^{i-1} \in \mathbb{R}^d$ constrained to local manifold

```
Algorithm: LoMAP Projection at Step i-1

1.  # Denoise current sample using Tweedie's formula
    τ̂^(0|i-1) ← (1/√α_{i-1}) * (τ^{i-1} - √(1 - α_{i-1}) * ε_θ(τ^{i-1}))
    # τ̂^(0|i-1) ∈ ℝ^d is the estimated clean trajectory

2.  # Retrieve k-nearest neighbors from offline dataset
    N = {n_1, ..., n_k} ← kNN(τ̂^(0|i-1), 𝒟)
    # Similarity measured by cosine distance in trajectory space

3.  # Forward-diffuse each neighbor to noise level i-1
    for j = 1 to k:
        ε_{n_j} ~ 𝒩(0, I)
        τ^{n_j, i-1} ← √α_{i-1} * τ^{n_j, 0} + √(1 - α_{i-1}) * ε_{n_j}

4.  # Construct local manifold via PCA
    X ← stack([τ^{n_1, i-1}, ..., τ^{n_k, i-1}])  # shape: k × d
    μ ← mean(X, axis=0)                              # shape: d
    C ← (X - μ)^T (X - μ)                           # d × d covariance
    U, Σ, V^T ← SVD(C)
    # Select r principal components explaining λ=0.99 of variance
    r ← min{r : Σ_1 + ... + Σ_r ≥ λ * sum(Σ)}
    U ← U[:, :r]   # U ∈ ℝ^{d × r}

5.  # Project current sample onto local subspace
    τ^{i-1} ← μ + U U^T (τ^{i-1} - μ)
    # Projects onto the r-dimensional affine subspace centered at μ

6.  return τ^{i-1}
```

## Integration with Guided Diffusion

LoMAP wraps the standard reverse diffusion loop. At each step, after computing the guided update, the projected sample replaces the unconstrained one before the next denoising step:

```
for i = N, N-1, ..., 1:
    τ^{i-1} ← guided_reverse_step(τ^i, ε_θ, J_φ)   # standard guided DDPM
    if apply_lomap(i):                                 # applied at intermediate steps
        τ^{i-1} ← LoMAP(τ^{i-1}, ε_θ, 𝒟, k, λ)
```

> [!NOTE]
> Projection is applied selectively at intermediate-to-later denoising steps (not at every step from $i = N$), which avoids over-constraining early steps where trajectories are still highly noisy. The exact range is tuned per environment.

## Efficient Neighbor Retrieval

Neighbor lookup uses **FAISS IVF (Inverted File)** indexing, which partitions the dataset into Voronoi cells and searches only relevant cells. This reduces retrieval cost from $O(|\mathcal{D}|)$ to approximately $O(\sqrt{|\mathcal{D}|})$ per query, making LoMAP practical on datasets of tens of thousands of trajectories.

# Comparison with Related Methods

| Method | Training Required | Manifold Awareness | Guidance Correction | Memory |
|--------|------------------|-------------------|---------------------|--------|
| Diffuser (baseline) | Yes (planner + guidance) | None | None | Low |
| RGG (Restoration Gap Guidance) | No | Indirect (via gap restoration) | Partial | Low |
| MCG (Manifold Constrained Guidance) | Yes (autoencoder) | Yes (learned manifold) | Yes | High |
| MPGD | Yes (diffusion autoencoder) | Yes (learned manifold) | Yes | High |
| **LoMAP (ours)** | **No** | **Yes (local PCA subspace)** | **No (constraint instead)** | Medium (FAISS index) |

> [!NOTE]
> Unlike MCG and MPGD which require training a separate autoencoder to learn the manifold, LoMAP is entirely training-free and constructs local approximations on the fly from offline data.

# Experiments

- **Datasets:**
  - **Maze2D** (D4RL): 2D maze navigation; 3 sizes (U-Maze, Medium, Large); state dim $= 4$, action dim $= 2$; also multi-task variants
  - **MuJoCo Locomotion** (D4RL): 3 environments — HalfCheetah (state $\in \mathbb{R}^{17}$, action $\in \mathbb{R}^6$), Hopper (state $\in \mathbb{R}^{11}$, action $\in \mathbb{R}^3$), Walker2d (state $\in \mathbb{R}^{17}$, action $\in \mathbb{R}^6$); 3 dataset qualities (Medium, Medium-Expert, Medium-Replay)
  - **AntMaze** (D4RL): Ant robot navigation; state $\in \mathbb{R}^{29}$, action $\in \mathbb{R}^8$; sizes Medium and Large with data distributions Play and Diverse

- **Optimizer:** Adam (for training base diffusion models; LoMAP itself requires no training)

- **Baselines:**
  - Offline RL: IQL, CQL, BC, MOPO, MOReL
  - Diffusion planners: Diffuser, Decision Diffuser (DD), Trajectory Transformer (TT), Decision Transformer (DT)
  - Refinement/guidance methods: RGG, TAT (Trajectory Aggregation Tree)

- **Results:**
  - **Maze2D-Large:** Diffuser+LoMAP achieves $151.9 \pm 2.66$ vs. $123.0$ for base Diffuser (23% gain)
  - **MuJoCo average:** $82.8$ (Diffuser+LoMAP) vs. $77.5$ (Diffuser); largest gains on Medium datasets
  - **AntMaze average:** $86.7$ (HD+LoMAP) vs. $55.3$ (Hierarchical Diffuser); LoMAP is particularly effective in larger, more complex mazes
  - **Artifact ratio:** LoMAP consistently reduces infeasible trajectory fractions across all Maze2D variants compared to Diffuser and RGG

# Limitations

- **High-dimensional pixel observations:** Cosine similarity in raw trajectory space is unreliable for image-based observations. Alternative embedding-space distance metrics would be needed.
- **Conservatism:** By projecting onto subspaces spanned by offline data neighbors, LoMAP inherently discourages behaviors not represented in the dataset — potentially limiting generalization to out-of-distribution states.
- **Trajectory stitching:** Benchmarks requiring creative combination of sub-optimal trajectory segments (e.g., OGBench) may not benefit, as the projection enforces locality relative to observed trajectories rather than enabling novel combinations.
