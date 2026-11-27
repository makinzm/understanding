# Meta Information

- URL: [DAGs with NO TEARS: Continuous Optimization for Structure Learning](https://arxiv.org/abs/1803.01422)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. *NeurIPS 2018*.

# DAGs with NO TEARS: Continuous Optimization for Structure Learning

## Background: DAG Structure Learning Problem

DAG structure learning (also called causal discovery or Bayesian network structure learning) aims to recover the underlying directed acyclic graph (DAG) of causal relationships from observational data $X \in \mathbb{R}^{n \times d}$, where $n$ is the number of samples and $d$ is the number of variables (nodes).

**Classical difficulty:** The number of DAGs over $d$ nodes grows super-exponentially (Robinson, 1977), making exhaustive search intractable. Traditional approaches either:
- Use greedy score-based search (FGS, GES) — local and prone to getting stuck
- Use constraint-based methods (PC algorithm) — rely on multiple conditional independence tests
- Impose structural constraints (bounded in-degree, treewidth) — limiting applicability

**Linear SEM formulation:** Under the linear structural equation model (SEM), data is generated as:

```math
\begin{align}
  X = XW + Z
\end{align}
```

where $W \in \mathbb{R}^{d \times d}$ is the weighted adjacency matrix of the DAG ($W_{ij} \neq 0$ means edge $j \to i$), and $Z \in \mathbb{R}^{n \times d}$ is noise. The goal is to estimate the sparsest $W$ consistent with data, subject to $W$ encoding a DAG.

**Score function (least squares with L1 regularization):**

```math
\begin{align}
  F(W) = \frac{1}{2n}\|X - XW\|_F^2 + \lambda\|W\|_1
\end{align}
```

The optimization problem is:

```math
\begin{align}
  \min_{W \in \mathbb{R}^{d \times d}} F(W) \quad \text{subject to } \mathcal{G}(W) \in \text{DAGs}
\end{align}
```

Prior work handled the DAG constraint via combinatorial search. NOTEARS converts it into a smooth, differentiable equality constraint.

## Key Innovation: Acyclicity as a Smooth Constraint

The central contribution is a novel algebraic characterization of acyclicity using the matrix exponential.

**Intuition:** For binary $B \in \{0,1\}^{d \times d}$, $[B^k]_{ii}$ counts walks of length $k$ from node $i$ back to itself (i.e., cycles through $i$). If $B$ contains no directed cycles, all such cycle-counts are zero. The matrix exponential accumulates all walk lengths:

```math
\begin{align}
  e^B = \sum_{k=0}^{\infty} \frac{B^k}{k!}
\end{align}
```

so $\text{tr}(e^B) = \sum_i [e^B]_{ii} = d$ iff no cycles exist (each diagonal contributes exactly 1 from the identity term).

**Proposition 1** (infinite series form): For $B \in \{0,1\}^{d \times d}$ with spectral radius $r(B) < 1$, $B$ is acyclic iff:

```math
\begin{align}
  \text{tr}(I - B)^{-1} = d
\end{align}
```

**Proposition 2** (matrix exponential form): For $B \in \{0,1\}^{d \times d}$, $B$ is acyclic iff:

```math
\begin{align}
  \text{tr}(e^B) = d
\end{align}
```

**Theorem 1** (main result — extension to real-valued $W$): Matrix $W \in \mathbb{R}^{d \times d}$ is acyclic (i.e., $\mathcal{G}(W)$ is a DAG) iff:

```math
\begin{align}
  h(W) = \text{tr}(e^{W \circ W}) - d = 0
\end{align}
```

where $W \circ W$ denotes element-wise squaring (Hadamard product) to ensure non-negative entries. The gradient is:

```math
\begin{align}
  \nabla h(W) = (e^{W \circ W})^T \circ 2W
\end{align}
```

$h(W) \geq 0$ always holds; $h(W) = 0$ iff $W$ is a DAG. This function is smooth ($C^\infty$) with easily computable gradients via automatic differentiation.

## Optimization Framework

The combinatorial DAG constraint is replaced by the smooth equality constraint $h(W) = 0$, yielding an equality-constrained program (ECP):

```math
\begin{align}
  \min_{W \in \mathbb{R}^{d \times d}} F(W) \quad \text{subject to } h(W) = 0
\end{align}
```

This is solved using the **augmented Lagrangian method**, forming the Lagrangian:

```math
\begin{align}
  \mathcal{L}^\rho(W, \alpha) = F(W) + \frac{\rho}{2}|h(W)|^2 + \alpha h(W)
\end{align}
```

where $\alpha \in \mathbb{R}$ is the dual variable (Lagrange multiplier) and $\rho > 0$ is a penalty parameter. The dual ascent update is:

```math
\begin{align}
  \alpha \leftarrow \alpha + \rho \cdot h(W^*_\alpha)
\end{align}
```

**Proposition 3 (convergence):** For $\rho$ sufficiently large and starting $\alpha_0$ near the optimal $\alpha^*$, the dual ascent update converges linearly to $\alpha^*$.

## Algorithms

### Algorithm 1: NOTEARS (Outer Loop)

```
Input:  Initial W₀, α₀; progress rate c ∈ (0,1); tolerance ε > 0; threshold ω > 0

For t = 0, 1, 2, ...:
  (a) Minimize: W_{t+1} ← argmin_W L^ρ(W, α_t)
      such that h(W_{t+1}) < c · h(W_t)    [ensure progress]
  (b) Dual update: α_{t+1} ← α_t + ρ · h(W_{t+1})
  (c) If h(W_{t+1}) < ε:
        Set W̃ = W_{t+1}, break

Return: Ŵ := W̃ ∘ 𝟙(|W̃| > ω)    [hard threshold to prune weak edges]
```

The progress condition in step (a) prevents the outer loop from terminating prematurely. Hard thresholding with $\omega$ removes spurious near-zero edges.

### Algorithm 2: Proximal Quasi-Newton (Inner Solver)

The inner subproblem $\min_W \mathcal{L}^\rho(W, \alpha_t)$ contains both a smooth part $f(W)$ and a non-smooth L1 penalty. A proximal quasi-Newton method with L-BFGS curvature approximation solves it:

```
Input: Initial w₀; active set S ← {1, ..., p}

For k = 0, 1, 2, ...:
  (a) Shrink active set S by removing variables where w_j = 0
      and subgradient condition is satisfied
  (b) If shrinking criterion met: reset S ← {1,...,p}, reset L-BFGS
  (c) Compute quasi-Newton direction d_k on active set:
        d_k = argmin_d g_k^T d + (1/2) d^T B_k d + λ‖w_k + d‖₁
  (d) Line search: find η by Armijo rule
  (e) w_{k+1} ← w_k + η · d_k
  (f) Update L-BFGS variables using curvature information on S
```

The coordinate update uses **soft-thresholding**:

```math
\begin{align}
  z^* = -c + \mathcal{S}\!\left(c - \frac{b}{a}, \frac{\lambda}{a}\right)
\end{align}
```

where $\mathcal{S}(x, t) = \text{sign}(x)\max(|x|-t, 0)$ is the soft-threshold operator, $a = B_{jj}$, $b = g_j + (Bd)_j$, $c = b/a$.

The L-BFGS approximation uses a compact form $B_k = \gamma_k I - QQ^T$ with memory size $m \ll p$, giving $O(m^2 p + m^3)$ complexity per iteration.

> [!NOTE]
> NOTEARS is named as an acronym: **N**on-combinatorial **O**ptimization via **T**racing **E**xponential matrix and **A**ugmented lagrangian for **R**egularized **S**tructure learning, as well as a play on "no tears" referencing the difficulty of prior methods.

## Comparison with Prior Methods

| Method | Search Strategy | Structural Assumptions | Noise Assumption | Scalability |
|---|---|---|---|---|
| **NOTEARS** | Continuous gradient | None | Model-agnostic | $O(d^3)$ per iter |
| **FGS** (Ramsey et al., 2016) | Greedy score-based | None | Gaussian | $O(d^2)$ per step |
| **PC algorithm** (Spirtes & Glymour, 1991) | Constraint (CI tests) | Faithfulness | Gaussian/nonparam | $O(d^{2k})$ |
| **LiNGAM** (Shimizu et al., 2006) | ICA-based | Linear, no Gaussian | Non-Gaussian | $O(d^3)$ |
| **GOBNILP** (Cussens, 2012) | Integer programming | Bounded in-degree | Gaussian | Exponential |

Key differentiators of NOTEARS:
- **Global search**: Unlike greedy methods (FGS, GES) that add/remove one edge at a time, NOTEARS optimizes $W$ globally as a matrix
- **No in-degree restriction**: Exact solvers like GOBNILP require bounding the maximum in-degree ($k \leq 3$) for tractability; NOTEARS does not
- **Model-agnostic**: Unlike LiNGAM (requires non-Gaussian noise) or parametric score methods, NOTEARS works with any noise distribution
- **Simple implementation**: ~50 lines of Python vs. hundreds for competing methods

> [!IMPORTANT]
> The acyclicity constraint $h(W) = \text{tr}(e^{W \circ W}) - d = 0$ requires computing the matrix exponential, which costs $O(d^3)$ per evaluation. This makes NOTEARS less efficient than greedy methods for very large graphs ($d > 1000$), but more accurate for moderate sizes ($d \leq 100$).

## Experiments

- **Dataset (synthetic)**: Randomly generated linear SEMs with the following configurations:
  - Graph types: Erdős-Rényi ER-1 ($d$ edges), ER-2 ($2d$ edges), ER-4 ($4d$ edges); Scale-Free SF ($\sim 4d$ edges)
  - Node counts: $d \in \{10, 20, 50, 100\}$
  - Sample sizes: $n = 20$ (high-dimensional) and $n = 1000$ (low-dimensional)
  - Noise: Gaussian $\mathcal{N}(0, I)$, Exponential $\text{Exp}(1)$, Gumbel$(0,1)$
- **Dataset (real)**: Sachs et al. (2005) protein signaling network, $d = 11$ nodes, $n = 7466$ samples
- **Hardware**: Not specified
- **Optimizer**: L-BFGS (inner), augmented Lagrangian with dual ascent (outer)
- **Results**:
  - NOTEARS achieves near-global optima: objective values within $\sim 5\%$ of GOBNILP's exact solutions on $d=10$ graphs
  - Outperforms FGS on denser graphs (ER-4, SF) in SHD (Structural Hamming Distance), FDR, TPR
  - Robust to non-Gaussian noise: FGS and PC degrade, while NOTEARS is stable
  - Recovers 11/17 edges in the Sachs protein network with low FDR, comparable to existing methods
  - Scales to $d = 100$ in seconds on a laptop

> [!TIP]
> The official implementation is available at [https://github.com/xunzheng/notears](https://github.com/xunzheng/notears). A JAX-based extension supporting nonlinear SEMs (NOTEARS-MLP) was later published as "DAG-GNN" and "NoCurl."

## Limitations and Future Directions

1. **Non-convexity**: The augmented Lagrangian objective is non-convex in $W$; NOTEARS finds stationary points, not guaranteed global optima
2. **Cubic complexity**: Matrix exponential computation costs $O(d^3)$ per evaluation, making very large graphs ($d \gg 100$) expensive
3. **Threshold selection**: The edge pruning threshold $\omega$ must be tuned manually; wrong values yield spurious or missing edges
4. **Linear SEM assumption**: The score function $F(W)$ assumes linear relationships; extensions to nonlinear SEMs require architectural modifications (addressed by later follow-up work)
5. **Identifiability**: Without non-Gaussianity or intervention data, the recovered DAG is only identifiable up to a Markov equivalence class under Gaussian noise
