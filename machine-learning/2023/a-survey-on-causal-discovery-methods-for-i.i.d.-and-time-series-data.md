# Meta Information

- URL: [A Survey on Causal Discovery Methods for I.I.D. and Time Series Data](https://arxiv.org/abs/2303.15027)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Hasan, U., Hossain, E., & Gani, M. O. (2023). A Survey on Causal Discovery Methods for I.I.D. and Time Series Data. arXiv preprint arXiv:2303.15027.

# A Survey on Causal Discovery Methods for I.I.D. and Time Series Data

## Overview

This survey unifies causal discovery (CD) algorithms for both I.I.D. (independent and identically distributed) data and time series data under one framework. Unlike earlier surveys that address only one setting, this work organizes methods into five algorithm families: constraint-based, score-based, functional causal model (FCM)-based, gradient-based, and prior-knowledge-augmented approaches. The target audience includes researchers and practitioners in machine learning, statistics, epidemiology, economics, and climate science who need to infer causal structure from observational data without performing randomized controlled trials.

## Fundamental Concepts

### Causal Graph Representations

A causal graph is a Directed Acyclic Graph (DAG) $\mathcal{G} = (V, E)$ where nodes $V = \{X_1, \ldots, X_d\}$ represent random variables and directed edges $E$ represent direct causal relationships. Three canonical building blocks determine the flow of information:

| Building Block | Structure | Independence Property |
|---|---|---|
| Chain | $X \to Y \to Z$ | $X \perp\!\!\!\perp Z \mid Y$ |
| Fork | $X \leftarrow Y \rightarrow Z$ | $X \perp\!\!\!\perp Z \mid Y$ |
| Collider | $X \to Z \leftarrow Y$ | $X \not\perp\!\!\!\perp Z \mid Z$ (conditioning on $Z$ opens the path) |

**d-separation** formalizes when a set of nodes $Z$ blocks all paths between node sets $X$ and $Y$ in $\mathcal{G}$, making $X \perp\!\!\!\perp Y \mid Z$ in the distribution. This is the basis for constraint-based algorithms: they test conditional independencies in data and use d-separation to recover edges.

### Structural Equation Models (SEMs)

Each variable $X_i$ is expressed as a function of its direct causes (parents $\text{PA}_{X_i}$) plus an independent noise term:

$$X_i = f_i(\text{PA}_{X_i}) + E_i, \quad E_i \perp\!\!\!\perp E_j \; \forall i \neq j$$

For the linear case: $X_i = \sum_{j: X_j \in \text{PA}_{X_i}} b_{ij} X_j + E_i$, where $b_{ij}$ is the causal strength. The full system can be written in matrix form as $\mathbf{X} = B^\top \mathbf{X} + \mathbf{E}$, where $B \in \mathbb{R}^{d \times d}$ is the weighted adjacency matrix with $B_{ij} \neq 0$ iff $X_j \to X_i$.

### Standard Assumptions

1. **Acyclicity**: No directed cycles in $\mathcal{G}$ (DAG constraint).
2. **Causal Markov Condition**: Each variable is conditionally independent of its non-descendants given its parents.
3. **Causal Faithfulness**: Every conditional independence in the distribution corresponds to a d-separation in $\mathcal{G}$ (no cancellations of paths).
4. **Causal Sufficiency**: No hidden common causes (no latent confounders).

Markov equivalence classes (MECs) are sets of DAGs that encode the same conditional independencies; most algorithms without additional assumptions recover only the MEC, represented as a Completed Partially Directed Acyclic Graph (CPDAG).

## Algorithm Families for I.I.D. Data

### Constraint-Based Methods

**Input**: Observational samples $\mathbf{X} \in \mathbb{R}^{n \times d}$ ($n$ samples, $d$ variables).
**Output**: CPDAG or PAG (Partial Ancestral Graph) representing the MEC.

These methods enumerate conditional independence (CI) tests to determine the skeleton, then orient edges using v-structure patterns.

**PC Algorithm** (Peter-Clark):
1. Start with a complete undirected graph.
2. For each pair $(X_i, X_j)$, test $X_i \perp\!\!\!\perp X_j \mid S$ for increasing subset sizes $|S|$; remove the edge if independence is found.
3. Orient v-structures: if $X_i - X_k - X_j$ with $X_i \not\sim X_j$ and $X_k \notin S_{ij}$, orient as $X_i \to X_k \leftarrow X_j$.
4. Apply Meek rules to orient remaining edges without creating new v-structures or cycles.

**Complexity concern**: Number of CI tests grows exponentially with degree of the graph; algorithms like **RFCI** (Really Fast Causal Inference) reduce this by performing fewer tests using orientation rules earlier.

**FCI** (Fast Causal Inference): Relaxes causal sufficiency. It outputs a PAG that explicitly represents possible latent confounders and selection bias using circle endpoints ($\circ$) instead of arrowheads.

### Score-Based Methods

**Input**: $\mathbf{X} \in \mathbb{R}^{n \times d}$.
**Output**: DAG maximizing a decomposable scoring function.

Instead of CI tests, these methods assign a score $S(\mathcal{G}, \mathbf{X})$ to each DAG and search for the highest-scoring one. Common scores:
- **BIC**: $S_{\text{BIC}}(\mathcal{G}) = \log P(\mathbf{X} \mid \hat{\theta}, \mathcal{G}) - \frac{k}{2} \log n$, where $k$ is the number of parameters.

**GES** (Greedy Equivalence Search):
1. **Forward phase**: Start from empty graph; greedily add edges that maximally increase score, staying within CPDAG space.
2. **Backward phase**: Greedily remove edges that maximally increase score.

**FGS** (Fast Greedy Search): A parallelized variant of GES that achieves scalability to 1,000,000 variables on sparse models in under 13 minutes.

**RL-BIC**: Uses reinforcement learning (policy gradient) to navigate DAG space, with BIC as the reward signal. Treats graph search as a sequence decision problem.

**Triplet A\***: Outperforms GES empirically for large dense networks by using a more informed heuristic.

### Functional Causal Model (FCM)-Based Methods

**Input**: $\mathbf{X} \in \mathbb{R}^{n \times d}$.
**Output**: Full DAG (not just MEC), by exploiting asymmetry between cause and effect.

FCMs achieve full identifiability by restricting the functional form $f_i$ and the noise distribution $E_i$.

**LiNGAM** (Linear Non-Gaussian Acyclic Model):

*Assumption*: $f_i$ is linear and $E_i$ is non-Gaussian (with non-zero variance). No latent confounders.

*Key insight*: In a bivariate linear Gaussian model, $X \to Y$ and $Y \to X$ are statistically indistinguishable. With non-Gaussian noise, only the true causal direction yields a residual that is independent of the predictor.

Algorithm (ICA-based):
1. Compute the ICA decomposition of $\mathbf{X}$: find mixing matrix $A$ such that $\mathbf{X} = A \mathbf{S}$ with independent components $\mathbf{S}$.
2. Identify permutation $P$ such that $W = PA^{-1}$ has no zeros on its diagonal.
3. The lower-triangular $W$ (after row/column permutation matching causal order $k(\cdot)$) gives $B = I - W$.

$$x_i = \sum_{k(j) < k(i)} b_{ij} x_j + e_i$$

**DirectLiNGAM**: Avoids the ICA step by successively identifying the root variable (the one whose residuals after regression are independent of all predictors) and reducing the problem dimension by one at each step. More robust than the ICA variant.

**ANM** (Additive Noise Model):

$$X_j = f_j(X_i) + E_j \quad \text{(bivariate case)}$$

$X_i \to X_j$ is identified if $E_j = X_j - f_j(X_i)$ is independent of $X_i$ (tested via HSIC), but $X_i - g(X_j)$ is NOT independent of $X_j$ for any $g$. Generalizes LiNGAM to nonlinear $f_j$.

**CAREFL** (Causal Autoregressive Flow):
Uses normalizing flows (autoregressive) to model $P(X_j \mid X_i)$ and $P(X_i)$ jointly; the causal direction is the one that factorizes better under the flow model.

### Gradient-Based Methods

**Input**: $\mathbf{X} \in \mathbb{R}^{n \times d}$.
**Output**: Weighted adjacency matrix $W \in \mathbb{R}^{d \times d}$ representing the DAG.

These methods reformulate the combinatorial DAG search as continuous optimization.

**NOTEARS** (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning):

*Core innovation*: The acyclicity constraint on $W$ is expressed as a smooth equality constraint:

$$h(W) = \text{tr}(e^{W \circ W}) - d = 0 \iff W \text{ is a DAG}$$

where $\circ$ denotes element-wise (Hadamard) product and $d$ is the number of nodes. This follows because $[e^{W \circ W}]_{ii} = 1 + \sum_{k} (W \circ W)^k_{ii}/k!$ counts the number of cycles through node $i$; the sum equals $d$ iff no cycles exist.

Optimization problem:

$$\min_{W} F(W) = \frac{1}{2n} \|X - XW\|_F^2 + \lambda \|W\|_1 \quad \text{s.t. } h(W) = 0$$

Solved via augmented Lagrangian method:

$$\min_W F(W) + \frac{\rho}{2} h(W)^2 + \alpha h(W), \quad \text{then update } \alpha \leftarrow \alpha + \rho h(W)$$

**GOLEM**: Replaces the least-squares objective with a log-likelihood:

$$S(B; \mathbf{X}) = \mathcal{L}(B; \mathbf{X}) + \lambda_1 R_{\text{sparse}}(B) + \lambda_2 R_{\text{DAG}}(B)$$

Uses a soft DAG penalty $R_{\text{DAG}}(B) = h(B)$ rather than a hard constraint. Outperforms NOTEARS on denser graphs by requiring fewer optimization iterations.

**GraN-DAG**: Replaces $W$ with a neural network $f_\theta$ per variable, enabling nonlinear relationships. Defines acyclicity on the induced graph where edge $(j, i)$ exists iff $\partial f_{\theta,i}/\partial x_j \neq 0$.

**DAG-GNN**: Uses a Variational Autoencoder (VAE) with an alternative acyclicity formulation:

$$h'(W) = \text{tr}[(I + \alpha W \circ W)^d] - d = 0$$

where $\alpha > 0$ is a small constant. Computationally cheaper than the matrix exponential.

**DAG-NoCurl**: Enforces acyclicity implicitly via Hodge decomposition of the graph, projecting gradients onto the space of cycle-free updates without an explicit constraint.

## Algorithm Families for Time Series Data

### Granger Causality

**Core idea**: Variable $X_j$ Granger-causes $X_i$ if past values of $X_j$ significantly improve prediction of $X_i$ beyond using only past values of $X_i$.

For a Vector Autoregressive (VAR) model of order $p$:

$$X_i(t) = \sum_{\tau=1}^{p} \sum_{j=1}^{d} A^{(\tau)}_{ij} X_j(t-\tau) + E_i(t)$$

where $A^{(\tau)} \in \mathbb{R}^{d \times d}$ are coefficient matrices at lag $\tau$. The causal graph has edge $X_j \to X_i$ iff $\exists \tau: A^{(\tau)}_{ij} \neq 0$, tested via F-test or likelihood-ratio test.

**Limitation**: Granger causality detects predictive causality, not interventional causality; it fails under hidden confounders.

### PCMCI (PC with Momentary Conditional Independence)

**Developed by**: Runge et al. (2019).

**Input**: Multivariate time series $\mathbf{X}(t) \in \mathbb{R}^d$ for $t = 1, \ldots, T$.
**Output**: Time-lagged causal graph with edges $X_j(t-\tau) \to X_i(t)$.

Algorithm:
1. **PC phase**: Apply PC-style skeleton discovery using lagged conditional independence tests to identify parents $\hat{\text{PA}}_i$ for each $X_i(t)$.
2. **MCI phase**: For each candidate edge $(X_j(t-\tau), X_i(t))$, test:
   $$X_j(t-\tau) \perp\!\!\!\perp X_i(t) \mid \hat{\text{PA}}_i(t), \hat{\text{PA}}_j(t-\tau)$$
   This conditioning on both parents makes the test robust to autocorrelation.

PCMCI controls false discovery rate better than standard Granger methods under autocorrelation.

### TiMINo (Time Series Models with Independence Noise)

Extends the FCM framework to time series. Models:

$$X_i(t) = f_i(\text{PA}_{X_i}(t), \text{PA}_{X_i}(t-1), \ldots) + E_i(t)$$

where $E_i(t) \perp\!\!\!\perp E_j(t)$ for all $i \neq j$ and $E_i(t) \perp\!\!\!\perp E_i(t')$ for $t \neq t'$. Causal direction is identified by finding the assignment where residuals are independent.

### DYNOTEARS

Extends NOTEARS to dynamic (time-lagged) settings. Augments the adjacency matrix to include lagged edges:

$$\mathbf{X}(t) = \mathbf{X}(t) W^{(0)} + \sum_{\tau=1}^{p} \mathbf{X}(t-\tau) W^{(\tau)} + E(t)$$

The acyclicity constraint applies only to $W^{(0)}$ (contemporaneous edges). Lagged edges $W^{(\tau)}$ for $\tau \geq 1$ are unconstrained. Solved with the same augmented Lagrangian framework as NOTEARS.

## Evaluation Metrics

| Metric | Description |
|---|---|
| SHD (Structural Hamming Distance) | Number of edge additions, deletions, reversals needed to convert predicted to true DAG |
| SID (Structural Intervention Distance) | Number of pairs $(i,j)$ where the predicted graph gives a wrong set of adjustment variables for $do(X_i)$ |
| Precision / Recall | For edge prediction: precision = TP/(TP+FP), recall = TP/(TP+FN) |
| F1 Score | Harmonic mean of precision and recall |

SID is often more informative than SHD because it measures intervention quality, not just structural correctness.

## Software Tools

| Tool | Language | Scope |
|---|---|---|
| CDT (Causal Discovery Toolbox) | Python | I.I.D.; integrates many algorithms |
| gCastle | Python | I.I.D. and time series; gradient-based focus |
| Tetrad Project | Java | Constraint-based and score-based; GUI available |
| tigramite | Python | Time series; PCMCI and variants |

## Experiments

- **Dataset**: Synthetic datasets generated from linear and nonlinear SEMs with varying graph sizes ($d = 10, 20, 50, 100$ nodes) and sample sizes ($n = 100$ to $10^6$). Erd\H{o}s-Rényi and scale-free graph topologies used.
- **Benchmarks**: Sachs protein signaling network (real $d=11$ node dataset with known ground truth from interventional data).
- **Results summary**:
  - FGS scales to $d = 1{,}000{,}000$ sparse variables; GES fails above $d = 50{,}000$.
  - GOLEM outperforms NOTEARS on dense graphs (average degree $> 2$) with fewer augmented Lagrangian iterations.
  - GAE achieves near-linear training time up to $d = 100$ nodes, average under 2 minutes.
  - Triplet A\* outperforms GES on large dense networks.
  - PCMCI achieves better FDR control than Granger causality under autocorrelated noise.

## Comparison: Key Algorithmic Differences

| Method | Functional Form | Noise | Identifiability | Latent Confounders | Data Type |
|---|---|---|---|---|---|
| PC / GES | Any (nonparametric) | Any | MEC only | No (causal sufficiency) | I.I.D. |
| FCI | Any | Any | PAG (partial) | Yes | I.I.D. |
| LiNGAM | Linear | Non-Gaussian | Full DAG | No | I.I.D. |
| ANM | Nonlinear additive | Any | Full DAG | No | I.I.D. |
| NOTEARS | Linear | Gaussian | Full DAG | No | I.I.D. |
| GraN-DAG | Nonlinear (NN) | Any | Full DAG | No | I.I.D. |
| Granger | Linear VAR | Any | Predictive only | No | Time series |
| PCMCI | Nonparametric | Any | MEC | Partial | Time series |
| DYNOTEARS | Linear VAR | Gaussian | Full DAG | No | Time series |
| TiMINo | Nonlinear | Independent noise | Full DAG | No | Time series |

> [!IMPORTANT]
> Constraint-based methods (PC, FCI) recover only the Markov equivalence class (CPDAG/PAG), not the full DAG. FCM-based and gradient-based methods achieve full DAG identifiability under their specific assumptions. Choosing between them requires knowing the data-generating mechanism's characteristics.

> [!NOTE]
> "Causal discovery algorithms can be broadly categorized into constraint-based, score-based, functional causal model-based, gradient-based, and approaches that incorporate prior knowledge." — Survey paper

> [!TIP]
> For practical entry points: CDT and gCastle provide Python interfaces to most methods discussed. Tetrad offers a GUI for constraint-based and score-based algorithms. For time series specifically, the `tigramite` package implements PCMCI.

> [!CAUTION]
> The acyclicity assumption (DAG) excludes feedback loops, which are common in real systems (e.g., biological regulatory networks, economic equilibria). Methods that handle cyclic causal structures (e.g., cyclic SEMs) are beyond the scope of this survey and require separate treatment.
