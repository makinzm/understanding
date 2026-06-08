# Meta Information

- URL: [Scaling Laws Across Model Architectures: A Comparative Analysis of Dense and MoE Models in Large Language Models](https://arxiv.org/abs/2410.05661)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Siqi Wang, Zhengyu Chen, Bei Li, Keqing He, Min Zhang, Jingang Wang (2024). Scaling Laws Across Model Architectures: A Comparative Analysis of Dense and MoE Models in Large Language Models. arXiv:2410.05661.

# Overview

This paper investigates whether the power-law scaling frameworks established for Dense Transformer LLMs also apply to Mixture of Experts (MoE) models, and whether optimal hyperparameters (batch size, learning rate) transfer between the two architectures. The authors extend the Chinchilla scaling law to incorporate a new variable — the number of experts $E$ — and validate the unified framework through experiments ranging from 50M to 7B parameters on over 100B tokens.

**Who benefits from this work:** ML practitioners deciding how to allocate compute budgets (more parameters vs. more data vs. more experts) when training large models, and researchers studying the fundamental efficiency differences between dense and sparse architectures.

## Architectures Compared

| Property | Dense Model | MoE Model |
|---|---|---|
| Activated parameters per token | All $N$ | $N / E$ (top-$k$ routing) |
| Total parameters | $N$ | $N \cdot E$ (approx.) |
| Training FLOPs for $D$ tokens | $\approx 6ND$ | $\approx 6(N/E)D$ per active path |
| Data efficiency vs. dense (same FLOPs) | baseline | +16.37% |

MoE models route each token to a subset of $k$ experts (typically $k=2$ out of $E=8$), so active parameter count per forward pass is much lower than total parameter count. The key question is how compute-optimal scaling laws change when $E$ is a free variable.

# Scaling Law Framework

## Unified Loss Equation

The paper proposes a single parametric form covering both architectures:

```math
\begin{align}
  \hat{L}(N, D, E) = \frac{A}{N^\alpha E^\gamma} + \frac{B}{D^\beta} + \sigma
\end{align}
```

- $N$: model scale (non-embedding parameter count), $N \in \mathbb{R}^+$
- $D$: number of training tokens, $D \in \mathbb{R}^+$
- $E$: number of experts (set to 1 for dense models), $E \in \mathbb{Z}^+$
- $A, B, \alpha, \beta, \gamma, \sigma$: fitted constants
- $\sigma$: irreducible loss (entropy of the data distribution)

When $E = 1$, the equation reduces to the standard Chinchilla form $A/N^\alpha + B/D^\beta + \sigma$.

> [!NOTE]
> The formula simplifies to this two-term form when expert count $E < 100$, a regime covering virtually all practical MoE architectures at the time of writing.

## Fitting the Constants

The authors fit $\{A, B, \alpha, \beta, \gamma, \sigma\}$ separately for dense and MoE families by minimizing Huber loss over a grid of (N, D) configurations. Key fitted values:

| Constant | Dense | MoE |
|---|---|---|
| $\alpha$ (model scale exponent) | 0.507 | 0.590 |
| $\beta$ (data exponent) | 0.493 | 0.410 |
| $\gamma$ (expert exponent) | — | ~0.13 |

The higher $\alpha$ for MoE means each doubling of active parameters yields a larger loss reduction than for dense models. Conversely, the lower $\beta$ means MoE benefits less from more data per unit compute than dense models do — a consequence of being more parameter-efficient.

## Compute-Optimal Allocation

For a fixed compute budget $C \approx 6ND$ (in FLOPs), the optimal split between $N^*$ and $D^*$ is derived by taking the partial derivative of $\hat{L}$ and setting it to zero. This yields:

```math
\begin{align}
  \frac{N^*}{D^*} \propto \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}
\end{align}
```

For **dense models** ($\alpha_N \approx \alpha_D \approx 0.5$), the ratio $N^*/D^*$ is nearly equal — matching the Chinchilla 1:1 rule. For **MoE models** ($\alpha_N = 0.59 > \alpha_D = 0.41$), practitioners should allocate more compute to increasing model scale (more experts or larger expert FFNs) relative to training tokens.

# Hyperparameter Scaling

## Optimal Batch Size

Following McCandlish et al. (2018), the noise scale $B_\text{noise}$ determines the optimal batch size at each loss level:

```math
\begin{align}
  B_\text{noise} = \frac{\text{tr}(H\Sigma)}{G^T H G}
\end{align}
```

where $H$ is the true Hessian, $\Sigma$ is the per-sample gradient covariance, and $G$ is the gradient mean. A lower noise scale means smaller batches suffice for reliable gradient estimates. The empirical fit is:

```math
\begin{align}
  B_\text{opt} \approx \frac{\lambda_B}{L^{\alpha_B}}
\end{align}
```

**Finding:** MoE models have a lower noise scale than dense models at the same loss value, so their $B_\text{opt}$ is smaller. At equivalent loss, MoE optimal batch sizes are consistently lower (~65.8% interval overlap between architectures).

## Optimal Learning Rate

Similarly, optimal learning rate scales with loss:

```math
\begin{align}
  \eta_\text{opt} \approx \lambda_\eta \cdot L^{\alpha_\eta}
\end{align}
```

**Finding:** MoE models support larger learning rates at the same loss level than dense models (~76.2% interval overlap). This is attributed to more efficient gradient estimation in sparse activation patterns, which reduces the risk of divergence at higher step sizes.

**Practical implication:** When transferring hyperparameters from a dense pilot run to a MoE production run, reduce batch size slightly and increase learning rate slightly. The paper's fitted exponents quantify by how much.

# Experimental Setup

## Pseudocode: Scaling Experiment Protocol

```
1. Sample (N, D) pairs uniformly in log-space from [50M, 7B] × [1B, 100B]
2. For each pair:
   a. Initialize model (Dense or MoE with E=8 experts, top-2 routing)
   b. Train on The Pile with AdamW, cosine LR schedule
   c. Record final validation loss L(N, D)
3. Fit scaling law constants via Huber regression on {(N, D, L)} triples
4. Predict L at held-out (N, D) pairs; measure relative error
5. Compare optimal token allocation αD/αN ratios between architectures
```

# Experiments

- **Dataset:** The Pile (825 GiB English text; Gao et al., 2020). Training runs use 1B–100B token subsets.
- **Dense model sizes:** 50M, 200M, 500M, 1B, 3B, 7B parameters
- **MoE model sizes:** 200M–3B active parameters; 8 experts, top-2 routing
- **Hardware:** Not specified in the paper
- **Optimizer:** AdamW with cosine learning rate schedule
- **Evaluation benchmarks:** TriviaQA, MATH, MMLU, CMMLU, MATH401
- **Key result:** MoE models achieve 16.37% better data efficiency at equivalent FLOPs. The unified scaling law predicts held-out loss with <2% relative error across both architectures.

# Differences from Prior Work

| Paper | Scope | Key gap addressed here |
|---|---|---|
| Kaplan et al. (2020) | Dense only; $L(N,D)$ | No expert variable $E$; suboptimal token/param ratio |
| Hoffmann et al. / Chinchilla (2022) | Dense only; corrects Kaplan | Still no MoE; equal $\alpha_N = \alpha_D$ assumption |
| Clark et al. (2022) | MoE scaling of expert count | Fixed $N$; no unified $(N,D,E)$ framework |
| **This paper** | Dense + MoE unified | First to fit $\hat{L}(N,D,E)$ jointly; hyperparameter transfer analysis |

> [!IMPORTANT]
> The paper finds that the Chinchilla 1:1 token-to-parameter ratio is suboptimal for MoE: MoE should be parameter-heavier (higher $\alpha_N$) relative to training tokens. This directly affects how practitioners should set up compute-optimal MoE training runs.

> [!CAUTION]
> Experiments are limited to $E \leq 8$ experts and models up to 7B parameters. The authors observe signs of overtraining beyond 100B tokens. Extrapolation to very large expert counts (e.g., $E \geq 64$) or trillion-token regimes should be treated cautiously.
