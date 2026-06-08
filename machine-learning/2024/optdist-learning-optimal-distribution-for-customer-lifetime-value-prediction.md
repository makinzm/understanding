# Meta Information

- URL: [OptDist: Learning Optimal Distribution for Customer Lifetime Value Prediction](https://arxiv.org/abs/2408.08585)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Weng, Y., Tang, X., Xu, Z., Lyu, F., Liu, D., Sun, Z., & He, X. (2024). OptDist: Learning Optimal Distribution for Customer Lifetime Value Prediction. Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24).

# OptDist: Learning Optimal Distribution for Customer Lifetime Value Prediction

## Background: CLTV Prediction Challenges

Customer Lifetime Value (CLTV) prediction is a fundamental task in e-commerce and financial services, aiming to predict the total future revenue a user will generate. Two structural properties make this difficult to model with a single distribution:

1. **Zero-inflation**: A large fraction of users never make a purchase (zero value). On the Criteo-SSC dataset, only 7.20% of samples have a positive label.
2. **Heavy tail**: High-value users spend orders of magnitude more than a typical user, producing a right-skewed long-tail distribution.

Prior work often assumes a single parametric distribution for all users (e.g., ZILN) or manually partitions users into predefined buckets for group-specific models. These approaches either sacrifice flexibility or require brittle manual tuning.

## Problem Formulation

Given a user feature vector $x_u \in \mathbb{R}^d$, the goal is to predict CLTV $\hat{y}_u \geq 0$. The prediction is drawn from a learned distribution $D$ parameterized by $\Theta$:

```math
\begin{align}
  \theta^\star = \arg\min_{\pi_u, \Theta} \mathcal{L}_u(f(x_u \mid D, \pi_u \odot \Theta), y_u)
\end{align}
```

where $\pi_u \in \{0,1\}^L$ is a one-hot selection vector over $L$ candidate sub-distributions and $\Theta = \{\theta_1, \ldots, \theta_L\}$ is the set of sub-distribution parameters. The constraint $\sum_i \pi_{u,i} = 1$ ensures exactly one sub-distribution is chosen per sample.

## Architecture Overview

OptDist consists of two modules connected by an alignment mechanism:

| Module | Role | Input | Output |
|---|---|---|---|
| Distribution Learning Module (DLM) | Trains $L$ sub-distribution networks in parallel | $x_u \in \mathbb{R}^d$ | $L$ distributions with per-sample losses $\mathcal{L}_{u,i}$ |
| Distribution Selection Module (DSM) | Selects the best sub-distribution per sample | $x_u \in \mathbb{R}^d$ | Gumbel-Softmax weights $\pi_u \in \Delta^L$ |
| Alignment Mechanism | Generates pseudo-labels from DLM losses to supervise DSM | $\{\mathcal{L}_{u,i}\}_{i=1}^L$ | Hard label $y_u^p$ and soft label $y_u^\omega$ |

## Distribution Learning Module (DLM)

Each sub-distribution network $i \in \{1, \ldots, L\}$ is a separate MLP that takes user features $x_u$ and outputs parameters for a Zero-Inflated Log-Normal (ZILN) distribution:

```math
\begin{align}
  \theta_i = (p_i,\ \mu_i,\ \sigma_i)
\end{align}
```

- $p_i \in [0,1]$: estimated conversion probability (probability of non-zero purchase)
- $\mu_i \in \mathbb{R}$: mean of the log-normal component
- $\sigma_i > 0$: standard deviation of the log-normal component

The predicted CLTV from sub-distribution $i$ is:

```math
\begin{align}
  \hat{y}_{u,i} = p_i \cdot \exp\!\left(\mu_i + \frac{\sigma_i^2}{2}\right)
\end{align}
```

Each sub-distribution is trained with the negative log-likelihood loss of ZILN, denoted $\mathcal{L}_{u,i}$.

> [!NOTE]
> Each sub-distribution network has the same architecture and input but independently learns different distribution parameters. Diversity emerges naturally from different random initializations and from the DSM routing during training.

## Distribution Selection Module (DSM)

The DSM is an MLP that outputs logits $\alpha_u \in \mathbb{R}^L$ over the $L$ sub-distributions. To enable gradient flow through discrete selection during training, Gumbel-Softmax relaxation is applied:

```math
\begin{align}
  \pi_{u,i} = \frac{\exp\!\left((\log \alpha_{u,i} + g_{u,i}) / \tau\right)}{\sum_{j=1}^{L} \exp\!\left((\log \alpha_{u,j} + g_{u,j}) / \tau\right)}
\end{align}
```

where $g_{u,i} \sim \text{Gumbel}(0,1)$ is an i.i.d. noise sample and $\tau > 0$ is the temperature parameter.

- At **high temperature** ($\tau \to \infty$): $\pi_u$ approaches a uniform distribution (soft selection).
- At **low temperature** ($\tau \to 0$): $\pi_u$ approaches a one-hot vector (hard discrete selection).

During inference, Gumbel noise is removed and the sub-distribution with the highest $\alpha_{u,i}$ is selected deterministically (argmax).

## Alignment Mechanism

The DLM produces per-sample losses $\{\mathcal{L}_{u,i}\}_{i=1}^L$, revealing which sub-distribution fits each sample best. The alignment mechanism converts these into supervision signals for the DSM:

**Hard pseudo-label** (one-hot, points to best sub-distribution):

```math
\begin{align}
  y_u^p = \text{one\_hot}\!\left(\arg\min_i \mathcal{L}_{u,i}\right)
\end{align}
```

**Soft pseudo-label** (probability proportional to inverse loss):

```math
\begin{align}
  y_u^\omega = \text{softmax}(-\mathcal{L}_u)
\end{align}
```

The DSM is trained with a combined loss:

```math
\begin{align}
  \mathcal{L}_{\text{DSM}} = \mathcal{L}_{\text{focal-CE}}(\pi_u,\ y_u^p) + \lambda \cdot \text{KL}(\pi_u \,\|\, y_u^\omega)
\end{align}
```

where focal cross-entropy $\mathcal{L}_{\text{focal-CE}}$ emphasizes hard-to-classify samples, KL divergence aligns the DSM's output distribution with the soft loss landscape, and $\lambda$ balances the two terms.

## Training Procedure

The full training loop alternates between updating DLM and DSM:

```
Input: training data {(x_u, y_u)}, number of sub-distributions L, temperature τ, weight λ
Initialize: L DLM sub-networks {Net_1, ..., Net_L}, DSM network Net_sel

For each mini-batch:
  1. Forward pass through all L DLM sub-networks:
       θ_i = Net_i(x_u)  for i = 1..L
       Compute per-sample losses L_{u,i} for each sub-distribution

  2. Generate pseudo-labels from DLM losses:
       y_u^p = one_hot(argmin_i L_{u,i})
       y_u^ω = softmax(-L_u)

  3. Forward pass through DSM with Gumbel-Softmax:
       π_u = GumbelSoftmax(Net_sel(x_u), τ)

  4. Compute final prediction as weighted combination:
       ŷ_u = Σ_i π_{u,i} · ŷ_{u,i}

  5. Compute total loss:
       L_total = L_NLL(ŷ_u, y_u) + L_focal-CE(π_u, y_u^p) + λ · KL(π_u || y_u^ω)

  6. Backpropagate and update all parameters
```

> [!IMPORTANT]
> The pseudo-labels are generated on-the-fly from the current DLM state, so DSM supervision improves iteratively as DLM distributions become more refined.

## Comparison with Related Methods

| Method | Distribution | Selection strategy | Manual bucketing |
|---|---|---|---|
| ZILN | Single ZILN | None | No |
| Two-stage | MSE / binary | Hard threshold | Manual |
| MTL-MSE | MSE | Shared MTL head | No |
| MDME | Multiple distributions | Expert gating (soft) | No |
| MDAN | Multiple distributions | Pre-defined buckets | Yes |
| **OptDist** | Multiple ZILN | Gumbel-Softmax per sample | **No** |

Key distinctions:
- **vs. MDAN**: MDAN partitions users into predefined buckets (e.g., by spend quantile) offline and uses separate networks per bucket. OptDist learns the selection end-to-end, adapts to each sample, and supports incremental training without re-partitioning.
- **vs. MDME**: MDME uses soft MoE-style expert gating but does not explicitly align selection with which distribution minimizes loss. OptDist's alignment mechanism provides direct training signal for which sub-distribution is best for each sample.
- **vs. ZILN**: ZILN fits one set of distribution parameters across all users, which is suboptimal for heterogeneous CLTV distributions.

# Experiments

- **Datasets**:
  - Criteo-SSC: 15.99M samples, 7.20% positive ratio (public)
  - Kaggle CLTV: 805.75K samples, 90.12% positive ratio (public)
  - Industrial: 4.535M samples from a financial platform, 6.35% positive ratio (private)
- **Hardware**: NVIDIA Tesla V100-PCIe-32GB GPU, 128 GB RAM
- **Framework**: TensorFlow
- **Batch size**: 2048 (public datasets), 512 (industrial)
- **Embedding dim**: 5 (public), 12 (industrial)
- **Learning rate**: searched in $\{5\times10^{-4},\ 1\times10^{-3},\ 1.5\times10^{-3},\ 2\times10^{-3},\ 2.5\times10^{-3}\}$
- **Evaluation metrics**: MAE, Norm-GINI, Spearman's $\rho$ (on all samples and positive samples separately)
- **Results**:
  - OptDist reduced MAE vs. ZILN by 24.4% on Criteo-SSC (15.784 vs. 20.880), 2.2% on Kaggle (70.929 vs. 72.528), and 17.2% on Industrial (0.322 vs. 0.389)
  - Online A/B test on a financial marketing platform showed ROI improvements of +11.45% to +21.90% across four campaigns at the 30-day mark
  - Optimal number of sub-distributions $L$ is 4–5 across all datasets; performance degrades slightly with $L > 6$ due to redundancy
