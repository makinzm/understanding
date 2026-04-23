# Meta Information

- URL: [Scalable Neural Contextual Bandit for Recommender Systems](https://arxiv.org/abs/2306.14834)
- LICENSE: [ Deed - Attribution 4.0 International - Creative Commons ]( https://creativecommons.org/licenses/by/4.0/ )
- Reference: Zhu, Z., & Van Roy, B. (2023). Scalable Neural Contextual Bandit for Recommender Systems. arXiv:2306.14834.

# Scalable Neural Contextual Bandit for Recommender Systems

## Overview

Traditional recommender systems rely on supervised learning that only **exploits** known user preferences, never **exploring** potentially better content for users whose interests are not fully captured in the training data.
This paper proposes **Epistemic Neural Recommendation (ENR)**, a scalable contextual bandit algorithm that balances exploration and exploitation via Thompson sampling with epistemic neural networks (EpiNet).

ENR achieves at least a **9% CTR improvement** on MIND and **6% rating improvement** on KuaiRec compared to supervised baselines, while requiring **orders of magnitude fewer compute resources** than prior neural contextual bandit methods.

> [!NOTE]
> "ENR significantly boosts click-through rates and user ratings by at least 9% and 6% respectively, achieves equivalent performance with at least 29% fewer user interactions, and demands orders of magnitude fewer computational resources than neural contextual bandit baseline algorithms."

## Problem Formulation: Contextual Bandit

A recommender system operates over discrete timesteps $t = 1, 2, \ldots$. At each step:

- **Context** $\psi_{S_t} \in \mathbb{R}^{d_S}$: user state (history, demographics, session features)
- **Action set** $\mathcal{A}$: candidate items, each with feature vector $\phi_a \in \mathbb{R}^{d_A}$
- **Action** $A_t \in \mathcal{A}$: the item recommended to the user
- **Reward** $R_{t+1} \in \{0, 1\}$ (click) or $R_{t+1} \in \mathbb{R}$ (rating): user feedback observed after recommendation

The agent's objective is to maximize cumulative reward $\sum_t R_{t+1}$, which requires trading off exploitation (recommending items with known high reward) and exploration (recommending items with uncertain but potentially high reward).

> [!TIP]
> The contextual bandit formulation generalizes multi-armed bandits by conditioning on a context at each timestep. See [Lattimore & Szepesvári (2020), *Bandit Algorithms*](https://tor-lattimore.com/downloads/book/book.pdf) for background.

## ENR Architecture

ENR consists of three components: an informative representation layer, an epistemic uncertainty estimation module (EpiNet), and a Thompson sampling decision rule.

### 1. Informative Neural Representation Layer

Raw context $\psi_{S_t} \in \mathbb{R}^{d_S}$ and action features $\phi_A \in \mathbb{R}^{d_A}$ are embedded separately, then combined:

**Context summarization** ($\beta_\text{context} \in \mathbb{R}^{d_S \times d_E}$):

```math
\begin{align}
  h_{\beta_\text{context}}(\psi_{S_t}) = \text{LayerNorm}(\text{ReLU}(\beta_\text{context}^\top \psi_{S_t})) \in \mathbb{R}^{d_E}
\end{align}
```

**Action summarization** ($\beta_\text{action} \in \mathbb{R}^{d_A \times d_E}$):

```math
\begin{align}
  h_{\beta_\text{action}}(\phi_A) = \text{LayerNorm}(\text{ReLU}(\beta_\text{action}^\top \phi_A)) \in \mathbb{R}^{d_E}
\end{align}
```

**Context-action interaction** (element-wise product):

```math
\begin{align}
  I(\psi_{S_t}, \phi_A) = h_{\beta_\text{context}}(\psi_{S_t}) \odot h_{\beta_\text{action}}(\phi_A) \in \mathbb{R}^{d_E}
\end{align}
```

**Combined representation** (concatenation):

```math
\begin{align}
  x_{t,A} = \text{concat}(h_{\beta_\text{context}}(\psi_{S_t}),\; h_{\beta_\text{action}}(\phi_A),\; I(\psi_{S_t}, \phi_A)) \in \mathbb{R}^{3d_E}
\end{align}
```

> [!NOTE]
> Layer Normalization is critical for online learning because incoming feature distributions shift continuously as user behavior evolves. Without it, training becomes unstable when new users or items with very different feature scales arrive in the stream.

### 2. Epistemic Neural Network (EpiNet)

ENR's uncertainty estimation follows the EpiNet design: a small auxiliary network is appended to the main predictor to produce epistemic (model) uncertainty without requiring matrix inversions or full posterior inference.

The full prediction given context-action pair and epistemic index $z \in \mathbb{R}^{d_z}$ is:

```math
\begin{align}
  f_\theta(\psi_{S_t}, \phi_A, z) = f_{\theta_x}(x_{t,A}) + \left(g_\sigma(\text{sg}[x_{t,A}], z) + g_{\sigma}^p(\text{sg}[x_{t,A}], z)\right)^\top z
\end{align}
```

| Symbol | Description |
|--------|-------------|
| $f_{\theta_x}: \mathbb{R}^{3d_E} \to \mathbb{R}$ | Main prediction network (point estimate) |
| $g_\sigma: \mathbb{R}^{3d_E} \times \mathbb{R}^{d_z} \to \mathbb{R}^{d_z}$ | Learnable epistemic network |
| $g_\sigma^p: \mathbb{R}^{3d_E} \times \mathbb{R}^{d_z} \to \mathbb{R}^{d_z}$ | Fixed prior network (Glorot-initialized, frozen) |
| $z \sim P_z$ | Epistemic index sampled from prior distribution (e.g., $\mathcal{N}(0, I_{d_z})$) |
| $\text{sg}[\cdot]$ | Stop-gradient operator: prevents gradients from flowing through the bracketed term |

The learned network $g_\sigma$ adjusts its outputs to reduce epistemic uncertainty as more data is observed, while the fixed prior $g_\sigma^p$ ensures epistemic perturbation is always present even with abundant data, preventing premature certainty.

> [!IMPORTANT]
> The stop-gradient on $x_{t,A}$ when passed to the epistemic heads prevents the representation layer from being corrupted by epistemic training objectives. Only $f_{\theta_x}$ trains the representation end-to-end.

### 3. Thompson Sampling Decision Rule

At each timestep $t$, one epistemic index $z$ is sampled, and the item with the highest predicted reward is selected:

```math
\begin{align}
  A_t = \arg\max_{a \in \mathcal{A}}\; f_\theta(\psi_{S_t}, \phi_a, z), \quad z \sim P_z
\end{align}
```

By fixing one $z$ per timestep, ENR commits to a single "hypothesis" about the reward function, which implements the Thompson sampling exploration strategy implicitly — items with high uncertainty will sometimes appear best under certain $z$ samples, naturally inducing exploration.

## Training Algorithm

ENR maintains a replay buffer $\mathcal{D}$ of past (context, action, reward) tuples and trains online with mini-batch gradient descent.

**Algorithm: ENR Thompson Sampling**

```
Initialize: θ (network parameters), P_z (prior over z), D = {} (replay buffer)

For t = 1, 2, ...:
  1. Observe context ψ_{S_t}
  2. Sample z ~ P_z
  3. Compute f_θ(ψ_{S_t}, φ_a, z) for all a ∈ A_t
  4. Select A_t = argmax_a f_θ(ψ_{S_t}, φ_a, z)
  5. Observe reward R_{t+1}, add (ψ_{S_t}, A_t, R_{t+1}) to D
  6. For k = 1, ..., K gradient steps:
       Sample minibatch B from D
       Sample Z = {z_1, ..., z_m} ~ P_z
       θ ← θ - α ∇_θ Σ_{i∈B} Σ_{z∈Z} L(R_i, f_θ(ψ_{S_i}, φ_{A_i}, z))
```

The loss $\mathcal{L}$ is:
- **Cross-entropy** for binary rewards (clicks): $\mathcal{L}(r, \hat{r}) = -r \log \hat{r} - (1-r) \log(1-\hat{r})$
- **Mean squared error** for continuous rewards (ratings): $\mathcal{L}(r, \hat{r}) = (r - \hat{r})^2$

## Comparison with Related Methods

| Method | Exploration Strategy | Covariance Matrix Size | Inference Time | Scalable? |
|--------|---------------------|----------------------|----------------|-----------|
| LinUCB | UCB on linear model | $d \times d$ (small) | Fast | Yes |
| NeuralUCB (full) | UCB with full network gradient | $10^6 \times 10^6$ | Intractable | No |
| NeuralTS (full) | Thompson sampling, full Laplace | $10^7 \times 10^7$ | Intractable | No |
| NeuralLinUCB (last layer) | UCB on last-layer features | $d_E \times d_E$ | 6–100× slower | Marginal |
| Ensemble sampling | Multiple network copies | N/A | Similar to ENR | Marginal |
| **ENR** | Thompson sampling via EpiNet | None (no inversion) | Fastest | **Yes** |

> [!NOTE]
> Full NeuralUCB and NeuralTS require inverting matrices of size $10^6 \times 10^6$ or $10^7 \times 10^7$, which is both memory-intractable and computationally infeasible in production recommender systems.

**Key design advantages of ENR over prior neural bandits:**
- No matrix inversion: uncertainty is parameterized via the EpiNet auxiliary network
- Single forward pass per action candidate: $O(|\mathcal{A}|)$ inference cost
- Ensemble sampling achieves comparable inference speed but requires 5× the training time due to maintaining multiple full network copies

## Ablation Studies

| Design Choice | Finding |
|--------------|---------|
| Layer Normalization | Essential; removal causes training divergence in online settings |
| Epistemic index width $d_z$ | Optimal at $d_z = 5$; too large degrades stability |
| Prior network scale | Best at scale $0.3$; too small loses exploration, too large corrupts learning |

## Experiments

- **Datasets**:
  - **MIND** (Microsoft News Dataset): ~1,000,000 users, ~160,000 news articles, ~15,000,000 interactions; binary reward (click/no-click)
  - **KuaiRec**: 1,411 users, 3,327 short videos, 4,676,570 interactions; continuous reward (user rating 1–5)
- **Hardware**: Not specified
- **Optimizer**: Adam with learning rate $\alpha$; $K$ gradient steps per timestep
- **Results**:
  - MIND: ENR CTR = 0.220 vs. best baseline 0.206 (+7% absolute, +9% relative); 56% fewer interactions to match supervised baseline performance
  - KuaiRec: ENR rating = 2.63 vs. best baseline 2.46 (+6%); 29% fewer interactions needed
  - Inference: ENR ~0.0009s per action vs. 0.01–1.6s for NeuralLinUCB variants

## Applicability

ENR is applicable when:

- **Who**: ML engineers and researchers building production recommendation pipelines that require online learning
- **When**: The item catalog and user base are large enough that full covariance matrix methods become computationally intractable
- **Where**: Content recommendation (news, video, e-commerce) where user feedback arrives as a sequential stream and the system must balance exploration of novel content with exploitation of known preferences

**Limitations**:
- Focused on pure online learning; no offline pretraining phase is considered
- Reward model assumes binary or scalar feedback; does not address multi-objective or listwise reward settings
- Feedback timing assumptions may not hold when there is a delay between recommendation and reward observation
