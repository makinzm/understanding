# Meta Information

- URL: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Cheng, H.-T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., Anderson, G., Corrado, G., Chai, W., Ispir, M., Anil, R., Haque, Z., Hong, L., Jain, V., Liu, X., & Shah, H. (2016). Wide & Deep Learning for Recommender Systems. arXiv:1606.07792.

# Overview

Wide & Deep Learning is a framework for recommendation systems that jointly trains a **wide** (linear) component and a **deep** (neural network) component to balance **memorization** (learning co-occurrence patterns from historical data) and **generalization** (transferring to unseen feature combinations). It is deployed in Google Play app store recommendations, serving over 1 billion users and 1 million apps.

| Component | Role | Strengths | Weaknesses |
|---|---|---|---|
| Wide (linear) | Memorization via cross-product features | Captures explicit, specific patterns | Cannot generalize to unseen pairs |
| Deep (neural net) | Generalization via embeddings | Handles sparse, high-dimensional inputs | May over-generalize, miss niche patterns |
| Wide & Deep | Joint training of both | Balances both | Requires more careful feature engineering |

> [!NOTE]
> "Memorization can be loosely defined as learning the frequent co-occurrence of items or features and exploiting the correlation available in the historical data." — Cheng et al. (2016)

# Model Architecture

## Input

- **Sparse categorical features**: user-installed apps, impression app (the candidate app), user demographic features (country, language), contextual features (device type, day of week, hour of day) — all as integer IDs after vocabulary lookup
- **Dense continuous features**: app age, historical statistics, all normalized to $[0, 1]$

## Wide Component

The wide component is a generalized linear model:

```math
\begin{align}
y_{\text{wide}} = \mathbf{w}_{\text{wide}}^T [\mathbf{x}, \phi(\mathbf{x})] + b
\end{align}
```

where $\mathbf{x} \in \mathbb{R}^d$ is the raw input, $b \in \mathbb{R}$ is the bias, and $\phi(\mathbf{x})$ is a set of cross-product transformations:

```math
\begin{align}
\phi_k(\mathbf{x}) = \prod_{i=1}^{d} x_i^{c_{ki}}, \quad c_{ki} \in \{0, 1\}
\end{align}
```

$c_{ki} = 1$ only if the $i$-th feature is part of the $k$-th cross-product. For example, the cross-product `AND(user_installed_app=netflix, impression_app=hulu)` equals 1 only when both features are present. This enables the model to memorize specific feature co-occurrences.

> [!IMPORTANT]
> The wide component only uses two features for cross-products in the Google Play deployment: **user installed apps** and **impression app**. All other features go through the deep component.

## Deep Component

The deep component is a feed-forward neural network with embedding layers:

**Embedding step** (input → dense representation):  
For each sparse categorical feature $i$ with vocabulary size $V_i$, learn embedding $\mathbf{e}_i \in \mathbb{R}^{32}$.

**Concatenation**:  
All embeddings and dense features are concatenated into a vector $\mathbf{a}^{(0)} \in \mathbb{R}^{\approx 1200}$.

**Hidden layers** (3 ReLU layers):

```math
\begin{align}
\mathbf{a}^{(l+1)} = \text{ReLU}(\mathbf{W}^{(l)} \mathbf{a}^{(l)} + \mathbf{b}^{(l)})
\end{align}
```

where $\mathbf{W}^{(l)}$ and $\mathbf{b}^{(l)}$ are the weight matrix and bias of layer $l$.

**Output**: final activation $\mathbf{a}^{(L_f)} \in \mathbb{R}^{h}$ fed into the joint prediction.

## Joint Training and Prediction

The wide and deep components are trained **simultaneously** (not independently stacked), with their outputs combined:

```math
\begin{align}
P(Y=1 \mid \mathbf{x}) = \sigma\!\left(\mathbf{w}_{\text{wide}}^T [\mathbf{x}, \phi(\mathbf{x})] + \mathbf{w}_{\text{deep}}^T \mathbf{a}^{(L_f)} + b\right)
\end{align}
```

where $\sigma(\cdot)$ is the sigmoid function and $Y \in \{0,1\}$ is the binary acquisition label.

> [!IMPORTANT]
> Joint training differs from **ensemble**: in an ensemble, each model is trained independently and combined only at inference. In joint training, gradients from the output loss flow through both components simultaneously, so each component learns with awareness of the other.

### Optimizers

| Component | Optimizer | Regularization |
|---|---|---|
| Wide | FTRL (Follow-the-Regularized-Leader) | L₁ |
| Deep | AdaGrad | — |

FTRL with L₁ is chosen for the wide component because it produces sparse weights, which is desirable for the large-dimensional cross-product features. AdaGrad adapts the learning rate for the dense embedding updates in the deep component.

# Feature Engineering

## Categorical Features

String-valued categorical features are converted to integer IDs via vocabularies that map each unique string value to a dense integer. Features with fewer than a minimum number of occurrences are mapped to a shared out-of-vocabulary (OOV) bucket.

## Continuous Features

Continuous real-valued features are normalized to $[0,1]$ using the **quantile normalization** approach:

```math
\begin{align}
\tilde{x}_i = \frac{\text{rank}(x_i) - 1}{n_q - 1}
\end{align}
```

where $n_q$ is the number of quantile buckets. Each quantile maps to a value in $[0,1]$, which stabilizes training compared to raw values with high variance.

# Training Pipeline

```
1. Data Generation
   ├─ Collect user impression logs (label: acquired=1, not acquired=0)
   ├─ Build vocabularies for categorical features
   └─ Compute quantile boundaries for continuous features

2. Model Training (Warm-Starting)
   ├─ Initialize embeddings and model weights from previous model checkpoint
   ├─ Train jointly on new data (500+ billion examples total)
   └─ Validate via offline AUC and dry runs

3. Model Export and Serving
   ├─ Push updated model to serving infrastructure
   ├─ Score top-k candidate apps per query via dot product + model
   └─ Multithreaded batch scoring: 10M apps/sec at 14ms latency
```

> [!NOTE]
> Warm-starting means new model parameters are initialized from a previously trained model rather than random initialization, which reduces the amount of retraining needed when new data arrives—critical for production systems receiving continuous new data.

# Differences from Similar Methods

| Method | Key Idea | Difference from Wide & Deep |
|---|---|---|
| Logistic Regression (wide-only) | Linear model on raw + cross-product features | Cannot generalize to unseen feature pairs |
| Deep Neural Networks (deep-only) | Embeddings + MLP | May over-generalize; less effective for niche patterns |
| Factorization Machines (FM) | Dot-product interactions between embeddings | Wide & Deep uses full MLP layers—more expressive nonlinear interactions |
| Collaborative Filtering (e.g., AppJoy) | User-item co-occurrence matrix factorization | No cross-product memorization; Wide & Deep directly uses impression features |

> [!TIP]
> Factorization machines use $\hat{y} = \mathbf{w}^T\mathbf{x} + \sum_{i < j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$, where interactions are limited to pairwise dot products. The deep component of Wide & Deep learns higher-order, nonlinear interactions through stacked ReLU layers.

# Experiments

- **Dataset**: Google Play app store recommendation logs — over **500 billion** training examples; binary label (app acquired = 1)
- **Users**: 1 billion+ active users; 1 million+ apps
- **Serving load**: ~10 million app scores per query (retrieval + ranking pipeline)
- **Hardware**: Not specified
- **Optimizer**: FTRL (L₁) for wide; AdaGrad for deep
- **Embedding dim**: 32 per categorical feature; ~1,200-dim concatenated input to MLP
- **Layers**: 3 hidden ReLU layers

### Results (Online A/B Test, 3 weeks)

| Model | App Acquisitions Gain (vs. Wide-only) |
|---|---|
| Wide-only (control) | baseline |
| Deep-only | +2.9% |
| Wide & Deep | **+3.9%** |

- Offline AUC: Wide & Deep = **0.728** vs Wide-only = 0.726
- Serving latency: **14ms** (multithreaded) vs 31ms (single batch)

> [!CAUTION]
> Offline AUC improvement (0.728 vs 0.726) is very small, while the online gain (+3.9%) is substantial. This gap between offline and online metrics is common in recommendation systems and suggests that AUC alone is insufficient for evaluating recommendation quality.
