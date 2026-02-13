# Meta Information

- URL: [[1606.07792] Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... & Shah, H. (2016). Wide & deep learning for recommender systems. Proceedings of the 1st workshop on deep learning for recommender systems, 7-10.

# Overview

Wide & Deep Learning is a framework for **large-scale recommendation systems** that jointly trains a linear model (wide) and a feed-forward neural network (deep) to balance two complementary properties:

- **Memorization**: Learning frequent co-occurrences of features directly from historical data (handled by the wide component).
- **Generalization**: Leveraging feature combinations never or rarely seen in training, achieved through learned dense embeddings (handled by the deep component).

The system was deployed on **Google Play** (over 1 billion active users) for app recommendation, achieving statistically significant improvements over wide-only and deep-only baselines.

> [!NOTE]
> "Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort."

## Comparison with Related Methods

| Method | Memorization | Generalization | Feature Engineering |
|---|---|---|---|
| Logistic regression (wide only) | High (explicit) | Low | Heavy manual cross-product features |
| Deep neural network (deep only) | Low (implicit) | High | Light (embeddings auto-generated) |
| Factorization Machines | Medium | Medium | Quadratic interaction terms |
| **Wide & Deep (proposed)** | High | High | Moderate (hybrid) |

The key novelty over ensemble methods is **joint training**: the wide and deep components are trained simultaneously so that their gradients are back-propagated through both paths at once. This requires fewer wide component parameters because the deep component compensates.

# Model Architecture

**Overall Input → Output:**
- Input: sparse categorical features (user query, item features, context) + dense continuous features
- Output: $P(Y=1 \mid \mathbf{x}) \in [0, 1]$, the probability that a user clicks/downloads an app

## 1. Wide Component

The wide component is a **generalized linear model**:

$$y_{\text{wide}} = \mathbf{w}_{\text{wide}}^T [\mathbf{x},\ \phi(\mathbf{x})] + b$$

- $\mathbf{x} \in \mathbb{R}^d$: raw input features (binary/sparse)
- $\phi(\mathbf{x})$: cross-product feature transformations
- $\mathbf{w}_{\text{wide}} \in \mathbb{R}^{d + |\phi|}$: learned weights
- $b \in \mathbb{R}$: bias

**Cross-product transformation** for feature set $\mathcal{T}_k$:

$$\phi_k(\mathbf{x}) = \prod_{i=1}^{d} x_i^{c_{ki}}, \quad c_{ki} \in \{0, 1\}$$

For example, AND(user\_installed\_app=netflix, impression\_app=pandora) $= 1$ only when both features are active simultaneously. This explicitly encodes co-occurrence patterns that are directly interpretable.

## 2. Deep Component

The deep component is a **feed-forward neural network** operating on dense representations.

**Embedding step** (sparse → dense):
- Each sparse categorical feature (e.g., app ID from a vocabulary of millions) is mapped to a $d_e = 32$-dimensional embedding vector.
- All embeddings are concatenated with continuous features to form the first-layer activation: $\mathbf{a}^{(0)} \in \mathbb{R}^{\sim 1200}$.

**Forward pass** through $L$ hidden layers:

$$\mathbf{a}^{(l)} = \text{ReLU}\!\left(W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l = 1, \dots, L$$

- $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$: weight matrix at layer $l$
- $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$: bias vector at layer $l$
- Three ReLU hidden layers are used (exact widths not published)

The final hidden-layer activation $\mathbf{a}^{(l_f)} \in \mathbb{R}^{n_{l_f}}$ is fed into the joint output unit.

## 3. Joint Training and Prediction

The combined prediction is:

$$P(Y=1 \mid \mathbf{x}) = \sigma\!\left(\mathbf{w}_{\text{wide}}^T [\mathbf{x},\ \phi(\mathbf{x})] + \mathbf{w}_{\text{deep}}^T \mathbf{a}^{(l_f)} + b\right)$$

- $\sigma(\cdot)$: sigmoid function
- $\mathbf{w}_{\text{deep}} \in \mathbb{R}^{n_{l_f}}$: output weights for the deep component

**Training algorithm (mini-batch stochastic optimization):**

```
Input: training set {(x_i, y_i)}, mini-batch size B
Initialize: embeddings, W^(l), w_wide, w_deep, b

For each mini-batch:
  1. Forward pass through wide component: y_wide = w_wide^T [x, φ(x)] + b
  2. Embed sparse features → a^(0)
  3. Forward pass through deep layers: a^(1), ..., a^(l_f)
  4. Compute joint prediction: P = σ(y_wide + w_deep^T a^(l_f) + b)
  5. Compute loss: L = -[y log P + (1-y) log(1-P)]
  6. Back-propagate gradients through both wide and deep paths
  7. Wide component: updated with FTRL + L1 regularization
  8. Deep component: updated with AdaGrad
```

> [!IMPORTANT]
> The wide and deep components use **different optimizers**: FTRL (Follow-the-Regularized-Leader) with L1 regularization for the wide component (promotes sparsity in feature weights) and AdaGrad for the deep component (adaptive learning rates for embeddings).

# System Pipeline

The production system has three stages:

## Data Generation

- Training data spans **several hundred billion examples** (user query/impression logs).
- Categorical string features are mapped to integer IDs using a vocabulary; features appearing fewer than a minimum count threshold are mapped to a single out-of-vocabulary bucket (OOV bucket).
- Continuous features are normalized to $[0, 1]$ using quantile-based bucketization: a value $x$ in quantile $i$ out of $n_q$ is mapped to $\tilde{x} = \frac{i-1}{n_q - 1}$.

## Model Training

- The model is warm-started from the previous version of the model whenever it is retrained on new data, rather than training from scratch, to reduce serving latency.
- Training is done on distributed infrastructure (Google Brain).

## Model Serving

- Servers retrieve the top $k$ apps from a candidate set using the model score.
- **Multithreaded batch inference** is used to keep latency low.
- At batch size 50 with 4 threads: **14 ms mean latency**.
- System scores **over 10 million app candidates per second** at peak traffic.

# Experiments

- **Dataset**: Google Play app store user interaction logs
  - Training: 500+ billion (app query, impression) examples
  - Evaluation: 3-week live A/B test on Google Play production traffic
  - No public dataset release
- **Hardware**: Google distributed training infrastructure (not fully specified)
- **Optimizer**: FTRL + L1 (wide), AdaGrad (deep)
- **Results**:

| Model | Offline AUC | Online App Acquisition Gain |
|---|---|---|
| Wide only (control) | 0.726 | baseline (0%) |
| Deep only | 0.722 | +2.9% |
| Wide & Deep | **0.728** | **+3.9%** |

The Wide & Deep model achieves the highest offline AUC and a **3.9% relative improvement** in app acquisition rate (statistically significant) over the optimized wide-only baseline.

> [!NOTE]
> The offline AUC of the deep-only model is slightly lower than the wide-only model, yet the deep-only model achieves a larger online gain (+2.9%), which the authors attribute to the difference between offline and online metrics (AUC measures ranking quality, not absolute conversion probability).

# Applicability

This framework is applicable when:

- The recommendation corpus is very large (millions to billions of items).
- User-item interaction history is available as sparse categorical features.
- Both precise memorization of known patterns (e.g., "users who installed app X also install app Y") and generalization to unseen feature combinations are important.
- Latency constraints require a single jointly-trained model rather than an ensemble of multiple separately-trained models.

Typical adopters are industry practitioners building production-scale recommender systems (e-commerce, app stores, news feeds) where historical interaction logs are abundant.
