# Meta Information

- URL: [Multi-Behavior Recommendation with Cascading Graph Convolution Networks](https://arxiv.org/abs/2303.15720)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Cheng, Z., Han, S., Liu, F., Zhu, L., Gao, Z., & Peng, Y. (2023). Multi-Behavior Recommendation with Cascading Graph Convolution Networks. *ACM Web Conference 2023 (WWW '23)*.

# Multi-Behavior Recommendation with Cascading Graph Convolution Networks (MB-CGCN)

## Overview

Multi-behavior recommendation systems leverage multiple types of user interactions (e.g., view, add-to-cart, purchase) to improve prediction accuracy for a **target behavior** (typically purchase). The core insight is that auxiliary behaviors such as clicking and carting reflect user intent that can be used to augment sparse purchase data.

MB-CGCN proposes a **cascading GCN architecture** that explicitly models the dependency chain between behaviors (view → cart → buy). Unlike prior work, it passes the learned embeddings of one behavior directly into the next GCN block via a feature transformation, making the chain dependency first-class rather than implicit.

> [!NOTE]
> "MB-CGCN achieves significant improvements of 33.7% and 35.9% on average over the two datasets in terms of Recall@10 and NDCG@10, respectively."

**Applicable to**: E-commerce platforms where users leave multiple interaction signals and purchase data is sparse. Useful when behaviors form a natural funnel (browsing → cart → buy).

## Problem Formulation

Let $\mathcal{U}$ be the set of users and $\mathcal{V}$ be the set of items. There are $B$ behavior types (e.g., view, cart, buy). Each behavior $b \in \{1, \dots, B\}$ is represented by a binary interaction matrix:

```math
\begin{align}
  Y^b \in \{0, 1\}^{|\mathcal{U}| \times |\mathcal{V}|}
\end{align}
```

where $Y^b_{ui} = 1$ if user $u$ performed behavior $b$ on item $i$. Behavior $b = B$ is the **target behavior** (buy). The goal is to predict the top-$K$ items for each user under behavior $B$.

**Input**: $B$ interaction matrices $\{Y^1, Y^2, \dots, Y^B\}$ and initial user/item embeddings.
**Output**: A ranked list of items per user for the target behavior $B$.

## Model Architecture

### Embedding Initialization

Users and items are mapped to a shared $d$-dimensional space via learnable embedding matrices:

```math
\begin{align}
  e_u^{(b,0)} \in \mathbb{R}^d, \quad e_i^{(b,0)} \in \mathbb{R}^d
\end{align}
```

For $b = 1$, the embeddings are randomly initialized. For $b > 1$, they are initialized from the transformation of the previous behavior's learned embeddings (see Feature Transformation below).

### LightGCN Block per Behavior

Each behavior $b$ has its own LightGCN that propagates embeddings over the corresponding user-item graph $G^b$ for $L$ layers. At each layer $l$:

```math
\begin{align}
  e_u^{(b,l)} &= \sum_{i \in \mathcal{N}_u^b} \frac{1}{\sqrt{|\mathcal{N}_u^b||\mathcal{N}_i^b|}} e_i^{(b,l-1)} \\
  e_i^{(b,l)} &= \sum_{u \in \mathcal{N}_i^b} \frac{1}{\sqrt{|\mathcal{N}_i^b||\mathcal{N}_u^b|}} e_u^{(b,l-1)}
\end{align}
```

where $\mathcal{N}_u^b$ is the set of items that user $u$ interacted with under behavior $b$.

The final embedding for behavior $b$ is the mean over layers:

```math
\begin{align}
  e_u^{(b)} = \frac{1}{L+1} \sum_{l=0}^{L} e_u^{(b,l)}
\end{align}
```

### Feature Transformation (Cascading Connection)

After learning embeddings for behavior $b$, MB-CGCN initializes the next GCN block via a linear transformation:

```math
\begin{align}
  e_u^{(b+1,0)} &= W_u^b \cdot e_u^{(b)} \\
  e_i^{(b+1,0)} &= W_i^b \cdot e_i^{(b)}
\end{align}
```

where $W_u^b, W_i^b \in \mathbb{R}^{d \times d}$ are trainable weight matrices. This step filters noise while preserving discriminative features from the previous behavior.

> [!IMPORTANT]
> The feature transformation is the key difference from CRGCN (the strongest prior baseline), which uses a residual connection that carries **all** prior features forward. MB-CGCN selectively distills features via the learned weight matrix, preventing noise accumulation across the chain.

### Embedding Aggregation

All behavior embeddings are aggregated to form the final user/item representation:

```math
\begin{align}
  e_u = \sum_{b=1}^{B} e_u^{(b)}, \quad e_i = \sum_{b=1}^{B} e_i^{(b)}
\end{align}
```

Summation is used over concatenation to avoid linear growth of embedding size with $B$.

### Prediction and Training

The predicted preference score is:

```math
\begin{align}
  \hat{y}_{ui} = e_u^\top \cdot e_i
\end{align}
```

Training uses **Bayesian Personalized Ranking (BPR)** loss:

```math
\begin{align}
  \mathcal{L}_{BPR} = -\sum_{(u,i,j) \in \mathcal{D}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
\end{align}
```

where $(u, i, j)$ means user $u$ has a positive interaction with item $i$ under behavior $B$ and has not interacted with item $j$. Only the target behavior provides supervision; no multi-task learning is used.

### Algorithm Summary

```
Input: Behavior interaction matrices {Y^1, ..., Y^B}, embedding dim d, layers L
Output: Ranked item list per user for behavior B

1. Initialize e_u^(1,0) and e_i^(1,0) randomly for all u, i
2. For b = 1 to B:
   a. Build graph G^b from Y^b
   b. Run LightGCN for L layers to get e_u^(b,l), e_i^(b,l)
   c. Pool: e_u^(b) = mean(e_u^(b,0), ..., e_u^(b,L))
   d. If b < B:
      e_u^(b+1,0) = W_u^b * e_u^(b)
      e_i^(b+1,0) = W_i^b * e_i^(b)
3. Aggregate: e_u = sum_b e_u^(b), e_i = sum_b e_i^(b)
4. Predict: y_hat_ui = e_u^T * e_i
5. Train with BPR loss on target behavior B
```

## Comparison with Related Methods

| Method | Multi-behavior | GNN Backbone | Behavior Dependency | Target-only Training |
|--------|---------------|--------------|---------------------|---------------------|
| LightGCN | No | LightGCN | — | Yes |
| NMTR | Yes | NCF | Cascade (multi-task) | No (MTL) |
| MBGCN | Yes | GCN | Shared graph | No (MTL) |
| CRGCN | Yes | LightGCN | Residual connection | Yes |
| **MB-CGCN** | **Yes** | **LightGCN** | **Feature transformation** | **Yes** |

> [!TIP]
> LightGCN (He et al., 2020) simplifies GCN for collaborative filtering by removing feature transformation and nonlinearity within each layer, relying purely on weighted sum aggregation. MB-CGCN restores a lightweight linear transformation **only between** behavior blocks, not within them.

### Why Not Multi-Task Learning?

MTL-based methods (e.g., NMTR, MBGCN) use auxiliary behavior losses to regularize target behavior training but require careful loss weighting. MB-CGCN avoids this by using behavior chain embeddings as initialization inputs, so auxiliary behaviors implicitly contribute without needing separate loss terms.

## Experiments

- **Datasets**: Beibei and Tmall (both from real-world e-commerce platforms)
  - **Beibei**: 21,716 users × 7,977 items; behaviors: view (2,412,586), cart (642,622), buy (304,576)
  - **Tmall**: 15,449 users × 11,953 items; behaviors: view (873,954), cart (195,476), buy (104,329)
  - 80/20 train/test split on the target (buy) behavior
- **Hardware**: Not specified
- **Optimizer**: Adam
- **Metrics**: Recall@K and NDCG@K for $K \in \{10, 20, 50\}$
- **Key Results**:
  - MB-CGCN achieves +23.2% Recall@10 on Beibei and +44.2% Recall@10 on Tmall over the best baseline (CRGCN)
  - NDCG@10 improvements: +17.6% (Beibei), +54.2% (Tmall)
  - Ablation: removing feature transformation drops Recall@20 by 9.0% (Beibei) and 0.7% (Tmall)
  - Correct behavior ordering (view→cart→buy) consistently outperforms incorrect orderings

> [!CAUTION]
> The behavior ordering experiment assumes a predefined chain. In domains without a clear funnel, the optimal ordering may need to be learned or searched.
