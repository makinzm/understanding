# Meta Information

- URL: [Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/abs/0902.2206)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J., & Smola, A. (2009). Feature Hashing for Large Scale Multitask Learning. *26th International Conference on Machine Learning (ICML 2009)*.

# Overview

Feature hashing (also called the "hashing trick") is a dimensionality reduction method that maps high-dimensional sparse feature vectors to a fixed-size lower-dimensional space using a hash function. Unlike random projections that require storing a dense projection matrix, hashing requires only a hash function and is thus memory-efficient. The key insight is that a *signed* hash function (mapping to $\pm 1$) preserves inner products in expectation with low variance, enabling accurate linear classifiers in the compressed space.

**Who should use this:** Engineers and researchers building large-scale machine learning systems (e.g., spam filters, recommender systems, NLP pipelines) where the feature space is enormous (millions of tokens, users, or categories) and storing full feature vectors or projection matrices is infeasible.

**When:** When the number of features $n \gg m$ (desired dimensionality), and the downstream task is a linear model (logistic regression, SVM, etc.).

**Where:** Particularly effective for text classification, personalized filtering, and multitask learning over hundreds of thousands of tasks.

# Problem Setting

## Input and Output

- **Input**: A sparse feature vector $x \in \mathbb{R}^n$ where $n$ is very large (e.g., $n = 40 \times 10^6$ unique tokens).
- **Output**: A compressed vector $\phi(x) \in \mathbb{R}^m$ where $m \ll n$ (e.g., $m = 2^{22} \approx 4 \times 10^6$).

The mapping is defined by two hash functions:
- $h: \{1, \ldots, n\} \to \{1, \ldots, m\}$ — assigns each feature index to a bucket.
- $\xi: \{1, \ldots, n\} \to \{+1, -1\}$ — assigns a random sign to each feature.

## Hash Feature Map

$$\phi_i^{(h,\xi)}(x) := \sum_{j : h(j) = i} \xi(j) \cdot x_j, \quad i \in \{1, \ldots, m\}$$

Multiple original features can collide into the same bucket $i$; the signed sum averages out collision noise.

> [!NOTE]
> The signed hashing (using $\xi$) is critical. Without it, collisions always add positively, causing bias. With $\xi$, colliding contributions cancel in expectation.

# Theoretical Guarantees

## Unbiasedness (Lemma 2)

For any two vectors $x, x' \in \ell_2$:

$$\mathbb{E}_{h,\xi}\left[\langle \phi(x), \phi(x') \rangle\right] = \langle x, x' \rangle$$

**Proof sketch**: For $j \neq j'$, $\mathbb{E}[\xi(j)\xi(j')] = 0$ because $\xi(j)$ and $\xi(j')$ are independent. For $j = j'$, $\xi(j)^2 = 1$, contributing $x_j x_j'$ to the expectation.

## Variance Bound

$$\text{Var}\left[\langle \phi(x), \phi(x') \rangle\right] \leq \frac{\|x\|^2 \|x'\|^2 - \langle x, x' \rangle^2}{m} \leq \frac{1}{m}$$

for unit-norm vectors. Variance decreases linearly with $m$, so larger hash tables give more accurate inner product approximations.

## Concentration Bound (Theorem 3)

For unit-norm $x$ and any $\varepsilon, \delta \in (0, 1)$:

$$\Pr\left[\left| \|\phi(x)\|^2 - 1 \right| \geq \varepsilon\right] \leq 2\delta \quad \text{when } m \geq \frac{72 \log(1/\delta)}{\varepsilon^2}$$

The required $m$ grows only *logarithmically* with the failure probability $1/\delta$ and inversely with $\varepsilon^2$, giving exponential tail bounds.

**Corollary 5** (dataset-level): To preserve norms of all $n$ training examples simultaneously with probability $\geq 1-\delta$:

$$m \geq \Omega\left(\frac{1}{\varepsilon^2} \log\frac{n}{\delta}\right)$$

The number of observations enters *logarithmically*.

## Multitask Interference (Theorem 7)

For $T$ tasks using independent hash functions $\phi_t$ and a shared global component $\phi_0$, the interference between task $t$'s parameters $w_t \in \mathbb{R}^n$ and the global classifier is bounded using Bernstein's inequality. Independent hashed subspaces are approximately orthogonal with high probability.

# Algorithm: Multiple Hashing (Lemma 6)

To reduce variance further without increasing $m$, replicate each feature entry $c$ times before hashing:

**Algorithm: c-Replication Hashing**
```
Input: feature index j, original value x_j, replication count c
For k = 1 to c:
    i ← h(j, k)                    # bucket from hash of (j, k)
    φ_i += ξ(j, k) * x_j / sqrt(c) # scaled signed contribution
```

**Effect**: Component magnitude decreases by $1/\sqrt{c}$, variance decreases by factor $c$, but sparsity increases (each original feature now occupies $c$ buckets).

# Multitask Learning Architecture

## Personalized Spam Filtering

The model maintains two sets of parameters:
- **Global classifier**: $w_0 \in \mathbb{R}^n$ shared across all users, hashed to $\phi_0(w_0) \in \mathbb{R}^{m_0}$.
- **Per-user classifiers**: $w_u \in \mathbb{R}^n$ for each user $u$, each hashed to $\phi_u(w_u) \in \mathbb{R}^{m_u}$ with an independent hash function $h_u$.

**Prediction** for user $u$ on email $x$:

$$\hat{y} = \text{sign}\left(\langle \phi_0(w_0), \phi_0(x) \rangle + \langle \phi_u(w_u), \phi_u(x) \rangle\right)$$

**Why independent hash functions?** Using $h_u \neq h_0$ ensures the personal and global subspaces are approximately orthogonal (Theorem 7), preventing interference during gradient updates.

**Memory savings**: For 433,167 users with 40M tokens at $m = 2^{22}$, storing per-user classifiers as full vectors is infeasible. Hashing compresses each to $2^{22}$ floats.

## Massively Multiclass Estimation

For $K$ classes, each class $k$ uses hash function $h_k$ to define $\phi_k(w_k)$, enabling efficient one-vs-all classification without storing $K \times n$ weight matrices.

# Differences from Similar Methods

| Method | Projection type | Memory | Bias | Tail bound |
|---|---|---|---|---|
| Random projections (Achlioptas 2001) | Dense matrix $\Phi \in \mathbb{R}^{m \times n}$ | $O(mn)$ | Unbiased | Polynomial |
| Feature hashing (this work) | Hash function pair $(h, \xi)$ | $O(1)$ | Unbiased | **Exponential** |
| Count-Min sketch (Cormode & Muthukrishnan) | Hash without sign | $O(m)$ | **Biased** (positive collisions) | — |
| Kernel approximation (Rahimi & Recht) | Random Fourier features | $O(mn)$ | Unbiased | Polynomial |

> [!IMPORTANT]
> The signed hash function $\xi$ is what distinguishes feature hashing from the Count-Min sketch and gives it unbiased inner products. Without $\xi$, collisions always add positively, corrupting the inner product estimate.

> [!TIP]
> Feature hashing is implemented in scikit-learn as [`sklearn.feature_extraction.FeatureHasher`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) and in many NLP frameworks as the default tokenization method.

# Experiments

- **Dataset**: 3.2 million emails from 433,167 users across 40 million unique words (proprietary email dataset).
- **Task**: Personalized spam filtering (binary classification per user).
- **Baseline**: Global classifier trained without personalization.
- **Optimizer**: Not specified (standard online learning for logistic regression implied).
- **Hardware**: Not specified.

**Key results**:
- Global hashed classifier converges with $m = 2^{22}$ dimensions (much smaller than $n = 40 \times 10^6$).
- Personalized approach achieves **30% average spam reduction** relative to global baseline.
- Users with **zero training emails** still get 20% spam reduction from the global classifier, showing the global component learns generalizable features.
- Users with 8–15 training emails achieve up to **65% spam reduction**, demonstrating rapid adaptation of personal classifiers.

> [!NOTE]
> "The number of observations enters logarithmically in the analysis" — this is the key result enabling practical deployment: $m$ grows as $O(\log n / \varepsilon^2)$, not $O(n)$.

# Corrections (Appendix D)

The ICML 2009 conference version contained three errors corrected in the arXiv revision:
1. **Contradiction with Alon's lower bounds**: The original bound on embedding dimensionality conflicted with known results for $\ell_1$ embeddings.
2. **Fatal proof error**: Incorrect handling of Kronecker delta summation in the concentration bound proof.
3. **Minor inequality error**: Off-by-one in a probability bound.

The corrected proof uses concentration results from Liberty et al. (2008) via Talagrand's inequality.
