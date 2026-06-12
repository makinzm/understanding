# Meta Information

- URL: [[1708.05031] Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural Collaborative Filtering. In Proceedings of the 26th International Conference on World Wide Web (WWW 2017), pp. 173–182.

# Neural Collaborative Filtering (NCF)

NCF is a general framework that replaces the fixed inner product in matrix factorization (MF) with a learnable neural network interaction function. It is designed for recommendation systems that rely on implicit feedback (clicks, views, purchases) where explicit ratings are unavailable.

**Applicable when:**
- User-item interaction data is implicit (binary: observed = 1, not observed = 0)
- The goal is to rank items for each user (top-N recommendation)
- One wants to model non-linear user-item interactions beyond MF's linear inner product

## 1. Problem Formulation

**Input:** User-item interaction matrix $Y \in \{0, 1\}^{M \times N}$, where $M$ is the number of users and $N$ is the number of items. Entry $y_{ui} = 1$ if user $u$ interacted with item $i$, and $y_{ui} = 0$ if there is no observed interaction (not necessarily a negative preference).

**Output:** Predicted interaction score $\hat{y}_{ui} \in [0, 1]$ representing the likelihood that user $u$ will interact with item $i$.

## 2. NCF Framework

All NCF models follow a unified architecture:

$$\hat{y}_{ui} = f(\mathbf{P}^T \mathbf{v}_u^U, \mathbf{Q}^T \mathbf{v}_i^I \mid \mathbf{P}, \mathbf{Q}, \Theta_f)$$

where:
- $\mathbf{v}_u^U \in \{0,1\}^M$: one-hot encoding of user $u$
- $\mathbf{v}_i^I \in \{0,1\}^N$: one-hot encoding of item $i$
- $\mathbf{P} \in \mathbb{R}^{M \times K}$: user latent factor matrix (embedding)
- $\mathbf{Q} \in \mathbb{R}^{N \times K}$: item latent factor matrix (embedding)
- $K$: embedding dimension (latent factor size)
- $\Theta_f$: parameters of the interaction function $f$
- $f$: the neural interaction function (replaces the inner product in standard MF)

The user and item embeddings are retrieved as $\mathbf{p}_u = \mathbf{P}^T \mathbf{v}_u^U \in \mathbb{R}^K$ and $\mathbf{q}_i = \mathbf{Q}^T \mathbf{v}_i^I \in \mathbb{R}^K$.

## 3. Objective Function

NCF uses the binary cross-entropy loss (log loss) to treat recommendation as a binary classification problem:

$$\mathcal{L} = -\sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} \left[ y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log(1 - \hat{y}_{ui}) \right]$$

where $\mathcal{Y}$ is the set of observed interactions and $\mathcal{Y}^-$ is the set of sampled negative instances.

> [!NOTE]
> Negative instances are uniformly sampled from unobserved interactions at each epoch. The sampling ratio (negatives per positive) is a hyperparameter; optimal performance was found at 3–6 negative samples per positive.

> [!IMPORTANT]
> This approach differs from standard MF methods (e.g., BPR) that use pairwise ranking loss. NCF adopts a pointwise approach with probabilistic treatment of implicit feedback, which empirically outperforms BPR in this framework.

## 4. Generalized Matrix Factorization (GMF)

GMF recovers standard MF as a special case and extends it by making the output layer learnable.

**Calculation order:**

1. Look up user embedding: $\mathbf{p}_u \in \mathbb{R}^K$
2. Look up item embedding: $\mathbf{q}_i \in \mathbb{R}^K$
3. Element-wise product: $\mathbf{p}_u \odot \mathbf{q}_i \in \mathbb{R}^K$
4. Output layer with learned weight vector $\mathbf{h} \in \mathbb{R}^K$:
   $$\hat{y}_{ui} = \sigma(\mathbf{h}^T (\mathbf{p}_u \odot \mathbf{q}_i))$$

where $\sigma$ is the sigmoid function.

> [!NOTE]
> Standard MF is recovered when $\mathbf{h}$ is fixed to the all-ones vector and $\sigma$ is the identity function. In GMF, $\mathbf{h}$ is learned from data, "allowing varying importance of latent dimensions."

## 5. Multi-Layer Perceptron (MLP)

MLP captures complex, non-linear user-item interactions by concatenating embeddings and passing them through multiple hidden layers.

**Separate embedding matrices** are used for MLP (distinct from GMF embeddings):
- $\mathbf{p}_u^{MLP} \in \mathbb{R}^{K}$: user embedding for MLP
- $\mathbf{q}_i^{MLP} \in \mathbb{R}^{K}$: item embedding for MLP

**Calculation order** (tower structure halving layer size each step):

1. Input: $\mathbf{z}_0 = [\mathbf{p}_u^{MLP}; \mathbf{q}_i^{MLP}] \in \mathbb{R}^{2K}$
2. For each hidden layer $l = 1, \dots, L$:
   $$\mathbf{z}_l = \text{ReLU}(\mathbf{W}_l^T \mathbf{z}_{l-1} + \mathbf{b}_l)$$
   where $\mathbf{W}_l \in \mathbb{R}^{d_{l-1} \times d_l}$, $\mathbf{b}_l \in \mathbb{R}^{d_l}$, and $d_l < d_{l-1}$ (tower pattern)
3. Output: $\hat{y}_{ui} = \sigma(\mathbf{h}^T \mathbf{z}_L)$

> [!NOTE]
> ReLU was empirically superior to sigmoid and tanh activations for the hidden layers. Using tanh/sigmoid causes vanishing gradients, while ReLU encourages sparse activations.

## 6. Neural Matrix Factorization (NeuMF)

NeuMF is the unified model combining GMF and MLP with separate embeddings for each component. This allows each sub-model to learn optimal embedding sizes independently.

**Architecture:**

1. GMF path: $\phi^{GMF} = \mathbf{p}_u^{GMF} \odot \mathbf{q}_i^{GMF} \in \mathbb{R}^{K_{GMF}}$
2. MLP path: $\phi^{MLP} = \mathbf{z}_L \in \mathbb{R}^{d_L}$ (last hidden layer output)
3. Concatenation and output:
   $$\hat{y}_{ui} = \sigma\left(\mathbf{h}^T \begin{bmatrix} \phi^{GMF} \\ \phi^{MLP} \end{bmatrix}\right)$$
   where $\mathbf{h} \in \mathbb{R}^{K_{GMF} + d_L}$

**Pre-training procedure:**

```
1. Train GMF independently to convergence
2. Train MLP independently to convergence
3. Initialize NeuMF:
   - P_GMF, Q_GMF ← converged GMF embeddings
   - P_MLP, Q_MLP ← converged MLP embeddings
   - h ← [h_GMF * α; h_MLP * (1-α)] where α = 0.5
4. Fine-tune NeuMF with a lower learning rate (Adam → SGD)
```

> [!NOTE]
> Pre-training addresses the non-convex optimization problem of NeuMF, providing better initialization than random initialization.

## 7. Comparison with Related Methods

| Method | Interaction Function | Non-linearity | Implicit Feedback |
|---|---|---|---|
| Standard MF (SVD) | Inner product $\mathbf{p}_u^T \mathbf{q}_i$ | None (linear) | No (requires explicit ratings) |
| BPR-MF | Inner product (pairwise) | None | Yes (pairwise ranking) |
| eALS | Weighted squared loss | None | Yes (pointwise) |
| GMF (this paper) | Learned weighted element-wise product | Sigmoid (output) | Yes |
| MLP (this paper) | Concatenation + hidden layers | ReLU | Yes |
| NeuMF (this paper) | GMF + MLP combined | ReLU + Sigmoid | Yes |

> [!IMPORTANT]
> The key limitation of standard MF is that the inner product $\mathbf{p}_u^T \mathbf{q}_i = \sum_k p_{uk} q_{ik}$ is a fixed linear function of the latent factors. The paper demonstrates a concrete case where this causes ranking inconsistencies with Jaccard similarity, motivating the use of neural interaction functions.

## 8. NCF Algorithm (Pseudocode)

```
Input: interaction matrix Y, number of negative samples ρ, number of factors K
Output: predicted scores ŷ_ui for all (u, i) pairs

Initialize P, Q randomly; θ_f randomly
for each epoch:
    for each observed interaction (u, i) in Y:
        sample ρ unobserved items j₁, ..., j_ρ for user u
        for each training instance (u, i, y_ui=1) and (u, jₖ, y=0):
            compute ŷ = f(p_u, q_i; θ_f)   # NCF forward pass
            compute loss L = -[y log ŷ + (1-y) log(1-ŷ)]
            update P, Q, θ_f via Adam optimizer

At inference:
    for each user u:
        score all items i: ŷ_ui = f(p_u, q_i)
        rank items by ŷ_ui descending
        return top-N items
```

# Experiments

- **Datasets:**
  - MovieLens-1M: 1,000,209 interactions, 6,040 users, 3,706 items, sparsity 95.53%
  - Pinterest: 1,500,809 interactions, 55,187 users, 9,916 items, sparsity 99.73%
  - Both converted to implicit feedback (rating/pin existence → 1)
  - Test split: leave-one-out evaluation (latest interaction per user as test, remaining as train)
- **Evaluation metrics:** Hit Ratio (HR@10) and NDCG@10 over 100 randomly sampled negative items per user
- **Baselines:** ItemPop (popularity), ItemKNN (item-based CF), BPR (pairwise MF), eALS (weighted MF)
- **Key results:**
  - NeuMF outperformed eALS and BPR by ~4.5% and ~4.9% relative improvement in HR@10, respectively
  - Deeper MLP (3 hidden layers) consistently better than shallower (1–2 layers)
  - Pre-trained NeuMF outperformed randomly initialized NeuMF
  - Optimal negative sampling ratio: 3–6 negatives per positive interaction
- **Optimizer:** Adam for pre-training GMF and MLP; SGD for fine-tuning NeuMF
- **Embedding size:** $K \in \{8, 16, 32, 64\}$; MLP hidden layer sizes: $[512, 256, 128, 64]$ tower
