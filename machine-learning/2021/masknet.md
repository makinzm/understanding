# Meta Information

- URL: [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/abs/2102.07619)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Wang, Z., She, Q., & Zhang, J. (2021). MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask. DLP-KDD 2021.

# MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask

## Overview

MaskNet is a deep learning architecture for Click-Through Rate (CTR) prediction in industrial recommendation systems. It introduces multiplicative feature interactions into standard DNN ranking models by applying instance-guided masks — dynamically generated element-wise multipliers derived from the full input embedding — to both feature embedding layers and hidden layers of a feed-forward network.

**Who uses this**: Engineers and researchers building CTR prediction models for online advertising, recommendation, or search ranking systems who want to go beyond purely additive MLP feature interactions.

**When**: Applicable when the downstream task is binary classification (click/no-click) over sparse, high-cardinality categorical features (e.g., user IDs, item IDs, ad categories) encoded as dense embeddings.

**Where**: Deployed in industrial-scale recommendation systems such as those at Sina Weibo, where the input is a large sparse feature space and the output is a predicted click probability.

## Motivation

Standard MLP-based ranking models rely exclusively on additive feature interactions. Rendle et al. (2020) empirically demonstrated that a simple dot-product baseline substantially outperforms MLP layers in collaborative filtering, suggesting that additive operations alone are insufficient to capture complex feature relationships. Prior work such as DeepFM, DCN, and xDeepFM partially addresses this by introducing explicit cross-product or polynomial feature interaction modules, but these are typically applied only at the feature level and do not propagate multiplicative signals into the hidden layers.

MaskNet's key insight is that element-wise multiplication of an instance-conditioned mask with a hidden layer (or an embedding layer) introduces multiplicative non-linearity throughout the entire forward pass, not just at the input feature level.

> [!NOTE]
> "The element-wise product between the mask and hidden layer or feature embedding layer brings multiplicative operation into DNN ranking system in a unified way." (Section 1)

## Problem Formulation

**Input**: A multi-field sparse feature vector $\mathbf{x} = [x_1, x_2, \ldots, x_f]$ where $f$ is the number of fields. Each $x_i$ is a one-hot or multi-hot vector over a high-cardinality vocabulary.

**Embedding layer**: Categorical feature $x_i$ is mapped to a dense embedding:

```math
\begin{align}
  \mathbf{e}_i = \mathbf{W}_e \mathbf{x}_i
\end{align}
```

where $\mathbf{W}_e \in \mathbb{R}^{k \times |\mathcal{V}_i|}$ is the embedding matrix for field $i$ and $k$ is the embedding dimension. Numerical features $x_j$ are scaled as $\mathbf{e}_j = \mathbf{v}_j x_j$ where $\mathbf{v}_j \in \mathbb{R}^k$ is a learned scaling vector.

**Concatenated embedding**: All field embeddings are concatenated to form the shared embedding:

```math
\begin{align}
  \mathbf{V}_{\text{emb}} = \text{concat}(\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_f) \in \mathbb{R}^{f \cdot k}
\end{align}
```

**Output**: A scalar click probability $\hat{y} \in (0, 1)$ produced by a sigmoid over the final prediction layer.

**Training objective**: Binary cross-entropy (log loss):

```math
\begin{align}
  \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]
\end{align}
```

## Instance-Guided Mask

The instance-guided mask generates a multiplicative weight vector conditioned on the full input embedding $\mathbf{V}_{\text{emb}}$. For a target layer of dimension $d$, the mask is computed as:

```math
\begin{align}
  \mathbf{V}_{\text{mask}} = \mathbf{W}_{d2} \left( \text{ReLU}(\mathbf{W}_{d1} \mathbf{V}_{\text{emb}} + \boldsymbol{\beta}_{d1}) \right) + \boldsymbol{\beta}_{d2}
\end{align}
```

where:
- $\mathbf{W}_{d1} \in \mathbb{R}^{(f \cdot k / r) \times (f \cdot k)}$ is the aggregation weight matrix with reduction ratio $r$
- $\mathbf{W}_{d2} \in \mathbb{R}^{d \times (f \cdot k / r)}$ is the projection weight matrix
- $\boldsymbol{\beta}_{d1}, \boldsymbol{\beta}_{d2}$ are bias terms
- $\mathbf{V}_{\text{mask}} \in \mathbb{R}^{d}$ has the same dimension as the target layer

The mask is applied via element-wise multiplication ($\odot$):

```math
\begin{align}
  \mathbf{V}_{\text{masked}} = \mathbf{V}_{\text{mask}} \odot \mathbf{V}_{\text{target}}
\end{align}
```

where $\mathbf{V}_{\text{target}}$ is either a normalized embedding vector or a normalized hidden layer vector.

> [!IMPORTANT]
> The mask weights are re-generated for each input instance from $\mathbf{V}_{\text{emb}}$. This is unlike static feature importance weights (e.g., attention weights learned over the whole training set) — each sample generates its own unique mask, enabling input-conditioned feature amplification or suppression.

## Layer Normalization

MaskNet applies Layer Normalization (Ba et al., 2016) before both the masked embedding and the feed-forward output. For an input vector $\mathbf{x} \in \mathbb{R}^d$:

```math
\begin{align}
  \text{LN}(\mathbf{x}) = \mathbf{g} \odot \frac{\mathbf{x} - \mu}{\sigma} + \mathbf{b}
\end{align}
```

where $\mu$ and $\sigma$ are the mean and standard deviation across the $d$ dimensions, and $\mathbf{g}, \mathbf{b} \in \mathbb{R}^d$ are learned scale and shift parameters. Layer normalization stabilizes training when multiplicative operations can amplify or suppress activations by large factors.

## MaskBlock: The Core Building Block

A MaskBlock takes the shared embedding $\mathbf{V}_{\text{emb}}$ (for mask generation) and a previous layer output (for transformation) as inputs. It produces:

```math
\begin{align}
  \mathbf{V}_{\text{output}} = \text{ReLU}\!\left(\text{LN}\!\left(\mathbf{W}_i \left(\mathbf{V}_{\text{mask}} \odot \text{LN}(\mathbf{V}_{\text{emb}})\right)\right)\right)
\end{align}
```

where:
- $\text{LN}(\mathbf{V}_{\text{emb}}) \in \mathbb{R}^{f \cdot k}$: layer-normalized shared embedding
- $\mathbf{V}_{\text{mask}} \in \mathbb{R}^{f \cdot k}$: instance-guided mask for the embedding, derived from $\mathbf{V}_{\text{emb}}$
- $\mathbf{W}_i \in \mathbb{R}^{d_{\text{out}} \times (f \cdot k)}$: linear projection weight
- Outer $\text{LN}(\cdot)$: normalization applied to the projected activations

The three components — layer normalization, instance-guided mask, and a feed-forward layer — together constitute the MaskBlock. The multiplication of the mask with the normalized embedding is what introduces the non-additive interaction, converting a purely additive DNN layer into a mixed additive/multiplicative computation.

> [!TIP]
> This design is analogous to Squeeze-and-Excitation (SE) Networks (Hu et al., 2018) in computer vision, which use channel-wise attention derived from global average pooling. MaskNet generalizes this idea: the mask is computed from the full input embedding (not pooled), is applied at every layer (not just once), and is re-derived per instance at each MaskBlock.

## MaskNet Architectures

### Serial MaskNet (SerMaskNet)

MaskBlocks are stacked sequentially. Each block applies a mask guided by the global shared embedding $\mathbf{V}_{\text{emb}}$ (not the output of the previous block) to the output of the previous block:

```
Vemb ──→ MaskBlock_1 ──→ MaskBlock_2 ──→ ... ──→ MaskBlock_K ──→ Dense ──→ Output
  └──────────────────────────────────────────────────────────────┘ (mask input to each block)
```

This design allows each stacked block to refine the representation using both the accumulated transformation and a globally-informed mask that does not drift as the stack deepens.

### Parallel MaskNet (ParaMaskNet)

Multiple independent MaskBlocks process the shared embedding in parallel. Each block applies a different mask (with its own learned parameters) to the same $\mathbf{V}_{\text{emb}}$, producing different interaction representations. Outputs are concatenated and fed through dense layers:

```
Vemb ──→ MaskBlock_1 ──┐
  ├──→ MaskBlock_2 ──┤──→ concat ──→ Dense layers ──→ Output
  └──→ MaskBlock_N ──┘
```

This promotes ensemble-like diversity: each branch specializes in capturing different multiplicative interaction patterns from the same input.

## Differences from Related Models

| Model | Feature Interaction Type | Where Multiplication Occurs |
|---|---|---|
| DNN | Additive only (MLP) | None |
| FM / DeepFM | Second-order pairwise dot products + additive MLP | Input feature level only |
| DCN | Explicit polynomial cross features + additive DNN | Input cross layer only |
| xDeepFM (CIN) | Compressed Interaction Network (vector-wise cross) | Input feature level only |
| AutoInt | Multi-head self-attention over embeddings | Input feature level only |
| **SerMaskNet** | Instance-guided mask × all layers, sequential | Embedding + every hidden layer |
| **ParaMaskNet** | Instance-guided mask × all layers, parallel | Embedding + every hidden layer |

> [!IMPORTANT]
> The key distinction from gating mechanisms in NLP (e.g., LSTM gates) and vision (SE-Net) is that MaskNet's mask is conditioned on the full instance embedding $\mathbf{V}_{\text{emb}}$ and is applied at every layer — not just at the attention or recurrent step. This makes multiplicative interactions pervasive throughout the model depth.

## Experiments

- **Datasets**:
  - **Criteo**: 45M instances, 39 fields (13 numerical + 26 categorical), ~30M unique features; split 80/10/10 train/validation/test.
  - **Avazu**: 40.43M instances, 23 fields, ~9.5M unique features; split 80/10/10.
  - **Malware**: 8.92M instances, 82 fields (all categorical), ~0.97M unique features; split 80/10/10.
- **Hardware**: 2 Tesla K40 GPUs
- **Optimizer**: Adam, learning rate $1 \times 10^{-4}$, L2 regularization
- **Batch size**: 1024
- **Embedding dimension**: 10 (fixed across all models for fair comparison)
- **Hidden layers**: 3 layers × 400 neurons, ReLU activation
- **Reduction ratio $r$**: 2 (default for instance-guided mask aggregation)

**Key results** (AUC, higher is better):

| Model | Criteo | Avazu | Malware |
|---|---|---|---|
| FM | 0.7895 | 0.7785 | 0.7166 |
| DNN | 0.8032 | 0.7843 | 0.7281 |
| DeepFM | 0.8057 | 0.7857 | 0.7344 |
| xDeepFM | 0.8064 | 0.7841 | 0.7310 |
| AutoInt | 0.8052 | 0.7844 | 0.7369 |
| **SerMaskNet** | **0.8119** | **0.7877** | **0.7413** |
| **ParaMaskNet** | **0.8124** | **0.7872** | **0.7410** |

Relative improvement (RelaImp) is measured as:

```math
\begin{align}
  \text{RelaImp} = \frac{\text{AUC}_{\text{model}} - 0.5}{\text{AUC}_{\text{base}} - 0.5} - 1
\end{align}
```

SerMaskNet achieves 7.74% RelaImp over FM and 1.55–4.46% over DeepFM/xDeepFM depending on dataset.

**Ablation findings**:
- Removing the instance-guided mask from MaskBlock (reducing it to a plain layer-norm + dense block) causes AUC to drop by roughly 0.3% on Criteo, confirming the mask as the primary source of improvement.
- Removing layer normalization also degrades performance, validating its stabilizing role in the presence of multiplicative operations.
- The feed-forward sublayer within MaskBlock is more critical for SerMaskNet than ParaMaskNet; in ParaMaskNet, the parallel structure provides sufficient capacity even without it.

**Hyperparameter sensitivity**:
- Embedding size: optimal around 30–50 dimensions (diminishing returns beyond that).
- Number of MaskBlocks: optimal at 5–7 for serial, 7–9 for parallel configurations.
- Reduction ratio $r$: AUC is relatively insensitive to $r \in \{1, 2, 3, 4, 5\}$, suggesting the aggregation bottleneck is not a critical design choice.

## References

- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv:1607.06450.
- Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. IJCAI 2017.
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. CVPR 2018.
- Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X., & Sun, G. (2018). xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. KDD 2018.
- Rendle, S., Krichene, W., Zhang, L., & Anderson, J. (2020). Neural Collaborative Filtering vs. Matrix Factorization Revisited. RecSys 2020.
- Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J. (2019). AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks. CIKM 2019.
- Wang, R., Fu, B., Fu, G., & Wang, M. (2017). Deep & Cross Network for Ad Click Predictions. ADKDD 2017.
