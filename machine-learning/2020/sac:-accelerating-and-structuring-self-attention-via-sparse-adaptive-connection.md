# Meta Information

- URL: [SAC: Accelerating and Structuring Self-Attention via Sparse Adaptive Connection](https://arxiv.org/abs/2003.09833)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Xiaoya Li, Yuxian Meng, Mingxin Zhou, Qinghong Han, Fei Wu, Jiwei Li (2020). SAC: Accelerating and Structuring Self-Attention via Sparse Adaptive Connection. arXiv:2003.09833.

# Introduction

Standard self-attention in Transformers computes attention weights between every pair of tokens, yielding $O(n^2)$ time and memory complexity for sequence length $n$. SAC (Sparse Adaptive Connection) addresses this by reformulating the self-attention mechanism as a learned graph problem: the input sequence is treated as a graph where nodes are token representations, and the model learns which edges (token pairs) to attend to. This reduces complexity to $O(\alpha n)$, where $\alpha$ is a user-defined sparsity hyperparameter controlling the number of attention edges per node.

> [!NOTE]
> SAC is designed for practitioners who need to apply Transformer-like models to long sequences (NMT, language modeling, document-level tasks) where quadratic attention cost is prohibitive, or to graph-structured domains where structure should be learned rather than predefined.

SAC generalizes many existing sparse attention methods: vanilla self-attention, Transformer-XL, Adaptive Span Transformer, and BP-Transformer are all recoverable as special cases of SAC by constraining the edge predictor appropriately.

# Problem Formulation

Given an input sequence of $n$ token representations $\mathbf{X} \in \mathbb{R}^{n \times d}$, standard self-attention computes:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d_k}$ are query, key, value projections of $\mathbf{X}$.

SAC replaces the full attention matrix with a sparse binary adjacency matrix $\mathbf{A} \in \{0, 1\}^{n \times n}$ where each node connects to exactly $\alpha$ other nodes. The sparse attention then only computes scores for the $\alpha N$ selected pairs per layer, reducing memory from $O(n^2)$ to $O(\alpha n)$.

# Method: Sparse Adaptive Connection (SAC)

## LSTM Edge Predictor

The core of SAC is an LSTM that sequentially selects which token pairs form attention edges. For a layer with $n$ nodes, the predictor generates $2\alpha n$ edge endpoints (forming $\alpha n$ directed edges) one step at a time.

**Algorithm (Edge Prediction for one layer):**

```
Input:  Node representations H ∈ ℝ^{n × d}, sparsity α
Output: Adjacency set E with |E| = αn edges

Initialize LSTM hidden state h_0, cell state c_0
For t = 1 to 2αn:
    1. Compute attention over all nodes using h_{t-1}:
       score_i = MLP([h_{t-1}; H_i])  for i = 1..n
       prob = softmax(score)           ∈ ℝ^n
    2. Sample destination node: a_t ~ Categorical(prob)
    3. Form edge (source, a_t) and add to E
    4. Update LSTM: h_t, c_t = LSTM([h_{t-1}; H_{a_t}])
Return E
```

The LSTM conditions each selection on previously chosen nodes via the hidden state, which encourages diverse, non-redundant edge selections. Distance encoding adds a distance matrix $\mathbf{V} \in \mathbb{R}^{n \times n}$ to the projection matrices to incorporate structural position information.

## Training via REINFORCE

Because edge selection is discrete (non-differentiable), SAC uses the REINFORCE policy gradient algorithm to train the edge predictor jointly with the Transformer weights.

The policy gradient objective is:

$$\nabla J(\Theta) = \sum_{t} \nabla \log p(a_t \mid a_{1:t-1}; \Theta) \cdot (\mathcal{R}(\Theta) - b)$$

where:
- $a_t$ is the $t$-th sampled edge endpoint
- $\mathcal{R}(\Theta)$ is the task reward (e.g., log probability of the correct translation)
- $b$ is a baseline reward (average reward over recent batches) used to reduce variance

The full training objective combines cross-entropy loss for the main task with the REINFORCE gradient for the edge predictor.

## Model Variants

| Variant | Description |
|---|---|
| Shared structure | A single edge structure is predicted once and reused across all attention heads and layers |
| Head-adaptive | Each attention head independently predicts its own sparse edge structure |
| All-nodes-connected | Every node is forced to attend to at least one neighbor (prevents isolated nodes) |

## Relationship to Prior Sparse Attention Methods

| Method | Edge Predictor Constraint in SAC |
|---|---|
| Vanilla self-attention | $\mathbf{A}$ is fully dense ($\alpha = n$) |
| Transformer-XL | Edges limited to fixed-size local window plus memory segment |
| Adaptive Span Transformer | Edges selected by a learned scalar span parameter per head |
| BP-Transformer | Edges follow binary partitioning tree structure |

SAC subsumes all of these as constrained versions of its general edge predictor.

# Experiments

## Neural Machine Translation

- **Dataset**: WMT 2014 English-German (4.5M sentence pairs), tokenized with BPE
- **Baseline**: Transformer-big (28.4 BLEU on newstest2014)
- **Results**:
  - SAC Large ($\alpha = 10n$ edges): **28.9 BLEU** with significantly lower GPU memory usage than Transformer-big
  - SAC Large + dependency parse edges: **29.5 BLEU**, showing that linguistically-motivated structure can further help

## Language Modeling

- **Datasets**: Enwiki8 (character-level) and Text8 (character-level), standard splits
- **Metric**: Bits-per-character (BPC; lower is better)
- **Results**:
  - Enwiki8: **1.00 BPC** (head-adaptive SAC), matching or outperforming Transformer-XL, Adaptive Span, and BP-Transformer
  - Text8: **1.06 BPC**

## Graph Representation Learning

SAC replaces the fixed graph adjacency in Graph Attention Networks (GAT) with a learned sparse connection, applied on top of the input graph structure.

- **Datasets**: Cora, Citeseer, Pubmed (citation networks), PPI (protein-protein interaction)
- **Results** (improvement over GAT baseline):

| Dataset | Metric | SAC improvement |
|---|---|---|
| Cora | Accuracy | +1.8% |
| Citeseer | Accuracy | +1.1% |
| Pubmed | Accuracy | +0.7% |
| PPI | F1 score | +1.1% |

## Image Classification

SAC is applied to vision Transformers by replacing dense self-attention in image patch processing.

- **Datasets**: CIFAR-100, ImageNet
- **Results**:
  - CIFAR-100: **82.4% top-1** (+0.8% vs. prior sparse attention baseline)
  - ImageNet: **78.7% top-1** (+1.0% improvement)

# Input / Output Specification

| Component | Input | Output |
|---|---|---|
| Edge Predictor (LSTM) | Node representations $\mathbf{H} \in \mathbb{R}^{n \times d}$, sparsity $\alpha$ | Adjacency set $E$ with $\alpha n$ edges |
| Sparse Self-Attention | Selected $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ pairs for edges in $E$ | Context vectors $\mathbf{C} \in \mathbb{R}^{n \times d}$ |
| Full SAC Layer | Token sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$ | Updated representations $\mathbf{X}' \in \mathbb{R}^{n \times d}$ |

# Key Properties and Applicability

- **Who**: Researchers and engineers applying Transformer models to long sequences (NLP), graph-structured data, or image patches
- **When**: When $O(n^2)$ memory is prohibitive, or when task-specific attention structure (rather than full attention) is expected to help
- **Where**: Neural machine translation, language modeling, graph neural networks, image classification
- **Limitation**: REINFORCE training adds optimization complexity; reward signal may be noisy for some tasks
- **Sparsity tradeoff**: Higher $\alpha$ recovers more of the full attention capacity at proportionally higher cost; $\alpha = n$ recovers vanilla self-attention exactly
