# Meta Information

- URL: [Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](https://arxiv.org/abs/2512.08892)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Xiong, G., He, Z., Liu, B., Sinha, S., & Zhang, A. (2025). Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders. arXiv:2512.08892.

# RAGLens: Faithful Retrieval-Augmented Generation with Sparse Autoencoders

## Overview

**RAGLens** is a lightweight, training-efficient hallucination detector for retrieval-augmented generation (RAG) systems. It leverages sparse autoencoder (SAE) features extracted from intermediate LLM hidden states to identify unfaithful outputs — generations that contradict or are unsupported by retrieved documents. Unlike prior detectors that require extensive labeled data or expensive external LLM judges, RAGLens uses a three-stage pipeline: instance-level feature aggregation via max pooling, mutual information (MI)-based feature selection, and a generalized additive model (GAM) classifier.

**Applicable when:**
- An engineer deploys a RAG pipeline using an open-weight LLM (e.g., Llama2, Llama3) and needs to detect hallucinations post-generation without retraining the base model.
- An ML researcher wants interpretable, local explanations (which SAE features triggered unfaithful generation) at inference time.
- A production system requires a lightweight probe (not another LLM call) that generalizes across domains with minimal labeled data.

## Background: Sparse Autoencoders in LLM Interpretability

A Sparse Autoencoder (SAE) is trained on LLM hidden states $h_t \in \mathbb{R}^{d}$ to produce a sparse, overcomplete representation:

$$z_t = \mathcal{E}(h_t), \quad z_t \in \mathbb{R}^{K}, \quad K \gg d$$

where $\mathcal{E}$ is the SAE encoder and $K$ is the dictionary size. Sparsity is enforced via Top-K or ReLU activation, so most $z_{t,k} = 0$. Each non-zero dimension corresponds to a monosemantic feature — an interpretable concept (e.g., "unsupported numeric", "temporal inconsistency") learned without supervision.

> [!NOTE]
> Pre-trained SAEs are used off-the-shelf from the EleutherAI SAE library (Llama2-7B, Llama3.1-8B) or trained by the authors (Llama2-13B, Qwen3-4B). No fine-tuning of the base LLM or SAE is performed.

## Method: Three-Stage Pipeline

### Stage 1 — Instance-Level Feature Summarization

Given $T$ generated tokens, the SAE produces activations $z_t \in \mathbb{R}^{K}$ at each token position. To obtain a fixed-size instance representation:

$$\bar{z}_k = \max_{1 \le t \le T} z_{t,k}, \quad k = 1, \ldots, K$$

Channel-wise max pooling preserves the maximum activation magnitude for each feature across the full generation, producing $\bar{z} \in \mathbb{R}^{K}$.

**Why max pooling?** The paper proves (Theorem 1) that in sparse activation regimes ($T \cdot \bar{p} \ll 1$, where $\bar{p}$ is average activation probability), the mutual information between $\bar{z}_k$ and hallucination label $\ell$ scales as:

$$I(\bar{z}; \ell) \approx \frac{\pi(1-\pi)}{2 \ln 2} \cdot \frac{T(\Delta p)^2}{\bar{p}}$$

where $\pi = P(\ell=1)$ and $\Delta p = p_1 - p_0$ is the activation probability difference between faithful/unfaithful instances. Max pooling thus amplifies discriminative signals linearly with sequence length while suppressing random activation noise.

### Stage 2 — Information-Based Feature Selection

From $K$ features (up to $K = 32 \times d$ in practice), only a subset carry hallucination-relevant information. Mutual information between each pooled feature and the binary label is estimated:

$$I(\bar{z}_k; \ell) = \int_{\mathbb{R}} \sum_{\ell \in \{0,1\}} p(\bar{z}_k, \ell) \log_2 \frac{p(\bar{z}_k, \ell)}{p(\bar{z}_k) p(\ell)} \, d\bar{z}_k$$

estimated via histogram binning (50 bins). The top $K' = 1000$ features by MI score are retained to form $\tilde{z} \in \mathbb{R}^{K'}$.

> [!IMPORTANT]
> MI-based selection (not random or variance-based) is critical. Ablations show that random selection degrades sharply at small $K'$, while MI selection remains robust.

### Stage 3 — Transparent Prediction via GAM

A Generalized Additive Model (GAM) maps selected features to a hallucination probability:

$$g\bigl(\mathbb{E}[\ell \mid \tilde{z}]\bigr) = \beta_0 + \sum_{j=1}^{K'} f_j(\tilde{z}_j)$$

where $g$ is the logit link function and each $f_j$ is a univariate shape function fitted by bagged gradient boosting (Explainable Boosting Machine, EBM). Training configuration: 32 max bins per feature, 1000 max boosting rounds, 10% validation split.

**Why GAM over MLP?** The additive constraint enforces interpretability: $f_j(\tilde{z}_j)$ directly shows the contribution of feature $j$ to the hallucination score, enabling global feature explanations. Despite this constraint, GAM outperforms MLP and XGBoost in ablations.

## Algorithm Summary

```
Input: RAG instance (query q, retrieved docs D, generated answer a)
       LLM with SAE pre-trained on layer L
       Labeled training set {(a_i, D_i, l_i)}

Training:
1. For each training instance i:
   a. Forward pass: extract hidden states {h_t} at layer L
   b. Apply SAE encoder: z_t = E(h_t), z_t ∈ R^K
   c. Max pool: z̄_k = max_t z_{t,k}  → z̄ ∈ R^K
2. Estimate MI(z̄_k; ℓ) for each k via histogram binning
3. Select top K' = 1000 features by MI → z̃ ∈ R^{K'}
4. Fit EBM-GAM on {(z̃_i, l_i)}

Inference:
1. Extract z̄ for new instance, project to K' features
2. Predict: ŷ = sigmoid(β_0 + Σ_j f_j(z̃_j))
3. Return hallucination score ŷ and top contributing features {f_j}
```

## Comparison with Related Methods

| Method | External LLM? | Fine-tunes LLM? | Interpretable? | AUC (RAGTruth) |
|--------|:---:|:---:|:---:|:---:|
| ReDeEP (2024) | No | Partial | No | 0.7458 |
| SEP (2023) | No | No | No | 0.7143 |
| ITI (2023) | No | Yes | No | 0.6714 |
| GPT-4o judge | Yes | No | No | ~0.75 |
| **RAGLens (ours)** | No | No | **Yes** | **0.8413–0.8964** |

> [!TIP]
> SEP (Semantic Entropy Probing) probes LLM hidden states directly for uncertainty. RAGLens differs by using SAE-disentangled features rather than raw hidden states, which concentrates discriminative signal and enables feature-level interpretation.

## Experiments

- **Datasets:**
  - RAGTruth (~18,000 instances): QA, summarization, data-to-text tasks with faithfulness labels
  - Dolly (Accurate Context subset): general instruction-following with retrieved context
  - AggreFact: news summarization faithfulness benchmark
  - TofuEval (MeetingBank subset): topic-focused dialogue summarization
- **Hardware:** Not explicitly specified
- **Optimizer:** EBM bagged gradient boosting (not gradient-based neural optimizer)
- **Models evaluated:**
  - Llama2-7B (Layer 15, expansion 32, Top-K K=192)
  - Llama2-13B (Layer 15, expansion 16, Top-K K=16)
  - Llama3.1-8B (Layer 19, expansion 16, ReLU activation)
  - Qwen3-4B (Layer 22, expansion 32, Top-K K=16)
- **Key results:**
  - Llama2-7B on RAGTruth: AUC 0.8413, Accuracy 0.7576, F1 0.7636
  - Llama2-13B on RAGTruth: AUC 0.8964, Accuracy 0.8333, F1 0.8148
  - Outperforms 25+ baselines including GPT-4o-based chain-of-thought judges
  - Token-level feedback from RAGLens reduces hallucination rate by 8–15% when used as signal to downstream LLM judges
- **Ablation findings:**
  - Mid-layer SAE features (layer ~15–19 for 7–8B models) carry the strongest hallucination signals
  - Pre-activation features consistently outperform post-activation
  - MI-based feature selection is robust; random selection collapses at low K'
  - GAM > MLP > XGBoost > Logistic Regression in predictive performance

## Interpretability Features

RAGLens provides two levels of explanation:

1. **Local (instance-level):** The shape function contributions $\{f_j(\tilde{z}_j)\}$ identify which SAE features most influenced a specific prediction (e.g., "feature #4221 activated strongly → unsupported numeric claim").

2. **Global (dataset-level):** Ranking SAE features by average MI across the dataset reveals common hallucination patterns: unsupported numerics, temporal inconsistencies, and ungrounded named entities are the top-ranked feature clusters.
