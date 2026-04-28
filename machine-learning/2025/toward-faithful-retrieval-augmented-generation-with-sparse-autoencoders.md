# Meta Information

- URL: [Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](https://arxiv.org/abs/2512.08892)
- LICENSE: [Deed - Attribution 4.0 International - Creative Commons](https://creativecommons.org/licenses/by/4.0/)
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

```math
\begin{align}
  z_t = \mathcal{E}(h_t), \quad z_t \in \mathbb{R}^{K}, \quad K \gg d
\end{align}
```

where $\mathcal{E}$ is the SAE encoder and $K$ is the dictionary size. Sparsity is enforced via Top-K or ReLU activation, so most $z_{t,k} = 0$. Each non-zero dimension corresponds to a monosemantic feature — an interpretable concept (e.g., "unsupported numeric", "temporal inconsistency") learned without supervision.

> [!NOTE]
> Pre-trained SAEs are used off-the-shelf from the EleutherAI SAE library (Llama2-7B, Llama3.1-8B) or trained by the authors (Llama2-13B, Qwen3-4B). No fine-tuning of the base LLM or SAE is performed.

> [!TIP]
> See Elhage et al. (2022) "Toy Models of Superposition" and Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable Features in Language Models" for foundational SAE background.

## Method: Three-Stage Pipeline

### Stage 1 — Instance-Level Feature Summarization

Given $T$ generated tokens, the SAE produces activations $z_t \in \mathbb{R}^{K}$ at each token position. To obtain a fixed-size instance representation, channel-wise max pooling is applied:

```math
\begin{align}
  \bar{z}_k = \max_{1 \le t \le T} z_{t,k}, \quad k = 1, \ldots, K
\end{align}
```

- Input: token-level activations $\{z_t\}_{t=1}^{T}$, each $z_t \in \mathbb{R}^{K}$
- Output: instance-level representation $\bar{z} \in \mathbb{R}^{K}$

**Why max pooling?** The paper proves (Theorem 1) that in sparse activation regimes where $T \cdot \bar{p} \ll 1$ (with $\bar{p}$ as average per-token activation probability), max pooling amplifies discriminative signal linearly with sequence length while suppressing random activation noise. Mean pooling, by contrast, dilutes rare hallucination-correlated activations.

### Stage 2 — Information-Based Feature Selection

From $K$ features (up to $K = 32 \times d$ in practice), only a subset carry hallucination-relevant information. Mutual information between each pooled feature $\bar{z}_k$ and the binary label $\ell \in \{0, 1\}$ is estimated:

```math
\begin{align}
  I(\bar{z}_k; \ell) = \int_{\mathbb{R}} \sum_{\ell \in \{0,1\}} p(\bar{z}_k, \ell) \log_2 \frac{p(\bar{z}_k, \ell)}{p(\bar{z}_k)\, p(\ell)} \, d\bar{z}_k
\end{align}
```

Estimated via histogram binning (50 bins). The top $K' = 1000$ features by MI score are retained to form $\tilde{z} \in \mathbb{R}^{K'}$.

> [!IMPORTANT]
> MI-based selection (not random or variance-based) is critical. Ablations show that random selection degrades sharply at small $K'$, while MI selection remains robust down to $K' = 20$.

### Stage 3 — Transparent Prediction via GAM

A Generalized Additive Model (GAM) maps selected features to a hallucination probability:

```math
\begin{align}
  g\bigl(\mathbb{E}[\ell \mid \tilde{z}]\bigr) = \beta_0 + \sum_{j=1}^{K'} f_j(\tilde{z}_j)
\end{align}
```

where $g$ is the logit link function and each $f_j$ is a univariate shape function fitted by bagged gradient boosting (Explainable Boosting Machine, EBM). Training: 32 max bins per feature, 1000 max boosting rounds, 10% validation split.

- Input: $\tilde{z} \in \mathbb{R}^{K'}$
- Output: predicted hallucination probability $\hat{p} \in [0,1]$

**Why GAM over MLP?** The additive constraint enforces interpretability: $f_j(\tilde{z}_j)$ directly shows the contribution of feature $j$ to the hallucination score, enabling global feature explanations. Despite this constraint, GAM outperforms MLP and XGBoost in ablations because cross-feature interactions add noise rather than signal for this task.

## Theoretical Justification of Max Pooling

**Theorem 1 (Max Pooling Signal Amplification in Sparse-Activation Regime):**

Let $p_\ell = P(z_{t,k} > 0 \mid \ell)$ be the per-token activation probability given label $\ell$, and $\bar{p} = (p_0 + p_1)/2$ the average activation rate. When $T \cdot \bar{p} \ll 1$:

```math
\begin{align}
  I(\bar{z}; \ell) = \frac{\pi(1-\pi)}{2\ln 2} \cdot \frac{T(\Delta p)^2}{\bar{p}} + O\!\left((T\bar{p})^2\right)
\end{align}
```

where:
- $\pi = P(\ell = 1)$ — base rate of hallucination
- $\Delta p = p_1 - p_0$ — difference in activation probability between hallucinated ($\ell=1$) and faithful ($\ell=0$) outputs
- $T$ — sequence length
- $\bar{p}$ — average per-token activation probability (small in sparse regime)

This shows MI scales as $T \cdot (\Delta p)^2 / \bar{p}$: rare, discriminative features are amplified rather than washed out, providing formal justification for max pooling over mean pooling.

## Algorithm Summary

```
Input:
  RAG instance (query q, retrieved docs C, generated answer y_{1:T})
  LLM with pre-trained SAE at layer L
  Labeled training set {(y_i, C_i, ℓ_i)}

--- TRAINING ---
1. For each training instance i:
   a. Forward pass: extract hidden states {h_t} at layer L, h_t ∈ R^d
   b. Apply SAE encoder: z_t = E(h_t), z_t ∈ R^K  (Top-K or ReLU)
   c. Max pool: z̄_k = max_{1≤t≤T} z_{t,k}  → z̄ ∈ R^K
2. Estimate MI(z̄_k; ℓ) for each k via 50-bin histogram
3. Select top K' = 1000 features by MI → project to z̃ ∈ R^{K'}
4. Fit EBM-GAM on {(z̃_i, ℓ_i)} with bagged gradient boosting

--- INFERENCE ---
1. Extract {h_t} at layer L for new instance
2. Apply SAE encoder and max pool → z̄ ∈ R^K, then project → z̃ ∈ R^{K'}
3. Predict: ŷ = sigmoid(β_0 + Σ_j f_j(z̃_j))
4. Return: hallucination score ŷ, top contributing features {(j, f_j(z̃_j))}
```

## Comparison with Related Methods

| Method | External LLM? | Fine-tunes LLM? | Interpretable? | AUC (RAGTruth, Llama2-7B) |
|--------|:---:|:---:|:---:|:---:|
| ReDeEP (2024) | No | Partial | No | 0.7458 |
| SEP (2023) | No | No | No | 0.7143 |
| ITI (2023) | No | Yes | No | 0.6714 |
| GPT-4o judge (CoT) | Yes | No | Partially | ~0.75 |
| **RAGLens (ours)** | No | No | **Yes** | **0.8413** |

> [!TIP]
> SEP (Semantic Entropy Probing) probes LLM hidden states directly for uncertainty. RAGLens differs by using SAE-disentangled features rather than raw hidden states, concentrating discriminative signal and enabling feature-level interpretation.

Key differences from ReDeEP (previous SOTA): RAGLens uses SAE features (interpretable, overcomplete dictionary) rather than raw hidden states, adds a formal theoretical justification for max pooling, and provides token-level attribution for targeted mitigation.

## Hallucination Mitigation

Once a hallucinated instance is detected, RAGLens provides two levels of interpretability feedback to prompt the LLM to revise:

**Instance-level feedback:** Inform the model which top-activated SAE features fired (e.g., "Feature 22790 — unsupported numeric specifics") and ask it to regenerate the answer avoiding those patterns.

**Token-level feedback:** Attribute feature contributions to individual tokens using a per-token score:

```math
\begin{align}
  s_t = \sum_{k=1}^{K} f_k(z_{t,k})
\end{align}
```

Tokens with the highest hallucination attribution score $s_t$ are highlighted in the prompt, guiding the model to specifically revise those claims. This approach converted 36 responses from hallucinated to faithful (vs. 29 for instance-level feedback) out of 450 tested responses.

## Experiments

### Datasets

| Dataset | Description | Size / Notes |
|---|---|---|
| RAGTruth | RAG hallucination benchmark: summarization, QA, and data-to-text tasks with binary faithfulness labels | ~18,000 LLM responses |
| Dolly (Accurate Context) | Instruction-following with retrieved context, faithfulness evaluation | Two-fold cross-validation |
| AggreFact | SOTA summarization faithfulness benchmark | Cross-dataset generalization target |
| TofuEval (MeetingBank) | Topic-focused dialogue summarization faithfulness benchmark | Cross-dataset generalization target |

### Models and SAE Configurations

| Model | Layer Used | SAE Expansion | Activation | Dictionary Size $K$ |
|---|---|---|---|---|
| Llama2-7B | 15 | 32× | Top-K ($K_{\text{active}}=192$) | $32d$ |
| Llama2-13B | 15 | 16× (custom trained) | Top-K ($K_{\text{active}}=16$) | $16d$ |
| Llama3.1-8B | 19 | 16× | ReLU | $16d$ |
| Llama3.2-1B | — | Public SAE | — | — |
| Qwen3-0.6B | — | Public SAE | — | — |
| Qwen3-4B | 22 | 32× | Top-K ($K_{\text{active}}=16$) | $32d$ |

- Hardware: Not explicitly stated in the paper.
- Optimizer: Bagged gradient boosting (EBM) for GAM; Adam assumed for custom SAE training.

### Main Detection Results

RAGLens substantially outperformed all baselines:
- Llama2-7B on RAGTruth: AUC **0.8413**, Accuracy **75.76%**, F1 **0.7636**
- Llama2-7B on Dolly: AUC **0.8764**, Accuracy **77.78%**, F1 **0.8070**
- Llama2-13B on RAGTruth: AUC **0.8964**, Accuracy **83.33%**, F1 **0.8148**
- Previous SOTA (ReDeEP on Llama2-7B/RAGTruth): AUC **0.7458** — RAGLens surpasses by +0.0955 AUC.

SAE-based detection consistently outperformed chain-of-thought self-judgment across all model scales (Llama3.2-1B, Llama3.1-8B, Qwen3-0.6B/4B), with Pearson correlation of **0.6731** ($p < 0.05$) between predicted and actual feature activations in counterfactual perturbation studies.

### Cross-Dataset Generalization

A detector trained only on RAGTruth achieved:
- **0.8019 AUC** on AggreFact (cross-dataset transfer, no retraining)
- **0.8191 AUC** on TofuEval/MeetingBank (strongest subtask generalization, summary subtask)

### Mitigation Results (450 Llama2-7B outputs)

Percentage of outputs still rated unfaithful after RAGLens feedback (lower is better):

| Judge | Original | Instance-Level Feedback | Token-Level Feedback |
|---|---|---|---|
| Llama3.3-70B | 43.78% | 42.22% | **39.11%** |
| Human | 71.11% | 62.22% | **55.56%** |

Token-level feedback reduced hallucination rate by an additional ~7 percentage points compared to instance-level feedback.

### Ablations

**Layer selection:** Mid-layer SAEs (normalized depth ~0.4–0.6, i.e., layers 15–19 for 7–8B models) are optimal for summarization and QA. Data-to-text tasks show flat performance across layers.

**Pre- vs. post-activation features:** Pre-activation (before ReLU/Top-K nonlinearity) consistently outperforms post-activation for both SAE and Transcoder architectures.

**Predictor comparison:** GAM > MLP > XGBoost > Logistic Regression — per-feature nonlinearity matters, but cross-feature interactions add noise.

**Feature count sensitivity:** MI-based selection is robust down to $K' = 20$; random selection collapses sharply below $K' = 500$.

## Interpretability: Example Features

| Feature ID | Semantic Interpretation | Correlation with Hallucination |
|---|---|---|
| 22790 | Activates on "unsupported numeric/time specifics" | Positive (high activation → likely hallucination) |
| 17721 | Activates on "grounded, high-salience tokens" | Negative (high activation → likely faithful) |

Counterfactual validation confirmed these features respond to RAG-specific grounding: replacing retrieved passages with unrelated documents shifts activations in the predicted direction, ruling out surface-level pattern matching as the explanation.

## Key Takeaways

1. **LLMs encode faithfulness internally:** SAE-based detection outperforms explicit chain-of-thought reasoning, confirming that hallucination information is present in intermediate representations but not always surfaced in generated text.
2. **Sparsity enables signal amplification:** Theorem 1 proves max pooling over sparse features concentrates label-relevant information, providing formal justification for the design choice.
3. **Compact feature sets suffice:** $K' = 1000$ (or even $K' = 20$) features selected by MI maintain near-peak performance, enabling lightweight deployment without the full SAE dictionary.
4. **Token-level attribution improves mitigation:** Pinpointing which specific tokens drove hallucination features enables more targeted regeneration than instance-level feedback alone.
5. **Middle layers are most informative:** Hallucination signals concentrate at intermediate depths (~40–60% of total layers), consistent with broader mechanistic interpretability findings.

## Code

Implementation: [gzxiong/RAGLens: ICLR 2026](https://github.com/gzxiong/RAGLens)
