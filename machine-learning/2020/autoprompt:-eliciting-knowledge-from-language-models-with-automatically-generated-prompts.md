# Meta Information

- URL: [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Shin, T., Razeghi, Y., Logan IV, R. L., Wallace, E., & Singh, S. (2020). AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. EMNLP 2020.

# AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts

AutoPrompt is a gradient-guided search method that automatically constructs prompts for pretrained masked language models (MLMs), enabling knowledge extraction without task-specific finetuning. The key insight is that MLMs contain factual, relational, and commonsense knowledge that is best accessed through fill-in-the-blank templates, and finding effective templates manually is slow and brittle. AutoPrompt addresses this by searching the discrete token vocabulary using gradients.

**Applicability**: Useful for practitioners who want to probe or use MLMs (BERT, RoBERTa) on classification, fact retrieval, or relation extraction tasks, especially in low-data regimes where finetuning is unstable or infeasible.

## Background and Motivation

Prior work (LAMA benchmark) showed that factual knowledge can be retrieved from pretrained MLMs using hand-crafted cloze-style prompts (e.g., "Paris is the capital of [MASK]"). However, manually designed prompts are sensitive to phrasing, non-transferable across models, and require human expertise. AutoPrompt replaces manual prompt engineering with a gradient-guided discrete search, building on the Universal Adversarial Triggers method (Wallace et al., 2019).

**Difference from finetuning**: Finetuning updates all model parameters and requires a separate checkpoint per task. AutoPrompt keeps the model frozen and only learns a small set of trigger tokens (typically 3–10) appended to the input, making it parameter-efficient and storage-friendly.

**Difference from manual prompts (LPAQA)**: LPAQA (Jiang et al., 2020) generates prompts using paraphrase mining and middle-word insertion. AutoPrompt uses gradient signals to search over the full vocabulary, yielding higher precision even though the resulting tokens may not be human-readable.

**Difference from probing classifiers**: Linear probes train a classifier on top of frozen representations but introduce their own parameters and risk attributing classifier capacity to the LM. AutoPrompt is parameter-free during inference.

## Method

### Template Structure

AutoPrompt reformulates every task as a masked language model (MLM) prediction problem. Each input $\mathbf{x}_{\text{inp}}$ is embedded in a prompt template:

```
[x_inp] [T] [T] ... [T] [P]
```

where $[T]$ denotes shared trigger tokens (same across all inputs) and $[P]$ is the `[MASK]` position at which the model predicts the output label. The number of trigger tokens is a hyperparameter (typically 3–6).

- **Input**: token sequence $\mathbf{x}_{\text{prompt}} = [\mathbf{x}_{\text{inp}}, \mathbf{x}_{\text{trig}}, \mathbf{x}_{[\text{MASK}]}]$
- **Output**: a probability distribution over the vocabulary at the `[MASK]` position via the MLM head

### Gradient-Guided Trigger Search

Trigger tokens are optimized iteratively using first-order token-embedding gradients. This avoids enumerating the full vocabulary by approximating the change in log-likelihood caused by substituting the current trigger token $\tilde{w}$ with a candidate $w$:

```math
\begin{align}
  \mathcal{V}_{\text{cand}} = \underset{w \in \mathcal{V}}{\text{top-}k}
  \left[ \mathbf{e}_w^\top \nabla_{\mathbf{e}_{\tilde{w}}} \log p(y \mid \mathbf{x}_{\text{prompt}}) \right]
\end{align}
```

where $\mathbf{e}_w \in \mathbb{R}^{d}$ is the input embedding of token $w$, and the gradient $\nabla_{\mathbf{e}_{\tilde{w}}} \log p(y \mid \mathbf{x}_{\text{prompt}}) \in \mathbb{R}^{d}$ is computed w.r.t. the current trigger token's embedding.

**Algorithm**:

```
Input: dataset D, template with n_T trigger positions, vocabulary V, top-k
Initialize: trigger tokens x_trig = [random tokens]
for each training batch:
    for each trigger position i in {1, ..., n_T}:
        compute gradient of log p(y | x_prompt) w.r.t. e_{trig_i}
        select top-k candidates V_cand via dot product with all word embeddings
        evaluate each candidate via forward pass
        replace trig_i with best candidate token
return x_trig
```

Each trigger update requires one backward pass plus $k$ forward passes. The total computational cost is comparable to standard gradient descent over a discrete vocabulary.

### Automated Label Token Selection

For tasks with abstract labels (e.g., "positive" vs. "negative" sentiment), AutoPrompt must map class labels to vocabulary tokens. Rather than manually choosing words like "great" or "terrible," it automates this via:

**Step 1 – Contextualized [MASK] embedding extraction**: For each training example, extract the contextual embedding $\mathbf{h}^{(i)} \in \mathbb{R}^{d}$ at the `[MASK]` position.

**Step 2 – Logistic regression on embeddings**:

```math
\begin{align}
  p(y \mid \mathbf{h}^{(i)}) \propto \exp(\mathbf{h}^{(i)} \cdot \mathbf{w}_y + \beta_y)
\end{align}
```

where $\mathbf{w}_y \in \mathbb{R}^{d}$ and $\beta_y \in \mathbb{R}$ are the learned class weights and bias.

**Step 3 – Token scoring**: Token $w$ is scored for class $y$ using the learned weights evaluated at the token's input embedding $\mathbf{e}_w$:

```math
\begin{align}
  s(y, w) = \mathbf{e}_w \cdot \mathbf{w}_y + \beta_y
\end{align}
```

**Step 4 – Label set construction**: Select the top-$k$ vocabulary tokens for each class $y$:

```math
\begin{align}
  \mathcal{V}_y = \underset{w \in \mathcal{V}}{\text{top-}k}\ [s(y, w)]
\end{align}
```

**Final class probability** (marginalizing over the label token set):

```math
\begin{align}
  p(y \mid \mathbf{x}_{\text{prompt}}) = \sum_{w \in \mathcal{V}_y} p([\text{MASK}] = w \mid \mathbf{x}_{\text{prompt}})
\end{align}
```

This allows the model to use multiple tokens per class (e.g., "good," "great," "excellent" for positive sentiment), increasing recall over single-token label assignment.

## Experiments

### Sentiment Analysis (SST-2)

- **Dataset**: Stanford Sentiment Treebank (SST-2), binary classification (positive/negative)
- **Model**: RoBERTa-large (355M parameters)
- **AutoPrompt result**: 91.4% accuracy with no finetuning
- **Comparison**: Outperforms manual prompts significantly; competitive with finetuned models
- **Hyperparameters searched**: candidate set size $k \in \{10, 100\}$, label set size $|\mathcal{V}_y| \in \{1, 3, 5\}$, trigger length $\in [3, 6]$

### Natural Language Inference (SICK-E)

- **Dataset**: SICK-E (Sentences Involving Compositional Knowledge – Entailment), ~10,000 sentence pairs, 3-way (entailment/contradiction/neutral) and 2-way variants
- **Model**: RoBERTa-large
- **AutoPrompt result**: 87.3% on 2-way classification
- **Hyperparameters searched**: candidate size $\in \{10, 50\}$, label size $\in \{1, 3, 5, 10\}$, trigger length $\in [1, 5]$

### Fact Retrieval (LAMA)

- **Dataset**: LAMA benchmark (Leslie et al., 2019), consisting of (subject, relation, object) triples from ConceptNet, Google-RE, T-REx, and SQuAD; 41 Wikidata relations tested
- **Training data**: Up to 1,000 facts per relation from T-REx
- **Metrics**: Precision@1 (P@1), Precision@10 (P@10), Mean Reciprocal Rank (MRR)
- **AutoPrompt result**: 43.3% P@1 vs. LPAQA ensemble 34.1% — a ~9 point improvement
- **Model**: BERT-large

### Relation Extraction (T-REx)

- **Dataset**: T-REx (subject–relation–object triples with supporting context sentences)
- **Comparison**: LSTM-based supervised RE baseline (Sorokin & Gurevych, 2017)
- **AutoPrompt result**: MLM with AutoPrompt achieves 90.73% vs. 57.95% for the supervised LSTM model
- **Perturbation experiment**: On perturbed data (artificially modified object entities), AutoPrompt performance degrades sharply, while a surface-form model does not — this confirms the MLM is relying on factual knowledge rather than surface patterns

### Low-Data Regime

AutoPrompt is compared to finetuning with 10–1,000 labeled examples. With as few as 10 examples, AutoPrompt achieves better average accuracy and lower variance than finetuning (which suffers from "failed runs" due to unstable optimization at very low data sizes).

## Limitations

- **Interpretability**: Generated trigger tokens (e.g., "orio Bauer 155 mente") are not human-readable, making it difficult to interpret what knowledge is being probed.
- **Labeled data dependency**: Despite avoiding finetuning, AutoPrompt still requires labeled examples to optimize trigger tokens.
- **Imbalanced datasets**: Performance degrades when class distributions are skewed.
- **Greedy local search**: The iterative one-at-a-time trigger token update does not guarantee global optimality over all trigger positions jointly.
- **Model-specific**: Triggers are optimized for a specific model and do not transfer across architectures.

> [!NOTE]
> The authors note: "Although AutoPrompt can elicit more knowledge than manual prompts, the resulting prompts are not always interpretable, which limits our ability to understand what the model has learned."

> [!TIP]
> Code and datasets: https://ucinlp.github.io/autoprompt/
