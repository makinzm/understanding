# Meta Information

- URL: [Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in NLP Using Fewer Parameters & Less Data](https://arxiv.org/abs/2009.09139)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Pilault, J., El hattami, A., & Pal, C. (2020). Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in NLP Using Fewer Parameters & Less Data. arXiv:2009.09139.

# Conditionally Adaptive Multi-Task Learning (CA-MTL)

CA-MTL is a parameter-efficient multi-task learning framework for NLP that modulates pretrained Transformer weights using learned task embeddings, combined with entropy-based task sampling to mitigate negative transfer. It targets practitioners who want to fine-tune a single shared model across many NLP tasks (8–26 tasks) without maintaining task-specific full models or suffering catastrophic forgetting.

## Problem Setting

Standard multi-task learning (MTL) on pretrained Transformers faces three interrelated problems:

1. **Overfitting on low-resource tasks**: Small datasets such as RTE (2.5k examples) are easily overfit when jointly trained with larger ones.
2. **Catastrophic forgetting**: Parameter sharing causes the model to lose performance on tasks it was previously good at.
3. **Negative transfer**: Some task pairs interfere, causing MTL to underperform even single-task fine-tuning.

CA-MTL addresses all three by (a) generating task-conditioned weights from a shared hypernetwork instead of maintaining separate weight sets, and (b) preferentially sampling tasks where model uncertainty is highest.

**Formal Objective (Eq. 1):**

$$\min_{\phi, \theta} \sum_{i=1}^{T} \mathcal{L}_i\bigl(f_{\phi(z_i), \theta_i}(x_i),\, y_i\bigr)$$

where $T$ is the number of tasks, $z_i \in \mathbb{R}^d$ is a learned task embedding for task $i$, $\phi(z_i)$ is a hypernetwork that generates task-conditioned weights, and $\theta_i$ are task-specific output-layer parameters (e.g., classification head).

**Conditional Weight Transformation (Definition 1):**

$$\phi(W \mid z_i) = \gamma_i(z_i)\, W + \beta_i(z_i)$$

The functions $\gamma_i(\cdot)$ and $\beta_i(\cdot)$ implement Feature-Wise Linear Modulation (FiLM): they map a task embedding $z_i \in \mathbb{R}^d$ to a scalar scale and shift for each weight matrix $W$.

## Architecture: Task-Conditioned Transformer

The base pretrained model (BERT or RoBERTa) is modified with four conditional components. The lower half of Transformer layers is kept **frozen** to preserve pretraining knowledge; conditional modules are inserted in or appended to the upper layers.

### 1. Conditional Attention

Standard dot-product attention is extended with a block-diagonal task-specific bias matrix $M(z_i) \in \mathbb{R}^{L \times L}$:

$$\text{Attention}(Q, K, V, z_i) = \text{softmax}\!\left[M(z_i) + \frac{QK^\top}{\sqrt{d}}\right] V$$

- $Q, K, V \in \mathbb{R}^{L \times d}$, where $L$ is sequence length and $d$ is head dimension.
- $M(z_i)$ is constructed as a direct sum of $N = d/L$ block matrices $\{A_n\}_{n=1}^{N}$.
- Each block is conditionally modulated: $A_n' = A_n \gamma_i(z_i) + \beta_i(z_i)$.
- The block-diagonal structure reduces parameters from $O(L^2)$ to $O(L^2/N^2)$ compared to a full bias matrix.

This allows the attention pattern to shift per task without retraining the full attention weights.

### 2. Conditional Alignment

A single shared alignment matrix $R \in \mathbb{R}^{V \times d}$ (where $V$ is vocabulary size) is conditioned on the task embedding to replace $T$ separate task-specific embedding matrices:

$$\hat{R}(z_i) = R\, \gamma_i(z_i) + \beta_i(z_i)$$

This alignment is applied between the embedding layer and the first Transformer layer. Using a single conditionally-modulated matrix instead of $T$ separate matrices yields a 1.26% improvement over the separate-matrix baseline on GLUE (3.61% vs. 2.35% improvement over BERT_BASE).

### 3. Conditional Layer Normalization (CLN)

Standard Layer Normalization statistics are overridden with task-conditioned scale and shift parameters:

$$h_i = \left(\frac{1}{\sigma} \odot (a_i - \mu)\right) \cdot \hat{\gamma}_i(z_i) + \hat{\beta}_i(z_i)$$

where $\hat{\gamma}_i(z_i) = \gamma'\, \gamma_i(z_i) + \beta'$, initialized from the pretrained LayerNorm weights $\gamma', \beta'$. The conditioning functions $\gamma_i(\cdot)$ and $\beta_i(\cdot)$ are trained during fine-tuning.

This extends conditional batch normalization (used in visual style transfer) to Transformer layer normalization.

### 4. Conditional Bottleneck

A two-layer feed-forward adapter module is appended to the upper Transformer layers. It applies the same FiLM transformation:

- **CA-MTL_BASE**: Adapter appended only to the topmost layers.
- **CA-MTL_LARGE**: Skip connections allow the adapter to integrate information across all layers.

The bottleneck has a down-projection to a smaller hidden size, applies task conditioning, then up-projects back, enabling efficient information routing without large parameter overhead.

### Architecture Summary

| Component | Location | Key Mechanism |
|---|---|---|
| Conditional Alignment | After embedding layer | FiLM on shared matrix $R$ |
| Conditional Attention | Multi-head attention in upper layers | Task-specific bias $M(z_i)$ |
| Conditional Layer Norm | All layer norms in upper layers | Task-conditioned $\gamma, \beta$ |
| Conditional Bottleneck | Appended to upper layers | Task-conditioned FFN adapter |

Parameter overhead: ~1.12× the pretrained model for 8 tasks (vs. 8× for 8 separate fine-tuned models, or 24× for 24 tasks).

## Multi-Task Uncertainty Sampling

To prevent catastrophic forgetting and mitigate negative transfer, CA-MTL uses an active-learning-inspired sampling strategy: at each training step, it selects the batch of samples across tasks where the model is least confident.

**Shannon Entropy per task:**

$$H_i = -\sum_{c=1}^{C_i} p_o \log p_o$$

where $C_i$ is the number of classes for task $i$ and $p_o$ is the predicted probability for class $o$.

**Normalized Uncertainty Score:**

$$\mathcal{U}(x_i) = \frac{H_i(f_{\phi(z_i), \theta_i}(x))}{\hat{H} \times H'_i}$$

- $\hat{H} = \max_{i \in \{1,\ldots,T\}} \bar{H}_i$: maximum average entropy across tasks (scales to the hardest task).
- $H'_i = -\sum_{c=1}^{C_i} \frac{1}{C_i} \log \frac{1}{C_i}$: uniform distribution entropy for task $i$ (normalizes for varying class counts).
- $\bar{H}_i = \frac{1}{b} \sum_{x \in x_i} H_i$: average entropy over a mini-batch of $b$ samples for task $i$.

**Algorithm: MT-Uncertainty Sampling**

```
Input: Tasks {T_1, ..., T_T}, batch size b, model f_{φ,θ}
For each training step:
  1. Sample b candidate examples per task → b×T total candidates
  2. Compute normalized uncertainty U(x_i) for each candidate
  3. Select top-b samples across all tasks by highest U(x_i)
  4. Update model parameters on the selected b-sample batch
```

This concentrates gradient updates on the samples where the model is least certain, naturally spending more time on low-resource or difficult tasks. Result: 66.3% of the total dataset is actively used for gradient updates, while achieving better performance than training on 100%.

> [!IMPORTANT]
> MT-Uncertainty differs from traditional active learning in that it selects one batch across all tasks jointly (not one sample per learner), and it accounts for differing number of classes per task via $H'_i$ normalization.

## Differences from Similar Methods

| Method | Weight Sharing | Task Conditioning | Uncertainty Sampling | Scales to 24+ Tasks |
|---|---|---|---|---|
| Single-task fine-tuning | None (separate models) | N/A | No | Requires 24× params |
| Standard MTL (shared top layers) | Full sharing | None | No | Degrades (−1.8% GLUE) |
| Adapter (Houlsby et al. 2019) | Shared pretrained + per-task adapters | None | No | Linear param growth |
| CA-MTL (ours) | Shared + conditionally modulated | Task embedding FiLM | Yes (entropy-based) | Improves (+4% GLUE) |

> [!NOTE]
> "Unlike existing approaches that insert multiple adapters, we learn a shared representation that is modulated by task embeddings, which allows for more efficient weight sharing and better generalization."

> [!TIP]
> The FiLM (Feature-wise Linear Modulation) technique was originally proposed for visual question answering: [FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018)](https://arxiv.org/abs/1709.07871). CA-MTL extends this idea to Transformer weight modulation in NLP.

## Experiments

### Datasets

| Group | Tasks | Notes |
|---|---|---|
| GLUE | CoLA, MNLI-m/mm, MRPC, QNLI, QQP, RTE, SST-2, STS-B | Standard NLP benchmark (8 tasks) |
| SuperGLUE | BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC | More challenging NLU tasks (8 tasks) |
| MRQA | SQuAD, NewsQA, TriviaQA, SearchQA, HotpotQA, NaturalQuestions | Machine reading comprehension (6 tasks) |
| NER | WNUT2017 | Token-level named entity recognition |
| Transfer | SciTail, SNLI | Zero-shot cross-task adaptation |

The 24-task model jointly trains on GLUE + SuperGLUE + MRQA + WNUT2017 simultaneously.

- Hardware: Not explicitly specified in the paper.
- Optimizer: Adam with warmup (following BERT fine-tuning conventions).
- Batch size: $b = 32$; 3 training epochs for 24-task experiments.
- Layer freezing: Bottom 6 of 12 layers frozen for BERT_BASE; similar strategy for LARGE.

### Key Results

**8-Task GLUE (test set):**
- CA-MTL_BERT-BASE: **80.9** avg (vs. 79.6 single-task BERT_BASE, vs. 77.5 baseline MTL)
- CA-MTL_BERT-LARGE: **82.8** avg (vs. 82.1 single-task BERT_LARGE)
- Outperforms other adapter-based MTL methods by ~2.8%

**24-Task Performance:**
- BERT_LARGE: **86.6** GLUE avg (+2.3% over single-task; baseline MTL degrades −1.8%)
- RoBERTa_LARGE: **89.4** GLUE avg (+1.2% over single-task)
- CA-MTL achieves parity or exceeds single-task models on 15–17 of 24 tasks

**State-of-the-Art Results:**
- WNUT2017 NER: **58.0 F1** (vs. XLM-R_LARGE: 57.1)
- SNLI: **92.1%** accuracy (vs. SemBERT: 91.9%)
- SciTail: **96.8%** (matches ALUM_RoBERTa-SMART)

**Ablation Study (GLUE avg, BERT_BASE):**

| Added Component | Avg GLUE | Task σ | % Data Used |
|---|---|---|---|
| Baseline MTL | 80.61 | 14.41 | 100% |
| + Conditional Attention | 82.41 | 10.67 | 100% |
| + Conditional Bottleneck | 82.90 | 11.27 | 100% |
| + Conditional Layer Norm | 83.12 | 10.91 | 100% |
| + MT-Uncertainty | **84.03** | **10.02** | **66.3%** |

Each component contributes incrementally; MT-Uncertainty reduces data usage to 66% while improving scores.
