# Meta Information

- URL: [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Shin, T., Razeghi, Y., Logan IV, R. L., Wallace, E., & Singh, S. (2020). AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. EMNLP 2021.

# Introduction

AutoPrompt is a gradient-based method for automatically constructing prompts that elicit knowledge from pretrained masked language models (MLMs) without any fine-tuning. Prior work on probing MLMs for factual or commonsense knowledge relied on either hand-crafted prompts — which are sensitive to exact wording and tedious to write — or specialized probe classifiers that add trainable parameters atop the frozen model. AutoPrompt instead reformulates tasks as fill-in-the-blank (cloze-style) problems and uses the MLM's own gradients to discover which token sequences best expose the model's stored knowledge.

> [!NOTE]
> The paper's central claim: "we develop AutoPrompt, an automated method to create prompts for a diverse set of tasks based on a gradient-based search over a shared vocabulary." This differs from probing classifiers (which add new parameters) and from manual templates (which are brittle to phrasing).

## Applicability

AutoPrompt is designed for:
- **Who**: NLP researchers probing what factual, commonsense, or relational knowledge is encoded in a pretrained MLM (BERT, RoBERTa), and practitioners who need lightweight multi-task deployment.
- **When**: When labeled data is available but fine-tuning is impractical (e.g., storage constraints, extreme low-data regimes).
- **Where**: Any task that can be cast as a fill-in-the-blank prediction — sentiment analysis, NLI, fact retrieval, relation extraction.

# Background and Notation

Given a masked language model $f_\theta$ with vocabulary $\mathcal{V}$, AutoPrompt maps an input $x_\text{inp}$ to a prompted version $x_\text{prompt}$ using a template $\lambda$.

| Symbol | Meaning |
|---|---|
| $x_\text{inp}$ | Original task input (text) |
| $x_\text{trig} = [t_1, \ldots, t_m]$ | Shared trigger tokens, $t_i \in \mathcal{V}$ |
| $x_\text{prompt} = \lambda(x_\text{inp}, x_\text{trig})$ | Full prompt fed to the MLM |
| $[\text{MASK}]$ | Single masked position where the MLM predicts |
| $\mathcal{V}_y$ | Set of label tokens assigned to class $y$ |
| $w_\text{in}, w_\text{out}$ | Input and output embedding vectors of token $w$ |
| $h^{(i)}$ | Contextualized embedding at position $i$ from the transformer |

A template is a fixed-structure string that concatenates the task input with the trigger tokens and a `[MASK]` slot. For example, for sentiment analysis:

```
{sentence} [T] [T] [T] [T] [T] [P].
```

where `[T]` are trigger positions and `[P]` is the `[MASK]` prediction slot.

## Class Probability via Label Token Marginalization

Because MLMs predict individual tokens at `[MASK]`, classes must be mapped to vocabulary tokens. The probability for class $y$ given a prompt is obtained by summing over all tokens assigned to that class:

```math
\begin{align}
  p(y \mid x_\text{prompt}) = \sum_{w \in \mathcal{V}_y} p([\text{MASK}] = w \mid x_\text{prompt})
\end{align}
```

- **Input**: Prompted sequence $x_\text{prompt} \in \mathcal{V}^*$, label token sets $\{\mathcal{V}_y\}$.
- **Output**: Scalar class probability $p(y \mid x_\text{prompt}) \in [0,1]$.

# Gradient-Based Prompt Search

## Algorithm

AutoPrompt uses a first-order gradient approximation (identical to the HotFlip token-swap attack) to efficiently search for trigger tokens that maximize log-likelihood.

**Pseudocode:**

```
Input:  dataset D = {(x_inp^(i), y^(i))},
        template λ with m trigger positions,
        number of candidates k, iterations T
Output: best trigger sequence x_trig*

Initialize: each trigger token t_j ← [MASK]

for iteration 1..T:
    sample mini-batch B from D
    for each trigger position j = 1..m:
        # Compute gradient w.r.t. token embedding at position j
        g_j ← ∇_{e(t_j)} log p(y | x_prompt)   # averaged over B
        # Rank all vocabulary tokens by first-order gain
        V_cand^(j) ← top-k_{w ∈ V} [ w_in^T · g_j ]
    # Evaluate all k candidates per position on a fresh batch B'
    for each candidate (j, w) in V_cand:
        score(j, w) ← log p(y | x_prompt with t_j ← w)   # on B'
    # Accept the swap with highest score
    (j*, w*) ← argmax_{(j,w)} score(j, w)
    t_{j*} ← w*
    track best x_trig seen on dev set

return x_trig*
```

**Gradient approximation detail:**

```math
\begin{align}
  \mathcal{V}_\text{cand}^{(j)} = \operatorname{top-k}_{w \in \mathcal{V}}\!\left[w_\text{in}^\top \nabla_{e(t_j)} \log p(y \mid x_\text{prompt})\right]
\end{align}
```

The inner product $w_\text{in}^\top g_j$ estimates the change in log-likelihood if $t_j$ were replaced by $w$, using only one forward-backward pass per trigger position.

- **Input at each step**: Mini-batch of (input, label) pairs; current trigger tokens.
- **Output**: Updated trigger sequence with one position improved per iteration.

> [!NOTE]
> This is identical in form to HotFlip (Ebrahimi et al., 2018), which was originally designed as an adversarial attack. AutoPrompt repurposes the same mechanics constructively to improve task performance.

## Comparison with Related Prompting Methods

| Method | Requires Fine-tuning? | Requires Manual Design? | Label tokens |
|---|---|---|---|
| Manual prompts (LAMA) | No | Yes (human labor) | Fixed keywords |
| Soft prompts / prefix-tuning | Yes (prompt params) | No | Continuous |
| **AutoPrompt** | No | No | Automated via search |
| Finetuning full model | Yes | No | Classification head |

> [!IMPORTANT]
> AutoPrompt occupies a unique niche: it uses gradient information but does **not** update model weights. Multiple task prompts can be served from a single frozen model by storing only the trigger sequences (a few dozen tokens each), not separate checkpoints.

# Automating Label Token Selection

For tasks with abstract class labels (e.g., "entailment", "positive"), meaningful vocabulary tokens must be assigned to each class. AutoPrompt automates this in two steps.

**Step 1 — Train a logistic classifier on contextualized mask embeddings:**

```math
\begin{align}
  p(y \mid h^{(i)}) \propto \exp\!\left(h^{(i)} \cdot \mathbf{y} + \beta_y\right)
\end{align}
```

where $h^{(i)} \in \mathbb{R}^d$ is the transformer's output at the `[MASK]` position, $\mathbf{y} \in \mathbb{R}^d$ is a learned weight vector per class, and $\beta_y$ is a scalar bias. The classifier is trained on the training set with the trigger tokens fixed to their initial values.

**Step 2 — Score vocabulary tokens against learned class vectors:**

```math
\begin{align}
  s(y, w) &= p(y \mid w_\text{out}) \\
  \mathcal{V}_y &= \operatorname{top-k}_{w \in \mathcal{V}}\, s(y, w)
\end{align}
```

Tokens with output embeddings $w_\text{out} \in \mathbb{R}^d$ most aligned with the learned class weight $\mathbf{y}$ are selected as label tokens.

- **Input**: Vocabulary embeddings $\{w_\text{out}\}$; trained logistic weights $\{\mathbf{y}, \beta_y\}$.
- **Output**: Label token sets $\{\mathcal{V}_y\}$, one per class.

> [!NOTE]
> An alternative is to manually inspect the top-k tokens and choose intuitive ones; the authors compare both and find automated selection is typically comparable or better.

# Experiments

## Datasets

| Task | Dataset | Split info |
|---|---|---|
| Sentiment Analysis | SST-2 | Standard train/dev/test |
| Natural Language Inference | SICK-E | Full dataset (balanced 3-way; authors also use 2-way subset) |
| Fact Retrieval (probing) | LAMA (T-REx + Google-RE + ConceptNet + SQuAD) | 41 relations, ~1M+ triples |
| Relation Extraction | T-REx (from LAMA) | Treated as supervised RE task |

## Models

- BERT-large-cased
- RoBERTa-large

## Key Results

**Sentiment Analysis (SST-2):**
- AutoPrompt (RoBERTa): 91.4% accuracy — outperforms manual prompts (85.2%) and approaches fine-tuned performance
- In very low-data regimes (10–100 examples), AutoPrompt outperforms standard fine-tuning

**Natural Language Inference (SICK-E, 2-way):**
- BERT: 85.7%, RoBERTa: 87.3%
- Comparable to probing with a linear classifier on the same representations

**Fact Retrieval (LAMA T-REx):**
- AutoPrompt achieves 43.3% P@1, vs. 34.1% for prior best (manual prompts from LAMA)
- Consistent gains across relation types; RoBERTa generally outperforms BERT

**Relation Extraction (T-REx):**
- BERT: 90.73% mean precision, exceeding supervised baselines
- When input sentences are perturbed to remove entity mentions, accuracy drops substantially — confirming the model relies on factual knowledge rather than surface extraction patterns

## Hardware / Optimizer

Not explicitly stated in the paper for final runs; gradient search is efficient (≈ one forward-backward pass per candidate iteration), making it far lighter than full fine-tuning.

# Discussion

## Advantages over Fine-tuning

1. **Storage efficiency**: Serving $N$ tasks requires only $N$ trigger sequences (a few tokens each), not $N$ full model copies.
2. **Low-data regime**: In experiments with fewer than ~100 labeled examples, AutoPrompt matches or exceeds fine-tuned models.
3. **Modular probing**: Trigger tokens can be analyzed (e.g., via nearest-neighbor lookup) to surface what semantic content drives predictions.

## Limitations

- **Requires labeled data**: Unlike zero-shot manual prompting, AutoPrompt needs a training set to run gradient search.
- **Brittle triggers**: Learned trigger tokens are often non-interpretable strings; small changes can degrade performance.
- **Imbalanced data**: The method struggles when class frequencies are highly skewed.
- **Restricted to MLMs**: Gradient search over discrete tokens is straightforward for masked LMs; extension to autoregressive models (e.g., GPT) requires adaptation (e.g., using the log-likelihood of the completion).

> [!CAUTION]
> The generated trigger tokens (e.g., "eses Inst") are semantically opaque. While the paper frames this as a probing tool, one should not interpret trigger token semantics at face value without further analysis.

# Conclusion

AutoPrompt demonstrates that pretrained MLMs encode substantial factual, commonsense, and task-specific knowledge that can be surfaced by automatically learned prompts rather than by fine-tuning or human engineering. The key technical contributions are: (1) a gradient-based trigger-search algorithm requiring no weight updates; (2) an automated label-token selection procedure using contextualized embeddings. Across sentiment analysis, NLI, fact retrieval, and relation extraction, AutoPrompt consistently improves over hand-crafted baselines and reveals that RoBERTa contains more accessible factual knowledge than BERT.

> [!TIP]
> Code and datasets are released by the authors at [github.com/ucinlp/autoprompt](https://github.com/ucinlp/autoprompt).
