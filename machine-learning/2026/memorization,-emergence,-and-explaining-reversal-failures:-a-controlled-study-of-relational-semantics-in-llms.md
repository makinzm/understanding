# Meta Information

- URL: [Memorization, Emergence, and Explaining Reversal Failures: A Controlled Study of Relational Semantics in LLMs](https://arxiv.org/abs/2601.02931)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhu, Y., Liu, Q., Wang, J., Cheng, F., Liu, C., Aizawa, A., Kurohashi, S., & Shimodaira, H. (2026). Memorization, Emergence, and Explaining Reversal Failures: A Controlled Study of Relational Semantics in LLMs. arXiv:2601.02931.

# Memorization, Emergence, and Explaining Reversal Failures

## Overview

This paper investigates two core questions about autoregressive language models and relational semantics:

1. **RQ1 (Memorization & Emergence)**: Can autoregressive models memorize relational facts and internalize their logical properties (symmetry, inversion)? Under what training conditions does this emerge?
2. **RQ2 (Reversal Curse)**: When models do understand relational semantics, does the reversal failure (worse accuracy on reverse queries) stem from a missing inversion representation, or from the left-to-right decoding bias inherent in autoregressive generation?

The approach is controlled: GPT-2-style models are trained from scratch on synthetic knowledge-graph (KG) corpora where the ground truth relational structure is fully known. This avoids confounders from naturalistic text and allows precise manipulation of training variables.

## 2. Problem Setting and Formal Definitions

### Knowledge Graph Structure

A knowledge graph is defined as $\mathcal{G} = (\mathcal{E}, \mathcal{R}, \mathcal{T})$ where $\mathcal{E}$ is the entity set, $\mathcal{R}$ is the relation type set, and $\mathcal{T} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$ is the triple set.

Two logical properties of binary relations are studied:

- **Symmetry**: $r_{\text{sym}} \in \mathcal{R}$ satisfies $(A, r_{\text{sym}}, B) \in \mathcal{T} \Rightarrow (B, r_{\text{sym}}, A) \in \mathcal{T}$ (e.g., "friend of")
- **Inversion**: A pair $(r_1, r_2)$ satisfies inversion if $(A, r_1, B) \in \mathcal{T} \Rightarrow (B, r_2, A) \in \mathcal{T}$ (e.g., "father/son", "parent/child")

Non-logic attributes (person–job pairs) serve as a control without relational structure.

### Reversal Curse

The reversal curse refers to the empirically observed asymmetry where autoregressive models answer forward queries (given subject, predict object) more accurately than reverse queries (given object, predict subject), even when the inverse fact is semantically equivalent.

## 3. Synthetic Corpus Construction

### Entity and Relation Setup

- **Entities**: Up to $10^6$ unique full names, constructed from three token pools of 100 tokens each. This prevents entity memorization by name lookup alone.
- **Occupations**: 300 real-world job titles as non-relational attributes.
- **Relations**: Symmetric (friend), inversion pairs (parent/child, father/son, etc.)

### Corpus Generation

Each training graph $T^{(n)}$ contains 7 triples: person–person relation triples (with logical structure) plus person–job triples (without). Each triple is verbalized using one of four surface templates, selected uniformly at random. The templates are shuffled within each graph paragraph.

```math
\begin{align}
  D_\alpha = \mathcal{P}_\alpha(N, K; \mathcal{F}) = \left\{ \left\{ \text{Shuffle}\left( \{ f^{(\tau_i^{(k)})}(T_i^{(n)}) \}_{i \in \mathcal{I}_\alpha} \right) \right\}_{k=1}^{K} \right\}_{n=1}^{N}
\end{align}
```

where $\tau_i^{(k)} \sim \text{Unif}(\{1,2,3,4\})$ selects a surface template, $N$ is the number of training graphs, and $K$ is the number of template paragraphs per graph.

**Controlled variables**:
- $N \in \{2{,}000, \; 20{,}000, \; 200{,}000\}$ (training graph count)
- $K \in \{10, \; 100\}$ (templates per graph)
- $L \in \{1, 2, 3, 5, 8, 12, 16, 20, 30\}$ (model depth)
- Training epochs $E$

**Evaluation graphs**: Held-out 500 samples with reverse facts withheld to prevent data leakage.

## 4. Evaluation Tasks

Four task types probe different aspects of relational understanding:

| Task | Abbreviation | What it Measures |
|---|---|---|
| Memorization QA (job) | QMemJob | Recall of explicitly seen person–job facts |
| Memorization QA (person) | QMemPeople | Recall of explicitly seen person–person facts |
| Logic QA (forward) | QLogic-F | Inference of logically implied facts in forward direction |
| Logic QA (reverse) | QLogic-R | Inference of logically implied facts in reverse direction |
| In-context completion | QICL-C | Completion of a sentence with an unseen entity via in-context examples |
| In-context QA | QICL-Q | Answer a QA-formatted query about an unseen entity in-context |

> [!NOTE]
> The Logic QA tasks (QLogic) require the model to answer queries about facts that can be derived via symmetry or inversion from seen facts, but were never explicitly stated in training. This distinguishes genuine logical inference from surface memorization.

## 5. Model Architecture and Training

### Architecture

GPT-2-style decoder-only transformer with the following standard configuration:

| Hyperparameter | Value |
|---|---|
| Layers (L) | 12 (ablated from 1–30) |
| Attention heads | 12 |
| Hidden dimension | 768 |
| Tokenizer | GPT-2 tokenizer |

### Pretraining Hyperparameters

- Batch size: 491,520 tokens/iteration
- Learning rate: $6 \times 10^{-4}$ with cosine decay to $6 \times 10^{-5}$
- Weight decay: 0.1
- Warmup: 500 iterations
- Precision: bf16

### Supervised Fine-Tuning (SFT)

- Learning rate: $3 \times 10^{-5}$
- Batch size: 32,768 tokens/iteration
- QA pairs generated only from training-set content (evaluation entities excluded)

### Inference

- Temperature: 0.8
- Top-k: 100

## 6. Findings

### Finding 1: Sharp Phase Transition (Emergence)

Memorization and logical inference emerge abruptly as the number of training graphs $N$ increases:

- QICL-C accuracy shows a sharp phase transition above approximately $N > 2{,}000$
- QICL-Q accuracy transitions at approximately $N > 20{,}000$
- Final logic QA accuracy reaches ~85% under optimal conditions (large $N$, large $K$)

This pattern is consistent with the **grokking** phenomenon: models first overfit to memorization of individual facts, then abruptly generalize to logical rules. The transition from near-zero to near-perfect accuracy occurs within a narrow range of training samples or epochs, and is not a gradual improvement.

```math
\begin{align}
  \text{Accuracy}(N) \approx \begin{cases} \approx 0 & N < N^* \\ \approx 1 & N \geq N^* \end{cases}
\end{align}
```

for some threshold $N^*$ that depends on the task and model depth.

> [!NOTE]
> "Clear grokking transition: QICL-C accuracies abruptly jump to near-perfect around epoch $E \approx 20$." (paper)

### Finding 2: Shallow Models Suffice for Emergence

Even a 3-layer GPT-2 model achieves ~85% on QICL-C with sufficient training data. A 1-layer model reaches only ~10%, suggesting a minimal architectural requirement of at least 2–3 layers for relational inference.

Models that successfully generalize develop **stable, logic-relevant representations in intermediate layers (5–8)**. Failed models show unstable activations in late layers. This supports the hypothesis that relational rules are encoded as distributional patterns in intermediate representations, not in the final prediction head.

### Finding 3: Reversal Failure = Decoding Bias, Not Semantic Deficiency

The paper provides three lines of evidence that the reversal curse is architectural, not semantic:

1. **Symmetry vs. inversion**: Symmetric relations (which are self-inverse) perform similarly in forward and reverse directions, while inversion pairs show a gap only when reverse is queried autoregressively.

2. **Bidirectional training**: When models are trained bidirectionally (both forward and reverse sentences are included), the accuracy gap between forward and reverse queries disappears entirely.

3. **Diffusion models**: Models trained with masked/diffusion objectives (no left-to-right constraint) show approximately equal forward and reverse accuracy, with a slight (~4%) reverse advantage.

> [!IMPORTANT]
> The reversal curse is not evidence that models fail to understand inversion — it is evidence that autoregressive decoding creates a positional asymmetry. A model that has internalized the inverse relation cannot express it when the decoding direction opposes the surface order of the training sentence.

### Finding 4: Memorization Asymmetry Between Relation Types

- **QMemJob** (person–job, non-relational): >95% accuracy with sufficient training
- **QMemPeople** (person–person, relational): peaks at ~50% under optimal conditions

The lower ceiling on person–person memorization reflects the higher cardinality of person–person entity pairs compared to person–job pairs. Person–job facts follow a many-to-one structure (many people share one job), while person–person facts form a more complex graph, making memorization harder.

## 7. Comparison with Related Work

| Aspect | This Work | Prior Work |
|---|---|---|
| Training corpus | Fully controlled synthetic KG | Naturalistic text (Wikipedia, books) |
| Relational structure | Ground-truth known | Unknown / approximate |
| Reversal cause | Decoding bias (shown empirically) | Hypothesized as missing reverse training signal |
| Emergence | Controlled phase transition study | Observed post-hoc in pretrained models |
| Grokking | Observed at the relational inference level | Studied in arithmetic / modular addition tasks |

> [!TIP]
> The "reversal curse" was named and empirically studied by Berglund et al. (2023), who showed GPT-4-class models fail on reverse queries. This paper provides a mechanistic explanation using a controlled setup.

### Limitations

- Only symmetric and inverse kinship/friendship relations are studied; transitivity and multi-hop reasoning are out of scope.
- All corpora are template-generated; findings may not directly transfer to naturalistic, noisy text distributions.
- Entity name pools are finite (up to $10^6$); real-world pretraining corpora have effectively unbounded entity diversity.

# Experiments

- **Dataset**: Synthetic KG-based corpora, generated by the authors. No publicly available benchmark is used. Training corpus size $N \in \{2{,}000, 20{,}000, 200{,}000\}$; templates $K \in \{10, 100\}$; fixed evaluation set of 500 graphs.
- **Hardware**: Not explicitly stated (bf16 training implies GPU).
- **Optimizer**: AdamW (inferred from GPT-2 pretraining conventions); learning rate $6 \times 10^{-4}$ with cosine decay.
- **Results**:
  - QMemJob: >95% accuracy with $N = 200{,}000$, $K = 100$
  - QMemPeople: ~50% peak accuracy
  - QICL-C phase transition above $N \approx 2{,}000$; ~85% final accuracy
  - QICL-Q phase transition above $N \approx 20{,}000$
  - Bidirectional training reduces forward–reverse accuracy gap to near zero
  - Diffusion model reverse accuracy ≈ forward accuracy (≈+4% reverse advantage)
