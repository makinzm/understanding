# Meta Information

- URL: [LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking](https://arxiv.org/abs/2311.02089)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yue, Z., Rabhi, S., Moreira, G. de S. P., Wang, D., & Oldridge, E. (2023). LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking. *PGAI@CIKM 2023*.

# LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking

## Overview

LlamaRec is a two-stage recommendation framework that combines a small, efficient sequential retriever with an LLM-based ranker. The key insight is that instead of using LLMs to *generate* ranked lists of items (slow, autoregressive), LlamaRec uses a **verbalizer** to map output logits directly to probability distributions over candidate items, enabling ranking in a single forward pass.

**Who benefits from this:** Practitioners building production-scale recommender systems where LLM inference latency matters. The framework is applicable to any domain with item sequences (e.g., movies, products, video games).

**Inputs and Outputs:**

| Component | Input | Output |
|---|---|---|
| Retriever (LRURec) | User interaction sequence $\mathbf{h} = [i_1, i_2, \ldots, i_{t-1}]$ | Top-$K$ candidate items $\mathcal{C} = \{c_1, \ldots, c_K\}$ |
| LLM Ranker (Llama 2) | Prompt with user history + candidates | Probability distribution over candidates |
| Full Framework | User history sequence | Ranked list of $K$ items |

## Problem Setup

Given a user interaction history $\mathbf{h} = [i_1, i_2, \ldots, i_{t-1}]$ sorted chronologically, the goal is to recommend the next item $i_t$. The training objective minimizes the negative log-likelihood over dataset $\mathcal{X}$:

$$\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{X}} \left[ \mathcal{L}(f(\theta; x), y) \right]$$

where $f = f_\text{ranker} \circ f_\text{retriever}$ composes the retrieval and ranking stages.

## Stage 1: Retrieval with LRURec

The retrieval stage uses LRURec (Linear Recurrent Unit-based Recommender), a sequential model that processes user interaction sequences using linear recurrence. LRURec computes a representation of the user state regardless of sequence length, and retrieves the top-$K = 20$ candidate items. This stage is not trained end-to-end with the ranker; it runs independently and its top-20 outputs form the candidate pool.

> [!TIP]
> LRURec is based on linear recurrent units (LRU), which are more efficient than self-attention for long sequences, enabling retrieval from large item catalogues in sub-linear time.

## Stage 2: LLM Ranking via Verbalizer

### Prompt Construction

The prompt given to Llama 2 (7B) follows a fixed template:

```
[Instruction]
Here is the purchase history of a user: <item_1>, <item_2>, ..., <item_n>
Please rank the following candidate items based on the user's preferences:
A. <candidate_title_1>
B. <candidate_title_2>
...
T. <candidate_title_20>
The most suitable item to recommend next is:
```

- History items are listed in chronological order, capped at 20 most recent items
- Each item's title is truncated to 32 tokens maximum
- Candidates $c_1, \ldots, c_{20}$ are assigned alphabetical labels A–T

### Verbalizer: Logit-to-Probability Mapping

Instead of generating a ranked sequence autoregressively, LlamaRec extracts the logit for each index token (A, B, C, …) from the LLM's output at the label position. These logits are converted to a probability distribution via softmax:

$$P(c_k \mid \mathbf{h}) = \frac{\exp(z_{l_k})}{\sum_{j=1}^{K} \exp(z_{l_j})}$$

where $z_{l_k}$ is the logit for the token corresponding to label letter $l_k$ (e.g., "A", "B", …) for candidate $c_k$, and $K = 20$.

> [!IMPORTANT]
> This verbalizer approach requires only a **single forward pass** through the LLM, eliminating autoregressive decoding, beam search, and output parsing. This reduces inference time from ~56 seconds (generation-based) to under 1 second for a batch of 20 candidates.

### Training with QLoRA

The ranker is fine-tuned using instruction tuning. Loss is computed only on the label token position (not on the instruction or history tokens):

$$\mathcal{L}_\text{ranker} = -\log P(l_{y} \mid \text{prompt})$$

where $l_y$ is the correct label letter for the ground-truth item.

**Parameter-efficient fine-tuning settings:**

| Hyperparameter | Value |
|---|---|
| Method | QLoRA (4-bit quantization + LoRA) |
| Trainable parameters | < 1% of 7B parameters |
| LoRA rank $r$ | 8 |
| LoRA $\alpha$ | 32 |
| LoRA dropout | 0.05 |
| Target modules | Query ($W_Q$) and Value ($W_V$) projection matrices |
| LoRA learning rate | $1 \times 10^{-4}$ |
| Ranker training epochs | 1 |

> [!NOTE]
> QLoRA combines 4-bit NormalFloat quantization of the frozen base model weights with low-rank adapters (LoRA) applied to the attention projection matrices. This allows fine-tuning a 7B model on a single GPU.

## Key Algorithmic Differences from Prior Work

| Method | Generation Strategy | Efficiency | Two-Stage |
|---|---|---|---|
| P5 | Autoregressive text generation | Slow (seq-len dependent) | No |
| GPT4Rec | Beam search over titles | Slow | No |
| PALR | LLM re-ranks retrieved candidates | Single forward pass | Yes |
| TALLRec | Binary yes/no classification per item | $O(K)$ forward passes | No |
| **LlamaRec** | Verbalizer over index tokens | **Single forward pass** | **Yes** |

> [!IMPORTANT]
> The key distinction from PALR (the closest prior work) is that LlamaRec uses a verbalizer over index tokens rather than generating text. PALR still relies on generating item indices as text, while LlamaRec reduces ranking to reading a single token's logit value from the vocabulary.

# Experiments

- **Datasets:**
  - ML-100k (MovieLens): 610 users, 3,650 items, 100k interactions (movie domain)
  - Beauty (Amazon Reviews): 22,332 users, 12,086 items, 198k interactions (cosmetics)
  - Games (Amazon Reviews): 15,264 users, 7,676 items, 148k interactions (video games)
  - All datasets use leave-one-out evaluation: last item as test, second-to-last as validation
- **Hardware:** NVIDIA A100 GPU (implied by NVIDIA co-authors and QLoRA setup)
- **Retriever Optimizer:** AdamW, learning rate $1 \times 10^{-3}$, up to 500 epochs
- **Ranker Optimizer:** AdamW, LoRA learning rate $1 \times 10^{-4}$, 1 epoch on ranker
- **Evaluation Metrics:** MRR@5, MRR@10, NDCG@5, NDCG@10, Recall@5, Recall@10
- **Results:**
  - LlamaRec outperforms LRURec (the retriever alone) by +11.99% on ML-100k, +3.99% on Beauty, +11.06% on Games (average over all metrics)
  - Largest single-metric gain: Recall@5 on ML-100k, +20.85% over LRURec
  - Against LLM-based baselines on Beauty: +14.31% average over second-best (PALR), +24.22% on NDCG@10
  - Inference time: < 1 second vs. 56.16 seconds for generation-based approach (title length = 20 tokens)
