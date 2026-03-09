# Meta Information

- URL: [BigBird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., & Ahmed, A. (2020). Big Bird: Transformers for Longer Sequences. NeurIPS 2020.

```bibtex
@article{zaheer2020bigbird,
  author    = {Manzil Zaheer and Guru Guruganesh and Avinava Dubey and Joshua Ainslie and Chris Alberti and Santiago Ontanon and Philip Pham and Anirudh Ravula and Qifan Wang and Li Yang and Amr Ahmed},
  title     = {Big Bird: Transformers for Longer Sequences},
  journal   = {CoRR},
  volume    = {abs/2007.14062},
  year      = {2020},
  url       = {https://arxiv.org/abs/2007.14062},
  eprinttype = {arXiv},
  eprint    = {2007.14062}
}
```

# Introduction

Transformer-based models like BERT achieve strong NLP performance but suffer from quadratic memory complexity $O(n^2)$ with respect to sequence length $n$, due to full self-attention computing pairwise interactions between all tokens. This limits practical sequence lengths to around 512 tokens for BERT and 1024 tokens for GPT-2, preventing use on long documents such as legal contracts, scientific papers, and genomic sequences.

BigBird introduces a sparse attention mechanism that reduces the quadratic dependency to linear $O(n)$, enabling sequences up to 8× longer than prior methods while maintaining the theoretical expressivity of full attention (universal approximation and Turing completeness).

## Comparison with Related Work

| Method | Complexity | Mechanism |
|---|---|---|
| Full Transformer (BERT) | $O(n^2)$ | All-pairs attention |
| Sparse Transformer | $O(n\sqrt{n})$ | Fixed strided + local patterns |
| Reformer | $O(n \log n)$ | LSH-based approximate attention |
| Longformer | $O(n)$ | Local windows + task-specific globals |
| BigBird | $O(n)$ | Random + local + global tokens (theoretically grounded) |

> [!TIP]
> Longformer (Beltagy et al., 2020) is the most direct predecessor; BigBird generalizes it by adding random attention and providing rigorous theoretical guarantees for universal approximation and Turing completeness.

# Model Architecture

## Generalized Attention

BigBird models attention via a directed graph $D$ on $n$ nodes (tokens), where edges define which key-value pairs each query attends to. The generalized attention output for token $i$ is:

$$\text{Attn}_D(X)_i = x_i + \sum_{h=1}^{H} \text{softmax}\!\left(Q_h(x_i)\, K_h(X_{N(i)})^\top\right) \cdot V_h(X_{N(i)})$$

where:
- $X \in \mathbb{R}^{n \times d}$ is the input sequence ($n$ tokens, $d$ model dimension)
- $Q_h, K_h, V_h : \mathbb{R}^d \to \mathbb{R}^{d_h}$ are query, key, value projections for head $h$
- $N(i)$ is the out-neighbor set of node $i$ in graph $D$
- $H$ is the number of attention heads

Full attention corresponds to the complete graph $D = K_n$, with $|N(i)| = n$ for all $i$.

## Three Attention Components

BigBird combines three complementary attention patterns to form the sparse graph $D$:

### 1. Global Tokens ($g$ tokens)

$g$ special tokens (either existing tokens designated as global, or new learnable CLS-style tokens) that attend to and are attended by **all** other tokens in the sequence. Global tokens ensure that information can flow across the entire sequence even with a sparse local graph.

- Each global token $g_j$ receives attention from all $n$ positions: $N(g_j) = \{1, \ldots, n\}$
- All tokens attend to each global token: $g_j \in N(i)$ for all $i$

### 2. Local Window Attention ($w$ window size)

Each token $i$ attends to its $w/2$ neighbors on each side, forming a sliding window of size $w$:

$$N_{\text{local}}(i) = \left\{i - \frac{w}{2}, \ldots, i + \frac{w}{2}\right\} \cap \{1, \ldots, n\}$$

This captures local syntactic and semantic dependencies efficiently.

### 3. Random Attention ($r$ random keys)

Each token $i$ attends to $r$ randomly selected keys from the full sequence:

$$N_{\text{random}}(i) = \text{Uniform-Sample}(\{1, \ldots, n\}, r)$$

Random attention enables long-range dependencies without the full $O(n^2)$ cost. The theoretical justification draws an analogy to random graph connectivity: sparse random graphs are well-connected with high probability (Erdős–Rényi theory), so random attention ensures that information can propagate between distant positions.

## Combined Complexity

The total number of attended positions per token is $g + w + r$, which is a constant independent of $n$. The total attention complexity across all tokens is thus $O(n \cdot (g + w + r)) = O(n)$.

> [!IMPORTANT]
> BigBird requires $g + w + r \geq 1$ for connectivity. In practice, the paper uses $g = 2$ global CLS tokens, $w = 3$ (window of 3 neighbors), and $r = 3$ random tokens as the minimum configuration with theoretical guarantees.

## Two Variants

BigBird is implemented in two flavors:

1. **BigBird-ITC** (Internal Transformer Construction): Uses existing tokens as global tokens (e.g., the first $g$ tokens). No new parameters added.
2. **BigBird-ETC** (Extended Transformer Construction): Adds $g$ new learnable global tokens prepended to the sequence, similar to extra CLS tokens. Slightly more expressive but requires additional parameters.

# Theoretical Properties

## Theorem 1: Universal Approximation

For any function $f$ in the class of continuous sequence-to-sequence functions $\mathcal{F}_{CD}$ (preserving sequence structure under permutations), and for any $\varepsilon > 0$, there exists a BigBird transformer $g \in \mathcal{T}_D^{(H,m,q)}$ such that:

$$d_p(f, g) \leq \varepsilon$$

where $D$ is any graph containing a **star graph** $S_n$ (one node connected to all others — this corresponds to the global token structure). This result holds for $1 < p < \infty$.

The proof proceeds in three steps:
1. Approximate $f$ by a piecewise-constant function using a contextual mapping (a function where tokens with the same local context map to the same output).
2. Construct contextual mappings using **selective shift operators** (Lemma 2): sparse attention layers that propagate positional information along graph edges to uniquely identify each token's position.
3. Approximate the resulting modified transformer by a standard BigBird transformer using ReLU/softmax activations.

> [!NOTE]
> The star graph $S_n$ (global tokens) is the key structural requirement. Pure local-window attention without any global tokens does NOT satisfy universal approximation.

## Theorem 2: Turing Completeness

A BigBird encoder-decoder with sparse attention (containing the star graph) can simulate any Turing machine, meaning it can compute any computable function given sufficient depth. This extends the full-attention Turing completeness result of Pérez et al. (2019) to sparse attention.

## Proposition 1: Limitations of Sparse Attention

For certain tasks (specifically the "furthest vector" problem — finding the key most similar to a given query), full attention solves the problem in a single layer, but any sparse attention mechanism requires $\Omega(n^{1-o(1)})$ layers under the Strong Exponential Time Hypothesis (SETH). This demonstrates that sparsity has real computational costs for specific retrieval-like tasks.

# Pre-training

BigBird is pre-trained with the same **Masked Language Modeling (MLM)** objective as BERT: randomly mask 15% of tokens and predict them from context.

- **Architecture base**: Transformer encoder with BigBird sparse attention
- **Sequence length**: Up to 4096 tokens (vs. 512 for BERT)
- **Training corpus**: Four standard text datasets (Books, CC-News, OpenWebText, Stories)
- **Evaluation metric**: Bits-per-character (BPC) on held-out data

| Model | Sequence Length | BPC |
|---|---|---|
| RoBERTa | 512 | 1.23 |
| Longformer | 4096 | 1.18 |
| BigBird | 4096 | **1.12** |

> [!NOTE]
> Lower BPC indicates better language modeling. BigBird achieves the best BPC by exploiting longer context during pretraining.

# Experiments

## Dataset Overview

| Task | Dataset | Metric |
|---|---|---|
| Pretraining (NLP) | Books + CC-News + OpenWebText + Stories | BPC |
| Question Answering | Natural Questions | Short Ans. F1, Long Ans. Accuracy |
| Question Answering | HotpotQA | Ans. F1, Support F1, Joint F1 |
| Question Answering | TriviaQA-wiki | Verified Accuracy |
| Question Answering | WikiHop | Accuracy |
| Summarization | ArXiv, PubMed, BigPatent | ROUGE-1/2/L |
| Genomics Pretraining | Human genome (GRCh37) | BPC |
| Promoter Prediction | Eukaryotic Promoter Database (EPD) | F1 |
| Chromatin Profile | ENCODE + Roadmap Epigenomics (919 profiles) | AUC (TF, HM, DHS) |

## Question Answering Results

For Natural Questions (long-document QA requiring full document reading):
- BigBird-ETC achieves state-of-the-art on **Long Answer (LA)** prediction, outperforming prior methods that use retrieval or chunking to handle length.
- BigBird also achieves top results on TriviaQA and WikiHop.

The improvement stems directly from BigBird's ability to encode the full document (up to 4096 tokens) rather than truncating to 512 tokens.

## Summarization

BigBird is combined with the Pegasus summarization model (BigBird-Pegasus). Key results on ArXiv summarization:
- ROUGE-1: **46.63** (vs. ~40.9 for prior best with truncation)
- Improvement of approximately 5 ROUGE-1 points, attributed to BigBird reading full papers rather than truncated abstracts/introductions.

## Genomics Applications

BigBird is pretrained on the human reference genome (GRCh37) treating nucleotide sequences (A, T, C, G) as tokens, with sequences up to 4096 base pairs.

**Promoter Region Prediction:**
- Task: Binary classification — does a 300bp DNA window contain a promoter?
- BigBird F1: **99.9** vs. prior CNN-based best of **95.6**

**Chromatin-Profile Prediction:**
- Task: Predict 919 binary epigenetic marks (transcription factor binding, histone modifications, DNase hypersensitivity) from 1000bp DNA sequences
- BigBird achieves improvements on histone marks (HM), where long-range context is most important
- AUC gains validate that 4096-length sequences capture regulatory element interactions that 512-length models miss

## Hardware and Implementation

- Training hardware: TPUs with 16GB memory per chip
- Batch sizes: 32–64 per device
- The sparse attention pattern is implemented using a custom blocked sparse matrix multiplication kernel for efficiency

# Differences from Similar Methods

| Aspect | Longformer | BigBird |
|---|---|---|
| Local window | Yes (sliding window) | Yes ($w$ neighbors) |
| Global tokens | Task-specific manual designation | Systematic ($g$ tokens with theory) |
| Random attention | No | Yes ($r$ random keys) |
| Theoretical guarantee | None stated | Universal approx. + Turing completeness |
| Complexity | $O(n)$ | $O(n)$ |
| Max sequence length | 4096 | 4096 |
| Genomics applications | No | Yes |

> [!CAUTION]
> The random attention component is drawn uniformly per token during each forward pass. In practice, this is typically implemented with a fixed random seed at initialization (rather than resampling each forward pass) for efficiency. The distinction matters for reproducibility.

> [!TIP]
> See the Longformer paper (arXiv:2004.05150) for the predecessor approach. BigBird's ETC variant was inspired by Google's Extended Transformer Construction for structured inputs.
