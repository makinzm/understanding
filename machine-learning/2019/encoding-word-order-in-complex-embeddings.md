# Meta Information

- URL: [Encoding Word Order in Complex Embeddings](https://arxiv.org/abs/1912.12333)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Wang, B., Zhao, D., Lioma, C., Li, Q., Zhang, P., & Simonsen, J. G. (2020). Encoding Word Order in Complex Embeddings. In *International Conference on Learning Representations (ICLR 2020)*.

# Encoding Word Order in Complex Embeddings

## Overview

This paper proposes a novel word embedding approach that encodes **positional (word order) information through the phase of complex-valued numbers**, and **semantic information through the amplitude**. Rather than treating position as a discrete lookup (as in standard positional encodings), words are represented as **continuous functions** over a position variable. This functional formulation admits a unique closed-form solution under two mathematical constraints, yielding complex exponential embeddings that generalize and subsume the Transformer's sinusoidal positional encoding as a special case.

**Applicability**: Useful for any NLP practitioner who needs position-sensitive word representations in CNN, RNN, or Transformer models — particularly for tasks where word order is crucial, such as machine translation, text classification, and language modeling.

## Problem: Limitations of Existing Positional Encodings

Standard position embeddings assign a fixed vector to each absolute position index. This approach has two shortcomings:

1. **Independence across positions**: Position $i$ and position $i+1$ embeddings are not explicitly related; the model must learn positional adjacency implicitly from data.
2. **Semantic–positional entanglement**: Token (semantic) embeddings and position embeddings are typically summed, which conflates two distinct types of information without a principled separation.

> [!NOTE]
> "Position embeddings capture the position of individual words, but not the ordered relationship (e.g., adjacency or precedence) between individual word positions."

## Proposed Method: Complex-Order Word Embeddings

### Core Idea

Represent each word's embedding as a **complex-valued function** $f: \mathbb{Z}_{\geq 0} \to \mathbb{C}^d$ over the position variable $\text{pos} \in \mathbb{Z}_{\geq 0}$. Each dimension $j \in \{1, \ldots, d\}$ of the embedding is:

$$f_j(\text{pos}) = r_j \cdot e^{i(\omega_j \cdot \text{pos} + \theta_j)}$$

where:
- $r_j \in \mathbb{R}_{>0}$: amplitude, encoding **semantic content** of the word in dimension $j$
- $\omega_j \in \mathbb{R}$: angular frequency, encoding **positional sensitivity** (how much the word's embedding varies with position)
- $\theta_j \in \mathbb{R}$: initial phase offset, encoding a **word-specific starting angle**
- $\text{pos}$: integer word position (0-indexed)

The per-word learnable parameters are thus $(r_j, \omega_j, \theta_j)$ for each dimension $j$.

### Desirable Properties and Their Derivation

The functional form above is derived — not assumed — from two principled constraints:

**Property 1 (Position-Free Offset Transformation)**: There exists a transformation $T$ such that $f(\text{pos} + 1) = T \circ f(\text{pos})$ for all $\text{pos}$, and $T$ does not explicitly depend on $\text{pos}$. This ensures that the relationship between adjacent positions is uniform and position-agnostic.

**Property 2 (Boundedness)**: $\|f(\text{pos})\|$ is bounded for all $\text{pos} \in \mathbb{Z}_{\geq 0}$, preventing embedding norms from exploding as sequence length grows.

> [!IMPORTANT]
> **Claim 1**: Under a linear witnessing assumption, the unique class of functions satisfying both properties is $g(\text{pos}) = z_2 \cdot z_1^{\text{pos}}$ with $|z_1| \leq 1$, $z_1, z_2 \in \mathbb{C}^d$. Setting $z_1 = e^{i\omega}$ (on the unit circle) and $z_2 = r \cdot e^{i\theta}$ recovers $f_j(\text{pos}) = r_j \cdot e^{i(\omega_j \cdot \text{pos} + \theta_j)}$.

### Input / Output

| Stage | Input | Output |
|---|---|---|
| Word lookup | Token index $w \in \{1,\ldots,|V|\}$, position $\text{pos} \in \mathbb{Z}_{\geq 0}$ | Complex vector $\mathbf{f}(w, \text{pos}) \in \mathbb{C}^d$ |
| Magnitude extraction | $\mathbf{f}(w, \text{pos}) \in \mathbb{C}^d$ | $\|\mathbf{f}(w, \text{pos})\| \in \mathbb{R}^d$ (for real-valued downstream layers) |
| Argument extraction | $\mathbf{f}(w, \text{pos}) \in \mathbb{C}^d$ | $\arg(\mathbf{f}(w, \text{pos})) \in \mathbb{R}^d$ (phase angles) |

For integration with real-valued architectures (CNN, RNN, Transformer), the complex vector is converted to a real vector of length $2d$ by concatenating real and imaginary parts, or by using the modulus and argument separately.

### Parameter Count

Each word $w$ has parameters $(r_j, \omega_j, \theta_j)$ for $j = 1, \ldots, d$: three real scalars per dimension. With vocabulary size $|V|$ and embedding dimension $d$:
- **Full parameterization**: $3 \times |V| \times d$ parameters
- **Shared frequencies**: $\omega_j$ can be shared across the vocabulary (reducing to $|V| \times 2d + d$ parameters), as frequency reflects positional sensitivity which may be word-type–independent.

## Pseudocode: Forward Pass for Complex-Order Embedding

```
Input:  token index w, position pos
Params: r[w,j], omega[w,j], theta[w,j]  for j = 1..d

For j = 1 to d:
    phase[j]  = omega[w,j] * pos + theta[w,j]
    embed[j]  = r[w,j] * exp(i * phase[j])
           = r[w,j] * cos(phase[j])  +  i * r[w,j] * sin(phase[j])

Output: embed ∈ C^d   (complex vector of length d)

# Convert to real for downstream layers:
real_embed = [Re(embed[1]), ..., Re(embed[d]),
              Im(embed[1]), ..., Im(embed[d])]  ∈ R^{2d}
```

## Relation to Transformer Positional Encoding (Vaswani et al., 2017)

The Transformer's sinusoidal positional encoding adds a position-dependent offset to a word embedding:

$$\text{PE}(\text{pos}, 2k) = \sin\left(\frac{\text{pos}}{10000^{2k/d}}\right), \quad \text{PE}(\text{pos}, 2k+1) = \cos\left(\frac{\text{pos}}{10000^{2k/d}}\right)$$

This corresponds to a **degenerate special case** of complex-order embeddings where:
- $r_j = 1$ (constant amplitude for all words — no semantic variation in the amplitude)
- $\theta_j = 0$ (no initial phase)
- $\omega_j = 1 / 10000^{2j/d}$ (fixed, shared frequency)

> [!NOTE]
> The Transformer's positional encoding does **not** separate semantic and positional information (it adds position to semantic embedding), whereas complex-order embeddings encode position in the **phase** and semantics in the **amplitude**, achieving a principled decoupling.

## Comparison with Similar Methods

| Method | Position Representation | Semantic–Positional Coupling | Learnable? | Continuous? |
|---|---|---|---|---|
| Learned positional embedding (BERT) | Lookup table $\mathbb{R}^{L \times d}$ | Additive (conflated) | Yes | No |
| Sinusoidal PE (Transformer) | Fixed $\sin/\cos$ | Additive (conflated) | No | Yes (in pos) |
| Relative PE (Shaw et al., 2018) | Pair-wise offsets | Attention-level | Yes | No |
| Complex-order (this paper) | Phase of complex number | **Decoupled** (phase vs. amplitude) | Yes | Yes |

## Experiments

### Text Classification

- **Datasets**: CR (product reviews), MPQA (opinion corpus), SUBJ (subjectivity), MR (movie reviews), SST-1 / SST-2 (Stanford Sentiment Treebank), TREC (question classification)
- **Baselines**: FastText, CNN (Kim 2014), LSTM, Transformer, and their positional variants
- **Metric**: Accuracy
- **Result**: Complex-order embeddings consistently outperformed all baselines across all six datasets with statistical significance ($p < 0.05$), demonstrating that the complex-valued positional encoding provides complementary information over standard semantic embeddings.

### Machine Translation

- **Dataset**: WMT 2016 English→German (approximately 29K sentence pairs)
- **Hardware**: Not specified
- **Optimizer**: Adam (matching Transformer training setup)
- **Metric**: BLEU score
- **Result**: Transformer + Complex-order achieved **35.8 BLEU**, a gain of **+1.3 BLEU** over vanilla Transformer (34.5 BLEU), showing the benefit of richer positional representations in sequence-to-sequence tasks.

### Language Modeling

- **Dataset**: Text8 (100 million character Wikipedia corpus)
- **Metric**: Bits Per Character (BPC)
- **Result**: Transformer-XL + Complex-order achieved **1.26 BPC**, outperforming the baseline Transformer-XL at **1.29 BPC**.

## Ablation Study

Experiments on the Transformer architecture revealed:

- Removing the initial phase $\theta_j$ degraded accuracy by approximately 0.028 points — phases carry word-specific positional identity.
- Applying frequency sharing (one $\omega_j$ per dimension across all words) reduced performance by 0.008–0.019 points but provided significant parameter savings.
- The amplitude $r_j$ as the primary semantic carrier was confirmed: removing its variability (fixing $r_j = 1$) degrades results toward the Transformer baseline.

## Qualitative Analysis: Learned Frequencies

In text classification experiments, words with **high positional sensitivity** (large $\|\omega_j\|$) tended to be **sentiment-bearing words** (e.g., "excellent", "terrible"). This indicates that the model learns a meaningful association: semantically important words exhibit greater positional variability, suggesting they carry different meaning in different syntactic positions.

## Limitations

- The derivation assumes a **linear witnessing** relationship between adjacent-position embeddings, which may not capture long-range non-linear positional dependencies.
- Evaluation is on relatively small WMT 2016 (29K pairs); the approach has not been validated on large-scale benchmarks (e.g., WMT 2014 with 4.5M pairs).
- Requires storing 3 real parameters per embedding dimension per word (versus 1 for standard embeddings), tripling the vocabulary embedding memory unless sharing is applied.
