# Meta Information

- URL: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.

---

# Scaling Laws for Neural Language Models

## Overview

This paper establishes empirical power-law relationships between language model cross-entropy loss $L$ (in nats/token) and three factors: non-embedding parameter count $N$, training dataset size $D$ (tokens), and compute budget $C$ (FLOPs). These relationships hold across more than seven orders of magnitude and are largely independent of architectural details such as depth vs. width. The key practical implication is that, given a fixed compute budget, it is always more efficient to train a significantly larger model on fewer tokens than convergence would require, rather than training a smaller model to full convergence.

## 1. Background and Methods

### 1.1 Model Architecture

All experiments use decoder-only Transformer models (same family as GPT-2). The non-embedding parameter count is:

```math
\begin{align}
  N \approx 12 \cdot n_\text{layer} \cdot d_\text{model}^2
\end{align}
```

where $n_\text{layer}$ is the number of layers and $d_\text{model}$ is the residual stream dimension. The standard architecture uses $d_\text{attn} = d_\text{ff}/4 = d_\text{model}$.

Embedding parameters ($n_\text{vocab} \times d_\text{model}$ for the token embedding and $n_\text{ctx} \times d_\text{model}$ for the positional embedding) are excluded from $N$ because they do not follow the same scaling laws as the non-embedding parameters.

The forward pass compute per token is:

```math
\begin{align}
  C_\text{forward} \approx 2N + 2 \cdot n_\text{layer} \cdot n_\text{ctx} \cdot d_\text{model}
\end{align}
```

For context lengths $n_\text{ctx} \leq d_\text{model}$, the second term is negligible, giving $C_\text{forward} \approx 2N$ FLOPs per token. Accounting for forward and backward passes, total training compute over $D$ tokens is:

```math
\begin{align}
  C \approx 6NBS
\end{align}
```

where $B$ is the batch size (tokens) and $S$ is the number of gradient update steps, so $D = B \cdot S$.

> [!NOTE]
> 1 PF-day = $10^{15}$ FLOPs/second × 86,400 seconds = $8.64 \times 10^{19}$ total FLOPs.

**Detailed per-operation breakdown** for a single Transformer layer with $d_\text{ff} = 4 d_\text{model}$ and $d_\text{attn} = d_\text{model}$:

| Operation | Parameters | FLOPs per Token |
|-----------|-----------|-----------------|
| Attention QKV projection | $n_\text{layer} \cdot 3 d_\text{model}^2$ | $2 \cdot n_\text{layer} \cdot 3 d_\text{model}^2$ |
| Attention mask (softmax) | — | $2 \cdot n_\text{layer} \cdot n_\text{ctx} \cdot d_\text{model}$ |
| Attention output projection | $n_\text{layer} \cdot d_\text{model}^2$ | $2 \cdot n_\text{layer} \cdot d_\text{model}^2$ |
| Feed-forward (2 layers) | $n_\text{layer} \cdot 8 d_\text{model}^2$ | $2 \cdot n_\text{layer} \cdot 8 d_\text{model}^2$ |
| De-embed | — | $2 \cdot d_\text{model} \cdot n_\text{vocab}$ |
| **Total (non-embed)** | $N \approx 12 n_\text{layer} d_\text{model}^2$ | $C_\text{forward} \approx 2N$ |

### 1.2 Dataset: WebText2

The primary training dataset is WebText2, an extended version of OpenAI's WebText corpus (Reddit outbound links filtered by quality, scraped through 2017–2018).

| Property | Value |
|----------|-------|
| Documents | 20.3 million |
| Raw text size | 96 GB |
| Token count (BPE) | $2.29 \times 10^{10}$ |
| Test set tokens | $6.6 \times 10^{8}$ |
| Vocabulary size | 50,257 (byte-pair encoding) |
| Context length | 1,024 tokens |

Transfer generalization was evaluated on Books Corpus, Common Crawl, English Wikipedia, and Public Internet Books.

### 1.3 Training Procedure

- **Optimizer**: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$); Adafactor for models above $\sim 1$ billion parameters
- **Learning rate**: linear warmup over 3,000 steps, then cosine decay to zero
- **Default run**: $2.5 \times 10^5$ steps, batch size 512 sequences × 1,024 tokens ($\approx 5.24 \times 10^5$ tokens/step)
- **Regularization**: 10% dropout
- **Model sizes explored**: $N \in [768,\; 1.5 \times 10^9]$ non-embedding parameters

---

## 2. Empirical Power Laws

### 2.1 Single-Variable Scaling Laws

When performance is not bottlenecked by the other two factors, loss follows an independent power law for each variable. All three laws are of the form $L = (X_c / X)^{\alpha}$:

**Loss vs. non-embedding parameters** (infinite data, trained to convergence):

```math
\begin{align}
  L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076,\quad N_c \approx 8.8 \times 10^{13}
\end{align}
```

**Loss vs. dataset size** (optimal model, trained to convergence on $D$ tokens):

```math
\begin{align}
  L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095,\quad D_c \approx 5.4 \times 10^{13} \text{ tokens}
\end{align}
```

**Loss vs. compute** (naive: fixed model trained with all of $C$):

```math
\begin{align}
  L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.057,\quad C_c \approx 1.6 \times 10^7 \text{ PF-days}
\end{align}
```

**Loss vs. optimally allocated compute** (best model size for each $C_\text{min}$):

```math
\begin{align}
  L(C_\text{min}) = \left(\frac{C_c^\text{min}}{C_\text{min}}\right)^{\alpha_{C_\text{min}}}, \quad \alpha_{C_\text{min}} \approx 0.050,\quad C_c^\text{min} \approx 3.1 \times 10^8 \text{ PF-days}
\end{align}
```

> [!IMPORTANT]
> $C_\text{min}$ is the minimum compute budget needed to reach a given loss, achieved by using the optimal model size. It is smaller than actual compute $C$ whenever the batch size $B$ exceeds the critical batch size $B_\text{crit}$. The optimal-compute exponent ($\alpha_{C_\text{min}} \approx 0.050$) is smaller than the naive-compute exponent ($\alpha_C \approx 0.057$), reflecting the efficiency gained by proper allocation.

### 2.2 Architecture Independence

When total non-embedding parameters $N$ are held fixed, varying the aspect ratio (depth vs. width) by 40× produces less than 3% variation in loss. This means $\alpha_N \approx 0.076$ holds across architectures; practitioners can choose depth/width based on hardware efficiency without significant loss penalty.

**Comparison with LSTMs**: At equivalent parameter counts, Transformers outperform LSTMs for tokens that appear later in the context window (positions $> 100$), where long-range dependencies matter. LSTMs match Transformer performance for the earliest context positions.

---

## 3. Joint Dependence on Model Size and Data

### 3.1 Combined L(N, D) Equation

When both model size $N$ and dataset size $D$ are finite, the joint loss is:

```math
\begin{align}
  L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}
\end{align}
```

Fitted parameters: $\alpha_N = 0.076$, $\alpha_D = 0.103$, $N_c = 6.4 \times 10^{13}$, $D_c = 1.8 \times 10^{13}$.

This form recovers $L(N)$ as $D \to \infty$ and $L(D)$ as $N \to \infty$.

### 3.2 Overfitting and Minimum Data Requirements

The relative overfitting penalty is:

```math
\begin{align}
  \delta L(N, D) \equiv \frac{L(N, D)}{L(N, \infty)} - 1
\end{align}
```

To avoid significant overfitting, the dataset must satisfy:

```math
\begin{align}
  D \gtrsim (5 \times 10^3) \cdot N^{0.74} \text{ tokens}
\end{align}
```

This sub-linear relationship ($N^{0.74}$ rather than $N^1$) means each 8× increase in model size requires only approximately 5× more data to avoid overfitting penalties.

---

## 4. Scaling with Training Steps and Batch Size

### 4.1 Critical Batch Size

There exists a critical batch size $B_\text{crit}$ that balances time-efficiency (fewer serial steps) and compute-efficiency (fewer total FLOPs). It scales as a power law in the current loss $L$:

```math
\begin{align}
  B_\text{crit}(L) = \frac{B^*}{L^{1/\alpha_B}}, \quad B^* \approx 2 \times 10^8 \text{ tokens},\quad \alpha_B \approx 0.21
\end{align}
```

The trade-off between training steps $S$ and total examples $E = B \cdot S$ satisfies:

```math
\begin{align}
  \left(\frac{S}{S_\text{min}} - 1\right)\left(\frac{E}{E_\text{min}} - 1\right) = 1
\end{align}
```

where $S_\text{min}$ and $E_\text{min}$ are the minima achieved at $B \ll B_\text{crit}$ and $B \gg B_\text{crit}$ respectively.

### 4.2 Learning Curve Law: L(N, S)

The loss for a model of size $N$ trained for $S_\text{min}$ adjusted steps (in the large-batch limit) is:

```math
\begin{align}
  L(N, S_\text{min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_\text{min}}\right)^{\alpha_S}
\end{align}
```

where $S_\text{min}(S) \equiv S / (1 + B_\text{crit}(L)/B)$ is the step count projected to the infinite-batch limit, and $\alpha_S \approx 0.76$, $S_c \approx 2.1 \times 10^3$.

Fitted parameters: $\alpha_N = 0.077$, $\alpha_S = 0.76$, $N_c = 6.5 \times 10^{13}$, $S_c = 2.1 \times 10^3$.

> [!NOTE]
> The two-term structure separates capacity limits ($N$-dependent term) from training-progress limits ($S$-dependent term). As $S_\text{min} \to \infty$ (fully converged), the formula recovers $L(N, \infty) = (N_c/N)^{\alpha_N}$.

---

## 5. Optimal Compute Allocation

### 5.1 Derivation of the Compute-Efficient Frontier

Given a fixed $C_\text{min}$ and the constraint $C_\text{min} = 6 N B_\text{crit} S_\text{min}$, the loss $L(N, S_\text{min})$ is minimized jointly over $N$ and $S_\text{min}$. The resulting effective compute exponent is:

```math
\begin{align}
  \alpha_{C_\text{min}} = \frac{1}{1/\alpha_S + 1/\alpha_B + 1/\alpha_N} \approx 0.054
\end{align}
```

This harmonic-mean form arises because the three terms in the denominator each contribute a "bottleneck": steps, batch size, and model size.

**Pseudocode for compute-optimal model selection**:

```
Given: compute budget C_target (in PF-days)
1. Compute optimal model size:
     N* = k_N * C_target^0.73    (k_N from fitted Table 6 constants)
2. Compute optimal dataset size:
     D* = k_D * C_target^0.27
3. Compute critical batch size:
     B* = B_crit( L(N*, inf) )   (from B_crit(L) = 2e8 / L^(1/0.21))
4. Compute training steps:
     S* = D* / B*
5. Train model with N* parameters on D* tokens with batch size B*
6. Stop training at S* steps (do NOT train to convergence)
```

### 5.2 Optimal Scaling Relationships

| Quantity | Power Law | Exponent | Scale |
|----------|-----------|----------|-------|
| $N_\text{opt}$ | $C_\text{min}^{p_N}$ | $p_N = 0.73$ | $N_e = 1.3 \times 10^9$ params |
| Batch size $B$ | $C_\text{min}^{p_B}$ | $p_B = 0.24$ | $B_e = 2.0 \times 10^6$ tokens |
| Steps $S_\text{min}$ | $C_\text{min}^{p_S}$ | $p_S = 0.03$ | $S_e = 5.4 \times 10^3$ |
| Data $D_\text{opt}$ | $C_\text{min}^{p_D}$ | $p_D = 0.27$ | $D_e = 2 \times 10^{10}$ tokens |

The exponents approximately sum to 1 ($0.73 + 0.24 + 0.03 = 1.00$). The dominant finding is that the vast majority of increased compute should go towards a larger model, not more training steps.

> [!IMPORTANT]
> "Convergence is inefficient": compute-optimal training stops at roughly $\approx 10\%$ above the fully converged loss, i.e., $L(N, C_\text{opt}) = (1 + f) \cdot L(N, \infty)$ with $f \approx 0.10$. Practitioners should prefer a large model trained briefly over a small model trained exhaustively.

### 5.3 Contradiction and Conjecture

Extrapolating both the compute-efficient frontier $N(C_\text{min})$ and the data-constrained limit $L(N, D)$ forward, the curves intersect near:

| Quantity | Conjectured Value |
|----------|-------------------|
| Compute $C^*$ | $\sim 10^4$ PF-days |
| Model size $N^*$ | $\sim 10^{12}$ parameters |
| Dataset $D^*$ | $\sim 10^{12}$ tokens |
| Loss $L^*$ | $\sim 1.7$ nats/token |

Beyond $C^*$, data limitations would prevent further improvement from scale alone. The value $L^* \approx 1.7$ nats/token may correspond to the irreducible entropy of natural language under the WebText2 distribution.

---

## 6. Generalization to Other Distributions

Models trained on WebText2 are evaluated on Books Corpus, Common Crawl, English Wikipedia, and Internet Books. Transfer loss on any target distribution correlates tightly with training loss, with a small positive offset $\delta$ that grows only slowly with scale:

```math
\begin{align}
  L_\text{transfer} \approx L_\text{train} + \delta
\end{align}
```

The offset $\delta$ is distribution-specific but does not depend on model size—the scaling exponents on all tested out-of-distribution datasets match those on WebText2. Improving in-distribution performance reliably predicts improved out-of-distribution performance.

---

## 7. Comparison with Related Work

| Aspect | This Work | Prior Work |
|--------|-----------|------------|
| Dataset–model scaling | $D \propto N^{0.74}$ (sub-linear) | Some prior work claimed super-linear |
| Dominant scale factor | Model size $N$ | Varied; often dataset size |
| Architectural sensitivity | Very weak for fixed $N$ | Depth/width often emphasized |
| LSTM vs. Transformer | Transformer better at long-range context | Mixed results in earlier literature |
| Universal Transformers | No advantage over standard when $N$ matched | Claimed benefits in prior work |

---

## Experiments

- **Dataset**: WebText2 — 20.3M documents, 96 GB, $2.29 \times 10^{10}$ tokens. No public train/dev/test split; the test set is a held-out portion ($6.6 \times 10^8$ tokens). Additional transfer evaluation on Books Corpus, Common Crawl, English Wikipedia, Internet Books.
- **Hardware**: Not explicitly stated; compute measured in PF-days.
- **Optimizer**: Adam for models $\leq 1$B params; Adafactor for larger models. Linear warmup over 3,000 steps, cosine decay over $2.5 \times 10^5$ steps.
- **Model sizes**: $N \in [768,\; 1.5 \times 10^9]$ non-embedding parameters.
- **Results**:
  - All three power laws ($L(N)$, $L(D)$, $L(C_\text{min})$) hold across 7+ orders of magnitude with consistent exponents.
  - Compute-optimal allocation: $N_\text{opt} \propto C_\text{min}^{0.73}$; most additional compute should go to larger models.
  - Data requirement grows slowly: $D \propto N^{0.74}$; trillion-parameter models require only $\sim 10^{12}$ tokens.
  - Stopping training at $\approx 10\%$ above convergence loss is compute-optimal.
  - Transformers outperform LSTMs at equivalent $N$, especially for tokens at long context positions.
  - Architectural shape (depth vs. width) has $< 3\%$ impact on loss for fixed $N$.

---

## Summary of Power-Law Parameters

| Scaling Law | Exponent | Scale Constant |
|-------------|----------|---------------|
| $\alpha_N$ (parameters) | 0.076 | $N_c = 8.8 \times 10^{13}$ |
| $\alpha_D$ (data) | 0.095 | $D_c = 5.4 \times 10^{13}$ tokens |
| $\alpha_C$ (naive compute) | 0.057 | $C_c = 1.6 \times 10^7$ PF-days |
| $\alpha_{C_\text{min}}$ (optimal compute) | 0.050 | $C_c^\text{min} = 3.1 \times 10^8$ PF-days |
| $\alpha_B$ (batch size) | 0.21 | $B^* = 2.1 \times 10^8$ tokens |
| $\alpha_S$ (steps) | 0.76 | $S_c = 2.1 \times 10^3$ steps |
