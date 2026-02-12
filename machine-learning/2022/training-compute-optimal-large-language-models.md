# Meta Information

- URL: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskiy, E., Cai, T., Millican, K., ... & Sifre, L. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556.

> [!CAUTION]
> NOTE comments are my personal understanding and may contain errors.

# 1. Introduction

This paper (commonly known as the "Chinchilla" paper) challenges the prevailing assumption that larger models always lead to better performance, given a fixed compute budget. Prior work by Kaplan et al. (2020) suggested that model size should scale faster than training data when compute is increased. The authors of this paper empirically demonstrate that **model size and training tokens should be scaled equally**, meaning that a 10× increase in compute should yield approximately 3.16× more parameters *and* 3.16× more training tokens.

> [!NOTE]
> 3.16 is calculated from $\sqrt{10}$

The core implication is that many large language models (e.g., Gopher at 280B parameters trained on 300B tokens) are "significantly undertrained" relative to their compute budget, and a smaller model trained on substantially more data would outperform them.

> [!NOTE]
> "We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant."

# 2. Estimating Optimal Model and Data Scaling

## 2.1 Setup

All models are **autoregressive transformer language models** trained on the **MassiveText** dataset. The compute budget $C$ (in FLOPs) for training a transformer model is approximated as:

$$C \approx 6ND$$

where:
- $N \in \mathbb{Z}^+$: number of model parameters
- $D \in \mathbb{Z}^+$: number of training tokens
- The factor of 6 accounts for one forward pass and two backward passes, each requiring $\approx 2ND$ FLOPs

The objective is: given a fixed compute budget $C$, find the optimal $(N_{\text{opt}}, D_{\text{opt}})$ that minimizes the final pre-training loss $L(N, D)$.

## 2.2 Approach 1 — Fix Model Size, Vary Tokens

Models ranging from 70M to 10B parameters are trained at four different token horizons (scales of tokens). For each model size, the point of minimum loss for a given FLOP budget is identified by fitting an envelope over training loss curves.

**Result**: Both $N_{\text{opt}}$ and $D_{\text{opt}}$ scale as $C^{0.50}$, i.e., $a = b = 0.50$.

## 2.3 Approach 2 — IsoFLOP Profiles

Multiple model sizes (from 70M to 16B parameters) are trained at each of nine fixed FLOP budgets ranging from $6 \times 10^{18}$ to $3 \times 10^{21}$ FLOPs. For each FLOP budget, the model size achieving minimum loss is identified.

**Result**: $a = 0.49$, $b = 0.51$ — closely matching Approach 1.

## 2.4 Approach 3 — Parametric Loss Modeling

A parametric loss function is fit over all $(N, D, L)$ triplets from all training runs:

$$\hat{L}(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

where:
- $E$: entropy of natural text (irreducible loss)
- $A / N^{\alpha}$: loss reduction from model approximation (decreases with more parameters)
- $B / D^{\beta}$: loss reduction from optimization (decreases with more data)

The fit is performed via L-BFGS minimization of Huber loss. Fitted parameters:

| Parameter | Value |
|-----------|-------|
| $E$       | 1.69  |
| $A$       | 406.4 |
| $B$       | 410.7 |
| $\alpha$  | 0.34  |
| $\beta$   | 0.28  |

**Result**: $a = 0.46$, $b = 0.54$.

## 2.5 Compute-Optimal Frontier

Given the parametric model, the optimal allocation can be derived analytically by minimizing $\hat{L}(N, D)$ subject to $C = 6ND$:

$$N_{\text{opt}}(C) \propto C^{a}, \quad D_{\text{opt}}(C) \propto C^{b}$$

All three approaches yield $a \approx b \approx 0.50$, indicating that **parameters and tokens should scale equally** with compute.

> [!IMPORTANT]
> Kaplan et al. (2020) predicted $a = 0.73$ and $b = 0.27$, placing far greater weight on model size. This paper's empirical finding directly contradicts that result. The key difference is that Kaplan et al. did not sufficiently train smaller models to convergence, biasing their estimates toward larger model sizes.

## 2.6 Optimal Token-to-Parameter Ratio

The compute-optimal rule implies:

$$\frac{D_{\text{opt}}}{N_{\text{opt}}} \approx 20 \text{ tokens/parameter}$$

This means: for every 1 billion parameters, a compute-optimal model should be trained on approximately **20 billion tokens**.

# 3. Chinchilla: Validation of the Scaling Law

## 3.1 Model Design

To empirically validate the compute-optimal hypothesis, the authors train **Chinchilla** using the same compute budget as Gopher (280B), but with $\approx 4\times$ fewer parameters and $\approx 4\times$ more training data:

| Property              | Chinchilla | Gopher   |
|-----------------------|-----------|----------|
| Parameters            | 70B       | 280B     |
| Training Tokens       | 1.4T      | 300B     |
| Layers                | 80        | 80       |
| Attention Heads       | 64        | 128      |
| Key/Value Size        | 128       | 128      |
| Model Dimension $d$   | 8,192     | 16,384   |
| Max Learning Rate     | $1\times10^{-4}$ | $4\times10^{-5}$ |
| Batch Size (tokens)   | 1.5M–3M   | 3M–6M    |

## 3.2 Architecture

Both Chinchilla and Gopher are **dense autoregressive transformer** language models (decoder-only). They use:
- **Input**: token sequence $x \in \mathbb{Z}^{T}$ (token IDs), embedded to $x \in \mathbb{R}^{T \times d}$
- **Output**: probability distribution over vocabulary at each position, $p \in \mathbb{R}^{T \times V}$, where $V$ is vocabulary size

## 3.3 Training Configuration

- **Optimizer**: AdamW with gradient clipping
- **Learning Rate Schedule**: cosine decay with $10\times$ decay
  - Max LR ranges from $2\times10^{-4}$ (smallest models) to $1\times10^{-4}$ (Chinchilla) to $4\times10^{-5}$ (Gopher)
- **Hardware**: TPUv3 and TPUv4 pods
- **Framework**: JAX + Haiku

# 4. Experimental Results

## 4.1 Datasets Used

| Dataset          | Proportion in MassiveText |
|-----------------|---------------------------|
| MassiveWeb       | 45%                       |
| Books            | 30%                       |
| C4               | 10%                       |
| News             | 10%                       |
| GitHub           | 4%                        |
| Wikipedia        | 1%                        |

- **MassiveText total size**: 12.8TB (filtered)
- **C4**: used as held-out validation set
- All models trained for less than one epoch to avoid data repetition

## 4.2 Key Benchmark Results

| Benchmark                  | Chinchilla | Gopher | GPT-3 (175B) |
|---------------------------|-----------|--------|-------------|
| MMLU (5-shot)             | **67.6%** | 60.0%  | 43.9%       |
| BIG-bench (avg, 62 tasks) | **65.1%** | 54.4%  | —           |
| LAMBADA                   | **77.4%** | 74.5%  | 76.2%       |
| RACE-middle               | **86.8%** | 75.1%  | —           |
| RACE-high                 | **82.3%** | 71.6%  | —           |
| Natural Questions (5-shot)| **31.5%** | 24.5%  | 29.9%       |
| TriviaQA unfiltered (5-shot) | **73.2%** | 63.6% | 71.2%   |

Chinchilla outperforms Gopher (4× larger), GPT-3 (2.5× larger), and MT-NLG 530B (7.5× larger) on most benchmarks, while using the same or less compute during pre-training.

# 5. Comparison with Prior Work (Kaplan et al. 2020)

| Aspect                         | Kaplan et al. (2020)     | This Work (Chinchilla)       |
|-------------------------------|--------------------------|------------------------------|
| Scaling recommendation        | $N \propto C^{0.73}$     | $N \propto C^{0.50}$         |
| Token scaling exponent        | $D \propto C^{0.27}$     | $D \propto C^{0.50}$         |
| 10× compute → model scale     | 5.5× larger model        | 3.16× larger model           |
| 10× compute → data scale      | 1.8× more tokens         | 3.16× more tokens            |
| Key claim                     | Prioritize model size    | Scale model and data equally |

**Why the discrepancy?** Kaplan et al. stopped training runs early (before convergence) for smaller models, making small-but-well-trained models appear inferior. When all models are trained to convergence, the benefit of smaller models trained on more data becomes apparent.

# 6. Practical Implications

1. **Inference efficiency**: Chinchilla (70B) is far cheaper to serve than Gopher (280B) or GPT-3 (175B), as inference cost scales with model size, not training tokens.
2. **Memory requirements**: Smaller models fit on fewer accelerators, reducing deployment costs.
3. **Data quality matters more**: As models shrink and token counts grow, the bottleneck shifts to obtaining large, high-quality training corpora.
4. **Who benefits**: Organizations training large language models from scratch, or researchers setting up new pre-training runs, should apply these scaling laws to avoid wasting compute.

> [!TIP]
> A practical rule of thumb derived from this work: train a model for approximately **20 tokens per parameter**. For a 7B model, this means ~140B tokens; for a 70B model, ~1.4T tokens.

# Experiments

- **Datasets**: MassiveText (MassiveWeb 45%, Books 30%, C4 10%, News 10%, GitHub 4%, Wikipedia 1%); C4 and GitHub for held-out validation
- **Hardware**: TPUv3 and TPUv4 clusters
- **Framework**: JAX + Haiku
- **Optimizer**: AdamW with cosine LR decay ($10\times$ decay), gradient clipping
- **Models trained**: 400+ models ranging from 70M to 16B parameters for scaling law derivation
- **Chinchilla**: 70B parameters, 1.4T tokens, same compute budget as Gopher (280B, 300B tokens)
- **Key results**: Chinchilla achieves 67.6% on MMLU (vs. 60.0% for Gopher), outperforming all prior models at equal or less compute
