# Meta Information

- URL: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. arXiv:2504.19874.

# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

## Overview

**TurboQuant** is a data-oblivious vector quantization algorithm that compresses $d$-dimensional vectors $x \in \mathbb{R}^d$ into $b$ bits per coordinate while minimizing reconstruction distortion. It targets two distortion objectives: mean-squared error (MSE) and inner product estimation error. Its core insight is that applying a random rotation to a high-dimensional vector makes its coordinates nearly independent, allowing near-optimal vector quantization to be reduced to per-coordinate scalar quantization.

**Applicability:** TurboQuant is designed for online, low-latency workloads (e.g., KV cache compression in LLM inference, real-time vector database indexing) where data-dependent preprocessing (like k-means for Product Quantization) is too expensive or infeasible.

## Problem Formulation

Given vectors $x$ drawn uniformly from the unit sphere $\mathbb{S}^{d-1}$, the goal is to design a quantization map $Q: \mathbb{R}^d \rightarrow \{0,1\}^{b \cdot d}$ that minimizes either:

**MSE distortion:**
```math
\begin{align}
  D_{\text{mse}}(Q) = \mathbb{E}\left[\|x - Q^{-1}(Q(x))\|_2^2\right]
\end{align}
```

**Inner product distortion:** for a query $y \in \mathbb{R}^d$,
```math
\begin{align}
  D_{\text{prod}}(Q) = \mathbb{E}\left[\left|\langle y, x \rangle - \langle y, Q^{-1}(Q(x)) \rangle\right|^2\right]
\end{align}
```
subject to the unbiasedness constraint $\mathbb{E}[Q^{-1}(Q(x))] = x$.

## Information-Theoretic Lower Bounds

The paper establishes lower bounds from Shannon's source coding theory, showing that for any quantizer $Q$ operating at $b$ bits per coordinate:

```math
\begin{align}
  D_{\text{mse}}(Q) &\geq \frac{1}{4^b} \\
  D_{\text{prod}}(Q) &\geq \frac{1}{d} \cdot \frac{1}{4^b}
\end{align}
```

> [!NOTE]
> The MSE bound follows from the Shannon lower bound for sources on the hypersphere, which states $D(B) \geq 2^{-2B/d}$ for total budget $B = b \cdot d$ bits.

TurboQuant achieves approximately $2.7\times$ the MSE lower bound (close to $1.45\times$ at $b=1$), making it provably near-optimal.

## Key Technical Component: Random Rotation and Beta Distribution

**Lemma:** If $x$ is drawn uniformly from $\mathbb{S}^{d-1}$ and $\Pi$ is a random orthonormal rotation matrix, then each coordinate $y_i = (\Pi x)_i$ follows a $\text{Beta}((d-1)/2, (d-1)/2)$ distribution scaled to $[-1, 1]$, which converges to $\mathcal{N}(0, 1/d)$ as $d \to \infty$.

Crucially, distinct coordinates of $y = \Pi x$ are approximately independent in high dimensions. This allows the optimal codebook for each coordinate to be precomputed analytically (via the Lloyd-Max algorithm on a Beta distribution), eliminating any need for data-dependent training.

## Algorithm 1: TurboQuant_mse

**Input:** $x \in \mathbb{R}^d$, bit-width $b$, precomputed codebooks $\mathcal{C}_b$ for Beta distribution.

**Output:** Quantized representation and reconstruction $\hat{x} \in \mathbb{R}^d$.

1. Sample random rotation matrix $\Pi \in \mathbb{R}^{d \times d}$ (fixed for a session)
2. Rotate: $y = \Pi x \in \mathbb{R}^d$
3. For each coordinate $i \in [d]$: find nearest centroid $c_i = \arg\min_{c \in \mathcal{C}_b} |y_i - c|$
4. Store index $j_i$ of $c_i$ using $b$ bits
5. **Dequantize:** $\hat{y}_i = c_{j_i}$; reconstruct $\hat{x} = \Pi^\top \hat{y}$

**Distortion bound:**
```math
\begin{align}
  D_{\text{mse}} \leq \frac{\sqrt{3}\pi}{2} \cdot \frac{1}{4^b}
\end{align}
```

At $b = 1, 2, 3, 4$ bits, the empirical distortion values are approximately $0.36, 0.117, 0.03, 0.009$.

> [!IMPORTANT]
> The random rotation $\Pi$ is data-oblivious — it does not depend on any training set. This makes TurboQuant applicable to any data distribution at runtime with no preprocessing overhead.

## Algorithm 2: TurboQuant_prod (Inner Product Quantizer)

**Problem with TurboQuant_mse for inner products:** The MSE-optimal quantizer is biased — $\mathbb{E}[\hat{x}] \neq x$ in general — which degrades inner product estimates.

**Solution:** A two-stage approach combining MSE quantization with a Quantized Johnson-Lindenstrauss (QJL) transform on the residual.

**Input:** $x \in \mathbb{R}^d$, query $y \in \mathbb{R}^d$, bit-width $b$.

**Output:** Unbiased estimate $\widehat{\langle y, x \rangle}$.

1. **Stage 1 – MSE quantization:** Apply TurboQuant_mse with bit-width $(b-1)$, obtaining $\hat{x}^{(1)} \in \mathbb{R}^d$
2. **Stage 2 – Residual quantization:** Compute residual $r = x - \hat{x}^{(1)}$; apply QJL 1-bit quantization to $r$, yielding $\hat{r}$ such that $\mathbb{E}[\hat{r}] = r$
3. **Reconstruct:** $\hat{x} = \hat{x}^{(1)} + \hat{r}$; estimate $\langle y, x \rangle \approx \langle y, \hat{x} \rangle$

**Distortion bound:**
```math
\begin{align}
  D_{\text{prod}} \leq \frac{\sqrt{3}\pi^2 \|y\|_2^2}{d} \cdot \frac{1}{4^b}
\end{align}
```

> [!NOTE]
> The QJL transform used in Stage 2 is a 1-bit sketch $\text{sign}(G x)$ for a Gaussian matrix $G$, scaled to ensure $\mathbb{E}[\hat{r}] = r$. The unbiasedness property carries through to the final inner product estimate.

## Comparison with Similar Methods

| Method | Data-oblivious | Unbiased IP | Near-optimal MSE | Indexing time |
|---|---|---|---|---|
| Product Quantization | No (requires k-means) | No | No | Minutes–hours |
| RabitQ | Partially | Yes | Partially (1-bit) | Fast |
| QJL | Yes | Yes | No (1-bit only) | Near-zero |
| **TurboQuant** | **Yes** | **Yes** | **Yes** | **~0.001s** |

**vs. Product Quantization:** PQ partitions the vector into subspaces and applies k-means within each, achieving good empirical distortion but requiring expensive offline codebook training. TurboQuant precomputes codebooks analytically from the Beta distribution and needs no training data.

**vs. RabitQ:** RabitQ also uses random rotations but applies a single 1-bit quantizer per coordinate. TurboQuant extends this to arbitrary $b$ bits and provides theoretical guarantees for both MSE and inner product objectives simultaneously.

**vs. QJL:** QJL achieves unbiased inner product estimation at 1 bit per coordinate. TurboQuant_prod generalizes QJL to arbitrary bit-widths by combining it with an MSE quantizer for lower total distortion at the same bit budget.

# Experiments

- **Datasets:**
  - DBpedia Entities: 100K vectors, OpenAI `text-embedding-3` embeddings, $d = 1536$ and $d = 3072$
  - GloVe: $d = 200$, 1.1M vectors
  - OpenAI `text-embedding-3-small`: $d = 1536$, 1M vectors
  - OpenAI `text-embedding-3-large`: $d = 3072$, 100K vectors
  - LongBench: long-context NLP tasks (single-doc QA, multi-doc QA, summarization, few-shot, code)
- **Models (KV cache experiments):** Llama-3.1-8B-Instruct, Ministral-7B-Instruct
- **Results:**
  - Empirical distortion on DBpedia closely matches theoretical bounds at all bit-widths; unbiasedness of TurboQuant_prod confirmed
  - Needle-in-a-Haystack (4× compression, 2 bits/value): score 0.997 vs. 1.000 full precision; outperforms SnapKV, PyramidKV, KIVI, PolarQuant (~0.95)
  - LongBench at 3.5 bits: matches 16-bit full precision; at 2.5 bits (>4.5× compression): minimal degradation, outperforms KIVI and PolarQuant
  - ANN Recall@10 on GloVe and OpenAI embeddings: surpasses Product Quantization and RabitQ at equal bit-width; indexing time ~0.001s vs. minutes for PQ
