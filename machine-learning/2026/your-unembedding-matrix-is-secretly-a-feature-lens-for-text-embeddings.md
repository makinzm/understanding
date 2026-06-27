# Meta Information

- URL: [Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings](https://arxiv.org/abs/2606.07502)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Wu, S., Chen, Z., Liu, Y., Cui, H., Li, C., & Yan, R. (2026). Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings. arXiv:2606.07502.

# Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings

## Applicability

This paper is relevant to engineers building zero-shot text embedding systems on top of decoder-only LLMs (e.g., Qwen, Llama, Mistral) for tasks such as semantic search, retrieval, clustering, and reranking. It applies specifically when embeddings are extracted from a pretrained autoregressive LLM's hidden states (e.g., via PromptEOL or ECHO-style prompting) without any embedding-specific fine-tuning, and when the practitioner wants a training-free post-hoc fix that also reduces storage cost through dimensionality reduction.

## 1. Problem: Frequency Bias in LLM-Derived Embeddings

Large decoder-only LLMs underperform as zero-shot text embedding models despite strong general language capabilities. The authors use **Logit Lens** — projecting an intermediate hidden state $h$ onto the vocabulary via the unembedding matrix to inspect which tokens it "decodes" to — and observe that the resulting top-probability tokens are dominated by high-frequency, semantically uninformative tokens (e.g., common stop-words), consistently across Qwen, Llama, and Mistral model families.

> [!NOTE]
> "text embeddings tend to align with frequent but uninformative tokens when projected onto the vocabulary space."

The authors hypothesize that raw sentence embeddings are pulled toward a corpus-level "average token" — the frequency-weighted centroid of the training vocabulary — which collapses representations into a narrow anisotropic cone and suppresses task-relevant semantic signal.

## 2. Mathematical Framework

### 2.1 Unembedding Matrix and SVD

Let $|\mathcal{V}|$ be the vocabulary size and $d$ the hidden-state dimension. The unembedding matrix $W_{\mathcal{U}} \in \mathbb{R}^{|\mathcal{V}| \times d}$ maps a hidden state $h \in \mathbb{R}^{d}$ to next-token probabilities:

```math
\begin{align}
  q = \mathrm{Softmax}(h W_{\mathcal{U}}^\top)
\end{align}
```

Its singular value decomposition is:

```math
\begin{align}
  W_{\mathcal{U}} = U \Sigma V^\top
\end{align}
```

where $U \in \mathbb{R}^{|\mathcal{V}| \times d}$ holds the left singular vectors, $\Sigma \in \mathbb{R}^{d \times d}$ is diagonal with singular values, and $V \in \mathbb{R}^{d \times d}$ holds the right singular vectors (an orthonormal basis of the $d$-dimensional hidden-state space, indexed $1, \dots, d$ from smallest to largest singular value).

### 2.2 Reverse-Engineering the "Average Token"

Given empirical unigram token frequencies $\hat{p} \in \mathbb{R}^{|\mathcal{V}|}$ estimated from a reference corpus (the authors use RedPajama, since the model's actual pretraining distribution is unavailable) and the Moore–Penrose pseudo-inverse $W_{\mathcal{U}}^{+} \in \mathbb{R}^{d \times |\mathcal{V}|}$:

```math
\begin{align}
  \hat{h} = \log(\hat{p}) \, W_{\mathcal{U}}^{+}
\end{align}
```

$\hat{h} \in \mathbb{R}^{d}$ is the reconstructed hidden vector that would decode into the corpus's frequency-weighted token distribution — i.e., an estimate of the "average token" direction embeddings collapse toward.

### 2.3 Logit Spectroscopy: Locating the Edge Spectrum

To find which singular directions of $W_{\mathcal{U}}$ are responsible for writing frequency information, the authors ablate each right-singular direction $V_{[i]} \in \mathbb{R}^{d}$ (column $i$ of $V$) one at a time with a projector that removes it:

```math
\begin{align}
  \Psi_i = I - V_{[i]} V_{[i]}^\top
\end{align}
```

For the $k$ most frequent tokens $\mathcal{V}^{+} \subset \mathcal{V}$, they measure how much ablating direction $i$ changes those tokens' logit weights $\hat{w}_j$ (row $j$ of $W_{\mathcal{U}}$) versus the ablated weights $\tilde{w}_j^{(i)}$:

```math
\begin{align}
  \Delta\pi^{(i)} = \frac{\sum_{j \in \mathcal{V}^{+}} \left| \tilde{w}_j^{(i)} - \hat{w}_j \right|}{\sum_{j \in \mathcal{V}^{+}} \left| \hat{w}_j \right|}
\end{align}
```

Plotting $\Delta\pi^{(i)}$ against singular index $i$ (from 1 to $d$) reveals that the directions with the *smallest* and *largest* singular values — the two ends of the spectrum — have disproportionately high $\Delta\pi$, i.e., they are the directions most responsible for encoding frequent-token information. The authors name this the **edge spectrum** subspace, in contrast to the **bulk spectrum** (the middle range of singular directions).

### 2.4 EmbedFilter

EmbedFilter removes the edge spectrum and keeps only the bulk. Let $\tau$ be a filtering ratio and $l_\tau, r_\tau$ the start/end column indices of $V$ that define the retained middle range (so that $r_\tau - l_\tau = d/\tau$). The **Bulk Spectrum Transformation** is:

```math
\begin{align}
  \Phi_\tau = V_{[l_\tau:r_\tau]} V_{[l_\tau:r_\tau]}^\top
\end{align}
```

where $V_{[l_\tau:r_\tau]} \in \mathbb{R}^{d \times (d/\tau)}$. Given a raw sentence embedding $e_i \in \mathbb{R}^{d}$ (extracted from the LLM via a prompting method such as PromptEOL or ECHO), the refined embedding is:

```math
\begin{align}
  \tilde{e}_i = e_i \, \Phi_\tau^\top
\end{align}
```

**Calculation order:**
1. Compute (or load precomputed) $U, \Sigma, V$ from the model's unembedding matrix $W_{\mathcal{U}}$ via SVD.
2. Determine the edge-spectrum boundary indices $l_\tau, r_\tau$ from the filtering ratio $\tau$ (no per-task calibration needed — this is a fixed, model-level computation done once).
3. Build $\Phi_\tau = V_{[l_\tau:r_\tau]} V_{[l_\tau:r_\tau]}^\top$.
4. For each input text, obtain raw embedding $e_i$ from the LLM.
5. Apply $\tilde{e}_i = e_i \Phi_\tau^\top$ to obtain the filtered, optionally lower-dimensional embedding.

### 2.5 Why This Also Reduces Dimensionality

Because $V$ is orthogonal, projecting onto $V_{[l_\tau:r_\tau]}$ (a $d/\tau$-dimensional subspace) preserves pairwise distances up to that projection:

```math
\begin{align}
  \lVert x \Phi_\tau^\top - y \Phi_\tau^\top \rVert_2 = \lVert x V_{[l_\tau:r_\tau]} - y V_{[l_\tau:r_\tau]} \rVert_2
\end{align}
```

so instead of materializing $\tilde{e}_i \in \mathbb{R}^d$, one can directly store the coordinates $e_i V_{[l_\tau:r_\tau]} \in \mathbb{R}^{d/\tau}$, cutting storage by a factor of $\tau$ while approximately preserving the geometry that downstream similarity search relies on.

> [!IMPORTANT]
> EmbedFilter is training-free and requires no task-specific calibration data: $\Phi_\tau$ depends only on the frozen unembedding matrix $W_{\mathcal{U}}$ and the chosen ratio $\tau$, not on the downstream task or any labeled examples.

## 3. Differences from Related Methods

| Method | Mechanism | Calibration data needed | Dimensionality reduction |
|---|---|---|---|
| PromptEOL | Prompts the LLM with a "summarize in one word" template to concentrate semantics into the final token's hidden state | None | No |
| ECHO | Duplicates the input sequence to let early tokens attend to later context, mitigating the causal-attention bottleneck | None | No |
| MetaEOL / GenEOL | Aggregate embeddings across multiple meta-task prompts or LLM-generated variants | None | No |
| BERT-Whitening | Whitens embeddings using statistics (mean, covariance) estimated on calibration data, addressing general anisotropy | Yes (calibration corpus) | Yes, via PCA-style truncation |
| **EmbedFilter (this paper)** | Removes a specific edge-spectrum subspace of the unembedding matrix's SVD, identified via Logit Spectroscopy as the source of frequency bias | No (only needs $W_{\mathcal{U}}$, already part of the model) | Yes, distance-preserving via orthogonal projection |

Unlike whitening-based approaches, which treat anisotropy as a generic statistical artifact to be corrected post hoc with corpus statistics, EmbedFilter targets a specific, mechanistically identified subspace of the model's own unembedding matrix, requiring no external calibration data and composing directly with existing prompting-based embedding extraction methods (PromptEOL, ECHO) as a drop-in refinement.

## 4. Ablation Study

Using Qwen2.5-0.5B with filtering ratio $\tau = 2$, six filtering strategies were compared (MTEB average score):

- **Truncation** (remove the first half of dimensions, Matryoshka-style): underperforms the unfiltered baseline.
- **Random selection** (randomly drop half the dimensions): underperforms the baseline.
- **Dominant** (filter out only the largest-singular-value subspace): 47.53, worst among edge-based strategies.
- **Secondary** (filter out only the smallest-singular-value subspace): 53.19, moderate improvement.
- **Bulk** (inverse of EmbedFilter — keep only the edges, discard the bulk): 47.13, poorest result, confirming the bulk subspace carries the useful semantic signal.
- **Optimal** (oracle: filter exactly the singular vectors with the highest $\Delta\pi^{(i)}$ per model): 54.19, near the practical ceiling.
- **EmbedFilter** (filter both edges via the fixed $\Phi_\tau$ rule, no per-model search): 54.57, matching or exceeding the oracle "Optimal" setting without requiring the expensive per-direction $\Delta\pi$ search at inference time.

This shows the gain comes specifically from removing *both* spectral edges (not just one), and that EmbedFilter's fixed rule is as effective as an oracle that searches for the best directions per model.

## 5. Limitations

- The filtering rule is described by the authors as "primarily heuristic" rather than derived from a closed-form theoretical justification of why exactly the spectral edges encode frequency.
- Token frequency statistics used to reverse-engineer the "average token" direction are approximated from the open RedPajama corpus rather than each model's actual (proprietary) pretraining data.
- Due to compute constraints, only a subset of MTEB's retrieval tasks was evaluated.
- The filtering boundaries $l_\tau, r_\tau$ are applied symmetrically; the authors note an asymmetric edge-filtering strategy is unexplored and left to future work.
- The authors explicitly state they leave a deeper mechanistic investigation of *why* frequency information concentrates at the spectral edges to future work.

> [!TIP]
> Code is released at https://github.com/CentreChen/EmbFilter, which is useful for reproducing the SVD-based filtering pipeline without re-deriving $\Phi_\tau$ from scratch.

# Experiments

- Dataset: MTEB (Massive Text Embedding Benchmark), covering 49 datasets across 7 task categories — Semantic Textual Similarity (10 datasets: STS12–17, SICK-R, STSB, BIOSSES), Classification (12), Clustering (11), Pair Classification (3), Reranking (4), Retrieval (8: SciFact, ArguAna, NFCorpus, FiQA2018, QuoraRetrieval, SCIDOCS, Touche2020, TRECCOVID), and Summarization (1). Token-frequency statistics for the "average token" reconstruction are estimated from the RedPajama corpus.
- Models: Qwen, Llama, and Mistral families, embeddings extracted via PromptEOL and ECHO prompting.
- Baselines: BERT-Whitening, SimCSE-BERT-sup, coCondenser-msmarco, alongside the unfiltered PromptEOL/ECHO embeddings.
- Results: EmbedFilter improves PromptEOL on Qwen-0.5B by +9.0 points (50.07 → 54.57 MTEB average) and ECHO on Qwen-0.5B by +14.1 points (46.03 → 52.55); on Llama-8B, ECHO improves by +7.8 points (53.52 → 57.70). These gains are retained even when simultaneously reducing embedding dimensionality by a factor of $\tau = 8$, and post-filtering Logit Lens inspection confirms embeddings now align with semantically meaningful tokens rather than high-frequency stop-words.
