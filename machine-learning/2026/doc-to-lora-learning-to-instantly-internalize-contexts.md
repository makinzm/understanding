# Meta Information

- URL: [Doc-to-LoRA: Learning to Instantly Internalize Contexts](https://arxiv.org/abs/2602.15902)
- LICENSE: [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Reference: Charakorn, R., Cetin, E., Uesaka, S., & Lange, R. T. (2026). Doc-to-LoRA: Learning to Instantly Internalize Contexts. arXiv:2602.15902.

# Doc-to-LoRA: Learning to Instantly Internalize Contexts

Doc-to-LoRA (D2L) is a hypernetwork that meta-learns to perform approximate **context distillation** (CD) in a single forward pass. Given a document (context) $c$, D2L generates a LoRA adapter for a frozen target LLM so that subsequent queries can be answered *without* re-consuming the original context. This removes the quadratic KV-cache overhead of long in-context learning while preserving—and often improving upon—the factual internalization quality of conventional CD.

**Applicability:** Useful when (1) the same document is queried repeatedly, (2) the context exceeds the target LLM's native window, or (3) low-latency knowledge internalization is required (e.g., on-device, interactive, personalized systems).

---

## 1. Background: Context Distillation

Context distillation (CD) is a self-distillation technique: the same LLM acts as both teacher (with context $c$) and student (without context). The teacher samples response $y \sim p_\theta(\cdot \mid x, c)$ for query $x$; the student is then trained to match the teacher without seeing $c$.

**Query-dependent objective (overfits on single $(c,x,y)$):**

```math
\begin{align}
\min_{\theta_c} \mathrm{KL}\!\left(p_\theta(y \mid x, c) \;\|\; p_{\theta_c}(y \mid x)\right)
\end{align}
```

**Query-independent objective (robust; uses multiple queries $\mathcal{D}_c = \{(x_i, y_i)\}$):**

```math
\begin{align}
\min_{\theta_c} \;\mathbb{E}_{(x,y)\sim\mathcal{D}_c}\!\left[\mathrm{KL}\!\left(p_\theta(y \mid x, c) \;\|\; p_{\theta_c}(y \mid x)\right)\right]
\end{align}
```

Conventional CD requires iterating gradient descent *per context*, making it impractically slow (>100 s per sample for realistic query budgets).

---

## 2. Method: Meta-Learning Context Distillation

### 2.1. Meta-Training Objective

D2L learns a hypernetwork $H_\phi$ that maps a context $c$ to LoRA weight deltas $\Delta W_c$, so that the adapted model $\theta + \Delta W_c$ imitates the context-conditioned teacher $p_\theta(\cdot \mid x, c)$ for arbitrary $(x, c)$ pairs.

**Meta-training dataset** $\mathcal{D} = \{(c_i, \mathcal{D}_{c_i})\}_{i=1}^n$ contains diverse (context, query, response) triplets.

**Meta-training objective:**

```math
\begin{align}
\min_\phi \;\mathbb{E}_{(c,\mathcal{D}_c)\sim\mathcal{D}}\; \mathbb{E}_{(x,y)\sim\mathcal{D}_c}\; \mathrm{KL}\!\left(p_\theta(y \mid x, c) \;\|\; p_{\theta + H_\phi(c)}(y \mid x)\right)
\end{align}
```

After meta-training, $H_\phi$ is frozen and reused for *any* new context: one forward pass through $H_\phi$ replaces the entire per-context training loop.

### 2.2. D2L Architecture

**Input encoding:** The frozen target LLM processes context $c$ and returns per-layer token activations $Z \in \mathbb{R}^{L \times N \times D}$, where $L$ = number of Transformer layers (including the embedding layer), $N$ = number of context tokens, $D$ = hidden size. Activations at layer $l$ are $Z_l \in \mathbb{R}^{N \times D}$.

**Perceiver-based hypernetwork $h_\phi$:** For each layer $l$, a shared Perceiver-style cross-attention module maps the variable-length token activations $Z_{l-1}$ to $r$ fixed latent queries $Q_m \in \mathbb{R}^{r \times d_q}$:

```math
\begin{align}
U_l = \mathrm{XAttn}\!\left(Q_m,\; K(Z_{l-1}),\; V(Z_{l-1})\right) \in \mathbb{R}^{r \times d_u}
\end{align}
```

Two linear output heads convert $U_l$ into LoRA matrices and apply them to the target layer:

```math
\begin{align}
W'_l = W_l + \alpha_l B_l A_l;\quad A_l \in \mathbb{R}^{r \times d^{\mathrm{in}}_l},\; B_l \in \mathbb{R}^{d^{\mathrm{out}}_l \times r}
\end{align}
```

where $\alpha_l$ is a learnable per-layer scalar and $r$ is the LoRA rank. The Perceiver design naturally handles variable $N$ (context length) and always produces a fixed-size adapter.

**Long-context chunking:** For contexts longer than training-time sequences, $c$ is split into $K$ equal-sized chunks $\{c^{(k)}\}_{k=1}^K$. Each chunk is processed independently, yielding $(A_l^{(k)}, B_l^{(k)})$. The chunk adapters are combined by concatenation along the rank dimension:

```math
\begin{align}
A_l = \begin{bmatrix} A^{(1)}_l \\ \vdots \\ A^{(K)}_l \end{bmatrix}, \quad B_l = \begin{bmatrix} B^{(1)}_l \cdots B^{(K)}_l \end{bmatrix}
\end{align}
```

This yields an effective rank of $r \cdot K$ without changing the hypernetwork's output shape. Crucially, D2L generalizes to chunk counts far beyond those seen in training.

### 2.3. Forward Pass (Pseudocode)

```python
def forward(LLM, hypernet, ctx_ids, ctx_attn_mask, input_ids, input_attn_mask):
    # 1) Encode context: [n_chunks, seq_len] -> [n_chunks, n_layers, seq_len, D]
    features = LLM.forward(ctx_ids, ctx_attn_mask).detach()
    # 2) Perceiver: [n_chunks, n_layers, r, d_latent]
    emb = hypernet.perceiver(features, ctx_attn_mask)
    # 3) Head: [n_chunks, n_layers, r, d_in + d_out]
    lora_flat = hypernet.head(emb)
    # 4) Combine across chunks: [n_layers, r*n_chunks, d_in + d_out]
    lora = combine_lora(lora_flat, n_chunks=ctx_ids.shape[0])
    # 5) Apply LoRA to base model layers; then run query
    apply_lora_to_layers(LLM, lora)
    return LLM.forward(input_ids, input_attn_mask)
```

> [!NOTE]
> The context encoder forward pass is run with `detach()` so gradients do not flow back into the frozen target LLM. Only the hypernetwork parameters $\phi$ are updated during meta-training.

### 2.4. Comparison with Related Methods

| Method | Internalization mechanism | Per-context gradient steps | Context length flexibility | Latency |
|--------|--------------------------|---------------------------|---------------------------|---------|
| **Standard ICL** | KV-cache (in-context) | 0 | Limited by window size | Fast (one pass) |
| **CD** | LoRA fine-tuning per context | Many (SGD loop) | Arbitrary (slow) | >40–465 s |
| **T2L (Charakorn et al., 2025)** | Hypernetwork on instruction text | 0 | Fixed input | Fast |
| **D2L (this work)** | Hypernetwork on any document | 0 (one forward pass) | Arbitrary via chunking | <1 s |
| **LLMLingua-2** | Prompt compression (token-level) | 0 | Reduces tokens in-context | Fast, but keeps context |

> [!TIP]
> T2L (Text-to-LoRA) by the same authors is trained on SFT-style instruction datasets with NTP loss. D2L uses a KL distillation objective that matches the full teacher distribution, leading to richer internalization. D2L also handles arbitrary documents, not just short instructions.

---

## 3. Needle-in-a-Haystack (NIAH) Experiments

### Task

The model must locate a "needle" sentence (e.g., `"The special magic number is 0042."`) within a long haystack document and recall the 4-digit number when prompted, *without* access to the original context at query time.

**Base LLM:** `gemma-2-2b-it` (8K native context window).

**D2L training:** Input 32–256 tokens; 1–8 random chunks; 640K samples; 1 epoch; lr $4 \times 10^{-5}$; ~3 hours on 1× H200 GPU.

**Inference chunking:** 1,024-token max chunk size (4× larger than training maximum of 256 tokens).

### Results

- D2L achieves **near-perfect retrieval accuracy** up to ~40K tokens (40 chunks), which is 5× the chunk count seen during training and >4× the base model's 8K context window.
- Beyond 40K tokens performance degrades gracefully; the base model with ICL degrades sharply at 8K.
- **Memory advantage:** The base model requires >12 GB additional VRAM for a 128K-token haystack; D2L consistently uses <50 MB regardless of context length, because the LoRA adapter size is constant.

---

## 4. Main Experiments: Question Answering

### 4.1. Experimental Setup

- **D2L config:** 8 cross-attention blocks; 8K token chunks; rank-8 LoRA on MLP "down projection" layers; 309M trainable parameters; two modes: *batched* (speed) and *iterative* (memory).
- **Baselines (in-parameter):** CD (oracle query), CD (generated queries, 5 and 25 queries), T2L.
- **Baselines (in-context):** ICL (upper bound), LLMLingua-2 (prompt compression).
- **Metric:** Word-level ROUGE-L F1 (normalized relative to base model with full ICL).

### 4.2. Training Data

- **Source:** FineWeb-Edu (~900M tokens) + passage-grounded QA datasets (PwC, SQuAD, ROPES, DROP).
- **After filtering:** ~3.2M unique contexts.
- **Query generation:** 10 context-grounded queries per sample via `gemma-3-12b-it`.
- **Response sampling:** single response from target model `gemma-2-2b-it`; top-16 logits saved as training targets.
- **Context length stats:** mean 277 tokens, max 2,344 tokens.
- **Total triplets:** ~101M context-query-response pairs.

### 4.3. Datasets

| Dataset | Type | Notes |
|---------|------|-------|
| SQuAD (Rajpurkar et al., 2016) | Reading comprehension | Short passages, factoid QA |
| DROP (Dua et al., 2019) | Reading comprehension | Requires discrete reasoning |
| ROPES (Lin et al., 2019) | Reading comprehension | Situational reasoning over effects |
| 2WikiMultihopQA (Ho et al., 2020) | Long-context multi-hop QA | LongBench; up to 32K tokens |
| MultiFieldQA | Long-context QA | LongBench; multi-domain |
| QASPER (Dasigi et al., 2021) | Long-context QA | Scientific paper QA |
| Imagenette (Howard, 2019) | Image classification | 10-class ImageNet subset; VLM transfer |

### 4.4. Reading Comprehension Results (Short Contexts)

- D2L outperforms all in-parameter baselines on SQuAD, DROP, and ROPES.
- D2L achieves **82.5% relative performance** vs. ICL upper bound on SQuAD.
- D2L is roughly equivalent to LLMLingua-2 compressing the context to 40% of original length—but D2L removes the context *entirely* at inference.
- **Internalization latency:** D2L <1 s (batched) vs. CD (oracle) ~40 s vs. vanilla CD >100 s.

### 4.5. Long-Context QA Results (Zero-Shot)

D2L's training contexts max at 2,344 tokens; test documents reach 32K tokens—a genuine zero-shot length generalization.

**2WikiMultihopQA (key numbers):**

| Method | Norm. Perf | Update Memory (GB) | Update Latency (s) |
|--------|-----------|-------------------|-------------------|
| CD (oracle) | 0.901 | 7.82 | 40.2 |
| D2L (batched) | **0.857** | 11.52 | **0.21** |
| D2L (iterative) | 0.844 | **3.79** | 0.55 |
| CD (25 generated queries) | 0.745 | 59.9 | 465 |
| CD (5 generated queries) | 0.704 | 79.4 | 72.5 |

D2L is within 5% of oracle CD but ~190× faster and uses 15–21× less memory than CD with generated queries.

> [!IMPORTANT]
> D2L's update memory includes the KV-cache of the target LLM used to encode the context. In iterative mode this is avoided by processing chunks one-at-a-time, dropping to 3.79 GB—competitive with oracle CD (7.82 GB) and far below multi-query CD (40–80 GB).

### 4.6. VLM → LLM Zero-Shot Visual Transfer

D2L is applied cross-modally: context activations come from a VLM (`gemma-3-4b-it`) encoding an image, and the target is a text-only LLM (`gemma-2-2b-it`). No non-text training data is used.

- Imagenette classification: **75.03% accuracy** (10-class) via internalized visual information alone.
- NLP tasks (SQuAD, DROP, ROPES) degrade modestly (~10–15%) compared to the LLM→LLM baseline, showing that cross-architecture transfer is feasible but not lossless.

---

## 5. Analyses

### 5.1. D2L vs. CD at Varying Query Budgets

On SQuAD (100 samples), D2L achieves normalized performance of **0.866**, substantially closer to oracle CD (0.988) than CD with 100 generated queries (0.650), while being ~7,000× faster (0.086 s vs. 631 s).

### 5.2. Training Objective: KL vs. NTP

KL distillation matches the full teacher token distribution and outperforms next-token prediction (NTP) loss (0.819 vs. 0.763 normalized F1 on SQuAD at 50% training). The soft labels from the teacher carry richer information than one-hot targets.

### 5.3. LoRA Rank Ablation

Increasing rank from 8 to 16 improves SQuAD from 0.814 → 0.896 and DROP from 0.655 → 0.711, confirming that higher adapter capacity helps. Main experiments use rank-8 for memory efficiency.

### 5.4. Knowledge Interference

When queries are unrelated to the internalized document, D2L performance drops significantly (SQuAD normalized: 0.201 base → 0.096 D2L), indicating the adapter introduces a strong topical prior. CD shows less interference (0.211 → 0.203). Mitigation: including irrelevant queries in training data.

---

## 6. Experiments Summary

- **Hardware:** Single H200 GPU for evaluation; 8× H200 for meta-training (~5 days for `gemma-2-2b-it`).
- **Optimizer:** Standard gradient descent for meta-training; lr $4 \times 10^{-5}$ for NIAH; batch packing with 4K-token sequences and gradient accumulation.
- **Datasets:** SQuAD, DROP, ROPES (reading comprehension); 2WikiMultihopQA, MultiFieldQA, QASPER (long-context QA); Imagenette (vision transfer); NIAH (synthetic).
- **Key result:** D2L matches or exceeds conventional CD across all benchmarks while reducing internalization latency from minutes to <1 second and peak memory by 10–20×.

---

## 7. Limitations

- Meta-training is expensive (~5 days on 8× H200 GPUs for a 2B-parameter target LLM).
- The hypernetwork must be retrained for each new target LLM architecture.
- A performance gap remains between D2L and the ICL upper bound (context directly in window).
- The method is evaluated only with LoRA parameterization; other adapter forms are not explored.
- Knowledge interference occurs when the internalized document is unrelated to the query.
