# Meta Information

- URL: [Recurrent Preference Memory for Efficient Long-Sequence Generative Recommendation](https://arxiv.org/abs/2602.11605)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yixiao Chen, Yuan Wang, Yue Liu, Qiyao Wang, Ke Cheng, Xin Xu, Juntong Yan, Shuojin Yang, Menghao Guo, Jun Zhang, Huan Yu, Jie Jiang (2026). Recurrent Preference Memory for Efficient Long-Sequence Generative Recommendation. arXiv:2602.11605 [cs.IR].

---

# Recurrent Preference Memory for Efficient Long-Sequence Generative Recommendation (Rec2PM)

## Background and Problem

Generative recommendation systems model user behavior as autoregressive sequences, predicting the next item $I_n$ from the full interaction history $S = (I_0, I_1, \ldots, I_{n-1})$:

$$P(I_n \mid S) = P(I_n \mid I_0, I_1, \ldots, I_{n-1}; \theta)$$

The autoregressive training objective is:

$$\mathcal{L}_{AR} = -\sum_j \sum_i \log P(I_{j,i} \mid I_{j,0:i-1}; \theta)$$

As users accumulate thousands of interactions over their lifetime, two bottlenecks emerge:

1. **Computational cost**: Full-sequence attention is $O(L^2)$ in sequence length $L$, making inference on millions of users with $L > 1000$ prohibitively slow.
2. **Storage cost**: KV-cache approaches store per-layer key-value pairs for the full history, consuming tens of kilobytes per user.

Existing approaches either truncate history (losing long-term preferences) or retain full KV caches (prohibitive storage/latency). Rec2PM addresses both by compressing the archived history into a small set of dense **preference memory tokens**.

---

## Tripartite Memory Architecture

Rec2PM partitions user context into three types of memory with distinct roles:

| Memory Type | Content | Storage |
|---|---|---|
| **Working Memory** ($S_k$) | Recent raw interactions (current segment, length $L_{seg}$) | Raw item embeddings |
| **Preference Memory** ($M_{k-1}$) | Compressed representation of all prior segments | $C$ token embeddings $\in \mathbb{R}^{C \times d}$ |
| **Parametric Memory** | Shared model weights $\theta$ | Model parameters |

- $C$: number of memory slots (default $C = 4$)
- $d$: embedding dimension ($d = 64$)
- $L_{seg}$: segment length (default 200 interactions)

The **unified input** to the transformer at segment $k$ is:

$$E_{\text{unified},k} = [M_{k-1}; S_k; Q_{\text{mem}}]$$

where $Q_{\text{mem}} \in \mathbb{R}^{C \times d}$ is a learnable query matrix appended to trigger memory generation. The encoder outputs:

$$H = \text{Encoder}([E_{\text{encode}}; Q_{\text{mem}}])$$

$$m = H_{|E_{\text{encode}}|+1 : |E_{\text{encode}}|+C}$$

The updated memory $m$ is the slice of output tokens corresponding to the query positions — a cross-attention-style extraction within the same forward pass.

---

## Self-Referential Teacher Forcing

Standard recurrent training (BPTT through segments) is serial and slow. Rec2PM enables **parallel training** with a two-stage procedure:

### Stage 1: Global Reference Memory Generation

A global input concatenates all segments with query tokens interleaved:

$$E_{\text{global}} = [S_0; Q_{\text{mem}}; S_1; Q_{\text{mem}}; \ldots; S_k; Q_{\text{mem}}]$$

A **custom causal attention mask** prevents any $Q_{\text{mem}}$ token from attending to subsequent $Q_{\text{mem}}$ tokens, while allowing each $Q_{\text{mem}}^{(h)}$ to attend to all raw interactions up to segment $h$. This produces reference memories:

$$m_h^{(\text{ref})} \in \mathbb{R}^{C \times d}, \quad h = 0, 1, \ldots, k$$

These reference memories encode the full prior history at each segment boundary in a single forward pass.

### Stage 2: Parallel Recurrent Update

For each segment $h$ independently (all in parallel):

$$E_{\text{local},h} = [M_{h-1}^{(\text{ref})}; S_h; Q_{\text{mem}}]$$

- Tokens in $S_h$ predict next items → supervised by $\mathcal{L}_{AR}$
- $Q_{\text{mem}}$ outputs updated memory $m_h^{(\text{upd})}$ → supervised by consistency loss

The consistency loss enforces the recurrent memory to match the globally-informed reference:

$$\mathcal{L}_{con} = \frac{1}{k} \sum_{h=1}^{k} \| m_h^{(\text{ref})} - m_h^{(\text{upd})} \|^2$$

**Combined objective:**

$$\mathcal{L} = \mathcal{L}_{AR} + \lambda \mathcal{L}_{con}, \quad \lambda = 1$$

> [!IMPORTANT]
> The consistency loss decouples training steps and prevents the "drift" phenomenon seen in RNN training, where errors accumulate across recurrent steps. By providing globally-informed supervision at every step, gradients remain accurate even for early segments.

---

## Inference Modes

At inference, two update strategies are supported without any additional fine-tuning:

**Overwriting (Rec2PM-O)**: Memory is replaced at each step.

$$M_k = m_k$$

Storage: $C \times d$ floats per user = $4 \times 64 \times 4$ bytes = **1 KB per user** (constant).

**Appending (Rec2PM-A)**: Memory grows by accumulating tokens.

$$M_k = [M_{k-1}; m_k]$$

Storage grows linearly with number of segments.

> [!NOTE]
> Overwriting outperforms appending empirically. The authors attribute this to the bottleneck effect: forcing all history into $C$ slots requires the model to distill genuine long-term preferences rather than memorizing recent noise.

Both modes work because the training Stage 2 already exposes the model to the overwriting pattern ($M_{h-1}^{(\text{ref})}$ is always a fixed-size memory).

---

## Comparison with Related Methods

| Method | Training | Memory Format | Storage | Inference Latency |
|---|---|---|---|---|
| **Tok-Serial-O/A** | Serial BPTT | Token embeddings | 1 KB | Low |
| **KV-Mask-O/A** (PersRec) | Mask-parallel | Per-layer KV caches | 32 KB | Low |
| **Rec2PM-O/A** | Parallel teacher-forcing | Token embeddings | 1 KB | ~10 ms |
| **Full-sequence (HSTU-Full)** | Standard | None (full KV at inference) | — | ~135 ms |

**Key differences from prior work:**

- **vs. RMT (Bulatov et al. 2022)**: RMT uses recurrent tokens but trains serially (BPTT across segments). Rec2PM achieves parallel training via the self-referential teacher forcing strategy, avoiding gradient truncation.
- **vs. ICAE (Ge et al. 2023)**: ICAE compresses text with a separate encoder and fixed prompts; Rec2PM uses the same transformer backbone for both prediction and memory update in one forward pass.
- **vs. Gist (Mu et al. 2023)**: Gist compresses prompts via attention masks in a single pass with no recurrent update; Rec2PM supports iterative multi-segment updates.
- **vs. KuaiFormer (Liu et al. 2024)**: Also uses token embedding compression for recommendation, but does not present a teacher-forcing parallel training scheme.
- **vs. KV-Mask-O (PersRec, Zhang et al. 2026)**: Stores KV caches (32× more storage), requires no special training trick since KV state is exact; Rec2PM achieves similar accuracy at 1/32 the storage.

---

## Information Bottleneck Interpretation

Compressing $L_{seg}$-length segments into $C \ll L_{seg}$ token slots creates an **information bottleneck**: the model must discard noisy, session-specific interactions and retain only preference-relevant signals. This parallels the IB objective of minimizing $I(S; M)$ while maximizing $I(M; \text{future items})$.

Empirically, this manifests as:
- Memory tokens cluster semantically by category (Appendix visualizations)
- Memory tokens exhibit temporal specialization (different slots capture different time horizons)
- Overwriting ($C = 4$ constant) outperforms appending (no forced compression)

---

## Applicability

Rec2PM targets large-scale industrial recommender systems where:
- **Users** have thousands of historical interactions (e.g., e-commerce, streaming platforms)
- **Inference latency** must remain low at hundreds of millions of QPS
- **Storage** per-user must be bounded (cannot afford full KV caches at scale)
- **Accuracy** must match full-sequence methods

It is **not** suited for scenarios where the full history can fit in context within latency/memory budgets, as full-attention naturally outperforms compressed memory when resources allow.

---

# Experiments

## Datasets

**MerRec (Public Benchmark):**
- Filtered to users with $\geq 1003$ interactions
- Sequence length per user: 1003 items (truncated)
- Evaluation: leave-one-out (validation = 2nd-to-last item, test = last item)
- Exact user/item counts not reported in public paper

**Industrial Dataset (Proprietary — Kuaishou):**
- Users: ~500 million
- Items: >550 million
- Interactions: >500 billion
- Training period: 2025/04/01 – 2025/05/16 (46 days)
- Test period: 2025/05/17 (1 day)
- Average sequence length: 1,147 items; maximum: 2,048 items
- Data size: ~200 TB

## Hardware & Optimizer

- **Backbone models**: SASRec (4 layers, 4 heads) and HSTU (16 layers, 8 heads), both with $d = 64$
- **Optimizer**: Not named explicitly; learning rate $10^{-3}$, weight decay $0.1$, batch size 8
- **Early stopping**: patience = 10 epochs; 5 runs with seeds 0–4 (MerRec)

## Results

**Efficiency (HSTU backbone, MerRec):**
- Rec2PM-O: 1 KB/user storage, ~10 ms inference latency
- KV-Mask-O: 32 KB/user storage, ~10 ms inference latency
- HSTU-Full: no stored memory, ~135 ms inference latency

**Accuracy (MerRec, HSTU):**
- Rec2PM-O: H@1=15.04, H@10=44.20, H@50=61.23, N@10=28.66, N@50=32.48
- Outperforms Tok-Serial-O and KV-Mask-O on most metrics

**Industrial Dataset:**
- Rec2PM achieves **107% of HSTU-Full's H@1 accuracy** at **8% of its inference latency**
- Performance is robust to temporal overlap between archived history and recent context (only 0.7% drop when segments overlap with recent 100 interactions)

**Ablation — Consistency Loss:**
- Without $\mathcal{L}_{con}$: H@1 drops from 15.04 → 14.43, H@10 from 44.20 → 43.38

**Ablation — Memory Slots $C$:**
- Stable performance across $C \in \{1, 2, 4, 8, 16\}$; $C = 4$ optimal

**Inference Mode — One-time vs. Iterative:**
- Models trained with iterative updates can perform one-time full-history compression with no fine-tuning and negligible performance difference (Table 2)
