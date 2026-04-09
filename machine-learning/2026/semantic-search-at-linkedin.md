# Meta Information

- URL: [Semantic Search At LinkedIn](https://arxiv.org/abs/2602.07309)
- LICENSE: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Borisyuk, F., Zheng, R., Ma, X., Wang, Z., et al. (2026). Semantic Search At LinkedIn. arXiv:2602.07309.

# Semantic Search At LinkedIn

## Overview

LinkedIn's production semantic search system combines LLM-based relevance judgment, embedding-based retrieval, and a compact Small Language Model (SLM) reranker trained via multi-teacher distillation. The system enables retrieval by semantic meaning rather than keyword overlap. Through co-designed inference optimizations including model pruning, context summarization, and shared-prefix amortization, the system achieves over 75× throughput improvement under fixed latency constraints while preserving near-teacher-level NDCG.

**Applicability:** Large-scale industrial search systems (job search, people search) requiring both relevance and engagement optimization at production throughput. The framework generalizes to any dual-objective ranking problem where relevance and engagement signals come from different teacher models.

## System Architecture

### Two-Stage Pipeline

**Stage 1 — Retrieval:**

- Input: user query $q \in \mathbb{R}^{d_q}$ (text)
- GPU-accelerated exhaustive embedding-based retrieval with contrastive-trained bi-encoder
- Retrieval-as-Ranking (RAR) combines embedding similarity with learned feature scores
- Output: top-$K$ candidate set ($K = 1000$)

**Stage 2 — Ranking:**

- Input: query $q$ + top-$K$ candidates from Stage 1
- SLM reranks top 250 results
- Joint optimization of relevance and engagement scores via multi-teacher distillation
- Output: ranked list with calibrated probability scores per objective

### Query Understanding Layer

A deterministic preprocessing module converts short, ambiguous queries into routing decisions, normalized attribute signals, and query reformulations. This establishes a stable semantic contract for downstream retrieval and ranking components.

## Embedding-Based Retrieval

### Contrastive Training

The bi-encoder is trained with a combined loss:

$$\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{InfoNCE}} + (1 - \lambda) \cdot \mathcal{L}_{\text{pair}}$$

where:
- $\mathcal{L}_{\text{InfoNCE}}$ captures global semantic similarity across the batch (in-batch negatives)
- $\mathcal{L}_{\text{pair}}$ enforces pairwise margin constraints for local decision boundaries
- $\lambda \in [0, 1]$ balances global structure vs. local precision

**Training data construction:**
- 1–2 positives per query: documents with relevance label $> 2$
- 2–3 hard negatives per query: documents with label $\leq 2$, sampled via policy-bucketed hard-negative mining
- Sampling respects query diversity while targeting top-rank failure modes

### Retrieval-as-Ranking (RAR)

The GPU RAR model combines embedding similarity with hand-crafted features:

$$S(q, d) = w_0 \cdot \langle e_q, e_d \rangle + \sum_{i=1}^{n} w_i \cdot f_i(q, d)$$

where $e_q, e_d$ are query and document embeddings, $f_i$ includes network proximity and profile popularity signals, and $w_i$ are learned weights. Trained with a weighted multi-task objective balancing relevance and engagement.

**Retrieval Results:**

| Metric | Baseline | Full Pipeline |
|---|---|---|
| Job Search Precision@50 | 0.414 | 0.505 (+22%) |
| People Search Precision@10 | 0.33 | 0.47 (+42%) |
| People Search Click AUC (RAR) | — | +1.7% |

## SLM Ranking

### Relevance Training via Oracle Distillation

An 8B "oracle" relevance model produces ordinal grades $g \in \{0, 1, 2, 3, 4\}$ as supervision. The SLM distills from this oracle via soft-label fine-tuning:

- **Binary soft-label format:** The oracle's binary Yes/No token probabilities $(p_{\text{yes}}, p_{\text{no}})$ are used as targets
- **Chat-template inference:** SLM uses a chat template; first-token logits for "Yes"/"No" tokens are extracted as the relevance score
- **Pairwise list-wise ranking loss** is applied jointly to enforce relative ordering

> [!NOTE]
> Soft-label fine-tuning with binary probabilities achieved +11.0% NDCG@10 over ordinal label training in Job Search. This suggests that distributional teacher uncertainty carries more information than discrete grades.

**Scaling observation:** Increasing training data from 200K to 8M query-document pairs (40×) improved performance; gains saturated beyond this point, indicating current architectural capacity rather than data scarcity.

### Multi-Task Engagement Teacher

Trained on user action logs to predict multiple engagement signals:
- **Job Search:** click, apply, dismiss, badfit, shortlist
- **People Search:** long-dwell, connect, follow, message

**Performance vs. DLRM baseline:**
- +4.4% Job Click AUROC
- Data engineering contribution: +1.48% AUROC
- Feature engineering contribution: +4.04% AUROC

### Multi-Teacher Distillation (MTD)

The student SLM distills simultaneously from a relevance teacher (large SLM) and an engagement teacher (multi-task model).

**Loss function:**

$$\mathcal{L}_{\text{MTD}} = \alpha \cdot D_{\text{KL}}(p_{\text{rel}} \| q_{\text{rel}}) + (1 - \alpha) \cdot D_{\text{KL}}(p_{\text{eng}} \| q_{\text{eng}})$$

where $p$ denotes teacher distributions and $q$ denotes student predictions. Weighted KL-divergence preserves teacher uncertainty.

**Training procedure:**
1. Train relevance teacher on oracle-labeled data
2. Train engagement teacher on action logs
3. Warm-start student from relevance-specialized SLM
4. Fine-tune student via weighted KL losses from both teachers

> [!NOTE]
> Warm-start initialization from the relevance specialist (step 3) outperformed open-source model initialization by +0.68% NDCG@10 and +1.98% Click AUROC.

### Loss Masking for Imbalanced Engagement Actions

For sparse engagement actions (Follow, Message), standard negative sampling treats all non-interacted documents as negatives, producing noisy gradients. The MTD framework masks training such that only documents surfaced alongside positive instances are treated as negatives.

**Effect:** Predicted scores for rare events increased 5× without degrading ranking metrics.

### Calibration

Post-hoc calibration uses isotonic regression with position-conditional probabilities:
- Separate probability vector per rank position (accounts for position bias)
- Independent training per objective (modular multi-head design)
- Job Search Click AUROC improved from 0.6704 → 0.7095

### Feature Engineering

**Member Profile Summarization:**
- A 1.7B RL-based summarizer condenses job descriptions
- Reward: prediction accuracy + length penalty + quality score
- Reduces document length by an order of magnitude with negligible quality loss

**Numerical Feature Encoding Ablation:**

| Encoding | AUC Improvement |
|---|---|
| Descriptive identifiers | +5.8% |
| CTR features | +5.1% |
| Boolean True/False | +1.7% |
| Decimal truncation (2 places) | minimal |

## Inference Optimizations

### Scoring-Optimized Prefill Execution

Ranking only requires a single forward pass returning final-token logits — no sampling or decoding is needed.

**Optimizations:**
- Disable per-token log-probability computation
- Bypass decoding kernel
- Consolidate device-host memory transfers

**Result:** +44% throughput (900 → 1300 items/second per GPU)

### Shared-Prefix Amortization

Ranking prompts for a given query share a long query prefix $q$ across all $N$ candidate documents $d_i$.

**Naive prefill complexity:**
$$F_{\text{naive}} \propto N \cdot (T_q + T_i)^2$$

**Amortized prefill complexity:**
$$F_{\text{amortized}} \propto T_q^2 + N \cdot (2 T_i T_q + T_i^2)$$

where $T_q$ is query token length and $T_i$ is candidate token length. Two implementations:
1. **In-batch prefix caching (IBPC):** Single KV computation for query prefix, shared across the batch
2. **Multi-item scoring:** Concatenated items with attention masking

**Result:** +25% throughput (1600 → 2000 items/second per GPU); additional +10% from CUDA graph execution.

### Context Summarization (Offline)

- 1.7B LLM compresses job descriptions offline; summaries stored for serving
- P95 prompt length: ~1500 → ~500 tokens
- **Result:** 4× throughput improvement in Job Search ranking

### Structured Model Pruning

OSSCAR-based structured pruning:
- 50% MLP neuron removal per transformer block
- Remove final 8 transformer layers
- Parameters: 600M → 375M

**Result:** Pruned model matched or exceeded dense baseline in People Search (NDCG@10: 0.8652 vs. 0.8629)

### MixLM: Text–Embedding Hybrid Architecture

Replaces long document text with cached learned embedding tokens:

- **Document encoder:** Compresses document into small embedding set $\{z_1, \ldots, z_m\}$ (cached nearline)
- **Ranker input:** Query text tokens + document embedding tokens $z_i$
- **Training:** Multi-stage distillation from text-based teacher

**Effect:** ~76× throughput improvement over raw-text baseline at equivalent latency budget.

### Cumulative Inference Speedup

| Model Variant | NDCG@10 | QPS (items/s/GPU) |
|---|---|---|
| Full Text (baseline) | 0.9432 | 290 |
| Summarized + Pruned | 0.9218 | 2,200 |
| MixLM | 0.9239 | 22,000 |

### CPU and Runtime Optimizations

| Optimization | Throughput Gain |
|---|---|
| Batch tokenization + sending | +20% |
| Multi-process gRPC (bypass Python GIL) | — |
| Scheduler/runtime tuning | +23% |
| Python heap freezing (`gc.freeze()`) | tail latency |

**Cumulative CPU speedup:** 2.93× over baseline.

### Middle-Tier Serving

- **Score caching:** Distributed Couchbase store keyed by (searcher, query); >50% of requests served from cache, reducing latency 8–10%
- **Adaptive depth (PID controller):** Dynamically adjusts ranking depth from 250 → 130 during peak load; reduces per-query GPU compute by 48%
- **Traffic shaping:** Defers latency-insensitive requests to idle windows; +25% throughput

## Training Infrastructure

### Asynchronous Multi-Teacher Distillation (SGLang-based)

**Online mode:** Asynchronous client queries teacher models during training; 3× speedup over synchronous distillation.

**Offline mode:** Teacher outputs precomputed and stored on HDFS/NFS.
- ~35% reduction in training time vs. online mode
- ~25% GPU-hours saved despite one-time precomputation cost

### Distributed Training Optimizations

| Technique | Speedup |
|---|---|
| LiGer fused kernels (2× larger batch) | memory reduction |
| Multi-node training | up to 3.5× |
| FSDP2 | +20% |
| H200 clusters | up to 30% |
| FP8 mixed precision | no benefit (<8B models) |

### Agentic GPU Optimization

An LLM-guided agent analyzes training code and metrics:
- BFS traversal + RAG to identify relevant workflow code
- Applies configuration tuning (gradient checkpointing, FSDP settings)
- Applied to MixLM training: 13% reduction in training time (256 GPU-hours saved on 64 H100s)

## SAGE: Relevance Governance Framework

**SAGE (Scalable Assessment and Governance Engine)** operationalizes relevance policy by combining:
- Explicit product policy specifications
- Curated human-labeled precedent data
- LLM surrogate judges (8B student distilled from frontier LLM judges)
- Simulation-driven iteration

Achieves linear kappa of 0.77 vs. human precedent and 0.81 vs. teacher judge.

## Comparison with Similar Systems

| Aspect | Traditional DLRM | LinkedIn Semantic Search |
|---|---|---|
| Retrieval signal | Keyword / BM25 | Embedding + LLM contrastive |
| Ranking model | Dense retrieval + linear ranker | SLM (600M→375M params) |
| Supervision | Click logs | Oracle LLM + engagement logs |
| Throughput optimization | N/A | Prefix amortization, MixLM, pruning |
| Calibration | None | Isotonic regression (position-aware) |
| Training infrastructure | Standard | Async MTD + agentic GPU optimization |

> [!TIP]
> The SGLang-based scoring-optimized inference stack is open-sourced at https://github.com/sgl-project/sglang

> [!IMPORTANT]
> The 76× throughput improvement is achieved by combining offline context summarization (4×), model pruning (reduces tokens and params), prefix amortization (+25%), scoring-optimized prefill (+44%), and the MixLM embedding replacement. MixLM alone accounts for most of the final gain.

# Experiments

- **Datasets:** LinkedIn Job Search and People Search production logs; query-document pairs with human relevance labels (0–4 ordinal scale); user action logs (click, apply, dismiss, connect, follow, message); Job Search training: 200K–8M query-document pairs
- **Hardware:** H100 and H200 GPU clusters (multi-node); inference on GPU-accelerated serving infrastructure
- **Optimizer:** Standard gradient-based optimization; FSDP2 for distributed training; FP8 mixed precision tested but not used for <8B models
- **Results:**
  - Job Search: +7.73% NDCG@10, -46.88% Poor Match Rate@10, >+1.2% DAU lift vs. DLRM baseline
  - People Search: >10% NDCG@10 improvement vs. baseline
  - Retrieval: +22% Precision@50 (Job Search), +42% Precision@10 (People Search)
  - Throughput: 76× improvement (290 → 22,000 QPS per GPU) from full-text to MixLM
  - Engagement teacher: +4.4% Job Click AUROC vs. DLRM
  - Calibration: Click AUROC 0.6704 → 0.7095
