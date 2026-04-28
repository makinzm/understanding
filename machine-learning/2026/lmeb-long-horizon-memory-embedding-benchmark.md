# Meta Information

- URL: [LMEB: Long-horizon Memory Embedding Benchmark](https://arxiv.org/abs/2603.12572)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhao, X., Hu, X., Xu, J., Tang, D., Zhang, X., Zhou, M., Zhong, Y., Zhou, Y., Shan, Z., Zhang, M., Hu, B., & Zhang, M. (2026). LMEB: Long-horizon Memory Embedding Benchmark. arXiv:2603.12572.

# LMEB: Long-horizon Memory Embedding Benchmark

## Overview

LMEB (Long-horizon Memory Embedding Benchmark) is a standardized evaluation framework designed to assess text embedding models on long-horizon memory retrieval tasks. Unlike traditional benchmarks such as MTEB, which focus on passage retrieval, LMEB targets the specific challenges that arise in memory-augmented systems: fragmented information, context-dependent queries, and temporally distant references. The benchmark is intended for researchers and practitioners building embedding models for agents, RAG systems, and dialogue systems where multi-hop or temporally distant memory recall is required.

The benchmark comprises 22 datasets, 193 zero-shot retrieval tasks, a total corpus of 1.69M documents, 193K+ queries, and 387K+ relevance judgments. Fifteen embedding models ranging from approximately 239M to 12B parameters are evaluated.

## Background: Why Traditional Benchmarks Fall Short

MTEB and similar benchmarks evaluate embeddings primarily on single-hop passage retrieval: given a query, retrieve the most relevant document from a corpus. This setup does not capture several properties that are critical for memory-augmented LLM applications:

- **Fragmentation**: Relevant information may be spread across multiple segments of a long document or conversation history.
- **Context dependence**: Queries implicitly reference prior turns, entities, or events that must be resolved from context.
- **Temporal distance**: Key facts may be separated from their reference point by many intervening turns or passages.

LMEB addresses all three by constructing tasks from sources that naturally exhibit these properties.

## Memory Taxonomy

LMEB organizes retrieval tasks along two cognitive dimensions: **abstraction level** (how concrete vs. general the stored knowledge is) and **temporal dependency** (how much the retrieval depends on temporal context). This yields four memory categories:

| Category | Description | Datasets | Tasks |
|---|---|---|---|
| Episodic | Retrieval of past events tied to temporal cues, entities, and spatial context | 2 | 69 |
| Dialogue | Multi-turn conversational memory; recalling prior turns and user preferences | 6 | 42 |
| Semantic | General, context-independent knowledge and facts | 8 | 15 |
| Procedural | Learned skills and action sequences for multi-step reasoning | 6 | 67 |

### Episodic Memory

Episodic memory tasks require a model to retrieve descriptions of specific past events given a query that references temporal markers (e.g., "What happened after the charity race?"). The two episodic datasets in LMEB include EPBench (MIT) and LongMemEval (MIT).

### Dialogue Memory

Dialogue tasks test whether an embedding model can retrieve the correct prior utterance or fact from a multi-turn conversation history. The six dialogue datasets include LoCoMo (CC BY-NC 4.0), ConvoMem (CC BY-NC 4.0), MemBench (MIT), MemGovern (MIT), KnowMeBench (Apache 2.0), and ReMe (Apache 2.0).

### Semantic Memory

Semantic tasks evaluate retrieval of general knowledge independent of temporal or situational context, similar to open-domain QA. The eight semantic datasets include MLDR (MIT), QASPER (CC BY-NC 4.0), SciFact (CC BY-NC 2.0), PeerQA (CC BY-NC-SA 4.0), LooGLE (CC BY-SA 4.0), NovelQA (Copyright), ESG-Reports (Copyright), and Covid-QA (Apache 2.0).

### Procedural Memory

Procedural tasks test retrieval of instructions, tool documentation, or action sequences needed for step-by-step reasoning. The six procedural datasets include Gorilla (Apache 2.0), ToolBench (Apache 2.0), Proced\_mem\_bench (Apache 2.0), DeepPlanning (Apache 2.0), REALTALK (Unspecified), and TMD (Unspecified).

## Evaluation Setup

### Task Format

Each task follows the standard information retrieval (IR) format:

- **Queries**: A set of natural-language questions or prompts, $q \in Q$
- **Corpus**: A collection of candidate documents, $d \in D$ (corpus size varies by task)
- **Qrels**: Relevance judgments mapping $(q, d)$ pairs to binary or graded relevance labels
- **Candidates**: An optional pre-filtered candidate set for efficiency

### Metrics

Two retrieval metrics are computed for each task:

- **NDCG@10** (Normalized Discounted Cumulative Gain at cutoff 10): measures ranking quality, rewarding models that place relevant documents at higher positions in the top-10 results.
- **Recall@10**: measures what fraction of relevant documents appear in the top-10 results.

Macro-averages are computed across all 193 tasks to produce a single LMEB score.

### Evaluation Protocol

All models are evaluated in a **zero-shot** setting — no task-specific fine-tuning is permitted. Two conditions are evaluated:

1. **With task instructions**: A short natural-language instruction describing the retrieval task is prepended to each query.
2. **Without task instructions**: Queries are encoded as-is.

This design tests whether task-aware encoding (as in instruction-following embedding models) provides consistent gains across memory retrieval tasks.

### Data Construction

The benchmark uses a mix of AI-generated and human-annotated data. Corpus documents range from short utterances (dialogue) to long passages (episodic narratives). The HuggingFace default subset contains 840 rows with fields:

- `id` (string): task-scoped question identifier (e.g., `scene_0_q_82`)
- `text` (string): query text (e.g., "What did the charity race raise awareness for?")

## Experimental Results

### Key Findings

**Finding 1 — Appropriate difficulty**: The best-performing model achieves 61.41 NDCG@10, indicating that no model trivially solves LMEB and that meaningful headroom exists for future improvement.

**Finding 2 — Size paradox**: Larger embedding models (up to ~12B parameters) do not consistently outperform smaller ones (~239M parameters) on LMEB tasks. Model size alone is an insufficient predictor of memory retrieval quality, suggesting that architecture and training data composition matter more than raw capacity for these tasks.

**Finding 3 — Variable instruction sensitivity**: The performance gap between the with-instruction and without-instruction conditions varies substantially across model families, indicating that different models have been trained with different assumptions about task-prompt formatting.

**Finding 4 — Orthogonality with MTEB**: LMEB scores are essentially uncorrelated with MTEB scores, confirming that LMEB measures a distinct set of capabilities:

| Metric | Value |
|---|---|
| Pearson correlation (LMEB vs. MTEB) | -0.115 |
| Spearman correlation (LMEB vs. MTEB) | -0.130 |
| Pearson correlation (LMEB dialogue vs. MTEB) | -0.496 |

The near-zero and slightly negative Pearson/Spearman values mean that optimizing for MTEB does not translate to LMEB performance. The strongly negative correlation for dialogue tasks specifically ($r = -0.496$) reveals that embeddings trained on standard passage retrieval are poorly suited for conversational memory retrieval — a significant practical implication for system designers selecting embedding models for dialogue agents.

## Comparison with Related Benchmarks

| Property | MTEB | LMEB |
|---|---|---|
| Primary task type | Passage retrieval, classification, clustering | Long-horizon memory retrieval |
| Information structure | Single coherent passages | Fragmented, temporally distributed |
| Context dependency | Low | High (prior turns, temporal cues) |
| Memory categories | None | Episodic, Dialogue, Semantic, Procedural |
| Zero-shot evaluation | Yes | Yes |
| Correlation with each other | — | Pearson $r = -0.115$ (orthogonal) |

> [!IMPORTANT]
> The near-zero correlation between LMEB and MTEB means that practitioners selecting embedding models based solely on MTEB leaderboard rankings will likely make suboptimal choices for memory-augmented applications. LMEB should be used alongside MTEB for a complete evaluation.

> [!TIP]
> The benchmark dataset is available at [KaLM-Embedding/LMEB on HuggingFace](https://huggingface.co/datasets/KaLM-Embedding/LMEB). Evaluation scripts and instructions are on [GitHub](https://github.com/KaLM-Embedding/LMEB).

## Applicability

LMEB is most relevant for:

- **Researchers** developing embedding models intended for use in LLM agent pipelines, RAG systems, or dialogue managers where queries reference past context.
- **Practitioners** selecting embedding models for production memory-augmented systems, where MTEB alone is an insufficient guide.
- **Benchmark designers** seeking a complementary evaluation axis to passage-retrieval benchmarks.

# Experiments

- **Datasets**: 22 datasets across four memory categories (see Memory Taxonomy section); total 1.69M documents, 193K+ queries, 387K+ relevance judgments
  - Episodic: EPBench (MIT), LongMemEval (MIT)
  - Dialogue: LoCoMo (CC BY-NC 4.0), ConvoMem (CC BY-NC 4.0), MemBench (MIT), MemGovern (MIT), KnowMeBench (Apache 2.0), ReMe (Apache 2.0)
  - Semantic: MLDR (MIT), QASPER (CC BY-NC 4.0), SciFact (CC BY-NC 2.0), PeerQA (CC BY-NC-SA 4.0), LooGLE (CC BY-SA 4.0), NovelQA (Copyright), ESG-Reports (Copyright), Covid-QA (Apache 2.0)
  - Procedural: Gorilla (Apache 2.0), ToolBench (Apache 2.0), Proced\_mem\_bench (Apache 2.0), DeepPlanning (Apache 2.0), REALTALK (Unspecified), TMD (Unspecified)
- **Models**: 15 embedding models, ~239M to ~12B parameters; includes NV-Embed-v2 as a representative large model
- **Evaluation**: Zero-shot; two conditions (with/without task instructions); metrics NDCG@10 and Recall@10
- **Results**: Best model NDCG@10 = 61.41; Pearson correlation with MTEB = -0.115; dialogue-specific Pearson with MTEB = -0.496
