# Meta Information

- URL: [Exploring Distributed Vector Databases Performance on HPC Platforms: A Study with Qdrant](https://arxiv.org/abs/2509.12384)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Ockerman, S. et al. (2025). Exploring Distributed Vector Databases Performance on HPC Platforms: A Study with Qdrant. arXiv:2509.12384.

# Distributed Vector Databases on HPC Platforms: A Study with Qdrant

## Overview

This paper characterizes the performance of Qdrant, a distributed vector database, on the Polaris supercomputer (Argonne National Laboratory). It evaluates insertion, indexing, and query operations at scale with up to 32 workers across 8 compute nodes, using a scientific text workload derived from the peS2o academic corpus and biological query terms from BV-BRC.

> [!NOTE]
> "To the best of our knowledge no studies have focused on understanding or optimizing vector database performance in the context of scientific workloads and HPC systems."

The target users are HPC practitioners and ML engineers who need to deploy retrieval-augmented generation (RAG) pipelines or similarity search systems over large scientific datasets on supercomputers.

## Background: Vector Databases

Vector databases store high-dimensional embedding vectors and enable approximate nearest neighbor (ANN) search using specialized index structures. The primary index type is the Hierarchical Navigable Small World (HNSW) graph, which supports $O(\log n)$ approximate nearest neighbor queries at the cost of $O(n \log n)$ index construction time.

Supported distance metrics include cosine similarity, Euclidean distance $\|x - y\|_2$, and inner product $x \cdot y$.

### Distributed Architectures

Two main architectural patterns exist for distributed vector databases:

| Architecture | Description | Examples |
|---|---|---|
| Stateful (shared-nothing) | Each worker stores data shards and local indexes | Qdrant, Vald, Weaviate |
| Stateless with storage separation | Compute and storage are decoupled | Vespa, Milvus |

Qdrant follows the stateful model: data is sharded across workers, and each worker independently maintains its shard's HNSW index.

### Query Execution in Qdrant (Distributed)

1. Client submits a query vector to a single designated worker.
2. That worker broadcasts the query to all other workers.
3. Each worker searches its local shards independently.
4. Partial top-$k$ results are returned to the initial worker.
5. The initial worker merges all partial results and returns the final top-$k$ to the client.

This fan-out/fan-in pattern means query latency is bounded by inter-worker communication, not just local search time.

### Comparison of Distributed Vector Databases

| System | Parallel Read/Write | Compute-Storage Separation | GPU Indexing | GPU ANN Search |
|---|---|---|---|---|
| Qdrant | Yes | No | No | No |
| Milvus | Yes | Yes | Yes | Yes |
| Weaviate | Yes | No | No | No |
| Vespa | Yes | Yes | No | No |
| Vald | Yes | No | No | No |

> [!TIP]
> HNSW index construction is currently CPU-only in Qdrant; GPU-accelerated alternatives like FAISS-GPU or RAPIDS cuVS may be worth evaluating for large HPC deployments.

## System Architecture and Experimental Setup

### Hardware: Polaris Supercomputer

- CPU: AMD EPYC Milan 7543P, 2.8 GHz, 32 cores per node
- Memory: 512 GB DDR4 RAM per node
- GPU: 4 × NVIDIA A100 (40 GB each) per node
- Interconnect: HPE Slingshot 11, Dragonfly topology
- Deployment: up to 32 Qdrant workers, 4 workers per node, across 8 nodes

### Dataset

- **Corpus**: peS2o — 8,293,485 full academic papers
- **Embedding model**: Qwen3-Embedding-4B (dense, 4B parameter transformer)
- **Query workload**: 22,723 genome-related biological terms sourced from BV-BRC (Bacterial and Viral Bioinformatics Resource Center)
- **Total dataset size**: ~80 GB of embedding vectors
- **Per-worker shard size**: $\approx 80\,\text{GB} / N_{\text{workers}}$
- **Tuning subset**: 1 GB subset used for hyperparameter search

### Embedding Generation Pipeline

An orchestrator submits single-node batch jobs to Polaris's PBS scheduler. Within each job:

1. Papers are batched by character count (max 150,000 characters per batch, up to ~4,000 papers).
2. Multiprocessing distributes batches across 4 A100 GPUs using heuristic bin-packing.
3. Out-of-memory errors fall back to sequential single-paper processing.

> [!NOTE]
> Model inference accounts for 98.5% of total embedding generation runtime (mean 2,381.97 s per batch), with I/O at 7.49 s and model loading at 28.17 s.

## Experiments and Results

### 1. Data Insertion

**Tuning**: Batch size and concurrent upload requests were varied on the 1 GB subset.

- Optimal batch size: **32 vectors per request**
- Optimal concurrent requests: **2**

**Finding on concurrency**: Amdahl's Law constrains the speedup from concurrent uploads. The batch object conversion step (CPU-bound, 45.64 ms) dominates over the insertion RPC (14.86 ms), yielding a theoretical maximum speedup of only 1.31× from concurrency alone. Multiprocessing is more effective than asyncio for this workload.

**Scaling results on full 80 GB dataset**:

| Workers | Insertion Time |
|---|---|
| 1 | 8.22 hours |
| 4 | 2.11 hours |
| 8 | 1.14 hours |
| 16 | 35.92 minutes |
| 32 | 21.67 minutes |

Near-linear speedup is observed, since each worker independently handles its own shard insertions with minimal cross-worker coordination.

### 2. Index Construction

Index construction (HNSW) is deferred until after all data is inserted to accelerate the process. Each worker builds its shard's index independently using all available CPU cores.

**Scaling speedup**: Maximum speedup of **21.32×** with 32 workers vs. 1 worker. Single-worker CPU utilization reaches 90–97%, confirming this phase is CPU-saturated.

Sublinear scaling is due to:
- Inter-worker coordination overhead at startup/teardown
- Fixed per-node overhead independent of shard size

> [!IMPORTANT]
> Index construction is entirely CPU-bound in Qdrant. GPU-accelerated HNSW construction (e.g., via cuVS) could substantially reduce this bottleneck in future HPC deployments.

### 3. Query Performance

**Tuning**: Batch size and concurrency were varied on the 1 GB subset.

- Optimal batch size: **16 queries per request**
- Optimal concurrent batches: **2**
- Single-worker query time improved from 139 s (batch size 1) to 73 s (batch size 16).

**Scaling results on full 80 GB dataset**:

- Maximum speedup: **3.57×** with multiple workers (well below linear)
- Parallelization benefit emerges only when dataset exceeds ~30 GB per shard
- Beyond 4 workers, diminishing returns due to fan-out communication overhead

**Root cause**: Since each query fans out to all workers, adding more workers increases inter-worker communication linearly while reducing local search work. For smaller datasets that fit in memory, the communication cost dominates.

## Key Contributions

1. First empirical characterization of Qdrant's distributed performance on an HPC supercomputer (Polaris) with scientific text workloads.
2. Identification of per-phase bottlenecks: inference-bound embedding, serialization-bound insertion, CPU-bound indexing, and communication-bound querying.
3. Release of a scientific embedding dataset (peS2o + BV-BRC queries) and evaluation workload for future research.

## Related Work

Prior work falls into two categories:

- **Feature surveys** (Taipalus, 2024; Pan et al., 2023, 2024): Compared vector database systems at the feature level without empirical scaling evaluation.
- **Single-node RAG evaluation** (Shen et al., 2024): Evaluated retrieval quality on a single GPU, not distributed performance.
- **Distributed database design** (Xu et al., 2025): Proposed a new distributed design benchmarked against FAISS, but not evaluated against production systems in an HPC context.

None of these addressed distributed performance on HPC infrastructure with scientific workloads.

## Conclusion

Qdrant scales effectively for insertion (near-linear) and index construction (21.32× at 32 workers), but query scaling is limited to 3.57× due to inter-worker communication overhead in the fan-out query model. The authors recommend:

- **GPU-accelerated index construction** to eliminate the CPU bottleneck in HNSW building
- **Adaptive scaling techniques** that match the number of workers to dataset size
- **Dataset-aware concurrency tuning**: parallelization only yields benefit for shards larger than ~30 GB

These findings apply directly to teams deploying vector similarity search for RAG or scientific retrieval pipelines on HPC clusters where data volumes reach tens to hundreds of GB.

# Experiments

- Dataset: peS2o academic corpus (8,293,485 papers, ~80 GB embeddings); BV-BRC genome-related query terms (22,723 terms)
- Hardware: Polaris supercomputer — AMD EPYC Milan 7543P (32-core), 512 GB DDR4, 4 × NVIDIA A100 40 GB per node; HPE Slingshot 11 Dragonfly interconnect
- Embedding model: Qwen3-Embedding-4B
- Optimizer: N/A (inference and database evaluation, no training)
- Results:
  - Insertion: 8.22 h (1 worker) → 21.67 min (32 workers), near-linear speedup
  - Index construction: 21.32× speedup at 32 workers; CPU-saturated at 90–97% utilization
  - Query: 3.57× max speedup; limited by fan-out communication overhead; benefit only above ~30 GB shard size
