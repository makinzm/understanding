# Meta Information

- URL: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP '23).

# Background

## Autoregressive LLM Generation

LLMs generate tokens sequentially, factorizing the joint probability as:

```math
\begin{align}
  P(x_1, \ldots, x_n) = \prod_{i=1}^{n} P(x_i \mid x_1, \ldots, x_{i-1})
\end{align}
```

Each forward pass computes self-attention over all previously generated tokens. The attention score for query $q_i \in \mathbb{R}^d$ and key $k_j \in \mathbb{R}^d$ is:

```math
\begin{align}
  a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{t=1}^{i} \exp(q_i^\top k_t / \sqrt{d})}
\end{align}
```

and the output is $o_i = \sum_j a_{ij} v_j$ where $v_j \in \mathbb{R}^d$ is the value vector.

## KV Cache and the Memory Bottleneck

To avoid recomputing key and value projections for previous tokens, serving systems store the **KV cache**: a set of $(k_j, v_j)$ pairs for each transformer layer. For OPT-13B (40 layers, $d = 5120$), storing one token's KV cache across all layers requires approximately 800 KB, and a 2048-token sequence requires 1.6 GB.

Serving concurrently with large batches is the primary way to amortize the memory cost of loading model weights (which occupy roughly 65% of GPU memory). The KV cache for active requests occupies the remaining ~30%. However, the authors measured that existing systems utilize only **20.4%–38.2%** of allocated KV cache memory for actual token states; the rest is wasted on:

1. **Reserved future slots** — systems pre-allocate for the worst-case output length.
2. **Internal fragmentation** — over-provisioned blocks within a request.
3. **External fragmentation** — free memory scattered between non-contiguous allocations.

## Decoding Phases

| Phase | Description | Compute characteristic |
|---|---|---|
| Prompt phase | Processes all input tokens in parallel | Compute-bound |
| Decoding phase | Generates one token per step, appending to KV cache | Memory-bound |

Fine-grained **iteration-level scheduling** (from Orca) allows dynamic batching: different requests can join or leave a batch at each step, improving GPU utilization but increasing memory pressure because more concurrent KV caches must coexist.

# Method: PagedAttention and vLLM

## Core Idea: Virtual Memory for KV Cache

The authors draw a direct analogy to OS virtual memory:

| OS Virtual Memory | vLLM KV Cache |
|---|---|
| Page | KV block |
| Byte | Token |
| Process | LLM request |
| Virtual address | Logical block index |
| Physical address | Physical block index |
| Page table | Block table |
| Copy-on-write | Block sharing with reference counts |

Instead of requiring a contiguous physical memory region per request, vLLM decouples logical KV blocks from physical storage via block tables, enabling flexible allocation and deallocation.

## PagedAttention Algorithm

KV cache is divided into blocks of size $B$ tokens. The key and value matrices for the $j$-th block are $K_j \in \mathbb{R}^{B \times d}$ and $V_j \in \mathbb{R}^{B \times d}$, stored in non-contiguous physical GPU memory. The attention computation for token $i$ becomes:

```math
\begin{align}
  A_{ij} &= \frac{\exp\!\left(q_i^\top K_j^\top / \sqrt{d}\right)}{\sum_{t=1}^{\lceil i/B \rceil} \exp\!\left(q_i^\top K_t^\top / \sqrt{d}\right) \cdot \mathbf{1}} \\
  o_i &= \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top
\end{align}
```

where $q_i \in \mathbb{R}^d$, $K_j, V_j \in \mathbb{R}^{B \times d}$, and $A_{ij} \in \mathbb{R}^{B}$. The kernel accesses each physical block via the block table and fuses the block reads with the attention computation.

**Input/Output:**
- Input: query vector $q_i \in \mathbb{R}^d$, block table mapping logical block indices to physical block addresses, cached $K_j$ and $V_j$ for blocks $j = 1, \ldots, \lceil i/B \rceil$
- Output: attention output $o_i \in \mathbb{R}^d$

**Difference from standard attention:** Standard implementations assume a single contiguous $K \in \mathbb{R}^{i \times d}$ and $V \in \mathbb{R}^{i \times d}$. PagedAttention instead iterates over blocks referenced through the block table, incurring ~20–26% attention kernel overhead but eliminating external fragmentation and enabling block-level sharing.

## KV Cache Manager

The manager maintains three data structures:

- **Logical KV blocks** — a per-request view of tokens ordered left-to-right.
- **Physical KV blocks** — fixed-size slots in GPU memory (or CPU RAM for swapped blocks).
- **Block table** — per-request mapping from logical block index to physical block number and fill count.

Physical blocks are allocated on demand:
1. On request arrival, the first physical block is reserved.
2. When a block is full ($B$ tokens stored), a new physical block is allocated.
3. At most 1 block per request is partially filled at any time, limiting internal fragmentation to $< B$ tokens.

## Decoding Algorithm (Single Sequence)

```
procedure DECODE(request):
    prompt_tokens ← request.prompt
    allocate physical blocks for prompt_tokens
    fill KV cache from prompt phase forward pass

    while not done:
        t ← generate next token (autoregressive step)
        if current logical block is full:
            allocate new physical block
        append KV(t) to current logical block
        if t == EOS or len(output) == max_len:
            free all physical blocks for this request
```

Maximum waste per sequence: $< B$ tokens × (KV size per token).

## Advanced Decoding Scenarios

### Parallel Sampling

When $k$ output sequences are sampled from one prompt, all $k$ sequences share the prompt's physical KV blocks using **reference counting**. Each block's reference count is incremented when shared and decremented on release. When a sequence writes to a shared block (its first generation step), **copy-on-write** creates a private copy.

Memory saving: sharing the prompt blocks reduces memory per request by up to 55% when the prompt is long (e.g., for parallel sampling width $k = 6$, empirically 6.1%–30.5% savings).

### Beam Search

With beam width $w$, at each step the top-$w$ candidates are retained. Candidates share prompt and common-prefix blocks until they diverge. When a beam is pruned, its exclusive blocks are freed; shared blocks are decremented and freed only when reference count reaches zero.

Empirical memory saving: 37.6%–66.3% for beam width 6 vs. Orca (Oracle).

**Difference from prior systems:** Systems like FasterTransformer copy KV cache across beams eagerly. vLLM's copy-on-write avoids all copies until actual divergence.

### Shared Prefix (Prompt Caching)

For workloads with a fixed system prompt (e.g., few-shot examples), the prompt KV blocks are pre-filled and marked as reserved (reference count held at 1). All incoming requests map their first logical blocks to these shared physical blocks. This reduces redundant prefill computation.

Throughput gain: 1.67× (1-shot) to 3.58× (5-shot) for translation tasks.

## Scheduling and Preemption

**Policy:** First-come-first-serve (FCFS); newest requests are preempted first when memory is exhausted.

**Granularity:** Sequences within the same **sequence group** (e.g., parallel samples from the same prompt) are always scheduled together to maintain logical consistency.

**Recovery options:**

| Method | Mechanism | Cost |
|---|---|---|
| Swapping | Move KV blocks to CPU RAM via PCIe | Scales linearly with block size |
| Recomputation | Discard KV cache; rerun prefill on prompt + generated tokens | ~20% of swapping cost at block size 16 |

> [!NOTE]
> Recomputation is faster than expected because the concatenated prompt + previously generated tokens can be processed in a single parallel forward pass (same as the original prompt phase), rather than token-by-token autoregressive steps.

## Distributed Execution

For models requiring tensor parallelism across $N$ GPUs (e.g., OPT-175B on 8 GPUs):
- The central scheduler manages a single logical block table.
- Each GPU worker stores KV cache for a subset of attention heads: $K_j^{(r)}, V_j^{(r)} \in \mathbb{R}^{B \times (d/N)}$ for worker $r$.
- Workers receive control signals (input token IDs + block table) and synchronize via all-reduce at each transformer layer.

# Implementation

**Codebase:** 8,500 lines of Python, 2,000 lines of C++/CUDA.

**Supported models:** OPT (13B, 66B, 175B), LLaMA (13B).

**Custom CUDA kernels:**

1. **Fused reshape + block write** — writes new KV entries into paged physical blocks.
2. **Fused block read + attention** — reads scattered blocks via block table and computes attention with coalesced memory access.
3. **Fused block copy** — copies non-contiguous blocks for copy-on-write operations.

**Decoding API:** Three primitives — `fork` (copy logical blocks with shared physical backing), `append` (add a token to current sequence), `free` (decrement reference counts and release zero-count blocks) — allow any decoding algorithm to be implemented on top of the memory manager.

# Experiments

- **Dataset:** ShareGPT (real-world conversations; mean prompt 408 tokens, mean output 294 tokens) and Alpaca (instruction-following; shorter, more uniform lengths).
- **Hardware:** NVIDIA A100 80GB GPUs.
- **Models:** OPT-13B (1 GPU), OPT-66B (2 GPUs), LLaMA-13B (1 GPU), OPT-175B (8 GPUs).
- **Baselines:** FasterTransformer (static batching), Orca-Oracle (iteration-level scheduling, oracle output lengths), Orca-Pow2 (2× length over-reservation), Orca-Max (maximum possible length reservation).
- **Metric:** Normalized latency (end-to-end latency / output length) at varying request arrival rates; throughput defined as sustainable request rate below a latency threshold.

**Key results:**

| Comparison | Throughput gain |
|---|---|
| vLLM vs. Orca (Oracle) on ShareGPT | 1.7×–2.7× |
| vLLM vs. Orca (Max) on ShareGPT | 2.7×–8× |
| vLLM vs. FasterTransformer | up to 22× |
| Concurrent batch size vs. Orca | 2.2×–4.3× |

- **Parallel sampling (k=4):** 1.3×–2.3× throughput gain over Orca (Oracle); 6.1%–30.5% memory savings.
- **Beam search (w=6):** Up to 2.3× gain; 37.6%–66.3% memory savings.
- **Shared prefix:** 1.67× (1-shot) and 3.58× (5-shot) throughput improvement.

**Block size ablation:** Block size 16 is the default. Smaller blocks underutilize GPU memory bandwidth; larger blocks increase internal fragmentation and reduce parallelism benefits.

**Attention kernel overhead:** PagedAttention is 20–26% slower than contiguous-memory attention in microbenchmarks, but this is dominated by memory savings that allow larger batches.

# Comparison with Related Methods

| System | Scheduling | Memory management | Sharing |
|---|---|---|---|
| FasterTransformer | Static batching | Contiguous, pre-allocated | None |
| Orca | Iteration-level | Contiguous, dynamic | None |
| **vLLM (PagedAttention)** | Iteration-level | Paged, on-demand | Block-level copy-on-write |

> [!IMPORTANT]
> vLLM and Orca's iteration-level scheduling are orthogonal optimizations: Orca focuses on when to schedule requests; vLLM focuses on how memory is allocated. They can be combined, and vLLM builds on iteration-level scheduling as its base.

> [!TIP]
> FlashAttention (Dao et al., 2022) targets compute efficiency (tiling for SRAM) and is orthogonal to PagedAttention's memory management goals. Both can be combined.

# Applicability

- **Who:** ML engineers and infrastructure teams deploying LLM inference at scale on GPUs.
- **When:** Workloads with variable-length inputs/outputs, advanced decoding (beam search, parallel sampling), or shared system prompts benefit most.
- **Where:** Single-GPU and multi-GPU tensor-parallel deployments; applicable to any transformer model with autoregressive KV caching.
- **Limitation:** The custom attention kernel introduces ~20–26% per-step overhead; for extremely short sequences or trivially small batches, contiguous attention may be preferable.
