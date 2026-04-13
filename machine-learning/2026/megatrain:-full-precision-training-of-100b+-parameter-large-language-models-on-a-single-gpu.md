# Meta Information

- URL: [MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU](https://arxiv.org/abs/2604.05091)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yuan, Z., Sun, H., Sun, L., & Ye, Y. (2026). MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU. arXiv:2604.05091.

# MegaTrain: Full Precision Training of 100B+ Parameter LLMs on a Single GPU

## Overview

MegaTrain is a memory-centric training system that inverts the conventional GPU-centric assumption: instead of treating GPU HBM as the primary parameter store and host (CPU) memory as a slower backup, MegaTrain places all persistent state (parameters, gradients, optimizer moments) in host memory and streams each layer's weights into GPU memory only when that layer is being computed. This design decouples model scale from GPU memory size, enabling full-precision (BF16/FP32) training of models exceeding 100B parameters on a single GPU.

**Target users**: ML researchers and practitioners who need to fine-tune or pre-train 7B–120B parameter LLMs but lack access to multi-GPU clusters. MegaTrain is applicable wherever a single high-memory host machine (1–1.5 TB DDR5/LPDDR5X) with a capable GPU (H200, GH200, A100) is available.

**Key results**:
- Trains Qwen2.5-72B and GPT-OSS-120B (MoE) on a single H200 (1.5TB host memory)
- Achieves 1.84× throughput over DeepSpeed ZeRO-3 at 14B scale on GH200
- Supports 512K token context on GH200 at 407.4 TFLOPS

## 1. Background and Problem

### Memory Decomposition

For a model with $P$ parameters, the total persistent memory required for full-precision training is approximately $12P$ bytes:

```math
\begin{align}
  \text{Total} = \underbrace{2P}_{\text{params (BF16)}} + \underbrace{2P}_{\text{grads (BF16)}} + \underbrace{4P}_{\text{m}_1\text{ (FP32)}} + \underbrace{4P}_{\text{m}_2\text{ (FP32)}}
\end{align}
```

where $m_1$ and $m_2$ are the first and second optimizer moments (Adam). For a 70B model this totals ~840 GB — far beyond any single GPU.

### Memory Hierarchy

MegaTrain exploits a four-level hierarchy present in modern GPU servers:

| Level | Bandwidth | Capacity (typical) |
|---|---|---|
| On-chip SRAM | ~80 TB/s | MBs |
| GPU HBM (HBM3/HBM3e) | ~3–5 TB/s | 24–141 GB |
| Host DRAM (DDR5/LPDDR5X) | 200–900 GB/s | 480 GB–1.5 TB |
| NVMe SSD | ~10 GB/s | TBs |

The PCIe 4.0 link between host and device achieves ~26 GB/s (unidirectional), and NVLink-C2C on GH200 reaches 900 GB/s.

### Why Existing Approaches Fall Short

- **ZeRO-3 with CPU offloading** (DeepSpeed): offloads parameters to CPU but still maintains CUDA graph state and persistent optimizer tensors on GPU; throughput collapses beyond 30B because the data movement is not pipelined to hide PCIe latency.
- **FSDP**: shards across GPUs but cannot operate on a single GPU for 70B+ models; encounters OOM beyond ~50 layers at fixed device-memory budgets.
- **Gradient checkpointing alone**: reduces activation memory but does not address persistent parameter memory.

## 2. MegaTrain System Design

### 2.1 Execution Workflow

MegaTrain splits a training step into three sequential phases over a Transformer with $L$ layers, grouped into $B$ blocks of $L/B$ layers each:

**Phase 1 — Streaming Forward Pass**

For layer $i = 1, \ldots, L$:
1. Prefetch weights $W_i \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$ from host DRAM into GPU staging buffer (H2D copy)
2. Execute forward computation: $A_i = f(A_{i-1}, W_i)$
3. Checkpoint activations every $L/K$ layers (keep $K$ checkpoints in GPU memory)
4. Evict $W_i$ from GPU; advance to $W_{i+1}$

**Phase 2 — Streaming Backward Pass**

For block $b = B, \ldots, 1$ (reverse order):
1. Recompute activations within the block from the nearest checkpoint (streaming $W_i$ forward again for recomputation)
2. Stream $W_i$ in reverse layer order for gradient computation
3. Compute $\frac{\partial \mathcal{L}}{\partial W_i}$ and $\frac{\partial \mathcal{L}}{\partial A_{i-1}}$
4. Immediately offload gradient $G_i$ to host DRAM (D2H copy)
5. Evict $W_i$ from GPU

**Phase 3 — CPU Optimizer Update**

All Adam updates execute on CPU using AVX-512 vectorized instructions directly on host-resident parameter and moment tensors, avoiding any round-trip of optimizer state through GPU memory.

### 2.2 Pipelined Double-Buffered Execution Engine

To hide the ~26 GB/s PCIe latency, MegaTrain runs three concurrent CUDA streams:

| Stream | Role |
|---|---|
| Compute Stream | Forward/backward kernel dispatch |
| Weight-Transfer Stream | H2D parameter copies (prefetch) |
| Gradient-Transfer Stream | D2H gradient evacuation |

**Double buffering** allocates two staging buffers (Buffer 0, Buffer 1) in both CPU pinned memory and GPU HBM. While the compute stream executes layer $i$ using Buffer 0, the weight-transfer stream concurrently copies $W_{i+1}$ into Buffer 1. After the compute stream finishes layer $i$, it switches to Buffer 1 for layer $i+1$ while Buffer 0 is recycled for $W_{i+2}$.

Synchronization uses three CUDA events:
1. **Weights-Ready**: gates the compute stream until the current layer's parameters are fully transferred
2. **Backward-Done**: triggers gradient evacuation D2H after the backward kernel completes
3. **Buffer-Free**: permits the weight-transfer stream to reuse a buffer slot once gradient evacuation finishes

This pipeline ensures that PCIe data movement is never on the critical computation path.

### 2.3 Stateless Execution Model

Standard PyTorch autograd maintains a persistent computation graph that assumes all parameter tensors remain allocated in GPU memory. Because MegaTrain evicts each layer's parameters immediately after use, those tensor addresses become invalid before the backward pass starts.

MegaTrain replaces the autograd graph with **stateless layer templates**:
- A template encapsulates the CUDA kernels for an Attention or MLP block but holds no persistent weight pointers
- A `Bind` primitive dynamically maps the current streaming buffer view into the template's input slots before each layer execution
- This allows layer $F_i$ to execute on Template A while layer $W_{i+1}$ is simultaneously being bound to Template B, eliminating weight-preparation latency

Because streamed buffer addresses change at every layer boundary, CUDA graph capture (which bakes pointer addresses at capture time) is incompatible; MegaTrain uses explicit StreamIn → Bind → Compute → Offload dispatch instead.

### 2.4 Memory Management Details

**Layer-Contiguous Tiling**: Parameters $W_i$, gradients $G_i$, and optimizer moments $(m_i^{(1)}, m_i^{(2)})$ for layer $i$ are packed into a single contiguous host-memory block aligned to 4KB pages. This enables the PCIe DMA engine to issue a single large transfer per layer, saturating available bandwidth rather than scattering small kernel launches.

**Pinned Slab Recycling**: Rather than pinning the full model ($O(P)$ pinned memory), a small fixed-size pool of pinned staging buffers (sized to the largest layer) is reused via JIT copying from pageable storage. Pinning overhead is $O(1)$ regardless of model depth $L$.

**Gradient Slab Pool**: $K = 12$ pinned host slabs store evacuated gradients while background CPU threads accumulate and apply optimizer updates. This decouples GPU buffer release from CPU optimizer latency.

**Workspace Stack**: Pre-allocated GPU workspaces managed as stacks eliminate CUDA allocator fragmentation and runtime jitter from cudaMalloc calls during training.

## 3. Relation to Prior Systems

| System | Parallelism | Parameter Location | Optimizer Location | Long Context | 120B on 1 GPU |
|---|---|---|---|---|---|
| PyTorch FSDP | Multi-GPU sharding | GPU HBM | GPU HBM | Limited | ✗ |
| DeepSpeed ZeRO-3 | Multi-GPU + CPU offload | GPU + CPU | CPU | Limited | ✗ |
| Gemini (Colossal-AI) | CPU offload | CPU | CPU | ✗ | ✗ |
| **MegaTrain** | **Single GPU streaming** | **CPU (streamed to GPU)** | **CPU (AVX-512)** | **512K tokens** | **✓** |

The key distinction from ZeRO-3 CPU offloading is the **pipelined double-buffered streaming**: ZeRO-3 fetches parameters synchronously (blocking GPU until transfer completes), whereas MegaTrain overlaps the next layer's prefetch with the current layer's compute, hiding PCIe latency entirely.

## 4. Algorithms

### Algorithm: MegaTrain Forward Pass

```
Input: Layer weights W[1..L] in host DRAM, input activation A[0]
Output: Loss L, activation checkpoints C[0..K]

Buffer = [Buf0, Buf1]  # Two GPU staging buffers (ping-pong)
checkpoint_interval = L / K

for i = 1 to L:
    active_buf = i mod 2
    next_buf   = (i+1) mod 2

    # Overlap: prefetch next layer while computing current
    async H2D_copy(W[i+1], Buffer[next_buf])  # Weight-Transfer Stream

    wait(Weights-Ready event for Buffer[active_buf])
    Bind(Template, Buffer[active_buf])
    A[i] = Compute(Template, A[i-1])           # Compute Stream
    
    if i mod checkpoint_interval == 0:
        C[i // checkpoint_interval] = A[i]     # Save checkpoint to GPU
    else:
        free(A[i-1])                            # Discard intermediate activation

    signal(Buffer-Free event for Buffer[active_buf])

return A[L], C
```

### Algorithm: MegaTrain Backward Pass

```
Input: Checkpoints C[0..K], weights W[1..L] in host DRAM, loss gradient dL/dA[L]
Output: Gradients G[1..L] in host DRAM

for block b = K downto 1:
    # Recompute activations in this block from checkpoint
    for i = (b-1)*block_size+1 to b*block_size:
        async H2D_copy(W[i], Buffer)
        wait(Weights-Ready)
        A[i] = Compute(Template, A[i-1])   # Recomputation forward

    # Backward through the block
    for i = b*block_size downto (b-1)*block_size+1:
        async H2D_copy(W[i], Buffer)
        wait(Weights-Ready)
        dL/dA[i-1], G[i] = Backward(Template, A[i-1], dL/dA[i])
        
        async D2H_copy(G[i], GradSlab)   # Gradient-Transfer Stream
        signal(Backward-Done)
        wait(Buffer-Free)
        free(Buffer)

# CPU optimizer update (off GPU critical path)
for i = 1 to L:
    W[i] -= lr * Adam(G[i], m1[i], m2[i])  # AVX-512 vectorized on CPU
```

## Experiments

- **Dataset**: MetaMathQA (~395,000 math problem-answer pairs; 70% training / 30% test); evaluation metric is exact-match accuracy on mathematical reasoning tasks
- **Models**: Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, Qwen2.5-72B; GPT-OSS-120B (MoE architecture)
- **Hardware**: GH200 (96GB HBM3, 480GB LPDDR5X, 900 GB/s NVLink-C2C), H200 SXM (141GB HBM3e, 1.5TB DDR5, 128 GB/s PCIe Gen4), A100 PCIe, RTX A6000, RTX 3090
- **Optimizer**: Adam with AVX-512 vectorized CPU execution
- **Baselines**: DeepSpeed ZeRO-3 with CPU offloading, PyTorch FSDP, Gemini (Colossal-AI)

**Key results**:
- GH200 14B: 264 TFLOPS — **1.84× speedup over ZeRO-3** (143.6 TFLOPS)
- H200 achieves stable training up to **120B parameters** (near-flat host memory growth vs. near-exponential growth for ZeRO-3 beyond 30B)
- Fixed 3.83 GB GPU budget, 56-layer depth: **6.14× speedup over FSDP** (264 vs 43 TFLOPS); FSDP hits OOM at 84+ layers while MegaTrain scales to 180 layers
- Long context on GH200: **407.4 TFLOPS at 512K tokens** with stable memory usage
- Consumer GPU (RTX A6000 48GB): trains 14B at 56.82 TFLOPS vs ZeRO-3 batch-size-1 limit
- Accuracy at 7B and 14B: statistically indistinguishable from PyTorch Native and ZeRO-3, confirming no numerical drift from explicit recomputation

**Ablation highlights**:
- Removing double buffering: **−31.3% throughput** (266.3 → 182.9 TFLOPS), largest single contribution
- Removing gradient slab pooling: minor effect (~−3%)
- Checkpoint interval = 1 (every layer): reduces feasible batch size and throughput to 240.45 TFLOPS due to GPU memory pressure from retained activations

> [!IMPORTANT]
> The throughput advantage of MegaTrain grows with model depth and width. At shallow depths or narrow widths, ZeRO-3 may match or exceed MegaTrain because the PCIe bandwidth cost becomes relatively larger compared to compute. MegaTrain is most beneficial when the model is depth-heavy (many layers, moderate width per layer) relative to GPU compute capacity.

> [!NOTE]
> "MegaTrain's explicit recompute and CPU-master design do not introduce numerical drift" — verified by near-identical MetaMathQA accuracy across training configurations.

> [!TIP]
> The authors plan to extend MegaTrain to multi-GPU via tensor/expert parallelism and to SSD-backed storage for trillion-parameter training. The code is available at https://github.com/DLYuanGod/MegaTrain.
