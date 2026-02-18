# Meta Information

- URL: [AgentSight: System-Level Observability for AI Agents Using eBPF](https://arxiv.org/abs/2508.02736)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zheng, Y., Hu, Y., Yu, T., & Quinn, A. (2025). AgentSight: System-Level Observability for AI Agents Using eBPF. PASM'2025.

# AgentSight: System-Level Observability for AI Agents Using eBPF

## Overview

AgentSight is an observability framework for AI agents that bridges the **semantic gap** between an agent's high-level intent (expressed in LLM prompts) and its low-level system actions (expressed as syscalls). It is designed for security teams, platform engineers, and AI developers who need to audit, debug, or secure deployed LLM agents without modifying agent code or framework internals.

**Applicability**: Any environment running LLM-based agents (LangChain, AutoGen, Claude Code, Cursor, Gemini CLI, etc.) on Linux systems. Particularly useful when:
- Auditing multi-agent workflows for unexpected behavior
- Detecting prompt injection attacks in production
- Diagnosing runaway reasoning loops causing resource waste
- Analyzing coordination bottlenecks in multi-agent systems

## Problem: The Semantic Gap in Agent Monitoring

Existing observability tools observe agent behavior from only one of two layers:

| Tool Type | What They See | What They Miss |
|-----------|--------------|----------------|
| Application-level (LangSmith, Langfuse, AgentOps) | LLM prompts and responses (intent) | Actual filesystem, network, process actions |
| System-level (Falco, Tracee) | Syscalls and kernel events (actions) | Why the agent issued those syscalls |

This gap makes it impossible to answer questions like "did the agent perform this file read because it was instructed to, or because it was injected?" AgentSight correlates intent with actions in a single unified trace.

## Technical Design: Boundary Tracing

AgentSight's core innovation is **boundary tracing**: instrumenting two stable, unavoidable system boundaries rather than agent-specific code paths.

### Two Observation Boundaries

1. **Network boundary** — intercepts TLS-encrypted LLM API traffic using uprobes attached to `SSL_read`/`SSL_write` in the OpenSSL library, extracting plaintext prompt/response pairs.
2. **Kernel boundary** — monitors process creation (`sched_process_exec`), file access (`openat2`), and network connections (`connect`) via stable eBPF tracepoints and kprobes.

Because both boundaries are at stable OS/TLS interfaces rather than framework APIs, AgentSight remains resilient to framework version changes and works across all agent frameworks without code instrumentation.

### Architecture

```
+--------------------+        +----------------------+
|   eBPF Programs    |        |   Userspace Daemon   |
|                    |        |   (Rust, ~6K LoC)    |
| - uprobe on        |------> | - Event ring buffer  |
|   SSL_read/write   |        | - Process lineage    |
| - tracepoints on   |        |   tracker            |
|   sched_process_   |        | - Correlation engine |
|   exec, openat2,   |        |                      |
|   connect, execve  |        +----------+-----------+
+--------------------+                   |
                                         v
                             +----------------------+
                             |  TypeScript Frontend |
                             |  (~3K LoC)           |
                             | - Event correlation  |
                             | - LLM analysis       |
                             | - Visualization      |
                             +----------------------+
```

**Input**: Raw kernel events (process, file, network) and decrypted LLM API payloads.
**Output**: Correlated event graph linking agent intent to system actions, plus LLM-generated behavioral analysis.

### Hybrid Correlation Engine

The correlation engine operates in two stages:

**Stage 1 — Heuristic linking** (real-time, low latency):
1. Build a process tree by tracking `fork`/`execve` events with parent PIDs
2. For each syscall event, walk up the process tree to find the agent process
3. Match the syscall timestamp to an LLM response within a **100–500 ms temporal proximity window**
4. Perform **direct argument matching**: check if filenames/URLs in the syscall match strings in the LLM response

**Stage 2 — Semantic linking** (asynchronous, LLM-assisted):
- Feed correlated event clusters to a secondary LLM
- Ask the LLM to reason about whether the observed actions match the agent's stated goal
- Flag discrepancies as potential injection, loop, or anomaly

**In-kernel filtering**: Aggressive eBPF filtering discards irrelevant events (e.g., syscalls from unrelated system processes) before they reach userspace, minimizing overhead.

## Implementation

- **Daemon**: ~6,000 lines of Rust and C for eBPF programs and the collection/correlation core
- **Frontend**: ~3,000 lines of TypeScript for analysis UI and LLM integration
- **Open source**: https://github.com/eunomia-bpf/agentsight

## Experiments

- **Dataset**: No standard ML dataset — evaluated on three designed case studies simulating real-world agent behaviors (prompt injection attack, reasoning loop, multi-agent coordination)
- **Hardware**: Not explicitly specified
- **Baseline**: Agent task execution without AgentSight monitoring

### Performance Overhead

| Task | Baseline (s) | With AgentSight (s) | Overhead |
|------|-------------|---------------------|----------|
| Understand repository | 127.98 | 132.33 | 3.4% |
| Code writing | 22.54 | 23.64 | 4.9% |
| Repository compilation | 92.40 | 92.72 | 0.4% |
| **Average** | — | — | **2.9%** |

### Case Study 1: Prompt Injection Detection

An agent tasked with summarizing a codebase was given a README containing a hidden prompt instructing it to exfiltrate `/etc/passwd`. AgentSight:
- Captured 521 raw events, reduced to 37 merged events after correlation
- Secondary LLM identified a logical inconsistency: the agent read `/etc/passwd` but this action had no plausible connection to the stated "summarize code" goal
- Successfully flagged the attack chain end-to-end

### Case Study 2: Reasoning Loop Detection

An agent got stuck in a "try–fail–re-reason" loop, repeatedly executing the same failing shell command. AgentSight:
- Detected anomalous repetition of identical syscall patterns
- Measured disproportionate resource consumption in LLM API calls relative to productive output
- Secondary LLM labeled the behavior as a reasoning loop rather than normal retry logic

### Case Study 3: Multi-Agent Coordination (6 agents)

Six software development agents collaborated concurrently. AgentSight captured 3,153 correlated events and revealed:
- Sequential dependency bottlenecks where agents waited on each other's outputs
- File locking contention during parallel write operations
- The complete causal graph of inter-agent communication and resource sharing

## Comparison with Related Approaches

| Aspect | Application-level Tracing (LangSmith) | System-level Tools (Falco) | AgentSight |
|--------|--------------------------------------|---------------------------|------------|
| Captures intent | Yes | No | Yes (via TLS interception) |
| Captures actions | Partial | Yes | Yes (via eBPF kernel events) |
| Correlates intent↔action | No | No | Yes (hybrid engine) |
| Framework agnostic | No | Yes | Yes |
| Code instrumentation needed | Yes | No | No |
| Overhead | Variable | ~1–5% | ~2.9% |

> [!NOTE]
> The "AI to watch AI" approach — using a secondary LLM as the reasoning engine for anomaly detection — differs fundamentally from rule-based intrusion detection. This allows detecting novel attack patterns without predefined signatures, at the cost of LLM inference latency and cost.

> [!IMPORTANT]
> The TLS interception approach (uprobes on `SSL_read`/`SSL_write`) depends on the agent process using a linked OpenSSL library. Agents using other TLS libraries (BoringSSL, mbedTLS) or in-process HTTP clients that bypass standard SSL functions may not be captured.

> [!TIP]
> eBPF tracing background: [BPF Performance Tools (Gregg, 2019)](http://www.brendangregg.com/bpf-performance-tools-book.html) and the [eBPF documentation](https://ebpf.io/what-is-ebpf/) provide foundational context for understanding how kernel-level instrumentation works.
