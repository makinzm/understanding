# Meta Information

- URL: [AI scientists produce results without reasoning scientifically](https://arxiv.org/abs/2604.18805)
- LICENSE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Reference: Ríos-García, M., Alampara, N., Gupta, C., Mandal, I., Mannan, S., Aghajani, A. A., Krishnan, N. M. A., & Jablonka, K. M. (2026). AI scientists produce results without reasoning scientifically. arXiv:2604.18805.

# Overview

This paper investigates a fundamental disconnect in LLM-based autonomous scientific agents: they frequently achieve correct experimental outcomes while bypassing genuine scientific reasoning. Running over 25,000 agent traces across 8 scientific domains in the CORRAL benchmark, the authors find that evidence is ignored in 68% of traces and refutation-driven belief revision (updating beliefs when evidence contradicts a hypothesis) occurs in only 26% of traces. Critically, the choice of base language model explains 41.4% of performance variance, while the agent scaffold (framework architecture) explains only 1.5% — indicating that scaffold engineering alone cannot fix the underlying reasoning failures. These problematic patterns persist even when agents are shown successful reasoning examples as in-context demonstrations.

> [!IMPORTANT]
> The core claim is an epistemological one: current AI science agents justify their results through pattern-matching rather than through a valid scientific process. Even if their final answers are correct, the reasoning path is not scientifically valid, which undermines reproducibility, interpretability, and trust.

# Background and Motivation

## The AI Scientist Paradigm

Recent work has proposed fully autonomous AI agents that can conduct scientific experiments end-to-end: forming hypotheses, running computational tools, interpreting results, and iterating. These systems are evaluated primarily on outcome metrics (e.g., prediction accuracy, optimization gain). The implicit assumption is that a correct outcome implies a sound process.

The authors challenge this assumption. They draw on philosophy of science — specifically Karl Popper's principle of falsificationism — to argue that scientific reasoning requires:
1. Forming explicit, testable hypotheses before gathering evidence
2. Treating disconfirming evidence as a signal for belief revision (refutation-driven updates)
3. Committing to conclusions only after systematic evidential support

Without these properties, a system cannot be said to reason scientifically even if it produces correct outputs.

## Related Work

| Work | Focus | Limitation |
|---|---|---|
| ScienceAgentBench | Task success on scientific coding tasks | No reasoning process assessment |
| BixBench | Bioinformatics agent evaluation | Outcome-focused metrics only |
| MLAgentBench | ML research automation | No epistemic validity check |
| CORRAL (this paper) | Epistemic trace analysis across domains | First to separate outcome from reasoning process |

# CORRAL Benchmark

## Architecture

CORRAL (Collaborative Reasoner and Research Agent Laboratory) is a microservice-based benchmark framework with three decoupled components:

- **Environments**: Simulate scientific domains; define the task space, available tools, and observable feedback
- **Agents**: LLM-based entities using scaffolds such as ReAct, ToolCalling, LLMPlanner, and Reflection
- **Tasks**: Problems with scoring functions for evaluation; can be chained into `TaskGroups` for multi-stage challenges

The system uses a client-server design: `CorralRunner` sends REST API calls to `CorralServer`, which manages environments via `CorralRouter`.

## Scientific Environments

CORRAL includes 9 pre-built environments spanning multiple scientific disciplines:

| Environment | Domain | Description |
|---|---|---|
| `corral_spectra` | Chemistry | Spectroscopy data analysis |
| `corral_molecular_dynamics` | Physics/Chemistry | Molecular dynamics setup and analysis |
| `corral_open_catalyst` | Catalysis | Heterogeneous catalysis task optimization |
| `corral_afm` | Materials Science | Atomic force microscopy interpretation |
| `corral_retrosynthesis` | Chemistry | Plan multi-step organic synthesis routes |
| `corral_resistor` | Physics | Infer circuit topology from measurements |
| `corral_wet_lab` | Biology/Chemistry | Virtual lab for ion concentration analysis |
| `corral_simple_ml` | Machine Learning | Basic supervised/unsupervised modeling tasks |
| `corral_samplemath` | Mathematics | Basic mathematical operations |

> [!NOTE]
> The paper evaluates agents across 8 domains. The exact subset used is described in the appendices of the full paper.

## Epistemic Traces

The central methodological contribution is **Epistemic Traces** — an epistemological graph analysis system that annotates the reasoning steps within each agent trajectory. Each step in an agent's trace is classified into one of six node types:

| Node Type | Meaning |
|---|---|
| **Hypothesis** | A proposed explanation or prediction before seeing evidence |
| **Test** | An experimental or analytical action intended to validate the hypothesis |
| **Evidence** | An observational or computational result returned from the environment |
| **Judgment** | An evaluative assessment of evidence relative to the hypothesis |
| **Update** | A modification of beliefs or plans based on the judgment |
| **Commitment** | A decision to commit to a specific conclusion or direction |

Each step also receives behavioral markers:

- **Positive**: explicit validation, recognizing dead ends and pivoting, articulating plans, connecting hypotheses to data, systematic task tracking
- **Neutral**: unremarkable iteration, reaching iteration limits
- **Negative**: absent validation, superfluous tool calls, incoherent outputs, repeated patterns, hallucinated information, premature answers, abandoned attempts

A scientifically valid reasoning trace should cycle through Hypothesis → Test → Evidence → Judgment → Update, repeating as needed until a justified Commitment is made.

# Key Findings

## Evidence Ignored in 68% of Traces

In 68% of agent traces, the agent does not acknowledge or incorporate available evidence into its subsequent reasoning. The agent proceeds to conclusions without connecting observations to hypotheses — a direct violation of evidential reasoning.

## Refutation-Driven Revision in Only 26% of Traces

Even when the agent observes evidence that contradicts its hypothesis, belief revision (updating the hypothesis or plan) occurs in only 26% of cases. The majority of the time, the agent either ignores the contradictory evidence or proceeds to commit to the original hypothesis anyway.

## Base Model Dominates Scaffold

A variance decomposition analysis across the full experimental matrix yields:

```math
\begin{align}
  \text{Var explained by base model} &= 41.4\% \\
  \text{Var explained by scaffold} &= 1.5\%
\end{align}
```

This means engineering better agent frameworks (ReAct vs. ToolCalling vs. Reflection) has negligible effect on outcome quality compared to the underlying LLM's capabilities. However, since both base model and scaffold have low epistemic trace quality, this is not a solution — it shows neither dimension captures scientific reasoning.

## In-Context Demonstrations Do Not Fix Reasoning

Providing agents with example traces that demonstrate proper scientific reasoning (hypothesis → test → evidence → update) as few-shot context does not substantially improve epistemic quality. The failures are not correctable by prompting alone.

> [!IMPORTANT]
> The authors conclude: until scientific reasoning is made an explicit training objective during model development, the reasoning processes of AI science agents cannot be justified — regardless of their outcome accuracy.

# Methodology

## Agent Variants Tested

| Scaffold | Description |
|---|---|
| ReAct | Interleaves reasoning steps (thought) with tool actions |
| ToolCalling | Uses structured tool-call outputs without explicit reasoning traces |
| LLMPlanner | Generates a multi-step plan before execution |
| Reflection | Includes a self-critique step after each action |

## Measurement of Scientific Reasoning

For each agent trajectory:
1. An LLM-based annotator classifies each reasoning step into Epistemic Trace node types
2. Behavioral markers (positive/neutral/negative) are assigned
3. The fraction of traces showing refutation-driven updates is computed
4. The fraction where evidence is not referenced in subsequent steps is computed

These metrics are computed separately from outcome correctness scores, allowing direct comparison of process quality vs. result quality.

## Variance Decomposition

Performance variance is partitioned using a mixed-effects model with base model identity and scaffold type as factors. The explained variance fractions (41.4% and 1.5%) quantify the relative contribution of each.

# Implications

## For Evaluation

Benchmarks that measure only final accuracy cannot detect these epistemic failures. A system can score well on ScienceAgentBench while reasoning non-scientifically. Evaluation frameworks must include process-level metrics.

## For Training

Since few-shot prompting cannot fix reasoning failures, and scaffold engineering has negligible effect, the authors argue that scientific reasoning must be targeted during pre-training or fine-tuning — for example via reinforcement learning from process-based feedback or supervision on reasoning traces rather than final answers.

## For Deployment

Researchers using AI science agents for autonomous discovery cannot trust that a correct result was obtained through a valid scientific process. Human oversight of the reasoning trajectory — not just the outcome — is necessary.

# Experiments

- **Benchmark**: CORRAL (open-source, available at lamalab-org.github.io/corral)
- **Total runs**: >25,000 agent trajectories
- **Domains**: 8 scientific environments (chemistry, physics, materials science, biology, ML, mathematics)
- **Agent scaffolds**: ReAct, ToolCalling, LLMPlanner, Reflection
- **Key metrics**:
  - Evidence incorporation rate: 32% (ignored in 68% of traces)
  - Refutation-driven revision rate: 26%
  - Variance explained by base model: 41.4%
  - Variance explained by scaffold: 1.5%
