# Meta Information

- URL: [From AGI to ASI](https://arxiv.org/abs/2606.12683)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Genewein, T., Franklin, M., Lerchner, A., Orseau, L., Albanie, S., Bales, A., Wyeth, C., Chan, S., Gabriel, I., Leibo, J. Z., Dafoe, A., Hutter, M., Graepel, T., & Legg, S. (2026). From AGI to ASI. arXiv preprint arXiv:2606.12683.

# From AGI to ASI

## Overview

This report from Google DeepMind (with contributors from the University of Waterloo, Australian National University, and University College London) analyzes the technological pathways and fundamental bottlenecks on the road from human-level Artificial General Intelligence (AGI) to Artificial Superintelligence (ASI). Rather than treating ASI as a singular event, the authors frame the transition as a continuum of capability improvements driven by four distinct technological mechanisms, each subject to specific bottlenecks.

This work is relevant to AI researchers, policymakers, and safety researchers who need to reason about AI development trajectories beyond current systems. It serves as a roadmap for understanding which technical levers drive capability improvements and where those levers might fail or be deliberately constrained.

## Characterizing Artificial Superintelligence

The paper establishes a three-level hierarchy of machine intelligence:

| Level | Definition | Notes |
|-------|-----------|-------|
| **AGI** | A system roughly as intelligent as a single human at median human-level on most cognitive tasks | Early AGI will already exceed humans in many narrow domains |
| **ASI** | A system significantly more capable than large organizations of human experts across virtually all tasks and domains | Threshold is deliberately set high to distinguish from narrow superhuman systems |
| **UAI (Universal AI)** | Theoretical upper bound of machine intelligence, formalized by AIXI | Incomputable in practice; serves as a formal ceiling |

> [!IMPORTANT]
> The report emphasizes that AGI and ASI are points on a continuum rather than discrete events. Current AI systems already exceed human performance in many domains (e.g., chess, protein folding) while remaining below human-level in others, placing them between narrow AI and AGI on this continuum.

## Universal AI — An Informal Overview

The formal theoretical grounding comes from the **Legg-Hutter intelligence measure**, which defines intelligence as average performance across all computable tasks, with simpler tasks (lower Kolmogorov complexity) receiving higher weight:

> [!NOTE]
> "The Legg-Hutter intelligence score is the average performance of an agent across all computable tasks, where lower-complexity tasks are weighted more heavily by their Kolmogorov complexity." — Paraphrased from the paper

The **AIXI agent** is the theoretical optimum under this framework. It selects actions by maximizing expected cumulative reward over all computable environments, weighted by Solomonoff's Universal Prior. AIXI resolves three fundamental challenges simultaneously:

1. **Acting under uncertainty**: Uses Bayesian hypothesis updating to maintain a distribution over all possible environments and updates it with each new observation
2. **Interactive decision-making**: Solves the credit assignment problem by explicitly planning over long horizons
3. **Exploration-exploitation**: Balances trying new strategies versus exploiting known rewards through the Bayesian prior

The AIXI framework is **incomputable** — it requires summing over all Turing machines — making it an idealized upper bound. Practical AI systems are approximations from below on the Legg-Hutter scale.

### Advantages of Digital Intelligence

Digital systems possess scale-dependent advantages over biological intelligence that amplify with more compute:

| Advantage | Description |
|-----------|-------------|
| **Bandwidth** | I/O rates far exceeding biological sensory/motor limits |
| **Processing speed** | Internal computation speed expandable by allocating more compute |
| **Working memory** | Context windows and external storage exceed biological RAM by orders of magnitude |
| **Substrate independence** | Can transition across hardware generations without loss of "personality" |
| **Lossless replication** | Perfect copies preserve all learned experience |
| **High-bandwidth sharing** | Agent collectives can share internal representations without lossy verbal communication |

> [!IMPORTANT]
> These advantages are not fixed — they scale with compute. An ASI with 1,000× more compute than current systems amplifies each of these advantages proportionally, creating compounding effects relative to human intelligence.

## Technological Pathways to ASI

### 5.1 Scaling Compute, Models, and Data

Effective compute has grown at an estimated **10× per year**, combining:

```
Effective Compute Growth ≈ 10×/year
  = Hardware improvement (~1.5×/year)
  × Investment growth (~2.5×/year)
  × Algorithmic efficiency (~3–6×/year)
```

Capability improvements follow **approximate power laws** with respect to parameters, data, and compute. These scaling laws are empirically measured but not guaranteed to hold indefinitely.

**Primary uncertainty**: Whether scale increases reliably translate to general capability advances, or whether they encounter diminishing returns in specific capability domains.

### 5.2 Algorithmic Paradigm Shifts

Current transformer-based pretraining may represent one paradigm among many. Potential successor paradigms include:

- **Novel architectures**: State-space models, mixture-of-experts with fundamentally different routing, neuromorphic systems
- **RL-based training**: Agents learning from interaction rather than passive data imitation
- **Spiking neural networks**: More energy-efficient computation inspired by biological neurons
- **Hybrid symbolic-neural systems**: Combining neural pattern recognition with explicit reasoning

Each paradigm shift historically produced step-change improvements (e.g., the transition from hand-crafted features to deep learning via AlexNet, or from supervised pretraining to RLHF).

### 5.3 Recursive Self-Improvement (RSI)

RSI occurs when AI systems accelerate AI research and development, creating positive feedback loops. The authors identify four improvement mechanisms:

| Mechanism | Biological Analogy | AI Equivalent |
|-----------|-------------------|--------------|
| **Genotypic RSI** | Genetic evolution of inherited instructions | Automated optimization of code, architectures, and hardware blueprints |
| **Memetic RSI** | Cultural transmission of learned knowledge | Automated dataset collection, synthetic data generation, recursive distillation of test-time compute |
| **Sociogenic RSI** | Specialization and division of labor in societies | Agent collectives specializing and freeing resources for larger collective intelligence |
| **Infrastructure RSI** | Tool-making and environmental modification | AI-driven chip design, data center optimization, and dataset curation |

**RSI Dynamics Pseudocode**:

```
Algorithm: Recursive Self-Improvement Loop
---------------------------------------------------------------------------
Given: AI system A with capability level C(A)
       Research task distribution T (benchmark for AI research speed)

1. WHILE C(A) < ASI_threshold:
   a. Use A to automate AI research tasks in T
   b. Measure AI research speed-up S relative to human-only baseline
   c. Apply discovered improvements to produce A' with C(A') > C(A)
   d. A ← A'
   
2. IF S is exponential in C(A):
   → growth transitions from exponential to hyperbolic
3. IF S is linear in C(A):
   → growth remains exponential but at an accelerated rate
4. IF S is sub-linear or saturates:
   → RSI provides limited additional acceleration
```

> [!CAUTION]
> Whether RSI produces hyperbolic vs. sustained exponential growth depends on empirical properties of the AI research task distribution — properties that are currently unknown. The paper highlights measuring S as a key open research problem.

### 5.4 Multi-Agent Coordination and Group Agency

ASI may emerge from **organized collectives of AGI agents** rather than a single monolithic system. Two organizational modes are discussed:

**Centralized coordination**: A hierarchical structure with a principal AI orchestrating specialized sub-agents, analogous to a corporation. Advantages include directed resource allocation but risks include single points of failure and misalignment amplification.

**Decentralized market mechanisms**: Self-organizing agent economies where tasks are allocated through bidding and market prices, analogous to a decentralized marketplace. More resilient but potentially less directed toward specific goals.

> [!NOTE]
> "Group intelligence can scale in ways that individual intelligence cannot — a multi-agent collective can divide labor, parallelize exploration, and accumulate institutional knowledge across agent lifetimes." — Paraphrased from the paper

The authors note that the **abstraction barrier** is particularly relevant here: current AI systems trained on human-generated data may inherit human conceptual abstractions. Multi-agent systems that develop internal communication languages may form novel abstractions inaccessible to human understanding.

## Potential Bottlenecks to ASI

Six categories of bottlenecks could slow or halt the ASI transition:

| Bottleneck | Core Challenge | Potential Mitigations |
|-----------|---------------|--------------------|
| **Data Wall** | High-quality text training data exhaustion estimated later this decade | Synthetic data generation, self-play, interaction-based learning from environments |
| **Resource Demands** | Economic viability and natural resource constraints for continued compute scaling | AI-driven efficiency improvements, infrastructure optimization |
| **Paradigm Sufficiency** | Transformer-based neural networks may be fundamentally insufficient for some capability domains | Paradigm evolution research, hybrid approaches |
| **Research Complexity** | Diminishing returns as AI research problems become harder (complexity wall) | AI-assisted research acceleration; RSI partially mitigates this |
| **Abstraction Barrier** | AI systems trained on human abstractions may lack capacity for genuinely novel concept formation | Scaling, interactive learning environments, multi-agent diversity |
| **Deliberate Slowdown** | Regulatory, safety, or geopolitical interventions limiting development | Economic and competitive pressures; governance frameworks |

### Fundamental Physical and Complexity Limits

Even a fully realized ASI remains bounded by:

- **Speed of light**: Limits communication latency across spatially distributed systems
- **Thermodynamic limits**: Energy dissipation imposes lower bounds on computation
- **Real-time constraints**: Complex physical environments cannot be simulated faster than real-time without full physical replicas
- **Complexity-theoretic barriers**: P vs. NP and similar results imply some problems cannot be efficiently solved regardless of intelligence level
- **Gödel incompleteness**: Limits provability within any sufficiently expressive formal system
- **Epistemic uncertainty**: Irreducible randomness in quantum systems and chaotic dynamics

## Comparison with Predecessor Frameworks

| Framework | Intelligence Measure | Primary Use | Limitations |
|-----------|---------------------|-------------|-------------|
| **Turing Test** | Behavioral indistinguishability from a human in conversation | Historical milestone | Single-task, excludes superhuman performance measurement |
| **Task-specific benchmarks** | Performance on fixed test sets (MMLU, HumanEval, etc.) | Current AI evaluation | Saturate quickly; can be gamed; don't measure general intelligence |
| **Legg-Hutter score (AIXI)** | Average performance over all computable tasks weighted by Kolmogorov complexity | Theoretical upper bound | Incomputable; no practical measurement procedure |
| **This paper's AGI/ASI framework** | Qualitative comparison to human individuals and expert organizations | Policy and forecasting | Lacks formal mathematical specification |

## Research Agenda

The authors identify the following open problems as highest priority:

1. **Scaling law extrapolation**: Modeling capability improvements as a function of effective compute with quantified uncertainty bands
2. **RSI measurement**: Empirically measuring how much AI accelerates AI research ($S$ in the RSI pseudocode above)
3. **Bottleneck binding analysis**: Determining when and how each bottleneck becomes the limiting constraint
4. **Multi-agent intelligence scaling**: Characterizing how group intelligence scales with collective size and coordination mechanism
5. **Abstraction barrier characterization**: Measuring when and how AI systems form novel abstractions versus recombining human ones
6. **Forecasting ensemble methods**: Combining multiple quantitative forecasting models to produce calibrated probability distributions over AI capability timelines

# Experiments

The paper does not conduct empirical experiments. It is a theoretical and analytical report.

- **Datasets used**: None (conceptual analysis)
- **Benchmarks referenced**: AlexNet on ImageNet (2012 baseline), AlphaFold, AlphaGo/AlphaZero (as historical examples of paradigm shifts), general LLM benchmarks (unnamed)
- **Quantitative claims**:
  - Effective compute growth: ~10× per year (combining hardware ×1.5, investment ×2.5, algorithmic efficiency ×3–6)
  - High-quality text data runway: estimated to exhaust later this decade (2020s)
  - AGI timeline: "within the next decade or less" (per the report's framing; no formal probability estimates given)
