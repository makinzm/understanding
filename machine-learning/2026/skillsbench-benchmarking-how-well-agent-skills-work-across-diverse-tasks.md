# Meta Information

- URL: [SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks](https://arxiv.org/abs/2602.12670)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, X., Chen, W., Liu, Y., et al. (2026). SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks. arXiv:2602.12670.

# SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks

SkillsBench evaluates a specific augmentation strategy for LLM-based agents: providing structured procedural knowledge files (called "skills") at inference time. The benchmark measures whether curated skills reliably improve agent performance across diverse real-world tasks, and quantifies how much of this benefit transfers from human-curated versus self-generated skills.

**Target users**: Researchers developing LLM agent frameworks, practitioners deploying agents for domain-specific tasks, and benchmark designers studying procedural knowledge augmentation.

## Background: Agent Skills vs. Related Concepts

A "skill" in this context is a file-based artifact combining procedural guidance with optional executable resources. It differs from adjacent techniques:

| Concept | Format | Content Type | Scope |
|---|---|---|---|
| System prompt | Text | Behavioral instructions | Entire session |
| Few-shot examples | Input-output pairs | Demonstrations | Query-specific |
| RAG retrieval | Passages | Factual knowledge | Query-specific |
| Tool documentation | API specs | Interface contracts | Tool-specific |
| **Skill (this paper)** | SKILL.md + resources | Procedural workflows | Task-class-wide |

Skills meet four defining criteria:
1. **Procedural content**: How-to guidance and step-by-step workflows (not factual statements)
2. **Task-class applicability**: One skill applies to a class of similar problems, not a single instance
3. **Structured components**: A `SKILL.md` file plus optional executable scripts, configs, or templates
4. **Portability**: File system-based, compatible across agent harnesses (Claude Code, Gemini CLI, etc.)

> [!NOTE]
> "Skills are structured packages of procedural knowledge that augment LLM agents." The key contrast is that skills encode *how to do* something rather than *what* something is.

## Benchmark Design

### Task Composition

- **84 evaluated tasks** across 11 domains
- **Three difficulty strata**:
  - Core: 17 tasks (< 60 min human completion time)
  - Extended: 43 tasks (1–4 hours)
  - Extreme: 26 tasks (> 4 hours)
- **Domains**: Healthcare, Manufacturing, Cybersecurity, Natural Science, Energy, Office Work, Finance, Media Production, Robotics, Mathematics, Software Engineering

Each task consists of:
- `instruction.md`: Human-authored task description
- `environment/skills/`: Directory containing curated skills for that task class
- Oracle verifier: Deterministic programmatic assertions for binary pass/fail scoring

### Quality Control Pipeline

Six requirements enforced for every contributed task:

1. **Human-authored instructions**: LLM-generated task descriptions were excluded
2. **Skill generality**: Skills must guide task classes; solutions to specific instances are rejected
3. **Deterministic verification**: Success criteria implemented as programmatic assertions (not LLM judges)
4. **Automated validation**: Structural checks, oracle execution, instruction quality assessment
5. **Human review**: Evaluation of task realism, oracle quality, skill quality, and anti-cheating measures
6. **Leakage prevention**: Validation agents flagged skills containing task-specific filenames, exact command sequences, or constants that would trivially solve the instance

### Evaluation Infrastructure

Built on the Harbor framework (Merrill et al., 2026) for containerized, reproducible evaluation.

**Input to agent**: `instruction.md` (always) + optionally `environment/skills/` directory
**Output scored**: Binary pass/fail via oracle verifier
**Sampling**: Temperature 0, 5 trials per task → averaged pass rate
**Total trajectories**: 7,308 across all configurations

## Experimental Setup

### Three Experimental Conditions

```
Condition 1 (Vanilla):
  Input: instruction.md only
  → Agent attempts task without procedural guidance

Condition 2 (Curated Skills):
  Input: instruction.md + environment/skills/ directory
  → Agent can read and apply human-authored procedural knowledge

Condition 3 (Self-Generated Skills):
  Input: instruction.md
  Pre-step: Agent prompted to generate procedural knowledge
  → Agent uses its own generated guidance before solving
```

### Agent-Model Configurations (7 total)

| Harness | Model |
|---|---|
| Claude Code | Claude Opus 4.5 |
| Claude Code | Claude Opus 4.6 |
| Claude Code | Claude Sonnet 4.5 |
| Claude Code | Claude Haiku 4.5 |
| Gemini CLI | Gemini 3 Pro |
| Gemini CLI | Gemini 3 Flash |
| Codex CLI | GPT-5.2 |

### Evaluation Metrics

**Pass Rate** $P$: Binary score per trial, averaged across 5 trials and then across all tasks:

$$P = \frac{1}{|T|} \sum_{t \in T} \frac{1}{5} \sum_{i=1}^{5} \mathbb{1}[\text{pass}_{t,i}]$$

where $T$ is the full task set and $\mathbb{1}[\text{pass}_{t,i}]$ is 1 if trial $i$ on task $t$ passes the oracle verifier.

**Normalized Gain** $g$ (Hake's formulation): Proportional improvement toward perfect performance:

$$g = \frac{P_{\text{skill}} - P_{\text{vanilla}}}{1 - P_{\text{vanilla}}}$$

where $P_{\text{skill}}$ is pass rate with skills and $P_{\text{vanilla}}$ is baseline pass rate. $g = 1.0$ means skills brought performance to 100%; $g = 0$ means no improvement.

## Results

### Overall Performance

Curated skills improved average pass rate by **+16.2 percentage points** (pp), varying from +13.6pp to +23.3pp across agent-model configurations. Self-generated skills yielded negligible average benefit of **−1.3pp**.

Best performing configuration: Gemini CLI + Gemini 3 Flash at 48.7% pass rate with skills.

### Domain-Level Heterogeneity

| Domain | Without Skills | With Skills | Delta |
|---|---|---|---|
| Healthcare | 34.2% | 86.1% | **+51.9pp** |
| Manufacturing | 1.0% | 42.9% | **+41.9pp** |
| Cybersecurity | 20.8% | 44.0% | +23.2pp |
| Natural Science | 23.1% | 44.9% | +21.9pp |
| Energy | 29.5% | 47.5% | +17.9pp |
| Office Work | 24.7% | 42.5% | +17.8pp |
| Finance | 12.5% | 27.6% | +15.1pp |
| Media Production | 23.8% | 37.6% | +13.9pp |
| Robotics | 20.0% | 27.0% | +7.0pp |
| Mathematics | 41.3% | 47.3% | +6.0pp |
| Software Engineering | 34.4% | 38.9% | +4.5pp |

> [!IMPORTANT]
> Larger gains appear in domains with specialized procedural knowledge likely underrepresented in pretraining data (Healthcare, Manufacturing). Domains where models already have strong priors (Mathematics, Software Engineering) benefit less.

### Skills Quantity Effect (Non-Monotonic)

| Skills Count | Average Improvement |
|---|---|
| 1 skill | +17.8pp |
| 2–3 skills (optimal) | **+18.6pp** |
| 4+ skills | +5.9pp |

Providing too many skills degrades performance, likely due to context budget constraints or conflicting guidance.

### Skills Complexity Effect

| Documentation Style | Pass Rate | Improvement |
|---|---|---|
| Detailed (focused, stepwise) | 42.7% | **+18.8pp** |
| Compact (terse but complete) | 37.6% | +17.1pp |
| Standard | 37.1% | +10.1pp |
| Comprehensive (exhaustive) | 39.9% | −2.9pp |

> [!NOTE]
> Comprehensive documentation produced *negative* improvement on average. Models appear to struggle extracting relevant guidance from lengthy skills, or hit context length limits.

### Self-Generated Skills

Models prompted to generate procedural knowledge before solving tasks achieved −1.3pp on average:

| Configuration | Self-Gen Delta |
|---|---|
| Claude Opus 4.6 | +1.4pp (only positive result) |
| Codex CLI / GPT-5.2 | −5.6pp |
| Average across all | −1.3pp |

This sharp contrast with curated skills (+16.2pp) demonstrates that current models cannot reliably author the domain expertise they benefit from consuming.

### Skills as Model-Scale Substitute

Claude Haiku 4.5 **with** curated skills (27.7%) outperformed Claude Opus 4.5 **without** skills (22.0%), achieving +5.7pp over a 2-tier larger model. This suggests skills partially compensate for model capacity gaps on procedural tasks.

### Task-Level Extremes

**Largest positive deltas** (curated skills):
- `mario-coin-counting`: +85.7pp (2.9% → 88.6%)
- `sales-pivot-analysis`: +85.7pp
- `flood-risk-analysis`: +77.1pp
- `sec-financial-report`: +74.3pp

**Negative deltas** (16 of 84 tasks):
- `taxonomy-tree-merge`: −39.3pp
- `energy-ac-optimal-power-flow`: −14.3pp
- `trend-anomaly-causal-inference`: −12.9pp
- `exoplanet-detection-period`: −11.4pp

Failure cases suggest skills introduced conflicting guidance for tasks models already handled effectively without assistance.

## Algorithms and Skill Design Patterns

### Skill Impact Categories (from case studies)

Four patterns observed in successful skill application:

```
1. API Gap Bridging
   Task: sales-pivot-analysis
   Problem: Agent unfamiliar with domain-specific API
   Skill provides: exact function signatures and call patterns
   Result: +85.7pp

2. Data Processing Pipeline
   Task: flood-risk-analysis
   Problem: Multi-step preprocessing with domain conventions
   Skill provides: ordered pipeline steps with format specs
   Result: +77.1pp

3. Regulatory Knowledge Encoding
   Task: sec-financial-report
   Problem: Domain-specific compliance rules
   Skill provides: applicable regulations and required fields
   Result: +74.3pp

4. Pitfall Prevention
   Task: manufacturing-fjsp-optimization
   Problem: Common implementation errors in optimization
   Skill provides: known failure modes and avoidance strategies
   Result: Large positive delta
```

### Recommended Skill Authoring Pattern (from paper findings)

```
SKILL.md structure:
  1. One-sentence scope statement (which task class this applies to)
  2. Prerequisites (tools, libraries, access)
  3. Step-by-step procedure (numbered, imperative)
  4. One worked example
  5. Known pitfalls (optional, brief)

Avoid:
  - Exhaustive API documentation (use links instead)
  - Task-specific constants or file paths
  - More than 3 skills per task class
```

## Harness-Specific Observations

| Harness | Behavior | Skills Improvement Range |
|---|---|---|
| Claude Code | Highest skills utilization; natively integrates skills format | +13.9pp to +23.3pp |
| Gemini CLI | Highest raw performance; efficient context use | +13.6pp to +17.4pp |
| Codex CLI | Competitive raw performance but frequently neglected skills despite acknowledging content | Not specified |

> [!IMPORTANT]
> Codex CLI acknowledged skill content but often failed to apply it, suggesting harness design (not just model capability) significantly influences skills utilization.

## Infrastructure Errors

164 infrastructure errors recorded across 7,308 trials:

| Error Type | Count |
|---|---|
| VerifierTimeoutError | 83 |
| RuntimeError | 46 |
| AgentSetupTimeoutError | 35 |

## Cost-Performance Analysis

| Configuration | Input Tokens/Task | Cost/Task | Pass Rate |
|---|---|---|---|
| Gemini 3 Flash | 1.08M | $0.55 | **48.7%** |
| Gemini 3 Pro | 0.47M | $0.98 | Lower |

Flash consumed 2.3× more input tokens but used a compensatory exploration strategy. At 4× lower per-token cost, Flash was 44% cheaper per task while achieving better performance—a Pareto-dominant configuration.

# Experiments

- **Datasets**: 84 tasks spanning 11 domains (Healthcare, Manufacturing, Cybersecurity, Natural Science, Energy, Office Work, Finance, Media Production, Robotics, Mathematics, Software Engineering); difficulty stratified as Core (17), Extended (43), Extreme (26)
- **Hardware**: Containerized execution via Harbor framework; infrastructure errors in 164/7,308 trials
- **Models**: Claude Opus 4.5/4.6, Claude Sonnet 4.5, Claude Haiku 4.5 (via Claude Code); Gemini 3 Pro/Flash (via Gemini CLI); GPT-5.2 (via Codex CLI)
- **Sampling**: Temperature 0, 5 trials per task per condition
- **Results**: Curated skills: +16.2pp average improvement (range +13.6pp to +23.3pp); Self-generated skills: −1.3pp average; Best configuration: Gemini CLI + Gemini 3 Flash at 48.7% with skills; Optimal skill count: 2–3 skills per task

## Limitations

- Evaluation covers only terminal-based, containerized tasks; results may not transfer to GUI agents or multi-agent coordination scenarios
- Model and harness set is limited; commercial API behavior and harness integration may evolve
- Injecting skills increases context length; causal attribution requires length-matched controls to isolate procedural content effects from context size effects
- Containerization provides isolation but not complete immunity from training data leakage

## Comparison with Related Work

| Aspect | This Work (SkillsBench) | Prior Agent Benchmarks |
|---|---|---|
| Augmentation type | Procedural knowledge files (skills) | Tool definitions, system prompts, few-shot |
| Evaluation condition | Vanilla vs. curated vs. self-generated | Typically single condition |
| Verification method | Deterministic oracle assertions | Often LLM-as-judge |
| Cross-harness | 3 harnesses, 7 model configs | Usually single framework |
| Scale | 84 tasks, 7,308 trajectories | Varies widely |

> [!TIP]
> The Harbor framework (Merrill et al., 2026) used for containerized evaluation is cited as the infrastructure foundation. SkillsBench builds on it to add skills injection and cross-harness evaluation.
