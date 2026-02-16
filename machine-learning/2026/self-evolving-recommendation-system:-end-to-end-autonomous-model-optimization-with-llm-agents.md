# Meta Information

- URL: [Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents](https://arxiv.org/abs/2602.10226)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Wang, H., Wu, Y., Chang, D., Wei, L., & Heldt, L. (2026). Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents. arXiv:2602.10226.

# Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents

## Overview

This paper presents an autonomous machine learning engineering system deployed at YouTube that uses Large Language Models (Gemini 2.5 Pro) to independently generate, train, and deploy improvements to production recommendation models. The system eliminates manual ML engineering from the critical path by closing a dual feedback loop: a high-frequency offline agent that proposes changes validated against proxy metrics, and a low-frequency online agent that validates candidates against live north star business metrics via A/B testing.

The system is applicable to large-scale industrial recommender systems where (a) the optimization objective is non-differentiable (e.g., long-term user satisfaction), (b) the experiment cycle is long and costly, and (c) human engineering bandwidth is a limiting factor. Engineers at internet-scale companies dealing with ranking model maintenance will benefit most.

## Problem Formulation

The core challenge is a **bi-level optimization problem** bridging the "alignment gap" between differentiable proxy losses and true north star business metrics.

**Lower Level — Model Training:**

$$\theta^*(\Phi) = \arg\min_\theta \mathcal{L}_{\text{proxy}}(\mathcal{D}; \theta, \Phi)$$

- $\theta \in \mathbb{R}^d$ — model parameters (ranking model weights)
- $\mathcal{D}$ — training dataset
- $\Phi$ — system meta-configuration (optimizer choice, model architecture, reward function)
- $\mathcal{L}_{\text{proxy}}$ — differentiable proxy loss (e.g., cross-entropy on click labels)

**Upper Level — Business Metric Optimization:**

$$\Phi^* = \arg\max_\Phi \mathbb{E}[M(\theta^*(\Phi))] \quad \text{s.t.} \quad G(\Phi) \leq C$$

- $M$ — non-differentiable north star metrics measured via live A/B experiments
- $G(\Phi) \leq C$ — system constraints (e.g., latency, memory budget)

The upper level cannot be optimized by gradient descent because $M$ is measured only in live traffic, with a delay of days to weeks per experiment.

## System Architecture

The system uses two nested feedback loops, each handled by a separate LLM-driven agent.

```
┌─────────────────────────────────────────────────────┐
│                   Online Agent (Outer Loop)          │
│  [Proposal Selection → Validation → Training →       │
│   Live A/B → Metric Synthesis → Experiment Journal] │
│                         ↑                            │
│                 Candidates (Φ candidates)            │
│                         │                            │
│  ┌──────────────────────┴──────────────────────────┐ │
│  │             Offline Agent (Inner Loop)           │ │
│  │   [Hypothesis → Code → Verify → Tool-call]      │ │
│  │   Personas: Optimizer | Architecture | Reward   │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Offline Agent (Inner Loop)

The offline agent generates configuration candidates at high frequency (targeting hundreds per week) using cheap proxy signals (loss computation, SQL correlation queries) without requiring full production training runs.

**Three Specialized Personas:**

| Persona | Optimization Target | Validation Tool | Proxy Metric |
|---|---|---|---|
| Optimizer | Training algorithm & hyperparameters | `compute_loss` | Minimize $\mathcal{L}_{\text{proxy}}$ |
| Architecture | Neural network topology mutations | `compute_loss` | Minimize $\mathcal{L}_{\text{proxy}}$ |
| Reward | Label logic & multi-objective reward composition | `run_sql_query` | Signal correlation with north star |

**Think-Code-Verify Cycle (pseudocode):**

```
for each iteration:
  1. THINK: Generate hypothesis based on Experiment Journal + persona framing
  2. CODE:  Produce precise configuration diff (delta-based, not full replacement)
  3. VERIFY: A secondary "linter" LLM persona checks syntax and schema validity
  4. TOOL:  invoke compute_loss(Φ_candidate) or run_sql_query(label_logic)
  5. if proxy metric improves:
       submit Φ_candidate to Online Agent queue
  6. append (Φ_candidate, outcome, diff) to Experiment Journal
```

> [!IMPORTANT]
> Delta-based configuration generation (producing only the changed fields as a diff) is critical. Full configuration regeneration leads to hallucination errors where the LLM drops unchanged fields or invents new ones.

### Online Agent (Outer Loop)

The online agent orchestrates the full production lifecycle as a five-phase directed acyclic graph (DAG):

| Phase | State | Function |
|---|---|---|
| I | PROPOSED | Accept candidates; maintain FIFO queue from offline agent |
| II | VALIDATED | Static analysis, compilation checks, baseline metric assessment |
| III | TRAINING | Monitor full model training run; verify weight export success |
| IV | LIVE | Divert a fraction of traffic; monitor metrics; abort if safety bounds exceeded |
| V | COMPLETED | Retrieve north star metrics from A/B system; serialize result to Experiment Journal |

The Experiment Journal is shared between both loops. It stores structured records containing:
- Configuration diff $\Delta\Phi$
- Proxy metric outcome (from offline evaluation)
- North star metric outcome (from online A/B, if completed)
- Timestamp and surface identifier

## Context Engineering

Each agent invocation constructs a prompt with the following components in order:

1. **Persona framing** — "You are an expert MLE specializing in [optimizer/architecture/reward] for large-scale recommendation systems."
2. **Primary objectives** — metric priorities, exploration vs. exploitation balance
3. **Steering instructions** — optional human-in-the-loop directives for specific research directions
4. **Safety guardrails** — explicit numeric constraint thresholds (e.g., "Metric#3 must not increase beyond +1%")
5. **Baseline configuration schema** — current production configuration with field descriptions
6. **Experiment Journal** — full sorted history of past experiments with diffs and outcomes

> [!NOTE]
> "Providing the full, sorted history from Experiment Journal is better than no history, restricted top-k, or unsorted history." The agent relies on trajectory context to avoid re-exploring failed directions and to build compound improvements.

## Key Technical Differences from Related Work

| Aspect | Neural Architecture Search (NAS) | AutoML / HPO | This Work |
|---|---|---|---|
| Search space | Discrete architecture graph | Hyperparameter grid | Open-ended: code diffs |
| Objective | Proxy metric (val accuracy) | Proxy metric | North star (live A/B) |
| Proposal mechanism | Gradient / evolutionary | Bayesian optimization | LLM + Experiment Journal |
| Human involvement | High (space design) | Medium | Minimal (guardrails only) |
| Scope | Architecture only | Parameters only | Architecture + optimizer + reward |

The critical novelty is that the LLM operates on the **semantic level** (proposing novel algorithmic components) rather than searching a pre-defined space, and validates against **non-differentiable business metrics** in production.

## Experiments

- **Dataset:** Production YouTube recommendation system training data (petabyte-scale interaction logs; exact size not disclosed)
- **Hardware:** Google production training infrastructure (TPU clusters; details not disclosed)
- **Base model:** YouTube ranking model (DNN-based; architecture details proprietary)
- **LLM:** Gemini 2.5 Pro (primary); Gemini 2.5 Flash (ablation comparison)
- **Optimizer:** Discovered by the system (started from Adagrad baseline, transitioned to RMSprop)

**Key Quantitative Results (production A/B tests, 95% confidence where marked with *):**

| Discovery | YouTube Metric | Surface Metric |
|---|---|---|
| RMSprop optimizer (from Adagrad) | +0.06%* | +0.12%* |
| Training efficiency (4–8× speedup) | −0.01% | +0.06% |
| Gated Path architecture (GLU-like gates) | +0.06%* | +0.14%* |
| Activation function refinement | −0.02% | +0.12%* |
| Multi-objective reward synthesis | +0.03%* | +0.13%* |

**Velocity comparison:**

| Workflow | Experiments per Week |
|---|---|
| Human engineering | $\Theta(1)$–$\Theta(10)$ |
| Self-Evolving System | $\Theta(100)$ |

## Ablation Studies

- **Gemini 2.5 Pro vs. Flash:** Pro achieves consistently lower proxy loss on held-out proposal tasks, confirming that stronger reasoning capability directly improves discovery quality.
- **Context history:** Full sorted Experiment Journal > top-k history > unsorted history > no history. Agents without history repeat failed attempts and fail to build on prior discoveries.
- **Persona framing:** Explicit expert persona framing significantly improves proposal relevance vs. generic prompting.
- **Delta vs. full config generation:** Delta-based generation reduces configuration errors from hallucination; full regeneration fails at higher rates.

## Lessons Learned

1. **Diversity requires explicit prompting.** Without instructions to explore diverse hypotheses, the agent defaults to incremental numerical tuning rather than structural innovations.
2. **Cold-start problem.** The system performs poorly in the initial cycles before sufficient Experiment Journal history accumulates. Warm-starting with a set of pre-run human experiments accelerates early convergence.
3. **Reward persona requires different validation.** Unlike optimizer/architecture changes, reward modifications shift the loss landscape itself, making $\mathcal{L}_{\text{proxy}}$ decrease an invalid signal. Correlation analysis between the modified label and north star metrics is used instead.
4. **Structural > numerical.** GLU gating and multi-objective reward synthesis yielded the largest cumulative gains, exceeding all hyperparameter tuning discoveries combined.
5. **Cross-surface transfer.** Architectural improvements discovered on one YouTube surface transferred successfully to other surfaces with minimal re-engineering.
