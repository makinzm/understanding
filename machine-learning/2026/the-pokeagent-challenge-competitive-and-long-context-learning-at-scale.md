# Meta Information

- URL: [The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](https://arxiv.org/abs/2603.15563)
- LICENSE: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Reference: Karten, S., Grigsby, J., Upaa Jr, T., Bae, J., et al. (2026). The PokeAgent Challenge: Competitive and Long-Context Learning at Scale. NeurIPS 2025 Competition.

# Overview

The PokeAgent Challenge is a standardized benchmark that uses the Pokemon franchise to evaluate frontier AI capabilities across three dimensions that remain open problems: **partial observability**, **adversarial (game-theoretic) reasoning**, and **long-horizon planning**. The benchmark comprises two tracks run as a NeurIPS 2025 competition with 100+ participating teams:

- **Battling Track**: Competitive two-player Pokemon battles in Pokemon Showdown (Gen 1 OU and Gen 9 OU rulesets)
- **Speedrunning Track**: Long-horizon RPG completion of Pokemon Emerald, targeting key milestones up to defeating the first gym leader (Roxanne)

A key contribution is the **Harness vs. Model Attribution Framework** — a five-dimensional analysis (state representation, tools, memory, feedback, fine-tuning) that disambiguates an agent's underlying model capability from its surrounding infrastructure. This framework reveals that LLM performance depends heavily on harness quality, not just the base model.

# Battling Track

## Environment Design

Pokemon Showdown is a two-player, zero-sum, stochastic game with imperfect information and simultaneous action selection. Its properties make it a rigorous strategic reasoning benchmark:

| Property | Value |
|---|---|
| Available actions per turn | ~9 |
| Typical battle length | 20–100 turns |
| Partial observability | Hidden team composition |
| Gen 9 OU state space | ~$10^{564}$ possible configurations |

Two rulesets are evaluated:

- **Gen 1 OU**: More compact state space, greater effective hidden information, smaller human demonstration datasets
- **Gen 9 OU**: Larger demonstration datasets, modern complexity with items, abilities, and Terastallization

Both formats have two timer variants: standard rules and **Extended Timer** (designed to give LLMs sufficient deliberation time for fair evaluation against RL agents).

## Datasets

| Dataset | Size |
|---|---|
| Human battle demonstrations | 4M trajectories (reconstructed from Showdown spectator replays to player perspective) |
| Synthetic self-play trajectories | 18M |
| NeurIPS 2025 competition battles | 100K+ |
| **Total** | **22M+ battles** |
| Competitive team configurations | 200K+ (inferred from replay analysis + expert-validated curated teams) |

## Evaluation Metrics

**Primary metric**: Full-History Bradley–Terry (FH-BT) rating, bootstrapped with uncertainty estimates across complete battle histories.

> [!NOTE]
> Glicko-1 was considered but rejected as too noisy for the dense, fixed-policy agent pool used in this benchmark. FH-BT is fit across the complete battle history rather than using online updates.

Minimum 250 games per agent required for statistical validity (enforced at the qualifying stage). **GXE scores** (Pokemon Showdown's public ladder rating) are used for cross-model correlation analysis against the BenchPress benchmark suite.

## Baselines

**LLM Baselines**: Extended PokéChamp framework supports frontier models (GPT, Claude, Gemini) and open-source models (Llama, Gemma, Qwen). Uses **depth-limited minimax search** with LLM-based position evaluation. Extended Timer is required for fair evaluation. Even small open-source models achieve meaningful performance when given a well-designed harness.

**RL Baselines**: Extended Metamon implementation with 30 agent checkpoints ranging from compact RNNs to 200M-parameter Transformers, trained on millions of human demonstrations and self-play battles. The strongest RL baselines are competitive with top human players on Pokemon Showdown's public ladder.

# Speedrunning Track

## Task Formulation

The speedrunning task is formalized as an episodic Markov Decision Process:

```math
\begin{align}
  \mathcal{M} = (\mathcal{S}, \mathcal{A}, T, R, \gamma)
\end{align}
```

- **Actions** $\mathcal{A}$: raw button inputs (A, B, directional pad, etc.)
- **Transitions** $T$: largely deterministic for navigation, stochastic for battles
- **Reward** $R$: $+1$ per milestone achieved, $\gamma = 1$ (no discounting)
- **Scope**: 15 standardized milestones from Littleroot Town through defeating Roxanne (first gym leader)
- **Scale**: Thousands of agent steps, millions of reasoning tokens

## State Space and Partial Observability

The agent receives:
- Visual frames (pixel input)
- Party composition: species, levels, status conditions, HP values

**Hidden from the agent**:
- Puzzle states and dynamic obstacle positions
- Detailed item inventories
- Opponent movesets

This partial observability makes the task significantly harder than it appears from a linear walkthrough perspective: even "straightforward" routes require substantial exploration and backtracking.

## Evaluation Criteria

| Metric | Description |
|---|---|
| Completion percentage | Progress through 15 standardized milestones |
| Completion time | For agents achieving 100%; lower is better |
| Tie-breaking | Action count (efficiency measure) |

**Human baselines:**
- Top speedrunner: 18 minutes
- Average human: 1:22:05

## PokéAgent Baseline System (Organizer Baseline)

The paper introduces the first open-source multi-agent orchestration system for long-horizon RPG play:

**Architecture components:**

| Component | Role |
|---|---|
| MCP tools | A* pathfinding, button inputs, knowledge retrieval |
| Battle strategy sub-agent | Handles in-game battles |
| Self-reflection sub-agent | Detects and corrects mistakes |
| Gym puzzle sub-agent | Solves specialized puzzle segments |
| Objective verification sub-agent | Confirms milestone completion |
| Central orchestrator | Maintains high-level route plan, dispatches sub-agents, implements automatic context compaction |

**Frontier model performance on speedrunning (organizer baseline):**

| Model | Mean Completion Time | Notes |
|---|---|---|
| Gemini 3 Flash | ~2:24 | Fastest mean; more actions than Pro |
| Claude Sonnet 4.5 | 6:25–20:45 (high variance) | Completes all milestones; 3–4× cost of Gemini variants |
| GPT-5.2 | Intermediate | — |
| Best organizer baseline | ~1.8× average human | Compared to 1:22:05 human average |
| Raw frontier VLMs (no harness) | ~0% completion | Fails to maintain coherence over thousands of sequential decisions |
| Coding-agent architectures (Claude Code, Codex CLI, Gemini CLI) | ~0% completion | Fail similarly without specialized orchestration |

> [!IMPORTANT]
> Without the multi-agent harness, even frontier VLMs achieve effectively 0% task completion. This demonstrates that the harness infrastructure — not just the base model — is the critical differentiator for long-horizon sequential decision-making.

# NeurIPS 2025 Competition Results

## Participation

- 100+ active teams, 650+ Discord community members
- 100K+ total battles on competition server
- 22 valid Speedrunning submissions; 6 teams achieving 100% completion

## Battling Track Results

Top 8 teams in each format qualified for head-to-head tournament brackets.

**Gen 1 OU**: Winner: PA-Agent; Finalist: 4thLesson. 13 of 16 qualifying teams extended the public RL baselines; 3 used independent approaches (Porygon2AI, FoulPlay).

**Gen 9 OU**: Winner: FoulPlay — used **root-parallelized MCTS** in an imperfect-information setting; Finalist: Team Q.

## Speedrunning Track Results

| Place | Team | Time | Method |
|---|---|---|---|
| 1st | Heatz | 40:13 | Scripted Policy Distillation (SPD) |
| 2nd | Hamburg PokéRunners | 1:19:47 | Pure RL (recurrent PPO, milestone-conditioned rewards) |
| Judge's Choice | Deepest | N/A (5th by time) | Fewest total actions (649) |
| Best pure-LLM harness | anthonys | 1:29:17 | LLM-based harness |

**Heatz's Scripted Policy Distillation (winning approach):**

1. An LLM decomposes the task into subgoals and generates scripted policies for each subgoal
2. Neural networks distill the scripted policies via imitation learning
3. RL refines the resulting agent

This hybrid approach is 2× faster than the best pure-LLM harness (1:29:17) and 2.2× slower than the best human speedrunner (18:00).

# Cross-Track Analysis

## Specialist vs. Generalist Methods

Both tracks show the same pattern: **RL and search methods outperform LLM reasoning alone**. Top battling participants used RL or MCTS; top speedrunning finishers used RL-based methods. This consistent finding across two very different tasks strengthens the generalizability of the conclusion.

## LLM Reasoning Failure Modes Under Adversarial Pressure

Weaker models exhibit "panic behavior": small tactical errors lead to compounded mistakes. Failure modes categorized:

- **Memory corruption cascades**: Accumulated context errors propagate across turns
- **Goal oscillation**: Switching between objectives without completing either
- **Excessive plan commitment**: Persisting with a failing plan despite negative feedback
- **Computational paralysis**: Inability to resolve ambiguous multi-step decisions

> [!NOTE]
> These failure modes are absent from standard coding and math benchmarks but appear consistently under multi-turn adversarial pressure, highlighting a gap between benchmark performance and real-world strategic reasoning.

## Orthogonality to Standard Benchmarks

Analysis against BenchPress (83 models × 49 benchmarks):

| Metric | Value |
|---|---|
| Maximum Spearman $\rho$ with GXE | 0.77 |
| Mean absolute Spearman $\rho$ | 0.45 |
| Variance explained by rank-2 SVD (standard benchmarks) | 91% |
| Variance of GXE captured by same rank-2 SVD | 27% |

Standard benchmarks are nearly rank-2 (highly correlated); competitive Pokemon GXE scores are largely orthogonal to this low-rank structure. This means Pokemon battling measures strategic reasoning capabilities that are not captured by existing evaluation suites.

# Comparison with Similar Benchmarks

| Aspect | PokeAgent Challenge | Standard LLM Benchmarks (e.g., MMLU, HumanEval) | Game-playing benchmarks (e.g., Atari, StarCraft) |
|---|---|---|---|
| Partial observability | Yes (hidden teams, puzzle states) | No | Varies |
| Adversarial reasoning | Yes (opponent adaptation) | No | Yes (StarCraft) |
| Long-horizon planning | Yes (thousands of steps) | No | Partial |
| Natural language interface | Yes (LLM harness) | Yes | Rarely |
| Living leaderboard | Yes | Rarely | Occasionally |
| Human baseline quality | Elite speedrunners (18 min) and ladder players | Crowdsourced annotators | Professional players |
| Correlation with standard benchmarks | Low ($\rho < 0.45$ mean) | — | Unknown |

# Experiments

- **Datasets**: 22M+ Pokemon Showdown battle trajectories (4M human + 18M synthetic self-play); 200K+ competitive team configurations; NeurIPS 2025 competition battles (100K+)
- **Battling hardware**: Distributed Pokemon Showdown servers (organizer-hosted competition server)
- **Speedrunning hardware**: Standardized Game Boy Advance emulator environment (self-contained, reproducible locally)
- **Baselines**: PokéChamp LLM framework (GPT, Claude, Gemini, Llama, Gemma, Qwen); Metamon RL (RNN to 200M-parameter Transformer, trained on human demonstrations + self-play); Metamon RL strongest checkpoints competitive with top human ladder players
- **Key results**:
  - Best battling agent (FoulPlay, MCTS) wins Gen 9 OU tournament
  - Best speedrunning agent (Heatz, SPD) completes in 40:13 vs. 18:00 human record (2.2× slower)
  - Raw frontier VLMs achieve ~0% speedrunning completion without harness
  - Pokemon GXE scores explain only 27% of variance captured by standard benchmark rank-2 SVD (near-orthogonal to existing benchmarks)

# Open Challenges

1. **VLM-SLAM**: Speedrunning agents struggle with spatial localization, action-distance estimation, and objective detection. Grounding VLM outputs to consistent spatial representations remains a bottleneck.
2. **LLM–RL Gap in Battling**: Specialist RL agents far outpace harness-LLM agents. Effective hybrid approaches are unsolved.
3. **Full-Game Completion with Open-Source Models**: No open-source model has completed the full RPG. Proprietary models require heavy harness support.
4. **Approaching Human Speedrun Times**: Best agent (40:13) is 2.2× slower than the best human speedrunner (18:00). Advances needed in navigation efficiency, obstacle avoidance, and objective sequencing.
