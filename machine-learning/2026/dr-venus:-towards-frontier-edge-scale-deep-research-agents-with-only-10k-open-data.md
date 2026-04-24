# Meta Information

- URL: [DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data](https://arxiv.org/abs/2604.19859)
- LICENSE: [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- Reference: Venus Team, Dai, S., Deng, Y., Lin, J., Song, Y., Wang, G., Wu, X., Zhou, Y., Yang, S., Ying, Z., Zhang, Z., Meng, C., & Wang, W. (2026). DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data. arXiv:2604.19859.

# DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data

## Overview

DR-Venus is a 4-billion-parameter deep research agent designed for resource-constrained (edge-scale) deployment. It solves complex, multi-step information-seeking tasks — such as navigating the web, searching for evidence, and synthesizing answers to hard research questions — without requiring access to large-scale proprietary data or extremely large models.

**Applicability**: Useful when deploying capable research agents under constraints of cost, latency, or data privacy. Targeted at practitioners who need agents comparable to 30B-class systems but must operate within 4B-parameter budget limits.

The central claim is that "careful improvement of training data quality and effective data utilization can compensate for a substantial portion of the scale gap" between small and large language models in agentic tasks.

## Problem Setting

### Task Definition

A deep research agent operates as a multi-turn system. At each turn $t$, the agent observes all prior history $h_{\leq t-1}$ and produces a turn output $u_t = (\tau_t, a_t)$ consisting of a reasoning trace $\tau_t$ and an action $a_t$ (either calling a tool or generating a final answer). The environment returns observation $o_t$ in response to actions.

The full trajectory is:

```math
\begin{align}
  H = (q,\; (u_1, o_1),\; (u_2, o_2),\; \ldots,\; (u_{T-1}, o_{T-1}),\; u_T)
\end{align}
```

where $q$ is the initial query and $u_T$ is the terminal turn with the final answer. The agent uses two tools: `search` (Google API via Serper) and `visit`/`browse` (Jina-based page reader). Trajectories can span up to 200 turns with a maximum sequence length of 256K tokens.

### Why Small Models Struggle

Large models trained on vast agentic data inherently benefit from scale. Small models (≤9B parameters) face two compounding challenges: (1) limited reasoning capacity and (2) scarcity of high-quality long-horizon training trajectories. Naively fine-tuning on available open data wastes long trajectories due to imbalanced sampling and fails to provide dense enough training signal for reinforcement learning.

## Methodology

### Stage 1: Agentic Supervised Fine-Tuning (SFT)

The SFT objective is standard cross-entropy loss over agent-generated tokens:

```math
\begin{align}
  \mathcal{L}_{\text{SFT}}(\theta) = -\sum_{H \in \mathcal{D}_{\text{SFT}}} \sum_{i \in M(H)} \log \pi_\theta(x_i \mid x_{<i})
\end{align}
```

where $M(H)$ is the set of model-generated token positions within trajectory $H$.

The data pipeline processes the REDSearcher open dataset through four sequential steps:

| Step | Operation | Input → Output |
|------|-----------|----------------|
| 1. Environment Alignment | Standardize tool-calling format | 10,001 trajectories → 10,001 |
| 2. Tool Pruning | Remove unsupported/duplicate tool calls | 3,378 disallowed calls removed; 15,728 duplicates removed |
| 3. Correctness Filtering | Retain only trajectories with correct final answers | 10,001 → 9,365 (93.65% pass rate) |
| 4. Turn-Aware Resampling | Upsample longer trajectories | 9,365 → 18,745 effective training examples |

**Turn-Aware Resampling** addresses the imbalance where long-horizon trajectories are underrepresented. Trajectories are weighted by length tier:

- $\leq$50 turns: weight $1\times$
- 51–100 turns: weight $2\times$
- $>$100 turns: weight $5\times$

This increases the proportion of trajectories exceeding 100 turns from 13.29% to 33.21% of training data. Ablation results confirm resampling alone yields +4.0 points on BrowseComp.

### Stage 2: Agentic Reinforcement Learning with IGPO

DR-Venus extends Information Gain-based Policy Optimization (IGPO) with several enhancements tailored to long-horizon research tasks.

#### Information Gain Reward

At each turn $t$ in rollout $i$, the model's log-probability of producing the correct answer $g$ given history up to turn $t$ is estimated by averaging over the $L$ answer tokens:

```math
\begin{align}
  \log \pi_\theta(g \mid h_{i,\leq t}) = \frac{1}{L} \sum_{j=1}^{L} \log \pi_\theta(g_j \mid h_{i,\leq t},\; g_{<j})
\end{align}
```

The turn-level information gain reward then measures how much a single turn's observation improved the model's confidence in the correct answer:

```math
\begin{align}
  r_{i,t}^{\text{IG}} = \log \pi_\theta(g \mid h_{i,\leq t}) - \log \pi_\theta(g \mid h_{i,\leq t-1})
\end{align}
```

> [!NOTE]
> This IG reward is only non-trivially positive when a browse/visit action retrieves genuinely useful information. Search-and-redirect actions that do not immediately change the context contribute near-zero IG.

#### Browse-Aware IG Assignment

Because `search` actions retrieve URLs (not content), meaningful information gain occurs at the subsequent `visit`/`browse` step. Browse-aware assignment propagates the IG reward from a browse turn backward to the immediately preceding search turn, so the search action also receives credit for enabling the evidence retrieval.

#### Format-Aware Penalty

Turn outputs that violate the required JSON-formatted tool-calling schema receive a penalty instead of the IG reward:

```math
\begin{align}
  \hat{r}_{i,t} =
  \begin{cases}
    r_{i,t} & \text{if output format is valid} \\
    -\lambda_{\text{fmt}} & \text{otherwise}
  \end{cases}
\end{align}
```

#### Reward Normalization

Rewards are normalized separately for turn-level IG rewards and the terminal outcome reward to keep gradient magnitudes comparable:

```math
\begin{align}
  \tilde{r}_{i,t} =
  \begin{cases}
    \dfrac{\hat{r}_{i,t}^{\text{IG}} - \mu^{\text{IG}}}{\sigma^{\text{IG}}} & 1 \leq t < T_i \\[6pt]
    \dfrac{\hat{r}_i^{O} - \mu^{O}}{\sigma^{O}} & t = T_i
  \end{cases}
\end{align}
```

where $\mu$ and $\sigma$ are computed across the rollout group.

#### IG-Scale Mechanism

The IG-Scale mechanism dynamically balances the magnitude of turn-level IG rewards relative to the outcome reward, preventing either from dominating training:

```math
\begin{align}
  s = \min\!\left(\frac{\max(M^O,\; \eta)}{M^{\text{IG}} + \delta},\; s_{\max}\right)
\end{align}
```

where $M^O$ is the mean absolute outcome reward, $M^{\text{IG}}$ is the mean absolute IG reward, $\eta = 0.3$, $\delta = 10^{-8}$, and $s_{\max} = 10$. The scaled IG reward is $\bar{r}_{i,t}^{\text{IG}} = s \cdot \tilde{r}_{i,t}^{\text{IG}}$.

#### Discounted Cumulative Reward

Final per-token advantage estimates use a discounted sum of future rewards rather than single-step rewards, encouraging the agent to plan across turns:

```math
\begin{align}
  \tilde{R}_{i,t} = \sum_{k=t}^{T_i} \gamma^{k-t} \bar{r}_{i,k}, \quad \gamma = 0.95
\end{align}
```

#### IGPO Objective

The full policy gradient objective clips importance weights (as in PPO) while adding a KL divergence penalty against a reference policy $\pi_{\text{ref}}$:

```math
\begin{align}
  \mathcal{J}_{\text{IGPO}}(\theta) = \mathbb{E}\!\left[
    \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|u_i|} \sum_{k=1}^{|u_i|}
    \min\!\left(
      \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} \tilde{R}_{i,k},\;
      \text{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\varepsilon, 1+\varepsilon\right) \tilde{R}_{i,k}
    \right)
    - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
  \right]
\end{align}
```

where $G$ is the group size, $\varepsilon$ is the clip ratio, and $\beta$ is the KL coefficient.

### Comparison with GRPO

| Aspect | GRPO | IGPO (DR-Venus) |
|--------|------|-----------------|
| Reward granularity | Outcome-level (terminal only) | Turn-level IG + terminal outcome |
| Credit assignment | Sparse (last turn only) | Dense (every browse turn) |
| Reward balance | N/A | IG-Scale mechanism |
| Future planning | Single-step | Discounted cumulative return |
| Result on BrowseComp vs SFT | −1.5 | +2.3 |

GRPO's performance degrades below SFT on BrowseComp because sparse terminal-only rewards are insufficient to guide long-horizon trajectories (200 turns). Dense IG rewards from IGPO provide a learning signal at every evidence-gathering step.

## Training Infrastructure

- **Backbone model**: Qwen3-4B-Thinking-2507
- **SFT hardware**: 8 × A100 GPUs
- **RL hardware**: 16 × A100 GPUs
- **Maximum sequence length**: 256K tokens
- **Maximum turns per trajectory**: 200

## Experiments

- **Datasets**:
  - SFT: REDSearcher open dataset — 10,001 raw trajectories → 9,365 after filtering → 18,745 after resampling
  - RL: 1,000 English QA pairs from REDSearcher
  - Evaluation: BrowseComp (English web browsing), BrowseComp-ZH (Chinese), GAIA Text-Only (multi-step reasoning), xBenchDS-2505, xBenchDS-2510, DeepSearchQA
- **Hardware**: 8 A100 GPUs (SFT), 16 A100 GPUs (RL)
- **Optimizer**: Standard policy gradient (IGPO variant of PPO)

### Main Results (6-Benchmark Comparison)

| Model | Params | BrowseComp | BrowseComp-ZH | GAIA (Text) | xBench-DS-2505 | xBench-DS-2510 | DeepSearchQA |
|-------|--------|-----------|--------------|-------------|----------------|----------------|-------------|
| DR-Venus-4B-SFT | 4B | 26.8 | 35.7 | 65.4 | 69.0 | 35.3 | 37.7 |
| DR-Venus-4B-RL | 4B | **29.1** | **37.7** | 64.4 | **74.7** | **40.7** | **39.6** |
| AgentCPM-Explore-4B | 4B | 24.1 | 29.1 | 63.9 | 70.0 | 34.0 | 32.8 |
| REDSearcher-30B | 30B | 42.1 | 49.8 | 80.1 | — | — | — |

DR-Venus-4B-RL surpasses all sub-9B baselines and closes the gap to the 30B-class REDSearcher model.

### Ablation Study

| Configuration | BrowseComp | BrowseComp-ZH |
|---------------|-----------|--------------|
| SFT w/o resampling | 22.8 | 33.9 |
| SFT w/ resampling (baseline) | 26.8 (+4.0) | 35.7 (+1.8) |
| RL w/ GRPO | 25.3 (−1.5) | 35.6 (−0.1) |
| RL w/ IGPO | **29.1 (+2.3)** | **37.7 (+2.0)** |

### Pass@K Analysis

At Pass@16, DR-Venus-4B-SFT reaches 78.5% on BrowseComp-ZH — exceeding several 30B-class commercial systems. This reveals a large gap between the model's best-case potential and its greedy-decoding performance, motivating future work on inference-time scaling strategies for small models.

### Tool Usage Analysis

Correct trajectories consistently exhibit higher browse-tool ratios than failed ones, confirming that evidence retrieval (not just search) drives success. RL training further increases the model's browse ratio from 17.49% (SFT) to 22.46% (RL), indicating that IGPO's browse-aware reward successfully teaches the agent to invest more effort in reading retrieved content.

> [!IMPORTANT]
> RL training is conducted exclusively on English data (1K English QA pairs), which may explain a distribution mismatch for larger Pass@K on BrowseComp-ZH. Cross-lingual generalization of RL-trained agents remains an open research question.

## Limitations

- The RL stage uses only English data, which limits cross-lingual credit assignment. Chinese browsing tasks (BrowseComp-ZH) show less consistent improvement at high Pass@K values.
- The Pass@K gap (e.g., 37.7% at Pass@1 vs. 78.5% at Pass@16 on BrowseComp-ZH) suggests the model often has the capability but not the consistency to produce correct answers under greedy decoding.
- Evaluation uses a live web environment (real Google/Jina APIs), so results depend on real-time document availability.

> [!TIP]
> For background on Information Gain-based Policy Optimization (IGPO), see the original IGPO paper (Wang et al., 2026) which DR-Venus directly extends. For the GAIA benchmark used in evaluation, see [Mialon et al., 2023](https://arxiv.org/abs/2311.12983).

## Related Work

DR-Venus positions itself at the intersection of three research threads:

1. **Deep research agents**: Systems like OpenAI Deep Research, Perplexity, and Tongyi-DeepResearch demonstrate that iterative search-and-browse pipelines can solve hard information-seeking tasks, but typically rely on 30B+ parameters or proprietary data.
2. **Small language model agents**: Prior work (AgentCPM-Explore, REDSearcher) shows 4–9B models can operate as agents but lag behind large models on multi-hop research tasks.
3. **Reinforcement learning for agents**: GRPO (from DeepSeekMath) provides a standard RL baseline, but its terminal-only reward is poorly suited to long-horizon tasks. IGPO addresses this with turn-level dense rewards.

## References

- Wang et al. (2026). IGPO: Information Gain-based Policy Optimization.
- Chu et al. (2026). REDSearcher. Hugging Face (Zchu).
- Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. (GRPO origin)
- Wei et al. (2025). BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents.
- Zhou et al. (2025). BrowseComp-ZH.
- Mialon et al. (2023). GAIA: A Benchmark for General AI Assistants.
- Chen et al. (2025). xBench-DeepSearch (xBenchDS-2505, xBenchDS-2510).
- Gupta et al. (2026). DeepSearchQA.
- Tongyi-DeepResearch Team (2025). Tongyi-DeepResearch.
