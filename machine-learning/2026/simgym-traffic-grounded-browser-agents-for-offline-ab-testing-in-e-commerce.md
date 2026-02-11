# Meta Information

- URL: [SimGym: Traffic-Grounded Browser Agents for Offline A/B Testing in E-Commerce](https://arxiv.org/abs/2602.01443)
- LICENSE: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Reference: Castelo, A., Zanjani Foumani, Z., Fan, A., Koay, K. Y., Malik, V., Zhu, Y., Li, H., Feghhi, M., Uliana, R., Xie, S., Zhang, Z., Ocana Martins, A., Zhao, M., Pelland, F., Faerman, J., LeBlanc, N., Glazer, A., McNamara, A., Wang, L., & Wu, Z. (2026). SimGym: Traffic-Grounded Browser Agents for Offline A/B Testing in E-Commerce. arXiv:2602.01443.

---

# SimGym: Traffic-Grounded Browser Agents for Offline A/B Testing in E-Commerce

## 1. Introduction

Traditional A/B testing in e-commerce requires diverting live traffic to unvalidated UI variants, exposing real customers to potentially harmful changes, and waiting weeks for statistical significance. SimGym addresses these limitations by deploying LLM-based synthetic buyers—modeled on each storefront's actual customer distribution—to pre-test UI variants offline before any real customer is exposed.

> [!IMPORTANT]
> The core contribution is demonstrating **causal predictive validity**: SimGym's agent-predicted outcome shifts align directionally with real human behavioral outcomes at 69% rate and 0.64 Pearson correlation, without any post-training alignment. Prior work validated against task completion or offline benchmarks but never against production human outcomes.

**Who uses this / when / where:**
- E-commerce platform operators (e.g., Shopify merchants) who want to pre-screen UI experiments before live rollout
- Applied when an A/B test is too risky or slow to run against live traffic
- Deployable in any storefront with sufficient clickstream history to cluster buyer sessions

---

## 2. SimGym Platform

### 2.1 Intent and Persona Generation Pipeline

**Input:** Raw clickstream sessions from a storefront's production data
**Output:** $n$ structured agent prompts, each encoding a buyer intent + behavioral persona

The pipeline has six stages:

#### Stage 1: Session-Level Clustering

Session features are extracted along five axes and $z$-score normalized before k-means clustering ($k=5$, kmeans++ initialization):

| Feature Axis | Examples |
|---|---|
| Engagement | session duration, event count |
| Exploration depth | number of product detail views |
| Search behavior | search query count |
| Funnel progression | add-to-cart, checkout, purchase rates |
| Economic value | cart and order values |

Each cluster represents a distinct customer archetype (e.g., quick purchaser, deep browser, deal seeker).

#### Stage 2: Product Preference Extraction

For each cluster, an LLM receives `{shop_metadata, cluster_id, aggregated_product_summaries}` and returns up to 10 product categories and 10 individual products with reasoning. This grounds buyer preferences in the shop's actual catalog.

#### Stage 3: Buyer Intent Generation

Purchase-intent ratio is calibrated to the cluster's observed add-to-cart rate $a^-$

```math
\text{purchase\_count} = \text{clip}(\text{round}(\bar{a} \cdot n),\; 1,\; n-1)
```

where $n$ is the total number of agents for this cluster and $\text{clip}(\cdot, 1, n-1)$ ensures at least one purchaser and one browser. Each agent receives a two-sentence intent:

- Purchaser: *"You are looking for [category]. You are ready to purchase."*
- Browser: *"You are looking for [category]. You are researching options."*

Constraints forbid referencing bundles, sizes, discounts, or UI elements to prevent leaking variant information.

#### Stage 4: Buyer Behavior Aggregation

The $n$ sessions closest to the cluster centroid are selected, and their behavioral statistics are aggregated: session counts, funnel rates, cart/order values, and product browsing histories. This provides empirical grounding for persona construction.

#### Stage 5: Buyer Persona Construction

Five continuous dimensions are computed:

| Dimension | Type | Computation |
|---|---|---|
| Price sensitivity | Behavioral | Normalized gap between max browsed price and avg purchased price, category-aware |
| Exploration depth | Behavioral | 0–1 score from duration + search count + product views; mapped to shallow (0–0.35), moderate (0.35–0.65), deep (0.65–1.0) |
| Premium focus | Values | Keyword ratio: luxury/craftsmanship/prestige terms in browsed vs. purchased products |
| Performance focus | Values | Keyword ratio: durability/reliability/specs terms |
| Ethics focus | Values | Keyword ratio: sustainability/ethical-sourcing terms |

An LLM assigns confidence scores and generates reasoning for each dimension.

#### Stage 6: Prompt Composition

Intents and personas are combined 1:1. Each agent prompt encodes: shopping goal, behavioral profile, values orientation, and product preferences.

**Pseudocode:**

```
IntentPersonaPipeline(sessions, shop, n):
  clusters ← kmeans(z_score(extract_features(sessions)), k=5)
  FOR each cluster c:
    preferences ← LLM_extract_products(shop, c)
    intents ← generate_intents(c.a2c_rate, n, preferences)
    agg_data ← aggregate_n_closest(c.centroid, n)
    persona ← construct_persona(agg_data)
    prompts[c] ← [combine(intent_i, persona_i) for i in 1..n]
  RETURN prompts
```

---

### 2.2 Agent Architecture

**Input per step:** current page accessibility tree + session memory + goal + persona
**Output per step:** structured action (click, type, scroll, navigate, or terminate) + reasoning

#### Web Perception

Agents perceive pages as **accessibility trees**—hierarchical representations of interactive elements with reference IDs mapped to DOM locations. This avoids screenshot-based perception, which requires vision models, and instead provides an unambiguous symbolic interface for action targeting.

> [!NOTE]
> この論文をまとめていての個人的な感想でしかないが、AOMは情報は含まれているものの、人間が確認できる情報と乖離があるためLLMが正確に理解できるかは不明だと感じた。

#### Perceive-Plan-Act Loop

```
AgentSession(goal, persona, url, step_limit, time_limit):
  memory ← {initial_context, goal, persona}
  step ← 0
  WHILE step < step_limit AND elapsed < time_limit:
    page ← get_accessibility_tree(current_url)
    prompt ← build_prompt(goal, persona, memory, page)
    response ← LLM(prompt, schema=structured)  # returns {reasoning, action, continue}
    IF NOT response.continue: BREAK
    result ← execute(response.action, page)
    memory.append({response.reasoning, response.action, result})
    IF detect_loop(memory) OR repeated_actions(memory, threshold=3): BREAK
    IF result.failed AND retries < max_retries:
      retry with error_context appended to prompt
      CONTINUE
    step ← step + 1
  RETURN memory
```

#### Episodic Memory

The full session log—navigation context, step-by-step reasoning, actions, and outcomes—is passed to the LLM at each step. This prevents the agent from repeating failed actions and maintains coherent goal-directed behavior across multi-step journeys.

> [!NOTE]
> "Without session context, over half of agents hit the step limit before reaching a decision, indicating they wander inefficiently rather than progressing toward their goal." (paper)

#### Guardrails

- Infinite loop detection (action sequence hashing)
- Hard step and time limits
- Model retry with error context injection
- Structured JSON output schema enforcement

---

## 3. Ground Truth Generation and Evaluation

### 3.1 Ground Truth Dataset

**Data source:** Real UI experiments (theme switches) from a production e-commerce platform
**Selection criteria:**
1. Sufficient pre/post traffic for reliable Add-to-Cart (A2C) rate estimation
2. Measurable A2C differences between variants

**Confounder removal** (aggressive filtering):
- Removed shops with active promotions, seasonal effects, pricing/assortment changes during experiment window
- Removed ramp-up periods
- Validated with **double machine learning** to confirm consistent treatment effects

**LLM evaluator:** Characterizes the nature of each theme change using visual screenshots + DOM parsing, enabling stratification by change type and magnitude.

**Final dataset:** 2,020 shops across 12 countries and diverse product categories.

### 3.2 Evaluation Metrics

| Metric | Definition |
|---|---|
| **Alignment Rate** | % of shops where agent A2C directional change matches human directional change |
| **Alignment Probability** | Bayesian posterior $P(\text{agent direction} = \text{human direction})$; continuous in $[0, 1]$ |
| **Pearson Correlation** | Correlation of agent-predicted A2C change magnitudes with observed human changes |

---

## 4. Experiments

### 4.1 Agent Sample Size Selection

Bootstrap resampling across $n \in \{50, 100, 150, \ldots, 700\}$ agents per shop, with 1,000 resamples per size:
- Sign alignment increases from ~51% (50 agents) to ~73% (700 agents)
- Pearson correlation stabilizes at ~0.65 after 300 agents, plateaus after 500

**Selected:** 600 agents per shop (above plateau, computationally feasible; total runs $\approx$ 2,400,000 across 2,020 shops $\times$ 2 independent runs).

### 4.2 Ablation Studies

#### Memory Ablation

| Metric | SimGym (full) | w/o Memory |
|---|---|---|
| Pearson Correlation | **0.64** | 0.29 |
| Alignment Rate | **69%** | 55% |
| Alignment Probability | **0.69** | 0.55 |
| Goal Reached | **90%** | 45% |
| Step Limit Hit (Timeout) | 9.59% | **54.93%** |
| Avg Journey Length | 11.7 steps | 13.7 steps |

Without memory, 45.70% of agents get stuck in loops (vs. 8.50%), and 45% never reach the add-to-cart decision stage. The correlation collapses to 0.29 (near random).

> [!NOTE]
>
> 個人的に、なぜメモリなしとありを比較したのか？という疑問がある。過去の行動の履歴を参照しないエージェントがどのように振る舞うかを確認したい、という意図は理解できるが、実際のユーザも過去の行動履歴を参照しないわけではないため、あまり意味のある比較ではないように思える。ただ、当たり前だと思われることも定量的に示すことは重要なので、その意味では価値があるのかもしれない。

#### Persona Ablation

| Configuration | Pearson | Alignment | Align. Prob. | Goal Reached |
|---|---|---|---|---|
| **SimGym (full)** | **0.64** | **69%** | **0.69** | **90%** |
| Intent Only (no persona) | 0.27 | 52% | 0.54 | 79% |
| Generic Persona (donor-derived) | 0.27 | 62% | 0.59 | 90% |

- **Intent Only**: Only 24.84% of agents reach the decision stage (vs. 40.27% for full SimGym), producing random-chance alignment. Without behavioral strategy, agents fail to act like purposeful shoppers.
- **Generic Persona**: Reaches decisions at comparable rate (90% goal completion) but predicts direction without magnitude—correlation 0.27 vs. 0.64—because donor-shop personas misrepresent the target shop's customer distribution.

> [!IMPORTANT]
> Custom, traffic-grounded personas are essential for both directional and magnitude alignment. Generic personas capture behavioral strategy but miss shop-specific customer distribution, yielding shallow predictive slopes (0.18 vs. 0.57).

---

## 5. Comparison with Related Work

| Approach | Validation Method | Persona Grounding | Live Browser |
|---|---|---|---|
| WebShop / ShoppingBench | Offline benchmark accuracy | Generic / none | No |
| AgentA/B | Task completion rate | Generic LLM personas | No |
| PAARS / Shop-R1 | Behavioral similarity to historical data | Historical data | No |
| **SimGym** | **Causal predictive validity vs. real outcomes** | **Shop-specific traffic-derived** | **Yes** |

> [!NOTE]
> The core differentiator is causal validation: prior work validates that agents *behave* like shoppers; SimGym validates that agents *predict the same A/B test outcomes* as real shoppers. This is a stricter and more practically useful bar.

---

## Experiments

- **Dataset:** 2,020 production storefronts with real UI A/B test outcomes (12 countries, diverse industries); confounders removed via double machine learning
- **Hardware:** Not specified; agents execute in live browser environment
- **LLM:** Claude (referenced implicitly; structured JSON output schema-constrained generation)
- **Hyperparameters:** $k=5$ clusters, 600 agents/shop, max 10 product categories, max 10 products per preference extraction, exploration depth thresholds at 0.35 and 0.65
- **Key results:**
  - Full SimGym: 69% directional alignment, 0.64 Pearson correlation, 0.69 alignment probability
  - Memory ablation drops to 55% alignment and 0.29 correlation, with 54.93% timeout rate
  - Intent-only ablation drops to 52% alignment (random chance) and 0.27 correlation
  - Custom personas vs. generic: 0.64 vs. 0.27 correlation; directional slope 0.57 vs. 0.18
