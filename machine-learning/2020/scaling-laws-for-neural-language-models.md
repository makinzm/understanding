# Meta Information

- URL: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.

# Introduction

This paper establishes empirical power-law relationships governing the performance of autoregressive Transformer language models as a function of model size $N$ (non-embedding parameters), dataset size $D$ (tokens), and compute budget $C$ (FLOPs). The central finding is that test loss $L$ follows smooth power laws spanning more than seven orders of magnitude in each variable, with each law holding independently when the other factors are not bottlenecks. These scaling laws allow practitioners to predict the performance of large models from small-scale experiments and to allocate compute budgets optimally.

> [!NOTE]
> "Performance depends strongly on scale, weakly on model shape: The most important factors determining model quality are the number of model parameters N (excluding embeddings) and the amount of compute C used for training."

The results challenge the intuition that model depth, width, or number of attention heads are critical architectural choices—when total parameter count is held fixed, varying these produces only a few percent variation in loss.

## Seven Key Empirical Findings

1. **Performance depends strongly on scale, weakly on model shape** — Architecture details (depth vs. width, number of attention heads) have minimal effects when total non-embedding parameter count $N$ is held fixed.
2. **Smooth power laws** — Loss scales as a power law across six orders of magnitude in $N$, $D$, and $C$ without plateaus or kinks.
3. **Universality of overfitting** — Performance penalty depends on the ratio $N^{0.74}/D$; overfitting begins predictably once $D < (5 \times 10^3) \cdot N^{0.74}$.
4. **Universality of training** — Training curves follow predictable power-law forms across model sizes.
5. **Transfer improves with test performance** — Cross-distribution generalization improves smoothly with in-distribution validation loss.
6. **Large models are more sample-efficient** — Larger models reach the same performance with fewer training examples and optimization steps.
7. **Convergence is inefficient** — Compute-optimal training stops significantly before convergence; allocating more compute to larger models is better than training smaller models longer.

# Background and Problem Setting

## Language Modeling Setup

The task is autoregressive language modeling: predict the next token given the preceding context. The model is trained to minimize cross-entropy loss (measured in nats/token) over a fixed vocabulary of 50,257 byte-pair encoded tokens.

**Input:** A sequence of tokens $x_1, x_2, \ldots, x_T$ where each $x_i \in \{0, \ldots, 50256\}$

**Output:** A probability distribution over the next token; the loss $L = -\frac{1}{T}\sum_{t=1}^{T} \log p(x_t | x_{< t})$ in nats/token

## Transformer Architecture

All experiments use decoder-only Transformers (same family as GPT-2). Key parameters:

| Symbol | Meaning |
|--------|---------|
| $n_\text{layer}$ | Number of transformer layers |
| $d_\text{model}$ | Residual stream dimension |
| $d_\text{ff} = 4 d_\text{model}$ | Feed-forward width (standard ratio) |
| $d_\text{attn} = d_\text{model}$ | Attention dimension (standard ratio) |
| $n_\text{heads}$ | Number of attention heads |
| $n_\text{ctx}$ | Context length |

The parameter count (excluding embeddings) is:

$$N \approx 12 \cdot n_\text{layer} \cdot d_\text{model}^2$$

This approximation holds under the standard ratio $d_\text{ff} = 4 \cdot d_\text{attn} = 4 \cdot d_\text{model}$. The embedding parameters ($n_\text{vocab} \times d_\text{model}$ and $n_\text{ctx} \times d_\text{model}$) are excluded from $N$ because they do not follow the same scaling laws.

**FLOPs per token** (broken down by operation):

| Operation | Parameters | FLOPs per Token |
|-----------|-----------|-----------------|
| Embedding | $(n_\text{vocab} + n_\text{ctx}) d_\text{model}$ | $4 d_\text{model}$ |
| Attention QKV | $n_\text{layer} \cdot d_\text{model} \cdot 3 d_\text{attn}$ | $2 n_\text{layer} \cdot d_\text{model} \cdot 3 d_\text{attn}$ |
| Attention Mask | — | $2 n_\text{layer} \cdot n_\text{ctx} \cdot d_\text{attn}$ |
| Attention Project | $n_\text{layer} \cdot d_\text{attn} \cdot d_\text{model}$ | $2 n_\text{layer} \cdot d_\text{attn} \cdot d_\text{model}$ |
| Feed-Forward | $n_\text{layer} \cdot 2 d_\text{model} \cdot d_\text{ff}$ | $2 n_\text{layer} \cdot 2 d_\text{model} \cdot d_\text{ff}$ |
| De-embed | — | $2 d_\text{model} \cdot n_\text{vocab}$ |
| **Total (non-embedding)** | $N = 2 d_\text{model} n_\text{layer}(2 d_\text{attn} + d_\text{ff})$ | $C_\text{forward} \approx 2N + 2 n_\text{layer} n_\text{ctx} d_\text{attn}$ |

For training (forward + backward passes), total compute per token is approximately $6N$ FLOPs, so a training run over $T$ tokens uses:

$$C \approx 6NBS \approx 6NT \text{ FLOPs}$$

where $B$ is batch size and $S$ is the number of gradient steps.

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $L$ | Cross-entropy loss in nats/token |
| $N$ | Non-embedding model parameters |
| $C \approx 6NBS$ | Total non-embedding compute (PF-days) |
| $D$ | Dataset size in tokens |
| $B_\text{crit}$ | Critical batch size |
| $C_\text{min}$ | Minimum compute to reach a given loss |
| $S_\text{min}$ | Minimum number of training steps |
| $\alpha_X$ | Power-law exponents for variable $X$ |

# Scaling Laws

## Individual Power Laws

When the other variables are not limiting, each of $N$, $D$, and $C$ independently determines performance via a power law:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13} \text{ tokens}$$

$$L(C_\text{min}) = \left(\frac{C_c^\text{min}}{C_\text{min}}\right)^{\alpha_{C_\text{min}}}, \quad \alpha_{C_\text{min}} \approx 0.050, \quad C_c^\text{min} \approx 3.1 \times 10^8 \text{ PF-days}$$

Here $C_\text{min}$ denotes the minimum compute required to reach a given loss (i.e., training optimally sized models at the critical batch size).

**Full table of empirical fitted values:**

| Exponent | Value | Scale | Value |
|----------|-------|-------|-------|
| $\alpha_N$ | 0.076 | $N_c$ | $8.8 \times 10^{13}$ params |
| $\alpha_D$ | 0.095 | $D_c$ | $5.4 \times 10^{13}$ tokens |
| $\alpha_C$ (naive) | 0.057 | $C_c$ | $1.6 \times 10^7$ PF-days |
| $\alpha_{C_\text{min}}$ | 0.050 | $C_c^\text{min}$ | $3.1 \times 10^8$ PF-days |
| $\alpha_B$ | 0.21 | $B^*$ | $2.1 \times 10^8$ tokens |
| $\alpha_S$ | 0.76 | $S_c$ | $2.1 \times 10^3$ steps |

## Joint N and D Dependence (Overfitting)

When both $N$ and $D$ are finite, the loss is described by:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

This formula captures the overfitting behavior: for a fixed $N$, performance stops improving when $D \lesssim D_c \cdot (N/N_c)^{\alpha_N/\alpha_D}$. To avoid overfitting, the dataset should satisfy:

$$D \gtrsim (5 \times 10^3) \cdot N^{0.74} \text{ tokens}$$

> [!IMPORTANT]
> This means dataset size should grow as $D \propto N^{0.74}$, which is sub-linear — doubling parameters requires less than double the data to avoid overfitting.

The overfitting measure is defined as:

$$\delta_L(N, D) \equiv \frac{L(N, D)}{L(N, \infty)} - 1 \approx \left[1 + \left(\frac{N}{N_c}\right)^{\alpha_N/\alpha_D} \cdot \frac{D_c}{D}\right]^{\alpha_D} - 1$$

## Training Dynamics and the N–Steps Law

For a model of size $N$ trained for $S_\text{min}$ gradient steps (at the optimal batch size), the loss trajectory follows:

$$L(N, S_\text{min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_\text{min}}\right)^{\alpha_S}, \quad \alpha_S \approx 0.76, \quad S_c \approx 2.1 \times 10^3$$

Training curves are power-law in steps, and this formula allows early-stopping prediction: training a large model for few steps can converge to the same loss as a smaller model trained to convergence.

The lower bound on when early stopping is beneficial:

$$S_\text{stop}(N, D) \geq \frac{S_c}{\left[L(N, D) - L(N, \infty)\right]^{1/\alpha_S}}$$

# Compute-Optimal Scaling

## Critical Batch Size

The critical batch size $B_\text{crit}$ is the batch size that maximizes compute efficiency (balancing time and compute). It scales with loss as:

$$B_\text{crit}(L) = \frac{B^*}{L^{1/\alpha_B}}, \quad B^* \approx 2 \times 10^8 \text{ tokens}, \quad \alpha_B \approx 0.21$$

Training at $B \ll B_\text{crit}$ wastes compute (too many serial steps); training at $B \gg B_\text{crit}$ wastes data (poor sample efficiency). At the critical batch size, the time-compute tradeoff satisfies:

$$\left(\frac{S}{S_\text{min}} - 1\right) \cdot \left(\frac{E}{E_\text{min}} - 1\right) = 1$$

where $E = B \cdot S$ is total tokens seen and $E_\text{min} = B_\text{crit} \cdot S_\text{min}$.

The effective minimum steps after accounting for batch size:

$$S_\text{min}(S, B) \equiv \frac{S}{1 + B_\text{crit}(L)/B}, \quad C_\text{min}(C, B) \equiv \frac{C}{1 + B/B_\text{crit}(L)}$$

## Optimal Allocation Under Fixed Compute Budget

Given a total compute budget $C$, the compute-optimal strategy allocates resources as:

| Resource | Optimal Scaling | Exponent | Scale |
|---|---|---|---|
| Model size $N$ | $\propto C^{0.73}$ | $p_N = 0.73$ | $N_e = 1.3 \times 10^9$ params |
| Batch size $B$ | $\propto C^{0.24}$ | $p_B = 0.24$ | $B_e = 2.0 \times 10^6$ tokens |
| Training steps $S$ | $\propto C^{0.03}$ | $p_S = 0.03$ | $S_e = 5.4 \times 10^3$ steps |
| Dataset tokens $D = B \cdot S$ | $\propto C^{0.27}$ | $p_D = 0.27$ | $D_e = 2 \times 10^{10}$ tokens |

The exponents sum to 1 ($0.73 + 0.24 + 0.03 = 1.00$). The dominant finding is that **most additional compute should go into larger models**, not more data or more training steps.

The composition of exponents satisfies:

$$\alpha_{C_\text{min}} = \frac{1}{1/\alpha_S + 1/\alpha_B + 1/\alpha_N} \approx \frac{1}{1.32 + 4.76 + 13.16} \approx 0.054$$

> [!NOTE]
> "We should therefore prioritize model size in setting up efficient training runs. Concretely, for every 10× increase in compute, model size should increase by about 5.5×, with modest increases in data and minimal increases in serial training time."

## Pseudocode: Compute-Optimal Model Selection

```
Input:  C_target  — compute budget in PF-days
        alpha_C   ≈ 0.050, alpha_N ≈ 0.076, alpha_B ≈ 0.21, alpha_S ≈ 0.76
        Fitted constants: N_e = 1.3e9, D_e = 2e10, B_e = 2e6, S_e = 5.4e3

Step 1: Predict target loss
        L* = (C_c_min / C_target)^alpha_C    [C_c_min = 3.1e8 PF-days]

Step 2: Compute optimal model size
        N* = N_e * C_target^0.73

Step 3: Compute critical batch size
        B* = B_e * C_target^0.24
        (equivalently: B_crit = B_star / L*^(1/alpha_B))

Step 4: Compute training steps
        S* = S_e * C_target^0.03

Step 5: Compute dataset size
        D* = B* * S*    (~2e10 * C_target^0.27 tokens)

Step 6: Train N*-parameter model on D* tokens with batch B*,
        stopping at S* steps (well before convergence)

Output: Trained model achieving approximately L* nats/token
```

# Architectural Independence

Experiments systematically vary:
- Depth: $n_\text{layer} \in [2, 207]$
- Width: $d_\text{model} \in [128, 4288]$
- Number of attention heads: varied while fixing $d_\text{attn} = d_\text{model}$

The key result: **performance depends only weakly on the specific shape** — losses vary by only a few percent across a wide range of depth/width ratios, as long as the model is not excessively shallow (fewer than 2 layers) or excessively narrow. A 40× variation in the aspect ratio ($n_\text{layer} / d_\text{model}$) produces only ~3% difference in loss.

> [!IMPORTANT]
> For fixed $N$, the optimal depth-to-width ratio is relatively flat. Practitioners can choose shape based on hardware efficiency (e.g., square matrices for tensor core utilization) without significant loss penalty.

**Comparison: Transformers vs. LSTMs vs. Universal Transformers**

| Architecture | Long-context performance | Sample efficiency | Architectural overhead |
|---|---|---|---|
| Transformer | Strong (power-law in context) | High | Standard |
| LSTM | Comparable for short context, degrades on long | Lower at large $N$ | Recurrent (sequential) |
| Universal Transformer | Marginal improvement via parameter reuse | Similar to Transformer | Increased depth per step |

# Transfer and Distribution Shift

Models trained on WebText2 are evaluated on Books Corpus, Common Crawl, English Wikipedia, and Internet Books. The transfer loss $L_\text{transfer}$ on any target distribution correlates strongly with training loss $L_\text{train}$:

$$L_\text{transfer} \approx L_\text{train} + \delta$$

where $\delta$ is a positive constant offset specific to the distribution pair. The offset does not depend on model size — scaling exponents on transfer distributions match those on the training distribution. **All scaling laws hold across distribution shift**, implying that improved in-distribution performance reliably predicts improved out-of-distribution performance.

# Comparison with Related Work

| Aspect | This work | Prior work |
|---|---|---|
| Dataset–model scaling | $D \propto N^{0.74}$ (sub-linear) | Some prior work found super-linear scaling |
| Dominant factor | Model size $N$ | Varied; often dataset size |
| Architectural sensitivity | Very weak for fixed $N$ | Often emphasized depth/width choices |
| LSTM vs. Transformer | Transformer outperforms on long contexts | Mixed results in earlier literature |
| Compute allocation | Model size dominates (exponent 0.73) | Not systematically studied before |

The paper specifically notes that LSTMs underperform Transformers at large scale (especially for long contexts), and that Universal Transformers (which reuse parameters across layers) show no advantage over standard Transformers when $N$ is matched.

> [!TIP]
> The follow-up "Chinchilla" paper (Hoffmann et al., 2022) refines these findings, concluding that compute-optimal training requires equal scaling of parameters and tokens (exponents closer to 0.50 each). The exponent difference stems from training beyond the 2.5×10⁵-step horizon used here.

# Extrapolation and Predictions

The authors extrapolate the scaling laws to identify a projected "performance wall" where further scaling yields diminishing returns:

$$C^* \sim 10^4 \text{ PF-days}, \quad N^* \sim 10^{12} \text{ parameters}, \quad D^* \sim 10^{12} \text{ tokens}, \quad L^* \sim 1.7 \text{ nats/token}$$

At $L \approx 1.7$ nats/token, the model may be approaching the entropy of natural language under the WebText2 distribution. This prediction is highly uncertain and depends on the power laws holding across eight or more orders of magnitude.

Two dataset growth rates emerge that are inconsistent with each other:
- From overfitting avoidance: $D \propto N^{0.74}$
- From compute-optimal training: $D \propto C_\text{min}^{0.27} \propto N^{0.37}$

These rates intersect at the projected limits above, implying a potential saturation of the current scaling regime.

# Experiments

- **Dataset:** WebText2 — an extended version of the WebText corpus (Reddit outbound links filtered for quality), comprising 20.3M documents, 96 GB of text, and $2.29 \times 10^{10}$ tokens. The test set is a held-out portion of $6.6 \times 10^8$ tokens.
- **Additional evaluation datasets:** Books Corpus, Common Crawl, English Wikipedia, Internet Books (for transfer experiments only).
- **Model sizes:** $N \in [768, 1.5 \times 10^9]$ non-embedding parameters (approximately six orders of magnitude).
- **Context length:** 1024 tokens (standard); varied in context-length ablations.
- **Optimizer:** Adam for smaller models; Adafactor for models $> 1$B parameters.
- **Training schedule:** Linear warmup over 3000 steps, then cosine decay to zero over $2.5 \times 10^5$ total steps; batch size 512 sequences × 1024 tokens = $5.24 \times 10^5$ tokens/step.
- **Regularization:** 10% dropout.
- **Hardware:** Not explicitly stated; compute measured in PF-days (petaflop-days, $\approx 10^{20}$ FLOPs).
- **Key result:** A $1.5 \times 10^9$ parameter model trained with $\sim 10^3$ PF-days of compute achieves test loss of approximately 3.0 nats/token on WebText2. Doubling compute (via model size alone) reduces loss by approximately $2^{0.05} \approx 3.5\%$ relative.
