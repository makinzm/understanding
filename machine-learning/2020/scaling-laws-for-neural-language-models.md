# Meta Information

- URL: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.

# Introduction

This paper establishes empirical power-law relationships governing the performance of autoregressive Transformer language models as a function of model size $N$ (non-embedding parameters), dataset size $D$ (tokens), and compute budget $C$ (FLOPs). The central finding is that test loss $L$ follows smooth power laws spanning more than seven orders of magnitude in each variable, with each law holding independently when the other factors are not bottlenecks. These scaling laws allow practitioners to predict the performance of large models from small-scale experiments and to allocate compute budgets optimally.

> [!NOTE]
> "Performance depends strongly on scale, weakly on model shape: The most important factors determining model quality are the number of model parameters N (excluding embeddings) and the amount of compute C used for training."

The results challenge the intuition that model depth, width, or number of attention heads are critical architectural choices—when total parameter count is held fixed, varying these produces only a few percent variation in loss.

# Background and Problem Setting

## Language Modeling Setup

The task is autoregressive language modeling: predict the next token given the preceding context. The model is trained to minimize cross-entropy loss (measured in nats/token) over a fixed vocabulary of 50,257 byte-pair encoded tokens. The evaluation metric is the test loss $L$ on held-out data from the same distribution.

**Input:** a sequence of tokens $x_1, x_2, \ldots, x_T$ where each $x_i \in \{0, \ldots, 50256\}$
**Output:** a probability distribution over the next token; the loss $L = -\frac{1}{T}\sum_{t=1}^{T} \log p(x_t | x_{< t})$ in nats/token

## Transformer Architecture

All experiments use decoder-only Transformers (same family as GPT-2). The parameter count (excluding embeddings) is:

$$N \approx 12 \cdot n_\text{layer} \cdot d_\text{model}^2$$

where the standard ratio $d_\text{ff} = 4 \cdot d_\text{attn} = 4 \cdot d_\text{model}$ is used. The embedding parameters ($n_\text{vocab} \times d_\text{model}$ and $n_\text{ctx} \times d_\text{model}$) are excluded from $N$ because they do not follow the same scaling laws.

The FLOPs per forward pass are approximately:

$$C_\text{forward} \approx 2N + 2 \cdot n_\text{layer} \cdot n_\text{ctx} \cdot d_\text{model}$$

For training (forward + backward), the total compute per token is approximately $6N$ FLOPs, so a training run over $T$ tokens uses $C \approx 6NT$ FLOPs total.

# Scaling Laws

## Individual Power Laws

When the other variables are not limiting, each of $N$, $D$, and $C$ independently determines performance via a power law:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13} \text{ tokens}$$

$$L(C_\text{min}) = \left(\frac{C_c^\text{min}}{C_\text{min}}\right)^{\alpha_{C_\text{min}}}, \quad \alpha_{C_\text{min}} \approx 0.050, \quad C_c^\text{min} \approx 3.1 \times 10^8 \text{ PF-days}$$

Here $C_\text{min}$ denotes the minimum compute required to reach a given loss (i.e., training optimally sized models). $L(N)$ is the irreducible loss achievable with unlimited data; $L(D)$ is the loss achievable with unlimited parameters.

## Joint N and D Dependence (Overfitting)

When both $N$ and $D$ are finite, the loss is described by:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

This formula captures the overfitting behavior: for a fixed $N$, performance stops improving when $D \lesssim D_c \cdot (N/N_c)^{\alpha_N/\alpha_D}$. To avoid overfitting, the dataset should satisfy:

$$D \gtrsim (5 \times 10^3) \cdot N^{0.74} \text{ tokens}$$

> [!IMPORTANT]
> This means dataset size should grow as $D \propto N^{0.74}$, which is sub-linear — doubling parameters requires less than double the data to avoid overfitting.

## Training Dynamics and the N–Steps Law

For a model of size $N$ trained for $S_\text{min}$ gradient steps (at the optimal batch size), the loss trajectory follows:

$$L(N, S_\text{min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_\text{min}}\right)^{\alpha_S}, \quad \alpha_S \approx 0.76, \quad S_c \approx 2.1 \times 10^3$$

Training curves are power-law in steps, and this formula allows early-stopping prediction: training a large model for few steps can converge to the same loss as a smaller model trained to convergence.

# Compute-Optimal Scaling

## Critical Batch Size

The critical batch size $B_\text{crit}$ is the batch size that maximizes compute efficiency (balancing time and compute). It scales with loss:

$$B_\text{crit}(L) = \frac{B^*}{L^{1/\alpha_B}}, \quad B^* \approx 2 \times 10^8 \text{ tokens}, \quad \alpha_B \approx 0.21$$

Training at $B \ll B_\text{crit}$ wastes compute (too many serial steps); training at $B \gg B_\text{crit}$ wastes data (poor sample efficiency). At the critical batch size, the time–compute tradeoff satisfies:

$$\left(\frac{S}{S_\text{min}} - 1\right) \cdot \left(\frac{E}{E_\text{min}} - 1\right) = 1$$

where $E = B \cdot S$ is total examples seen and $E_\text{min} = B_\text{crit} \cdot S_\text{min}$.

## Optimal Allocation Under Fixed Compute Budget

Given a total compute budget $C$, the compute-optimal strategy allocates resources as:

| Resource | Optimal Scaling |
|---|---|
| Model size $N$ | $\propto C^{0.73}$ |
| Batch size $B$ | $\propto C^{0.24}$ |
| Training steps $S$ | $\propto C^{0.03}$ |
| Dataset tokens $D = B \cdot S$ | $\propto C^{0.27}$ |

The exponents sum to 1 ($0.73 + 0.24 + 0.03 = 1.00$). The dominant finding is that **most additional compute should go into larger models**, not more data or more training steps.

> [!NOTE]
> "We should therefore prioritize model size in setting up efficient training runs. Concretely, for every 10× increase in compute, model size should increase by about 5.5×, with modest increases in data and minimal increases in serial training time."

**Pseudocode for compute-optimal model selection:**

```
Given: compute budget C_target (in PF-days)
1. Compute optimal N* = k_N * C_target^0.73    (k_N from fitted constants)
2. Compute optimal D* = k_D * C_target^0.27
3. Compute optimal B* = B_crit(L*)
4. Set S* = D* / B*  (training steps)
5. Train N*-parameter model on D* tokens with batch B*
```

# Architectural Independence

Experiments systematically vary depth $n_\text{layer} \in [2, 207]$, width $d_\text{model} \in [128, 4288]$, and number of attention heads, holding $N$ approximately fixed. The key result is that **performance depends only weakly on the specific shape** — losses vary by only a few percent across a wide range of depth/width ratios, as long as the model is not excessively shallow (fewer than 2 layers) or excessively narrow.

> [!IMPORTANT]
> For fixed $N$, the optimal depth-to-width ratio is relatively flat. Practitioners can choose shape based on hardware efficiency (e.g., square matrices for tensor core utilization) without significant loss penalty.

# Transfer and Distribution Shift

Models trained on WebText2 are evaluated on Books Corpus, Common Crawl, English Wikipedia, and Internet Books. The transfer loss $L_\text{transfer}$ on any target distribution correlates strongly with training loss $L_\text{train}$:

$$L_\text{transfer} \approx L_\text{train} + \delta$$

where $\delta$ is a positive constant offset specific to the distribution pair. The offset does not depend on model size, meaning the scaling exponents on transfer distributions match those on the training distribution. **All scaling laws hold across distribution shift**, implying that improved in-distribution performance reliably predicts improved out-of-distribution performance.

# Comparison with Related Work

| Aspect | This work | Prior work |
|---|---|---|
| Dataset–model scaling | $D \propto N^{0.74}$ (sub-linear) | Some prior work found super-linear scaling |
| Dominant factor | Model size $N$ | Varied; often dataset size |
| Architectural sensitivity | Very weak for fixed $N$ | Often emphasized depth/width choices |
| LSTM vs. Transformer | Transformer outperforms on long contexts | Mixed results in earlier literature |

The paper specifically notes that LSTMs underperform Transformers at large scale (especially for long contexts), and that Universal Transformers (which reuse parameters across layers) show no advantage over standard Transformers when $N$ is matched.

# Extrapolation and Predictions

The authors extrapolate the scaling laws to identify a "performance wall" where further scaling yields diminishing returns:

$$C^* \sim 10^4 \text{ PF-days}, \quad N^* \sim 10^{12} \text{ parameters}, \quad D^* \sim 10^{12} \text{ tokens}, \quad L^* \sim 1.7 \text{ nats/token}$$

At $L \approx 1.7$ nats/token, the model may be approaching the entropy of natural language under the WebText2 distribution. This prediction is highly uncertain and depends on the power laws holding across eight or more orders of magnitude.

# Experiments

- **Dataset:** WebText2 — an extended version of the WebText corpus (Reddit outbound links filtered for quality), comprising 20.3M documents, 96 GB of text, and $2.29 \times 10^{10}$ tokens. No standard train/dev/test split is reported; the test set is a held-out portion of WebText2.
- **Additional evaluation datasets:** Books Corpus, Common Crawl, English Wikipedia, Internet Books (for transfer experiments only).
- **Model sizes:** $N \in [768, 1.5 \times 10^9]$ non-embedding parameters (approximately six orders of magnitude).
- **Context length:** 1024 tokens (standard); varied in context-length ablations.
- **Optimizer:** Adam for smaller models; Adafactor for models $> 1$B parameters.
- **Training schedule:** Linear warmup over 3000 steps, then cosine decay to zero over $2.5 \times 10^5$ total steps; batch size 512 sequences × 1024 tokens = $5.24 \times 10^5$ tokens/step.
- **Regularization:** 10% dropout.
- **Hardware:** Not explicitly stated; compute measured in PF-days (petaflop-days, $\approx 10^{20}$ FLOPs).
- **Key result:** A $1.5 \times 10^9$ parameter model trained with $\sim 10^3$ PF-days of compute achieves test loss of approximately 3.0 nats/token on WebText2. Doubling compute (via model size alone) reduces loss by approximately $2^{0.05} \approx 3.5\%$ relative.

> [!IMPORTANT]
> These scaling laws were later revised by Hoffmann et al. (2022) in the Chinchilla paper (arXiv:2203.15556), which found that the optimal allocation is **equal** scaling of model parameters and training tokens ($N \propto C^{0.5}$, $D \propto C^{0.5}$), rather than the heavily model-size-biased allocation ($N \propto C^{0.73}$) suggested here. The discrepancy arises because this paper used fixed training budgets and did not explore training smaller models for much longer.
