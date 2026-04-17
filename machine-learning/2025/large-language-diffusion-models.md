# Meta Information

- URL: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). Large Language Diffusion Models. arXiv:2502.09992.

# Large Language Diffusion Models (LLaDA)

LLaDA (Large Language Diffusion with mAsking) is a masked diffusion model trained at the 8B parameter scale that demonstrates LLM-level capabilities — scalability, in-context learning, and instruction following — without using autoregressive (AR) next-token prediction. It targets researchers and practitioners exploring non-AR alternatives for language modeling, particularly those interested in bidirectional generation and improved reversal reasoning.

## Motivation: Generative Modeling vs. Autoregressive Modeling

Standard LLMs are trained with an autoregressive factorization:

```math
\begin{align}
p_\theta(x) = p_\theta(x^1) \prod_{i=2}^{L} p_\theta(x^i \mid x^1, \ldots, x^{i-1})
\end{align}
```

The core hypothesis of LLaDA is that emergent LLM capabilities arise from sound generative modeling principles — specifically, maximizing the likelihood $\mathbb{E}_{p_\text{data}(x)} \log p_\theta(x)$ — rather than from the AR formulation itself. LLaDA replaces the AR factorization with a masked diffusion process while keeping the same Transformer backbone.

## Masked Diffusion Model

### Forward Process

Given a sequence $x_0 \in \mathcal{V}^L$ of $L$ tokens, a forward noising process independently masks each token with probability $t \in (0, 1]$:

```math
\begin{align}
q_{t|0}(x_t^i \mid x_0^i) =
\begin{cases}
1 - t, & \text{if } x_t^i = x_0^i \\
t,     & \text{if } x_t^i = \texttt{[MASK]}
\end{cases}
\end{align}
```

At $t = 1$, every token is masked; at $t = 0$, the sequence is clean. The joint over the full sequence factorizes as:

```math
\begin{align}
q_{t|0}(x_t \mid x_0) = \prod_{i=1}^{L} q_{t|0}(x_t^i \mid x_0^i)
\end{align}
```

### Reverse Process

The reverse transition from time $t$ to $s < t$ is defined per token. For a masked token at position $i$:

```math
\begin{align}
q_{s|t}(x_s^i \mid x_t) =
\begin{cases}
1,                                        & \text{if } x_t^i \neq \texttt{[MASK]},\ x_s^i = x_t^i \\
s/t,                                      & \text{if } x_t^i = \texttt{[MASK]},\ x_s^i = \texttt{[MASK]} \\
\frac{t-s}{t} q_{0|t}(x_s^i \mid x_t),  & \text{if } x_t^i = \texttt{[MASK]},\ x_s^i \neq \texttt{[MASK]} \\
0,                                        & \text{otherwise}
\end{cases}
\end{align}
```

A neural network $p_\theta(x_0^i \mid x_t)$ approximates $q_{0|t}(x_s^i \mid x_t)$ — the probability of the original clean token given the partially masked sequence. Unlike AR models, $p_\theta$ uses **bidirectional (non-causal) attention** and predicts all masked tokens simultaneously.

### Training Objective

The loss is a cross-entropy over masked positions, weighted by $1/t$:

```math
\begin{align}
\mathcal{L}(\theta) \triangleq -\mathbb{E}_{t \sim U(0,1],\, x_0,\, x_t}\left[\frac{1}{t} \sum_{i=1}^{L} \mathbb{1}[x_t^i = \texttt{[MASK]}] \log p_\theta(x_0^i \mid x_t)\right]
\end{align}
```

This is a variational upper bound on the negative log-likelihood: $-\mathbb{E}_{p_\text{data}}[\log p_\theta(x_0)] \leq \mathcal{L}(\theta)$.

An equivalent discrete-time form samples the number of masked tokens $l \sim \{1, \ldots, L\}$ uniformly:

```math
\begin{align}
\mathcal{L}(\theta) = -\mathbb{E}_{l,\, x_0,\, x_l}\left[\frac{L}{l} \sum_{i=1}^{L} \mathbb{1}[x_l^i = \texttt{[MASK]}] \log p_\theta(x_0^i \mid x_l)\right]
\end{align}
```

### Pre-training Algorithm

```
Input: mask predictor p_θ, data distribution p_data
repeat
  sample x_0 ~ p_data  (with 1% prob, length ~ U[1, 4096])
  sample t ~ U(0, 1]
  sample x_t ~ q_{t|0}(x_t | x_0)       # mask each token independently with prob t
  compute L = -(1 / (t * |x_0|)) * Σ_i 1[x_t^i = MASK] * log p_θ(x_0^i | x_t)
  update θ via ∇_θ L
until converged
```

## Supervised Fine-Tuning (SFT)

For instruction following, the model conditions on a prompt $p_0 \in \mathcal{V}^{L_p}$ and generates a response $r_0 \in \mathcal{V}^{L'}$. During SFT, only response tokens are masked; the prompt is kept clean:

```math
\begin{align}
\mathcal{L}_\text{SFT}(\theta) = -\mathbb{E}_{t,\, p_0,\, r_0,\, r_t}\left[\frac{1}{t} \sum_{i=1}^{L'} \mathbb{1}[r_t^i = \texttt{[MASK]}] \log p_\theta(r_0^i \mid p_0, r_t)\right]
\end{align}
```

**Input**: concatenation of clean prompt tokens $p_0 \in \mathbb{R}^{L_p}$ and partially masked response tokens $r_t \in \mathbb{R}^{L'}$, processed by a bidirectional Transformer.
**Output**: probability distribution over vocabulary for each masked response position.

### SFT Algorithm

```
Input: mask predictor p_θ, paired data (p_0, r_0)
repeat
  sample (p_0, r_0) ~ p_data
  sample t ~ U(0, 1]
  sample r_t ~ q_{t|0}(r_t | r_0)     # mask only response tokens
  compute L = -(1 / (t * L')) * Σ_i 1[r_t^i = MASK] * log p_θ(r_0^i | p_0, r_t)
  update θ via ∇_θ L
until converged
```

## Sampling (Inference)

### Standard Reverse Sampling

```
Input: p_θ, prompt p_0, answer length L, number of steps N
Initialize r_1 = [MASK, MASK, ..., MASK]   # fully masked, length L
for t = 1, (N-1)/N, ..., 1/N  do:
  s = t - 1/N
  r̂_0 = argmax_i p_θ(r_0^i | p_0, r_t)   # predict all masked tokens
  for each position i:
    if r_t^i ≠ MASK:   r_s^i = r_t^i      # keep unmasked tokens
    else: remask r̂_0^i = MASK with prob s/t, else keep r̂_0^i
  r_s = result
return r_0
```

### Low-Confidence Remasking

Instead of random remasking, positions with the lowest predicted token probability are re-masked at each step:

```
Initialize r_1 = fully masked, length L
for t = 1, (N-1)/N, ..., 1/N  do:
  s = t - 1/N
  for each position i:
    if r_t^i ≠ MASK: r̂_0^i = r_t^i, confidence c^i = 1.0
    else: r̂_0^i = argmax p_θ(r_0^i | p_0, r_t), c^i = max probability
  n_keep = floor(L * (1 - s))
  remask the floor(L * s) positions with lowest confidence
  r_s = r̂_0 with those positions remasked
return r̂_0
```

> [!NOTE]
> Low-confidence remasking improves base model performance (e.g., HumanEval-FIM: 52.3 → 64.7) but degrades instruction-tuned model performance (GSM8K: 72.0 → 12.9). A **semi-autoregressive** variant generates left-to-right in blocks, applying diffusion within each block, which recovers instruct performance (73.8) while preserving base gains (64.4).

### Classifier-Free Guidance

For improved conditional generation without labeled data, guidance scales the ratio of conditional to unconditional predictions:

```math
\begin{align}
\tilde{p}_\theta(r_0 \mid p_0, r_t) \propto \frac{[p_\theta(r_0 \mid p_0, r_t)]^{1+w}}{[p_\theta(r_0 \mid \varnothing, r_t)]^w}
\end{align}
```

where $\varnothing$ is a null/empty prompt and $w \geq 0$ is the guidance weight.

## Architecture

LLaDA uses a standard Transformer identical in size to LLaMA3 8B, except it uses **full multi-head attention (MHA)** (no grouped query attention) and reduces FFN dimensions to compensate. Crucially, there is **no causal mask** — all positions attend to all positions bidirectionally.

| Parameter | LLaDA 8B | LLaMA3 8B |
|---|---|---|
| Layers | 32 | 32 |
| Model dim $d$ | 4096 | 4096 |
| Attention heads | 32 | 32 |
| KV heads | 32 (full MHA) | 8 (GQA) |
| FFN dim | 12,288 | 14,336 |
| Vocab size | 126,464 | 128,000 |
| Total parameters | 8.02B | 8.03B |

A 1.49B variant was also trained for scaling law experiments (up to $10^{23}$ FLOPs), confirming comparable scaling behavior to AR models.

## Training Details

- **Pre-training data**: 2.3 trillion tokens
- **Compute**: 0.13 million H800 GPU hours
- **Context length**: 4,096 tokens (1% of samples with random lengths in $[1, 4096]$)
- **SFT data**: 4.5 million (prompt, response) pairs
- **Masking rate**: $t \sim U(0, 1]$ per sample

## Comparison with AR Models

| Property | LLaDA (Masked Diffusion) | AR Models (e.g., LLaMA3) |
|---|---|---|
| Attention direction | Bidirectional (all positions see all) | Causal (left-to-right only) |
| Generation order | Parallel / iterative denoising | Strictly left-to-right |
| Reversal reasoning | Balanced (forward ≈ reversal) | Severely degraded on reversal tasks |
| Fill-in-the-middle | Native (bidirectional attention) | Requires special training |
| KV-cache | Not applicable (no sequential decode) | Efficient with KV-cache |
| Inference speed | Slower (no KV-cache optimization) | Faster at equivalent compute |
| Alignment (RLHF) | Not yet explored | Standard |

> [!IMPORTANT]
> The "reversal curse" — where AR models trained on "A is B" fail to learn "B is A" — is structurally avoided by LLaDA because the bidirectional Transformer has no directional bias. GPT-4o drops 48.4 points on Chinese poem reversal completion (82.7% → 34.3%); LLaDA drops only 6.4 points (48.8% → 42.4%).

> [!TIP]
> BERT (Devlin et al., 2018) also uses masked token prediction with bidirectional attention, but is not trained to generate full sequences and cannot be scaled as a generative LLM. LLaDA extends this idea by deriving a rigorous variational training objective and applying it at the 8B-parameter scale.

## Experiments

- **Datasets**: MMLU, BBH, ARC-C, Hellaswag, TruthfulQA, WinoGrande, PIQA, GSM8K, MATH (Hendrycks), GPQA, HumanEval, HumanEval-FIM, MBPP, CMMLU, C-Eval, iGSM (synthetic multi-step math reasoning), MMLU-pro
- **Hardware**: H800 GPUs (0.13M GPU hours for pre-training)
- **Optimizer**: Not specified in the extracted content
- **Key Results**:
  - LLaDA 8B Base surpasses LLaMA3 8B Base on GSM8K (70.7 vs 53.1), MATH (27.3 vs 15.1), HumanEval-FIM (73.8 vs 73.3), CMMLU (69.9 vs 50.7), and C-Eval (70.5 vs 51.7)
  - LLaDA 8B Instruct is competitive with LLaMA3 8B Instruct across 15 benchmarks; surpasses it on ARC-C (88.5 vs 82.4) and GSM8K (78.6 vs 78.3)
  - On multi-step iGSM math reasoning (6 steps), LLaDA 8B (44.0%) substantially outperforms LLaMA3 8B (34.0%)
  - Scaling experiments confirm that LLaDA follows a comparable power-law scaling curve to AR models up to $10^{23}$ FLOPs
