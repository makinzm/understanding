# Meta Information

- URL: [Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](https://arxiv.org/abs/2507.07685)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yamaguchi, S., Nishida, K., & Chijiwa, D. (2025). Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought. arXiv:2507.07685.

# Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought

**RED** is a training-free decoding strategy that improves how large vision-language models (LVLMs) utilize intermediate reasoning steps during chain-of-thought (CoT) prompting. Rather than modifying model weights, RED adjusts the decoding procedure itself by combining two probability distributions: one conditioned on the visual input and one conditioned on the generated rationale.

## Problem: LVLMs Ignore Their Own Rationales

Standard CoT prompting for LVLMs generates a text rationale $r$ before producing the final answer $y$. The expected flow is:

```math
\begin{align}
  r &\sim p_\theta(r \mid x, q) \\
  y &\sim p_\theta(y \mid x, r, q)
\end{align}
```

where $x \in \mathbb{R}^{H \times W \times 3}$ is the image, $q$ is the text question, $r$ is the generated rationale, and $y$ is the final answer.

Through preliminary experiments, the authors find that:

1. Replacing the model's own rationale $r$ with an irrelevant rationale causes little performance degradation.
2. Attention to rationale tokens is substantially weaker than attention to image tokens.
3. Models rely disproportionately on visual information and largely bypass the intermediate reasoning captured in $r$.

> [!NOTE]
> "Existing LVLMs often ignore the contents of generated rationales in CoT reasoning."

This under-utilization problem motivates a method that explicitly enforces the model to ground its output predictions on the rationale.

## Method: RED (Rationale-Enhanced Decoding)

### KL-Constrained Reward Maximization

RED is derived from the following KL-constrained optimization:

```math
\begin{align}
  \max_\pi \; \mathbb{E}_\pi[R] - \beta D_{\mathrm{KL}}[\pi \| \pi_{\mathrm{ref}}]
\end{align}
```

- $\pi$ is the target decoding policy
- $R = \log p_\theta(y_i \mid y_{<i}, r, q)$ is the rationale-grounding reward, measuring how consistent each output token is with the rationale alone
- $\pi_{\mathrm{ref}} = p_\theta(y_i \mid y_{<i}, x, q)$ is the reference policy conditioned on the image
- $\beta > 0$ controls the KL regularization strength

**Proposition 1.** The closed-form optimal policy for the above objective is:

```math
\begin{align}
  \pi^*(a \mid s) = \frac{1}{Z(s)} \, \pi_{\mathrm{ref}}(a \mid s) \exp\!\left(\frac{1}{\beta} R(s, a)\right)
\end{align}
```

Substituting the reward definition with $\lambda = 1/\beta$, this yields the RED decoding distribution:

```math
\begin{align}
  \hat{p}_\theta(y_i) := \frac{1}{Z_\theta} \, p_\theta(y_i \mid y_{<i}, x, q) \cdot p_\theta(y_i \mid y_{<i}, r, q)^\lambda
\end{align}
```

### Practical Implementation (Log-Space Combination)

Taking the log, sampling from $\hat{p}_\theta$ is equivalent to combining logit scores from two forward passes:

```math
\begin{align}
  \widehat{\mathrm{logits}}_\theta(y_i) := \log\mathrm{softmax}(\mathrm{logits}_\theta(y_i \mid y, x, q)) + \lambda \cdot \log\mathrm{softmax}(\mathrm{logits}_\theta(y_i \mid y, r, q))
\end{align}
```

- First term: standard image-conditioned logits (reference policy)
- Second term: rationale-conditioned logits, upweighted by $\lambda$
- $\lambda \in \{0.1, 0.3, 0.5, 1.0, 10.0\}$ is selected on a validation set

> [!IMPORTANT]
> The rationale-conditioned pass replaces the image tokens $x$ with the text rationale $r$. This isolates the model's belief about the answer given only the textual reasoning, without visual distraction.

### Algorithm

**Input:** Image $x$, question $q$, LVLM $p_\theta$, hyperparameter $\lambda$

1. Generate rationale: $r \leftarrow \text{generate}_\theta(x, q)$
2. Initialize output $y \leftarrow \emptyset$
3. For each position until max length $L$:
   - Compute $\mathrm{logits}^{(1)} = \log\mathrm{softmax}(\mathrm{logits}_\theta(y_i \mid y, x, q))$ — image-conditioned
   - Compute $\mathrm{logits}^{(2)} = \log\mathrm{softmax}(\mathrm{logits}_\theta(y_i \mid y, r, q))$ — rationale-conditioned
   - Combine: $\widehat{\mathrm{logits}} = \mathrm{logits}^{(1)} + \lambda \cdot \mathrm{logits}^{(2)}$
   - Sample: $y_i \sim \mathrm{softmax}(\widehat{\mathrm{logits}})$
   - Append $y_i$ to $y$
4. Return $y$

> [!NOTE]
> This is a "power-of-experts" formulation: tokens with high probability under both distributions are strongly amplified, enforcing agreement between visual and textual reasoning.

### Difference from Contrastive Decoding Methods

| Method | Contrast Pair | Goal |
|---|---|---|
| VCD (Visual Contrastive Decoding) | Corrupted image vs. original image | Reduce visual hallucination |
| ICD (Instruction Contrastive Decoding) | Distracted instruction vs. original | Improve instruction following |
| **RED** | Image-conditioned vs. rationale-conditioned | Enforce rationale grounding |

Unlike VCD and ICD, RED does not subtract a degraded distribution. Instead, it multiplies two complementary distributions (image-side and rationale-side), rewarding tokens supported by both sources of information.

### Compatibility with CCoT

RED can also be applied to **CCoT** (Compositional CoT), which uses structured scene graphs as rationales instead of free-text rationales. In this setting, $r$ is a scene graph generated by an additional object detection component, and the same logit combination formula applies.

## Experiments

- **Datasets:** GQA (visual question answering), TextVQA (text-in-image VQA), MME (multi-modal evaluation benchmark), SEED-I (image reasoning), LLaVA-Bench (open-ended instruction following), MM-Vet (multi-capability VQA)
- **Hardware:** 24-core Intel Xeon CPU, NVIDIA H100 GPU (80 GB VRAM)
- **Decoding:** Greedy decoding throughout all experiments
- **Hyperparameter:** $\lambda$ tuned from $\{0.1, 0.3, 0.5, 1.0, 10.0\}$ using held-out validation split
- **Models:** Gemma-3 (4B, 12B, 27B), Qwen-2.5-VL (7B, 32B, 72B), Llama3-LLaVA-Next (8B)
- **Baselines:** Standard inference (no CoT), standard CoT, CCoT, VCD, ICD

**Results:**

- RED consistently improves accuracy over standard CoT across all six benchmarks and all tested model families.
- Using higher-quality rationales (GPT-4 generated) further boosts performance, confirming that RED effectively uses rationale quality.
- Replacing the rationale with an irrelevant one degrades performance with RED, demonstrating that RED—unlike standard CoT—genuinely conditions on the rationale content.
- Ablation on GQA with Gemma-3-12B: the power-of-experts formulation (RED, Eq. above) outperforms mixture-of-experts $((1-\lambda)p + \lambda q)$ by +2.16% and rationale-only decoding ($p_\theta(y \mid r, q)$) by a larger margin.

## Applicability

RED is suited for practitioners who deploy LVLMs with CoT prompting and want improved answer accuracy without retraining. The method applies to any auto-regressive LVLM at inference time. It is most beneficial when:

- The generated rationale is of moderate to high quality (improves with rationale quality)
- The task requires multi-step visual reasoning (GQA, TextVQA, MM-Vet)
- Compute overhead from a second forward pass is acceptable (roughly 2× decoding cost)

> [!CAUTION]
> The two-pass decoding cost scales with sequence length. For very long rationales or real-time applications, latency may be a concern.
