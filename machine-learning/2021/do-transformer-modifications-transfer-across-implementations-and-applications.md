# Meta Information

- URL: [Do Transformer Modifications Transfer Across Implementations and Applications?](https://arxiv.org/abs/2102.11972)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Narang, S., Chung, H. W., Tay, Y., Fedus, W., Fevry, T., Matena, M., Malkan, K., Fiedel, N., Shazeer, N., Lan, Z., Zhou, Y., Li, W., Ding, N., Marcus, J., Roberts, A., & Raffel, C. (2021). Do Transformer Modifications Transfer Across Implementations and Applications? arXiv:2102.11972.

# Overview

This paper conducts a large-scale empirical study evaluating approximately 50 proposed Transformer architecture modifications under a single unified experimental framework across multiple NLP tasks. The central finding is that most proposed modifications do **not** reliably improve performance when evaluated outside the codebase in which they were originally developed. Only a small subset of modifications—primarily minor changes and parameter-intensive approaches—show consistent gains.

**Who should use this**: Researchers and practitioners designing or evaluating Transformer architecture modifications. This paper provides critical guidance on how to properly validate architectural improvements to avoid overfitting to implementation-specific details.

**When and where**: Applicable to any scenario where architectural modifications to Transformer models are being proposed or adopted, especially for NLP pre-training and fine-tuning.

# Background: Transformer Architecture

## Input/Output Specification

The baseline encoder-decoder Transformer processes:
- **Input (encoder)**: A sequence of tokens $x = (x_1, \ldots, x_n)$, embedded as $h_{e,0} \in \mathbb{R}^{n \times d_{\text{model}}}$
- **Input (decoder)**: A target sequence prefix, embedded as $h_{d,0} \in \mathbb{R}^{m \times d_{\text{model}}}$
- **Output**: A probability distribution over the vocabulary at each decoder position, $p(y_t \mid y_{<t}, x)$

The baseline model has 12 encoder layers and 12 decoder layers, with $d_{\text{model}} = 768$, $d_{\text{ff}} = 2048$, $H = 12$ attention heads, and 223M total parameters.

## Attention Mechanism

For encoder self-attention at layer $l$, head $h$, token position $t$:

$$q_{e,l,h}[t] = h_{e,l-1}[t] \cdot Q_{e,l,h}, \quad Q_{e,l,h} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$

$$k_{e,l,h}[t] = h_{e,l-1}[t] \cdot K_{e,l,h}, \quad k_{e,l,h}[t] \in \mathbb{R}^{d_k}$$

$$v_{e,l,h}[t] = h_{e,l-1}[t] \cdot V_{e,l,h}, \quad v_{e,l,h}[t] \in \mathbb{R}^{d_v}$$

$$a_{e,l,h} = \text{softmax}\!\left(\frac{q_{e,l,h} \cdot k_{e,l,h}^{\top}}{\sqrt{d_k}}\right) v_{e,l,h} \in \mathbb{R}^{n \times d_v}$$

The attention output with residual connection:

$$s_{e,l}[t] = \text{LayerNorm}\!\left([a_{e,l,1}[t]; \ldots; a_{e,l,H}[t]] \cdot O_{e,l} + h_{e,l-1}[t]\right), \quad O_{e,l} \in \mathbb{R}^{H d_v \times d_{\text{model}}}$$

## Feedforward Network

$$f_{e,l}[t] = \text{ReLU}(s_{e,l}[t] \cdot W_{e,l,1} + b_{e,l,1}) \cdot W_{e,l,2} + b_{e,l,2}$$

$$h_{e,l} = \text{LayerNorm}(s_{e,l} + f_{e,l})$$

where $W_{e,l,1} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ and $W_{e,l,2} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.

For **causal decoder self-attention**, a mask $M$ prevents attending to future tokens:

$$M[i, j] = \begin{cases} 0 & \text{if } i \leq j \\ -\infty & \text{if } i > j \end{cases}$$

## Normalization Variants

**LayerNorm** (baseline): Normalizes using mean $\mu$ and variance $\sigma^2$ per token position $t \in \mathbb{R}^{d_{\text{model}}}$:

$$\text{LayerNorm}(h)[t] = \frac{\gamma}{\sigma[t]} (h[t] - \mu[t]) + \beta, \quad \mu[t] = \frac{1}{d}\sum_i h[t,i], \quad \sigma[t] = \sqrt{\frac{1}{d}\sum_i (h[t,i] - \mu[t])^2}$$

**RMS Norm**: Omits mean subtraction, reducing computation:

$$\text{RMSNorm}(h)[t] = \frac{\gamma}{\text{RMS}(h)} \cdot h[t], \quad \text{RMS}(h) = \sqrt{\frac{1}{d_{\text{model}}} \sum_i h[t,i]^2}$$

## GLU Activation Variants

Gated Linear Unit (GLU) variants replace the standard ReLU feedforward with a gated structure:

$$\text{GLU}(x, W, V) = \sigma(xW) \odot (xV) \in \mathbb{R}^{d_{\text{ff}}/2}$$

$$\text{GeGLU}(x, W, V) = \text{GELU}(xW) \odot (xV)$$

$$\text{SwiGLU}(x, W, V) = \text{Swish}_\beta(xW) \odot (xV)$$

$$\text{ReGLU}(x, W, V) = \max(0, xW) \odot (xV)$$

where $\odot$ denotes element-wise multiplication, and $x \in \mathbb{R}^{d_{\text{model}}}$, $W, V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}/2}$.

> [!NOTE]
> GLU variants use a gating mechanism that multiplies two linear projections, where one acts as a "gate" controlling information flow. This increases expressiveness without adding attention heads.

# Experimental Setup

## Pre-training

- **Dataset**: C4 (Colossal Clean Crawled Corpus) — web-crawled English text
- **Objective**: Span corruption (T5-style masked span prediction)
- **Steps**: 524,288 gradient steps
- **Batch size**: 65,536 tokens per step
- **Optimizer**: Adafactor
- **Variance estimation**: 5 independent runs of 65,536 steps each to measure variability

## Fine-tuning Tasks (Downstream Evaluation)

| Task | Type | Metric |
|---|---|---|
| SuperGLUE | Language understanding (multi-task) | Average score |
| XSum | Abstractive summarization | ROUGE-2 |
| WebQuestions | Closed-book QA | Exact match accuracy |
| WMT'14 En-De | Machine translation | BLEU |

## Baseline Model Configuration

12-layer encoder-decoder Transformer, 223M parameters, pre-norm (LayerNorm before attention/FFN), relative positional attention (Shaw et al., 2018).

# Modifications Evaluated

## Categories

| Category | Examples |
|---|---|
| Activation functions | GeLU, Swish, ELU, SeLU, Sigmoid, Softplus, GLU, ReGLU, GeGLU, SwiGLU, LiGLU |
| Normalization | RMS Norm, ReZero, ReZero+LayerNorm, Fixup |
| Depth/width tradeoffs | Varying layers (6–24) and $d_{\text{ff}}$ (1536–6144) at fixed parameter count |
| Embeddings | Factorized, tied (enc/dec, dec in/out), adaptive input |
| Parameter sharing | ALBERT-style block sharing, encoder/decoder only |
| Softmax variants | Adaptive softmax, Mixture of Softmaxes |
| Full architectures | Evolved Transformer, Synthesizer, Switch Transformer, MoE, Universal Transformer, etc. |

## Key Results

**Modifications that improve performance**:
- **GeGLU, SwiGLU, ReGLU**: Reduce pre-training loss from 1.838 to ~1.79 and improve SuperGLUE/XSum, at the cost of extra weight matrices in the FFN
- **RMS Norm**: Small but consistent improvement (loss 1.821) with reduced computation vs. LayerNorm
- **Switch Transformer / MoE**: Best pre-training loss (1.758/1.785) by significantly increasing parameter count via sparsely-activated expert layers
- **Tied decoder input/output embeddings**: Minor improvement (loss 1.827) with no parameter overhead

**Modifications that fail to transfer**:
- **Universal Transformer**: Loss 2.053, SuperGLUE 70.13 — underperforms baseline despite recursive computation
- **Dynamic/Lightweight Convolutions**: Loss >2.0, BLEU collapses to 17.03 for dynamic convolutions
- **Transparent Attention**: SuperGLUE 54.31, XSum 10.40 — catastrophic failure
- **Synthesizer (dense/random base variants)**: Loss 1.96–2.0 — only hybrid "plus" variants approach baseline
- **ReZero, Fixup**: Significantly hurt performance (SuperGLUE 61.7 and 58.6 respectively)

## Correlation: Pre-training Loss vs. Downstream Performance

Pre-training perplexity correlates well with fine-tuning, with Spearman's $\rho$:
- SuperGLUE: $\rho = 0.87$
- XSum: $\rho = 0.80$
- WebQuestions: $\rho = 0.69$

> [!IMPORTANT]
> The weaker correlation for WebQuestions (knowledge-intensive QA) suggests that pre-training loss is a less reliable proxy for tasks requiring factual recall than for language understanding or generation.

# Key Findings

## Primary Conjecture: Implementation Dependence

The authors conjecture that most architectural modifications succeed primarily within the codebase where they were developed (e.g., Mesh TensorFlow for MoE and Synthesizer). When evaluated in a different implementation, these gains vanish. Evidence:

- Methods originating in Mesh TensorFlow (Switch Transformer, MoE, Synthesizer dense-plus/random-plus) perform well in this Mesh TensorFlow-based study
- Methods from other codebases (PyTorch-based Evolved Transformer, Funnel Transformer, etc.) underperform

## Three Categories of Successful Modifications

1. **Minor adjustments**: GLU activation variants (SwiGLU, GeGLU, ReGLU), RMS Norm, decoder embedding tying — small, robust gains with minimal hyperparameter sensitivity
2. **Parameter-increasing methods**: Switch Transformer, MoE, deeper models — improve performance by adding compute/parameters, not architectural novelty
3. **Same-codebase methods**: Modifications developed and tested in the same framework as this study

## Hyperparameter Sensitivity

A 25-configuration hyperparameter search for Universal Transformer achieved only ~6% relative improvement, still failing to match the vanilla Transformer. This suggests that robust architectural improvements should be hyperparameter-agnostic.

# Comparison with Similar Work

| Aspect | This Work | Individual Proposed Methods |
|---|---|---|
| Evaluation framework | Single unified codebase (Mesh TF) | Each paper uses its own codebase |
| Tasks covered | Pre-training + 4 downstream tasks | Often 1–2 tasks |
| Parameter matching | Fixed 223M params (except MoE/Switch) | May increase parameters |
| Reproducibility | 5 variance seeds | Often single run |
| Finding | Most modifications fail to transfer | Original papers show improvement |

> [!TIP]
> For practitioners choosing Transformer modifications: prefer GLU variants (SwiGLU/GeGLU), RMS Norm, and tied decoder embeddings as safe, well-validated improvements. Treat complex architectural claims from single-codebase papers with skepticism.

# Recommendations for Future Research

1. **Multi-codebase validation**: Test modifications across at least two independent implementations
2. **Diverse task evaluation**: Cover pre-training, supervised learning, language modeling, and multiple downstream applications
3. **Hyperparameter robustness**: Measure performance with consistent hyperparameters or explicitly test hyperparameter sensitivity
4. **Statistical reporting**: Report mean ± standard deviation across multiple seeds to avoid cherry-picking

# Experiments

- **Dataset (pre-training)**: C4 (Colossal Clean Crawled Corpus) — English web text
- **Dataset (fine-tuning)**: SuperGLUE, XSum, WebQuestions, WMT'14 English-German
- **Hardware**: TPUs (Google internal infrastructure)
- **Optimizer**: Adafactor (adaptive learning rate with sublinear memory)
- **Results**: Switch Transformer achieves best pre-training loss (1.758) and competitive SuperGLUE (75.38); GLU variants (GeGLU/SwiGLU/ReGLU) achieve pre-training loss ~1.79 with consistent downstream gains; most complex architectural modifications (Universal Transformer, Synthesizer dense, dynamic convolutions) fail to match the vanilla Transformer baseline
