# Meta Information

- URL: [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv preprint arXiv:1606.08415.

# Gaussian Error Linear Units (GELUs)

## Overview

GELU is a neural network activation function that outperforms ReLU and ELU across NLP, computer vision, and speech tasks. Unlike ReLU which gates inputs by sign ($x \cdot \mathbf{1}(x > 0)$), GELU weights inputs by their magnitude relative to the Gaussian distribution, combining ideas from dropout regularization and input-dependent stochastic masking.

**Applicability:** Used as a drop-in replacement for ReLU in feedforward layers of deep neural networks (MLPs, CNNs, Transformers). Especially effective in NLP models — adopted as the default activation in BERT, GPT, and subsequent large language models.

## Motivation: Stochastic Regularizer Interpretation

The key insight is that GELU arises naturally from the expectation of a stochastic masking operation:

1. For input $x \in \mathbb{R}$, draw a stochastic mask $m \sim \text{Bernoulli}(\Phi(x))$, where $\Phi(x) = P(X \leq x)$ for $X \sim \mathcal{N}(0, 1)$.
2. Apply $m \cdot x$: with probability $\Phi(x)$ the input is retained, otherwise zeroed.
3. Taking the expectation over $m$:

$$\mathbb{E}[m \cdot x] = \Phi(x) \cdot x + (1 - \Phi(x)) \cdot 0 = x \cdot \Phi(x)$$

This expected transformation is the deterministic GELU nonlinearity. Inputs with high absolute value are more likely retained (high $\Phi(x)$ for large $x$) or zeroed (low $\Phi(x)$ for large negative $x$), while near-zero inputs are stochastically masked — analogous to Adaptive Dropout but applied without explicit sampling at inference.

> [!NOTE]
> "GELU nonlinearity weights inputs by their value, rather than gates inputs by their sign." — Hendrycks & Gimpel (2016)

## GELU Definition

**Exact form** using the standard Gaussian CDF $\Phi(x)$ or equivalently the error function $\text{erf}$:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Input/Output:** scalar or element-wise applied to tensor $x \in \mathbb{R}^{d}$; output has same shape as input.

**Approximations** (computationally efficient, used in practice):

1. Tanh approximation (used in GPT-2/BERT implementations):

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

2. Sigmoid approximation (simpler):

$$\text{GELU}(x) \approx x \cdot \sigma(1.702x)$$

where $\sigma$ is the logistic sigmoid function.

## Comparison with Related Activation Functions

| Activation | Formula | Properties |
|---|---|---|
| **ReLU** | $x \cdot \mathbf{1}(x > 0)$ | Convex, monotonic; gates by sign; zero gradient for $x < 0$ |
| **ELU** | $x$ if $x > 0$ else $\alpha(e^x - 1)$ | Can output negative values; can increase training speed |
| **GELU** | $x \cdot \Phi(x)$ | Non-convex, non-monotonic; curvature throughout; input-magnitude weighting |
| **SiLU** | $x \cdot \sigma(x)$ | Uses logistic CDF instead of Gaussian CDF; nearly identical to GELU |

Key differences from ReLU:
- GELU is **non-monotonic**: has a small negative region near $x \approx -0.17$ where the output dips slightly below zero before the curve rises.
- GELU is **smooth**: differentiable everywhere, no kink at zero.
- GELU converges to ReLU as the standard deviation of the Gaussian approaches 0.

> [!TIP]
> SiLU (Sigmoid Linear Unit), also known as "Swish" (proposed by Google Brain in 2017), is $x \cdot \sigma(x)$ and was independently proposed. The Appendix of this paper documents a credit dispute: the authors coined SiLU first in their June 2016 GELU paper. PyTorch and TensorFlow later renamed "swish" to "SiLU" to reflect prior attribution.

## Algorithm: Computing GELU (Tanh Approximation)

```
Input:  x ∈ ℝ^d  (pre-activation vector)
Output: y ∈ ℝ^d  (activated vector)

c ← sqrt(2/π) ≈ 0.7978845608
inner ← c * (x + 0.044715 * x^3)      # element-wise cubic expansion
y ← 0.5 * x * (1 + tanh(inner))        # tanh-approximated Gaussian CDF
return y
```

Gradient (for backpropagation):

$$\frac{\partial \text{GELU}(x)}{\partial x} = \Phi(x) + x \cdot \phi(x)$$

where $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ is the standard Gaussian PDF.

## Experiments

### Datasets

| Task | Dataset | Train | Dev | Test |
|---|---|---|---|---|
| Image classification | MNIST | 60,000 | — | 10,000 |
| Reconstruction | MNIST Autoencoder | 60,000 | — | 10,000 |
| NLP (POS tagging) | Twitter POS | 1,000 | 327 | 500 |
| Speech | TIMIT (frame classification) | — | — | — (39 phoneme labels) |
| Image classification | CIFAR-10 | 50,000 | — | 10,000 |
| Image classification | CIFAR-100 | 50,000 | — | 10,000 |

### Architectures and Hyperparameters

**MNIST Classification:**
- 8-layer fully connected network, 128 neurons per layer
- Adam optimizer, learning rates ∈ {$10^{-3}$, $10^{-4}$, $10^{-5}$}, 50 epochs, batch size 128

**MNIST Autoencoder:**
- Encoder-decoder: $1000 \to 500 \to 250 \to 30 \to 250 \to 500 \to 1000$
- Adam optimizer, learning rates ∈ {$10^{-3}$, $10^{-4}$}, batch size 64, MSE loss

**Twitter POS Tagging:**
- 2-layer network, 256 neurons/layer, dropout keep probability 0.8

**TIMIT Frame Classification:**
- 5-layer network, 2048 neurons/layer, 39 output labels
- Dropout rate 0.5; input: 11 frames × 26 MFCC features

**CIFAR-10 (Shallow CNN):**
- 9-layer CNN with batch normalization
- Adam optimizer, 200 epochs, learning rate decay at epoch 100

**CIFAR-100 (Wide ResNet, 40 layers, widening factor 4):**
- 50 epochs, Nesterov momentum, dropout keep probability 0.7

### Key Results

| Task | GELU | ReLU | ELU |
|---|---|---|---|
| MNIST classification (median train loss) | **lowest** | higher | higher |
| Twitter POS error | **12.57%** | 12.67% | 12.91% |
| TIMIT frame error | **29.3%** | 29.5% | 29.6% |
| CIFAR-10 test error | **7.89%** | 8.16% | 8.41% |
| CIFAR-100 test error | **20.74%** | 21.77% | 22.98% |

GELU consistently outperforms ReLU and ELU across all five task domains.

> [!IMPORTANT]
> GELU became the default activation in transformer-based language models (BERT, GPT-2/3, and their successors), where smooth, non-monotonic activations improve gradient flow and expressivity compared to piecewise-linear ReLU.
