# Meta Information

- URL: [[2312.00752] Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- LICENSE: [Deed - Attribution 4.0 International - Creative Commons](https://creativecommons.org/licenses/by/4.0/)


> [!CAUTION]
> NOTE comments is my personal understanding and may contain errors.

# 1. Introduction

There is a foundation model which is called Transformer. However, it has quadratic complexity with respect to sequence length.

In 2022, structured state space sequence models (S4) have been proposed, which is combination of RNNs and CNNs, and it can handle long-range dependencies in sequences with linear complexity.

Here, the authors propose a new class of selective state space models.

## a. Selection Mechanism

- The critical weakness of prior S4-based models is that they use select the information whether is relevant or irrelevant.
- By parametrizing the S4 parametrizing based on the input, the authors propose a selection mechanism to select the relevant information.

## b. Hardware-aware Algorithm

- The essential constraint of prior all the models parameter is fixed (invariant) regardless of time and input to make computation efficient.
- The authors propose scan instead of convolution to make the model more hardware-efficient.

## c. Architecture

Mamba is a new architecture which combines prior SSM models with MLP blocks of Transformers, so it is simple and effective.

Selective SSMs guarantee the following properties:

1. High quality: Selectivity improves the model's performance on various tasks including language and genomics.
2. Fast training and inference: Computation and memory scale linearly with sequence length, which does not require previous elements as a cache.
3. Long context: The quality and efficiency enables training with 1M token context length.

> [!NOTE]
> Transformer computation is O($L^2$) because generating each token requires attending to all previous O(L) tokens. In contrast, Mamba's computation is O(L) because each token is generated using a fixed amount of computation that does not depend on the sequence length.

# 2. State Space Models

## Overview

```math
\begin{align}
t \in \mathbb{R} & : \text{time} \\
x(t) \in \mathbb{R} & : \text{input signal} \\
y(t) \in \mathbb{R} & : \text{output signal} \\
N \in \mathbb{Z}^{+} & : \text{state dimension} \\
A \in \mathbb{R}^{N \times N} & : \text{state matrix} \\
B \in \mathbb{R}^{N \times 1} & : \text{input matrix} \\
C \in \mathbb{R}^{1 \times N} & : \text{output matrix} \\
\Delta \in \mathbb{R} & : \text{discretization step} \\
h(t) \in \mathbb{R}^{N} & : \text{hidden state}
\end{align}
```

The continuous-time SSM is defined as follows:

```math
\begin{align}
h'(t) & = A h(t) + B x(t) & \text{`'` indicates derivative w.r.t. time} \\
y(t) & = C h(t)
\end{align}
```

The discrete-time SSM is defined as follows:

```math
\begin{align}
\bar{A} &= \exp(\Delta A) \in \mathbb{R}^{N \times N} \\
\bar{B} &= (\Delta A)^{-1} (\exp(\Delta A) - I) \Delta B \in \mathbb{R}^{N \times 1} \\
h_t & = \bar{A} h_{t-1} + \bar{B} x_t\\
y_t & = C h_t
\end{align}
```

The previous calculation is written in convolution form as follows:

```math
\begin{align}
L & : \text{sequence length} \\
\bar{K} & = (C \bar{B}, C \bar{AB}, \dots, C \bar{A}^{k}\bar{B}, \dots) \in \mathbb{L} \\
y & = \bar{K} * x \in \mathbb{R}^{L}
\end{align}
```
