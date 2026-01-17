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

# 3. Selective State Space Models

## 3.1. Motivation: Selection as a Means of Compression

Sequence modeling is compressiong context into a smlaller state, but it is a problem.

There is a trade-off between capacity and efficiency.

## 3.2. Improving SSMs with Selection

### Selective State Space Model (S6)

**Dimensions:**
```math
\begin{align}
B & \in \mathbb{Z}^{+} & : \text{batch size} \\
L & \in \mathbb{Z}^{+} & : \text{sequence length} \\
D & \in \mathbb{Z}^{+} & : \text{model dimension (number of independent SSMs)} \\
N & \in \mathbb{Z}^{+} & : \text{state dimension (for each SSM)}
\end{align}
```

**Input and State:**
```math
\begin{align}
x & \in \mathbb{R}^{B \times L \times D} & : \text{input sequence} \\
h_t & \in \mathbb{R}^{B \times D \times N} & : \text{hidden state at time } t \\
y & \in \mathbb{R}^{B \times L \times D} & : \text{output sequence}
\end{align}
```

> [!NOTE]
> S6 contains D independent SSMs, each processing one channel of the input.

**Parameters:**
```math
\begin{align}
A & \in \mathbb{R}^{D \times N} & : \text{structured representation of D independent } N \times N \text{ matrices}
\end{align}
```

> [!NOTE]  
> **Structured Matrix Representation**
> 
> HiPPO Matrix from [[2111.00396] Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
> 
> Each $A_d \in \mathbb{R}^N$ is a **compressed representation** of a structured $N \times N$ matrix $A_d^{(N \times N)}$.
> The actual $N \times N$ matrix is reconstructed when needed using one of these structures:
> 
> 1. **Diagonal**: $A_d^{(N \times N)} = \text{diag}(A_d) \in \mathbb{R}^{N \times N}$
>    - Storage: $N$ parameters per SSM
> 
> 2. **Diagonal + Low-Rank (DPLR)**: $A_d^{(N \times N)} = \Lambda_d - P_d Q_d^\top$
>    - Storage: $N + 2Nr$ parameters (where $r \ll N$)
>    - Used in S4 for efficiency
> 
> 3. **HiPPO Matrix**: Specific structured matrix from theory
>    - Can be parameterized with few values
> 
> **Why use structured representation?**
> - Memory efficiency: $O(DN)$ instead of $O(DN^2)$
> - Computational efficiency: Fast algorithms for $\exp(\Delta A)$
> - Better initialization: Theoretically motivated structures (e.g., HiPPO)

**Selection mechanism (input-dependent parameters):**
```math
\begin{align}
B(x) & = s_B(x) \in \mathbb{R}^{B \times L \times N} & : \text{input-dependent input matrix} \\
C(x) & = s_C(x) \in \mathbb{R}^{B \times L \times N} & : \text{input-dependent output matrix} \\
\Delta(x) & = \tau_\Delta(\Theta_\Delta + s_\Delta(x)) \in \mathbb{R}^{B \times L \times D} & : \text{input-dependent step size}
\end{align}
```

where $\Theta_\Delta \in \mathbb{R}^D$ is a learnable parameter, and:
```math
\begin{align}
s_B(x) &= \text{Linear}_N(x) & : \mathbb{R}^{B \times L \times D} \to \mathbb{R}^{B \times L \times N} \\
s_C(x) &= \text{Linear}_N(x) & : \mathbb{R}^{B \times L \times D} \to \mathbb{R}^{B \times L \times N} \\
s_\Delta(x) &= \text{Broadcast}_D(\text{Linear}_1(x)) & : \mathbb{R}^{B \times L \times D} \to \mathbb{R}^{B \times L \times D} \\
\tau_\Delta(z) &= \log(1 + \exp(z)) & : \text{softplus activation}
\end{align}
```

> [!NOTE]
> **Why softplus for $\Delta$?**
> - $\Delta$ must be positive (discretization step size)
> - $\text{softplus}(z) > 0$ for all $z \in \mathbb{R}$
> - Smoother than ReLU, better gradients
> - Connection to RNN gating mechanisms (Section 3.5)

**At each time step $t \in \{1, \ldots, L\}$:**
```math
\begin{align}
x_t &= x[:, t, :] \in \mathbb{R}^{B \times D} & : \text{input at time } t \\
B_t &= B(x)[:, t, :] \in \mathbb{R}^{B \times N} & : \text{input matrix at time } t \\
C_t &= C(x)[:, t, :] \in \mathbb{R}^{B \times N} & : \text{output matrix at time } t \\
\Delta_t &= \Delta(x)[:, t, :] \in \mathbb{R}^{B \times D} & : \text{step size at time } t
\end{align}
```

**Discretization (for each batch $b$ and dimension $d$):**

For each $(b, d) \in [B] \times [D]$, we discretize independently:
```math
\begin{align}
\bar{A}_{t,b,d} &= \text{discretize}_A(\Delta_{t,b,d}, A_d) \in \mathbb{R}^{N \times N} \\
\bar{B}_{t,b,d} &= \text{discretize}_B(\Delta_{t,b,d}, A_d, B_{t,b}) \in \mathbb{R}^{N}
\end{align}
```

> [!NOTE]
> **Discretization Process**
> 
> The structured representation $A_d \in \mathbb{R}^N$ is first expanded to the full matrix $A_d^{(N \times N)}$, then discretized.

Using Zero-Order Hold (ZOH) discretization:
```math
\begin{align}
\bar{A}_{t,b,d} &= \exp(\Delta_{t,b,d} \cdot A_d^{(N \times N)}) \in \mathbb{R}^{N \times N} \\
\bar{B}_{t,b,d} &= (\Delta_{t,b,d} \cdot A_d^{(N \times N)})^{-1}(\bar{A}_{t,b,d} - I) \cdot (\Delta_{t,b,d} \cdot B_{t,b}) \in \mathbb{R}^{N}
\end{align}
```

**Tensor form:**
```math
\begin{align}
\bar{A}_t & \in \mathbb{R}^{B \times D \times N \times N} & : \text{discretized state matrices} \\
\bar{B}_t & \in \mathbb{R}^{B \times D \times N} & : \text{discretized input matrices}
\end{align}
```

> [!NOTE]
> Although $A$ is stored as $\mathbb{R}^{D \times N}$, $\bar{A}_t$ requires the full $N \times N$ form for each $(b,d)$ pair.

**Recurrence (time-varying):**

For each $(b, d) \in [B] \times [D]$:
```math
\begin{align}
h_{t,b,d} &= \bar{A}_{t,b,d} \cdot h_{t-1,b,d} + \bar{B}_{t,b,d} \cdot x_{t,b,d} & \in \mathbb{R}^{N} \\
y_{t,b,d} &= C_{t,b}^\top \cdot h_{t,b,d} & \in \mathbb{R}
\end{align}
```

**Tensor form:**
```math
\begin{align}
h_t & \in \mathbb{R}^{B \times D \times N} & : \text{hidden state at time } t \\
y_t & \in \mathbb{R}^{B \times D} & : \text{output at time } t
\end{align}
```

> [!IMPORTANT]
> **Key difference from S4:**
> 
> | Aspect | S4 | S6 |
> |--------|----|----|
> | Parameters | Time-invariant: $\Delta, B, C$ are constants | Time-varying: $\Delta_t, B_t, C_t$ depend on input $x$ |
> | Computation | Recurrence OR convolution | ONLY recurrence (scan) |
> | Efficiency | Parallel convolution enables fast training | Sequential scan is slower |
> | Effectiveness | Cannot select/filter inputs | Can selectively focus on relevant inputs |
> | Context compression | Fixed dynamics | Input-dependent dynamics |

> [!NOTE]
> **Why can't S6 use convolution?**
> 
> Convolution form requires time-invariance:
> - S4: $\bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots)$ is a fixed kernel
> - S6: $\bar{A}_t, \bar{B}_t, C_t$ change at each time step → no fixed kernel exists
