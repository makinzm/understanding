# Meta Information

- URL: [SCANN: Synthesis of Compact and Accurate Neural Networks](https://arxiv.org/abs/1904.09090)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Hassantabar, S., Wang, Z., & Jha, N. K. (2019). SCANN: Synthesis of Compact and Accurate Neural Networks. arXiv preprint arXiv:1904.09090. Princeton University, Department of Electrical Engineering.

# SCANN: Synthesis of Compact and Accurate Neural Networks

SCANN is a framework for automatically synthesizing compact and accurate neural networks by dynamically changing the architecture during training. Unlike prior methods that fix the network depth beforehand, SCANN allows depth to change through iterative applications of three architecture-modifying operations: connection growth, neuron growth, and connection pruning. For non-image datasets with many features, DR+SCANN extends the framework by prepending a dimensionality reduction (DR) step that shrinks the input feature space before synthesis, enabling multiplicative compression gains.

**Applicability:** Practitioners targeting resource-constrained deployment (IoT sensors, edge devices, battery-operated hardware) who need significantly smaller and more energy-efficient networks than standard architectures, without hand-tuning layers or pruning schedules.

---

## Problem Setting

Standard DNNs overparameterize for their tasks. Manual compression via pruning or neural architecture search (NAS) either requires domain expertise or thousands of GPU days (e.g., NASNet requires ~2000 GPU days). SCANN automates architecture synthesis in ~20 GPU days for ImageNet-scale tasks and within minutes for small datasets, operating iteratively on an existing or freshly initialized network.

**Input:** A labeled dataset $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$ and an optional initial architecture.  
**Output:** A trained compact network $\mathcal{N}^*$ achieving competitive accuracy with a fraction of the parameters of the baseline.

---

## Dimensionality Reduction Step (DR)

For datasets with high feature counts relative to sample count, directly training even small networks is inefficient. DR first maps input $x \in \mathbb{R}^d$ to a lower-dimensional representation $z \in \mathbb{R}^k$ ($k \ll d$), which simultaneously reduces network input width and can remove noise.

Nine DR methods are evaluated via Scikit-learn:

| Category | Methods |
|----------|---------|
| Linear | PCA, Factor Analysis (FA), Independent Component Analysis (ICA) |
| Manifold | Isomap, Spectral Embedding |
| Random Projection | Gaussian (scaled), Gaussian (unit variance), Sparse binary, Sparse ternary |

**Random projection matrices** (Johnson-Lindenstrauss lemma guarantees distance preservation with dimension $O(\log d / \varepsilon^2)$ for approximation error $\varepsilon$):

```math
\begin{align}
  \phi_{ij}^{\text{Gaussian-scaled}} &\sim \mathcal{N}(0, 1/k) \\
  \phi_{ij}^{\text{sparse ternary}} &= \sqrt{3/k} \times \begin{cases} +1 & \text{w.p. } 1/6 \\ 0 & \text{w.p. } 2/3 \\ -1 & \text{w.p. } 1/6 \end{cases}
\end{align}
```

**Algorithm 1 — DR selection:**

```
Input: dataset D, compression ratios R = {r_1, ..., r_m}, DR methods M = {m_1, ..., m_9}
1. Normalize D: x ← (x - min(x)) / (max(x) - min(x))
2. best_acc ← 0
3. for each r in R, each m in M:
     D_r ← apply DR method m with ratio r to D
     Train a baseline MLP on D_r
     acc ← evaluate on validation set
     if acc > best_acc:
         best_acc ← acc
         best_config ← (r, m, D_r)
4. return best_config
```

---

## Three Core Architecture-Changing Operations

Each operation modifies the network topology and is followed by retraining before the next operation. The network is represented by a weight matrix $W$ and a binary mask (adjacency) matrix $C$, where $c_{ij} = 1$ indicates an active connection from neuron $i$ to neuron $j$.

### Connection Growth

Reactivates dormant or new connections based on the gradient signal. For neuron $i$ with activity $x_i = f(u_i)$ and neuron $j$ with pre-activity $u_j$:

```math
\begin{align}
  g_{ij} = \left| \frac{\partial \mathcal{L}}{\partial u_j} \cdot x_i \right|
\end{align}
```

If $g_{ij} > t$ for threshold $t$, set $c_{ij} = 1$ and initialize $w_{ij} = 0$.

> [!NOTE]
> The criterion follows Hebbian theory: neurons with correlated pre- and post-synaptic activity ("neurons that wire together fire together") are candidates for new connections.

**Full-growth variant:** Sets all $C$ entries to 1, restoring all possible connections without selectivity—used in aggressive recovery phases.

### Neuron Growth

Increases network capacity by duplicating the highest-activation neuron, then adding noise for symmetry breaking:

```
Input: network N, batch B, count K (neurons to add)
1. Forward propagate B through N
2. Select neuron i* = argmax_i u_i
3. Add new neuron j:
     c_{j·} ← c_{i*·},  c_{·j} ← c_{·i*}   (copy connectivity)
     w_{j·} ← w_{i*·} + ε,  w_{·j} ← w_{·i*} + ε  (copy weights + noise ε)
4. Repeat K times
```

Noise $\varepsilon$ breaks weight symmetry so the duplicate neuron diverges during subsequent training.

### Connection Pruning

Removes low-magnitude connections whose contribution to the output is negligible:

```
Input: network N, threshold t (e.g., p-th percentile of |w|)
1. For each active connection (i, j) where c_{ij} = 1:
     if |w_{ij}| < t:  set c_{ij} = 0
2. Remove any neuron with no remaining in-connections or out-connections
```

Pruned neurons that become orphaned (no in- or out-connections) are deleted. Accuracy lost by pruning is recovered through retraining.

---

## Training Schemes

SCANN provides three training schemes corresponding to different synthesis strategies.

### Scheme A — Constructive (Grow from Small)

Starts with a small network and grows it toward a target size. Supports skip connections and general feed-forward topologies (not restricted to layer-by-layer adjacency).

```
Start: N neurons (e.g., 300), target: M neurons (e.g., 500)
Loop until convergence:
  1. Connection growth (gradient-based, top ~80% candidates)
  2. Neuron growth (+5–10 neurons per iteration)
  3. Connection pruning (~25% of weights per iteration)
  4. Retrain for 10–20 epochs
```

### Scheme B — Aggressive Destructive

Starts large and aggressively prunes to a very small size, then recovers via growth cycles. Also supports general feed-forward topology.

```
Start: large network (e.g., 400 neurons)
1. Aggressively prune to minimal size
2. Loop:
     Full connection growth (restore 70–90% of connections)
     Connection pruning
     Retrain for 10–20 epochs
     (5–10 total iterations)
```

### Scheme C — MLP Destructive

Restricts topology to multilayer perceptrons (adjacent-layer connections only). Implements iterated dense-sparse-dense training.

```
Start: standard MLP with fixed layer count
1. Drastic pruning → sparse MLP
2. Loop:
     Full connection restoration (dense phase)
     Retrain
     Aggressive pruning (sparse phase)
```

> [!NOTE]
> Scheme A tends to produce the smallest networks when no accuracy constraint exists. Schemes B and C are preferred when baseline architectures and layer counts are fixed.

---

## DR+SCANN Pipeline

**Algorithm 2 — Full DR+SCANN:**

```
Input: dataset D, max iterations I_max
Step 1: DR Selection (Algorithm 1)
  (D_reduced, arch) ← DR-Select(D)

Step 2: SCANN Synthesis
  N ← initialize network with arch
  for iter = 1 to I_max:
    op ← select architecture-changing operation
    apply op to N
    train N on D_reduced
    acc ← evaluate on validation set
    if acc > best_acc: save N as N*
  return N*
```

The synergy arises because DR shrinks both the feature dimension and the first network layer width before SCANN further compresses, resulting in compression that multiplies rather than adds.

---

## Comparison with Related Methods

| Method | Depth change? | Skip connections? | GPU days (ImageNet) | Key mechanism |
|--------|--------------|-------------------|---------------------|---------------|
| SCANN (ours) | Yes (dynamic) | Yes | ~20 | Grow-prune + DR |
| NASNet | No | Yes | ~2000 | RL-based search |
| DARTS | No | Yes | ~4 | Gradient-based NAS |
| Han et al. pruning | No | No | — | Magnitude pruning |
| Lottery Ticket | No (fixed init) | No | — | Iterative pruning |

> [!IMPORTANT]
> The key differentiator of SCANN is dynamic depth: existing pruning and NAS methods fix the number of layers before training, whereas SCANN can add or remove entire neurons and their connections, effectively changing depth.

---

# Experiments

## Datasets

| Dataset | Train | Val | Test | Features | Classes |
|---------|-------|-----|------|----------|---------|
| Sensorless Drive (SenDrive) | 40,509 | 9,000 | 9,000 | 48 | 11 |
| Human Activity Recognition (HAR) | 5,881 | 1,471 | 2,947 | 561 | 6 |
| Musk v2 | 4,100 | 1,000 | 1,974 | 166 | 2 |
| Pen-Based Digits | 5,995 | 1,499 | 3,498 | 16 | 10 |
| Landsat Satellite Image | 3,104 | 1,331 | 2,000 | 36 | 6 |
| Letter Recognition | 10,500 | 4,500 | 5,000 | 16 | 26 |
| Epileptic Seizure | 6,560 | 1,620 | 3,320 | 178 | 2 |
| Smartphone HAR (SHAR) | 6,121 | 153 | 3,277 | 561 | 12 |
| DNA | 1,400 | 600 | 1,186 | 180 | 3 |
| MNIST | 60,000 | — | 10,000 | 784 (28×28) | 10 |
| ImageNet ILSVRC | ~1.2M | 50,000 | 100,000 | 224×224×3 | 1,000 |

## Hardware & Energy Model

- **Training hardware:** GPU (not specified; ~20 GPU days for ImageNet FC layers)
- **Energy model:** 130 nm CMOS standard cell library
  - Multiply-accumulate (MAC): 11.8 pJ
  - SRAM access: 34.6 pJ
  - Comparison: 6.16 fJ

## Key Results

**MNIST (vs. LeNet-5 Caffe baseline, 430.5K parameters, 0.72% error):**
- Scheme C: 9.3K parameters → **46.3× compression**, same 0.72% error
- Scheme B: 19.3K parameters → 22.3× compression
- Scheme A: 184.6K parameters → 2.3× compression, **improved** to 0.68% error

**ImageNet (VGG-16 baseline: 138.4M parameters, 28.4% top-1 error):**
- SCANN VGG-16: 17.2M parameters → **8.0× compression**, improved to 26.7% top-1

**ImageNet (MobileNetV2 baseline: 3.4M parameters, 28.0% top-1 error):**
- SCANN MobileNetV2: 2.6M parameters → 1.3× compression, 28.2% top-1

**Non-image datasets (DR+SCANN):**
- Parameter compression range: 1.2× – 5078.7× (geometric mean: **82.1×**)
- 6 of 9 datasets show accuracy improvement (0.41%–9.43%)
- Energy reduction: up to **5082× (Seizure dataset)**, geometric mean ~288×

## Optimizer

- Standard SGD / Adam (not explicitly stated for all experiments); learning rates and schedules follow each baseline architecture's standard training protocol.
