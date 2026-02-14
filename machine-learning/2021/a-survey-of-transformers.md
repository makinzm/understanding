# Meta Information

- URL: [A Survey of Transformers](https://arxiv.org/abs/2106.04554)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Lin, T., Wang, Y., Liu, X., & Qiu, X. (2022). A Survey of Transformers. *AI Open*, 3, 111–132.

---

# Overview

This survey provides a systematic taxonomy of Transformer variants (called "X-formers") along three axes: (1) module-level modifications (attention, position encoding, layer normalization, FFN), (2) architecture-level changes (connectivity, adaptive computation, divide-and-conquer), and (3) pre-training paradigms and application domains. The paper is useful for researchers who want to understand the design space of Transformers and practitioners choosing among variants for long-sequence, low-resource, or domain-specific tasks.

---

# 1. Vanilla Transformer

## 1.1 Self-Attention

Given input sequence length $T$ and hidden dimension $D$, the input $X \in \mathbb{R}^{T \times D}$ is projected into queries, keys, and values:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

where $W^Q, W^K \in \mathbb{R}^{D \times D_k}$ and $W^V \in \mathbb{R}^{D \times D_v}$.

Scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{D_k}}\right) V$$

- Input: $Q \in \mathbb{R}^{T \times D_k}$, $K \in \mathbb{R}^{M \times D_k}$, $V \in \mathbb{R}^{M \times D_v}$ (for cross-attention $M \neq T$; for self-attention $M = T$)
- Output: $Z \in \mathbb{R}^{T \times D_v}$
- Complexity: $O(T^2 \cdot D)$ time, $O(T^2)$ memory (from the attention matrix)

## 1.2 Multi-Head Attention (MHA)

$H$ parallel attention heads, each with its own projections $W_h^Q, W_h^K, W_h^V$:

$$\text{head}_h = \text{Attention}(Q W_h^Q,\ K W_h^K,\ V W_h^V)$$

$$\text{MultiHeadAttn}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

- Each head dimension: $D_k = D_v = D / H$
- Output projection: $W^O \in \mathbb{R}^{D \times D}$

## 1.3 Position-wise Feed-Forward Network (FFN)

Applied independently to each token position $t$:

$$\text{FFN}(H') = \text{ReLU}(H' W^1 + b^1) W^2 + b^2$$

- $W^1 \in \mathbb{R}^{D \times D_f}$, $W^2 \in \mathbb{R}^{D_f \times D}$, typically $D_f = 4D$
- Complexity: $O(T \cdot D^2)$ — dominates over attention when $T \ll D$

## 1.4 Residual Connections and Layer Normalization

Each sub-layer (attention or FFN) uses:
$$H^{(l)} = \text{LayerNorm}(H^{(l-1)} + \text{Sublayer}(H^{(l-1)}))$$

**Post-LN** (original): normalization after residual addition — harder to train without warmup.
**Pre-LN** (common variant): normalization before sub-layer input — more stable gradients.

## 1.5 Complexity Summary

| Module       | Time Complexity  | Space Complexity |
|-------------|-----------------|-----------------|
| Self-Attention | $O(T^2 \cdot D)$ | $O(T^2)$      |
| FFN          | $O(T \cdot D^2)$ | $O(T \cdot D)$ |

As sequence length $T$ grows, self-attention becomes the bottleneck. For $T \gg D$, the $T^2$ term dominates; this motivates most X-former work.

---

# 2. Attention Mechanism Variants

## 2.1 Sparse Attention

Instead of attending to all $T$ tokens, each query attends to a restricted subset. Five atomic sparse patterns:

| Pattern | Description | Example Models |
|---------|-------------|----------------|
| **Global** | A set of hub tokens attends to/from all positions | Longformer, BigBird |
| **Band** | Each token attends to $w$ local neighbors (sliding window) | Longformer, LongT5 |
| **Dilated** | Band with dilation stride $d$, covering wider context | Longformer |
| **Random** | Each query attends to $r$ randomly chosen keys | BigBird |
| **Block Local** | Sequence partitioned into blocks; attend within blocks | Blockwise Attention |

Compound patterns mix these: BigBird = global + band + random; Longformer = global + band + dilated.

**Content-based sparse attention** selects positions dynamically:
- **Routing Transformer**: K-means clustering of queries and keys into $k$ clusters; each query attends only within its cluster
- **Reformer**: Locality-Sensitive Hashing (LSH) groups similar queries/keys into buckets

### Reformer LSH Algorithm

$$h(x) = \arg\max([xR;\, -xR])$$

where $R \in \mathbb{R}^{D_k \times b/2}$ is a random matrix. Queries and keys hashed to the same bucket attend to each other. Reduces complexity from $O(T^2)$ to $O(T \log T)$.

## 2.2 Linearized Attention

Approximates the softmax kernel by a decomposable feature map $\phi$:

$$\exp(q^\top k / \sqrt{D}) \approx \phi(q)^\top \phi(k)$$

This allows reordering the computation:

$$Z_i = \frac{\phi(q_i) \sum_j \phi(k_j)^\top v_j}{\phi(q_i) \sum_j \phi(k_j)}$$

Computing $\sum_j \phi(k_j)^\top v_j$ once for all $j$ reduces complexity to $O(T \cdot D^2_k)$ (linear in $T$).

| Model | Feature Map $\phi$ |
|-------|-------------------|
| Linear Transformer | $\phi_i(x) = \text{elu}(x_i) + 1$ |
| Performer | Random Fourier features approximating Gaussian kernel |
| RFA | Random features for arbitrary kernels |

> [!IMPORTANT]
> Linearized attention loses the normalizing softmax, which can affect numerical stability and expressiveness. Performer addresses this via FAVOR+ (Fast Attention Via positive Orthogonal Random features).

## 2.3 Low-Rank Approximation

Exploits the empirical observation that the $T \times T$ attention matrix is approximately low-rank.

**Linformer**: Projects $K$ and $V$ from $T \times D$ to $k \times D$ using learned projection $E \in \mathbb{R}^{T \times k}$:
$$\bar{K} = E^\top K \in \mathbb{R}^{k \times D}, \quad \bar{V} = E^\top V \in \mathbb{R}^{k \times D}$$
Reduces complexity to $O(T \cdot k \cdot D)$ where $k \ll T$.

**Nyströmformer**: Selects $m$ landmark (Nyström) points $\tilde{Q}, \tilde{K} \in \mathbb{R}^{m \times D_k}$ and approximates:
$$\tilde{A} = \text{softmax}(Q \tilde{K}^\top)\, [\text{softmax}(\tilde{Q} \tilde{K}^\top)]^{-1}\, \text{softmax}(\tilde{Q} K^\top)$$
Complexity: $O(Tm)$ where $m \ll T$.

## 2.4 Query Prototyping and Memory Compression

Reduces the number of key-value pairs processed:

- **Compressed Attention** (Liu et al.): Strided convolution over $K$ and $V$ to reduce sequence length
- **Set Transformer**: Inducing points (learned anchor queries) aggregate information before attending to full input
- **Funnel Transformer**: Progressively reduces sequence length by pooling hidden states at intermediate layers

## 2.5 Attention with Priors

Incorporates inductive biases beyond content-based attention:

- **Positional bias**: Add a learned or formula-based scalar $b_{ij}$ to $q_i k_j^\top / \sqrt{D}$ before softmax (e.g., ALiBi uses a linear bias $-|i - j|$)
- **Gaussian attention**: Prior centered on the diagonal — each position biased toward attending to nearby positions (local assumption)
- **Cross-layer priors**: Realformer reuses attention maps from the previous layer as priors, then refines them

## 2.6 Multi-Head Mechanism Improvements

| Technique | Description |
|-----------|-------------|
| Head pruning | Remove redundant heads at inference time |
| Diverse heads | Regularize to encourage different heads to attend differently |
| Talking-Heads (Shazeer et al.) | Linear mixing across heads before and after softmax |
| Multi-Query Attention | All heads share one set of $K, V$ projections; reduces memory |
| Adaptive span | Each head learns a different context window size |

---

# 3. Position Encoding

## 3.1 Absolute Positional Encodings

**Sinusoidal (Vaswani et al., 2017)**:
$$\text{PE}(t)_{2i} = \sin\!\left(\frac{t}{10000^{2i/D}}\right), \quad \text{PE}(t)_{2i+1} = \cos\!\left(\frac{t}{10000^{2i/D}}\right)$$

Allows model to learn attention based on position difference $i - j$ via dot products, since $\text{PE}(t)^\top \text{PE}(t+k)$ depends only on $k$.

**Learned absolute embeddings**: Train a lookup table $E \in \mathbb{R}^{T_{\max} \times D}$; cannot generalize beyond $T_{\max}$.

## 3.2 Relative Positional Encodings

Encode the offset $\delta = i - j$ between query at position $i$ and key at position $j$:

**Shaw et al. (2018)**:
$$A_{ij} = \frac{(q_i)(k_j + r_{ij})^\top}{\sqrt{D_k}}, \quad r_{ij} = R_{\text{clip}(i-j, -k, k)}$$

where $R$ is a learned embedding of clipped relative distances.

**Transformer-XL (disentangled)**:
$$A_{ij} = q_i k_j^\top + q_i (R_{i-j} W^{K,R})^\top + u^1 k_j^\top + u^2 (R_{i-j} W^{K,R})^\top$$

where $u^1, u^2$ are global bias vectors and $R$ is a sinusoidal relative PE.

**DeBERTa** (content-to-position and position-to-content):
$$A_{ij} = q_i k_j^\top + q_i (r_{ij} W^{K,R})^\top + k_j (r_{ij} W^{Q,R})^\top$$

## 3.3 Other Approaches

- **Rotary PE (RoPE)**: Multiplies $q$ and $k$ by rotation matrices so that $q_i^\top k_j$ depends only on $i - j$; generalizes beyond training length
- **ALiBi**: No position embeddings; adds a linear penalty $-m \cdot |i - j|$ to attention logits, where $m$ is a head-specific slope

---

# 4. Layer Normalization Variants

| Variant | Formula | Notes |
|---------|---------|-------|
| Post-LN | $\text{LN}(x + \text{Sub}(x))$ | Original; unstable at large scale without warmup |
| Pre-LN | $x + \text{Sub}(\text{LN}(x))$ | Stable training; slightly reduced performance |
| RMS Norm | $\text{RMSNorm}(x) = x / \text{RMS}(x)$ | Removes mean-centering; faster |
| Sandwich-LN | LN before and after sub-layer | Combines stability with expressiveness |

---

# 5. Feed-Forward Network Variants

The standard FFN uses ReLU. Variants:

| Variant | Activation | Notes |
|---------|-----------|-------|
| GELU FFN | GELU instead of ReLU | Used in GPT-2, BERT |
| GLU FFN | $x \odot \sigma(x W_g)$ gating | Better gradient flow |
| Mixture-of-Experts (MoE) | Top-$k$ experts per token | Sparse; scales parameters without compute increase |
| Switch Transformer | MoE with $k=1$ (single expert) | Efficient routing; load balancing loss |

---

# 6. Architecture-Level Variants

## 6.1 Lightweight Transformers

- **Lite Transformer**: Replaces one attention head with a convolution per layer; reduces complexity while preserving performance for long-range + local patterns
- **DeLighT**: Depth-wise lightweight transformer; expands input, applies grouped linear transform, contracts; reduces parameters by 2–3×

## 6.2 Recurrence-Based (Divide and Conquer)

- **Transformer-XL**: Processes segments sequentially; previous segment hidden states are cached and attended as context, enabling $O(T + M)$ memory with $M$-length memory
- **Compressive Transformer**: Compresses old memories before discarding, maintaining fine-grained recent and coarse distant context

## 6.3 Hierarchical Transformers

- **HIBERT**: Sentence-level encoder whose output feeds document-level encoder; each level is a Transformer
- **Funnel Transformer**: Pools sequence length after each block, reducing $T$ progressively; upsamples for generation tasks

## 6.4 Adaptive Computation

- **Early exit**: Each layer outputs a confidence score; if confident enough, skip remaining layers for that token
- **Universal Transformer**: Weight-sharing across layers with learned halting probability; more computation for difficult tokens

---

# 7. Pre-trained Transformers

| Type | Architecture | Examples |
|------|-------------|---------|
| Encoder-only | Bidirectional, masked LM | BERT, RoBERTa, ALBERT |
| Decoder-only | Causal LM | GPT-2, GPT-3 |
| Encoder-Decoder | Sequence-to-sequence | BART, T5, mT5, Switch Transformer |

Pre-training objectives:
- **Masked Language Modeling (MLM)**: Predict masked tokens from bidirectional context (BERT)
- **Causal LM (CLM)**: Predict next token left-to-right (GPT)
- **Span prediction**: Predict contiguous masked spans (T5, SpanBERT)
- **Denoising**: Reconstruct corrupted sequences (BART: shuffle, deletion, infilling)

---

# 8. Applications

| Domain | Representative Models | Task Examples |
|--------|----------------------|--------------|
| NLP | BERT, GPT-3, T5 | Classification, NER, translation, summarization |
| Computer Vision | ViT, DeiT, Swin | Image classification, detection, segmentation |
| Audio | Wav2Vec 2.0, Speech Transformer | ASR, music generation |
| Multimodal | DALL-E, CogView, VL-BERT | Image captioning, text-to-image, VQA |

> [!NOTE]
> ViT (Vision Transformer) splits an image into $16 \times 16$ patches, flattens them, and applies standard Transformer encoder. It outperforms CNNs at large scale, confirming Transformer generality beyond NLP.

---

# 9. Comparison: Key Efficiency Trade-offs

| Model | Complexity | Key Idea | Trade-off |
|-------|-----------|---------|-----------|
| Vanilla Transformer | $O(T^2 D)$ | Full attention | Exact, expensive |
| Longformer | $O(T \cdot w)$ | Band + global | Loses some long-range |
| Reformer | $O(T \log T)$ | LSH bucketing | Approximate; hashing overhead |
| Linformer | $O(T \cdot k)$ | Low-rank projection | Fixed projection; less flexible |
| Performer | $O(T \cdot D^2)$ | Random features | Approximate softmax |
| Nyströmformer | $O(T \cdot m)$ | Landmark points | Depends on landmark quality |
| Transformer-XL | $O(T \cdot (T+M))$ | Segment recurrence | Sequential; limited parallelism |

---

# Experiments

- **Dataset**: No original experiments; all results cited from primary papers (language modeling benchmarks: WikiText-103, enwiki8, text8, BooksCorpus; downstream: GLUE, SuperGLUE; long-document: document-level classification, long QA)
- **Hardware**: Not applicable (survey paper)
- **Optimizer**: Not applicable
- **Results**: Summarized comparisons show: (1) sparse attention models approach full-attention performance on long sequences; (2) linearized methods have larger gaps on tasks requiring fine-grained attention (translation vs. LM); (3) pre-trained encoder-decoder models (T5-11B) achieve best performance on most NLP benchmarks at time of writing

> [!TIP]
> For a detailed comparison of efficient Transformers with code, see the [Long Range Arena benchmark](https://arxiv.org/abs/2011.04006), which systematically evaluates X-formers across tasks requiring long sequence understanding.
