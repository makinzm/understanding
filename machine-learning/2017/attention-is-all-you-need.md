# Meta Information

- URL: [[1706.03762] Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

# Model Architecture

From the beginning of chapter 3 and 3.5, $x \in \mathbb{R}^n \rightarrow z \in \mathbb{R}^{n \times d_{model}} \rightarrow y \in \mathbb{R}^m$.

We use N times block of encoder and decoder.

## 1. Embedding

Firstly, the input $x$ is passed through an embedding layer.

Output of embedding layer is $z \in \mathbb{R}^{n \times d_{model}}$.

> [!NOTE]
> "we use learned embeddings to convert the input tokens and output tokens to vectors"

## 2. Positional Encoding
And then, $z$ is added with a positional encoding ( depending on the position, the dimension and odd/even posiiton).

Output of this addition is also $z \in \mathbb{R}^{n \times d_{model}}$.

## 3. MHA

## 3.1. Basic Idea

And then, passed throgh Multi-Head Attention layer.
MHA consists of several scaled dot-product attention layers.

- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix
- $K \in \mathbb{R}^{n \times d_k}$: Key matrix
- $V \in \mathbb{R}^{n \times d_v}$: Value matrix
- softmax($\frac{QK^T}{\sqrt{d_k}}$) $\in \mathbb{R}^{n \times n}$: Attention weights matrix (each row sums to 1 not each column)
- Attention(Q, K, V) = softmax($\frac{QK^T}{\sqrt{d_k}}$)V $\in \mathbb{R}^{n \times d_v}$: Scaled dot-product attention output
- Let $A_i \in \mathbb{R}^{n \times d_v}$, Concat($A_1, A_2, ..., A_h$)　$\in \mathbb{R}^{n \times hd_v}$: Concatenated output of h attention heads
- MHA(Q, K, V) = Concat($head_1, head_2, ..., head_h$)$W^O$ $\in \mathbb{R}^{n \times d_{model}}$: Multi-Head Attention output
  - where $head_i$ = Attention($QW_i^Q$, $KW_i^K$, $VW_i^V$)
  - $W^O \in \mathbb{R}^{hd_v \times d_{model}}$
  - $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
  - $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
  - $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
  - $i$ is the index of attention head (number of head is a hyperparameter)

## 3.2. How to use MHA in Model

### 3.2.1. Encoder & Decoder MHA (Self-Atttention)

linear projections of queries, keys and values are created from the same input $z$. i.e. $Q = zW_i^Q$, $K = zW_i^K$, $V = zW_i^V$ for each head $i$ ($W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$).

> [!NOTE]
> "we found it beneficial to linearly project the queries, keys and values h times with different"
> "The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.

However, we need to mask out in encoder attention, but in encoder attention we can use all the positions.

> [!NOTE]
> "We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. 

### 3.2.2. Decoder-Encoder MHA

> [!NOTE]
> "the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder"


## 4. Add & Norm
After MHA, a residual connection is employed followed by layer normalization.
Let the output of MHA be $mha\_out \in \mathbb{R}^{n \times d_{model}}$.

Output of this layer is $z \in \mathbb{R}^{n \times d_{model}}$.

$z = LayerNorm(z + mha\_out)$
- LayerNorm(x) = $\frac{x - \mu}{\sigma} * \gamma + \beta$
  - where $\mu$ is mean of x, $\sigma$ is standard deviation of x, $\gamma$ and $\beta$ are learnable parameters.

> [!TIP]
> [LayerNorm — PyTorch 2.9 documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

## 5. Feed Forward

Each position is passed through a fully connected feed-forward network (FFN) independently and identically.

FFN(x) = max(0, $xW_1 + b_1$)$W_2 + b_2$
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- $b_1 \in \mathbb{R}^{d_{ff}}$, $b_2 \in \mathbb{R}^{d_{model}}$

The output dimension is $\mathbb{R}^{n \times d_{model}}$.

## 6. Final Linear & Softmax

Finally, the output of decoder is passed through a linear layer followed by a softmax layer to produce the final output probabilities.

- Linear layer: $y = zW_{final} + b_{final}$
  - where $W_{final} \in \mathbb{R}^{d_{model} \times V}$, $b_{final} \in \mathbb{R}^{V}$, $V$ is the vocabulary size.

> [!IMPORTANT]
> There is no explanation about the final output dimension in the paper. However, Figure 3~5 supports that the final output dimension is $\mathbb{R}^{V}$.

# Difference between Transformer and Seq2Seq with Attention

- Parallelization: Transformer allows for parallel processing of input sequences, while traditional Seq2Seq models with attention are often sequential in nature ( from Section 4 of the paper)
- Path lengths: $O(1)$ in Transformer vs. $O(n)$ in RNN (from Table 1 of the paper)

# Experiments

- Dataset: 
  - WMT 2014 English-German dataset
  -  WMT 2014 English-French dataset
- Hardware: 8 NVIDIA P100 GPUs
- Optimizer: Adam with learning rate schedule
- Regularization: Dropout and label smoothing

> [!TIP]
> [マイナーな正則化手法「ラベル平滑化」は実は効果アリアリらしい #機械学習 - Qiita](https://qiita.com/wing_man/items/a1d5ab1bba7d763d9369)
> [全ての学習率スケジューリングを過去にするOptimizer](https://zenn.dev/dena/articles/6f04641801b387)

- Results:
    - English-German: state-of-the-art BLEU score of 28.4
    - English-French: state-of-the-art BLEU score of 41.0
