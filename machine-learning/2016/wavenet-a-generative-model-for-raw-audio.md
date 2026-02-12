# Meta Information

- URL: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., & Kavukcuoglu, K. (2016). WaveNet: A Generative Model for Raw Audio. arXiv:1609.03499. Google DeepMind.

# WaveNet: A Generative Model for Raw Audio

WaveNet is an autoregressive deep generative model that synthesizes raw audio waveforms sample by sample. Unlike traditional speech synthesis pipelines that use hand-crafted vocoders or concatenative approaches, WaveNet learns a direct mapping from conditioning signals (e.g., text linguistic features, speaker identity) to audio samples. The architecture is based on dilated causal convolutions, which enable large receptive fields with limited depth and no recurrent computation.

**Applicability:** WaveNet is applicable to any task requiring high-fidelity audio generation—text-to-speech (TTS), voice conversion, music generation, and audio modeling. It is particularly suited for tasks where naturalness of the output matters more than inference speed. The autoregressive nature makes it slow for real-time inference but it excels in offline generation quality.

## 1. Autoregressive Formulation

WaveNet models the joint probability of a raw audio waveform $\mathbf{x} = \{x_1, x_2, \ldots, x_T\}$ as a product of conditional distributions:

$$p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

Each audio sample $x_t \in \{0, 1, \ldots, 255\}$ (after quantization) is predicted from a softmax over all preceding samples in the sequence. This is a fully autoregressive model: generation proceeds sequentially, and each prediction depends on all prior outputs.

**Comparison with RNNs:** LSTMs also model sequences autoregressively but process one step at a time via hidden states. WaveNet replaces recurrence with dilated convolutions, enabling full parallelization during training (all $T$ predictions are computed simultaneously), while generation remains sequential.

## 2. Causal and Dilated Convolutions

### 2.1 Causal Convolutions

A causal convolution ensures that the prediction for $x_t$ only depends on $x_1, \ldots, x_{t-1}$. This is implemented by shifting the convolution output so that position $t$ in the output only sees positions $\leq t-1$ in the input. No future information leaks into the prediction.

### 2.2 Dilated Causal Convolutions

Standard causal convolutions have a receptive field that grows linearly with depth. Dilated convolutions skip over inputs by a factor (the dilation rate), growing the receptive field exponentially:

- Layer 0: dilation $d = 1$ (standard convolution)
- Layer 1: dilation $d = 2$ (skip every other sample)
- Layer 2: dilation $d = 4$
- ...
- Layer $k$: dilation $d = 2^k$

The dilation sequence $1, 2, 4, 8, \ldots, 512$ is repeated across multiple blocks. With 10 layers per block and dilation up to 512, a single block covers a receptive field of $1 + 2 \times (1 + 2 + 4 + \cdots + 512) = 1023$ samples. Stacking multiple blocks multiplies this further while keeping parameter count modest.

> [!NOTE]
> "We found that using causal convolutions with dilations allows the model to have a very large receptive field with a small number of parameters." — Paper, Section 2.3

**Advantages over RNNs:** Training is fully parallelizable because no step depends on the hidden state from the previous step. The fixed computational graph makes gradient flow more predictable than through recurrent backpropagation.

### 2.3 Receptive Field Size

For $B$ blocks each with layers $k = 0, 1, \ldots, K-1$ and filter width $f$:

$$\text{Receptive field} = B \times (f - 1) \times \sum_{k=0}^{K-1} 2^k + 1 = B \times (f - 1) \times (2^K - 1) + 1$$

With $f = 2$, $K = 10$, $B = 5$: receptive field $= 5 \times 1023 + 1 = 5116$ samples. At 16 kHz, this covers ~320 ms.

## 3. Gated Activation Unit

Each dilated convolution layer uses a gated activation function inspired by gated recurrent units:

$$\mathbf{z} = \tanh(W_{f,k} \ast \mathbf{x}) \odot \sigma(W_{g,k} \ast \mathbf{x})$$

where:
- $\mathbf{x} \in \mathbb{R}^{T \times r}$ is the input to the layer ($r$ = residual channels, e.g., 32 or 64)
- $W_{f,k}$ and $W_{g,k} \in \mathbb{R}^{f \times r \times r}$ are learned filter and gate kernels at layer $k$
- $\ast$ denotes dilated causal convolution
- $\odot$ denotes element-wise multiplication
- $\mathbf{z} \in \mathbb{R}^{T \times r}$ is the layer output

The tanh branch learns "what information to pass forward"; the sigmoid branch acts as a gate controlling "how much" passes. This architecture outperforms ReLU activations for audio modeling.

## 4. Residual and Skip Connections

WaveNet uses both residual and skip connections to stabilize training of deep models:

**Residual connection:** The input $\mathbf{x}$ is added to the gated output via a $1 \times 1$ convolution, forming the input to the next layer:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + W_{\text{res}} \mathbf{z}_k$$

**Skip connection:** Each layer's output is also projected via a separate $1 \times 1$ convolution and accumulated into a global sum $\mathbf{s}$:

$$\mathbf{s} = \sum_{k} W_{\text{skip},k} \mathbf{z}_k$$

The final output stack processes $\mathbf{s}$ through ReLU → $1 \times 1$ conv → ReLU → $1 \times 1$ conv → softmax over 256 classes.

> [!IMPORTANT]
> The skip connections ensure that all layers contribute directly to the output distribution, not just the final layer. This prevents gradient vanishing in very deep models (often 30–60 dilated conv layers).

## 5. Output Quantization: μ-law Companding

Raw audio is stored as 16-bit integers (65,536 levels). WaveNet quantizes samples to 256 levels using the μ-law companding transformation:

$$f(x_t) = \text{sign}(x_t) \cdot \frac{\ln(1 + \mu |x_t|)}{\ln(1 + \mu)}, \quad \mu = 255$$

where $x_t \in (-1, 1)$ is the normalized sample amplitude. The mapping is logarithmic: it uses finer resolution near zero (where most audio energy resides) and coarser resolution at large amplitudes. This nonlinear quantization preserves perceptual quality significantly better than uniform (linear) quantization at 256 levels.

## 6. Conditional Generation

WaveNet supports conditioning on external signals $\mathbf{h}$ to control generation.

### 6.1 Global Conditioning (Time-Invariant)

A single embedding vector $\mathbf{h} \in \mathbb{R}^{d_h}$ (e.g., one-hot speaker identity) is projected and broadcast across all time steps:

$$\mathbf{z} = \tanh(W_{f,k} \ast \mathbf{x} + V_{f,k}^{\top} \mathbf{h}) \odot \sigma(W_{g,k} \ast \mathbf{x} + V_{g,k}^{\top} \mathbf{h})$$

$V_{f,k}^{\top} \mathbf{h} \in \mathbb{R}^r$ is broadcast over all time steps $T$.

### 6.2 Local Conditioning (Time-Varying)

A time-series conditioning signal $\mathbf{h} \in \mathbb{R}^{T' \times d_h}$ (e.g., mel-spectrogram, linguistic features at 200 Hz frame rate) is upsampled to audio resolution $T$ via transposed convolution $y = f(\mathbf{h}) \in \mathbb{R}^{T \times d_h}$, then used per time step:

$$\mathbf{z} = \tanh(W_{f,k} \ast \mathbf{x} + V_{f,k} \ast \mathbf{y}) \odot \sigma(W_{g,k} \ast \mathbf{x} + V_{g,k} \ast \mathbf{y})$$

This allows the model to follow fine-grained temporal structure such as phoneme sequences.

> [!TIP]
> For a modern TTS system, local conditioning on mel-spectrograms is analogous to what Tacotron 2 uses — WaveNet as the vocoder conditioned on Tacotron-generated mel-spectrograms.

## 7. Algorithm: WaveNet Forward Pass

```
Input: previous samples x[1..t-1], conditioning h (optional)
Output: distribution p(x_t | x[1..t-1], h)

1. Quantize input to 256 levels via μ-law companding
2. Embed quantized samples: x_embed = Embedding(x)  # shape: [T, r]
3. Apply initial causal conv: h_0 = CausalConv1d(x_embed, W_init)
4. For each block b in 1..B:
   For each layer k in 0..K-1:
     d = 2^k  # dilation rate
     filter = tanh(DilatedCausalConv(h_prev, W_f,k, dilation=d))
     gate   = σ(DilatedCausalConv(h_prev, W_g,k, dilation=d))
     if conditioning:
       filter += V_f,k(h)  # add conditioning bias
       gate   += V_g,k(h)
     z = filter ⊙ gate
     skip_sum += Conv1x1(z, W_skip,k)     # accumulate skip
     h_prev = h_prev + Conv1x1(z, W_res,k)  # residual update
5. out = ReLU(skip_sum)
6. out = Conv1x1(out, W_out1)
7. out = ReLU(out)
8. out = Conv1x1(out, W_out2)
9. logits = Softmax(out)  # shape: [T, 256]
10. Return categorical distribution over 256 levels
```

**Training:** Cross-entropy loss between logits and quantized targets, summed over all $T$ positions. Optimized with Adam.

**Generation:** Sample $x_t \sim \text{Categorical}(\text{logits}[t])$, then feed $x_t$ back as input for $x_{t+1}$.

## 8. Comparison with Related Models

| Property | WaveNet | LSTM-RNN TTS | HMM Concatenative | SampleRNN |
|---|---|---|---|---|
| Audio representation | Raw waveform | Vocoder parameters | Waveform units | Raw waveform |
| Conditioning | Local + global | Local features | Text+prosody | Frame-level RNN |
| Parallelism (training) | Full (CNN) | Partial (BPTT) | None | Partial |
| Receptive field | Exponential (dilations) | Theoretically unlimited | Fixed window | Hierarchical |
| Output quality (MOS) | 4.21 (EN) | 3.67 (EN) | 3.86 (EN) | — |
| Inference speed | Slow (sequential) | Moderate | Fast | Slow |

> [!NOTE]
> WaveNet's main limitation vs. SampleRNN is that both are slow at inference. Subsequent work (WaveRNN, WaveGlow, HiFi-GAN) addressed this. WaveNet's main advantage is modeling long-range dependencies efficiently within a CNN framework.

# Experiments

- **Dataset (TTS - English):** Internal Google TTS dataset, 24.6 hours, single North American English female speaker
- **Dataset (TTS - Mandarin):** Internal Google TTS dataset, 34.8 hours, single Mandarin Chinese female speaker
- **Dataset (Multi-speaker):** CSTR VCTK Corpus — 44 hours total, 109 different English speakers, sampled at 16 kHz
- **Dataset (Music - Piano):** YouTube piano recordings, ~60 hours of solo piano
- **Dataset (Music - General):** MagnaTagATune, ~200 hours, tagged with 188 genre/mood labels
- **Dataset (Speech recognition):** TIMIT phoneme recognition benchmark (standard train/dev/test split)
- **Hardware:** Not explicitly stated (Google DeepMind infrastructure)
- **Optimizer:** Adam (learning rate and batch size not specified in the paper)
- **Audio sample rate:** 16 kHz (speech); variable (music)
- **Quantization:** 256 levels via μ-law companding

**Key Results:**
- WaveNet (conditioned on linguistic + F0 features): MOS **4.21 ± 0.081** (English) vs. natural speech 4.55 ± 0.072
- WaveNet: MOS **4.08 ± 0.085** (Mandarin) vs. natural speech 4.21 ± 0.071
- Gap to natural speech reduced by **51% (EN) and 69% (Mandarin)** compared to previous best
- Multi-speaker model: single model handles 109 speakers with speaker conditioning; increasing training speakers improves quality
- TIMIT phoneme error rate: **18.8%** — best result at the time for models trained on raw waveforms (compared to ~19.6% for spectrogram-based models)
