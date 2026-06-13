# Meta Information

- URL: [StyleStream: Real-Time Zero-Shot Voice Style Conversion](https://arxiv.org/abs/2602.20113)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Liu, Y., Lee, N., & Anumanchipalli, G. (2026). StyleStream: Real-Time Zero-Shot Voice Style Conversion. arXiv preprint arXiv:2602.20113.

# StyleStream: Real-Time Zero-Shot Voice Style Conversion

StyleStream is the first streamable zero-shot voice style conversion system achieving approximately 1-second end-to-end latency. It converts an input utterance to match a target speaker's **timbre, accent, and emotion** simultaneously while preserving the source's original linguistic content—all without any fine-tuning on the target speaker.

Voice style conversion differs from text-to-speech voice cloning in that it operates entirely on acoustic signals rather than using text as a naturally disentangled content representation. The core challenge is separating linguistic content from stylistic attributes at the signal level.

## Problem Definition

**Input:**
- Source speech utterance (arbitrary speaker, $x_{\text{src}} \in \mathbb{R}^{T}$, 16 kHz waveform)
- Target reference speech utterance (desired style, $x_{\text{ref}} \in \mathbb{R}^{T'}$, 16 kHz waveform)

**Output:**
- Converted waveform $\hat{x} \in \mathbb{R}^{\hat{T}}$ that preserves the source's linguistic content while expressing the target speaker's timbre, accent, and emotion.

**Zero-shot** means the target speaker is unseen during training. **Streamable** means the system processes audio in chunks and can emit output with bounded latency rather than requiring the full source utterance.

## System Architecture

StyleStream consists of two main modules—the **Destylizer** and the **Stylizer**—plus a causal **Vocoder**.

```
Source Speech
    │
    ▼
┌─────────────┐
│ Destylizer  │  → Content features f_c (style removed)
└─────────────┘
                        ┌───────────────┐
Target Reference ──────►│ Style Encoder │  → Style embedding s
                        └───────────────┘
    f_c + s
    │
    ▼
┌──────────────┐
│  Stylizer    │  → Mel-spectrogram (stylized)
└──────────────┘
    │
    ▼
┌──────────────┐
│   Vocoder    │  → Waveform
└──────────────┘
```

## Destylizer

The Destylizer extracts linguistic content while removing all style information from the source utterance.

**Architecture:**

1. **HuBERT-Large encoder** (frozen, 18th layer features): Produces frame-level representations at 50 Hz.
2. **Six Conformer blocks**: Hidden size 768, FFN size 3072. Refine the representations with both local (convolution) and global (attention) context.
3. **Finite Scalar Quantization (FSQ)**: Levels $[5, 3, 3]$ → $5 \times 3 \times 3 = 45$ total codes. Acts as an information bottleneck.
4. **Four Transformer decoder layers**: Hidden size 768, FFN size 3072, with ALiBi positional encoding (enables length extrapolation). Predict character-level ASR tokens.

**Content features $f_c$** are the continuous representations **immediately before** the FSQ layer—not the discrete codes. This is critical: using discrete FSQ indices causes WER to jump from ~9% to 123.5% because quantization discards fine-grained phonetic detail.

**Training objective:** End-to-end sequence-to-sequence ASR loss (character token prediction). This forces the bottleneck to retain only linguistically useful information and discard style.

**Why ASR supervision works for disentanglement:**

- **Text supervision** directs only linguistic content through the bottleneck; style information that does not help ASR is suppressed.
- **Compact codebook (45 codes)** creates a narrow information bottleneck compared to CosyVoice 2's 6,561-code vocabulary, forcing stronger content-style separation.
- **Continuous pre-quantization features** preserve more phonetic nuance than discrete tokens while still benefiting from the style-suppressing training objective.

**Training config:** 100k steps, 8 × RTX A6000, batch size 32, peak LR $10^{-4}$, cosine annealing with 4k warmup steps. Data: 1,300 hours (LibriTTS + MSP-Podcast + GLOBE).

## Style Encoder

The style encoder extracts a speaker/style embedding from the reference utterance.

**Architecture (WavLM-TDNN):**

1. **Frozen WavLM**: Provides multi-layer representations.
2. **Learnable layer weighting**: Scalar coefficients aggregate information across all WavLM layers.
3. **Time Delay Neural Network (TDNN)**: Processes the weighted representations.
4. **Attentive Statistics Pooling (ATSP)**: Collapses the temporal dimension into a fixed-size embedding $s$.

The style embedding $s$ is injected into the Stylizer's diffusion transformer via **adaLN-Zero** at each transformer layer.

## Stylizer

The Stylizer generates a mel-spectrogram conditioned on the content features $f_c$ and style embedding $s$, using a **diffusion transformer** with a spectrogram inpainting objective.

**Mel-spectrogram format:** 100-bin mel-spectrograms, 50 Hz frame rate (320-sample hop, 16 kHz audio).

**Architecture:**

- 16 Transformer layers, hidden size 768, FFN size 3072.
- Time embedding via sinusoidal positional encoding (for diffusion timestep $t$).
- Style embedding injected via adaLN-Zero at each layer.

**Spectrogram Inpainting:**

The model is trained to reconstruct the source mel-spectrogram in masked regions, conditioned on:
- The unmasked **context** $(1 - m) \odot x_1$ (the source spectrogram where $m$ is 0).
- **Content features** $f_c$ from the Destylizer.
- **Style embedding** $s$ from the reference.

Input to the transformer at each denoising step:
```math
\begin{align}
  \text{input} = \text{concat}(x_t,\ (1-m)\odot x_1,\ f_c)\ \text{along channel dim}
\end{align}
```
where $x_t$ is the noisy spectrogram at diffusion time $t$.

**Training objective (Conditional Flow Matching with Optimal Transport):**

```math
\begin{align}
  \mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}\left[\left\|m \odot \left(\hat{v} - \frac{d}{dt}\psi_t(x_0)\right)\right\|^2\right]
\end{align}
```

Velocity prediction loss is computed only on the **masked regions** ($m \odot$), so the model learns to stylize while preserving unmasked (context) regions.

**Dropout during training:** Content, context, and style are dropped with probabilities 0.2, 0.3, 0.3 respectively to enable classifier-free guidance.

**Inference:**
- Classifier-Free Guidance (CFG) with fixed strength $\alpha = 2$.
- Number of Function Evaluations (NFE): 16 steps using Euler sampling.
- The mask $m$ covers 100% of the source region (full conversion).

**Training config:** 400k steps, 8 × RTX A6000, batch size 64, 6-second segments, random masking 70–100%.

## Vocoder

A causal Vocos-based vocoder converts mel-spectrograms to waveforms. It is trained with GAN loss, reconstruction loss, and feature matching loss. Causal convolutions ensure it is compatible with streaming.

## Streaming Mechanism

Both the Destylizer and Stylizer are adapted from their offline (non-causal) versions.

**Chunked-Causal Attention:** Each chunk of frames attends to its own features and all preceding chunks but not future chunks. This provides causal behavior while retaining limited long-range context.

**Streaming-specific modifications:**
- HuBERT layers are unfrozen and converted to chunked-causal attention.
- All convolution layers are converted to causal convolutions.
- The streaming Destylizer is fine-tuned with an MSE distillation loss against the offline (non-streaming) teacher's pre-FSQ features.

**Runtime configuration:**
- Target utterance length: 5 seconds (fixed-length input to the Stylizer).
- Ring buffer of incoming content feature chunks feeds the Stylizer.
- Chunk size: 600 ms (optimal balance between quality and latency).

**Latency:**
```math
\begin{align}
  L = t_{\text{chunk}} + t_{\text{proc}} = 600\,\text{ms} + 412.7\,\text{ms} \approx 1{,}013\,\text{ms}
\end{align}
```

Processing time (412.7 ms) remains below chunk duration (600 ms), ensuring real-time feasibility on an RTX A6000.

## Comparison with Related Methods

| Method | Architecture | Codebook | WER ↓ | A-SIM ↑ | E-SIM ↑ | Streaming |
|---|---|---|---|---|---|---|
| FACodec (NaturalSpeech3) | Gradient reversal + codec | large | 15.5% | — | — | No |
| CosyVoice 2.0 | ASR-supervised tokenizer | 6,561 codes | 9.5% | Low | Low | No |
| SeedVC v2 | Style conversion variant | — | 21.7% | — | — | No |
| Vevo (prior SOTA) | Autoregressive, 32-code | 32 codes | 17.5% | 0.596 | 0.712 | No |
| **StyleStream (Offline)** | DiT + FSQ 45-code | **45 codes** | **9.2%** | **0.640** | **0.827** | No |
| **StyleStream (Streaming)** | DiT + FSQ 45-code (causal) | **45 codes** | 15.3% | 0.635 | 0.803 | **Yes** |

> [!NOTE]
> CosyVoice 2.0 achieves lower WER (9.5%) but its large codebook retains accent and emotion information in the tokens, making it unable to convert those style attributes effectively.

> [!IMPORTANT]
> StyleStream is the only system in the comparison that supports real-time streaming. Its streaming variant degrades WER from 9.2% to 15.3% but still outperforms all non-streaming baselines on speaker and emotion similarity.

## Experiments

- **Datasets:**
  - Destylizer training: 1,300 hours — LibriTTS (read English), MSP-Podcast (conversational, emotion-varied), GLOBE (accented English)
  - Stylizer training: 50,000 hours — Emilia dataset (English subset)
  - Vocoder training: LibriTTS
  - Evaluation (StyleStream-Test): 300 source utterances × 10 target utterances = **3,000 source-target pairs**, covering 5 emotions and 5 accents
- **Hardware:** 8 × RTX A6000 GPUs for training; RTX A6000 for inference timing
- **Evaluation metrics:**
  - **WER** (Whisper-large-v3): Intelligibility
  - **S-SIM**: Speaker (timbre) similarity via Resemblyzer cosine similarity
  - **A-SIM**: Accent similarity via accent-id-commonaccentecapa embeddings
  - **E-SIM**: Emotion similarity via emotion2vec embeddings
  - **UTMOS**: Predicted mean opinion score for naturalness
  - **Subjective MOS** (crowd-sourced via Prolific, native US/UK English speakers, 400 ratings per condition): NMOS (naturalness), S-SMOS, A-SMOS, E-SMOS (similarity)
- **Disentanglement probe:** ECAPA-TDNN classifiers trained on Destylizer's content features; StyleStream achieves ~3% speaker classification accuracy vs. 20%+ for competing methods.

## Ablation Study

| Configuration | WER ↓ | A-SIM ↑ | Observation |
|---|---|---|---|
| FSQ [3,3,3] (27 codes) | 19.0% | — | Too restrictive; loses phonetic detail |
| FSQ [5,3,3] (45 codes) ← **optimal** | 9.2% | 0.640 | Best balance |
| FSQ [7,5,5,5,5] (4,375 codes) | ~9% | 0.439 | Large codebook retains style |
| Discrete FSQ indices (not continuous) | 123.5% | — | Catastrophic intelligibility loss |
| Without style encoder | — | 0.509 | Style not transferred |
| Destylizer trained on 50k hrs | — | — | Diminishing returns vs. 1.3k hrs |
| Chunk size 200 ms | Higher | Lower | Faster but worse quality |
| Chunk size 1,000 ms | Lower | Higher | Slower but better quality |
| Target reference 2 s (vs. 5 s) | — | Degraded | Shorter reference degrades all style metrics |

> [!IMPORTANT]
> Using continuous pre-quantization features instead of discrete FSQ codes is non-negotiable—the discrete codes lose too much phonetic information, leading to catastrophic intelligibility failure (WER 123.5%).

## Applicability

StyleStream is suited for:
- **Real-time communication**: Video calls or live broadcasts requiring instant voice style adaptation.
- **Accessibility**: Converting speech to a style more familiar or comfortable to a listener (e.g., accent normalization).
- **Entertainment/gaming**: Character voice adaptation without re-recording.
- **Research on speech disentanglement**: The disentanglement probing methodology is independently useful for evaluating content-style separation.

It requires at least a reference utterance of 5 seconds for best style transfer quality. Inference needs a GPU capable of processing a 600 ms chunk in under 600 ms (an RTX A6000 completes it in ~413 ms).
