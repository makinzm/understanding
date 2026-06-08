# Meta Information

- URL: [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html); model weights released under Apache 2.0 License.
- Reference: Qwen Team — Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, et al. (2026). Qwen3-TTS Technical Report. arXiv:2601.15621.

---

# Qwen3-TTS Technical Report

## Overview

Qwen3-TTS is a family of multilingual, controllable, and streaming text-to-speech (TTS) models trained on over 5 million hours of speech data in 10 languages. The system is designed for practitioners who need production-grade TTS with low-latency streaming, 3-second voice cloning, and natural-language-driven voice control.

**Key properties:**

| Property | Description |
|---|---|
| Controllability | Users describe voice characteristics in natural language (e.g., "speak with a calm, deep voice") to create or modify output voices |
| Voice Cloning | Clones any speaker's voice from a 3-second reference audio without additional fine-tuning |
| Naturalness | Human-like prosody and expressiveness evaluated by UTMOS (4.16 on LibriSpeech) |
| Multilinguality | 10 languages: Chinese, English, German, Italian, Portuguese, Spanish, Japanese, Korean, French, Russian |
| Streaming | First-packet latency as low as 97 ms for the 0.6B variant |

---

## 1. System Architecture

Qwen3-TTS uses a **dual-track token representation** that concatenates text tokens (from the standard Qwen tokenizer) with speech tokens produced by one of two custom speech tokenizers. A learnable speaker encoder is jointly trained with the backbone to condition generation on a reference voice.

```
Input text  →  [Qwen tokenizer]  →  text tokens  ─┐
Reference audio → [Speaker encoder] → speaker embed ─┤→  LLM backbone → speech tokens → Code2Wav → waveform
```

Two model variants correspond to two different tokenizers:

- **Qwen3-TTS-25Hz** (1.7B and 0.6B): Uses `Qwen-TTS-Tokenizer-25Hz` (single codebook, 25 Hz frame rate, latency ~150 ms)
- **Qwen3-TTS-12Hz** (1.7B and 0.6B): Uses `Qwen-TTS-Tokenizer-12Hz` (16-layer RVQ, 12.5 Hz frame rate, latency ~97–101 ms)

---

## 2. Qwen-TTS-Tokenizer-25Hz

### Design

- **Architecture**: Single-codebook VQ tokenizer derived from Qwen2-Audio.
- **Frame rate**: 25 Hz (one token per 40 ms of audio).
- **Codebook size**: 32,768 entries.
- **Decoder**: Diffusion Transformer (DiT) with Flow Matching reconstructs mel-spectrograms; BigVGAN converts mel-spectrograms to waveforms.

### Two-Stage Training

1. **Stage 1 – ASR pretraining**: Insert a Vector Quantization (VQ) layer into the Qwen2-Audio encoder and fine-tune for ASR to ensure speech tokens carry strong semantic content.
2. **Stage 2 – Reconstruction fine-tuning**: Train the DiT decoder to reconstruct mel-spectrograms from VQ tokens, using a multi-scale mel-spectrogram reconstruction loss.

### Streaming Mechanism

A **block-wise attention** scheme restricts the DiT's attention window so that each block can see the current block, three preceding blocks, and one lookahead block. This lets the decoder produce audio incrementally: after receiving 16 speech tokens (640 ms of context), the DiT emits one 320 ms audio packet, giving a first-packet latency of 150 ms for the 1.7B model.

> [!NOTE]
> "The block-wise DiT with Flow Matching enables streaming synthesis by reconstructing mel content in 320 ms chunks, balancing quality with low latency."

---

## 3. Qwen-TTS-Tokenizer-12Hz

### Design

- **Architecture**: 16-layer Residual Vector Quantization (RVQ) applied at 12.5 Hz.
- **Codebook size**: 2,048 per layer (16 layers total).
- **Encoder/Decoder**: Fully causal ConvNet — no look-ahead — enabling real-time streaming.

### Semantic-Acoustic Disentanglement

The training objective explicitly separates semantic and acoustic information across codebook layers:

- **Layer 0 (semantic codebook)**: Trained with WavLM teacher guidance so the zeroth codebook tokens capture linguistic content (phonemes, word identity).
- **Layers 1–15 (acoustic refinement)**: Residual quantizers capture speaker timbre, prosody, and fine-grained acoustic detail beyond the semantic content of layer 0.

```
Raw waveform → Causal Encoder → Semantic VQ (layer 0) + RVQ (layers 1–15) → Acoustic tokens
Acoustic tokens → Causal ConvNet Decoder → Waveform (no diffusion needed)
```

Training uses a GAN-based framework (generator + discriminator) with a multi-scale mel-spectrogram reconstruction loss.

> [!IMPORTANT]
> The causal ConvNet decoder eliminates the need for a diffusion model, enabling lower first-packet latency (101 ms vs. 150 ms for 25Hz) while maintaining high reconstruction quality.

### Comparison: 25Hz vs. 12Hz Tokenizer

| Property | Tokenizer-25Hz | Tokenizer-12Hz |
|---|---|---|
| Frame rate | 25 Hz | 12.5 Hz |
| Codebooks | 1 | 16 (RVQ) |
| Codebook size | 32,768 | 2,048 per layer |
| Decoder type | Diffusion Transformer + BigVGAN | Causal ConvNet |
| First-packet latency (1.7B) | 150 ms | 101 ms |
| UTMOS (LibriSpeech) | — | 4.16 |
| Speaker Similarity | — | 0.95 |
| Streaming look-ahead | Block-wise (1 block) | None (fully causal) |

---

## 4. Training Pipeline

### Pre-Training (3 Stages)

| Stage | Name | Data | Purpose |
|---|---|---|---|
| S1 | General | 5M+ hours, 10 languages | Learn broad speech patterns |
| S2 | High-Quality | Curated subset with quality stratification | Improve naturalness and clarity |
| S3 | Long-Context | Same data, extended seq length from 8,192 → 32,768 tokens | Enable consistent long-form synthesis (10+ minutes) |

### Post-Training (3 Stages)

**Stage 1 – Direct Preference Optimization (DPO):**
Human annotators label speech sample pairs (preferred vs. dispreferred) for multilingual output quality. The model is fine-tuned with DPO to shift the output distribution toward preferred samples.

**Stage 2 – GSPO (Generalized Score-based Preference Optimization):**
Rule-based reward signals supplement human feedback and are used in a Group Sampling Policy Optimization (GSPO) framework to comprehensively enhance intelligibility, naturalness, and instruction-following.

**Stage 3 – Speaker Fine-Tuning:**
A lightweight fine-tuning pass adapts the model to specific speaker identities (predefined voice profiles), improving naturalness and expressiveness for cloning use cases without retraining the full model.

---

## 5. Multi-Token Prediction (MTP) for 12Hz Models

The 12Hz tokenizer produces 16 tokens per audio frame (one per RVQ layer). Predicting all 16 autoregressively at each step would be prohibitively slow. Instead, Qwen3-TTS-12Hz uses a **Multi-Token Prediction (MTP) module**:

```
LLM backbone → aggregated codebook features
  → Linear head predicts layer-0 token (semantic)
  → MTP module predicts layers 1–15 tokens in parallel (acoustic)
```

This reduces the number of sequential LLM forward passes while maintaining per-layer token diversity.

---

## 6. Streaming Inference

One "speech packet" is defined as 4 speech tokens, representing 320 ms of audio.

| Model | First-Packet Latency (1 req) | RTF (6 concurrent) |
|---|---|---|
| Qwen3-TTS-12Hz-0.6B | 97 ms | 0.434 |
| Qwen3-TTS-12Hz-1.7B | 101 ms | 0.463 |
| Qwen3-TTS-25Hz-0.6B | — | — |
| Qwen3-TTS-25Hz-1.7B | 150 ms | 0.725 |

RTF (Real-Time Factor) < 1.0 means the model generates audio faster than real time. All variants satisfy this at 6 concurrent requests.

> [!NOTE]
> "For the 12Hz models, the fully causal ConvNet decoder eliminates any look-ahead requirement, enabling true streaming with minimal buffering."

---

## 7. Controllable Generation

Qwen3-TTS supports natural-language voice descriptions as input, allowing users to specify gender, age, speaking style, emotion, and accent without providing a reference audio clip. This is evaluated on the **InstructTTSEval** benchmark, which measures:

- **APS (Attribute Prediction Score)**: How accurately the generated speech matches the described attributes.
- **DSD (Description-Speech Consistency)**: Consistency between the text description and the acoustic output.
- **RP (Reference Proximity)**: Proximity to a reference voice given only a description.

---

# Experiments

### Datasets

| Dataset | Language | Purpose |
|---|---|---|
| Internal 5M-hour corpus | 10 languages | Pre-training speech data |
| Seed-TTS eval set | Chinese, English | Zero-shot voice cloning benchmark |
| CommonVoice | English, multilingual | Tokenizer ASR evaluation (WER) |
| LibriSpeech | English | Tokenizer reconstruction quality (PESQ, STOI, UTMOS, Speaker Similarity) |
| InstructTTSEval | Chinese, English | Controllable TTS evaluation (APS, DSD, RP) |
| Long speech test set | Chinese, English | Long-form synthesis (texts up to 2000 words) |

### Tokenizer Quality (Qwen-TTS-Tokenizer-12Hz on LibriSpeech)

| Metric | Score |
|---|---|
| PESQ_WB | 3.21 |
| PESQ_NB | 3.68 |
| STOI | 0.96 |
| UTMOS | 4.16 |
| Speaker Similarity | 0.95 |

### Zero-Shot Voice Cloning (Seed-TTS benchmark, WER ↓)

| Model | test-zh | test-en |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B | **0.77** | 1.24 |
| CosyVoice 3 | 0.71 | 1.45 |
| Seed-TTS | 1.12 | 2.25 |

Qwen3-TTS-12Hz-1.7B achieves best or near-best intelligibility across Chinese and English.

### Multilingual Speaker Similarity

Qwen3-TTS achieves the highest speaker similarity scores across all 10 languages (range: 0.775–0.829), demonstrating that the speaker encoder generalizes well across languages.

### Cross-Lingual Generation (Chinese speaker → Korean output)

| Model | WER ↓ |
|---|---|
| Qwen3-TTS-12Hz-1.7B | **4.82** |
| CosyVoice 3 | 14.4 |

A 66% reduction in WER demonstrates strong cross-lingual consistency.

### Long-Form Generation (texts up to 2000 words)

| Model | WER-ZH ↓ | WER-EN ↓ |
|---|---|---|
| Qwen3-TTS-25Hz-1.7B | **1.517** | **1.225** |
| Qwen3-TTS-12Hz-1.7B | 2.356 | 2.812 |

The 25Hz model benefits from the long-context pre-training stage (S3) and produces more stable prosody across extended utterances.

### Hardware

Not explicitly stated in the paper.

### Optimizer

Not explicitly stated beyond mentioning DPO and GSPO post-training; pre-training optimizer details are not disclosed.

---

## Comparison with Related TTS Systems

| System | Streaming | Voice Cloning | Controllable | Languages | Notes |
|---|---|---|---|---|---|
| Qwen3-TTS-12Hz | Yes (97–101 ms) | 3-sec ref | Yes (NL description) | 10 | Fully causal, RVQ decoder |
| Qwen3-TTS-25Hz | Yes (150 ms) | 3-sec ref | Yes | 10 | DiT + BigVGAN decoder |
| CosyVoice 3 | Partial | Yes | Limited | Multi | Prior SOTA baseline |
| Seed-TTS | No | Yes | No | 2 | Non-streaming baseline |
| VALL-E style | No | Yes | No | 1–2 | Autoregressive, no streaming |

> [!TIP]
> For practitioners deploying TTS in real-time applications (voice assistants, live translation), the 12Hz model's fully causal decoder and sub-100 ms first-packet latency make it preferable. For offline high-quality synthesis or long audiobooks, the 25Hz model's lower WER in long-form generation is advantageous.

> [!CAUTION]
> The paper does not disclose the exact composition of the 5M-hour training corpus or licensing details of the training data beyond stating it spans 10 languages. Use in production should account for potential data biases.
