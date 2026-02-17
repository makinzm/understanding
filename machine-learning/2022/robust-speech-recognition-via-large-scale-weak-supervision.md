# Meta Information

- URL: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. ICML 2023.

# Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)

Whisper is a speech recognition system trained on 680,000 hours of multilingual and multitask audio data collected from the internet. Unlike prior work that relies heavily on unsupervised pre-training followed by supervised fine-tuning on curated datasets, Whisper uses large-scale **weak supervision** from naturally occurring audio-transcript pairs and achieves zero-shot generalization that approaches human-level accuracy without dataset-specific fine-tuning.

**Applicable when**: You need robust speech recognition that generalizes to diverse acoustic conditions, multiple languages, or out-of-distribution audio without fine-tuning. Also applicable for speech translation from 99 languages to English.

## Problem Setup

### Input / Output

| Stage | Input | Output |
|-------|-------|--------|
| Feature extraction | Raw audio waveform | 80-channel log-magnitude Mel spectrogram, $x \in \mathbb{R}^{T \times 80}$ (30-second chunks at 25ms hop) |
| Encoder | Mel spectrogram $x \in \mathbb{R}^{T \times 80}$ | Audio representations $h \in \mathbb{R}^{T' \times d}$ |
| Decoder | Audio representations $h$ + token prefix | Next token probability distribution over vocabulary $\mathbb{R}^{|V|}$ |
| Full model | 30-second audio chunk | Text tokens (transcription or translation) |

### Multitask Output Format

The decoder generates a structured token sequence that encodes task specification:

```
<|startoftranscript|> <|lang_id|> <|transcribe|> [<|notimestamps|>] text tokens <|endoftext|>
```

- `<|lang_id|>`: one of 99 language tokens (e.g., `<|en|>`, `<|ja|>`)
- `<|transcribe|>` or `<|translate|>`: task indicator
- `<|notimestamps|>`: suppresses timestamp prediction when not needed
- Timestamp tokens `<|0.00|>` through `<|30.00|>` can interleave with text tokens

## Model Architecture

Whisper uses a standard **encoder-decoder Transformer** architecture:

### Encoder
1. Two 1D convolutional layers (kernel size 3, stride 1 and 2) with GELU activation process the Mel spectrogram into $\mathbb{R}^{T'/2 \times d}$
2. Sinusoidal position embeddings are added
3. $N$ Transformer encoder blocks with multi-head self-attention and feed-forward layers
4. Output: $h \in \mathbb{R}^{T' \times d}$ audio representations

### Decoder
1. Learned position embeddings (unlike encoder's sinusoidal)
2. $N$ Transformer decoder blocks with masked self-attention and cross-attention over encoder output
3. Linear projection to vocabulary size, with weights **tied to the input token embedding matrix**

### Model Sizes

| Model | Parameters | Layers | $d_{model}$ | Heads |
|-------|-----------|--------|-------------|-------|
| Tiny | 39M | 4+4 | 384 | 6 |
| Base | 74M | 6+6 | 512 | 8 |
| Small | 244M | 12+12 | 768 | 12 |
| Medium | 769M | 24+24 | 1024 | 16 |
| Large | 1550M | 32+32 | 1280 | 20 |

## Dataset Construction

### Scale
- **Total**: 680,000 hours of audio with transcripts
- **English speech recognition**: 563,000 hours
- **X→English translation**: 117,000 hours across 96 languages
- **Multilingual transcription**: additional 125,000 hours across 75 languages

### Quality Filtering Pipeline

Raw internet audio-transcript pairs undergo several filtering stages to remove low-quality or machine-generated content:

```
Internet audio + transcripts
    → Language identification (VoxLingua107 model)
    → Detect and remove ASR-generated captions
       (heuristics: low-quality characters, abnormal repetition, etc.)
    → Fuzzy deduplication
    → Manual inspection of high-error sources
    → Final dataset: 680,000 hours
```

> [!NOTE]
> The authors explicitly avoid using automatic speech recognition systems to generate training transcripts, as this would create a feedback loop that degrades diversity and introduces systematic errors.

## Multitask Training

All tasks share a single model. The **input format** determines what the model learns:

- **Transcription** (English): `<|en|> <|transcribe|>` → English text
- **Transcription** (multilingual): `<|xx|> <|transcribe|>` → text in language xx
- **Translation** (X→English): `<|xx|> <|translate|>` → English text
- **Timestamp prediction**: interleaved `<|t|>` tokens indicate segment boundaries

This multitask setup allows one model checkpoint to serve all use cases, and the authors find **positive transfer** between tasks at sufficient scale (large models outperform English-only baselines on English transcription).

## Long-form Decoding Algorithm

For audio longer than 30 seconds, Whisper uses **buffered transcription**:

```
Algorithm: Long-form Transcription
Input: audio waveform of arbitrary length
Output: full transcript

1. Split audio into 30-second windows with overlap
2. For each window:
   a. Run beam search (beam size 5) with log-prob scoring
   b. Apply temperature fallback:
      - Start at temperature τ=0.0
      - If compression ratio < 2.4 or avg log-prob < -1.0:
        retry with τ ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
   c. Apply voice activity detection to detect silence
   d. Condition on previous segment's text tokens
   e. Use timestamp tokens to find precise segment boundaries
3. Concatenate segments, resolve overlaps
```

> [!IMPORTANT]
> The temperature fallback mechanism is critical: greedy decoding (τ=0) can enter repetition loops, while higher temperatures sacrifice accuracy for diversity. The fallback triggers only when quality metrics indicate failure.

## Comparison with Prior Methods

| Aspect | Whisper | Wav2Vec 2.0 / HuBERT | Conformer |
|--------|---------|----------------------|-----------|
| Pre-training | Supervised (680K hrs) | Self-supervised + fine-tune | Supervised (domain-specific) |
| Fine-tuning required | No (zero-shot) | Yes, on LibriSpeech | Yes |
| LibriSpeech WER | 2.7% (test-clean) | ~1.8% (fine-tuned) | ~1.9% (fine-tuned) |
| Out-of-distribution | Robust (~55% error reduction) | Degrades (~2× more errors) | Degrades |
| Languages | 99 | Primarily English | Primarily English |
| Translation | Yes (X→en) | No | No |

> [!NOTE]
> "Models trained on LibriSpeech are roughly twice as many errors on other datasets compared to Whisper, showing that the in-distribution performance advantage of fine-tuned models does not translate to real-world robustness."

The key distinction: supervised models over-specialize to their training distribution, while Whisper's diverse weak supervision enforces generalization across acoustic conditions.

## Text Normalization

To enable fair evaluation, the authors implemented comprehensive **text normalization** rules:

- Expand contractions ("you're" → "you are")
- Standardize numeric expressions ("$1.5B" → "1.5 billion dollars")
- Remove symbols and diacritics where appropriate
- Convert British to American spelling
- Normalize punctuation and capitalization

These rules are applied to both hypothesis and reference before computing WER, preventing unfair penalties for formatting differences.

## Scaling Behavior

### Model Scale
Zero-shot performance improves reliably with model size across all tasks. English ASR shows diminishing returns at large scale (approaching human performance ceiling), while multilingual recognition and translation continue improving.

### Data Scale
Performance follows a **power-law** relationship with training data volume per language:

$$\text{WER} \propto D^{-\alpha}$$

where $D$ is training hours and $\alpha \approx 0.5$ (WER halves for every 16× increase in data). This suggests low-resource language performance could improve substantially with targeted data collection.

### Multitask Scale
Negative transfer occurs on smaller models (joint multilingual training hurts English ASR), but large models show positive transfer—joint training outperforms English-only baselines.

# Experiments

- **Datasets**:
  - LibriSpeech (960h train; test-clean, test-other)
  - Multilingual LibriSpeech (MLS) — 8 European languages
  - Common Voice (multiple languages)
  - VoxPopuli (European Parliament speech)
  - FLEURS (multilingual)
  - CoVoST2 (speech translation, 21 X→en pairs)
  - TED-LIUM (TED talk transcription)
  - Kincaid46 (25 recordings for human comparison)
  - 7 long-form datasets for buffered decoding evaluation
- **Hardware**: Not specified in detail; training uses large-scale GPU clusters
- **Optimizer**: AdamW with gradient clipping; linear warmup + cosine decay schedule
- **Training**: 1M update steps, batch size of 256 segments; 30-second audio chunks
- **Results**:
  - LibriSpeech test-clean WER: **2.7%** (zero-shot, large model)
  - Multilingual LibriSpeech WER: **7.3** average (zero-shot)
  - CoVoST2 X→en BLEU: **29.1** (zero-shot, new state-of-the-art at time of publication)
  - Human transcriber WER on Kincaid46: **5.9%** vs. Whisper: **7.1%**
  - Zero-shot Whisper reduces out-of-distribution error by **55.2%** on average vs. fine-tuned LibriSpeech models
