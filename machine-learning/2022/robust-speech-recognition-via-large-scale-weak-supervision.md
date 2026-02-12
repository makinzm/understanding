# Meta Information

- URL: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. OpenAI.

# Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)

**Whisper** is an encoder-decoder Transformer trained on 680,000 hours of internet audio paired with transcripts, covering 96 languages, multitask transcription, and speech-to-English translation — all in a single zero-shot model.

> [!NOTE]
> "We study the capabilities of speech processing systems trained simply to predict large amounts of existing labels on the internet."

The core thesis is that prior work over-emphasized self-supervised pre-training (wav2vec 2.0, HuBERT) with supervised fine-tuning, which yields brittle models that generalize poorly to new domains. By scaling *weakly supervised* training data by over an order of magnitude and training without fine-tuning on specific benchmarks, Whisper achieves zero-shot robustness that substantially outperforms fine-tuned baselines on out-of-distribution data.

## Applicability

Whisper targets practitioners who need robust, multi-domain, multilingual ASR without per-dataset fine-tuning: speech application developers, researchers building downstream NLP pipelines on transcribed audio, and language technology teams working on low-resource or code-switched speech.

## Input / Output

| Component | Input | Output |
|-----------|-------|--------|
| Audio preprocessing | Raw audio, resampled to 16,000 Hz | 80-channel log-Mel spectrogram, $x \in \mathbb{R}^{T' \times 80}$ where $T' = \lceil T / 0.01 \rceil$ (10 ms stride), globally scaled to $[-1, 1]$ |
| Encoder | Mel spectrogram (padded to 30 s → 3,000 frames) | Encoded representation $h \in \mathbb{R}^{1500 \times d}$ (halved by stride-2 convolution) |
| Decoder (input) | Sequence of special tokens + BPE text tokens | N/A |
| Decoder (output) | Cross-attends encoder $h$ and prior tokens | Next-token logits over vocabulary |
| Full system | 30 s audio segment | Transcript text (and optionally: language ID, timestamps, translated text) |

## Architecture

### Audio Encoder

1. **Conv stem**: Two 1D convolutional layers with filter width 3 and GELU activation. The second convolution uses stride 2 to downsample 3,000 frames → 1,500 frames.
2. **Positional encoding**: Sinusoidal position embeddings added to the 1,500-frame sequence.
3. **Transformer blocks**: Pre-activation residual blocks (LayerNorm before attention/MLP).
4. **Final layer norm**: Applied after all blocks.

### Text Decoder

1. **Positional encoding**: Learned position embeddings (unlike encoder's fixed sinusoidal).
2. **Transformer blocks**: Same depth and width as encoder; uses masked self-attention + cross-attention to encoder output.
3. **Output projection**: Tied to input token embeddings (Press & Wolf, 2017) — the same weight matrix is reused for embedding lookup and final logit computation.

### Model Family

| Model | Encoder/Decoder Layers | Width | Attention Heads | Parameters |
|-------|------------------------|-------|-----------------|------------|
| Tiny  | 4                      | 384   | 6               | 39M        |
| Base  | 6                      | 512   | 8               | 74M        |
| Small | 12                     | 768   | 12              | 244M       |
| Medium| 24                     | 1,024 | 16              | 769M       |
| Large | 32                     | 1,280 | 20              | 1,550M     |

English-only variants exist for Tiny through Medium; Large is multilingual only.

### Tokenization

- **English-only models**: Byte-level BPE inherited from GPT-2 (50,257 token vocabulary).
- **Multilingual models**: BPE vocabulary refitted to multilingual text corpora at the same vocabulary size, reducing over-representation of English tokens.

## Multitask Token Format

All tasks are expressed as a sequence of special tokens prepended to the decoder, enabling a single model to handle transcription, translation, language identification, voice activity detection, and timestamp prediction simultaneously.

**Decoder input sequence:**

```
<|startoftranscript|>
<|LANG|>            ← one of 99 language tokens (e.g., <|en|>, <|ja|>)
<|transcribe|>      ← or <|translate|> for X→English translation
<|notimestamps|>    ← or omit to enable timestamp prediction
... BPE text tokens, interleaved with timestamp tokens if enabled ...
<|endoftranscript|>
```

Special tokens also include:
- `<|nospeech|>`: predicted for silent or non-speech segments.
- Timestamp tokens: quantized at 20 ms resolution, spanning 0.00 s to 30.00 s (1,501 distinct tokens). Format: `<|0.00|> text <|1.40|>` for a segment from 0 to 1.4 s.

> [!IMPORTANT]
> Training loss is computed only over the output tokens (transcript, timestamps), not over the conditioning prefix. This forces the decoder to use the task/language tokens as true conditioning signals rather than learning to predict them.

## Data Pipeline

**Scale**: 680,000 hours total internet audio-transcript pairs.
- 438,000 hours: English transcription
- 117,000 hours: Non-English transcription (96 languages)
- 125,000 hours: Speech-to-English translation (from 23 languages)

**Filtering steps applied before training:**
1. Machine-generated transcript detection: heuristics reject all-uppercase, all-lowercase, repeated-text patterns.
2. Language verification: a VoxLingua107-fine-tuned classifier checks that spoken language matches transcript language; mismatches are excluded or re-routed to translation tasks.
3. Audio segmentation: 30-second chunks with aligned transcript subsets; silent-only segments subsampled to balance voice activity detection training.
4. Quality filtering: error rate analysis across providers identifies low-quality data sources; partially transcribed or misaligned transcripts are manually flagged.
5. Deduplication: training data deduplicated against all evaluation sets (especially TED-LIUM 3).

## Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (Loshchilov & Hutter, 2017) |
| Learning rate schedule | Linear decay to zero after 2,048 warmup steps |
| Batch size | 256 segments (each 30 s) |
| Total updates | $2^{20}$ (~2–3 passes over full dataset) |
| Precision | FP16 with dynamic loss scaling |
| Gradient clipping | Gradient norm clipping |
| Memory efficiency | Activation checkpointing (Chen et al., 2016) |
| Data augmentation | None in base training (dataset diversity is the only regularizer) |

**Large V2** variant additionally uses SpecAugment, Stochastic Depth, and BPE Dropout, and is trained 2.5× longer.

## Long-Form Transcription Algorithm

Because Whisper processes 30-second windows, long audio requires a sliding-window strategy with five heuristics that together lower average WER:

```
INPUT: audio of length L, model M
window_start ← 0
previous_text ← ""
WHILE window_start < L:
    segment ← audio[window_start : window_start + 30s]
    FOR temperature IN [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        output ← beam_search_decode(M, segment, previous_text, temperature)
        IF NOT repetitive(output) AND avg_log_prob(output) > threshold:
            BREAK
    IF nospeech_prob(output) > nospeech_threshold:
        window_start ← window_start + 30s  # VAD skip
        CONTINUE
    timestamps ← extract_timestamps(output)
    EMIT text_tokens(output)  # without timestamp tokens
    previous_text ← last_n_tokens(output)  # for context conditioning
    window_start ← last_timestamp(timestamps)
```

The five strategies and their incremental gains (avg WER across 7 long-form datasets):

| Strategy | Avg WER |
|----------|---------|
| Greedy only | 11.0 |
| + Beam search | 10.6 |
| + Temperature fallback | 10.6 |
| + Voice activity detection | 10.2 |
| + Previous text conditioning | 10.0 |
| + Initial timestamp constraint | 10.0 |

## Comparison with Related Methods

| Aspect | wav2vec 2.0 / HuBERT | Whisper |
|--------|----------------------|---------|
| Pre-training paradigm | Self-supervised (masked prediction) | Weakly supervised (internet transcripts) |
| Fine-tuning required | Yes, per dataset | No (zero-shot) |
| LibriSpeech clean WER | ~2.0–2.7% (fine-tuned) | 2.7% (zero-shot) |
| OOD robustness | Degrades rapidly | 55% relative error reduction over fine-tuned wav2vec 2.0 on 12 OOD datasets |
| Multilingual support | Via XLS-R (self-supervised, no task conditioning) | 96 languages, multitask, single model |
| Translation | Not directly supported | 125k hrs X→En training, SOTA on CoVoST 2 zero-shot |
| Model size | Up to 1B params | 39M–1,550M |
| Training data | 960h (LibriSpeech) + unlabeled | 680,000h weakly labeled |

> [!TIP]
> SpeechStew (Chan et al., 2021) also mixes supervised datasets, but uses only ~5,140 hours. BigSSL (Zhang et al., 2021) scales supervised+semi-supervised to millions of hours on proprietary data. Whisper is the first large-scale *publicly released* model in this paradigm.

**Key difference from CLIP analogy**: Whisper performs the same weakly-supervised scaling that CLIP applied to vision-language, but for audio-text, and demonstrates that task diversity (transcription + translation + timestamps) in data collection is sufficient to train a single specialist model.

## Datasets

### Training

| Split | Hours | Description |
|-------|-------|-------------|
| English transcription | 438,000 | Internet audio-transcript pairs, English |
| Non-English transcription | 117,000 | 96 languages |
| Speech-to-English translation | 125,000 | 23 source languages |
| **Total** | **680,000** | |

### Evaluation (English ASR)

| Dataset | Domain | Notes |
|---------|--------|-------|
| LibriSpeech (test-clean/other) | Audiobooks | Panayotov et al., 2015 |
| TED-LIUM 3 | TED talks | Hernandez et al., 2018 |
| Common Voice | Crowdsourced read speech | Ardila et al., 2019 |
| Artie Bias Corpus | Misc | Meyer et al., 2020 |
| VoxPopuli (en) | EU Parliament | Wang et al., 2021 |
| CORAAL | African American Vernacular English | Kendall & Farrington, 2021 |
| CHiME-6 | Far-field dinner table | Watanabe et al., 2020 |
| AMI IHM / SDM1 | Meeting speech | |
| Switchboard / CallHome | Telephone conversations | |
| WSJ | Wall Street Journal read speech | |
| GigaSpeech | Audiobook + podcast + YouTube | Chen et al., 2021 |
| Earnings-21 / Earnings-22 | Earnings calls | Del Rio et al., 2021 |
| Rev16 / Kincaid46 | Podcast / benchmark for human comparison | |

### Evaluation (Multilingual)

| Dataset | Task | Notes |
|---------|------|-------|
| Multilingual LibriSpeech (MLS) | ASR, 8 languages | Pratap et al., 2020b |
| VoxPopuli (multilingual) | ASR, 15 languages | Wang et al., 2021 |
| Fleurs (102 languages) | ASR + language ID | Conneau et al., 2022 |
| CoVoST 2 | X→En translation, 21 languages | Wang et al., 2020b |

# Experiments

- **Datasets**: See complete list above; primary English ASR evaluation uses 12+ benchmarks in zero-shot setting.
- **Hardware**: Not specified in the paper.
- **Optimizer**: AdamW with linear LR decay, 2,048 warmup steps.
- **Results**:
  - **English OOD robustness**: Whisper Large V2 achieves **55.2% average relative WER reduction** over wav2vec 2.0 Large across 12 out-of-distribution datasets.
  - **Human-level ASR**: On Kincaid46, Whisper Large V2 achieves **4.1% WER** vs. 4.4% average for human transcribers.
  - **Multilingual ASR**: 7.3% WER on MLS (best among zero-shot models); 64.5% language ID accuracy on Fleurs (80.3% on 82 seen languages).
  - **Speech translation**: **29.1 BLEU** on CoVoST 2 zero-shot, state-of-the-art overall, with +6.7 BLEU advantage on low-resource languages.
  - **Dataset scaling**: Multilingual WER halves for every ~16× increase in per-language training hours ($R^2 = 0.83$ on log-log scale).
  - **Multitask/multilingual transfer**: Negative at small scale, positive at large scale — multilingual Large outperforms English-only Large on English ASR.
