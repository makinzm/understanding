# Meta Information

- URL: [SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Taku Kudo and John Richardson (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226.

# Introduction

SentencePiece is a language-independent subword tokenization library designed for neural text processing systems. Unlike traditional tokenizers (e.g., Moses tokenizer, KyTea) that require pre-tokenized input and language-specific preprocessing, SentencePiece trains directly from raw sentences, enabling truly end-to-end neural systems. This approach eliminates dependencies on external tokenizers and ensures perfect reproducibility across different environments.

The system implements two subword segmentation algorithms:
1. **Byte-Pair Encoding (BPE)**: Iteratively merges the most frequent character pairs
2. **Unigram Language Model**: Probabilistic segmentation based on unigram statistics

> [!NOTE]
> The key innovation is treating whitespace as a normal symbol (U+2581 "▁") rather than using special boundary markers like `@@` in subword-nmt, enabling lossless tokenization.

# System Architecture

SentencePiece consists of four main components working in sequence:

## 1. Normalizer

**Input**: Raw text string (arbitrary Unicode)
**Output**: Normalized text string (canonical Unicode form)

The normalizer converts semantically-equivalent Unicode characters to their canonical forms. By default, it applies Unicode NFKC normalization, but supports custom normalization rules via TSV files.

Implementation uses the Aho-Corasick automaton for efficient pattern matching, allowing O(n) complexity where n is the input length.

Custom normalization example:
```
U+41 U+302 U+300 → U+1EA6
```

## 2. Trainer

**Input**: Normalized corpus (collection of sentences)
**Output**: Subword vocabulary and segmentation model

The trainer learns a subword segmentation model from the normalized corpus. It supports two algorithms:

### Byte-Pair Encoding (BPE)

BPE iteratively merges the most frequent adjacent character pairs until reaching the target vocabulary size $V$.

**Algorithm complexity**: O(N log N) where N is the corpus size
- Uses a binary heap to maintain symbol pair frequencies
- Each merge operation updates the heap in logarithmic time

**Parameters**:
- Vocabulary size $V$: final number of subword tokens
- Character coverage: percentage of characters to include (default: 0.9995)

### Unigram Language Model

The unigram model treats segmentation as a probabilistic problem, selecting the segmentation that maximizes the likelihood under the trained unigram distribution.

**Training algorithm**:
1. Initialize with a large seed vocabulary (e.g., all characters + common n-grams)
2. Iteratively remove symbols that increase the overall likelihood the least
3. Continue until reaching target vocabulary size $V$

**Segmentation objective**:
$$
\arg\max_{s \in S(x)} P(s) = \arg\max_{s \in S(x)} \prod_{i=1}^{|s|} P(s_i)
$$

where $x$ is the input sentence, $S(x)$ is the set of all possible segmentations, and $s_i$ are the subword tokens.

**Complexity**: Linear in corpus size for both training and inference

## 3. Encoder

**Input**: Raw text sentence
**Output**: Sequence of subword tokens

The encoder first normalizes the input, then applies the learned segmentation model to produce a sequence of subword tokens.

**Processing steps**:
1. Apply normalization (if configured)
2. Segment into subwords using the trained model
3. Return token sequence

**Output representation**:
- Whitespace encoded as `_` (U+2581)
- Example: `"New York"` → `["▁New", "▁York"]`

## 4. Decoder

**Input**: Sequence of subword tokens
**Output**: Normalized text

The decoder reverses the encoding process:

$$
\text{Decode}(\text{Encode}(\text{Normalize}(\text{text}))) = \text{Normalize}(\text{text})
$$

This lossless property ensures perfect reconstruction, critical for applications requiring exact text recovery.

**Decoding steps**:
1. Concatenate all subword tokens
2. Replace `_` (U+2581) with whitespace
3. Return normalized text

# Key Design Features

## Lossless Tokenization

Traditional tokenization with `@@` boundary markers (used in subword-nmt) can create ambiguities:
- `"quoted text"` → `["qu", "ot", "ed", "@@", "text"]`
- `"consecutive  spaces"` → information loss

SentencePiece solves this by:
1. Treating whitespace as a regular token (`_`)
2. Never using special boundary markers
3. Ensuring bijective mapping between raw and tokenized text

## Vocabulary Management

Unlike BPE-specific tools that specify merge operations, SentencePiece allows direct specification of the final vocabulary size $V$. This provides:
- Consistent interface across algorithms (BPE and unigram)
- Easier hyperparameter tuning
- Predictable model size

**Example vocabulary sizes**:
- Small models: 8k tokens
- Medium models: 16k-32k tokens
- Large models: 64k tokens
- Shared multilingual: 32k-64k tokens

## Efficient Implementation

**BPE segmentation complexity**: O(N log N)
- Binary heap maintains symbol pair frequencies
- Each merge updates heap in O(log N)
- Contrast with naive O(N²) implementations

**Unigram model complexity**: O(N)
- Forward-backward algorithm for computing marginal probabilities
- Viterbi algorithm for finding best segmentation
- Both linear in sentence length

**Performance benchmarks** (440k sentences from KFTT):

| Algorithm | Japanese (raw) | English (raw) |
|-----------|----------------|---------------|
| subword-nmt training | 528.0 sec | 94.7 sec |
| SentencePiece training | 217.3 sec | 21.8 sec |
| subword-nmt segmentation | 216.2 sec | 36.1 sec |
| SentencePiece segmentation | 5.9 sec | 20.3 sec |

**Speed**: Approximately 21k sentences/second for English, 74k sentences/second for Japanese on single-threaded CPU.

## Self-Contained Model Files

SentencePiece models are fully self-contained using Protocol Buffer serialization:

**Model file contains**:
1. Vocabulary (subword tokens with IDs and scores)
2. Segmentation algorithm type (BPE or unigram)
3. Normalization rules (pre-compiled FST)
4. All hyperparameters

**Benefits**:
- No external dependencies
- Perfect reproducibility across platforms
- Version compatibility guaranteed
- Easy deployment in production systems

## Multiple API Support

### C++ API
Native implementation providing maximum performance:
```cpp
sentencepiece::SentencePieceProcessor sp;
sp.Load("model.model");
std::vector<std::string> pieces;
sp.Encode("Hello world", &pieces);
```

### Python API
High-level interface with identical functionality:
```python
import sentencepiece as sp
sp_model = sp.SentencePieceProcessor()
sp_model.Load("model.model")
pieces = sp_model.EncodeAsPieces("Hello world")
```

### TensorFlow API
Integrates directly into TensorFlow computation graphs:
```python
import tf_sentencepiece as tfsp
indices = tfsp.encode(["Hello world"], model_file="model.model")
```

This enables:
- On-the-fly tokenization during training
- Dynamic data augmentation via subword regularization
- Deployment without external preprocessing

## Subword Regularization

SentencePiece supports stochastic segmentation for data augmentation. Instead of deterministic encoding, it samples from multiple possible segmentations according to their probabilities under the unigram model.

**Sampling example** for `"New York"`:
1. `['▁', 'N', 'e', 'w', '▁York']`
2. `['▁New', '▁York']`
3. `['▁New', '▁Y', 'o', 'r', 'k']`
4. `['▁New', '▁York']`
5. `['▁New', '▁York']`

**API usage**:
```python
sp.SampleEncodeAsPieces('New York', nbest_size=-1, alpha=0.1)
```

**Parameters**:
- `nbest_size`: Number of segmentation candidates (-1 for all)
- `alpha`: Smoothing parameter (higher = more randomness)

This technique improves model robustness by injecting controlled noise during training, similar to dropout or word dropout.

# Experiments

## Dataset: Kyoto Free Translation Task (KFTT)

- **Domain**: Wikipedia articles related to Kyoto
- **Language pair**: English ↔ Japanese
- **Training set**: 440,000 sentence pairs
- **Development set**: 1,166 sentence pairs
- **Test set**: 1,160 sentence pairs

## Model Architecture: Google Neural Machine Translation (GNMT)

- **Architecture**: Encoder-decoder with attention
- **LSTM nodes**: 512 units per layer
- **LSTM layers**: 6 layers (encoder) + 6 layers (decoder)
- **Attention mechanism**: Luong-style multiplicative attention
- **Evaluation metric**: Case-sensitive BLEU

## Baseline Tokenization

- **English**: Moses tokenizer
- **Japanese**: KyTea word segmenter
- **Vocabulary**: 80k tokens for source, 80k tokens for target (160k total)

## SentencePiece Configuration

- **Vocabulary**: 8k shared tokens (source and target use same vocabulary)
- **Algorithm**: BPE and unigram (both tested)
- **Character coverage**: 1.0 (100% coverage)
- **Shared vocabulary**: Enables parameter sharing in multilingual models

## Translation Quality Results

### Japanese → English

| System | Vocabulary Size | BLEU |
|--------|----------------|------|
| Word baseline | 80k + 80k | 28.24 |
| SentencePiece (raw) | 8k shared | 29.55 |
| SentencePiece (pre-tok) | 8k shared | **29.85** |

**Key findings**:
- SentencePiece without pre-tokenization achieves **+1.31 BLEU** over word baseline
- With pre-tokenization (KyTea + Moses), gains **+1.61 BLEU**
- Shared vocabulary (8k total) outperforms separate vocabularies (160k total)

### English → Japanese

| System | Vocabulary Size | BLEU |
|--------|----------------|------|
| Word baseline | 80k + 80k | 20.06 |
| SentencePiece (pre-tok) | 8k shared | **21.62** |
| SentencePiece (raw) | 8k shared | 20.86 |

**Key findings**:
- SentencePiece with pre-tokenization achieves **+1.56 BLEU** over baseline
- Without pre-tokenization, gains **+0.80 BLEU**
- Japanese benefits more from linguistic pre-tokenization due to lack of whitespace

> [!IMPORTANT]
> The asymmetry between ja→en and en→ja suggests that language-specific preprocessing remains beneficial for morphologically complex languages without whitespace delimiters, even with SentencePiece.

## Training and Segmentation Speed

### Training Time (440k sentences)

| Tool | Japanese (raw) | English (raw) |
|------|----------------|---------------|
| subword-nmt | 528.0 sec | 94.7 sec |
| SentencePiece | 217.3 sec | 21.8 sec |

**Speedup**: 2.4x for Japanese, 4.3x for English

### Segmentation Time (440k sentences)

| Tool | Japanese (raw) | English (raw) |
|------|----------------|---------------|
| subword-nmt | 216.2 sec | 36.1 sec |
| SentencePiece | 5.9 sec | 20.3 sec |

**Speedup**: 36.6x for Japanese, 1.8x for English

> [!NOTE]
> The dramatic speedup for Japanese is due to eliminating the need for external morphological analyzers (KyTea), which are computationally expensive.

# Comparison with Existing Tools

## SentencePiece vs. subword-nmt

| Feature | subword-nmt | SentencePiece |
|---------|-------------|---------------|
| Pre-tokenization | Required | Optional |
| Whitespace handling | `@@` markers | `_` (U+2581) meta-symbol |
| Lossless tokenization | No | Yes |
| Training complexity | O(N²) | O(N log N) |
| Vocabulary specification | Merge operations | Final size |
| Self-contained models | No | Yes (Protocol Buffer) |
| Language support | Requires language-specific tokenizers | Language-independent |
| Unigram model | No | Yes |
| Subword regularization | No | Yes |

## SentencePiece vs. WordPiece (BERT tokenizer)

| Feature | WordPiece | SentencePiece |
|---------|-----------|---------------|
| Algorithm | Greedy longest match | BPE or unigram |
| Training data | Pre-tokenized | Raw text |
| Whitespace | `##` prefix for continuations | `_` prefix for word starts |
| Decoding | Requires special handling | Lossless concatenation |
| Open source | No (only inference) | Yes (training + inference) |

## SentencePiece vs. tiktoken (OpenAI tokenizer)

| Feature | tiktoken | SentencePiece |
|---------|----------|---------------|
| Algorithm | BPE with regex pre-splitting | BPE or unigram on raw text |
| Pre-tokenization | Regex-based splitting | Optional |
| Training | Not publicly available | Fully open source |
| Performance | Rust implementation (fast) | C++ implementation (fast) |
| Subword regularization | No | Yes |

# Key Innovations

## 1. Direct Raw Text Training

Traditional pipeline:
```
Raw text → Language-specific tokenizer → Subword segmenter → Neural model
```

SentencePiece pipeline:
```
Raw text → SentencePiece → Neural model
```

**Benefits**:
- Eliminates dependency on external tokenizers
- Reduces error propagation from tokenization mistakes
- Enables true end-to-end learning
- Works for any language (including low-resource languages)

## 2. Lossless Tokenization via Meta-Symbol

**Problem with `@@` markers**:
- Ambiguous for consecutive whitespace: `"a  b"` → `["a", "@@", "b"]` loses information
- Special case handling required for decoder

**SentencePiece solution**:
- Represent whitespace as `_` (U+2581)
- Example: `"Hello world"` → `["▁Hello", "▁world"]`
- Decoding: concatenate and replace `_` with space

**Mathematical property**:
$$
\forall \text{text} : \text{Decode}(\text{Encode}(\text{Normalize}(\text{text}))) = \text{Normalize}(\text{text})
$$

## 3. Unified Vocabulary Management

**Traditional BPE approach**:
- Specify number of merge operations (e.g., 10,000 merges)
- Final vocabulary size is unpredictable
- Different tools produce different vocabulary sizes for same merge count

**SentencePiece approach**:
- Directly specify final vocabulary size $V$ (e.g., 32,000 tokens)
- Works identically for BPE and unigram algorithms
- Predictable model size and memory usage

## 4. Custom Meta-Symbols for Contextual Information

SentencePiece allows arbitrary meta-symbols to encode contextual information:

**Multilingual models**:
```
<2ja> 今日は良い天気です
<2en> The weather is nice today
```

**Domain-specific models**:
```
<medical> diagnosis: hypertension
<legal> defendant pleads not guilty
```

**Sentiment markers**:
```
<pos> This movie is amazing!
<neg> This movie is terrible!
```

These meta-symbols are treated as regular tokens and included in the vocabulary, enabling the model to condition on context.

# Applications and Use Cases

## 1. Neural Machine Translation

**Advantages**:
- Shared vocabulary for multilingual models
- Handles rare words via subword decomposition
- Eliminates out-of-vocabulary (OOV) problem
- Reduces vocabulary size from 100k+ to 8k-32k

**Typical configuration**:
- Vocabulary: 32k shared tokens
- Algorithm: BPE or unigram
- Character coverage: 0.9995 (covers 99.95% of characters)

## 2. Language Modeling (BERT, GPT, etc.)

**BERT-style models**:
- Pre-training: SentencePiece with 32k vocabulary
- Fine-tuning: Same tokenizer ensures compatibility
- Multilingual BERT: 110k vocabulary covering 104 languages

**GPT-style models**:
- GPT-2: BPE with 50k vocabulary
- GPT-3: BPE with 50k vocabulary
- Llama: SentencePiece with 32k vocabulary

## 3. Speech Recognition

**End-to-end ASR**:
- Input: Acoustic features (e.g., Mel spectrograms)
- Output: SentencePiece tokens
- Advantage: Reduces output vocabulary, improving convergence

**Typical configuration**:
- Vocabulary: 1k-5k tokens (smaller than NMT)
- Algorithm: Unigram (better for speech due to probabilistic nature)

## 4. Text-to-Speech

**Neural TTS models**:
- Input: SentencePiece tokens
- Output: Acoustic features
- Advantage: Handles rare words and foreign names gracefully

## 5. Cross-Lingual Transfer

**Zero-shot transfer**:
- Train on high-resource language (e.g., English)
- Transfer to low-resource language (e.g., Swahili)
- Shared vocabulary enables parameter sharing

**Few-shot adaptation**:
- Pre-train on multiple languages
- Fine-tune on target language with limited data

# Implementation Details

## Model File Format

SentencePiece uses Protocol Buffers for serialization:

```protobuf
message ModelProto {
  repeated SentencePiece pieces = 1;
  TrainerSpec trainer_spec = 2;
  NormalizerSpec normalizer_spec = 3;
}

message SentencePiece {
  string piece = 1;    // Subword token
  float score = 2;     // Log probability
  Type type = 3;       // NORMAL, UNKNOWN, CONTROL, etc.
}
```

**Fields**:
- `pieces`: Vocabulary with tokens, scores, and types
- `trainer_spec`: Training configuration (algorithm, vocab size, etc.)
- `normalizer_spec`: Normalization rules (pre-compiled FST)

## Character Normalization FST

The normalizer is implemented as a finite-state transducer (FST):

**States**:
- $q_0$: Initial state
- $q_1, q_2, \ldots$: Intermediate states
- $q_f$: Final state

**Transitions**:
- $(q_i, c, c', q_j)$: From state $q_i$, reading character $c$, output $c'$, go to $q_j$

**Implementation**:
- Aho-Corasick automaton for pattern matching
- O(n) time complexity where n is input length
- Pre-compiled into model file for efficiency

## Training Algorithms

### BPE Training Pseudocode

```
function TrainBPE(corpus, vocab_size):
    # Initialize vocabulary with characters
    vocab = {c: count(c) for c in unique_chars(corpus)}
    pairs = compute_pair_frequencies(corpus)
    heap = BinaryHeap(pairs)

    while len(vocab) < vocab_size:
        # Get most frequent pair
        (left, right), freq = heap.pop_max()

        # Merge pair into new token
        new_token = left + right
        vocab[new_token] = freq

        # Update corpus and pair frequencies
        corpus = replace_all(corpus, (left, right), new_token)
        pairs = update_pair_frequencies(corpus, new_token)
        heap.update(pairs)

    return vocab
```

### Unigram Training Pseudocode

```
function TrainUnigram(corpus, vocab_size):
    # Initialize with large seed vocabulary
    vocab = initialize_seed_vocab(corpus)

    # Iteratively prune vocabulary
    while len(vocab) > vocab_size:
        # Compute likelihood for each token
        likelihoods = {}
        for token in vocab:
            # Likelihood of corpus without this token
            likelihoods[token] = compute_likelihood(corpus, vocab - {token})

        # Remove token with smallest likelihood loss
        token_to_remove = argmin(likelihoods)
        vocab.remove(token_to_remove)

        # Re-estimate probabilities
        vocab = update_probabilities(corpus, vocab)

    return vocab
```

## Encoding Algorithm (Unigram)

```
function Encode(sentence, vocab, probs):
    # Dynamic programming to find best segmentation
    n = len(sentence)
    best_score = [-inf] * (n + 1)
    best_score[0] = 0
    best_segmentation = [None] * (n + 1)

    for i in range(n):
        for token in vocab:
            if sentence[i:].startswith(token):
                j = i + len(token)
                score = best_score[i] + log(probs[token])
                if score > best_score[j]:
                    best_score[j] = score
                    best_segmentation[j] = (i, token)

    # Backtrack to get segmentation
    result = []
    pos = n
    while pos > 0:
        i, token = best_segmentation[pos]
        result.append(token)
        pos = i

    return reversed(result)
```

**Complexity**: O(n × |V|) where n is sentence length and |V| is vocabulary size.

# Limitations and Considerations

## 1. Pre-tokenization Trade-off

**Observation from experiments**:
- English → Japanese benefits from linguistic pre-tokenization (+0.76 BLEU)
- Japanese → English shows marginal gain (+0.30 BLEU)

**When to use pre-tokenization**:
- Target language lacks whitespace (Japanese, Chinese, Thai)
- Linguistic features are important (morphology, compounds)
- Training data is limited (pre-tokenization provides inductive bias)

**When to skip pre-tokenization**:
- Source language has clear word boundaries (English, Spanish)
- Large training data available
- Deployment requires language-agnostic system

## 2. Vocabulary Size Selection

**Trade-offs**:
- Small vocabulary (4k-8k): Faster training, longer sequences
- Medium vocabulary (16k-32k): Balanced performance
- Large vocabulary (64k+): More parameters, shorter sequences

**Rule of thumb**:
- Translation: 16k-32k shared
- Language modeling: 32k-50k
- Speech recognition: 1k-5k
- Character-level tasks: 1k-2k

## 3. Character Coverage

**Default**: 0.9995 (99.95% of characters)

**Effect of coverage parameter**:
- 1.0: Include all characters (may include noise/typos)
- 0.9995: Exclude rare characters (recommended)
- 0.995: More aggressive filtering

**Characters with frequency below threshold** are mapped to `<unk>` token.

## 4. Computational Cost

**Training**:
- BPE: O(N log N) where N is corpus size
- Unigram: O(N × I) where I is number of iterations
- Typical training time: Minutes to hours depending on corpus size

**Inference**:
- BPE: O(n log n) per sentence
- Unigram: O(n × |V|) per sentence (dynamic programming)
- Typical speed: 20k-75k sentences/second on CPU

# Availability and Licensing

- **GitHub**: https://github.com/google/sentencepiece
- **License**: Apache License 2.0
- **Languages**: C++ (core), Python (bindings), TensorFlow (integration)
- **Platforms**: Linux, macOS, Windows
- **Installation**:
  ```bash
  pip install sentencepiece
  ```

**Command-line tools**:
```bash
# Train model
spm_train --input=data.txt --model_prefix=model --vocab_size=32000

# Encode
spm_encode --model=model.model < input.txt > output.txt

# Decode
spm_decode --model=model.model < input.txt > output.txt
```

# Differences from Similar Algorithms

## SentencePiece vs. BPE (Original)

| Aspect | Original BPE (Sennrich 2016) | SentencePiece BPE |
|--------|------------------------------|-------------------|
| Input | Pre-tokenized words | Raw sentences |
| Whitespace | `@@` suffix markers | `_` prefix meta-symbol |
| Lossless | No | Yes |
| Training | O(N²) naive | O(N log N) optimized |
| Vocabulary | Merge operations | Final size |

**Key innovation**: SentencePiece eliminates pre-tokenization requirement while improving algorithmic efficiency.

## SentencePiece vs. Unigram LM (Original)

| Aspect | Original Unigram LM (Kudo 2018) | SentencePiece Unigram |
|--------|---------------------------------|----------------------|
| Training | EM algorithm | Iterative pruning |
| Regularization | Multiple samples | Same (supports sampling) |
| Implementation | Research code | Production-ready C++ |

**Key innovation**: SentencePiece provides efficient implementation with self-contained models.

## SentencePiece vs. CharBPE

| Aspect | CharBPE | SentencePiece |
|--------|---------|---------------|
| Initial units | Characters | Characters + meta-symbols |
| Whitespace | Ignored or special token | `_` meta-symbol |
| Vocabulary | Character-derived only | Includes special tokens |

**Key innovation**: SentencePiece's meta-symbol approach enables seamless whitespace handling.

# Related Work and Context

## Subword Segmentation History

1. **Schütze (1992)**: Character n-grams for information retrieval
2. **Goldsmith (2001)**: Unsupervised morphology learning (Linguistica)
3. **Creutz & Lagus (2002)**: Morfessor for morphological segmentation
4. **Sennrich et al. (2016)**: BPE for neural machine translation
5. **Wu et al. (2016)**: WordPiece for Google NMT
6. **Kudo (2018)**: Subword regularization
7. **Kudo & Richardson (2018)**: SentencePiece (this work)

## Influence on Later Work

**Models using SentencePiece**:
- T5 (Raffel et al., 2020): 32k vocabulary
- ALBERT (Lan et al., 2020): Shared embeddings
- XLM-R (Conneau et al., 2020): 250k multilingual vocabulary
- mT5 (Xue et al., 2021): 101 languages
- Llama (Touvron et al., 2023): 32k vocabulary

**Extensions**:
- Provilkov et al. (2020): BPE-Dropout for subword regularization
- Bostrom & Durrett (2020): Learning to segment for better generalization

# Conclusion

SentencePiece provides a language-independent, efficient, and reproducible solution for subword tokenization in neural text processing. Its key contributions are:

1. **Language independence**: Trains directly from raw text without external tokenizers
2. **Lossless tokenization**: Perfect reconstruction via meta-symbol representation
3. **Efficient algorithms**: O(N log N) BPE and O(N) unigram implementations
4. **Self-contained models**: Protocol Buffer serialization ensures reproducibility
5. **Multiple APIs**: C++, Python, and TensorFlow integration
6. **Subword regularization**: Stochastic segmentation for robust training

The system has been widely adopted in production NMT systems (Google Translate), large language models (T5, Llama), and multilingual models (XLM-R, mT5), demonstrating its practical effectiveness and scalability.

**Applicability**: SentencePiece is suitable for any neural text processing task requiring vocabulary reduction, including machine translation, language modeling, speech recognition, and cross-lingual transfer. It is particularly valuable for low-resource languages and multilingual systems where pre-tokenization tools are unavailable or inconsistent.
