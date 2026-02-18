# Meta Information

- URL: [[2302.13971] LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., & Lample, G. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971. Meta AI.

> [!CAUTION]
> NOTE comments are my personal understanding and may contain errors.

# 1. Introduction

Prior large language models (GPT-3, PaLM, Chinchilla) achieve strong performance by scaling model size and proprietary training data. LLaMA challenges this paradigm by demonstrating that **training smaller models on more publicly available tokens** produces better inference-time efficiency.

The key insight is that inference cost (not training cost) dominates in production. A 13B-parameter LLaMA model, trained on 1T tokens of public data, outperforms GPT-3 (175B parameters) on most standard benchmarks—delivering the same quality at ~13× smaller inference cost.

> [!IMPORTANT]
> LLaMA is released under a non-commercial research license, making competitive foundation models accessible to researchers without access to proprietary datasets or compute.

# 2. Pre-Training Data

LLaMA uses exclusively **publicly available** data. Training corpus totals approximately **1.4T tokens** for the 33B and 65B models, and **1.0T tokens** for the 7B and 13B models.

| Source | Proportion | Epochs |
|---|---|---|
| English CommonCrawl (CCNet) | 67.0% | 1.10 |
| C4 | 15.0% | 1.06 |
| GitHub | 4.5% | 0.64 |
| Wikipedia (20 languages) | 4.5% | 2.45 |
| Books (Gutenberg + Books3) | 4.5% | 2.23 |
| ArXiv | 2.5% | 1.06 |
| Stack Exchange | 2.0% | 1.03 |

**Preprocessing steps:**
- **CommonCrawl**: Line-level deduplication; language identification with fastText; quality filtering using an n-gram LM trained on Wikipedia references; linear classifier distinguishes Wikipedia-referenced vs. non-referenced content.
- **C4**: Deduplication + language identification; heuristic quality filters (punctuation presence, word/sentence count thresholds).
- **GitHub**: File-level deduplication; low-quality file removal based on line length and alphanumeric proportion; boilerplate stripping via regex.
- **Wikipedia**: Removal of hyperlinks, comments, and formatting boilerplate from 20-language dumps (June–August 2022).
- **ArXiv**: Pre-first-section content and bibliography removed; LaTeX macros expanded for consistency.

**Tokenizer**: Byte-pair encoding (BPE) using SentencePiece. All numbers are split into individual digits; unknown UTF-8 characters fall back to raw bytes. Vocabulary size: 32,000 tokens.

# 3. Architecture

LLaMA is a decoder-only Transformer. Three modifications from the vanilla Transformer (Vaswani et al., 2017) improve training stability and efficiency:

## 3.1 Pre-Normalization with RMSNorm

Instead of post-layer normalization, each sub-layer input is normalized using **RMSNorm** (Zhang & Sennrich, 2019):

```math
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot g, \quad \text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}
```

where $x \in \mathbb{R}^{d}$ is the input vector, $g \in \mathbb{R}^{d}$ is a learned gain parameter, and $d$ is the model hidden dimension. Pre-normalization stabilizes training without needing the bias term present in LayerNorm.

## 3.2 SwiGLU Activation

The FFN block replaces ReLU with the **SwiGLU** activation (Shazeer, 2020):

```math
\text{FFN}_{\text{SwiGLU}}(x) = \text{SiLU}(xW_1) \otimes (xW_3) \cdot W_2
```

where $W_1, W_3 \in \mathbb{R}^{d \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, and $\otimes$ denotes element-wise multiplication. To keep parameter count comparable to PaLM, the hidden FFN dimension is set to $d_{\text{ff}} = \frac{2}{3} \cdot 4d$ (rather than $4d$).

## 3.3 Rotary Positional Embeddings (RoPE)

Absolute positional embeddings are replaced with **Rotary Positional Embeddings** (RoPE, Su et al., 2021) applied to each Transformer layer's query and key vectors. RoPE encodes position by rotating the embedding in 2D subspaces:

```math
q_m^\top k_n = \text{Re}\left[(W_q x_m) \odot e^{im\theta}\right]^\top \left[(W_k x_n) \odot e^{in\theta}\right]
```

where $m, n$ are token positions, $\theta$ defines rotation frequencies, and $\text{Re}[\cdot]$ takes the real part. This approach encodes relative positions directly in the attention scores without modifying value vectors.

## 3.4 Efficient Attention

The implementation uses an efficient causal multi-head attention kernel (xformers library) that does not materialize the full $L \times L$ attention matrix, reducing memory usage for long sequences.

## 3.5 Model Configurations

| Model | Dim $d$ | Heads | Layers | FFN Dim | Params |
|---|---|---|---|---|---|
| LLaMA-7B | 4096 | 32 | 32 | ~11,008 | 6.7B |
| LLaMA-13B | 5120 | 40 | 40 | ~13,824 | 13.0B |
| LLaMA-33B | 6656 | 52 | 60 | ~17,920 | 32.5B |
| LLaMA-65B | 8192 | 64 | 80 | ~22,016 | 65.2B |

Input: token sequence $x \in \mathbb{Z}^{L}$ (integer token IDs, sequence length $L$).
Output: logit distribution over vocabulary at each position, $\hat{y} \in \mathbb{R}^{L \times V}$ where $V = 32{,}000$.

# 4. Training

**Optimizer**: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$)
**Learning rate schedule**: Cosine decay; peak LR decays to 10% of maximum
**Peak learning rates**: 3.0e-4 (7B, 13B), 1.5e-4 (33B, 65B)
**Batch size**: 4M tokens (achieved via gradient accumulation over 2,048-token sequences)
**Weight decay**: 0.1
**Gradient clipping**: 1.0
**Warmup steps**: 2,000

**Infrastructure**: 2,048 A100-80GB GPUs interconnected via InfiniBand. Throughput: ~380 tokens/sec/GPU for the 65B model. The 65B model trained for ~21 days.

**Estimated CO₂ emissions**: 1,015 tCO₂eq total (7B: 14 tCO₂eq, 65B: 173 tCO₂eq).

# 5. Main Results

## 5.1 Common Sense Reasoning (Zero-shot)

Evaluated on BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, OpenBookQA. LLaMA-65B surpasses Chinchilla-70B on most of these tasks and matches PaLM-540B, despite being 8× smaller.

> [!NOTE]
> "LLaMA-13B outperforms GPT-3 on most benchmarks" (from the abstract). GPT-3 has 175B parameters — LLaMA-13B is ~13× smaller.

## 5.2 Question Answering (Closed-book)

On **NaturalQuestions** and **TriviaQA**, LLaMA achieves state-of-the-art zero-shot and few-shot results. LLaMA-65B achieves 73.0% on TriviaQA (64-shot).

## 5.3 Mathematical Reasoning

On **GSM8k** (grade school math word problems), LLaMA-65B outperforms Minerva-62B despite no specialized mathematical fine-tuning. LLaMA-65B scores 69.7 on MATH.

## 5.4 Code Generation

On **HumanEval** (Python function synthesis from docstrings) and **MBPP** (programming challenges), LLaMA is competitive with specialized code models like CodeGen and PaLM.

## 5.5 Massive Multitask Language Understanding (MMLU)

LLaMA-65B achieves 63.4% (5-shot). This is slightly below Chinchilla-70B (67.5%), which the authors attribute to limited academic-domain text in training data compared to Chinchilla's Gopher dataset.

## 5.6 Instruction Fine-Tuning (LLaMA-I)

A brief fine-tuning on instruction-following data (as in Chung et al., 2022) brings LLaMA-65B to **68.9%** on MMLU, closely approaching specialized instruction-tuned baselines.

# 6. Bias and Toxicity Evaluation

| Benchmark | Finding |
|---|---|
| **RealToxicityPrompts** | Toxicity increases with model size; "respectful" prompts surprisingly elicit more toxic completions than "basic" prompts |
| **CrowS-Pairs** | LLaMA-65B shows higher bias than OPT-175B in the religion category (+10 pp) |
| **WinoGender** | Gender bias in co-reference resolution; performance drops for "gotcha" sentences where grammatical gender conflicts with stereotypical gender |
| **TruthfulQA** | Models frequently generate plausible-sounding but incorrect information; larger models are not necessarily more truthful |

> [!IMPORTANT]
> Bias and toxicity do not reliably decrease with scale. Researchers and practitioners using LLaMA must apply additional mitigations (RLHF, constitutional AI, etc.) before deploying in user-facing applications.

# 7. Comparison with Related Work

| Method | Params | Public Data Only | Inference Efficiency Focus |
|---|---|---|---|
| **GPT-3** | 175B | No (proprietary) | No |
| **PaLM** | 540B | No (proprietary) | No |
| **Chinchilla** | 70B | No | Yes (compute-optimal) |
| **OPT** | 175B | Yes | No |
| **BLOOM** | 176B | Yes | No |
| **LLaMA** | 7B–65B | **Yes** | **Yes (inference-optimal)** |

LLaMA differs from Chinchilla (Hoffmann et al., 2022) in objective: Chinchilla optimizes for **training compute** given a fixed budget, while LLaMA optimizes for **inference cost** by training smaller models on far more tokens than compute-optimally prescribed.

> [!TIP]
> The Chinchilla scaling laws suggest training a 70B model on 1.4T tokens is "over-trained" relative to training compute. LLaMA intentionally over-trains to get better per-parameter quality at inference.

# Experiments

- **Datasets**: CommonCrawl (CCNet), C4, GitHub code, Wikipedia (20 languages), Books (Project Gutenberg + Books3), ArXiv, Stack Exchange. Total: ~1.4T training tokens.
- **Evaluation benchmarks**: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, OpenBookQA (common sense); NaturalQuestions, TriviaQA (QA); RACE (reading comprehension); MATH, GSM8k (math); HumanEval, MBPP (code); MMLU (multitask); RealToxicityPrompts, CrowS-Pairs, WinoGender, TruthfulQA (safety).
- **Hardware**: 2,048 × NVIDIA A100 80GB GPUs connected via InfiniBand.
- **Optimizer**: AdamW, cosine LR schedule.
- **Key results**: LLaMA-13B outperforms GPT-3 (175B) on most benchmarks; LLaMA-65B matches or surpasses Chinchilla-70B and PaLM-540B on common sense and QA tasks.
