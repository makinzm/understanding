# Meta Information

- URL: [Bridging the Language Gap: Enhancing Multilingual Prompt-Based Code Generation in LLMs via Zero-Shot Cross-Lingual Transfer](https://arxiv.org/abs/2408.09701)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Li, M., Mishra, A., & Mujumdar, U. (2024). Bridging the Language Gap: Enhancing Multilingual Prompt-Based Code Generation in LLMs via Zero-Shot Cross-Lingual Transfer. arXiv:2408.09701.

# Bridging the Language Gap: Enhancing Multilingual Prompt-Based Code Generation in LLMs via Zero-Shot Cross-Lingual Transfer

Code-generation LLMs exhibit a significant quality gap when prompted in non-English languages versus English, because their pretraining corpora are heavily English-dominated. This paper diagnoses that gap across Chinese, Spanish, Japanese, Russian, and Hindi, evaluates naive fixes (back-translation via Chain-of-Thought, bootstrapped fine-tuning), and proposes a lightweight zero-shot remedy: a two-layer linear projection that maps LASER cross-lingual encoder embeddings into the LLM's token embedding space. Training requires only English examples, yet the model generalises to all languages supported by LASER at inference time.

**Applicability**: Practitioners deploying code-generation models in multilingual products (developer tools, educational platforms, enterprise IDEs) where end-users write prompts in their native language. Also relevant to NLP researchers studying cross-lingual transfer for generation tasks.

## Background and Motivation

### The Multilingual Code Generation Problem

Current instruction-tuned LLMs such as CodeLLaMa, CodeGemma, and Mistral are trained on corpora that are heavily skewed toward English. When a user submits a programming task in Japanese or Hindi, the model may misunderstand the intent, hallucinate semantically incorrect code, or produce syntax errors that do not arise for the equivalent English prompt.

Three straightforward remedies exist, each with a known failure mode:

| Approach | Idea | Observed Failure |
|---|---|---|
| Direct multilingual prompting | Feed the non-English prompt as-is | High logical and syntax error rates |
| Chain-of-Thought back-translation | Ask the model to first translate the prompt to English, then generate code | Marginal and inconsistent improvement; translation errors propagate |
| Bootstrapped fine-tuning (BFT) | Generate synthetic English problems, translate them, fine-tune on round-tripped pairs | Reduced syntax errors but increased hallucination (logical errors) |

The authors argue that none of these fully closes the gap because they either ignore the cross-lingual embedding structure or introduce noisy training signal.

### LASER: Cross-Lingual Sentence Embeddings

LASER (Language-Agnostic SEntence Representations) is a pre-trained multilingual encoder that maps text in more than 200 languages to a shared 1024-dimensional vector space. Its key property is language-agnostic alignment: semantically equivalent sentences from different languages are mapped to nearby points. For example, the English token "add" and its Hindi counterpart produce similar embedding vectors.

> [!NOTE]
> "A pre-trained multilingual encoder like LASER encodes inputs into a joint vector space supporting over 200 languages, where the English token 'add' and its Hindi counterpart are embedded similarly."

## Proposed Method: LASER Projection (LP)

### Architecture

The method inserts a small learnable projector between the LASER encoder and the frozen LLM. The pipeline is:

1. **Tokenise** the non-English prompt using a language-appropriate word tokeniser (NLTK for most languages; Jieba for Chinese; Janome for Japanese).
2. **Encode** each token with LASER to obtain a sequence of 1024-dimensional embeddings.
3. **Mean-pool** the subword-level embeddings produced by LASER for each word token, yielding one vector per word.
4. **Project** each pooled embedding from 1024 dimensions into the LLM's embedding dimension (typically 4096) via two linear layers.
5. **Prepend** the projected embeddings to the LLM's own embedding sequence and run autoregressive decoding as usual.

### Projection Formula

Let $H_{\text{laser}} \in \mathbb{R}^{T \times 1024}$ be the sequence of LASER embeddings for $T$ word tokens. The projection produces:

```math
\begin{align}
  H_{\text{llm}} = W_{\text{llm}} \cdot H_{\text{laser}} + b_{\text{llm}}
\end{align}
```

where $W_{\text{llm}} \in \mathbb{R}^{d_{\text{llm}} \times 1024}$ and $b_{\text{llm}} \in \mathbb{R}^{d_{\text{llm}}}$ are the only trainable parameters, and $d_{\text{llm}} = 4096$ for all three tested models. The projected sequence $\hat{H}_{\text{llm}} \in \mathbb{R}^{T \times 4096}$ is prepended to the LLM's standard token embeddings before the transformer layers.

### Training Objective

The projector is trained to minimise mean squared error between the projected LASER embeddings and the LLM's own English token embeddings for the same text:

```math
\begin{align}
  \mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \hat{H}^{(i)}_{\text{llm}} - H^{(i)}_{\text{llm}} \right\|^2
\end{align}
```

where $N$ is the number of training examples and $H^{(i)}_{\text{llm}}$ is the LLM's embedding of the $i$-th English sentence. The LLM weights are fully frozen throughout.

> [!IMPORTANT]
> Training uses only 127 English MBPP examples. No multilingual supervision is required at training time. Generalisation to other languages at inference is entirely due to LASER's cross-lingual alignment.

### Algorithm: LP Training Procedure

```
Input:  English MBPP training set D_en (127 examples),
        frozen LLM with embedding layer E_llm,
        LASER encoder f_laser

Output: Trained projection W_llm, b_llm

For epoch = 1 to 200:
    For each sentence s_i in D_en:
        tokens   = word_tokenise(s_i)              # NLTK
        H_laser  = mean_pool(f_laser(tokens))      # R^{T x 1024}
        H_hat    = W_llm @ H_laser + b_llm         # R^{T x 4096}
        H_ref    = E_llm(tokenise_llm(s_i))        # R^{T' x 4096}
        loss     = MSE(H_hat, H_ref)
        update W_llm, b_llm via Adam
Return W_llm, b_llm
```

Training completes in under one hour on a single NVIDIA RTX 4060 consumer GPU.

### Comparison with Bootstrapped Fine-Tuning (BFT)

BFT generates synthetic training data using a strong LLM (ChatGPT), translates the generated problems to a target language, back-translates to English, filters pairs with BLEU ≥ 0.8, and fine-tunes the target model with LoRA. Unlike LP, BFT:

- Requires target-language data and a translation service.
- Modifies the LLM weights, risking catastrophic forgetting and hallucination.
- Must be repeated separately for each new language.

LP avoids all three issues: no target-language training data, no weight modification, and a single trained projector generalises to all LASER-supported languages.

## Datasets

### Base Dataset: MBPP (Mostly Basic Python Problems)

- **Source**: Austin et al. (2021)
- **Size**: 257 problems used in evaluation; 127 English examples used for LP training; 378 examples for BFT fine-tuning
- **Format**: Each problem consists of a natural language description, a reference Python solution, and three unit tests
- **Language**: Originally English only

### Multilingual MBPP Translation

The authors constructed translations of the 257 evaluation problems into five languages:

| Language | Code | Translation Tool | Quality Validation |
|---|---|---|---|
| Chinese (Simplified) | zh-cn | Google Translate API | Human + GPT-4 |
| Spanish | es | Google Translate API | Human + GPT-4 |
| Japanese | ja | Google Translate API | Human + GPT-4 |
| Russian | ru | Google Translate API | Human + GPT-4 |
| Hindi | hi | Google Translate API | Human + GPT-4 |

Quality was assessed by expert bilingual raters via Amazon Mechanical Turk, achieving 89–91% inter-rater agreement. GPT-4 ratings averaged 4.87–4.95 out of 5.0 across all languages, indicating high translation fidelity.

## Evaluation Metrics

Four metrics quantify code generation quality. All are computed by running the generated Python code against the three MBPP unit tests.

| Metric | Symbol | Definition |
|---|---|---|
| Syntax Error Rate | SER | Fraction of samples with Python syntax errors (parse failure) |
| Logical Error Rate | LER | Fraction of samples that run without syntax error but fail ≥ 1 test case |
| Total Error Rate | TotalER | Fraction of samples that fail ≥ 1 test case (SER + LER combined) |
| All Tests Passed Rate | ATPR | Fraction of samples that pass all three test cases |
| Code Completion Rate | CCR | Fraction of model responses containing a complete code block |

$\text{TotalER} = \text{SER} + \text{LER}$ and $\text{ATPR} + \text{TotalER} = 100\%$ when CCR = 100%.

## Experiments

### Setup

- **Models**: CodeLLaMa-7B-Instruct, CodeGemma-7B-IT, Mistral-7B-Instruct-v0.3
- **Reference baseline**: GPT-4 (closed-source; English only)
- **Hardware**: Single NVIDIA RTX 4060 GPU
- **Baselines**: Orig. (direct multilingual prompt), CoT (back-translation chain-of-thought), BFT (bootstrapped fine-tuning with LoRA)

### Key Results

LP (LASER Projection) consistently achieves lower TotalER and higher ATPR compared to all baselines across most language-model pairs.

Selected results for CodeLLaMa-7B:

| Language | Orig. ATPR | CoT ATPR | BFT ATPR | LP ATPR |
|---|---|---|---|---|
| English | 12.84 | — | — | 24.51 |
| Chinese | 6.23 | ~6 | ~8 | 17.90 |
| Spanish | ~10 | ~10 | ~12 | 18.29 |
| Hindi | ~2 | ~2 | ~3 | 4.28 |

> [!NOTE]
> Exact values for Orig./CoT/BFT for non-English languages are reported per-model in the paper's tables; the LP column shows consistent improvements in ATPR and reductions in LER. Hindi benefits least due to its lower resource availability in LASER's training data.

> [!IMPORTANT]
> BFT reduces SER but simultaneously increases LER across most settings, resulting in overall TotalER that is comparable to or worse than direct prompting. This indicates that fine-tuning on noisy round-tripped data causes the model to hallucinate semantically incorrect code more often.

All three models with LP maintained CCR above 90% across all languages, comparable to the original prompting baseline.

## Limitations

- **Word tokenisation dependency**: The pipeline relies on language-specific tokenisers (NLTK, Jieba, Janome). Languages without available tokenisers cannot be supported without additional tooling.
- **Low-resource language ceiling**: LASER's cross-lingual alignment degrades for very low-resource languages, placing an upper bound on LP's effectiveness in those settings.
- **Python only**: Evaluation is limited to Python code generation; transfer to other programming languages is not validated.
- **Five languages**: Evaluation covers only zh-cn, es, ja, ru, hi; broader language coverage is left to future work.
- **Bias risks**: LLM-generated code for low-resource language prompts may reflect cultural and racial biases present in the pretraining data; the authors recommend establishing monitoring protocols.

## References

- Artetxe, M., & Schwenk, H. (2019). Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and beyond. *TACL*.
- Austin, J., et al. (2021). Program synthesis with large language models. *arXiv:2108.07732*. (MBPP dataset)
- Rozière, B., et al. (2023). Code LLaMa: Open foundation models for code. *arXiv:2308.12950*.
- Team, C. (2024). CodeGemma: Open code models based on Gemma. *arXiv:2406.11409*.
- Jiang, A. Q., et al. (2023). Mistral 7B. *arXiv:2310.06825*.
- Hu, E., et al. (2021). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
- Papineni, K., et al. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL*.
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*.
- Liu, P., et al. (2023). Pre-train, prompt, and predict: A systematic survey of prompting methods in NLP. *ACM Computing Surveys*.
