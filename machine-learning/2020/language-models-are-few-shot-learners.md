# Meta Information

- URL: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)

```bibtex
@article{DBLP:journals/corr/abs-2005-14165,
  author       = {Tom B. Brown and
                  Benjamin Mann and
                  Nick Ryder and
                  Melanie Subbiah and
                  Jared Kaplan and
                  Prafulla Dhariwal and
                  Arvind Neelakantan and
                  Pranav Shyam and
                  Girish Sastry and
                  Amanda Askell and
                  Sandhini Agarwal and
                  Ariel Herbert{-}Voss and
                  Gretchen Krueger and
                  Tom Henighan and
                  Rewon Child and
                  Aditya Ramesh and
                  Daniel M. Ziegler and
                  Jeffrey Wu and
                  Clemens Winter and
                  Christopher Hesse and
                  Mark Chen and
                  Eric Sigler and
                  Mateusz Litwin and
                  Scott Gray and
                  Benjamin Chess and
                  Jack Clark and
                  Christopher Berner and
                  Sam McCandlish and
                  Alec Radford and
                  Ilya Sutskever and
                  Dario Amodei},
  title        = {Language Models are Few-Shot Learners},
  journal      = {CoRR},
  volume       = {abs/2005.14165},
  year         = {2020},
  url          = {https://arxiv.org/abs/2005.14165},
  eprinttype    = {arXiv},
  eprint       = {2005.14165},
  timestamp    = {Thu, 25 May 2023 10:38:31 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2005-14165.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

# Introduction

GPT-3 is a 175-billion parameter autoregressive language model that demonstrates task-agnostic performance through in-context learning, where the model receives only textual task descriptions and a few demonstrations without requiring gradient-based fine-tuning. This approach contrasts with the dominant paradigm in NLP where models are pre-trained on large corpora then fine-tuned on specific downstream tasks with thousands of labeled examples.

> [!NOTE]
> The paper defines "in-context learning" as using the model's text input to specify tasks, rather than through weight updates. This allows the model to perform tasks it was never explicitly trained on.

The central hypothesis is that scaling up language models greatly improves task-agnostic, few-shot performance, eventually reaching competitiveness with prior state-of-the-art fine-tuning approaches.

## Limitations of Fine-Tuning

Traditional fine-tuning has three key drawbacks:
1. **Data requirement**: Thousands or tens of thousands of labeled examples are needed for each task
2. **Exploitation potential**: Poor out-of-distribution generalization enables adversarial exploitation
3. **Human learning mismatch**: Humans typically require far fewer examples to learn new tasks

Large language models can potentially address these limitations by learning tasks from natural language descriptions and a handful of demonstrations.

# Approach

## Model Architecture

GPT-3 uses the same architecture as GPT-2, which is a transformer-based decoder-only model with the following modifications:
- Alternating dense and locally banded sparse attention patterns in the layers
- Input sequence length: $n_{ctx} = 2048$ tokens
- Vocabulary size: 50,257 tokens (same as GPT-2)

> [!IMPORTANT]
> Unlike GPT-2, GPT-3 uses sparse attention patterns similar to the Sparse Transformer to improve efficiency at scale.

> [!NOTE]
> In my opinion, decoder-only needs time direction (like causal masking) but encoder-only like BERT does not need time direction.
>
> I'm not sure which is better overall including encoder-decoder models like T5.
>
> [どのトランスフォーマーアーキテクチャが最適ですか？エンコーダのみ vs エンコーダ・デコーダ vs デコーダのみモデル - YouTube](https://www.youtube.com/watch?v=wOcbALDw0bU)

The model processes input text autoregressively:
1. Input tokens $x_1, x_2, ..., x_n$ are embedded into vectors $\mathbf{e}_i \in \mathbb{R}^{d_{model}}$
2. Transformer layers compute contextualized representations
3. Output layer predicts next token probabilities: $P(x_{n+1} | x_1, ..., x_n)$

## Model Sizes

Eight model sizes were trained, spanning over three orders of magnitude:

| Model | $n_{params}$ | $n_{layers}$ | $d_{model}$ | $n_{heads}$ | $d_{head}$ | Batch Size |
|-------|--------------|--------------|-------------|-------------|------------|------------|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M tokens |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M tokens |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M tokens |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M tokens |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M tokens |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M tokens |
| GPT-3 13B | 13.0B | 40 | 5120 | 40 | 128 | 2M tokens |
| GPT-3 175B | 175.0B | 96 | 12288 | 96 | 128 | 3.2M tokens |

All models are trained for 300 billion tokens.

## Training Data

The training dataset is a mixture of five corpora, with sampling weights determined not by size but by quality estimates:

| Dataset | Tokens (billions) | Weight in Training Mix | Epochs |
|---------|-------------------|------------------------|--------|
| Common Crawl (filtered) | 410 | 60% | 0.44 |
| WebText2 | 19 | 22% | 2.9 |
| Books1 | 12 | 8% | 1.9 |
| Books2 | 55 | 8% | 0.43 |
| Wikipedia | 3 | 3% | 3.4 |

> [!IMPORTANT]
> Common Crawl was heavily filtered using quality-based classifiers trained to distinguish high-quality references (WebText, books, Wikipedia) from raw Common Crawl, reducing it from several trillion tokens to 410 billion.

The data was de-duplicated at the document level using Spark's MinHashLSH implementation with 10 hashes and a Jaccard similarity threshold of 0.87.

## In-Context Learning Evaluation Settings

The paper systematically evaluates three conditions:

1. **Few-Shot (FS)**: The model receives $K$ examples of the task $(context, completion)$ pairs, plus one final context for which it must generate the completion. Typically $K$ ranges from 10 to 100.

2. **One-Shot (1S)**: Exactly one example demonstration is provided along with the task description ($K=1$).

3. **Zero-Shot (0S)**: No examples are provided, only a natural language task description.

**Input format for few-shot**:
```
[Task description]
[Example 1 context] → [Example 1 completion]
[Example 2 context] → [Example 2 completion]
...
[Example K context] → [Example K completion]
[Test context] →
```

The model then generates the completion for the test context. All examples and the test input fit within the $n_{ctx} = 2048$ token context window.

> [!NOTE]
> The paper emphasizes that few-shot learning does not involve any gradient updates or fine-tuning. The model weights remain frozen, and task specification happens purely through conditioning on the input context.

# Results

## Language Modeling

Evaluated on Penn Tree Bank (PTB) test set:
- GPT-3 175B: **20.50 perplexity** (zero-shot)
- Previous SOTA (fine-tuned): 35.8 perplexity

Evaluated on LAMBADA (predicting the final word of sentences):
- GPT-3 175B: **76.2% accuracy** (zero-shot), **86.4% accuracy** (few-shot)
- Previous SOTA: 68.0% accuracy
- Human performance: ~95% accuracy

## Closed-Book Question Answering

Models answer questions without access to external documents, relying solely on knowledge in parameters.

**TriviaQA** (filtered subset):
- GPT-3 175B few-shot: **71.2% accuracy**
- Previous best closed-book: 65.1% (T5-11B fine-tuned)
- Open-book retrieval-augmented SOTA: 72.5%

**Natural Questions** (filtered):
- GPT-3 175B few-shot: **29.9% accuracy**
- Previous best closed-book: 36.6% (T5-11B fine-tuned)
- Open-book SOTA: 44.7%

**WebQuestions**:
- GPT-3 175B few-shot: **41.5% accuracy**
- Previous best closed-book: 37.4% (T5-11B fine-tuned)

> [!IMPORTANT]
> GPT-3 achieves these results without any task-specific fine-tuning, using only in-context examples. On TriviaQA, it matches the performance of retrieval-augmented systems despite having no external knowledge retrieval mechanism.

## Translation

Evaluated on WMT benchmarks for English↔French, German, and Romanian translation:

**En→Fr**:
- GPT-3 175B few-shot: **25.2 BLEU** (learns from 32 examples)
- Unsupervised NMT baseline: 33.3 BLEU
- Supervised SOTA: 45.6 BLEU

**Fr→En**:
- GPT-3 175B few-shot: **32.6 BLEU**
- Unsupervised NMT baseline: 33.4 BLEU
- Supervised SOTA: 45.9 BLEU

**En→De**:
- GPT-3 175B few-shot: **24.8 BLEU**
- Unsupervised NMT baseline: 31.2 BLEU
- Supervised SOTA: 42.1 BLEU

**De→En**:
- GPT-3 175B few-shot: **29.7 BLEU**
- Unsupervised NMT baseline: 35.0 BLEU
- Supervised SOTA: 43.1 BLEU

Translation performance shows a clear pattern: the model is stronger at translating into English (the dominant language in the training corpus) than from English into other languages.

## SuperGLUE Benchmark

SuperGLUE is a benchmark suite of eight NLU tasks designed to be difficult for modern NLP systems.

**GPT-3 175B few-shot performance**:
- Overall: **71.8 average** (vs. 69.0 for fine-tuned BERT++)
- Best task (ReCoRD): 89.8 F1
- Worst task (WiC): 51.4% accuracy (barely above random 50%)

**Notable results by task**:
- **COPA** (Commonsense reasoning): 92.0% accuracy (few-shot) vs. 91.9% human baseline
- **ReCoRD** (Reading comprehension): 89.8 F1 (few-shot) vs. 91.1 human
- **BoolQ** (Yes/No questions): 77.5% accuracy (few-shot) vs. 91.0 human
- **MultiRC**: 75.4 F1a (few-shot) vs. 86.4 human

**Challenging tasks**:
- **WiC** (Word sense disambiguation): 51.4% accuracy (near random)
- **RTE** (Textual entailment): 63.5% accuracy vs. 93.6 human

> [!IMPORTANT]
> GPT-3 shows large variance across SuperGLUE tasks, excelling at commonsense reasoning and reading comprehension but struggling with tasks requiring fine-grained textual entailment or word sense disambiguation.

## Reading Comprehension

**CoQA** (Conversational Question Answering):
- GPT-3 175B few-shot: **85.0 F1**
- Fine-tuned SOTA: 90.7 F1
- Human performance: 89.8 F1

**DROP** (Discrete reasoning over paragraphs):
- GPT-3 175B few-shot: **36.5 F1**
- Fine-tuned SOTA: 83.1 F1
- Human performance: 96.4 F1

**QuAC** (Question Answering in Context):
- GPT-3 175B few-shot: **44.3 F1**
- Fine-tuned SOTA: 73.7 F1
- Human performance: 80.8 F1

**SQuADv2**:
- GPT-3 175B few-shot: **69.8 F1**
- Fine-tuned SOTA: 90.7 F1
- Human performance: 86.9 F1

Performance on reading comprehension tasks is mixed: GPT-3 approaches human-level performance on CoQA but shows significant gaps on tasks requiring discrete reasoning (DROP) or handling null answers (SQuADv2).

## Winograd-Style Tasks

Coreference resolution tasks requiring commonsense reasoning:

**Winogrande**:
- GPT-3 175B few-shot: **84.6% accuracy**
- Fine-tuned SOTA (RoBERTa-large): 79.1%
- Human performance: 94.0%

**Winograd** (original):
- GPT-3 175B few-shot: **88.6% accuracy**
- Fine-tuned SOTA: 90.1%

GPT-3 exceeds fine-tuned models on Winogrande, demonstrating strong commonsense reasoning capabilities.

## Arithmetic

Tested on synthetically generated arithmetic problems:

**2-digit addition**:
- GPT-3 175B few-shot: **100.0% accuracy**

**3-digit addition**:
- GPT-3 175B few-shot: **80.2% accuracy**
- GPT-3 13B few-shot: 50.3%

**4-digit addition**:
- GPT-3 175B few-shot: **25.5% accuracy**
- GPT-3 13B few-shot: 8.7%

**5-digit addition**:
- GPT-3 175B few-shot: **9.3% accuracy**

**2-digit subtraction**:
- GPT-3 175B few-shot: **94.5% accuracy**

**2-digit multiplication**:
- GPT-3 175B few-shot: **29.2% accuracy**

> [!IMPORTANT]
> Arithmetic performance shows smooth degradation with problem difficulty. The model achieves perfect performance on 2-digit addition but performance drops significantly as the number of digits increases, suggesting it partially learns algorithmic patterns but fails to generalize fully.

## SAT Analogies

A collection of 374 analogy problems from SAT exams:
- GPT-3 175B few-shot: **65.2% accuracy**
- Average college applicant: ~57% accuracy
- GPT-3 175B zero-shot: 53.7% accuracy

The model exceeds average human performance on this task designed to test verbal reasoning.

## News Article Generation

Human evaluators were shown articles and asked to judge whether they were written by humans or AI:

**Detection accuracy by article type**:
- Human-written baseline: **~88% detection** (correctly identified as human)
- GPT-3 175B generated with title + subtitle context: **~52% detection** (near random guessing)
- Control (deliberately poor GPT-3 outputs): ~86% detection

> [!CAUTION]
> Human evaluators could not reliably distinguish GPT-3-generated news articles from human-written ones when given only titles and subtitles as context. This poses significant risks for misinformation and propaganda.

Mean human evaluator accuracy judging if articles were machine-generated was 52%, essentially random chance. This suggests GPT-3's text generation quality has reached a threshold where synthetic content is indistinguishable from human writing in short-form news contexts.

## Synthetic and Qualitative Tasks

**Novel word unscrambling**:
Given examples of anagrams with systematic character insertion patterns, GPT-3 learns the pattern:
- Random character insertion then removal: **67.2% accuracy** few-shot

**Correcting English grammar**:
- GPT-3 175B few-shot shows qualitative improvement in grammar correction over smaller models

**Using novel words in sentences**:
Given a definition, GPT-3 generates appropriate usage examples demonstrating semantic understanding.

# Measuring and Preventing Memorization

## Data Contamination Analysis

The paper systematically analyzes overlap between training data and evaluation benchmarks:

**Methodology**:
1. Generate $n$-gram overlap statistics between each benchmark's test set and training data
2. Use a Bloom filter for efficient membership testing
3. Report "clean" vs. "dirty" accuracy, where "dirty" examples have substantial $n$-gram overlap

**Findings**:
- Most benchmarks show minimal contamination effects (<1% difference)
- LAMBADA shows significant contamination: zero-shot accuracy drops from 76.2% to 72.5% on clean subset
- PIQA, ReCoRD, and DROP show moderate contamination effects
- Overall conclusion: contamination exists but does not invalidate results

> [!NOTE]
> The paper acknowledges that even this analysis may underestimate contamination, as it cannot detect paraphrased or semantically similar content.

## Training Loss Analysis

Training loss follows a smooth power law: $L(C) \propto C^{-\alpha}$ where $C$ is compute budget and $\alpha \approx 0.050$.

The relationship between model size $N$ and loss also follows a power law when training on fixed data: $L(N) \propto N^{-0.076}$.

> [!IMPORTANT]
> These scaling laws suggest that further increases in model scale will continue to improve performance, with no indication of plateauing within the tested range.

# Limitations

The paper honestly documents several significant limitations:

## Task-Specific Weaknesses

1. **Natural Language Inference**: Performance on ANLI rounds 2 and 3 remains near random chance (33-40% for 3-way classification)
2. **Fill-in-the-blank tasks**: Cloze tests like RACE-h show large gaps compared to fine-tuned models
3. **Comparison questions**: Tasks requiring discrete comparison or arithmetic reasoning over text (DROP) show poor performance
4. **Reading comprehension formats**: Some RC tasks like QuAC and RACE show substantial human-model gaps

## Structural Limitations

1. **Autoregressive pre-training bias**: The model fills in text from left to right, which may not suit bidirectional tasks
2. **Few-shot sample efficiency**: Despite improvements, GPT-3 still requires more examples than humans for many tasks
3. **Uncertain capabilities**: Many capabilities work well in some contexts but fail in others without clear patterns

## Text Synthesis Weaknesses

1. **Long-range coherence**: Repetition and loss of coherence over sufficiently long passages
2. **Semantic repetition**: The model may repeat the same idea using different phrasing
3. **Contradictions**: Generated text occasionally contains self-contradictory statements
4. **Non-sequiturs**: Sentences may not logically follow from preceding context

> [!NOTE]
> The paper emphasizes that these issues occur non-trivially but not on the majority of outputs, making them difficult to systematically study.

## Methodological Limitations

1. **Few-shot evaluation variance**: Performance is sensitive to the specific examples provided and their ordering
2. **Task description phrasing**: Results depend on exact wording of prompts
3. **Limited task coverage**: The paper cannot test all possible NLP tasks
4. **Compute requirements**: Training GPT-3 175B requires thousands of petaflop/s-days of compute

# Broader Impacts

## Potential Beneficial Uses

1. **Code generation and assistance**: Helping programmers write and understand code
2. **Creative writing**: Assisting with content generation while keeping humans in the loop
3. **Education**: Answering questions and explaining concepts
4. **Accessibility**: Helping people with writing difficulties communicate effectively

## Misuse Potential

1. **Misinformation**: Generating misleading news articles or social media content at scale
2. **Spam and phishing**: Producing more persuasive fraudulent content
3. **Academic dishonesty**: Students using the model to complete assignments
4. **Social manipulation**: Generating personalized propaganda

> [!IMPORTANT]
> The paper notes that GPT-3-generated news articles are largely indistinguishable from human-written ones (52% detection rate), raising significant concerns about misinformation campaigns.

## Fairness, Bias, and Representation

**Gender bias**:
- Occupational stereotypes: "He is a [occupation]" vs "She is a [occupation]" show stereotypical patterns
- Higher-status occupations more associated with male pronouns
- Over 83% association of "nurse" with female pronouns

**Race/ethnicity/religion bias**:
- Sentiment analysis shows more negative associations for Black, Asian, and Jewish descriptors
- Religious terms show varying sentiment patterns (Islam more negative, Christianity more positive)

**Methodological challenges**:
- Difficult to disentangle descriptor semantics from bias
- Training data reflects societal biases in web text
- Bias metrics themselves may embed assumptions

## Environmental Impact

Training GPT-3 175B consumed approximately:
- Energy equivalent: Not precisely reported
- CO₂ footprint: Estimated in hundreds of metric tons range (comparing to automotive emissions)

> [!NOTE]
> The paper acknowledges that large-scale model training has environmental costs but argues that once trained, a single model can be used by millions without additional training.

# Comparison with Prior Work

## Differences from GPT-2

1. **Scale**: GPT-3 175B has 116× more parameters than GPT-2 1.5B
2. **Training data**: 300B tokens vs ~40B tokens; updated dataset with better quality filtering
3. **Sparse attention**: GPT-3 uses alternating dense/sparse patterns; GPT-2 uses all-dense
4. **Evaluation focus**: Systematic few-shot evaluation vs. zero-shot only
5. **Performance gap**: GPT-3 shows qualitatively different few-shot learning capabilities

## Differences from T5/BERT

1. **Architecture**: Decoder-only autoregressive vs. encoder-decoder (T5) or bidirectional encoder (BERT)
2. **Fine-tuning**: GPT-3 avoids fine-tuning; T5/BERT require task-specific fine-tuning for best results
3. **Task specification**: Natural language prompts vs. task-specific heads
4. **Scale**: GPT-3 175B vs. T5-11B or BERT-large 340M

> [!IMPORTANT]
> On several benchmarks, GPT-3 few-shot matches or exceeds fine-tuned T5/BERT models despite no gradient updates, suggesting that sufficiently large models can learn tasks from demonstration alone.

## Differences from Meta-Learning Approaches

Traditional meta-learning (MAML, Prototypical Networks) learns from tasks with explicit meta-training:
1. **Explicit meta-training**: These methods train on multiple tasks to learn how to adapt
2. **Support sets**: Structured divisions of examples for adaptation
3. **Gradient-based adaptation**: Inner-loop optimization updates

GPT-3's in-context learning:
1. **Implicit meta-learning**: Emerges from language modeling pre-training without explicit task distributions
2. **Unstructured context**: Examples simply concatenated as text
3. **Forward-pass only**: No gradient computation at test time

The paper hypothesizes that broad pre-training on diverse text creates implicit meta-learning capabilities as the model learns to recognize and continue patterns within its context window.

# Related Work

## Scaling Laws Literature

The paper builds on prior work (Kaplan et al., 2020) establishing that language model performance scales smoothly as a power law with:
- Model parameters $N$: $L \propto N^{-\alpha_N}$
- Dataset size $D$: $L \propto D^{-\alpha_D}$
- Compute $C$: $L \propto C^{-\alpha_C}$

GPT-3's results extend these scaling laws to much larger model sizes and demonstrate that task-specific performance improvements also follow scaling trends.

## In-Context Learning and Prompting

Prior work showed language models could adapt to tasks through careful prompting, but GPT-3 demonstrates this at unprecedented scale:
- **GPT-1**: Basic task transfer through pre-training + fine-tuning
- **GPT-2**: Zero-shot task transfer through prompting
- **GPT-3**: Strong few-shot learning from demonstrations

## Large-Scale Language Models

Contemporary work on large language models:
- **T5** (11B parameters): Unified text-to-text framework with fine-tuning
- **Megatron-LM** (8.3B): Efficient training of large transformers
- **Turing-NLG** (17B): Large-scale model for generation

GPT-3 advances this lineage by demonstrating that scale enables task learning without fine-tuning.

# Experiments

- **Datasets**: Penn Tree Bank, LAMBADA, HellaSwag, StoryCloze, TriviaQA, Natural Questions, WebQuestions, WMT14/16 (Fr↔En, De↔En, Ro↔En), SuperGLUE (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC), CoQA, DROP, QuAC, SQuADv2, RACE-m, RACE-h, ANLI, Winograd, Winogrande, ARC, OpenBookQA, SAT Analogies, synthetic arithmetic tasks, synthetic news article generation (titles from newser.com)

- **Hardware**: Training used V100 GPUs in Microsoft Azure's high-bandwidth clusters. The 175B parameter model required special infrastructure for model parallelism.

- **Optimizer**: Adam optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$. Gradient clipping at global norm 1.0. Learning rate warmup over first 375 million tokens, then cosine decay to 10% of max learning rate.

- **Training details**: Batch sizes vary by model from 0.5M to 3.2M tokens. All models trained for 300 billion tokens (approximately one epoch over the weighted training mixture).

- **Results**: GPT-3 175B achieves competitive or state-of-the-art performance on numerous benchmarks in few-shot settings without fine-tuning:
  - Language modeling: 20.5 perplexity on PTB (SOTA)
  - Question answering: 71.2% on TriviaQA (matches retrieval-augmented systems)
  - Reading comprehension: 85.0 F1 on CoQA (approaches human performance)
  - Commonsense reasoning: 84.6% on Winogrande (exceeds fine-tuned SOTA)
  - Text generation: 52% human detection rate (indistinguishable from human writing)

# Conclusion

GPT-3 demonstrates that scaling language models to 175 billion parameters enables strong task-agnostic performance through in-context learning. The model achieves competitive results with fine-tuned approaches across diverse NLP benchmarks while requiring only task descriptions and demonstrations—no gradient updates. Key findings include:

1. **Smooth scaling**: Performance improvements follow power laws across three orders of magnitude in model size
2. **Few-shot learning emergence**: Larger models show qualitatively better in-context learning abilities
3. **Task breadth**: Competitive performance on language modeling, question answering, translation, reasoning, and synthesis
4. **Limitations remain**: Natural language inference, discrete reasoning, and long-range coherence remain challenging

The work suggests that continued scaling of language models may further improve capabilities, though significant limitations and societal implications require careful consideration.

> [!TIP]
> For an accessible introduction to GPT-3 and its implications, see Jay Alammar's visual guide: [How GPT-3 Works](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)

> [!IMPORTANT]
> This paper represents a paradigm shift in NLP: rather than adapting models to tasks through fine-tuning, sufficiently large models can adapt to tasks through conditioning on textual demonstrations alone. This suggests pre-trained models develop meta-learning capabilities during unsupervised training.
