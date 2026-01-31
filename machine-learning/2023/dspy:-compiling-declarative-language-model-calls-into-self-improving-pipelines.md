# Meta Information

- [ [2310.03714] DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines ]( https://arxiv.org/abs/2310.03714 )
- LICENSE: [ arXiv.org - Non-exclusive license to distribute ]( https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html )

---

# 1. Introduction

LMs (Language Models) performance depends on prompt quality for each task.

Prompt tuning is implemented interactively widely but is inefficient and not reusable.

More systematic approaches are needed to optimize prompts and compose LM calls, so we propose DSPy.

- DSPy programming model: A framework that replaces manual prompt engineering with modular, declarative components that can be composed like neural network layers to build LM pipelines.
- DSPy compiler: An optimizer that automatically generates effective prompts and fine-tuning strategies for DSPy modules by bootstrapping examples from training data and a validation metric.

# 2. Related Work

DSPy draws inspiration from deep learning framework like Torch and Chainer.

In-context learning has enabled sophisticated LM behavior through prompting techniques like Chain of Thought and tool usage including retrieval models and APIs. Existing toolkits like LangChain, Semantic Kernel, and LlamaIndex facilitate this but rely on hand-written prompt templates, which DSPy aims to replace.

Recent work applies discrete optimization and RL to find effective prompts, typically for single LM calls. DSPy generalizes this by optimizing arbitrary multi-stage pipelines through bootstrapping demonstrations from high-level declarative signatures, using techniques like cross-validation or potentially RL and Bayesian optimization.

This paper demonstrates that DSPy enables building strong LM systems from modular components without hand-crafted prompts, systematically exploring the design space at a high level of abstraction.

# 3. The DSPy Programming Model

DSPy provides three main abstractions:

1. Signatures: input-output behavior.
1. Modules: hand-prompting techniques and arbitrary pipelines.
1. Teleprompters: optimization to maximize a metric.

## 3.1. Natural Language Signatures can abstract prompting & finetuning

"A DSPy signature is natural-language typed declaration of a function" like `question -> answer`.

It specifies what not how and can be automatically compiled into optimized prompts or fine-tuned models.

## 3.2. Parameterized & Templated Modules abstract prompting techniques

Modules are reusable components that implement a DSPy signature.

They can be composed like functions and automatically learn their behavior (prompts/demonstrations) through parameterization, replacing hand-written prompting techniques.

Example: `ChainOfThought`, `Predict`, `ReAct` are all interchangeable modules that work with any signature.

# 4. The DSPy Compiler

The DSPy compiler automatically optimizes programs through a teleprompter (optimizer) in three stages:

Stage 1: Candidate Generation
- Run the program on training data, filter examples that pass the metric, and use successful traces as demonstration candidates for each module.

Stage 2: Parameter Optimization
- Select the best demonstrations (or instructions) using hyperparameter tuning methods like random search or Optuna, or finetune the LM using the bootstrapped demonstrations.

Stage 3: Higher-Order Program Optimization
- Modify the program structure itself (e.g., create ensembles that run multiple versions in parallel and combine their outputs).

In essence: The compiler automatically finds good examples by running your program, then uses those examples to improve the program's prompts or finetune models.

# 5. Goals of Evaluation

1. Free hand crafted prompts
1. Parameterizing the modules
1. Find complex pipelines automatically

# 6. Case Study: Math Word Problems

Dataset: GSM8K grade school math questions

- Training: 200 examples
- Development: 300 examples  
- Test: 1.3k examples

Three DSPy Programs Evaluated:

1. vanilla = `dspy.Predict("question -> answer")` - one-step prediction
2. CoT = `dspy.ChainOfThought("question -> answer")` - two-step reasoning
3. reflection = ThoughtReflection module - generates 5 reasoning chains, compares them, produces final answer

Compilation Strategies:

1. none - zero-shot (no compilation)
2. fewshot - random k=8 demonstrations from training
3. bootstrap - auto-generate demonstrations + optimize selection via random search
4. bootstrap×2 - use bootstrapped program as teacher to bootstrap again
5. ensemble - combine top-7 bootstrapped programs with majority voting

Key Results:

- Bootstrap compilation dramatically improves all programs across both GPT-3.5 and Llama2-13b
- Without human reasoning chains, DSPy bootstrap matches or exceeds performance with expert hand-crafted CoT prompts
- reflection program performs best overall
- Accuracy improved from 4-20% (zero-shot) to 49-88% (compiled) by composing generic modules rather than hand-crafting prompts
- Llama2-13b with DSPy competitive with Llama2-34b using manual prompts

DSPy's optimization approach can match or outperform expert-written prompts without requiring hand-crafted reasoning chains.

# 7. Case Study: Complex Question Answering

# 8. Conclusion

