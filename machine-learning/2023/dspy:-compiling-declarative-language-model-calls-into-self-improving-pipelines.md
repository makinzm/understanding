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

# 5. Goals of Evaluation

# 6. Case Study: Math Word Problems

# 7. Case Study: Complex Question Answering

# 8. Conclusion

