# Meta Information

- URL: [[2312.00752] Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- LICENSE: [Deed - Attribution 4.0 International - Creative Commons](https://creativecommons.org/licenses/by/4.0/)

# 1. Introduction

There is a foundation model which is called Transformer. However, it has quadratic complexity with respect to sequence length.

In 2022, structured state space sequence models (S4) have been proposed, which is combination of RNNs and CNNs, and it can handle long-range dependencies in sequences with linear complexity.

Here, the authors propose a new class of selective state space models.

## a. Selection Mechanism

- The critical weakness of prior S4-based models is that they use select the information whether is relevant or irrevant.
- By parametrizing the S4 parametrizing based on the input, the authors propose a selection mechanism to select the relevant information.

## b. Hardware-aware Algorithm

- The essential constraint of prior all the models parameter is fixed (invariant) regardless of time and input to make computation efficient.
- The authors propose scan instead of convolution to make the model more hardware-efficient.

## c. Architecture


