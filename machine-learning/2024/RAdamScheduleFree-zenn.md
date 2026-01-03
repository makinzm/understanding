# Meta Information

- URL: [全ての学習率スケジューリングを過去にするOptimizer](https://zenn.dev/dena/articles/6f04641801b387)
- Copyright: [利用規約 | Zenn](https://zenn.dev/terms)の第6条
- Author: @nhamanasu

# How to use

https://github.com/facebookresearch/schedule_free

# Problem before ScheduleFree

We have to carefully tune learning rate schedules and optimizer hyperparameters for each ml task.

# What is ScheduleFree?

We have to catch up some words before explaining ScheduleFree.

- Momentum: Like inertia（慣性）, it helps smooth out the updates and can accelerate convergence.
    - $w_{t+1} = w_t - \eta \cdot v_t$
    - $v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \frac{\partial L}{\partial w_t}$
    - $v_0 = 0$
    - Ref: [12.6. Momentum — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_optimization/momentum.html)
- Polyak Averaging: It averages the weights over time to reduce variance and improve generalization.
    - $\bar{w}_t = \frac{1}{t} \sum_{i=1}^{t} w_i$
    - Ref: [[1709.03342] Optimal non-asymptotic bound of the Ruppert-Polyak averaging without strong convexity](https://arxiv.org/abs/1709.03342)
- Primal averaging: Gradient is computed based on the averaged weights.
    - $g_t = \frac{\partial L}{\partial \bar{w}_t}$
    - Ref: [[2010.00406] Momentum via Primal Averaging: Theoretical Insights and Learning Rate Schedules for Non-Convex Optimization](https://arxiv.org/abs/2010.00406)

[[2405.15682] The Road Less Scheduled](https://arxiv.org/abs/2405.15682)

ScheduleFree is based on Momentum and Primal Averaging.
$x$ is the model parameter, $y$ is the primal averaged weight, and $z$ is the momentum weight.

## Schedule Free SGD

- Let $\zeta_t$ be the random variable sampled from dataset at step t.
- $x_{t+1} = (1 - c_t) x_t + c_{t+1} z_{t+1}$
- $z_{t+1} = z_t - \eta_t \frac{\partial L(x_t; \zeta_{t+1})}{\partial y_t}$
- $y_{t+1} = (1 - \beta) z_{t} + \beta x_t$

# Why ScheduleFree works well?

[[2405.15682] The Road Less Scheduled](https://arxiv.org/abs/2405.15682) Theorem 1 shows the convergence guarantee of ScheduleFree because of G-Lipschitz of loss function.

> [!IMPORTANT]

> G-Lipschitz: $\| f(x) - f(y) \| \leq G \| x - y \|$ for all $x, y$ in the domain of $f$.

> [Lipschitz continuity - Wikipedia](https://en.wikipedia.org/wiki/Lipschitz_continuity)


# Schedule Free RAdam

We can also apply the ScheduleFree framework to RAdam optimizer.

- https://zenn.dev/dena/articles/6f04641801b387#%E3%81%9D%E3%81%97%E3%81%A6-radamschedulefree-%E3%81%B8

# Experiments

## In the blog

There is no experiment about ScheduleFree RAdam in the blog.

## In the paper

[[2405.15682] The Road Less Scheduled](https://arxiv.org/abs/2405.15682)

### Deep Learning Experiments

- Datasets & Architectures
    - CIFAR-10, Wide ResNet-16-8
    - CIFAR-100, DenseNet
    - SVHN(Street View House Numbers), deep ResNet
    - ImageNet, ResNet-50
    - IWSLT14(ge-en), LSTM
    - Criteo Kaggle Display Ads, DLRM(Deep Learning Recommendation Model)
    - MRI, U-Net
    - ILSVRC2012 ImageNet, Masked Autoencoder ViT
    - OpenWebText, GPT-2
- Optimizers
    - ScheduleFree SGD
    - ScheduleFree AdamW
- Results
    - ScheduleFree shows very competitive performance.
    - Learning rate tends to be larger than the conventional optimizers.

### Machine Learning Experiments

- Datasets & Models
    - WMT17(ge-en), Encoder-Decoder Transformer
    - ILSVRC2012 ImageNet, ViT
    - Knee MRI Dataset, U-Net
    - LibriSpeech ASR, Conformer
    - Open Graph Benchmark, Graph Neural Network
    - Crickthrough Prediction, DLRM
    - LibriSpeech ASR dataset, Deep Speech Model
- Optimizers
    - ScheduleFree AdamW
    - NAdamW
- Results
    - ScheduleFree AdamW outperforms the baseline.

### Others

- Schedule Free outperforms conventional linear decay approach.
- Memory requirement is almost the same as conventional optimizers.

