# Reference

[最適輸送の理論とアルゴリズム | 書籍情報 | 株式会社 講談社サイエンティフィク](https://www.kspub.co.jp/book/detail/5305140.html)

---

# 第1章　確率分布を比較するツールとしての最適輸送

Optimal Transport Cost is also called Earth Mover's Distance.

KL Divergence cannot meet axioms of a metric, but Wasserstein Distance, which is one of the optimal transport cost, can meet axioms of a metric. 

KL Divergence diverges to infinity when the support of one distribution is not contained in the support of the other distribution.

# 第2章　最適化問題としての定式化

Kantorovich Formulation of Optimal Transport Problem:

```math
\begin{aligned}
\text{OT}(\mu, \nu, C) := \text{minimize}_{\pi \in \mathcal{P}(\mathcal{X} \times \mathcal{Y})} \quad & \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) d\pi(x, y) \\
\text{subject to} \quad & \pi(A \times \mathcal{Y}) = \mu(A) \quad \forall A \subseteq \mathcal{F}(\mathcal{X}) \\
& \pi(\mathcal{X} \times B) = \nu(B) \quad \forall B \subseteq \mathcal{F}(\mathcal{Y})
\end{aligned}
```

Here $\mathcal{F}$ is a $\sigma$-algebra on $\mathcal{X}$ and $\mathcal{Y}$, and $\mathcal{P}(\mathcal{X} \times \mathcal{Y})$ is the set of all probability measures on $\mathcal{X} \times \mathcal{Y}$.


> [!NOTE]
> 教科書内では、移動先と移動元の集合が同じ$\mathcal{Y} = \mathcal{X}$ としているが、他の資料などは $\mathcal{Y} \neq \mathcal{X}$ としていることもある。
>
> 教科書内でモンジュの定式化も紹介されていたが、サポートが同じではないと定義できない制約があることもあり、カントロビッチの定式化が一般的に用いられていると紹介されていた。

Wasserstein Distance is defined as follows:

```math
W_p(\mu, \nu) := \left( \text{OT}(\mu, \nu, d^p) \right)^{\frac{1}{p}}
```

Here $d$ is a metric on $\mathcal{X}$ and $\mathcal{Y}$, and $p \geq 1$.

- [Wasserstein metric - Wikipedia](https://en.wikipedia.org/wiki/Wasserstein_metric)

Dual Problem

```math
\begin{aligned}
\text{OT}(\mu, \nu, C) = \text{maximize}_{f \in L^1(\mu), g \in L^1(\nu)} \quad & \int_{\mathcal{X}} f(x) d\mu(x) + \int_{\mathcal{Y}} g(y) d\nu(y) \\
\text{subject to} \quad & f(x) + g(y) \leq c(x, y) \quad \forall (x, y) \in \mathcal{X} \times \mathcal{Y}
\end{aligned}
```

> [!NOTE]
> 教科書にも記載があるが、$f$と$g$は、符号制約がないため、$-$をつけられたものも同じ問題を解いていることに注意する。

c-transform is a way to find the optimal $g$ given $f$:

```math
f^c(y) := \inf_{x \in \mathcal{X}} c(x, y) - f(x)
```

The problem of Optimal Transport is related to minimum cost flow problem in combinatorial optimization.


# 第3章　エントロピー正則化とシンクホーンアルゴリズム

Entropy Regularization of Optimal Transport Problem:

```math
\begin{aligned}
\text{OT}_\varepsilon(\mu, \nu, C) := \text{minimize}_{\pi \in \mathcal{P}(\mathcal{X} \times \mathcal{Y})} \quad & \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) d\pi(x, y) - \varepsilon H(\pi) \\
\text{subject to} \quad & \pi(A \times \mathcal{Y}) = \mu(A) \quad \forall A \subseteq \mathcal{F}(\mathcal{X}) \\
& \pi(\mathcal{X} \times B) = \nu(B) \quad \forall B \subseteq \mathcal{F}(\mathcal{Y}) \\
& H(\pi) := -\int_{\mathcal{X} \times \mathcal{Y}} \log \pi(x, y) d\pi(x, y)
\end{aligned}
```

This problem can be written by the independent $\tilde{P} = \mu \otimes \nu$ as follows:

```math
\begin{aligned}
\text{OT}_\varepsilon(\mu, \nu, C) = \text{minimize}_{\pi \in \mathcal{P}(\mathcal{X} \times \mathcal{Y})} \quad & \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) d\pi(x, y) + \varepsilon \text{KL}(\pi \| \tilde{P}) \\
\text{subject to} \quad & \pi(A \times \mathcal{Y}) = \mu(A) \quad \forall A \subseteq \mathcal{F}(\mathcal{X}) \\
& \pi(\mathcal{X} \times B) = \nu(B) \quad \forall B \subseteq \mathcal{F}(\mathcal{Y})
\end{aligned}
```

```math
\begin{aligned}
\otimes \text{ is the product measure, and } \quad & \tilde{P}(A \times B) = \mu(A) \nu(B) \quad \forall A \subseteq \mathcal{F}(\mathcal{X}), B \subseteq \mathcal{F}(\mathcal{Y})
\end{aligned}
```

OT with Entropy Regularization is a convex optimization problem, and its dual problem is as follows:

```math
\begin{aligned}
\text{OT}_\varepsilon(\mu, \nu, C) = \text{maximize}_{f \in L^1(\mu), g \in L^1(\nu)} \quad & \int_{\mathcal{X}} f(x) d\mu(x) + \int_{\mathcal{Y}} g(y) d\nu(y) - \varepsilon \int_{\mathcal{X} \times \mathcal{Y}} e^{\frac{f(x) + g(y) - c(x, y)}{\varepsilon}} d\tilde{P}(x, y)
\end{aligned}
```

This problem can be solved by Sinkhorn Algorithm, which is an iterative algorithm that updates $f$ and $g$ as follows:

```math
\begin{aligned}
f^{(t+1)}(x) & = \varepsilon \log \frac{d \mu}{d x} - \varepsilon \log \int_{\mathcal{Y}} e^{\frac{g^{(t)}(y) - c(x, y)}{\varepsilon}} d\nu(y) \\
g^{(t+1)}(y) & = \varepsilon \log \frac{d \nu}{d y} - \varepsilon \log \int_{\mathcal{X}} e^{\frac{f^{(t+1)}(x) - c(x, y)}{\varepsilon}} d\mu(x)
\end{aligned}
```

This algorithm can be written by $u = e^{\frac{f}{\varepsilon}}$, $v = e^{\frac{g}{\varepsilon}}$ and $K(x, y) = e^{-\frac{c(x, y)}{\varepsilon}}$ as follows:

```math
\begin{aligned}
u^{(t+1)}(x) & = \frac{d \mu}{d x} \left( \int_{\mathcal{Y}} K(x, y) v^{(t)}(y) d\nu(y) \right)^{-1} \\
v^{(t+1)}(y) & = \frac{d \nu}{d y} \left( \int_{\mathcal{X}} K(x, y) u^{(t+1)}(x) d\mu(x) \right)^{-1}
\end{aligned}
```

So it is calculated by matrix-vector multiplication, and it can be implemented by GPU.

Sinkhorn Algorithm converges to the optimal solution of OT with Entropy Regularization, and it has a linear convergence rate, which is proven by the Hilbert's projective metric.

OT with Entropy Regularization is differentiable with respect to $\mu$, $\nu$ and $C$, and its gradient can be calculated by the optimal $f$ and $g$, which is proven by Danskin's Theorem and Lagrangian relaxation.

However, OT with Entropy Regularization does not converge to the optimal solution of OT as $\varepsilon \to 0$, and it has a bias that is proportional to $\varepsilon$ and does not meet the axioms of a metric. So, synchorn divergence is proposed as a way to remove the bias of OT with Entropy Regularization, which is defined as follows:

```math
\text{SD}_\varepsilon(\mu, \nu, C) := \text{OT}_\varepsilon(\mu, \nu, C) - \frac{1}{2} \text{OT}_\varepsilon(\mu, \mu, C) - \frac{1}{2} \text{OT}_\varepsilon(\nu, \nu, C)
```

# 第4章　敵対的ネットワーク

GAN is a framework for training generative models, which consists of a generator $g$ and a discriminator $d$, which are trained by the following min-max problem:

```math
\begin{aligned}\text{minimize}_g \text{maximize}_d \quad & \mathbb{E}_{x \sim p_{\text{data}}} [\log d(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - d(g(z)))]\end{aligned}
```

However, this problem is not easy to optimize because of vanishing gradients, when the discriminator is too good, and mode collapse, when the generator is too good.

Wasserstein GAN is a framework for training generative models, which consists of a generator $g$ and a discriminator $d$, which are trained by the following min-max problem:

```math
\begin{aligned}
\text{minimize}_g \text{maximize}_d \quad & \mathbb{E}_{x \sim p_{\text{data}}} [d(x)] - \mathbb{E}_{z \sim p_z} [d(g(z))] \\
\text{subject to} \quad & d \text{ is 1-Lipschitz}
\end{aligned}
```

# 第5章　スライス法

There is a method to calculate OT by slicing the distributions into one-dimensional distributions, which is called Sliced Optimal Transport, and it can be calculated by the closed-form solution of OT for one-dimensional distributions.

[最適輸送が遅すぎる（スライス法による解法） - Speaker Deck](https://speakerdeck.com/joisino/zui-shi-shu-song-gachi-sugiru-suraisufa-niyorujie-fa)

# 第6章　他のダイバージェンスとの比較

1. [Divergence (statistics) - Wikipedia](https://en.wikipedia.org/wiki/Divergence_%28statistics%29)
1. [Integral probability metric - Wikipedia](https://en.wikipedia.org/wiki/Integral_probability_metric)


# 第7章　不均衡最適輸送

Unbaalanced Optimal Transport is a framework for comparing unnormalized measures, which is defined as follows by $\mathcal{F}$ and $\mathcal{G}$, which are convex functions that measure the divergence from $\mu$ and $\nu$ to the set of probability measures, respectively:

```math
\begin{aligned}
\text{UOT}(\mu, \nu, C) := \text{minimize}_{\pi \in \mathcal{M}(\mathcal{X} \times \mathcal{Y})} \quad & \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) d\pi(x, y) + \mathcal{F}(\pi(\cdot \times \mathcal{Y})) + \mathcal{G}(\pi(\mathcal{X} \times \cdot)) \\
\text{subject to} \quad & \pi \in \mathcal{M}(\mathcal{X} \times \mathcal{Y})
\end{aligned}
```

This framework is useful for dealing with outliers and for comparing distributions with different total masses.

There are a lot of $\mathcal{F}$ and $\mathcal{G}$, such as KL Divergence, Total Variation Distance, and so on.

- [[1607.05816] Scaling Algorithms for Unbalanced Transport Problems](https://arxiv.org/abs/1607.05816)

# 第8章　ワッサースタイン重心

# 第9章　グロモフ・ワッサースタイン距離

# 第10章　おわりに
