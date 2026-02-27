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



# 第3章　エントロピー正則化とシンクホーンアルゴリズム

# 第4章　敵対的ネットワーク

# 第5章　スライス法

# 第6章　他のダイバージェンスとの比較

# 第7章　不均衡最適輸送

# 第8章　ワッサースタイン重心

# 第9章　グロモフ・ワッサースタイン距離

# 第10章　おわりに
