# Meta Information

- URL: [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., & Ba, J. (2022). Large Language Models Are Human-Level Prompt Engineers. arXiv preprint arXiv:2211.01910.

# Large Language Models Are Human-Level Prompt Engineers

Automatic Prompt Engineer (APE) は、人手によるプロンプト設計を不要にする自動命令探索フレームワークである。LLM を使って命令候補を生成し、スコア関数で評価・選別することで、人間が設計したプロンプトと同等以上の性能を達成する。**対象ユーザー**: タスク固有の命令設計に時間を要する ML 実務者や研究者。**適用場面**: 少数のデモ (入出力例) が用意できるあらゆる NLP タスク。

## Problem Formulation

命令探索を自然言語プログラム合成として定式化する。求めたい命令 $\rho$ は次の最適化問題の解である:

```math
\begin{align}
  \rho^{\star} = \arg\max_{\rho} \mathbb{E}_{(Q,A) \sim \mathcal{D}}\bigl[f(\rho, Q, A)\bigr]
\end{align}
```

- $\mathcal{D}$: デモンストレーション集合 (入力 $Q$、正解 $A$ のペア)
- $f(\rho, Q, A)$: 命令の品質を測るスコア関数 (後述)

この最適化は解析的に解けないため、LLM を利用したサンプリングベースの探索を行う。

## APE Algorithm

**入力**: デモンストレーション集合 $\mathcal{D}$、提案モデル $p_{\text{prop}}$、スコアモデル $p_{\text{score}}$、候補数 $m$、保持率 $k$
**出力**: 最良命令 $\rho^{\star}$

```
APE(D, p_prop, p_score, m, k):
  1. meta_prompt ← construct_meta_prompt(D)          // デモを埋め込んだメタプロンプトを作成
  2. C ← sample m instructions from p_prop(· | meta_prompt) // 命令候補を m 個サンプリング
  3. C_filtered ← top k% of C ranked by f on D_small  // 小サブセットで効率的に絞り込む
  4. (optional) C_refined ← MC_resample(C_filtered, p_prop) // 上位候補周辺を再探索
  5. ρ* ← arg max_{ρ ∈ C_filtered ∪ C_refined} f(ρ, Q, A) on full D
  6. return ρ*
```

### Step 1: Instruction Proposal

2 種類のモードで LLM から命令候補を生成する。

| モード | 入力プロンプト形式 | 適用モデル例 |
|--------|-------------------|--------------|
| Forward | `[demos] Instruction: ___` → LLM が末尾を補完 | InstructGPT, GPT-3 |
| Reverse (infilling) | `I: ___ [demos]` → LLM が空白を埋める | T5, GLM |

Forward モードでは、デモを `Input: Q` `Output: A` 形式で並べたあと `Instruction: ` と付記し、LLM に後続を予測させる。

### Step 2: Score Functions

| スコア関数 | 定義 | 特性 |
|------------|------|------|
| 実行精度 (Execution Accuracy) | $f = \mathbb{1}[\hat{A} = A]$ | 離散的・解釈しやすい |
| 対数確率 (Log Probability) | $f = \log p_{\text{score}}(A \mid \rho, Q)$ | 連続的・微分可能 |

実験結果として、実行精度の方がテスト性能との相関が高いことが確認されている。

> [!NOTE]
> 効率化のため、まず小サブセット (例: 全デモの 20%) で全候補を評価し、上位 $k\%$ のみをフルデータセットで再評価する多段階フィルタリングを採用している。

### Step 3: Iterative Refinement (Optional)

上位候補 $\rho_{\text{top}}$ を "アンカー" として、メタプロンプトに埋め込み再サンプリングするモンテカルロ探索。意味的に類似した変形を生成することで局所探索を実現する。実験的には限界的な改善しか得られなかった。

## Comparison with Related Methods

| 手法 | 命令の生成方法 | 探索空間 | 勾配の要否 |
|------|--------------|----------|-----------|
| Manual Prompt Engineering | 人手 | 離散 | 不要 |
| Soft Prompt Tuning | 勾配降下 | 連続埋め込み | 必要 |
| **APE (本手法)** | LLM サンプリング | 離散 (自然言語) | 不要 |
| AutoPrompt | 勾配ベーストークン探索 | 離散 | 必要 |

APE の最大の差別化点は「**解釈可能な自然言語命令**を**勾配なし**で最適化できる」点にある。Soft Prompt Tuning は高精度だが人間が読めない埋め込みを最適化するのに対し、APE は LLM の生成能力を直接活用して可読性のある命令を生成する。

## Experiments

- **Dataset (Instruction Induction)**: Honovich et al. (2022) の 24 タスク。各タスクに少数 (約 5〜10 件) のデモを使用。
- **Dataset (BIG-Bench Instruction Induction)**: 21 タスク。複雑な推論・言語理解を含む。
- **Dataset (Zero-Shot Chain-of-Thought)**: MultiArith (600 問) と GSM8K (1,319 問の算術文章題)。
- **Dataset (TruthfulQA)**: 817 問の真実性評価ベンチマーク (MC1/MC2)。
- **Scoring Model**: InstructGPT (`text-davinci-002`)
- **Proposal Model**: InstructGPT (`text-davinci-002`)、一部実験で T5・GLM も比較

### Key Results

| ベンチマーク | Baseline (human prompt) | APE |
|-------------|------------------------|-----|
| Instruction Induction (IQM) | 0.749 | **0.810** |
| BIG-Bench (# tasks won) | — | 17/21 |
| MultiArith (ZS-CoT) | 78.7% | **82.0%** |
| GSM8K (ZS-CoT) | 40.7% | **43.0%** |

> [!IMPORTANT]
> APE が発見した Zero-Shot CoT プロンプト "Let's work this out in a step by step way to be sure we have the right answer" は、人間が設計した "Let's think step by step" を数値的に上回った。ただしこの結果は同一分布のデモ上で最適化されており、厳密な zero-shot 評価ではない点に注意。

### TruthfulQA の知見

APE は真実性 (truthfulness) と情報量 (informativeness) のトレードオフを Pareto 最適化できることを示した。単一スコア (true & informative の積) を最大化した場合、ベースラインの約 30% から 40% 以上に改善した。

## Practical Considerations

- **Context 圧縮**: APE 命令は in-context learning に必要なデモの代替になり、プロンプトトークン数を最大 10 倍削減できる。
- **モデル規模**: 提案モデルが大きいほど候補品質が高く、$m = 64$ 程度でほぼ収束する。
- **命令の転移性**: InstructGPT 用に生成した命令は GPT-3 ではほとんど機能しない。提案モデルとスコアリングモデルのアライメントが重要。

> [!TIP]
> 公式実装: [https://github.com/keirp/automatic_prompt_engineer](https://github.com/keirp/automatic_prompt_engineer)
