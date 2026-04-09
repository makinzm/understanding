# Meta Information

- URL: [Synthesizer: Rethinking Self-Attention for Transformer Models](https://arxiv.org/abs/2005.00743)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, Che Zheng (2021). Synthesizer: Rethinking Self-Attention for Transformer Models. ICML 2021.

# Overview

Synthesizer は、Transformer モデルのドット積セルフアテンションの必要性を根本から問い直した論文である。著者らは「トークン間のペアワイズ相互作用なしに合成注意重みを学習する」モデルを提案し、ランダムに初期化されたアテンション行列でも競争力のある性能を発揮することを示した。

## 背景：標準的なドット積セルフアテンション

標準的な Multi-Head Attention では、入力 $X \in \mathbb{R}^{n \times d}$ に対して以下の計算を行う：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

ここで $W_Q, W_K \in \mathbb{R}^{d \times d_k}$、$W_V \in \mathbb{R}^{d \times d_v}$ である。このアプローチはトークン $i$ とトークン $j$ の間のスコアを $q_i \cdot k_j$ として計算するため、全トークンペアの相互作用を陽に計算する。

本論文の主要な問い：「トークン間のドット積相互作用は本当に必要なのか？」

# Synthesizer の各バリアント

## Dense Synthesizer

入力トークンの埋め込みのみから注意行列を合成する。ペアワイズ相互作用を一切使用しない。

**入力**: 各トークン $x_{i} \in \mathbb{R}^{d}$（$i = 1, \ldots, n$）

**計算式**:

$$B_{i} = W_2 \cdot \text{ReLU}(W_1 \cdot x_i)$$

- $W_1 \in \mathbb{R}^{d \times d}$：第1層の重み行列
- $W_2 \in \mathbb{R}^{d \times n}$：第2層の重み行列（出力次元が系列長 $n$）
- $B_i \in \mathbb{R}^{n}$：トークン $i$ から生成されるアテンションスコアベクトル

全トークンをスタックすると $B \in \mathbb{R}^{n \times n}$ を形成する。

**出力計算**:

$$Y = \text{softmax}(B) \cdot G(X)$$

- $G(X) = XW_V \in \mathbb{R}^{n \times d_v}$：Value の線形変換
- $Y \in \mathbb{R}^{n \times d_v}$：出力行列

> [!NOTE]
> Dense Synthesizer はトークン $i$ のアテンションスコアを「そのトークン自身の埋め込みだけ」から生成する。トークン $j$ の情報はスコア計算に入らない（Value として間接的にのみ寄与）。

## Random Synthesizer

アテンション行列を固定またはランダム初期化された学習可能パラメータとして扱う。入力には一切依存しない。

**定義**: $R \in \mathbb{R}^{n \times n}$ を学習可能なパラメータ行列とする（各ヘッドで独立）。

**出力計算**:

$$Y = \text{softmax}(R) \cdot G(X)$$

- $R$ は入力 $X$ から独立して学習される
- パラメータ数は $n^2$（系列長の二乗、ヘッドごと）

**Fixed Random Synthesizer**: $R$ をランダム初期化して固定（学習しない）。これは実質的にランダムな混合係数で Value を集約するだけである。

> [!IMPORTANT]
> Fixed Random Synthesizer が WMT'14 EnDe で 27.27 BLEU を達成した（ベースラインの 27.67 に対して）。これはトークン間相互作用なしでも多くのタスクで競争力ある性能が出ることを示す驚くべき結果である。

## Factorized バリアント

### Factorized Dense Synthesizer

Dense の行列 $B \in \mathbb{R}^{n \times n}$ を低ランク分解で近似する。

$$A_i = F_A(x_i) \in \mathbb{R}^{k}, \quad B_i = F_B(x_i) \in \mathbb{R}^{k}$$

$$C = H_A(A) \odot H_B(B) \in \mathbb{R}^{n \times n}$$

- $k \ll n$：ランク（通常 $k=8$）
- パラメータ数を $d^2 + dn$ から $d^2 + d(k_1 + k_2)$ へ削減

### Factorized Random Synthesizer

$$R = R_1 R_2^T, \quad R_1, R_2 \in \mathbb{R}^{n \times k}$$

- パラメータ数を $n^2$ から $2nk$ へ削減
- $k = 8$ では大幅なパラメータ削減が可能

## Mixture（混合）モデル

複数のSynthesizerを加重和で組み合わせる：

$$Y = \alpha_1 \cdot \text{softmax}(A_1)V + \alpha_2 \cdot \text{softmax}(A_2)V + \ldots$$

実験では Random + Dense、Random + Vanilla Attention などの組み合わせを検証した。

# パラメータ数の比較

| モデル | パラメータ数（ヘッドごと） | 備考 |
|--------|----------------------|------|
| Dot Product | $2d^2$ | $W_Q, W_K$ のみ（$W_V$ は別途） |
| Random | $n^2$ | 系列長依存 |
| Factorized Random | $2nk$ | $k=8$ |
| Dense | $d^2 + dn$ | 2層 FFN |
| Factorized Dense | $d^2 + d(k_1+k_2)$ | 低ランク近似 |

# 標準的セルフアテンションとの比較

| 比較軸 | Dot Product Attention | Dense Synthesizer | Random Synthesizer |
|--------|----------------------|-------------------|-------------------|
| トークン依存性 | Yes（クエリ・キー） | Yes（トークン自身のみ） | No |
| ペアワイズ相互作用 | Yes | No | No |
| 入力非依存コンポーネント | No | No | Yes（完全） |
| グローバルパターン学習 | No（クエリ・キー依存） | No | Yes |
| 系列長依存パラメータ | No | No（$W_2$ が $d \times n$） | Yes（$R \in \mathbb{R}^{n \times n}$） |

> [!NOTE]
> 本論文の主張：「Dense Synthesizer は各トークン位置について、他の全トークンとの整合性を独立して学習できる」。これは位置ごとのグローバルパターンを暗黙的に符号化していると解釈できる。

# アルゴリズム（Dense Synthesizer の前向き計算）

```
Input:
  X ∈ R^{n×d}  - シーケンスの埋め込み行列
  W1 ∈ R^{d×d}, W2 ∈ R^{d×n}  - Dense 変換の重み（ヘッドごと）
  WV ∈ R^{d×dv}  - Value 変換の重み

Output:
  Y ∈ R^{n×dv}  - 出力埋め込み

Algorithm DenseSynthesizerAttention(X, W1, W2, WV):
  // Step 1: Value の計算
  V ← X · WV                   // V ∈ R^{n×dv}

  // Step 2: 各トークンから Attention スコアを生成
  for i = 1 to n:
    h_i ← ReLU(W1 · x_i)      // h_i ∈ R^d
    B_i ← W2 · h_i             // B_i ∈ R^n（系列全体への attention スコア）

  // Step 3: 行列 B の組み立て
  B ← stack([B_1, ..., B_n])   // B ∈ R^{n×n}

  // Step 4: Softmax 正規化と出力
  A ← softmax(B, dim=-1)       // A ∈ R^{n×n}（行方向で正規化）
  Y ← A · V                    // Y ∈ R^{n×dv}

  return Y
```

# Experiments

## Datasets

| タスク | データセット | 規模 |
|--------|------------|------|
| 機械翻訳 | WMT 2014 English-German (EnDe) | 標準ベンチマーク |
| 機械翻訳 | WMT 2014 English-French (EnFr) | 標準ベンチマーク |
| 言語モデリング | LM1B（One Billion Word） | 大規模言語モデリング |
| テキスト生成 | CNN/DailyMail | 抽象的要約 |
| テキスト生成 | PersonaChat | 対話生成 |
| 事前学習 | C4 Dataset | 524K ステップ（2x2 TPU V3） |
| 分類 | GLUE（9 タスク） | CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE, STSB, WNLI |
| 分類 | SuperGLUE（8 タスク） | BoolQ, CB, CoPA, MultiRC, ReCoRD, RTE, WiC, WSC |
| エンコーダ分類 | AGNews | ニュースカテゴリ分類 |
| エンコーダ分類 | Movie Reviews | 感情分析 |

## Hardware

- 事前学習実験: 2x2 TPU V3 Chips（$2 \times 2$ トポロジー）
- パラメータ数: ベースライン Transformer（T5-Base 相当）223M

## 主要結果

### 機械翻訳（WMT'14 EnDe、BLEU）

| モデル | BLEU |
|--------|------|
| Vanilla Transformer | 27.67 |
| Dense Synthesizer | 27.43 |
| Random Synthesizer | 27.27 |
| Mixture (R + D) | 27.68 |
| Mixture (R + Vanilla) | **28.47** |

### 言語モデリング（LM1B、Perplexity ↓）

| モデル | PPL |
|--------|-----|
| Vanilla Transformer | 38.21 |
| Dense Synthesizer | 40.88 |
| Factorized Random | 42.40 |
| Mixture (Dense + Vanilla) | **37.27** |

### 事前学習効率（C4データセット）

| モデル | log PPL ↓ | 速度（steps/sec） |
|--------|-----------|-----------------|
| Dynamic Convolution | 2.040 | 2.65 |
| Random Synthesizer | **1.972** | **4.26** |

Random Synthesizer は Dynamic Convolution に対して **60% の速度向上** と **3.5% の相対 PPL 改善** を達成。

### GLUE / SuperGLUE（T5 ファインチューニング）

| モデル | GLUE | SuperGLUE |
|--------|------|-----------|
| Vanilla T5 Base | 83.5 | 70.3 |
| Mixture (R + Vanilla) | **84.1** | **72.2** |

混合モデルがベースラインを **GLUE +0.6、SuperGLUE +1.9** 上回る。

# 主要な知見と考察

## Random Attention の驚くべき有効性

Fixed Random Synthesizer（学習なし、ランダム固定重み）が WMT'14 で 27.27 BLEU を達成したことは、Transformer の多くの性能が Value の集約によるものであり、精緻なアテンションスコアの計算によるものではない可能性を示唆する。

## 混合モデルの優位性

純粋な Synthesizer モデルは単独ではドット積アテンションに劣ることが多いが、混合すると一貫してベースラインを上回る。これは合成アテンションとドット積アテンションが**補完的な情報**を学習していることを示す。

## タスク依存性

- **クロスセンテンス照合が不要なタスク**（翻訳、言語モデリング）：Synthesizer は競争力がある
- **クロスセンテンス照合が必要なタスク**（GLUE/SuperGLUE）：純粋 Synthesizer は劣化
- **対話生成**（PersonaChat）：ドット積アテンションを追加するとむしろ性能が低下するケースも

## 系列長依存パラメータの制限

Random Synthesizer は $R \in \mathbb{R}^{n \times n}$ という系列長依存パラメータを持つため、アーキテクチャ定義時に系列長を固定する必要がある。可変長系列への適用には Factorized バリアントが必要。

# 関連手法との比較

| 手法 | アテンション計算 | 計算量 | 特徴 |
|------|----------------|--------|------|
| Vanilla Transformer | $QK^T / \sqrt{d}$ | $O(n^2 d)$ | 全ペアワイズ |
| Synthesizer Dense | $F(x_i)$ | $O(n d^2 + n^2 d)$ | 単一トークンから生成 |
| Synthesizer Random | 学習済み $R$ | $O(n^2)$ | 入力非依存 |
| Linformer | 低ランク $QK^T$ | $O(nkd)$ | KV の低ランク近似 |
| Dynamic Convolution | 畳み込みカーネル | $O(nkd)$ | 深さ方向畳み込み |

> [!TIP]
> Synthesizer は [Linformer](https://arxiv.org/abs/2006.04768) と同時期に提案されたが、アプローチが異なる。Linformer は KV 行列を低ランク近似するのに対し、Synthesizer はアテンション行列自体を別の方法で生成する。

> [!CAUTION]
> この論文の結論「トークン間ペアワイズ相互作用は必須ではない」は示唆的だが、タスク依存性が強い。複雑な推論タスクや長文理解タスクでは依然としてドット積アテンションが優れる可能性が高い。
