## Meta Information

Key Value Cacheが初めて登場した論文については見つけることはできませんでしたが、次の参考資料が参考になります。

1. [The KV Cache: Memory Usage in Transformers from Efficient NLP - YouTube](https://www.youtube.com/watch?v=80bIUggRJf4)
2. [KV Cache Secrets: Boost LLM Inference Efficiency | by Shoa Aamir | Medium](https://medium.com/@shoa.devs/kv-cache-secrets-boost-llm-inference-efficiency-ae6c53968857)

### 1/3. なぜキャッシュが有効なのか？

Transformerの自己注意機構では、各トークン生成時に**過去のすべてのトークンとの注意計算**が必要です。キャッシュなしだと：

- 1トークン目：1個のトークンで計算
- 2トークン目：2個のトークンで計算（1個目を再計算）
- 3トークン目：3個のトークンで計算（1-2個目を再計算）
- n トークン目：n個のトークンで計算

→ 行列計算を考えると、計算量は $O(\Sigma_{i=1}^{n} i^2) = O(n^3)$ にもなり、非常に非効率です。

### 2/3. 何をキャッシュし、何を更新するか？

**キャッシュするもの：**
- 過去に生成した全トークンの**KeyとValue**の行列
- 各レイヤーごとに保存

**毎回更新するもの：**
- 新しく生成したトークン1つ分のK, Vを**追加**
- 新しいトークンのQuery（Q）を計算し、キャッシュされたK, V全体と注意計算

```
ステップ t:
  新しいQ_t を計算
  K_cache = [K_1, K_2, ..., K_{t-1}]  # 既存
  V_cache = [V_1, V_2, ..., V_{t-1}]  # 既存
  
  Attention(Q_t, K_cache, V_cache) を計算
  
  K_t, V_t を計算してキャッシュに追加
  K_cache = [K_1, K_2, ..., K_{t-1}, K_t]
  V_cache = [V_1, V_2, ..., V_{t-1}, V_t]
```

### Encoder側？Decoder側？

**主にDecoder側で使用：**
- GPTなどの自己回帰モデル（デコーダーのみ）
- Decoder-only architectureでの逐次生成時

**Encoderでは通常不要：**
- 入力全体を一度に処理（並列計算）
- 逐次生成がないため

**Encoder-Decoderモデル（T5、BARTなど）の場合：**
- Encoderの出力（K, V）は1回計算して全デコードステップで再利用
- Decoderの自己注意部分でKVキャッシュを使用

### メリット
- 計算量：$O(n^2)$ → $O(n)$ に削減
- 生成速度が大幅に向上
- メモリ使用量とのトレードオフ

## 3/3. なぜQは最新のものだけでよいのか？

### 自己回帰生成の仕組み(Decoder-onlyモデル)

各ステップで**「今生成しようとしているトークン」が過去のトークンすべてを参照する**という一方向的な関係があります：

```
ステップ1: Q_1 が K_1, V_1 を参照 → token_1 生成
ステップ2: Q_2 が K_1, K_2 と V_1, V_2 を参照 → token_2 生成
ステップ3: Q_3 が K_1, K_2, K_3 と V_1, V_2, V_3 を参照 → token_3 生成
```

### 重要なポイント

**過去のQ（例：Q_1, Q_2）は二度と使われない**

- Q_1は token_1 を生成したら役目終了
- Q_2は token_2 を生成したら役目終了
- Q_3は token_3 を生成する時**だけ**必要

**一方、K と V は繰り返し使われる**

- K_1, V_1 は token_2, token_3, token_4... の生成時すべてで参照される
- だからキャッシュする価値がある

### 具体例

```
"The cat sat on the" まで生成済みで "mat" を生成する場合：

Q_6: "The cat sat on the の次は何？"（位置6の "the" のEmbeddingから計算）
↓ 注意を向ける先 ↓
K_1="The", K_2="cat", K_3="sat", K_4="on", K_5="the": 「これまでの文脈情報」
V_1="The", V_2="cat", V_3="sat", V_4="on", V_5="the": 「その内容」

このときに新たに計算されるのは:
- Q_6, K_6, V_6（位置6の "the" から）のみ
- K_1~K_5, V_1~V_5 はキャッシュから取得

過去のQueryは不要:
- Q_1, Q_2, Q_3, Q_4, Q_5 は各トークン生成時に使い捨て
- "mat" 生成時には参照されない
```

### まとめ

- **Q**: 「今これから生成するトークンの視点」→ 使い捨て
- **K, V**: 「過去の文脈情報」→ 今後も繰り返し参照される

---

KVキャッシュは長文生成を実用的な速度で実現する重要な最適化技術です。

