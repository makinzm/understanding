# What this Document Is

[実践MLOps 作って理解する機械学習システムの構築と運用 | Ohmsha](https://www.ohmsha.co.jp/book/9784274233982/)の内容を一通り読みました。

こちらをもとに、どのようなツール群について学習すればよいかを改めて考察をしたドキュメントです。

MLOpsの全体像を、インフラ層から本番運用まで含めて再構成します。

## MLOpsにおけるDoD(Definition of Done)とツールマッピング

## 0. インフラストラクチャ層

### DoD: MLOps環境が再現可能かつコード管理されている

達成条件:
- すべてのインフラがコードで定義されている
- 環境の作成・破棄が自動化されている
- インフラの変更履歴が追跡可能

ツール:
- [Terraform](https://github.com/hashicorp/terraform) / [Pulumi](https://github.com/pulumi/pulumi): Infrastructure as Code (IaC)
- [AWS CDK](https://github.com/aws/aws-cdk) / [Azure Bicep](https://github.com/Azure/bicep): クラウド特化型IaC
- [Ansible](https://github.com/ansible/ansible) / [Chef](https://github.com/chef/chef): 設定管理
- [Helm](https://github.com/helm/helm): Kubernetesパッケージ管理

なぜ学ぶ必要があるか:
- 手動で構築した環境は再現不可能。障害時の復旧や、開発・本番の環境差異がトラブルの原因になる

## 1. CI/CDパイプライン層

### DoD: コード変更が自動的にテスト・デプロイされる

達成条件:
- コミット時に自動テストが実行される
- コンテナイメージが自動ビルド・プッシュされる
- デプロイが承認フローを経て自動実行される
- パイプラインの実行履歴が記録されている

ツール:
- [GitHub Actions](https://github.com/actions/runner): CI/CDプラットフォーム
- [Jenkins](https://github.com/jenkinsci/jenkins): エンタープライズCI/CD
- [CircleCI](https://circleci.com/) / [Travis CI](https://github.com/travis-ci/travis-ci): マネージドCI/CD
- [Argo Workflows](https://github.com/argoproj/argo-workflows): Kubernetes-nativeワークフロー
- [Docker](https://github.com/docker/docker-ce) / [Podman](https://github.com/containers/podman): コンテナ化
- Container Registry ([AWS ECR](https://aws.amazon.com/jp/ecr/) / [GCP Artifact Registry](https://cloud.google.com/artifact-registry) / [Azure ACR](https://azure.microsoft.com/ja-jp/products/container-registry)): イメージ管理

なぜ学ぶ必要があるか:
- 手動デプロイは遅く、エラーが起きやすい。1日に何度もデプロイできる体制がビジネス速度を決める

## 2. 実験管理フェーズ

### DoD: すべての実験が再現可能である

達成条件:
- ハイパーパラメータが構造化されて記録されている
- 実験結果が検索・比較可能
- データバージョンとモデルバージョンが紐づいている
- 環境・依存関係が固定されている

ツール:

Deep Learningフレームワーク:
- [PyTorch](https://github.com/pytorch/pytorch): 最も人気のあるディープラーニングフレームワーク
- [TensorFlow](https://github.com/tensorflow/tensorflow): Google製のエンドツーエンドMLプラットフォーム
- [JAX](https://github.com/jax-ml/jax): 高速な自動微分ライブラリ
- [Burn](https://github.com/tracel-ai/burn): Rust製の次世代DLフレームワーク（型安全・高速・マルチバックエンド対応）

設定・実験管理:
- [Hydra](https://github.com/facebookresearch/hydra) / [OmegaConf](https://github.com/omry/omegaconf): 設定管理、マルチラン実験
- [MLflow](https://github.com/mlflow/mlflow) / [Weights & Biases](https://github.com/wandb/wandb) / [Neptune.ai](https://github.com/neptune-ai/neptune-client): 実験トラッキング
- [DVC](https://github.com/iterative/dvc) / [Pachyderm](https://github.com/pachyderm/pachyderm): データバージョン管理
- [uv](https://github.com/astral-sh/uv) / [Poetry](https://github.com/python-poetry/poetry) / [Conda](https://github.a): 依存関係管理

モデルフォーマット・相互運用:
- [ONNX](https://github.com/onnx/onnx): Open Neural Network Exchange（フレームワーク間でモデルを交換する標準フォーマット）
  - PyTorch/TensorFlow → ONNX → Burn/ONNX Runtimeで推論
  - フレームワーク非依存な推論を実現

なぜ学ぶ必要があるか:
- 「3ヶ月前のあの実験、どの設定だったっけ？」が再現できないと、過去の知見が活かせない

## 3. データパイプライン層

### DoD: データ処理が自動化・スケーラブルである

達成条件:
- データ取得から特徴量生成まで自動化
- 依存関係に基づく実行順序制御
- 失敗時のリトライ・アラート機能
- データ系譜(Lineage)の追跡可能性

ツール:
- [Airflow](https://github.com/apache/airflow) / [Prefect](https://github.com/PrefectHQ/prefect) / [Dagster](https://github.com/dagster-io/dagster): ワークフローオーケストレーション
- [Kubeflow Pipelines](https://github.com/kubeflow/pipelines): ML特化型パイプライン
- [dbt](https://github.com/dbt-labs/dbt-core): データ変換パイプライン
- [Apache Spark](https://github.com/apache/spark) / [Apache Flink](https://github.com/apache/flink): 大規模データ処理
- [Feast](https://github.com/feast-dev/feast) : 特徴量ストア

なぜ学ぶ必要があるか:
- データパイプラインの障害がML全体を止める。自動化されていないと、データサイエンティストの時間がパイプライン保守に消える

## 4. モデル検証・テスト層

### DoD: モデルの品質が定量的に保証されている

達成条件:
- データ品質チェックが自動実行される
- モデル性能が閾値を満たしている
- バイアス・公平性が検証されている
- 単体テスト・統合テストが通過している

ツール:
- [Great Expectations](https://github.com/great-expectations/great_expectations) / [Deequ](https://github.com/awslabs/deequ): データ品質テスト
- [Evidently AI](https://github.com/evidentlyai/evidently): モデル評価・ドリフト検証
- [pytest](https://github.com/pytest-dev/pytest) / [unittest](https://docs.python.org/ja/3/library/unittest.html): コードテスト
- [Fairlearn](https://github.com/fairlearn/fairlearn) / [AIF360](https://github.com/Trusted-AI/AIF360): 公平性評価
- [SHAP](https://github.com/shap/shap) / [LIME](https://github.com/marcotcr/lime): 説明可能性

なぜ学ぶ必要があるか:
- テストなしで本番に出すのは、ブレーキのない車で高速道路を走るようなもの

## 5. モデルデプロイメント層

### DoD: モデルが安全に本番環境で稼働している

達成条件:
- カナリアリリース・Blue-Greenデプロイが可能
- ロールバックが即座に実行できる
- 推論APIがSLA要件を満たす
- デプロイ履歴が記録されている

ツール:

モデルサービング:
- [Seldon Core](https://github.com/SeldonIO/seldon-core) / [KServe](https://github.com/kserve/kserve) / [BentoML](https://github.com/bentoml/BentoML): 汎用モデルサービング
- [TensorFlow Serving](https://github.com/tensorflow/serving): フレームワーク特化サービング
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): ONNX形式モデルの高速推論エンジン
- [Burn](https://github.com/tracel-ai/burn): Rust製フレームワーク（学習コードをそのまま推論に使用可能、WebAssembly/no_std対応）
- [FastAPI](https://github.com/tiangolo/fastapi) / [Flask](https://github.com/pallets/flask): カスタムAPI構築

コンテナオーケストレーション:
- [Kubernetes](https://github.com/kubernetes/kubernetes) / [AWS ECS](https://aws.amazon.com/jp/ecs/) / [GCP Cloud Run](https://cloud.google.com/run?hl=ja): コンテナオーケストレーション
- [Istio](https://github.com/istio/istio) / [Linkerd](https://github.com/linkerd/linkerd2): サービスメッシュ
- [ArgoCD](https://github.com/argoproj/argo-cd) / [FluxCD](https://github.com/fluxcd/flux2): GitOpsデプロイ

デプロイパターン別の選択:
- クラウド/サーバー推論: Seldon Core, KServe, TorchServe, ONNX Runtime
- エッジデバイス推論: ONNX Runtime, Burn (no_std), TensorFlow Lite
- WebAssembly/ブラウザ推論: Burn (WGPU/NdArray backend), ONNX Runtime Web

なぜ学ぶ必要があるか:
- デプロイ時の障害が最もビジネスインパクト大。段階的リリースとロールバック機能が必須
- ONNXを使えば、学習フレームワークに依存せず推論環境を選択できる
- Burnは学習コード=推論コードなので、モデル変換の手間が不要でPythonランタイムも不要

## 6. オブザーバビリティ層(監視・ログ・トレース)

### DoD: システムの状態が常に可視化されている

達成条件:
- メトリクス・ログ・トレースの3本柱が整備
- 異常検知時の自動アラート
- ダッシュボードで健全性が一目で分かる
- インシデント時の原因調査が迅速に可能

ツール:

メトリクス監視:
- [Prometheus](https://github.com/prometheus/prometheus) + [Grafana](https://github.com/grafana/grafana): メトリクス収集・可視化
- [Datadog](https://www.datadoghq.com/) / [New Relic](https://newrelic.com): 統合監視プラットフォーム
- [AWS CloudWatch](https://aws.amazon.com/jp/cloudwatch/) / [Azure Monitor](https://azure.microsoft.com/ja-jp/products/monitor): クラウドネイティブ監視

ログ管理:
- [Fluent Bit](https://github.com/fluent/fluent-bit) / [Fluentd](https://github.com/fluent/fluentd): ログ収集・転送
- [ELK Stack](https://www.elastic.co/jp/elastic-stack) (Elasticsearch, Logstash, Kibana): ログ分析
- [Loki](https://github.com/grafana/loki) + [Grafana](https://github.com/grafana/grafana): 軽量ログ管理

分散トレーシング:
- [OpenTelemetry (OTel)](https://github.com/open-telemetry/opentelemetry-specification): 統合オブザーバビリティ標準
- [Jaeger](https://github.com/jaegertracing/jaeger) / [Zipkin](https://github.com/openzipkin/zipkin): 分散トレーシング
- [Tempo](https://github.com/grafana/tempo): Grafanaのトレーシングバックエンド

アラート:
- [PagerDuty](https://www.pagerduty.com/) / [Opsgenie](https://www.atlassian.com/ja/software/opsgenie): インシデント管理
- [AlertManager](https://github.com/prometheus/alertmanager): Prometheus連携アラート

なぜ学ぶ必要があるか:
- 「なぜ推論が遅い？」「エラーはどこで起きた？」が分からないと、障害対応が泥沼化する

## 7. モデルモニタリング層

### DoD: モデルの劣化が早期検知される

達成条件:
- データドリフト・コンセプトドリフトが検出される
- 予測品質の低下が自動アラートされる
- 入力データの異常が検知される
- モデル間の性能比較が可視化されている

ツール:
- [Evidently AI](https://github.com/evidentlyai/evidently) / [Arize](https://arize.com/) / [phoenix](https://github.com/Arize-ai/phoenix): MLモニタリング特化
- [WhyLogs](https://github.com/whylabs/whylogs): データ品質・ドリフト監視
- [Alibi Detect](https://github.com/SeldonIO/alibi-detect): 異常検知・ドリフト検出
- [Grafana](https://github.com/grafana/grafana) + Custom Metrics: カスタムMLダッシュボード

なぜ学ぶ必要があるか:
- 本番モデルは必ず劣化する。インフラ監視だけではML特有の問題(ドリフト、バイアス悪化)は見つけられない

## 8. A/Bテスト・実験プラットフォーム層

### DoD: 新モデルの効果が統計的に検証されている

達成条件:
- トラフィック分割が正確に制御されている
- 統計的有意性が自動計算される
- ビジネスメトリクスへの影響が測定可能
- 実験履歴が記録されている

ツール:
- [GrowthBook](https://github.com/growthbook/growthbook): 実験プラットフォーム
- [Unleash](https://github.com/Unleash/unleash): オープンソースフィーチャーフラグ

なぜ学ぶ必要があるか:
- モデル精度向上がビジネス成果に繋がるとは限らない。実際の効果測定なしでは、何が本当に価値を生んでいるか分からない

## 9. モデルガバナンス・コンプライアンス層

### DoD: モデルが規制・内部基準を満たしている

達成条件:
- モデルの承認フローが機能している
- 監査ログが記録されている
- バイアス・公平性レポートが生成される
- モデルカードが作成されている

ツール:
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/): モデルライフサイクル管理
- [Vertex AI Model Registry](https://cloud.google.com/vertex-ai/docs/model-registry/introduction) / [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html): クラウドネイティブレジストリ
- [Model Cards](https://huggingface.co/docs/hub/model-cards): モデルドキュメンテーション

なぜ学ぶ必要があるか:
- 金融・医療など規制業界では必須。規制違反はビジネスリスク

## 10. 再学習・自動更新層

### DoD: モデル更新サイクルが自動化されている

達成条件:
- トリガー条件(時間/ドリフト/性能低下)で自動再学習
- 新モデルの自動検証・デプロイ
- 更新履歴・ロールバックポイントの記録

ツール:
- [Airflow](https://github.com/apache/airflow) / [Prefect](https://github.com/PrefectHQ/prefect): スケジュール・条件付きトリガー
- [Kubeflow Pipelines](https://github.com/kubeflow/pipelines): エンドツーエンド自動化
- [MLflow](https://github.com/mlflow/mlflow) + [Airflow](https://github.com/apache/airflow): 実験→登録→デプロイの自動化

なぜ学ぶ必要があるか:
- 監視で問題を見つけても、再学習が手動では対応が遅れる。自動化が最終ゴール

## 全体アーキテクチャ図

```
┌─────────────────────────────────────────────────────────┐
│ 0. Infrastructure Layer (Terraform, Ansible)           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 1. CI/CD Layer (GitHub Actions, ArgoCD)                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌──────────────┬──────────────┬─────────────────────────┐
│ 2. Experiment│ 3. Data      │ 4. Validation           │
│ (PyTorch,    │ (Airflow,dbt)│ (Great Expectations)    │
│  Hydra,      │              │                         │
│  MLflow,     │              │                         │
│  ONNX, Burn) │              │                         │
└──────────────┴──────────────┴─────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Deployment (Seldon, KServe, ONNX Runtime,          │
│                Burn, Kubernetes)                        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ 6. Observability │ 7. ML Monitoring │ 8. A/B Testing   │
│ (OTel, FluentBit)│ (Evidently)      │ (LaunchDarkly)   │
└──────────────────┴──────────────────┴──────────────────┘
                           ↓
┌──────────────────────────┬──────────────────────────────┐
│ 9. Governance            │ 10. Retraining               │
│ (Model Registry)         │ (Airflow + Triggers)         │
└──────────────────────────┴──────────────────────────────┘
```

## 学習優先順位マトリクス

| 優先度 | 基盤必須ツール | 応用ツール |
|--------|----------------|------------|
| 最優先 | Terraform, GitHub Actions, Docker, Kubernetes, Prometheus, Fluent Bit | OpenTelemetry |
| 高 | PyTorch, Hydra, MLflow, Airflow | ONNX, Evidently, Seldon Core |
| 中 | DVC, Great Expectations, ArgoCD | A/Bテストツール, Feast, BentoML |
| 低 | dbt, Burn (Rust志向の場合は高) | Fairlearn, WhyLabs |

## 推奨学習パス

### Month 1: 基盤構築
- Terraform でインフラコード化
- GitHub Actions で基本的なCI/CD
- Docker + Kubernetes 基礎

### Month 2: 実験・データ管理
- PyTorch でモデル構築
- Hydra で設定管理
- MLflow で実験トラッキング
- DVC でデータバージョン管理

### Month 3: オーケストレーション
- Airflow でパイプライン構築
- Great Expectations でデータテスト
- dbt でデータ変換(必要に応じて)

### Month 4: 監視基盤
- Prometheus + Grafana でメトリクス監視
- Fluent Bit でログ収集
- OpenTelemetry でトレーシング統合

### Month 5: MLデプロイ・監視
- ONNX でモデルエクスポート
- Seldon Core / KServe / ONNX Runtime でモデルサービング
- Evidently でMLモニタリング
- ArgoCD で GitOpsデプロイ

### Month 6: 高度な運用
- A/Bテストプラットフォーム (LaunchDarkly, Statsig)
- 自動再学習パイプライン
- インシデント対応フロー確立

### 番外編: Rust-native MLOps (興味がある場合)
- Burn でRust製ML実装
- WebAssembly推論
- 組込みデバイス向けno_std推論

## ツール選定の判断基準

### フレームワーク・推論エンジン選択

| ユースケース | 推奨ツール | 理由 |
|-------------|-----------|------|
| 汎用的な研究・開発 | PyTorch + MLflow | エコシステムが最も充実 |
| 本番推論（クラウド） | ONNX Runtime, TorchServe, Seldon Core | 高速・フレームワーク非依存 |
| エッジデバイス推論 | ONNX Runtime, Burn (no_std) | 軽量・依存少ない |
| WebAssembly推論 | Burn (WGPU backend), ONNX Runtime Web | ブラウザでGPU利用可能 |
| Rust統一開発 | Burn | 学習から推論まで同じコードベース |
| フレームワーク移行 | ONNX | PyTorch → TensorFlow等の移行を容易化 |
