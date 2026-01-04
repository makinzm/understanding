# Meta Information

- URL: 
    - [Recruit Data Blog | 【解法紹介】RecSys Challenge 2025 で優勝しました](https://blog.recruit.co.jp/data/articles/recsys-challenge-2025/)
    - [RecSys Challenge 2025](https://www.recsyschallenge.com/2025/)
    - [yukia18/recsys-challenge-2025-1st-place: 1st place solution by team rec2 for RecSys Challenge 2025.](https://github.com/yukia18/recsys-challenge-2025-1st-place)
    - [Synerise/recsys2025: Synerise RecSys Challenge 2025](https://github.com/Synerise/recsys2025)
    - [RecSys Challenge 2025 by Synerise](https://recsys.synerise.com/)
    - [Toward Universal User Representations: Contrastive Learning with Transformers and Embedding Ensembles](https://dl.acm.org/doi/epdf/10.1145/3758126.3758137)
- LICENSE: Thesis License is [Deed - Attribution 4.0 International - Creative Commons](https://creativecommons.org/licenses/by/4.0/)

# Task

Universal Behavioral Profiles is developed to predict use behavior to several models.

## Open Tasks

1. Churn Prediction of Active User (Moving out of the service)
2. Product Propensity
3. Category Propensity

## Hidden Task during the competition

1. Churn Prediction of All Users (Including Inactive Users)
2. Unknown Product Propensity
3. Price Range Propensity

# Evaluation Metric

0.8 * AUROC + 0.1 * Novelty + 0.1 * Diversity

> [!IMPORTANT]
> AUROC: Area Under the Receiver Operating Characteristic Curve: [AUROC — PyTorch-Metrics 1.8.2 documentation](https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html)
> Diversity: Measures the diversity of predictions by calculating the entropy of the predicted probability distribution, normalized by log₂(number of items). Range: [0, 1]. A value of 1 indicates perfectly uniform predictions across all items, while 0 indicates concentration on specific items. : [recsys2025/training_pipeline/metrics.py at main · Synerise/recsys2025](https://github.com/Synerise/recsys2025/blob/main/training_pipeline/metrics.py)
> Novelty: Measures how much the model recommends less popular items. Calculated as (1 - normalized_popularity)^100, where popularity is the weighted sum of the top-k predictions' popularity scores from training data. Range: [0, 1]. A value of 1 indicates recommending only the least popular items, while 0 indicates recommending only the most popular items. The 100th power amplifies sensitivity to small differences near 1.: [recsys2025/training_pipeline/metrics.py at main · Synerise/recsys2025](https://github.com/Synerise/recsys2025/blob/main/training_pipeline/metrics.py)

# Dataset

[RecSys Challenge 2025 by Synerise | Data set](https://recsys.synerise.com/data-set)

All the table other than product_properties include client_id and timestamp columns.

# 1st Place Solution

Create three types of user representations and ensemble them.

1. Contrasive Learning Transformer
2. Multi-task Learning
3. Aggregated Embeddings

## 1. Contrasive Learning Transformer

Code: [recsys-challenge-2025-1st-place/cl_transformer at main · yukia18/recsys-challenge-2025-1st-place](https://github.com/yukia18/recsys-challenge-2025-1st-place/tree/main/cl_transformer)

### Model Architecture

Contrasive Learning which make the same user's two different time windows' representations closer.

Tower Structure is different for input and target but Transformer based.

NT-Xent Loss for Contrastive Learning and BCE for predicting whether the future interaction happens.

Whether the user is in the evaluation set is also predicted, and then weighed loss is used for training.

### Code Details

1. Use Hydra to manage configuration: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/scripts/create_embeddings.py#L1448-L1449
2. Use Pytorch Lightning to manage training loop: 
  - https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/scripts/create_embeddings.py#L1192
  - https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/scripts/create_embeddings.py#L1611-L1629
3. Input Term Tower: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/scripts/create_embeddings.py#L909-L1071
4. Target Term Tower: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/scripts/create_embeddings.py#L1073-L1189
5. Flash Attention: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/src/torch/models/flash_attn_v2.py#L106-L190
6. Weighted Loss: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/scripts/create_embeddings.py#L404-L424
  - https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/scripts/create_embeddings.py#L1272C9-L1274
  - https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/scripts/create_embeddings.py#L1304
7. Optimizer: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/cl_transformer/scripts/create_embeddings.py#L1362-L1372

## 2. Multi-task Learning

### Model Architecture

Progressive Layered Extraction (PLE) based Multi-task Learning Transformer model.

Sequential User Behavior is processed by Transformer layers.
Numerical Feature is processed by MLP.
And then, these outputs are concatenated and passed to Task-specific Towers.

### Code Details

1. Create Dataset: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/mtl_transformer/src/create_dataset.py
  - Sequencial Behavior Dataset: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/mtl_transformer/src/create_dataset.py#L303-L308
  - Numerical Feature Dataset: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/mtl_transformer/src/create_dataset.py#L288-L298C24
2. Training: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/mtl_transformer/src/train.py
  - Concatenate Shared Expert and Task-specific Expert: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/mtl_transformer/src/train.py#L772-L798
  - Compute User Embedding: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/mtl_transformer/src/train.py#L910-L932
3. Create Embeddings: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/mtl_transformer/src/create_embeddings.py
  - Create Embedding: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/mtl_transformer/src/create_embeddings.py#L874-L896

>[!NOTE]
> There is duplicated code in creating embeddings and training (e.g., `LightningRecsysModel` class)

>[!NOTE]
> Optimizer is AdamW though contrasive learning transformer uses RAdamScheduleFree.

## 3. Aggregated Embeddings

### Model Architecture

Calculate various aggregated features from user behavior history.

### Code Details

- Event Type: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/feature_engineering/feature_engineering/aggregated_features_baseline/constants.py#L25-L31
- Calculator Class: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/feature_engineering/feature_engineering/aggregated_features_baseline/calculators.py#L51-L73
- Merge all features: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/feature_engineering/feature_engineering/aggregated_features_baseline/features_aggregator.py#L358-L395


## Ensemble

Use Gaussian Error Linear Unit (GELU) based Stacking model to ensemble three types of user representations.

This ensemble model is also trained.

### Code Details

https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/stacking/run/stacking.py

- Model: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/stacking/run/stacking.py#L275-L283
  - https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/stacking/src/ensemble/stacking/torch/model.py#L80-L128
- DataSet: https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/stacking/run/stacking.py#L250-L273
  - https://github.com/yukia18/recsys-challenge-2025-1st-place/blob/main/stacking/src/ensemble/stacking/torch/dataset.py#L8

## Results

Ensemble model outperformed each single model.
