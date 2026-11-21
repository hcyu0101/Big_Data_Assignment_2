# GitHub User Activity Prediction

Predicting future GitHub user activities using historical activity records, relational features, and temporal information.

This project is developed for **Assignment 2 of Big Data Analytics Techniques and Applications**.

## Overview

We formulate GitHub user activity prediction as a multi-label classification task, predicting six types of activities within the next seven days.

Our framework explores three feature groups:

- **User History (U):** Historical activity counts.
- **Relational Features (R):** Repository breadth, object diversity, and activity concentration.
- **Temporal Features (T):** Recent momentum, time decay, recency, and activity rhythm.

## Models & Experiments

| Experiment | Model | Features |
|---|---|---|
| E0 | MLP | U |
| E1 | MLP | U + R |
| E2 | MLP | U + T |
| E3 | GRU Hybrid | U + R + T + Sequential Features |
| E4 | MLP Hybrid | U + R + T |
| E5 | Ensemble | Copy Last Week + E4 |

The MLP Hybrid combines user history, relational, and temporal features. The GRU Hybrid additionally incorporates a 28-day activity sequence.

## Dataset

The dataset is derived from GH Archive and processed into eight Parquet tables:

Users, Repositories, PullRequests, Issues, IssueComments, PRReviews, PRReviewComments, and Pushes.

- Historical feature window: 30 days
- GRU sequence window: 28 days
- Prediction horizon: 7 days

## Evaluation

Models are evaluated using:

$\text{Score}=\frac{\text{Macro-F1}+\text{Micro-F1}}{2}$

Decision thresholds are tuned independently for each activity using the validation set.

## Repository Structure

```text
├── scripts/
│   ├── preprocess.py
│   ├── evaluate.py
│   └── cohort.txt
├── submission/
├── data/
│   └── db/
├── github_activity_hybrid/
│   ├── prepare.py
│   ├── train.py
│   ├── predict.py
│   ├── src/
│   └── artifacts/
└── README.md
```

## Usage

1. Preprocess the raw GH Archive data into Parquet tables.
2. Prepare training, validation, and prediction features.
3. Train the desired model using the selected feature groups.
4. Generate predictions using the trained checkpoint.

See the scripts in `github_activity_hybrid/` for implementation details and configurable arguments.

MLP Prediction
```
python github_activity_hybrid/predict.py  \
   --features artifacts/predict.npz  \
   --checkpoint artifacts/model_E4.pt  \
   --test-input submission/test_input.csv  \   
   --output submission/output_E4.csv  \
   --model mlp
```

GRU Prediction
```python
python github_activity_hybrid/predict.py \
    --features artifacts/predict.npz \
    --checkpoint artifacts/model_E3.pt \
    --test-input submission/test_input.csv \
    --output submission/output_E3.csv \
    --model gru
```

Copy Last Week Baseline
```python
python github_activity_hybrid/predict.py \
    --features artifacts/predict.npz \
    --test-input submission/test_input.csv \
    --output submission/output_baseline.csv \
    --model copy_last_week
```

Ensemble Prediction
```python
python github_activity_hybrid/predict.py \
    --features artifacts/predict.npz \
    --checkpoint artifacts/model_E4.pt \
    --test-input submission/test_input.csv \
    --output submission/output_E5.csv \
    --model ensemble
```

## Data Claim

The data used in this assignment is sourced from [GH Archive](https://www.gharchive.org), a
publicly available archive of GitHub events. This assignment is intended for educational
purposes within this course.

