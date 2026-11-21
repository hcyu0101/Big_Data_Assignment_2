# GitHub Activity Prediction — Hybrid C baseline

This is the **modeling code only**. It does not replace or modify the course preprocessing.
Run the assignment's original `scripts/preprocess.py` first; this project reads the eight
Parquet tables it creates.

## 1. Expected layout

```text
assignment2/
  scripts/preprocess.py
  scripts/cohort.txt
  scripts/evaluate.py
  submission/test_input.csv
  data/db/
    Users.parquet
    Repositories.parquet
    PullRequests.parquet
    Issues.parquet
    IssueComments.parquet
    PRReviews.parquet
    PRReviewComments.parquet
    Pushes.parquet
  github_activity_hybrid/
    prepare.py
    train.py
    predict.py
    src/
      common.py
      features.py
```

## 2. Install

```bash
pip install pandas pyarrow numpy scikit-learn torch
```

## 3. Prepare chronological training and validation data

From the assignment root:

```bash
python github_activity_hybrid/prepare.py \
  --db-dir data/db \
  --cohort-file scripts/cohort.txt \
  --data-start 2026-10-01 \
  --out-dir github_activity_hybrid/artifacts
```

Default rolling training cutoffs are Oct 8, 15, 22 and 29, 2026. Each sample
uses the next seven days as its target. Validation uses Nov 10–17, 2026. The final training target ends on Nov 5, leaving a five-day gap before validation to prevent target-window overlap.

All timestamps are UTC, as specified by the assignment. The validation window must be fully
covered by the downloaded data. Do not use any event after a snapshot cutoff as a feature.

## 4. Train

```bash
python github_activity_hybrid/train.py \
  --artifacts github_activity_hybrid/artifacts \
  --epochs 80
```

The baseline is a multi-label MLP with weighted binary cross entropy. Per-task thresholds are
tuned on the chronological validation snapshot; these thresholds are for the first experiment
only and should be treated cautiously because validation contains one target week.

## 5. Create a feature snapshot and predict

Choose a UTC cutoff that is no later than the latest complete data available. For example:

```bash
python github_activity_hybrid/prepare.py \
  --db-dir data/db \
  --cohort-file scripts/cohort.txt \
  --data-start 2026-10-01 \
  --out-dir github_activity_hybrid/artifacts \
  --predict-cutoff 2026-11-17T00:00:00Z

python github_activity_hybrid/predict.py \
  --features github_activity_hybrid/artifacts/predict.npz \
  --checkpoint github_activity_hybrid/artifacts/hybrid_mlp.pt \
  --test-input submission/test_input.csv \
  --output output_hybrid.csv
```

The prediction cutoff above is only an example. It is a UTC timestamp, and the feature builder
uses only rows whose official row-time is strictly before it.

## Important implementation notes

- `Users.user_id` and all foreign keys are period-dependent row numbers. The code maps them
  through `Users.github_user_id`, which is stable across processing periods and matches
  `test_input.csv` / `cohort.txt`.
- The cohort list is read from the course-provided `scripts/cohort.txt`, so users with no
  observed activity in the downloaded period still receive zero-valued features and labels.
- Historical features use the assignment's row-time columns: `first_seen_at` for PRs, issues
  and reviews; `created_at` for comments and pushes.
- Target labels use the corresponding activity timestamps: `created_at` for PRs/issues/comments,
  `submitted_at` for reviews, and `created_at` for pushes.
- `open_issue` excludes issue rows where `is_pull_request=True`.
- This first version is a deliberately simple baseline. Next we should compare 7-day vs 30-day
  features, relationship aggregates, and temporal decay separately before adding a GRU or GNN.

## Data Claim

The data used in this assignment is sourced from [GH Archive](https://www.gharchive.org), a
publicly available archive of GitHub events. This assignment is intended for educational
purposes within this course.

