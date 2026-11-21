from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

EVENTS = [
    "open_pr", "open_issue", "issue_comment",
    "pr_review", "pr_review_comment", "push",
]

TABLE_COLUMNS = {
    "Users": ["user_id", "github_user_id", "is_cohort", "is_bot"],
    "PullRequests": ["pr_id", "author_id", "repo_id", "first_seen_at", "created_at"],
    "Issues": ["issue_id", "author_id", "repo_id", "is_pull_request", "first_seen_at", "created_at"],
    "IssueComments": ["comment_id", "author_id", "repo_id", "issue_id", "created_at"],
    "PRReviews": ["review_id", "reviewer_id", "repo_id", "pr_id", "first_seen_at", "submitted_at"],
    "PRReviewComments": ["comment_id", "author_id", "repo_id", "pr_id", "created_at"],
    "Pushes": ["push_id", "user_id", "repo_id", "created_at"],
}

def load_tables(db_dir: str | Path):
    """Read only the columns used by this baseline to reduce RAM consumption."""
    db_dir = Path(db_dir)
    tables = {}
    for name, cols in TABLE_COLUMNS.items():
        path = db_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run the course preprocess.py first.")
        tables[name] = pd.read_parquet(path, columns=cols)
    return tables

def load_cohort(cohort_file: str | Path) -> np.ndarray:
    values = [int(x) for x in Path(cohort_file).read_text().split() if x.strip()]
    return np.asarray(sorted(set(values)), dtype=np.int64)

def to_datetime_columns(tables):
    for name, df in tables.items():
        for col in df.columns:
            if col.endswith("_at"):
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return tables

def make_user_map(users: pd.DataFrame) -> dict:
    """Map preprocess.py's period-dependent Users.user_id to stable github_user_id."""
    clean = users.dropna(subset=["user_id", "github_user_id"])
    return dict(zip(clean["user_id"].astype("int64"), clean["github_user_id"].astype("int64")))

def add_user_code(df: pd.DataFrame, id_col: str, user_map: dict) -> pd.DataFrame:
    out = df.copy()
    out["user_code"] = out[id_col].map(user_map)
    return out
