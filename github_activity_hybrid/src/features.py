from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import EVENTS

def make_event_frames(tables, user_map):
    specs = {
        "open_pr": ("PullRequests", "author_id", "first_seen_at", "repo_id", "pr_id"),
        "open_issue": ("Issues", "author_id", "first_seen_at", "repo_id", "issue_id"),
        "issue_comment": ("IssueComments", "author_id", "created_at", "repo_id", "comment_id"),
        "pr_review": ("PRReviews", "reviewer_id", "first_seen_at", "repo_id", "review_id"),
        "pr_review_comment": ("PRReviewComments", "author_id", "created_at", "repo_id", "comment_id"),
        "push": ("Pushes", "user_id", "created_at", "repo_id", "push_id"),
    }
    frames = {}
    for event, (table, actor_col, time_col, repo_col, object_col) in specs.items():
        df = tables[table]
        if event == "open_issue":
            df = df[df["is_pull_request"].fillna(False) == False]
        x = df[[actor_col, time_col, repo_col, object_col]].copy()
        x = x.rename(columns={
            actor_col: "internal_user_id",
            time_col: "row_time",
            repo_col: "repo_id",
            object_col: "object_id",
        })
        x = x[x["internal_user_id"].notna() & x["row_time"].notna()]
        x["user_code"] = x["internal_user_id"].map(user_map)
        x = x[x["user_code"].notna()].copy()
        x["user_code"] = x["user_code"].astype("int64")
        frames[event] = x[["user_code", "row_time", "repo_id", "object_id"]]
    return frames

def make_label_frames(tables, user_map):
    specs = {
        "open_pr": ("PullRequests", "author_id", "created_at"),
        "open_issue": ("Issues", "author_id", "created_at"),
        "issue_comment": ("IssueComments", "author_id", "created_at"),
        "pr_review": ("PRReviews", "reviewer_id", "submitted_at"),
        "pr_review_comment": ("PRReviewComments", "author_id", "created_at"),
        "push": ("Pushes", "user_id", "created_at"),
    }
    frames = {}
    for event, (table, actor_col, time_col) in specs.items():
        df = tables[table]
        if event == "open_issue":
            df = df[df["is_pull_request"].fillna(False) == False]
        x = df[[actor_col, time_col]].copy()
        x = x.rename(columns={actor_col: "internal_user_id", time_col: "event_time"})
        x = x[x["internal_user_id"].notna() & x["event_time"].notna()]
        x["user_code"] = x["internal_user_id"].map(user_map)
        x = x[x["user_code"].notna()].copy()
        x["user_code"] = x["user_code"].astype("int64")
        frames[event] = x[["user_code", "event_time"]]
    return frames

def build_features(cohort_ids, event_frames, users, cutoff, data_start, use_U=True, use_R=False, use_T=False):
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    
    # 這裡將歷史窗口固定最大為 30 天，避免資料長度不一造成分佈飄移
    window_start = cutoff - pd.Timedelta(days=30)
    
    X = pd.DataFrame(index=pd.Index(cohort_ids, name="UserID"))
    feature_names = []

    u = users.dropna(subset=["github_user_id"]).copy()
    u["github_user_id"] = u["github_user_id"].astype("int64")
    bot_map = u.drop_duplicates("github_user_id").set_index("github_user_id")["is_bot"]
    X["is_bot"] = bot_map.reindex(X.index).fillna(False).astype(float)
    feature_names.append("is_bot")

    windows = {"7d": 7, "14d": 14, "30d": 30}
    
    for event in EVENTS:
        df = event_frames[event]
        # 只取過去 30 天的事件
        hist = df[(df["row_time"] < cutoff) & (df["row_time"] >= window_start)]

        # --- [U] User Baseline Features ---
        if use_U:
            for win_name, days in windows.items():
                part = hist[hist["row_time"] >= cutoff - pd.Timedelta(days=days)]
                
                # 自身計數 + log1p
                counts = part.groupby("user_code").size()
                col = f"{event}_{win_name}_count_log"
                X[col] = np.log1p(counts.reindex(X.index).fillna(0).astype(float))
                feature_names.append(col)

            # 多個半衰期 decay
            for half_life in [7, 14]:
                if len(hist):
                    age_days = (cutoff - hist["row_time"]).dt.total_seconds() / 86400.0
                    weighted = hist.assign(_weight=np.exp(-age_days / half_life)).groupby("user_code")["_weight"].sum()
                    col = f"{event}_decay{half_life}"
                    X[col] = weighted.reindex(X.index).fillna(0).astype(float)
                else:
                    col = f"{event}_decay{half_life}"
                    X[col] = 0.0
                feature_names.append(col)
                
            # 距離最後一次活動天數
            if len(hist):
                last = hist.groupby("user_code")["row_time"].max()
                age = (cutoff - last).dt.total_seconds() / 86400.0
                col = f"{event}_days_since_last"
                X[col] = age.reindex(X.index)
            else:
                col = f"{event}_days_since_last"
                X[col] = np.nan
            feature_names.append(col)
        
        # --- [R] Relational Features ---
        if use_R:
            for win_name, days in [("7d", 7), ("30d", 30)]:
                part = hist[hist["row_time"] >= cutoff - pd.Timedelta(days=days)]
                
                # 碰過的 Repo 數量
                repo_counts = part.groupby("user_code")["repo_id"].nunique()
                col = f"{event}_{win_name}_repos"
                X[col] = np.log1p(repo_counts.reindex(X.index).fillna(0).astype(float))
                feature_names.append(col)

                # 碰過的不同 Object 數量
                obj_counts = part.groupby("user_code")["object_id"].nunique()
                col = f"{event}_{win_name}_unique_objects"
                X[col] = np.log1p(obj_counts.reindex(X.index).fillna(0).astype(float))
                feature_names.append(col)
                
                # 專注度：最常互動的 repo 佔比 (Top-1 Ratio)
                if len(part):
                    top_repo_count = part.groupby(["user_code", "repo_id"]).size().groupby(level=0).max()
                    total_count = part.groupby("user_code").size()
                    ratio = top_repo_count / total_count
                    col_ratio = f"{event}_{win_name}_top_repo_ratio"
                    X[col_ratio] = ratio.reindex(X.index).fillna(0.0).astype(float)
                    feature_names.append(col_ratio)
                else:
                    col_ratio = f"{event}_{win_name}_top_repo_ratio"
                    X[col_ratio] = 0.0
                    feature_names.append(col_ratio)

        # --- [T] Temporal Features ---
        if use_T:
            for win_name, days in [("7d", 7), ("30d", 30)]:
                part = hist[hist["row_time"] >= cutoff - pd.Timedelta(days=days)]
                
                # 活躍天數 (穩定性通常高於單純計數)
                active_days = part.assign(date=part["row_time"].dt.date).groupby("user_code")["date"].nunique()
                col = f"{event}_{win_name}_active_days"
                X[col] = active_days.reindex(X.index).fillna(0).astype(float)
                feature_names.append(col)

            # 週末活動佔比 (30d)
            if len(hist):
                is_weekend = hist["row_time"].dt.dayofweek.isin([5, 6])
                weekend_counts = hist[is_weekend].groupby("user_code").size()
                total_counts = hist.groupby("user_code").size()
                weekend_ratio = weekend_counts / total_counts
                col = f"{event}_weekend_ratio"
                X[col] = weekend_ratio.reindex(X.index).fillna(0.0).astype(float)
            else:
                col = f"{event}_weekend_ratio"
                X[col] = 0.0
            feature_names.append(col)

            # 趨勢：7天與30天活動率的比值
            if len(hist):
                part_7d = hist[hist["row_time"] >= cutoff - pd.Timedelta(days=7)]
                count_7d = part_7d.groupby("user_code").size().reindex(X.index).fillna(0)
                count_30d = hist.groupby("user_code").size().reindex(X.index).fillna(0)
                # +1 smoothing
                trend = (count_7d + 1) / (count_30d + 1)
                col = f"{event}_trend_7_over_30"
                X[col] = trend.astype(float)
            else:
                col = f"{event}_trend_7_over_30"
                X[col] = 1.0
            feature_names.append(col)

    X = X.replace([np.inf, -np.inf], np.nan)
    if use_U:
        for event in EVENTS:
            col = f"{event}_days_since_last"
            # 視窗最大30天，所以補值用 30 即可
            X[col] = X[col].fillna(30.0).clip(0, 30.0)
            
    X = X.fillna(0).astype("float32")
    return X, feature_names

def build_labels(cohort_ids, label_frames, start, horizon_days=7):
    start = pd.Timestamp(start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    end = start + pd.Timedelta(days=horizon_days)
    y = pd.DataFrame(index=pd.Index(cohort_ids, name="UserID"))
    for event in EVENTS:
        df = label_frames[event]
        selected = df[(df["event_time"] >= start) & (df["event_time"] < end)]
        active = selected.groupby("user_code").size().gt(0)
        y[event] = active.reindex(y.index).fillna(False).astype("int8")
    return y
