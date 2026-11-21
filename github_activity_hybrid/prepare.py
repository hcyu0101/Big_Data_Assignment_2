from __future__ import annotations

import argparse
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd

from src.common import EVENTS, load_tables, load_cohort, make_user_map, to_datetime_columns
from src.features import make_event_frames, make_label_frames, build_features, build_labels

def generate_daily_cutoffs(start_date, end_date):
    """
    生成每日快照 cutoff，強制對齊台灣時間 00:00 (即 UTC 16:00)
    """
    start = pd.to_datetime(start_date).tz_localize('UTC')
    end = pd.to_datetime(end_date).tz_localize('UTC')
    
    cutoffs = []
    current = start
    while current <= end:
        cutoff = current.replace(hour=16, minute=0, second=0, microsecond=0)
        cutoffs.append(cutoff)
        current += timedelta(days=1)
        
    return [c.strftime('%Y-%m-%dT%H:%M:%SZ') for c in cutoffs]

def build_gru_sequence(cohort_ids, event_frames, cutoff, window_days=28):
    """
    為 GRU 建立時間序列特徵: shape = (num_users, window_days, num_events)
    每天計算每種 event 的數量 (加上 log1p)
    """
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
        
    start = cutoff - pd.Timedelta(days=window_days)
    num_users = len(cohort_ids)
    num_events = len(EVENTS)
    
    # 建立 user_id -> row_index 的對應表
    user_idx = {uid: i for i, uid in enumerate(cohort_ids)}
    
    # 初始化 (num_users, window_days, num_events) 全為 0 的張量
    seq_data = np.zeros((num_users, window_days, num_events), dtype=np.float32)
    
    for e_idx, event in enumerate(EVENTS):
        df = event_frames[event]
        # 只取視窗內的事件
        hist = df[(df["row_time"] >= start) & (df["row_time"] < cutoff)].copy()
        if len(hist) == 0:
            continue
            
        # 計算事件發生在哪一天 (相對於 start 的天數, 0 to window_days-1)
        hist["day_idx"] = (hist["row_time"] - start).dt.days
        
        # 每天每人的事件計數
        daily_counts = hist.groupby(["user_code", "day_idx"]).size().reset_index(name="count")
        
        for _, row in daily_counts.iterrows():
            uid = row["user_code"]
            d_idx = int(row["day_idx"])
            count = row["count"]
            if uid in user_idx and 0 <= d_idx < window_days:
                # 儲存 log1p 後的計數
                seq_data[user_idx[uid], d_idx, e_idx] = np.log1p(count)
                
    return seq_data

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db-dir", required=True, help="Folder containing the course preprocess.py output db/*.parquet")
    p.add_argument("--cohort-file", default="scripts/cohort.txt")
    p.add_argument("--data-start", default="2026-10-01", help="UTC date at which the downloaded data starts")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--train-start", default="2026-10-31")
    p.add_argument("--train-end", default="2026-11-02")
    p.add_argument("--valid-cutoffs", default="2026-11-03,2026-11-10")
    p.add_argument("--predict-cutoff", default="2026-11-21T16:00:00Z", help="對齊測試起點 11/22 的 cutoff")
    return p.parse_args()

def save_npz(path, X, seq, y, user_ids, feature_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X": X.to_numpy(dtype=np.float32),
        "seq": seq,  # 加入 GRU 的序列資料
        "user_ids": np.asarray(user_ids, dtype=np.int64),
        "feature_names": np.asarray(feature_names, dtype=str),
    }
    if y is not None:
        payload["y"] = y.to_numpy(dtype=np.int8)
    np.savez_compressed(path, **payload)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    cohort = load_cohort(args.cohort_file)
    print(f"Loaded cohort: {len(cohort):,} users")

    tables = to_datetime_columns(load_tables(args.db_dir))
    user_map = make_user_map(tables["Users"])
    event_frames = make_event_frames(tables, user_map)
    label_frames = make_label_frames(tables, user_map)

    # 每日快照 (stride=1 day)
    train_cutoffs = generate_daily_cutoffs(args.train_start, args.train_end)
    X_parts, seq_parts, y_parts, ids_parts = [], [], [], []
    feature_names = None

    for cutoff in train_cutoffs:
        X, names = build_features(cohort, event_frames, tables["Users"], cutoff, args.data_start, use_U=True, use_R=True, use_T=True)
        seq = build_gru_sequence(cohort, event_frames, cutoff, window_days=28)
        y = build_labels(cohort, label_frames, cutoff, horizon_days=7)
        feature_names = names
        
        X_parts.append(X.to_numpy(dtype=np.float32))
        seq_parts.append(seq)
        y_parts.append(y.to_numpy(dtype=np.int8))
        ids_parts.append(cohort)
        print(f"Train snapshot {cutoff}: X={X.shape}, seq={seq.shape}, positive labels={int(y.to_numpy().sum())}")
    
    X_train = np.concatenate(X_parts, axis=0)
    seq_train = np.concatenate(seq_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    save_npz(out_dir / "train.npz", pd.DataFrame(X_train), seq_train, None, np.concatenate(ids_parts), feature_names)
    np.savez_compressed(out_dir / "train_labels.npz", y=y_train)

    # 驗證
    valid_cutoffs = [generate_daily_cutoffs(x.strip(), x.strip())[0] for x in args.valid_cutoffs.split(",") if x.strip()]
    Xv_parts, seqv_parts, yv_parts, idsv_parts = [], [], [], []
    for cutoff in valid_cutoffs:
        Xv, names = build_features(cohort, event_frames, tables["Users"], cutoff, args.data_start, use_U=True, use_R=True, use_T=True)
        seqv = build_gru_sequence(cohort, event_frames, cutoff, window_days=28)
        yv = build_labels(cohort, label_frames, cutoff, horizon_days=7)
        
        Xv_parts.append(Xv.to_numpy(dtype=np.float32))
        seqv_parts.append(seqv)
        yv_parts.append(yv.to_numpy(dtype=np.int8))
        idsv_parts.append(cohort)
        print(f"Validation snapshot {cutoff}: X={Xv.shape}, seq={seqv.shape}, positive labels={int(yv.to_numpy().sum())}")
    
    X_valid = np.concatenate(Xv_parts, axis=0)
    seq_valid = np.concatenate(seqv_parts, axis=0)
    y_valid = np.concatenate(yv_parts, axis=0)
    save_npz(out_dir / "valid.npz", pd.DataFrame(X_valid), seq_valid, pd.DataFrame(y_valid), np.concatenate(idsv_parts), names)

    if args.predict_cutoff:
        Xp, names = build_features(cohort, event_frames, tables["Users"], args.predict_cutoff, args.data_start, use_U=True, use_R=True, use_T=True)
        seqp = build_gru_sequence(cohort, event_frames, args.predict_cutoff, window_days=28)
        save_npz(out_dir / "predict.npz", Xp, seqp, None, cohort, names)
        print(f"Prediction snapshot {args.predict_cutoff}: X={Xp.shape}, seq={seqp.shape}")

    print(f"Saved artifacts to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
