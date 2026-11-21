from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from train import MLP, GRU_Hybrid, filter_features
from src.common import EVENTS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/predict.npz")
    ap.add_argument("--checkpoint", default="artifacts/hybrid_mlp.pt")
    ap.add_argument("--test-input", default="submission/test_input.csv")
    ap.add_argument("--output", default="output_hybrid.csv")
    ap.add_argument("--model", default="mlp", choices=["mlp", "gru", "copy_last_week", "ensemble"])
    args = ap.parse_args()

    data = np.load(args.features)
    user_ids = data["user_ids"].astype(np.int64)
    names = data["feature_names"].tolist()
    
    # 1. 永遠先準備 Baseline 預測 (Copy Last Week)
    X_raw = data["X"]
    baseline_pred = []
    for event in EVENTS:
        col_name = f"{event}_7d_count_log"
        if col_name in names:
            idx = names.index(col_name)
            active = (X_raw[:, idx] > 0).astype(np.int8)
            baseline_pred.append(active)
        else:
            baseline_pred.append(np.zeros(len(X_raw), dtype=np.int8))
    baseline_pred = np.column_stack(baseline_pred)

    if args.model == "copy_last_week":
        print("Running Copy Last Week baseline...")
        pred = baseline_pred
    else:
        # 2. 準備神經網路預測 (載入 checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        X_model = data["X"].astype(np.float32)
        seq_model = data.get("seq")
        
        if seq_model is not None:
             seq_model = seq_model.astype(np.float32)
        
        use_U = ckpt.get("use_U", True)
        use_R = ckpt.get("use_R", False)
        use_T = ckpt.get("use_T", False)
        model_type = ckpt.get("model_type", "mlp")
        
        # 允許覆蓋 checkpoint 中的 model_type 以支援 ensemble 呼叫 GRU
        if args.model != "ensemble":
            if model_type != args.model:
                 print(f"Warning: Checkpoint was trained as '{model_type}', but predicting as '{args.model}'")
        
        X, model_names = filter_features(X_model, names, use_U, use_R, use_T)

        if model_names != ckpt["feature_names"]:
            raise ValueError("Feature columns differ from training. Re-run prepare.py with the same code.")

        X = np.clip((X - ckpt["mean"]) / ckpt["std"], -8, 8)
        
        if model_type == "gru":
            if seq_model is None:
                 raise ValueError("GRU model requires sequence features, but 'seq' is not in predict.npz")
            model = GRU_Hybrid(input_dim=ckpt["input_dim"], seq_dim=ckpt["seq_dim"], output_dim=len(EVENTS))
        else:
            model = MLP(ckpt["input_dim"], len(EVENTS))
            
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            if model_type == "gru":
                probs = torch.sigmoid(model(torch.from_numpy(X), torch.from_numpy(seq_model))).numpy()
            else:
                probs = torch.sigmoid(model(torch.from_numpy(X))).numpy()
        nn_pred = (probs >= ckpt["thresholds"]).astype(np.int8)

        # 3. 判斷是否為 Ensemble (E5)
        if args.model == "ensemble":
            print(f"Running E5 Ensemble (Copy Last Week OR {model_type.upper()})...")
            # 融合：上週有做(Baseline=1) 或 模型預測會做(NN=1)，就預測為1
            pred = np.logical_or(baseline_pred, nn_pred).astype(np.int8)
        else:
            print(f"Running normal {model_type.upper()}...")
            pred = nn_pred

    # 4. 整理並輸出檔案
    result = pd.DataFrame(pred, columns=EVENTS)
    result.insert(0, "UserID", user_ids)
    test_ids = pd.read_csv(args.test_input)["UserID"].astype("int64")
    result = result.set_index("UserID").reindex(test_ids).reset_index()
    result[EVENTS] = result[EVENTS].fillna(0).astype("int8")
    result.to_csv(args.output, index=False)
    print(f"Saved {args.output} ({len(result)} users)")
    print("Check format with:")
    print(f"python scripts/evaluate.py --pred {args.output} --input {args.test_input}")

if __name__ == "__main__":
    main()
