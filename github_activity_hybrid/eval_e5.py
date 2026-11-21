import numpy as np
import torch
from train import MLP, filter_features, f1_scores
from src.common import EVENTS

def main():
    # 1. 載入驗證集與真實答案 (11/10 ~ 11/17)
    va = np.load("artifacts/valid.npz")
    X_raw = va["X"].astype(np.float32)
    y_valid = va["y"].astype(np.int8)
    names = va["feature_names"].tolist()

    # 2. 計算 Copy Last Week 預測
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

    # 3. 載入 E4 模型預測
    ckpt = torch.load("artifacts/model_E4.pt", map_location="cpu", weights_only=False)
    use_U, use_R, use_T = ckpt.get("use_U", True), ckpt.get("use_R", False), ckpt.get("use_T", False)
    
    X, _ = filter_features(X_raw, names, use_U, use_R, use_T)
    X = np.clip((X - ckpt["mean"]) / ckpt["std"], -8, 8)
    
    model = MLP(ckpt["input_dim"], len(EVENTS))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X))).numpy()
    nn_pred = (probs >= ckpt["thresholds"]).astype(np.int8)

    # 4. E5: 融合 (Baseline OR E4)
    ensemble_pred = np.logical_or(baseline_pred, nn_pred).astype(np.int8)

    # 5. 印出各組分數
    print("=== [對照組] Copy Last Week (Validation) ===")
    per, macro, micro, score = f1_scores(y_valid, baseline_pred)
    print(f"macro-F1={macro:.4f} micro-F1={micro:.4f} Score={score:.4f}\n")

    print("=== [E4] MLP (U+R+T) (Validation) ===")
    per, macro, micro, score = f1_scores(y_valid, nn_pred)
    print(f"macro-F1={macro:.4f} micro-F1={micro:.4f} Score={score:.4f}\n")

    print("=== [E5] Ensemble (E4 + Copy Last Week) (Validation) ===")
    per, macro, micro, score = f1_scores(y_valid, ensemble_pred)
    for name, f in zip(EVENTS, per):
        print(f"  {name:18s} F1={f:.4f}")
    print(f"\n>> macro-F1={macro:.4f} micro-F1={micro:.4f} Score={score:.4f} <<\n")

if __name__ == "__main__":
    main()