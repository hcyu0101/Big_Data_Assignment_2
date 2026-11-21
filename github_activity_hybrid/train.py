from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common import EVENTS

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, output_dim),
        )
    def forward(self, x, seq=None):
        return self.net(x)

class GRU_Hybrid(nn.Module):
    def __init__(self, input_dim, seq_dim, hidden_dim=64, output_dim=6):
        super().__init__()
        # 處理靜態統計特徵 (U, R, T)
        self.static_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 處理時間序列 (28天的 event counts)
        self.gru = nn.GRU(input_size=seq_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.gru_dropout = nn.Dropout(0.2)
        
        # 融合兩者
        self.classifier = nn.Sequential(
            nn.Linear(64 + hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, seq):
        # 1. 靜態特徵編碼
        static_feat = self.static_mlp(x)
        
        # 2. 序列特徵編碼 (取最後一個 time step 的 hidden state)
        # seq shape: (batch_size, window_days, seq_dim)
        gru_out, hidden = self.gru(seq) 
        # hidden shape: (num_layers, batch_size, hidden_dim) -> (batch_size, hidden_dim)
        seq_feat = self.gru_dropout(hidden[-1])
        
        # 3. Concatenate (類似 E5 GRU_CONCAT_AFTER_GRU 的設計)
        combined = torch.cat([static_feat, seq_feat], dim=1)
        return self.classifier(combined)

def f1_scores(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0)
    denom = 2 * tp + fp + fn
    per = np.divide(2 * tp, denom, out=np.ones_like(tp, dtype=float), where=denom != 0)
    micro_denom = 2 * tp.sum() + fp.sum() + fn.sum()
    micro = (2 * tp.sum() / micro_denom) if micro_denom else 1.0
    return per, float(per.mean()), float(micro), float((per.mean() + micro) / 2)

def tune_thresholds(y, probs):
    thresholds = []
    for j in range(y.shape[1]):
        best_t, best_f = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.025):
            pred = (probs[:, j] >= t).astype(np.int8)
            tp = np.sum((y[:, j] == 1) & (pred == 1))
            fp = np.sum((y[:, j] == 0) & (pred == 1))
            fn = np.sum((y[:, j] == 1) & (pred == 0))
            f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f > best_f:
                best_f, best_t = f, float(t)
        thresholds.append(best_t)
    return np.asarray(thresholds, dtype=np.float32)

def filter_features(X, feature_names, use_U, use_R, use_T):
    selected_indices = []
    selected_names = []
    
    for i, name in enumerate(feature_names):
        if name == "is_bot":
            selected_indices.append(i)
            selected_names.append(name)
            continue
            
        is_u = any(k in name for k in ["count_log", "decay", "days_since_last"])
        is_r = any(k in name for k in ["repos", "unique_objects", "top_repo_ratio"])
        is_t = any(k in name for k in ["active_days", "weekend_ratio", "trend_7_over_30"])
        
        keep = False
        if use_U and is_u: keep = True
        if use_R and is_r: keep = True
        if use_T and is_t: keep = True
        
        if keep:
            selected_indices.append(i)
            selected_names.append(name)
            
    return X[:, selected_indices], selected_names

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="artifacts/hybrid_mlp.pt")
    
    ap.add_argument("--model", default="mlp", choices=["mlp", "gru"])
    ap.add_argument("--use_U", type=str2bool, default=True)
    ap.add_argument("--use_R", type=str2bool, default=False)
    ap.add_argument("--use_T", type=str2bool, default=False)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base = Path(args.artifacts)
    tr = np.load(base / "train.npz")
    tr_y = np.load(base / "train_labels.npz")["y"].astype(np.float32)
    va = np.load(base / "valid.npz")
    
    X_train_raw = tr["X"].astype(np.float32)
    seq_train = tr["seq"].astype(np.float32)
    y_train = tr_y
    
    X_valid_raw = va["X"].astype(np.float32)
    seq_valid = va["seq"].astype(np.float32)
    y_valid = va["y"].astype(np.float32)
    
    feature_names_raw = tr["feature_names"].tolist()

    X_train, feature_names = filter_features(X_train_raw, feature_names_raw, args.use_U, args.use_R, args.use_T)
    X_valid, _ = filter_features(X_valid_raw, feature_names_raw, args.use_U, args.use_R, args.use_T)
    print(f"Using {len(feature_names)} features for static MLP component.")

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0
    X_train = np.clip((X_train - mean) / std, -8, 8)
    X_valid = np.clip((X_valid - mean) / std, -8, 8)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(seq_train), torch.from_numpy(y_train))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.model == "gru":
        print("Initializing GRU_Hybrid model...")
        model = GRU_Hybrid(input_dim=X_train.shape[1], seq_dim=seq_train.shape[2], output_dim=len(EVENTS)).to(device)
    else:
        print("Initializing MLP model...")
        model = MLP(input_dim=X_train.shape[1], output_dim=len(EVENTS)).to(device)
        
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    pos_weight = torch.tensor(np.clip(negatives / np.maximum(positives, 1), 1.0, 30.0),
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    xv = torch.from_numpy(X_valid).to(device)
    seqv = torch.from_numpy(seq_valid).to(device)
    yv = torch.from_numpy(y_valid).to(device)
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, seqb, yb in loader:
            xb, seqb, yb = xb.to(device), seqb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if args.model == "gru":
                logits = model(xb, seqb)
            else:
                logits = model(xb)
                
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            if args.model == "gru":
                val_loss = float(criterion(model(xv, seqv), yv).cpu())
            else:
                val_loss = float(criterion(model(xv), yv).cpu())
                
        train_loss = float(np.mean(losses))
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if args.model == "gru":
            probs = torch.sigmoid(model(xv, seqv)).cpu().numpy()
        else:
            probs = torch.sigmoid(model(xv)).cpu().numpy()
            
    thresholds = tune_thresholds(y_valid.astype(np.int8), probs)
    pred = (probs >= thresholds).astype(np.int8)
    per, macro, micro, score = f1_scores(y_valid.astype(np.int8), pred)

    print("Validation results (thresholds tuned on validation):")
    for name, f, t in zip(EVENTS, per, thresholds):
        print(f"  {name:18s} F1={f:.4f} threshold={t:.3f}")
    print(f"  macro-F1={macro:.4f} micro-F1={micro:.4f} Score={score:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "thresholds": thresholds,
        "feature_names": feature_names,
        "events": EVENTS,
        "input_dim": X_train.shape[1],
        "seq_dim": seq_train.shape[2] if args.model == "gru" else 0,
        "model_type": args.model,
        "use_U": args.use_U,
        "use_R": args.use_R,
        "use_T": args.use_T,
    }, output)
    print("Saved:", output.resolve())

if __name__ == "__main__":
    main()
