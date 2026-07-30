import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


TARGETS = ["PM2.5", "PM10", "aqi"]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class T7ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class T7ImageGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        projection_dim=512,
        hidden_dim=256,
        num_layers=2,
        dropout=0.25,
        output_dim=3,
    ):
        super().__init__()

        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.project(x)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def make_random_split(n, seed=42, train_frac=0.80, val_frac_of_train=0.15):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    test_size = int(round(n * (1 - train_frac)))
    test_idx = idx[:test_size]
    trainval_idx = idx[test_size:]

    val_size = int(round(len(trainval_idx) * val_frac_of_train))
    val_idx = trainval_idx[:val_size]
    train_idx = trainval_idx[val_size:]

    return train_idx, val_idx, test_idx


def build_arrays(seq_df, embeddings, T=7):
    X = []
    y = []
    keep_rows = []

    for i, r in seq_df.iterrows():
        start = int(r["seq_start_row_id"])
        end = int(r["seq_end_row_id"])
        ids = np.arange(start, end + 1)

        if len(ids) != T:
            continue

        if ids.min() < 0 or ids.max() >= len(embeddings):
            continue

        target_values = r[TARGETS].values.astype(float)

        if not np.isfinite(target_values).all():
            continue

        X.append(embeddings[ids])
        y.append(target_values)
        keep_rows.append(i)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    seq_kept = seq_df.loc[keep_rows].reset_index(drop=True)

    return X, y, seq_kept


def scale_arrays(X, y, train_idx):
    n, T, D = X.shape

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X[train_idx].reshape(-1, D))
    y_scaler.fit(y[train_idx])

    X_scaled = x_scaler.transform(X.reshape(-1, D)).reshape(n, T, D).astype(np.float32)
    y_scaled = y_scaler.transform(y).astype(np.float32)

    return X_scaled, y_scaled, x_scaler, y_scaler


def train_model(model, train_loader, val_loader, device, epochs, lr, weight_decay, patience):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No valid best state found.")

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history), best_epoch, best_val


def predict_model(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy())

    return np.vstack(preds)


def evaluate(y_true, y_pred):
    rows = []

    for i, target in enumerate(TARGETS):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        rows.append({
            "target": target,
            "R2": float(r2_score(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)),
        })

    rows.append({
        "target": "Average",
        "R2": float(np.mean([r["R2"] for r in rows])),
        "RMSE": float(np.mean([r["RMSE"] for r in rows])),
        "MAE": float(np.mean([r["MAE"] for r in rows])),
    })

    return rows


def make_prediction_df(seq_kept, idx, y_true, y_pred, split_name):
    key_cols = [
        "sequence_id",
        "target_row_id",
        "target_created_at",
        "seq_start_row_id",
        "seq_end_row_id",
    ]

    pred_df = seq_kept.iloc[idx][key_cols].copy()
    pred_df["split"] = split_name

    for i, target in enumerate(TARGETS):
        pred_df[f"actual_{target}"] = y_true[:, i]
        pred_df[f"predicted_{target}"] = y_pred[:, i]
        pred_df[f"residual_{target}"] = y_pred[:, i] - y_true[:, i]

    return pred_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence-table", required=True)
    parser.add_argument("--embedding-npy", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--T", type=int, default=7)
    parser.add_argument("--split-mode", choices=["random"], default="random")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac-of-train", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)

    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    seq_df = pd.read_csv(args.sequence_table)
    seq_df = seq_df.dropna(subset=TARGETS).reset_index(drop=True)

    embeddings = np.load(args.embedding_npy, mmap_mode="r")

    X, y, seq_kept = build_arrays(seq_df, embeddings, T=args.T)

    train_idx, val_idx, test_idx = make_random_split(
        len(seq_kept),
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac_of_train=args.val_frac_of_train,
    )

    X_scaled, y_scaled, x_scaler, y_scaler = scale_arrays(X, y, train_idx)

    train_ds = T7ImageDataset(X_scaled[train_idx], y_scaled[train_idx])
    val_ds = T7ImageDataset(X_scaled[val_idx], y_scaled[val_idx])
    test_ds = T7ImageDataset(X_scaled[test_idx], y_scaled[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = X_scaled.shape[2]

    model = T7ImageGRU(
        input_dim=input_dim,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=len(TARGETS),
    ).to(device)

    print("=" * 90)
    print("T7 IMAGE-ONLY RESNET50 FRONT-REAR CONCAT GRU")
    print("=" * 90)
    print("Sequence rows:", len(seq_kept))
    print("Train/Val/Test:", len(train_idx), len(val_idx), len(test_idx))
    print("Input dim:", input_dim)
    print("Targets:", TARGETS)
    print("=" * 90)

    model, history, best_epoch, best_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    test_pred_scaled = predict_model(model, test_loader, device)
    val_pred_scaled = predict_model(model, val_loader, device)

    test_pred = y_scaler.inverse_transform(test_pred_scaled)
    val_pred = y_scaler.inverse_transform(val_pred_scaled)

    y_test_true = y[test_idx]
    y_val_true = y[val_idx]

    metrics = pd.DataFrame(evaluate(y_test_true, test_pred))
    metrics["model"] = "t7_resnet50_front_rear_concat_gru_image_only"
    metrics["split_mode"] = args.split_mode
    metrics["train_rows"] = len(train_idx)
    metrics["val_rows"] = len(val_idx)
    metrics["test_rows"] = len(test_idx)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = best_val
    metrics["seed"] = args.seed
    metrics["input_dim"] = input_dim

    test_pred_df = make_prediction_df(seq_kept, test_idx, y_test_true, test_pred, "test")
    val_pred_df = make_prediction_df(seq_kept, val_idx, y_val_true, val_pred, "val")

    metrics_path = out_dir / "metrics.csv"
    test_pred_path = out_dir / "predictions.csv"
    val_pred_path = out_dir / "val_predictions.csv"
    hist_path = out_dir / "training_history.csv"
    config_path = out_dir / "config.json"
    split_path = out_dir / "split_indices.npz"

    metrics.to_csv(metrics_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)
    val_pred_df.to_csv(val_pred_path, index=False)
    history.to_csv(hist_path, index=False)

    np.savez(
        split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    config = vars(args)
    config["input_dim"] = int(input_dim)
    config["n_sequence_rows"] = int(len(seq_kept))
    config["targets"] = TARGETS

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("T7 IMAGE-ONLY GRU COMPLETE")
    print("=" * 90)
    print(metrics.to_string(index=False))
    print()
    print("Saved:", metrics_path)
    print("Saved:", test_pred_path)
    print("Saved:", val_pred_path)
    print("Saved:", hist_path)
    print("Saved:", config_path)
    print("Saved:", split_path)


if __name__ == "__main__":
    main()