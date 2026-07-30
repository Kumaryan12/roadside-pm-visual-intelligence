import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader


TARGET = "PM2.5"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ImageEmbeddingDataset(Dataset):
    def __init__(self, df, embeddings, y_scaled=None):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.y_scaled = y_scaled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row_id = int(row["row_id"])

        x = self.embeddings[row_id].astype(np.float32)

        if self.y_scaled is None:
            y = np.array([0.0], dtype=np.float32)
        else:
            y = np.array([self.y_scaled[idx]], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


class SingleImageMLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, dropout=0.25):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_split(df, split_mode, train_frac, val_frac, seed):
    df = df.copy()

    if split_mode == "random":
        idx = np.arange(len(df))

        trainval_idx, test_idx = train_test_split(
            idx,
            train_size=train_frac + val_frac,
            random_state=seed,
            shuffle=True,
        )

        relative_val_frac = val_frac / (train_frac + val_frac)

        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=relative_val_frac,
            random_state=seed,
            shuffle=True,
        )

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        return train_df, val_df, test_df

    if split_mode == "chrono":
        df = df.sort_values("created_at").reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)

        return train_df, val_df, test_df

    raise ValueError(f"Unknown split_mode: {split_mode}")


def predict(model, loader, y_scaler, device):
    model.eval()
    preds_scaled = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds_scaled.append(pred)

    preds_scaled = np.concatenate(preds_scaled).reshape(-1, 1)
    preds = y_scaler.inverse_transform(preds_scaled).reshape(-1)

    return preds


def compute_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )
    parser.add_argument(
        "--embedding-npy",
        default="experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/traqid_paper_resnet50_front_rear_concat_gap_features.npy",
    )
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--split-mode", choices=["random", "chrono"], default="random")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    df = pd.read_csv(args.manifest)

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    if "created_at" not in df.columns:
        raise ValueError("created_at column missing from manifest.")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at", TARGET, "row_id"]).copy()
    df["row_id"] = df["row_id"].astype(int)

    embeddings = np.load(args.embedding_npy, mmap_mode="r")
    print("Embedding shape:", embeddings.shape)

    df = df[df["row_id"] < embeddings.shape[0]].copy()
    df = df.reset_index(drop=True)

    train_df, val_df, test_df = make_split(
        df,
        split_mode=args.split_mode,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Test rows:", len(test_df))

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(train_df[[TARGET]].values).reshape(-1)
    y_val_scaled = y_scaler.transform(val_df[[TARGET]].values).reshape(-1)

    train_ds = ImageEmbeddingDataset(train_df, embeddings, y_train_scaled)
    val_ds = ImageEmbeddingDataset(val_df, embeddings, y_val_scaled)
    test_ds = ImageEmbeddingDataset(test_df, embeddings, None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SingleImageMLP(
        input_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.squeeze(-1).to(device)

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
                yb = yb.squeeze(-1).to(device)
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

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_left = args.patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    val_pred = predict(model, val_loader, y_scaler, device)
    test_pred = predict(model, test_loader, y_scaler, device)

    val_actual = val_df[TARGET].values.astype(float)
    test_actual = test_df[TARGET].values.astype(float)

    val_metrics = compute_metrics(val_actual, val_pred)
    test_metrics = compute_metrics(test_actual, test_pred)

    metrics_df = pd.DataFrame([
        {
            "split": "val",
            "target": TARGET,
            **val_metrics,
            "model": "single_image_resnet50_front_rear_mlp",
            "split_mode": args.split_mode,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
        {
            "split": "test",
            "target": TARGET,
            **test_metrics,
            "model": "single_image_resnet50_front_rear_mlp",
            "split_mode": args.split_mode,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
    ])

    val_predictions = val_df[["row_id", "created_at", TARGET]].copy()
    val_predictions = val_predictions.rename(columns={TARGET: "actual_PM2.5"})
    val_predictions["predicted_PM2.5"] = val_pred
    val_predictions["split"] = "val"

    test_predictions = test_df[["row_id", "created_at", TARGET]].copy()
    test_predictions = test_predictions.rename(columns={TARGET: "actual_PM2.5"})
    test_predictions["predicted_PM2.5"] = test_pred
    test_predictions["split"] = "test"

    metrics_path = out_dir / "metrics.csv"
    val_pred_path = out_dir / "val_predictions.csv"
    test_pred_path = out_dir / "predictions.csv"
    history_path = out_dir / "training_history.csv"
    config_path = out_dir / "config.json"

    metrics_df.to_csv(metrics_path, index=False)
    val_predictions.to_csv(val_pred_path, index=False)
    test_predictions.to_csv(test_pred_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SINGLE-IMAGE PM2.5 RESNET50-MLP COMPLETE")
    print("=" * 90)
    print(metrics_df.to_string(index=False))
    print()
    print("Saved:", metrics_path)
    print("Saved:", val_pred_path)
    print("Saved:", test_pred_path)
    print("Saved:", history_path)
    print("Saved:", config_path)


if __name__ == "__main__":
    main()