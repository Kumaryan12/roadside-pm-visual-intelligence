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


TARGETS = ["PM2.5", "PM10", "aqi"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class T7EmbeddingDataset(Dataset):
    def __init__(self, seq_df, embedding_array, y_scaled):
        self.seq_df = seq_df.reset_index(drop=True)
        self.embedding_array = embedding_array
        self.y_scaled = y_scaled.astype(np.float32)

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]

        start_id = int(row["seq_start_row_id"])
        end_id = int(row["seq_end_row_id"])

        ids = np.arange(start_id, end_id + 1)

        if len(ids) != 7:
            raise ValueError(f"Expected T=7 but got {len(ids)} for idx={idx}")

        x = self.embedding_array[ids].astype(np.float32)
        y = self.y_scaled[idx]

        return torch.from_numpy(x), torch.from_numpy(y)


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=3,
        projection_dim=512,
        hidden_dim=256,
        num_layers=2,
        dropout=0.25,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.projector = nn.Sequential(
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
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.projector(x)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def compute_metrics(y_true, y_pred):
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

    return pd.DataFrame(rows)


def make_split(df, split_mode, seed, train_frac):
    if split_mode == "random":
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            train_size=train_frac,
            random_state=seed,
            shuffle=True,
        )

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        train_df["eval_split"] = "train"
        test_df["eval_split"] = "test"

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    if split_mode == "chrono":
        if "target_split_date_chrono" not in df.columns:
            raise ValueError("target_split_date_chrono column missing for chrono split.")

        train_df = df[df["target_split_date_chrono"].isin(["train", "val"])].copy()
        test_df = df[df["target_split_date_chrono"] == "test"].copy()

        train_df["eval_split"] = "train"
        test_df["eval_split"] = "test"

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    raise ValueError(f"Unknown split_mode: {split_mode}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-table",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_engineered_T7_sequence_table.csv",
    )
    parser.add_argument(
        "--embedding-npy",
        default="experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/traqid_paper_resnet50_front_rear_concat_gap_features.npy",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/t7_resnet50_concat_gru_paper_style_random",
    )

    parser.add_argument("--split-mode", choices=["random", "chrono"], default="random")
    parser.add_argument("--train-frac", type=float, default=0.8)
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    seq = pd.read_csv(args.sequence_table)
    seq = seq.dropna(subset=TARGETS).copy()
    seq = seq[seq["T"] == 7].copy() if "T" in seq.columns else seq.copy()
    seq = seq.reset_index(drop=True)

    embeddings = np.load(args.embedding_npy, mmap_mode="r")
    print("Embedding shape:", embeddings.shape)

    max_needed_row = int(seq["seq_end_row_id"].max())
    if max_needed_row >= embeddings.shape[0]:
        raise ValueError(
            f"Sequence table needs row {max_needed_row}, but embedding array has only {embeddings.shape[0]} rows."
        )

    train_df, test_df = make_split(seq, args.split_mode, args.seed, args.train_frac)

    # internal validation from train only
    train_inner, val_inner = train_test_split(
        train_df,
        test_size=0.15,
        random_state=args.seed,
        shuffle=True,
    )

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(train_inner[TARGETS].values)
    y_val_scaled = y_scaler.transform(val_inner[TARGETS].values)
    y_test_scaled = y_scaler.transform(test_df[TARGETS].values)

    train_ds = T7EmbeddingDataset(train_inner, embeddings, y_train_scaled)
    val_ds = T7EmbeddingDataset(val_inner, embeddings, y_val_scaled)
    test_ds = T7EmbeddingDataset(test_df, embeddings, y_test_scaled)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    input_dim = embeddings.shape[1]

    model = GRURegressor(
        input_dim=input_dim,
        output_dim=len(TARGETS),
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.MSELoss()

    history = []
    best_val = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
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

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(float(loss.item()))

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
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
            patience_left = args.patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            print("Early stopping.")
            break

    model.load_state_dict(best_state["model"])

    preds_scaled = []
    true_scaled = []

    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()

            preds_scaled.append(pred)
            true_scaled.append(yb.numpy())

    preds_scaled = np.vstack(preds_scaled)
    true_scaled = np.vstack(true_scaled)

    preds = y_scaler.inverse_transform(preds_scaled)
    y_true = test_df[TARGETS].values

    metrics = compute_metrics(y_true, preds)

    predictions = test_df[
        [
            "sequence_id",
            "target_row_id",
            "target_created_at",
            "seq_start_row_id",
            "seq_end_row_id",
        ]
    ].copy()

    predictions["split_mode"] = args.split_mode

    for i, target in enumerate(TARGETS):
        predictions[f"actual_{target}"] = y_true[:, i]
        predictions[f"predicted_{target}"] = preds[:, i]

    metrics["split_mode"] = args.split_mode
    metrics["model"] = "resnet50_front_rear_concat_gru"
    metrics["embedding_file"] = args.embedding_npy
    metrics["train_rows"] = len(train_inner)
    metrics["val_rows"] = len(val_inner)
    metrics["test_rows"] = len(test_df)
    metrics["best_epoch"] = best_state["epoch"]
    metrics["best_val_loss"] = best_state["val_loss"]

    metrics_path = out_dir / "metrics.csv"
    predictions_path = out_dir / "predictions.csv"
    history_path = out_dir / "training_history.csv"
    config_path = out_dir / "config.json"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 90)
    print("TRAQID T7 RESNET50 GRU PAPER-STYLE COMPLETE")
    print("=" * 90)
    print("Sequence table:", args.sequence_table)
    print("Embedding:", args.embedding_npy)
    print("Split mode:", args.split_mode)
    print("Train rows:", len(train_inner))
    print("Val rows:", len(val_inner))
    print("Test rows:", len(test_df))
    print()
    print(metrics.to_string(index=False))
    print()
    print("Saved:", metrics_path)
    print("Saved:", predictions_path)
    print("Saved:", history_path)


if __name__ == "__main__":
    main()