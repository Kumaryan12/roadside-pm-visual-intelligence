from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer: str) -> torch.device:
    prefer = prefer.lower()

    if prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("WARNING: MPS requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError("device must be auto, mps, or cpu")


def parse_index_sequence(s: str) -> list[int]:
    return [int(x) for x in str(s).split("|") if str(x).strip()]


def encode_target(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if mode == "raw":
        return y

    if mode == "log1p":
        return np.log1p(y)

    raise ValueError(mode)


def inverse_target(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if mode == "raw":
        return y

    if mode == "log1p":
        return np.expm1(y)

    raise ValueError(mode)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)

    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        out["Pearson"] = float(pearsonr(y_true, y_pred).statistic)
        out["Spearman"] = float(spearmanr(y_true, y_pred).statistic)
    else:
        out["Pearson"] = float("nan")
        out["Spearman"] = float("nan")

    return out


class T7EmbeddingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        target_mode: str,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.target_mode = target_mode
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        self.seq_indices = [
            parse_index_sequence(s)
            for s in self.df["seq_embedding_indices"].tolist()
        ]

        self.y_raw = self.df["target_value"].astype(float).to_numpy(dtype=np.float32)
        self.y_model = encode_target(self.y_raw, target_mode).astype(np.float32)

        self.sequence_ids = self.df["sequence_id"].astype(int).to_numpy()
        self.target_row_ids = self.df["target_row_id"].astype(int).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idxs = self.seq_indices[idx]
        x = np.asarray(self.embeddings[idxs], dtype=np.float32)

        x = (x - self.mean) / self.std

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor([self.y_model[idx]], dtype=torch.float32),
            "y_raw": torch.tensor([self.y_raw[idx]], dtype=torch.float32),
            "sequence_id": torch.tensor(self.sequence_ids[idx], dtype=torch.long),
            "target_row_id": torch.tensor(self.target_row_ids[idx], dtype=torch.long),
        }


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, h = self.gru(x)
        last = out[:, -1, :]
        y = self.head(last)
        return y


def compute_train_scaler(train_df: pd.DataFrame, embeddings: np.ndarray):
    used = set()

    for s in train_df["seq_embedding_indices"]:
        used.update(parse_index_sequence(s))

    used = sorted(used)

    x = np.asarray(embeddings[used], dtype=np.float32)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.inference_mode()
def predict(model, loader, device, target_mode: str):
    model.eval()

    preds_model = []
    actual_raw = []
    sequence_ids = []
    target_row_ids = []

    for batch in loader:
        x = batch["x"].to(device)

        pred = model(x).detach().cpu().numpy().reshape(-1)
        y_raw = batch["y_raw"].cpu().numpy().reshape(-1)

        preds_model.append(pred)
        actual_raw.append(y_raw)
        sequence_ids.append(batch["sequence_id"].cpu().numpy().reshape(-1))
        target_row_ids.append(batch["target_row_id"].cpu().numpy().reshape(-1))

    preds_model = np.concatenate(preds_model)
    actual_raw = np.concatenate(actual_raw)
    sequence_ids = np.concatenate(sequence_ids)
    target_row_ids = np.concatenate(target_row_ids)

    preds_raw = inverse_target(preds_model, target_mode)
    preds_raw = np.maximum(preds_raw, 0)

    return actual_raw, preds_raw, sequence_ids, target_row_ids


def save_scatter(y_true, y_pred, out_path: Path, title: str):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)

    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.title(title)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_T7_front_sequence_manifest.csv",
    )

    parser.add_argument(
        "--embeddings",
        default="experiments/traqid_pretraining_v1/embeddings/traqid_mobilenetv2_torch_front_embeddings.npy",
    )

    parser.add_argument("--target-mode", default="log1p", choices=["raw", "log1p"])
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/models/t7_embedding_gru",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/t7_embedding_gru",
    )

    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures/t7_embedding_gru",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    sequence_manifest = Path(args.sequence_manifest)
    embeddings_path = Path(args.embeddings)

    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    fig_dir = Path(args.fig_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seq_df = pd.read_csv(sequence_manifest)
    embeddings = np.load(embeddings_path, mmap_mode="r")

    required = [
        "sequence_id",
        "split_date_chrono",
        "seq_embedding_indices",
        "target_value",
        "target_row_id",
    ]

    missing = [c for c in required if c not in seq_df.columns]
    if missing:
        raise ValueError(f"Sequence manifest missing columns: {missing}")

    train_df = seq_df[seq_df["split_date_chrono"] == "train"].copy()
    val_df = seq_df[seq_df["split_date_chrono"] == "val"].copy()
    test_df = seq_df[seq_df["split_date_chrono"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"Empty split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    device = get_device(args.device)

    print("=" * 90)
    print("TRAQID T=7 EMBEDDING GRU TRAINING")
    print("=" * 90)
    print("Sequence manifest:", sequence_manifest)
    print("Embeddings:", embeddings_path)
    print("Embedding shape:", embeddings.shape)
    print("Sequences:", len(seq_df))
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    print("Target mode:", args.target_mode)
    print("Device:", device)

    print("\nSplit dates:")
    print("train:", sorted(train_df["date"].unique().tolist()) if "date" in train_df.columns else "")
    print("val  :", sorted(val_df["date"].unique().tolist()) if "date" in val_df.columns else "")
    print("test :", sorted(test_df["date"].unique().tolist()) if "date" in test_df.columns else "")

    # Mean baseline.
    y_train = train_df["target_value"].astype(float).to_numpy()
    y_val = val_df["target_value"].astype(float).to_numpy()
    y_test = test_df["target_value"].astype(float).to_numpy()

    if args.target_mode == "log1p":
        mean_model_value = np.log1p(y_train).mean()
        val_mean_pred = np.expm1(np.full_like(y_val, mean_model_value, dtype=float))
        test_mean_pred = np.expm1(np.full_like(y_test, mean_model_value, dtype=float))
    else:
        mean_model_value = y_train.mean()
        val_mean_pred = np.full_like(y_val, mean_model_value, dtype=float)
        test_mean_pred = np.full_like(y_test, mean_model_value, dtype=float)

    baseline = {
        "val": compute_metrics(y_val, val_mean_pred),
        "test": compute_metrics(y_test, test_mean_pred),
    }

    print("\nMean baseline:")
    print(json.dumps(baseline, indent=2))

    print("\nComputing train scaler...")
    mean, std = compute_train_scaler(train_df, embeddings)

    train_ds = T7EmbeddingDataset(train_df, embeddings, args.target_mode, mean, std)
    val_ds = T7EmbeddingDataset(val_df, embeddings, args.target_mode, mean, std)
    test_ds = T7EmbeddingDataset(test_df, embeddings, args.target_mode, mean, std)

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
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_path = out_dir / "best_t7_embedding_gru.pt"

    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_actual, val_pred, _, _ = predict(model, val_loader, device, args.target_mode)
        test_actual, test_pred, _, _ = predict(model, test_loader, device, args.target_mode)

        val_metrics = compute_metrics(val_actual, val_pred)
        test_metrics = compute_metrics(test_actual, test_pred)

        row = {
            "epoch": epoch,
            "train_loss": loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }

        history.append(row)

        print(
            f"[epoch {epoch:02d}/{args.epochs}] "
            f"loss={loss:.4f} | "
            f"VAL RMSE={val_metrics['RMSE']:.3f}, Spearman={val_metrics['Spearman']:.3f} | "
            f"TEST RMSE={test_metrics['RMSE']:.3f}, Spearman={test_metrics['Spearman']:.3f}"
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "baseline": baseline,
                    "embedding_mean": mean,
                    "embedding_std": std,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_actual, val_pred, val_seq_ids, val_row_ids = predict(model, val_loader, device, args.target_mode)
    test_actual, test_pred, test_seq_ids, test_row_ids = predict(model, test_loader, device, args.target_mode)

    final_metrics = {
        "baseline": baseline,
        "best_checkpoint": {
            "path": str(best_path),
            "epoch": checkpoint["epoch"],
        },
        "val": compute_metrics(val_actual, val_pred),
        "test": compute_metrics(test_actual, test_pred),
    }

    print("\nFINAL BEST MODEL METRICS:")
    print(json.dumps(final_metrics, indent=2))

    history_path = report_dir / "training_history_t7_embedding_gru.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    pred_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "sequence_id": val_seq_ids,
                    "target_row_id": val_row_ids,
                    "split": "val",
                    "actual_PM25": val_actual,
                    "predicted_PM25": val_pred,
                }
            ),
            pd.DataFrame(
                {
                    "sequence_id": test_seq_ids,
                    "target_row_id": test_row_ids,
                    "split": "test",
                    "actual_PM25": test_actual,
                    "predicted_PM25": test_pred,
                }
            ),
        ],
        ignore_index=True,
    )

    predictions_path = report_dir / "predictions_t7_embedding_gru.csv"
    pred_df.to_csv(predictions_path, index=False)

    metrics_path = report_dir / "metrics_t7_embedding_gru.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    config_path = report_dir / "config_t7_embedding_gru.json"
    config = {
        "args": vars(args),
        "sequence_manifest": str(sequence_manifest),
        "embeddings": str(embeddings_path),
        "embedding_shape": list(embeddings.shape),
        "rows": {
            "sequences": int(len(seq_df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    save_scatter(
        val_actual,
        val_pred,
        fig_dir / "scatter_val_t7_embedding_gru.png",
        "TRAQID T=7 GRU VAL",
    )

    save_scatter(
        test_actual,
        test_pred,
        fig_dir / "scatter_test_t7_embedding_gru.png",
        "TRAQID T=7 GRU TEST",
    )

    print("\nSaved:")
    print(" - best model:", best_path)
    print(" - history:", history_path)
    print(" - predictions:", predictions_path)
    print(" - metrics:", metrics_path)
    print(" - config:", config_path)
    print(" - figures:", fig_dir)


if __name__ == "__main__":
    main()