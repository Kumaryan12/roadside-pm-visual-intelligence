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
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError("device must be auto, mps, or cpu")


def parse_ids(s: str):
    return [int(x) for x in str(s).split("|") if str(x).strip()]


def encode_target(y: np.ndarray, mode: str):
    y = np.asarray(y, dtype=np.float32)

    if mode == "raw":
        return y

    if mode == "log1p":
        return np.log1p(y)

    raise ValueError(mode)


def inverse_target(y: np.ndarray, mode: str):
    y = np.asarray(y, dtype=np.float32)

    if mode == "raw":
        return y

    if mode == "log1p":
        return np.expm1(y)

    raise ValueError(mode)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "finite_fraction": float(finite.mean()),
    }

    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        out["Pearson"] = float(pearsonr(y_true, y_pred).statistic)
        out["Spearman"] = float(spearmanr(y_true, y_pred).statistic)
    else:
        out["Pearson"] = float("nan")
        out["Spearman"] = float("nan")

    return out


def compute_embedding_scaler(train_df, embeddings):
    used = set()

    for s in train_df["seq_row_ids"]:
        used.update(parse_ids(s))

    used = sorted(used)
    x = np.asarray(embeddings[used], dtype=np.float32)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


class HVAQSequenceDataset(Dataset):
    def __init__(self, df, embeddings, target_mode, emb_mean, emb_std):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.target_mode = target_mode
        self.emb_mean = emb_mean.astype(np.float32)
        self.emb_std = emb_std.astype(np.float32)

        self.seq_row_ids = [parse_ids(s) for s in self.df["seq_row_ids"].tolist()]
        self.y_raw = self.df["target_PM2.5"].astype(float).to_numpy(dtype=np.float32)
        self.y_model = encode_target(self.y_raw, target_mode).astype(np.float32)
        self.sequence_ids = self.df["sequence_id"].astype(int).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ids = self.seq_row_ids[idx]
        x = np.asarray(self.embeddings[ids], dtype=np.float32)
        x = (x - self.emb_mean) / self.emb_std

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor([self.y_model[idx]], dtype=torch.float32),
            "y_raw": torch.tensor([self.y_raw[idx]], dtype=torch.float32),
            "sequence_id": torch.tensor(self.sequence_ids[idx], dtype=torch.long),
        }


class VGG16LSTMRegressor(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm):
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

        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.inference_mode()
def predict(model, loader, device, target_mode):
    model.eval()

    preds_model = []
    actual_raw = []
    sequence_ids = []

    for batch in loader:
        x = batch["x"].to(device)
        pred_model = model(x).detach().cpu().numpy().reshape(-1)

        preds_model.append(pred_model)
        actual_raw.append(batch["y_raw"].cpu().numpy().reshape(-1))
        sequence_ids.append(batch["sequence_id"].cpu().numpy().reshape(-1))

    preds_model = np.concatenate(preds_model)
    actual_raw = np.concatenate(actual_raw)
    sequence_ids = np.concatenate(sequence_ids)

    preds_raw = inverse_target(preds_model, target_mode)
    preds_raw = np.maximum(preds_raw, 0)

    return actual_raw, preds_raw, sequence_ids


def save_scatter(y_true, y_pred, path, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.45)

    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title(title)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-manifest",
        default="experiments/hvaq_vgg16_lstm_v1/data/processed/hvaq_T7_sequence_manifest.csv",
    )
    parser.add_argument(
        "--embeddings",
        default="experiments/hvaq_vgg16_lstm_v1/embeddings/hvaq_vgg16_avgpool512_row_embeddings.npy",
    )

    parser.add_argument(
        "--split-col",
        default="split_random",
        choices=["split_random", "split_purged_block", "split_chrono_date"],
    )

    parser.add_argument("--target-mode", default="log1p", choices=["raw", "log1p"])
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--fig-dir", required=True)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    fig_dir = Path(args.fig_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seq_df = pd.read_csv(args.sequence_manifest)
    seq_df.columns = [str(c).strip() for c in seq_df.columns]

    if args.split_col not in seq_df.columns:
        raise ValueError(f"Missing split column: {args.split_col}")

    seq_df = seq_df[seq_df[args.split_col].isin(["train", "val", "test"])].copy()

    train_df = seq_df[seq_df[args.split_col] == "train"].copy()
    val_df = seq_df[seq_df[args.split_col] == "val"].copy()
    test_df = seq_df[seq_df[args.split_col] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"Empty split using {args.split_col}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    embeddings = np.load(args.embeddings, mmap_mode="r")
    device = get_device(args.device)

    print("=" * 90)
    print("HVAQ T=7 VGG16 + LSTM")
    print("=" * 90)
    print("Sequence manifest:", args.sequence_manifest)
    print("Embeddings:", args.embeddings)
    print("Embedding shape:", embeddings.shape)
    print("Split column:", args.split_col)
    print("Sequences:", len(seq_df))
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    print("Target mode:", args.target_mode)
    print("Device:", device)

    print("\nSplit date counts:")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(name)
        print(part["date"].value_counts().sort_index())

    y_train = train_df["target_PM2.5"].astype(float).to_numpy()
    y_val = val_df["target_PM2.5"].astype(float).to_numpy()
    y_test = test_df["target_PM2.5"].astype(float).to_numpy()

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

    print("\nComputing embedding scaler from train split...")
    emb_mean, emb_std = compute_embedding_scaler(train_df, embeddings)

    train_ds = HVAQSequenceDataset(train_df, embeddings, args.target_mode, emb_mean, emb_std)
    val_ds = HVAQSequenceDataset(val_df, embeddings, args.target_mode, emb_mean, emb_std)
    test_ds = HVAQSequenceDataset(test_df, embeddings, args.target_mode, emb_mean, emb_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = VGG16LSTMRegressor(
        embedding_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_path = out_dir / "best_hvaq_t7_vgg16_lstm.pt"

    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip_norm=args.grad_clip_norm,
        )

        val_actual, val_pred, _ = predict(model, val_loader, device, args.target_mode)
        test_actual, test_pred, _ = predict(model, test_loader, device, args.target_mode)

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
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"loss={loss:.4f} | "
            f"VAL RMSE={val_metrics['RMSE']:.3f}, R2={val_metrics['R2']:.3f}, Spearman={val_metrics['Spearman']:.3f} | "
            f"TEST RMSE={test_metrics['RMSE']:.3f}, R2={test_metrics['R2']:.3f}, Spearman={test_metrics['Spearman']:.3f}"
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "baseline": baseline,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "emb_mean": emb_mean,
                    "emb_std": emb_std,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_actual, val_pred, val_ids = predict(model, val_loader, device, args.target_mode)
    test_actual, test_pred, test_ids = predict(model, test_loader, device, args.target_mode)

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

    history_path = report_dir / "training_history_hvaq_t7_vgg16_lstm.csv"
    metrics_path = report_dir / "metrics_hvaq_t7_vgg16_lstm.json"
    pred_path = report_dir / "predictions_hvaq_t7_vgg16_lstm.csv"
    config_path = report_dir / "config_hvaq_t7_vgg16_lstm.json"

    pd.DataFrame(history).to_csv(history_path, index=False)
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    pred_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "sequence_id": val_ids,
                    "split": "val",
                    "actual_PM25": val_actual,
                    "predicted_PM25": val_pred,
                }
            ),
            pd.DataFrame(
                {
                    "sequence_id": test_ids,
                    "split": "test",
                    "actual_PM25": test_actual,
                    "predicted_PM25": test_pred,
                }
            ),
        ],
        ignore_index=True,
    )

    pred_df.to_csv(pred_path, index=False)

    config = {
        "args": vars(args),
        "embedding_shape": list(embeddings.shape),
        "rows": {
            "all": int(len(seq_df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    save_scatter(
        val_actual,
        val_pred,
        fig_dir / "scatter_val_hvaq_t7_vgg16_lstm.png",
        f"HVAQ VGG16-LSTM VAL ({args.split_col})",
    )

    save_scatter(
        test_actual,
        test_pred,
        fig_dir / "scatter_test_hvaq_t7_vgg16_lstm.png",
        f"HVAQ VGG16-LSTM TEST ({args.split_col})",
    )

    print("\nSaved:")
    print(" - best model:", best_path)
    print(" - history:", history_path)
    print(" - metrics:", metrics_path)
    print(" - predictions:", pred_path)
    print(" - config:", config_path)
    print(" - figures:", fig_dir)


if __name__ == "__main__":
    main()