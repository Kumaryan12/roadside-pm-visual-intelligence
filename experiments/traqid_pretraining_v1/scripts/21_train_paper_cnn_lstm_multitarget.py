from __future__ import annotations

import argparse
import json
import math
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


TARGET_COLS = ["target_PM2.5", "target_PM10", "target_aqi"]


def get_device(prefer: str) -> torch.device:
    prefer = prefer.lower()

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError(f"Unknown device: {prefer}")


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_corr(y_true, y_pred):
    out = {}

    for i, name in enumerate(["PM2.5", "PM10", "AQI"]):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        if np.std(yp) == 0 or np.std(yt) == 0:
            out[name] = {"Pearson": float("nan"), "Spearman": float("nan")}
            continue

        try:
            p = pearsonr(yt, yp).statistic
        except Exception:
            p = float("nan")

        try:
            s = spearmanr(yt, yp).statistic
        except Exception:
            s = float("nan")

        out[name] = {"Pearson": float(p), "Spearman": float(s)}

    return out


def compute_metrics(y_true, y_pred):
    metrics = {}

    for i, name in enumerate(["PM2.5", "PM10", "AQI"]):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        finite = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[finite]
        yp = yp[finite]

        if len(yt) == 0:
            metrics[name] = {
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "R2": float("nan"),
                "Pearson": float("nan"),
                "Spearman": float("nan"),
                "finite_fraction": 0.0,
            }
            continue

        mae = mean_absolute_error(yt, yp)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)

        if np.std(yp) == 0 or np.std(yt) == 0:
            pear = float("nan")
            spear = float("nan")
        else:
            pear = pearsonr(yt, yp).statistic
            spear = spearmanr(yt, yp).statistic

        metrics[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "Pearson": float(pear),
            "Spearman": float(spear),
            "finite_fraction": float(finite.mean()),
        }

    avg_r2 = np.nanmean([metrics[k]["R2"] for k in metrics])
    avg_rmse = np.nanmean([metrics[k]["RMSE"] for k in metrics])

    metrics["average"] = {
        "R2": float(avg_r2),
        "RMSE": float(avg_rmse),
    }

    return metrics


class PaperSequenceDataset(Dataset):
    def __init__(self, seq_df: pd.DataFrame, features: np.ndarray, target_min, target_max):
        self.seq_df = seq_df.reset_index(drop=True)
        self.features = features
        self.target_min = np.asarray(target_min, dtype=np.float32)
        self.target_max = np.asarray(target_max, dtype=np.float32)
        self.target_range = np.maximum(self.target_max - self.target_min, 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]

        row_ids = [int(x) for x in str(row["seq_row_ids"]).split("|")]
        x = self.features[row_ids, :].astype(np.float32)

        y_raw = row[TARGET_COLS].to_numpy(dtype=np.float32)
        y_norm = (y_raw - self.target_min) / self.target_range
        y_norm = np.clip(y_norm, 0.0, 1.0).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y_norm), torch.from_numpy(y_raw), int(row["sequence_id"])


class CNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        head_hidden_dim: int,
        output_dim: int = 3,
        use_sigmoid: bool = True,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, output_dim),
        )

        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y = self.head(h)

        if self.use_sigmoid:
            y = self.sigmoid(y)

        return y


def evaluate(model, loader, device, target_min, target_max):
    model.eval()

    all_pred_norm = []
    all_true_raw = []
    all_seq_ids = []

    target_min = np.asarray(target_min, dtype=np.float32)
    target_max = np.asarray(target_max, dtype=np.float32)
    target_range = np.maximum(target_max - target_min, 1e-6).astype(np.float32)

    with torch.inference_mode():
        for x, y_norm, y_raw, seq_ids in loader:
            x = x.to(device)
            pred_norm = model(x).detach().cpu().numpy()

            all_pred_norm.append(pred_norm)
            all_true_raw.append(y_raw.numpy())
            all_seq_ids.extend(seq_ids.numpy().tolist())

    pred_norm = np.concatenate(all_pred_norm, axis=0)
    true_raw = np.concatenate(all_true_raw, axis=0)

    pred_raw = pred_norm * target_range[None, :] + target_min[None, :]

    metrics = compute_metrics(true_raw, pred_raw)

    return metrics, true_raw, pred_raw, np.asarray(all_seq_ids)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def plot_history(history_df, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_PM2.5_RMSE"], marker="o", label="Val PM2.5 RMSE")
    plt.plot(history_df["epoch"], history_df["test_PM2.5_RMSE"], marker="o", label="Test PM2.5 RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.title("Paper-style CNN-LSTM training curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "training_curve_pm25.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["val_PM2.5_R2"], marker="o", label="Val PM2.5 R²")
    plt.plot(history_df["epoch"], history_df["test_PM2.5_R2"], marker="o", label="Test PM2.5 R²")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Paper-style CNN-LSTM PM2.5 R²")
    plt.axhline(0, linewidth=1)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "r2_curve_pm25.png", dpi=220)
    plt.close()


def plot_scatter(y_true, y_pred, fig_dir, split_name):
    fig_dir.mkdir(parents=True, exist_ok=True)

    names = ["PM2.5", "PM10", "AQI"]

    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        lo = min(float(np.min(yt)), float(np.min(yp)))
        hi = max(float(np.max(yt)), float(np.max(yp)))

        plt.figure(figsize=(6, 6))
        plt.scatter(yt, yp, s=12, alpha=0.45)
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
        plt.xlabel(f"Actual {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"{split_name}: actual vs predicted {name}")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        safe_name = name.replace(".", "_")
        plt.savefig(fig_dir / f"scatter_{split_name}_{safe_name}.png", dpi=220)
        plt.close()


def run_single_split(args, seq_df, features, split_col, train_name, val_name, test_name, run_suffix):
    device = get_device(args.device)
    seed_everything(args.seed)

    out_dir = Path(args.out_dir) / run_suffix
    report_dir = Path(args.report_dir) / run_suffix
    fig_dir = Path(args.fig_dir) / run_suffix

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_df = seq_df[seq_df[split_col] == train_name].copy()
    val_df = seq_df[seq_df[split_col] == val_name].copy()
    test_df = seq_df[seq_df[split_col] == test_name].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Empty split. train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    if len(val_df) == 0:
        val_df = test_df.copy()

    target_min = train_df[TARGET_COLS].min().to_numpy(dtype=np.float32)
    target_max = train_df[TARGET_COLS].max().to_numpy(dtype=np.float32)

    train_ds = PaperSequenceDataset(train_df, features, target_min, target_max)
    val_ds = PaperSequenceDataset(val_df, features, target_min, target_max)
    test_ds = PaperSequenceDataset(test_df, features, target_min, target_max)

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    model = CNNLSTMRegressor(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head_hidden_dim=args.head_hidden_dim,
        output_dim=3,
        use_sigmoid=True,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = float("inf")
    best_payload = None
    history = []

    print("\n" + "=" * 90)
    print("TRAINING PAPER-STYLE CNN-LSTM")
    print("=" * 90)
    print("Run:", run_suffix)
    print("Split col:", split_col)
    print("Train/val/test names:", train_name, val_name, test_name)
    print("Train/val/test sizes:", len(train_df), len(val_df), len(test_df))
    print("Feature shape:", features.shape)
    print("Target min:", target_min.tolist())
    print("Target max:", target_max.tolist())
    print("Device:", device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x, y_norm, y_raw, seq_ids in tqdm(train_loader, desc=f"epoch {epoch:03d}/{args.epochs}", leave=False):
            x = x.to(device)
            y_norm = y_norm.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y_norm)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(losses))

        val_metrics, val_true, val_pred, val_ids = evaluate(model, val_loader, device, target_min, target_max)
        test_metrics, test_true, test_pred, test_ids = evaluate(model, test_loader, device, target_min, target_max)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_avg_RMSE": val_metrics["average"]["RMSE"],
            "val_avg_R2": val_metrics["average"]["R2"],
            "test_avg_RMSE": test_metrics["average"]["RMSE"],
            "test_avg_R2": test_metrics["average"]["R2"],
        }

        for target in ["PM2.5", "PM10", "AQI"]:
            row[f"val_{target}_RMSE"] = val_metrics[target]["RMSE"]
            row[f"val_{target}_R2"] = val_metrics[target]["R2"]
            row[f"test_{target}_RMSE"] = test_metrics[target]["RMSE"]
            row[f"test_{target}_R2"] = test_metrics[target]["R2"]
            row[f"test_{target}_Spearman"] = test_metrics[target]["Spearman"]

        history.append(row)

        # Select by average validation RMSE across all three paper targets.
        score = val_metrics["average"]["RMSE"]

        if score < best_score:
            best_score = score
            best_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "target_min": target_min.tolist(),
                "target_max": target_max.tolist(),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "args": vars(args),
                "run_suffix": run_suffix,
            }

            torch.save(best_payload, out_dir / "best_paper_cnn_lstm_multitarget.pt")

            pred_rows = []
            for split_label, ids, true_arr, pred_arr in [
                ("val", val_ids, val_true, val_pred),
                ("test", test_ids, test_true, test_pred),
            ]:
                for sid, yt, yp in zip(ids, true_arr, pred_arr):
                    pred_rows.append(
                        {
                            "split": split_label,
                            "sequence_id": int(sid),
                            "actual_PM2.5": float(yt[0]),
                            "predicted_PM2.5": float(yp[0]),
                            "actual_PM10": float(yt[1]),
                            "predicted_PM10": float(yp[1]),
                            "actual_aqi": float(yt[2]),
                            "predicted_aqi": float(yp[2]),
                        }
                    )

            pd.DataFrame(pred_rows).to_csv(report_dir / "predictions_paper_cnn_lstm_multitarget.csv", index=False)

        print(
            f"[{run_suffix}] epoch {epoch:03d}/{args.epochs} "
            f"loss={train_loss:.5f} | "
            f"VAL PM2.5 R2={val_metrics['PM2.5']['R2']:.3f}, RMSE={val_metrics['PM2.5']['RMSE']:.2f} | "
            f"TEST PM2.5 R2={test_metrics['PM2.5']['R2']:.3f}, RMSE={test_metrics['PM2.5']['RMSE']:.2f}"
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(report_dir / "training_history_paper_cnn_lstm_multitarget.csv", index=False)

    metrics = {
        "best_checkpoint": {
            "path": str(out_dir / "best_paper_cnn_lstm_multitarget.pt"),
            "epoch": best_payload["epoch"],
        },
        "target_min": best_payload["target_min"],
        "target_max": best_payload["target_max"],
        "val": best_payload["val_metrics"],
        "test": best_payload["test_metrics"],
        "config": vars(args),
        "run_suffix": run_suffix,
        "split": {
            "split_col": split_col,
            "train_name": train_name,
            "val_name": val_name,
            "test_name": test_name,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
        },
    }

    (report_dir / "metrics_paper_cnn_lstm_multitarget.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plot_history(history_df, fig_dir)

    pred_df = pd.read_csv(report_dir / "predictions_paper_cnn_lstm_multitarget.csv")
    for split_label in ["val", "test"]:
        part = pred_df[pred_df["split"] == split_label]
        y_true = part[["actual_PM2.5", "actual_PM10", "actual_aqi"]].to_numpy()
        y_pred = part[["predicted_PM2.5", "predicted_PM10", "predicted_aqi"]].to_numpy()
        plot_scatter(y_true, y_pred, fig_dir, split_label)

    print("\nBEST METRICS:")
    print(json.dumps(metrics, indent=2))

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-manifest",
        required=True,
    )
    parser.add_argument(
        "--features",
        required=True,
    )
    parser.add_argument("--split-col", default="split_random")
    parser.add_argument("--train-name", default="train")
    parser.add_argument("--val-name", default="val")
    parser.add_argument("--test-name", default="test")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/models/paper_cnn_lstm",
    )
    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/paper_cnn_lstm",
    )
    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures/paper_cnn_lstm",
    )

    args = parser.parse_args()

    seq_df = pd.read_csv(args.sequence_manifest)
    features = np.load(args.features, mmap_mode="r")

    print("=" * 90)
    print("PAPER-STYLE CNN-LSTM MULTITARGET TRAINER")
    print("=" * 90)
    print("Sequence manifest:", args.sequence_manifest)
    print("Sequences:", len(seq_df))
    print("Features:", args.features)
    print("Feature shape:", features.shape)
    print("Split column:", args.split_col)
    print("Targets:", TARGET_COLS)

    run_suffix = Path(args.features).stem.replace("_features", "")
    run_suffix += f"_{Path(args.sequence_manifest).stem}"
    run_suffix += f"_{args.split_col}"

    run_single_split(
        args=args,
        seq_df=seq_df,
        features=features,
        split_col=args.split_col,
        train_name=args.train_name,
        val_name=args.val_name,
        test_name=args.test_name,
        run_suffix=run_suffix,
    )


if __name__ == "__main__":
    main()