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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

        if np.std(yt) == 0 or np.std(yp) == 0:
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

    metrics["average"] = {
        "R2": float(np.nanmean([metrics[k]["R2"] for k in ["PM2.5", "PM10", "AQI"]])),
        "RMSE": float(np.nanmean([metrics[k]["RMSE"] for k in ["PM2.5", "PM10", "AQI"]])),
    }

    return metrics


def add_time_features(df: pd.DataFrame):
    df = df.copy()

    if "target_time" not in df.columns:
        raise ValueError("sequence manifest must contain target_time")

    dt = pd.to_datetime(df["target_time"], errors="coerce")
    hour = dt.dt.hour.fillna(0).astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    return df


def build_target_context(seq_df: pd.DataFrame, base_manifest: pd.DataFrame):
    """
    Adds Temperature, Humidity, Season, Day_or_Night from the target row.
    Sequence manifest has target_row_id.
    Base manifest has row_id and metadata.
    """

    cols = [
        "row_id",
        "Temperature",
        "Humidity",
        "Season",
        "Day_or_Night",
    ]

    missing = [c for c in cols if c not in base_manifest.columns]
    if missing:
        raise ValueError(f"Base manifest missing required context columns: {missing}")

    meta = base_manifest[cols].copy()
    meta = meta.rename(columns={"row_id": "target_row_id"})

    out = seq_df.merge(meta, on="target_row_id", how="left")
    out = add_time_features(out)

    return out


class TabularBuilder:
    def __init__(self):
        self.num_cols = ["Temperature", "Humidity", "hour_sin", "hour_cos"]
        self.cat_cols = ["Season", "Day_or_Night"]

        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        self.num_medians = None

    def fit(self, df: pd.DataFrame):
        num = df[self.num_cols].copy()
        self.num_medians = num.median(numeric_only=True)

        num = num.fillna(self.num_medians)
        cat = df[self.cat_cols].astype(str).fillna("missing")

        self.scaler.fit(num)
        self.encoder.fit(cat)

    def transform(self, df: pd.DataFrame):
        num = df[self.num_cols].copy()
        num = num.fillna(self.num_medians)

        cat = df[self.cat_cols].astype(str).fillna("missing")

        num_x = self.scaler.transform(num).astype("float32")
        cat_x = self.encoder.transform(cat).astype("float32")

        return np.concatenate([num_x, cat_x], axis=1).astype("float32")

    def summary(self):
        return {
            "numeric_cols": self.num_cols,
            "categorical_cols": self.cat_cols,
            "num_medians": {k: float(v) for k, v in self.num_medians.to_dict().items()},
            "onehot_categories": {
                col: list(map(str, cats))
                for col, cats in zip(self.cat_cols, self.encoder.categories_)
            },
        }


class PaperFusionDataset(Dataset):
    def __init__(
        self,
        seq_df: pd.DataFrame,
        features: np.ndarray,
        tabular_x: np.ndarray,
        target_min,
        target_max,
    ):
        self.seq_df = seq_df.reset_index(drop=True)
        self.features = features
        self.tabular_x = tabular_x.astype("float32")

        self.target_min = np.asarray(target_min, dtype=np.float32)
        self.target_max = np.asarray(target_max, dtype=np.float32)
        self.target_range = np.maximum(self.target_max - self.target_min, 1e-6).astype(np.float32)

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]

        row_ids = [int(x) for x in str(row["seq_row_ids"]).split("|")]
        x_seq = self.features[row_ids, :].astype("float32")
        x_tab = self.tabular_x[idx]

        y_raw = row[TARGET_COLS].to_numpy(dtype=np.float32)
        y_norm = (y_raw - self.target_min) / self.target_range
        y_norm = np.clip(y_norm, 0.0, 1.0).astype("float32")

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(x_tab),
            torch.from_numpy(y_norm),
            torch.from_numpy(y_raw),
            int(row["sequence_id"]),
        )


class CNNLSTMTabularFusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        tabular_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        tab_hidden_dim: int,
        fusion_hidden_dim: int,
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

        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, tab_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
            nn.ReLU(),
        )

        self.fusion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + tab_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, x_tab):
        out, _ = self.lstm(x_seq)
        h = out[:, -1, :]

        t = self.tabular_net(x_tab)

        fused = torch.cat([h, t], dim=1)
        y = self.fusion_head(fused)

        if self.use_sigmoid:
            y = self.sigmoid(y)

        return y


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def evaluate(model, loader, device, target_min, target_max):
    model.eval()

    all_pred_norm = []
    all_true_raw = []
    all_seq_ids = []

    target_min = np.asarray(target_min, dtype=np.float32)
    target_max = np.asarray(target_max, dtype=np.float32)
    target_range = np.maximum(target_max - target_min, 1e-6).astype(np.float32)

    with torch.inference_mode():
        for x_seq, x_tab, y_norm, y_raw, seq_ids in loader:
            x_seq = x_seq.to(device)
            x_tab = x_tab.to(device)

            pred_norm = model(x_seq, x_tab).detach().cpu().numpy()

            all_pred_norm.append(pred_norm)
            all_true_raw.append(y_raw.numpy())
            all_seq_ids.extend(seq_ids.numpy().tolist())

    pred_norm = np.concatenate(all_pred_norm, axis=0)
    true_raw = np.concatenate(all_true_raw, axis=0)
    pred_raw = pred_norm * target_range[None, :] + target_min[None, :]

    metrics = compute_metrics(true_raw, pred_raw)

    return metrics, true_raw, pred_raw, np.asarray(all_seq_ids)


def plot_history(history_df, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_PM2.5_RMSE"], marker="o", label="Val PM2.5 RMSE")
    plt.plot(history_df["epoch"], history_df["test_PM2.5_RMSE"], marker="o", label="Test PM2.5 RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.title("CNN-LSTM + tabular fusion training curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "training_curve_pm25.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["val_PM2.5_R2"], marker="o", label="Val PM2.5 R²")
    plt.plot(history_df["epoch"], history_df["test_PM2.5_R2"], marker="o", label="Test PM2.5 R²")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("CNN-LSTM + tabular fusion PM2.5 R²")
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--features", required=True)

    parser.add_argument("--split-col", default="split_random")
    parser.add_argument("--train-name", default="train")
    parser.add_argument("--val-name", default="val")
    parser.add_argument("--test-name", default="test")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tab-hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/models/paper_cnn_lstm_tabular_fusion",
    )
    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/paper_cnn_lstm_tabular_fusion",
    )
    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures/paper_cnn_lstm_tabular_fusion",
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)

    seq_df = pd.read_csv(args.sequence_manifest)
    base_df = pd.read_csv(args.base_manifest)
    features = np.load(args.features, mmap_mode="r")

    seq_df = build_target_context(seq_df, base_df)

    train_df = seq_df[seq_df[args.split_col] == args.train_name].copy()
    val_df = seq_df[seq_df[args.split_col] == args.val_name].copy()
    test_df = seq_df[seq_df[args.split_col] == args.test_name].copy()

    if len(val_df) == 0:
        val_df = test_df.copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Empty split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    tab_builder = TabularBuilder()
    tab_builder.fit(train_df)

    train_tab = tab_builder.transform(train_df)
    val_tab = tab_builder.transform(val_df)
    test_tab = tab_builder.transform(test_df)

    target_min = train_df[TARGET_COLS].min().to_numpy(dtype=np.float32)
    target_max = train_df[TARGET_COLS].max().to_numpy(dtype=np.float32)

    train_ds = PaperFusionDataset(train_df, features, train_tab, target_min, target_max)
    val_ds = PaperFusionDataset(val_df, features, val_tab, target_min, target_max)
    test_ds = PaperFusionDataset(test_df, features, test_tab, target_min, target_max)

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    model = CNNLSTMTabularFusion(
        input_dim=features.shape[1],
        tabular_dim=train_tab.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tab_hidden_dim=args.tab_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        output_dim=3,
        use_sigmoid=True,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_suffix = Path(args.features).stem.replace("_features", "")
    run_suffix += f"_{Path(args.sequence_manifest).stem}_{args.split_col}_tabular_fusion"

    out_dir = Path(args.out_dir) / run_suffix
    report_dir = Path(args.report_dir) / run_suffix
    fig_dir = Path(args.fig_dir) / run_suffix

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PAPER-STYLE CNN-LSTM + TABULAR FUSION")
    print("=" * 90)
    print("Sequence manifest:", args.sequence_manifest)
    print("Base manifest:", args.base_manifest)
    print("Features:", args.features)
    print("Feature shape:", features.shape)
    print("Train/val/test:", len(train_df), len(val_df), len(test_df))
    print("Tabular shape:", train_tab.shape)
    print("Tabular summary:", json.dumps(tab_builder.summary(), indent=2))
    print("Target min:", target_min.tolist())
    print("Target max:", target_max.tolist())
    print("Device:", device)
    print("Run:", run_suffix)

    best_score = float("inf")
    best_payload = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x_seq, x_tab, y_norm, y_raw, seq_ids in tqdm(
            train_loader,
            desc=f"epoch {epoch:03d}/{args.epochs}",
            leave=False,
        ):
            x_seq = x_seq.to(device)
            x_tab = x_tab.to(device)
            y_norm = y_norm.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_seq, x_tab)
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

        score = val_metrics["average"]["RMSE"]

        if score < best_score:
            best_score = score

            best_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "target_min": target_min.tolist(),
                "target_max": target_max.tolist(),
                "tabular_summary": tab_builder.summary(),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "args": vars(args),
                "run_suffix": run_suffix,
            }

            torch.save(best_payload, out_dir / "best_paper_cnn_lstm_tabular_fusion.pt")

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

            pd.DataFrame(pred_rows).to_csv(
                report_dir / "predictions_paper_cnn_lstm_tabular_fusion.csv",
                index=False,
            )

        print(
            f"[tabular fusion] epoch {epoch:03d}/{args.epochs} "
            f"loss={train_loss:.5f} | "
            f"VAL PM2.5 R2={val_metrics['PM2.5']['R2']:.3f}, RMSE={val_metrics['PM2.5']['RMSE']:.2f} | "
            f"TEST PM2.5 R2={test_metrics['PM2.5']['R2']:.3f}, RMSE={test_metrics['PM2.5']['RMSE']:.2f}"
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(report_dir / "training_history_paper_cnn_lstm_tabular_fusion.csv", index=False)

    metrics = {
        "best_checkpoint": {
            "path": str(out_dir / "best_paper_cnn_lstm_tabular_fusion.pt"),
            "epoch": best_payload["epoch"],
        },
        "target_min": best_payload["target_min"],
        "target_max": best_payload["target_max"],
        "tabular_summary": best_payload["tabular_summary"],
        "val": best_payload["val_metrics"],
        "test": best_payload["test_metrics"],
        "config": vars(args),
        "run_suffix": run_suffix,
        "split": {
            "split_col": args.split_col,
            "train_name": args.train_name,
            "val_name": args.val_name,
            "test_name": args.test_name,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
        },
    }

    (report_dir / "metrics_paper_cnn_lstm_tabular_fusion.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plot_history(history_df, fig_dir)

    pred_df = pd.read_csv(report_dir / "predictions_paper_cnn_lstm_tabular_fusion.csv")
    for split_label in ["val", "test"]:
        part = pred_df[pred_df["split"] == split_label]
        y_true = part[["actual_PM2.5", "actual_PM10", "actual_aqi"]].to_numpy()
        y_pred = part[["predicted_PM2.5", "predicted_PM10", "predicted_aqi"]].to_numpy()
        plot_scatter(y_true, y_pred, fig_dir, split_label)

    print("\nBEST METRICS:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()