#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_ids(s):
    if pd.isna(s):
        return []
    return [int(float(x)) for x in str(s).split("|") if str(x).strip()]


def parse_values(s):
    if pd.isna(s):
        return []
    return [float(x) for x in str(s).split("|") if str(x).strip()]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def all_metrics(y_true, y_pred, horizon):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    pm25_true = y_true[:, :horizon]
    pm25_pred = y_pred[:, :horizon]
    pm10_true = y_true[:, horizon:]
    pm10_pred = y_pred[:, horizon:]

    out = {
        "PM2.5_all_horizons": regression_metrics(pm25_true.reshape(-1), pm25_pred.reshape(-1)),
        "PM10_all_horizons": regression_metrics(pm10_true.reshape(-1), pm10_pred.reshape(-1)),
        "all_outputs": regression_metrics(y_true.reshape(-1), y_pred.reshape(-1)),
        "PM2.5_by_horizon": [],
        "PM10_by_horizon": [],
    }

    for h in range(horizon):
        out["PM2.5_by_horizon"].append({
            "horizon_step": h + 1,
            **regression_metrics(pm25_true[:, h], pm25_pred[:, h]),
        })
        out["PM10_by_horizon"].append({
            "horizon_step": h + 1,
            **regression_metrics(pm10_true[:, h], pm10_pred[:, h]),
        })

    out["average_R2"] = float(
        (out["PM2.5_all_horizons"]["R2"] + out["PM10_all_horizons"]["R2"]) / 2
    )
    out["average_RMSE"] = float(
        (out["PM2.5_all_horizons"]["RMSE"] + out["PM10_all_horizons"]["RMSE"]) / 2
    )
    return out


def build_arrays(df, hourly_df, horizon, scaler=None, fit_scaler=False):
    hourly_df = hourly_df.copy()
    hourly_df["hourly_row_id"] = hourly_df["hourly_row_id"].astype(int)
    hourly = hourly_df.set_index("hourly_row_id")

    X_seq = []
    X_static_raw = []
    y = []

    static_cols = ["Temperature", "Humidity", "hour_sin", "hour_cos"]

    for _, r in df.iterrows():
        input_ids = parse_ids(r["input_row_ids"])
        if len(input_ids) != horizon:
            continue

        hist = hourly.loc[input_ids]

        seq_features = []
        for _, hr in hist.iterrows():
            seq_features.append([
                float(hr["PM2.5"]),
                float(hr["PM10"]),
                float(hr["Temperature"]) if "Temperature" in hr else 0.0,
                float(hr["Humidity"]) if "Humidity" in hr else 0.0,
            ])

        fut_pm25 = parse_values(r["target_PM2.5_values"])
        fut_pm10 = parse_values(r["target_PM10_values"])

        if len(fut_pm25) != horizon or len(fut_pm10) != horizon:
            continue

        static_features = []
        for c in static_cols:
            static_features.append(float(r[c]) if c in r and not pd.isna(r[c]) else 0.0)

        # simple categorical encodings
        daynight = str(r["Day_or_Night"]) if "Day_or_Night" in r else "Unknown"
        season = str(r["Season"]) if "Season" in r else "Unknown"

        static_features += [
            1.0 if daynight == "Day" else 0.0,
            1.0 if daynight == "Night" else 0.0,
            1.0 if season == "Summer" else 0.0,
            1.0 if season == "Monsoon" else 0.0,
            1.0 if season == "Winter" else 0.0,
        ]

        X_seq.append(seq_features)
        X_static_raw.append(static_features)
        y.append(fut_pm25 + fut_pm10)

    X_seq = np.asarray(X_seq, dtype=np.float32)
    X_static = np.asarray(X_static_raw, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    n, t, f = X_seq.shape
    X_seq_flat = X_seq.reshape(n, t * f)
    X_all_for_scaling = np.concatenate([X_seq_flat, X_static], axis=1)

    if fit_scaler:
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all_for_scaling)
    else:
        if scaler is None:
            raise ValueError("Scaler required when fit_scaler=False")
        X_all_scaled = scaler.transform(X_all_for_scaling)

    X_seq_scaled = X_all_scaled[:, :t * f].reshape(n, t, f).astype(np.float32)
    X_static_scaled = X_all_scaled[:, t * f:].astype(np.float32)

    return X_seq_scaled, X_static_scaled, y, scaler


class ForecastDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


class NumericalLSTMForecaster(nn.Module):
    def __init__(self, seq_input_dim, static_dim, hidden_dim, num_layers, dropout, horizon):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, horizon * 2),
        )

    def forward(self, x_seq, x_static):
        out, (h, c) = self.lstm(x_seq)
        last = h[-1]
        z = torch.cat([last, x_static], dim=1)
        return self.head(z)


def run_epoch(model, loader, optimizer, criterion, device, train=True, grad_clip=1.0):
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    n = 0

    for x_seq, x_static, y in loader:
        x_seq = x_seq.to(device)
        x_static = x_static.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x_seq, x_static)
            loss = criterion(pred, y)

            if train:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total += float(loss.item()) * len(y)
        n += len(y)

    return total / max(n, 1)


def predict(model, loader, device):
    model.eval()
    preds = []
    ys = []

    with torch.no_grad():
        for x_seq, x_static, y in loader:
            x_seq = x_seq.to(device)
            x_static = x_static.to(device)
            pred = model(x_seq, x_static).cpu().numpy()
            preds.append(pred)
            ys.append(y.numpy())

    return np.vstack(ys), np.vstack(preds)


def evaluate_one_split(df, hourly_df, split_col, train_name, eval_name, args, device):
    train_df = df[df[split_col] == train_name].copy()
    eval_df = df[df[split_col] == eval_name].copy()

    # Use validation from same split if available, otherwise split 15% from train internally.
    if split_col == "split_random_forecast":
        val_df = df[df[split_col] == "val"].copy()
    elif split_col == "split_chrono_forecast":
        val_df = df[df[split_col] == "val"].copy()
    else:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(train_df))
        rng.shuffle(idx)
        n_val = max(1, int(0.15 * len(train_df)))
        val_idx = idx[:n_val]
        keep_idx = idx[n_val:]
        val_df = train_df.iloc[val_idx].copy()
        train_df = train_df.iloc[keep_idx].copy()

    print("\n" + "=" * 90)
    print(f"LSTM SPLIT: {split_col} | train={train_name} | eval={eval_name}")
    print("=" * 90)
    print("Train rows:", len(train_df), "Val rows:", len(val_df), "Eval rows:", len(eval_df))

    Xtr_seq, Xtr_static, ytr, scaler = build_arrays(train_df, hourly_df, args.horizon, fit_scaler=True)
    Xva_seq, Xva_static, yva, _ = build_arrays(val_df, hourly_df, args.horizon, scaler=scaler, fit_scaler=False)
    Xte_seq, Xte_static, yte, _ = build_arrays(eval_df, hourly_df, args.horizon, scaler=scaler, fit_scaler=False)

    print("X train seq/static/y:", Xtr_seq.shape, Xtr_static.shape, ytr.shape)
    print("X val   seq/static/y:", Xva_seq.shape, Xva_static.shape, yva.shape)
    print("X eval  seq/static/y:", Xte_seq.shape, Xte_static.shape, yte.shape)

    train_loader = DataLoader(
        ForecastDataset(Xtr_seq, Xtr_static, ytr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ForecastDataset(Xva_seq, Xva_static, yva),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    eval_loader = DataLoader(
        ForecastDataset(Xte_seq, Xte_static, yte),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = NumericalLSTMForecaster(
        seq_input_dim=Xtr_seq.shape[-1],
        static_dim=Xtr_static.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        horizon=args.horizon,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, criterion, device, train=True, grad_clip=args.grad_clip)
        va_loss = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} best_epoch={best_epoch}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best_val={best_val:.6f}")
            break

    model.load_state_dict(best_state)

    y_eval, pred_eval = predict(model, eval_loader, device)
    metrics = all_metrics(y_eval, pred_eval, args.horizon)

    print("EVAL METRICS:")
    print(json.dumps({
        "PM2.5_R2": metrics["PM2.5_all_horizons"]["R2"],
        "PM10_R2": metrics["PM10_all_horizons"]["R2"],
        "Avg_R2": metrics["average_R2"],
        "PM2.5_RMSE": metrics["PM2.5_all_horizons"]["RMSE"],
        "PM10_RMSE": metrics["PM10_all_horizons"]["RMSE"],
        "Avg_RMSE": metrics["average_RMSE"],
    }, indent=2))

    return {
        "model": "numerical_lstm",
        "split_col": split_col,
        "train_name": train_name,
        "eval_name": eval_name,
        "best_epoch": best_epoch,
        "PM2.5_R2": metrics["PM2.5_all_horizons"]["R2"],
        "PM10_R2": metrics["PM10_all_horizons"]["R2"],
        "Avg_R2": metrics["average_R2"],
        "PM2.5_RMSE": metrics["PM2.5_all_horizons"]["RMSE"],
        "PM10_RMSE": metrics["PM10_all_horizons"]["RMSE"],
        "Avg_RMSE": metrics["average_RMSE"],
        "details": metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-manifest", required=True)
    ap.add_argument("--hourly-base", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=200)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    seed_everything(args.seed)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("Using device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.forecast_manifest)
    hourly_df = pd.read_csv(args.hourly_base)

    split_specs = [
        ("split_random_forecast", "train", "test"),
        ("split_twofold_forecast", "fold1_train", "fold1_test"),
        ("split_chrono_forecast", "train", "test"),
    ]

    results = []
    details = {}

    for split_col, train_name, eval_name in split_specs:
        r = evaluate_one_split(df, hourly_df, split_col, train_name, eval_name, args, device)
        results.append({k: v for k, v in r.items() if k != "details"})
        details[f"{split_col}__{eval_name}"] = r["details"]

    summary = pd.DataFrame(results)

    csv_path = out_dir / "forecast_numerical_lstm_summary.csv"
    md_path = out_dir / "forecast_numerical_lstm_summary.md"
    json_path = out_dir / "forecast_numerical_lstm_details.json"

    summary.to_csv(csv_path, index=False)
    md_path.write_text(summary.to_markdown(index=False))
    json_path.write_text(json.dumps(details, indent=2))

    print("\n" + "=" * 90)
    print("FINAL NUMERICAL LSTM SUMMARY")
    print("=" * 90)
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(csv_path)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
