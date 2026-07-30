"""Train an image-embedding GRU or LSTM on fixed leakage-safe splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from roadside_pm.features.images.resnet import select_device
from roadside_pm.modeling.image_temporal import ImageTemporalRegressor


def transform_targets(values, transform):
    values = np.asarray(values, dtype="float32")
    if transform == "none": return values
    if transform == "log1p":
        if np.any(values < 0): raise ValueError("log1p target transform requires non-negative targets")
        return np.log1p(values)
    raise ValueError(f"Unknown target transform: {transform}")


def inverse_targets(values, transform):
    if transform == "none": return values
    if transform == "log1p": return np.maximum(np.expm1(values), 0.0)
    raise ValueError(f"Unknown target transform: {transform}")


class Sequences(Dataset):
    def __init__(self, frame, embeddings, targets, target_mean, target_std, target_transform):
        self.frame = frame.reset_index(drop=True); self.embeddings = embeddings; self.targets = targets
        self.mean = target_mean; self.std = target_std; self.target_transform = target_transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        positions = [int(value) for value in str(row["embedding_rows"]).split("|")]
        x = self.embeddings[positions].astype("float32")
        y = transform_targets(row[self.targets].to_numpy(dtype="float32"), self.target_transform)
        return torch.from_numpy(x), torch.from_numpy((y - self.mean) / self.std), int(row["sequence_id"])


def metrics(y_true, y_pred, names):
    result = {}
    for index, name in enumerate(names):
        result[name] = {"MAE": float(mean_absolute_error(y_true[:, index], y_pred[:, index])), "RMSE": float(mean_squared_error(y_true[:, index], y_pred[:, index]) ** 0.5), "R2": float(r2_score(y_true[:, index], y_pred[:, index]))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-manifest", required=True); parser.add_argument("--embeddings", required=True)
    parser.add_argument("--split-col", required=True); parser.add_argument("--target-cols", nargs="+", required=True)
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru"); parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--patience", type=int, default=15); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--dropout", type=float, default=0.3); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="auto")
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="none")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--balance-by-date", action="store_true")
    parser.add_argument("--extreme-iqr-multiplier", type=float, default=3.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    frame = pd.read_csv(args.sequence_manifest); embeddings = np.load(args.embeddings, mmap_mode="r")
    train = frame[frame[args.split_col] == "train"].copy(); val = frame[frame[args.split_col] == "val"].copy(); test = frame[frame[args.split_col] == "test"].copy()
    if min(len(train), len(val), len(test)) == 0: raise ValueError(f"Empty split: train={len(train)}, val={len(val)}, test={len(test)}")
    raw_train_targets = train[args.target_cols].to_numpy(dtype="float32")
    train_targets = transform_targets(raw_train_targets, args.target_transform)
    target_mean = train_targets.mean(axis=0); target_std = np.maximum(train_targets.std(axis=0), 1e-6)
    datasets = {name: Sequences(part, embeddings, args.target_cols, target_mean, target_std, args.target_transform) for name, part in (("train", train), ("val", val), ("test", test))}
    sampler = None
    balance_summary = {"enabled": False}
    if args.balance_by_date:
        if "target_time" not in train:
            raise ValueError("balance-by-date requires target_time in the sequence manifest")
        dates = pd.to_datetime(train["target_time"], errors="raise").dt.date.astype(str)
        counts = dates.value_counts()
        weights = dates.map(lambda value: 1.0 / counts[value]).to_numpy(dtype="float64", copy=True)
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True, generator=generator)
        balance_summary = {"enabled": True, "training_sequence_counts_by_date": counts.sort_index().to_dict(), "sampling": "inverse_training_date_frequency_with_replacement"}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }
    device = select_device(args.device); model = ImageTemporalRegressor(embeddings.shape[1], len(args.target_cols), cell=args.cell, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); loss_fn = nn.SmoothL1Loss(beta=args.huber_beta); best = float("inf"); best_state = None; bad = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y, _ in loaders["train"]:
            optimizer.zero_grad(); prediction = model(x.to(device)); loss = loss_fn(prediction, y.to(device)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(loss.item())
        model.eval(); val_losses = []
        with torch.inference_mode():
            for x, y, _ in loaders["val"]: val_losses.append(loss_fn(model(x.to(device)), y.to(device)).item())
        val_loss = float(np.mean(val_losses)); history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best: best, bad, best_state = val_loss, 0, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else: bad += 1
        if bad >= args.patience: break
    model.load_state_dict(best_state); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "target_mean": target_mean.tolist(), "target_std": target_std.tolist(), "date_balancing": balance_summary}, output / "best_model.pt")
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    report = {}; robustness_report = {
        "target_transform": args.target_transform,
        "loss": "SmoothL1Loss",
        "huber_beta_standardized_target_units": args.huber_beta,
        "date_balancing": balance_summary,
        "extreme_threshold_source": "outer-training partition only",
        "extreme_iqr_multiplier": args.extreme_iqr_multiplier,
        "targets": {},
    }
    thresholds = {}
    for target in args.target_cols:
        values = pd.to_numeric(train[target], errors="raise")
        q1, q3 = values.quantile([0.25, 0.75])
        thresholds[target] = float(q3 + args.extreme_iqr_multiplier * (q3 - q1))
        robustness_report["targets"][target] = {"upper_extreme_threshold": thresholds[target]}
    for name in ("val", "test"):
        truth, predictions, ids = [], [], []
        model.eval()
        with torch.inference_mode():
            for x, y, sequence_ids in loaders[name]:
                pred_transformed = model(x.to(device)).cpu().numpy() * target_std + target_mean
                true_transformed = y.numpy() * target_std + target_mean
                pred = inverse_targets(pred_transformed, args.target_transform)
                true = inverse_targets(true_transformed, args.target_transform)
                predictions.append(pred); truth.append(true); ids.extend(sequence_ids.numpy().tolist())
        actual = np.concatenate(truth); predicted = np.concatenate(predictions); report[name] = metrics(actual, predicted, args.target_cols)
        table = pd.DataFrame({"sequence_id": ids});
        table["prediction_origin"] = "outer_val" if name == "val" else "outer_test"
        for index, target in enumerate(args.target_cols):
            table[f"actual_{target}"] = actual[:, index]; table[f"predicted_{target}"] = predicted[:, index]
            inlier = actual[:, index] <= thresholds[target]
            table[f"extreme_by_train_threshold_{target}"] = ~inlier
            target_report = robustness_report["targets"][target].setdefault(name, {})
            target_report["rows_all"] = int(len(actual))
            target_report["rows_inlier"] = int(inlier.sum())
            target_report["rows_extreme"] = int((~inlier).sum())
            target_report["metrics_all"] = report[name][target]
            target_report["metrics_inlier"] = metrics(actual[inlier][:, [index]], predicted[inlier][:, [index]], [target])[target] if inlier.sum() >= 2 else None
            target_report["metrics_extreme"] = metrics(actual[~inlier][:, [index]], predicted[~inlier][:, [index]], [target])[target] if (~inlier).sum() >= 2 else None
        table.to_csv(output / f"predictions_{name}.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2)); (output / "metrics_robustness.json").write_text(json.dumps(robustness_report, indent=2)); print(json.dumps(report, indent=2)); print(json.dumps(robustness_report, indent=2))


if __name__ == "__main__": main()
