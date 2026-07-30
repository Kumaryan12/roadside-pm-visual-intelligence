from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor


TARGET_COLS = ["target_PM2.5", "target_PM10", "target_aqi"]
NUM_COLS = ["Temperature", "Humidity", "hour_sin", "hour_cos"]
CAT_COLS = ["Season", "Day_or_Night"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["target_time"], errors="coerce")
    hour = dt.dt.hour.fillna(0).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return df


def attach_context(seq_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["row_id", "Temperature", "Humidity", "Season", "Day_or_Night"]
    missing = [c for c in cols if c not in base_df.columns]
    if missing:
        raise ValueError(f"Base manifest missing columns: {missing}")

    meta = base_df[cols].rename(columns={"row_id": "target_row_id"})
    out = seq_df.merge(meta, on="target_row_id", how="left")
    out = add_time_features(out)
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )


def build_model(name: str, seed: int):
    name = name.lower()

    if name == "ridge":
        return Pipeline(
            steps=[
                ("prep", make_preprocessor()),
                ("model", MultiOutputRegressor(Ridge(alpha=1.0))),
            ]
        )

    if name == "rf":
        return Pipeline(
            steps=[
                ("prep", make_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=3,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if name == "mlp":
        return Pipeline(
            steps=[
                ("prep", make_preprocessor()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError("model must be one of: ridge, rf, mlp")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest.csv",
    )
    parser.add_argument(
        "--base-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )
    parser.add_argument("--split-col", default="split_random")
    parser.add_argument("--train-name", default="train")
    parser.add_argument("--val-name", default="val")
    parser.add_argument("--test-name", default="test")
    parser.add_argument("--model", default="rf", choices=["ridge", "rf", "mlp"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/tabular_only_multitarget",
    )

    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    seq_df = pd.read_csv(args.sequence_manifest)
    base_df = pd.read_csv(args.base_manifest)

    df = attach_context(seq_df, base_df)

    train_df = df[df[args.split_col] == args.train_name].copy()
    val_df = df[df[args.split_col] == args.val_name].copy()
    test_df = df[df[args.split_col] == args.test_name].copy()

    if len(val_df) == 0:
        val_df = test_df.copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Empty split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    X_train = train_df[NUM_COLS + CAT_COLS]
    y_train = train_df[TARGET_COLS].to_numpy()

    X_val = val_df[NUM_COLS + CAT_COLS]
    y_val = val_df[TARGET_COLS].to_numpy()

    X_test = test_df[NUM_COLS + CAT_COLS]
    y_test = test_df[TARGET_COLS].to_numpy()

    model = build_model(args.model, args.seed)

    print("=" * 90)
    print("TABULAR-ONLY MULTITARGET BASELINE")
    print("=" * 90)
    print("Model:", args.model)
    print("Sequence manifest:", args.sequence_manifest)
    print("Split:", args.split_col, args.train_name, args.val_name, args.test_name)
    print("Train/val/test:", len(train_df), len(val_df), len(test_df))
    print("Numeric:", NUM_COLS)
    print("Categorical:", CAT_COLS)

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    run_name = f"tabular_only_{args.model}_{Path(args.sequence_manifest).stem}_{args.split_col}"
    out_dir = report_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = []

    for split_name, part_df, y_true, y_pred in [
        ("val", val_df, y_val, val_pred),
        ("test", test_df, y_test, test_pred),
    ]:
        for sid, yt, yp in zip(part_df["sequence_id"], y_true, y_pred):
            pred_rows.append(
                {
                    "split": split_name,
                    "sequence_id": int(sid),
                    "actual_PM2.5": float(yt[0]),
                    "predicted_PM2.5": float(yp[0]),
                    "actual_PM10": float(yt[1]),
                    "predicted_PM10": float(yp[1]),
                    "actual_aqi": float(yt[2]),
                    "predicted_aqi": float(yp[2]),
                }
            )

    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions_tabular_only.csv", index=False)

    metrics = {
        "model": args.model,
        "features": {
            "numeric": NUM_COLS,
            "categorical": CAT_COLS,
        },
        "sequence_manifest": args.sequence_manifest,
        "base_manifest": args.base_manifest,
        "split": {
            "split_col": args.split_col,
            "train_name": args.train_name,
            "val_name": args.val_name,
            "test_name": args.test_name,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
        },
        "val": val_metrics,
        "test": test_metrics,
    }

    (out_dir / "metrics_tabular_only.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nVAL:")
    print(json.dumps(val_metrics, indent=2))

    print("\nTEST:")
    print(json.dumps(test_metrics, indent=2))

    print("\nSaved:", out_dir)


if __name__ == "__main__":
    main()