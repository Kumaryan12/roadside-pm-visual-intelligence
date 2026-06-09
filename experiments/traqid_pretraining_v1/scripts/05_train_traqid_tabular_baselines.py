from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = ["PM2.5", "PM10", "aqi"]


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def safe_corr(y_true, y_pred, method: str):
    s1 = pd.Series(y_true)
    s2 = pd.Series(y_pred)
    if s1.nunique() < 2 or s2.nunique() < 2:
        return np.nan
    return float(s1.corr(s2, method=method))


def evaluate(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    y_pred = np.maximum(y_pred, 0)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": safe_corr(y_true, y_pred, "pearson"),
        "Spearman": safe_corr(y_true, y_pred, "spearman"),
    }


def save_scatter(y_true, y_pred, title, out_path):
    y_pred = np.maximum(np.asarray(y_pred), 0)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)

    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def add_time_features(df):
    df = df.copy()
    df["created_at_parsed"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["hour_of_day"] = df["created_at_parsed"].dt.hour.fillna(0)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/tabular_baselines",
    )

    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures/tabular_baselines",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df = add_time_features(df)

    # Use only valid paired observations.
    df = df[(df["front_exists"] == True) & (df["rear_exists"] == True)].copy()

    # For tabular/weather baseline, remove impossible weather readings.
    df = df[
        df["Temperature"].between(-10, 60)
        & df["Humidity"].between(0, 100)
    ].copy()

    numeric_cols = ["Temperature", "Humidity", "hour_sin", "hour_cos"]
    categorical_cols = ["Season", "Day_or_Night"]

    feature_cols = numeric_cols + categorical_cols

    train_df = df[df["split_date_chrono"] == "train"].copy()
    val_df = df[df["split_date_chrono"] == "val"].copy()
    test_df = df[df["split_date_chrono"] == "test"].copy()

    print("=" * 90)
    print("TRAQID TABULAR BASELINES")
    print("=" * 90)
    print("Rows after cleaning:", len(df))
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    print("Features:", feature_cols)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "elasticnet": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=20000)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=14,
                        min_samples_leaf=20,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    results = []
    pred_rows = []

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    for target in TARGETS:
        print("\n" + "=" * 60)
        print("Target:", target)
        print("=" * 60)

        y_train_raw = train_df[target].astype(float).values
        y_val_raw = val_df[target].astype(float).values
        y_test_raw = test_df[target].astype(float).values

        for target_mode in ["raw", "log1p"]:
            if target_mode == "raw":
                y_train = y_train_raw
            else:
                y_train = np.log1p(y_train_raw)

            for model_name, model_template in models.items():
                model = clone(model_template)
                model.fit(X_train, y_train)

                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)

                if target_mode == "log1p":
                    val_pred = np.expm1(val_pred)
                    test_pred = np.expm1(test_pred)

                val_pred = np.maximum(val_pred, 0)
                test_pred = np.maximum(test_pred, 0)

                val_metrics = evaluate(y_val_raw, val_pred)
                test_metrics = evaluate(y_test_raw, test_pred)

                row_base = {
                    "target": target,
                    "target_mode": target_mode,
                    "model": model_name,
                }

                for k, v in val_metrics.items():
                    row_base[f"val_{k}"] = v

                for k, v in test_metrics.items():
                    row_base[f"test_{k}"] = v

                results.append(row_base)

                print(
                    f"{target_mode:5s} | {model_name:22s} | "
                    f"VAL MAE={val_metrics['MAE']:.3f}, RMSE={val_metrics['RMSE']:.3f}, R2={val_metrics['R2']:.3f}, Spearman={val_metrics['Spearman']:.3f} | "
                    f"TEST MAE={test_metrics['MAE']:.3f}, RMSE={test_metrics['RMSE']:.3f}, R2={test_metrics['R2']:.3f}, Spearman={test_metrics['Spearman']:.3f}"
                )

                scatter_name = f"scatter_{target.replace('.', '_')}_{target_mode}_{model_name}.png"

                save_scatter(
                    y_test_raw,
                    test_pred,
                    f"TRAQID Test: {target} | {target_mode} | {model_name}",
                    fig_dir / scatter_name,
                )

                for split_name, split_df, actual, pred in [
                    ("val", val_df, y_val_raw, val_pred),
                    ("test", test_df, y_test_raw, test_pred),
                ]:
                    temp = pd.DataFrame(
                        {
                            "row_id": split_df["row_id"].values,
                            "created_at": split_df["created_at"].values,
                            "date": split_df["date"].values,
                            "split": split_name,
                            "target": target,
                            "target_mode": target_mode,
                            "model": model_name,
                            "actual": actual,
                            "predicted": pred,
                        }
                    )
                    pred_rows.append(temp)

    results_df = pd.DataFrame(results)
    pred_df = pd.concat(pred_rows, ignore_index=True)

    results_out = out_dir / "traqid_tabular_baseline_results.csv"
    pred_out = out_dir / "traqid_tabular_baseline_predictions.csv"

    results_df.to_csv(results_out, index=False)
    pred_df.to_csv(pred_out, index=False)

    # Best model table by target using validation RMSE first.
    best_val = (
        results_df.sort_values(["target", "val_RMSE"])
        .groupby("target")
        .head(5)
        .reset_index(drop=True)
    )
    best_val.to_csv(out_dir / "traqid_tabular_best_by_val_rmse.csv", index=False)

    summary = {
        "manifest": str(manifest_path),
        "rows_after_cleaning": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "features": feature_cols,
        "targets": TARGETS,
        "warning": (
            "This is a non-image baseline. Vision models must beat or add value "
            "beyond this baseline before we claim visual information is useful."
        ),
    }

    (out_dir / "traqid_tabular_baseline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(" -", results_out)
    print(" -", pred_out)
    print(" -", out_dir / "traqid_tabular_best_by_val_rmse.csv")
    print(" -", out_dir / "traqid_tabular_baseline_summary.json")
    print(" - figures:", fig_dir)

    print("\nBest by validation RMSE:")
    print(best_val.to_string(index=False))


if __name__ == "__main__":
    main()