from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "target_created_at" in df.columns:
        dt = pd.to_datetime(df["target_created_at"], errors="coerce")
    elif "created_at" in df.columns:
        dt = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        dt = pd.Series(pd.NaT, index=df.index)

    hour = dt.dt.hour.fillna(0).astype(float)

    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    return df


def encode_target(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)

    if mode == "raw":
        return y

    if mode == "log1p":
        return np.log1p(y)

    raise ValueError(f"Unknown target mode: {mode}")


def inverse_target(y_pred: np.ndarray, mode: str) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=float)

    if mode == "raw":
        return y_pred

    if mode == "log1p":
        return np.expm1(y_pred)

    raise ValueError(f"Unknown target mode: {mode}")


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


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]):
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


def get_feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_T7_front_sequence_manifest_purged_block_split.csv",
    )

    parser.add_argument("--target-mode", default="log1p", choices=["raw", "log1p"])

    parser.add_argument(
        "--numeric-cols",
        default="Temperature,Humidity,hour_sin,hour_cos",
    )

    parser.add_argument(
        "--categorical-cols",
        default="Season,Day_or_Night",
    )

    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/tabular_context_only_purged_block",
    )

    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures/tabular_context_only_purged_block",
    )

    args = parser.parse_args()

    sequence_manifest = Path(args.sequence_manifest)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sequence_manifest)
    df.columns = [str(c).strip() for c in df.columns]
    df = add_time_features(df)

    # Ignore purged rows.
    df = df[df["split_date_chrono"].isin(["train", "val", "test"])].copy()

    numeric_cols = [c.strip() for c in args.numeric_cols.split(",") if c.strip()]
    categorical_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]

    for c in numeric_cols + categorical_cols:
        if c not in df.columns:
            raise ValueError(f"Requested feature column not found: {c}")

    required = [
        "sequence_id",
        "split_date_chrono",
        "target_value",
        "target_row_id",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Sequence manifest missing columns: {missing}")

    train_df = df[df["split_date_chrono"] == "train"].copy()
    val_df = df[df["split_date_chrono"] == "val"].copy()
    test_df = df[df["split_date_chrono"] == "test"].copy()

    print("=" * 90)
    print("TRAQID TABULAR CONTEXT-ONLY BASELINE")
    print("=" * 90)
    print("Sequence manifest:", sequence_manifest)
    print("Rows:", len(df))
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    print("Target mode:", args.target_mode)
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    print("\nSplit dates:")
    print("train:", sorted(train_df["date"].unique().tolist()) if "date" in train_df.columns else "")
    print("val  :", sorted(val_df["date"].unique().tolist()) if "date" in val_df.columns else "")
    print("test :", sorted(test_df["date"].unique().tolist()) if "date" in test_df.columns else "")

    X_train = train_df[numeric_cols + categorical_cols]
    X_val = val_df[numeric_cols + categorical_cols]
    X_test = test_df[numeric_cols + categorical_cols]

    y_train_raw = train_df["target_value"].astype(float).to_numpy()
    y_val_raw = val_df["target_value"].astype(float).to_numpy()
    y_test_raw = test_df["target_value"].astype(float).to_numpy()

    y_train = encode_target(y_train_raw, args.target_mode)

    results = []
    prediction_frames = []

    # ------------------------------------------------------------------
    # Mean baseline
    # ------------------------------------------------------------------

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)

    val_pred_model = dummy.predict(np.zeros((len(y_val_raw), 1)))
    test_pred_model = dummy.predict(np.zeros((len(y_test_raw), 1)))

    val_pred = inverse_target(val_pred_model, args.target_mode)
    test_pred = inverse_target(test_pred_model, args.target_mode)

    baseline = {
        "val": compute_metrics(y_val_raw, val_pred),
        "test": compute_metrics(y_test_raw, test_pred),
    }

    print("\nMean baseline:")
    print(json.dumps(baseline, indent=2))

    results.append(
        {
            "model": "mean_baseline",
            "target_mode": args.target_mode,
            **{f"val_{k}": v for k, v in baseline["val"].items()},
            **{f"test_{k}": v for k, v in baseline["test"].items()},
        }
    )

    # ------------------------------------------------------------------
    # Candidate tabular models
    # ------------------------------------------------------------------

    candidate_models = []

    for alpha in [0.1, 1.0, 10.0, 100.0]:
        candidate_models.append(
            (
                f"ridge_alpha_{alpha:g}",
                Pipeline(
                    steps=[
                        ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                        ("model", Ridge(alpha=alpha)),
                    ]
                ),
            )
        )

    candidate_models.append(
        (
            "hist_gradient_boosting",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=400,
                            learning_rate=0.04,
                            max_leaf_nodes=31,
                            min_samples_leaf=40,
                            l2_regularization=0.1,
                            random_state=args.random_state,
                        ),
                    ),
                ]
            ),
        )
    )

    candidate_models.append(
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=12,
                            min_samples_leaf=20,
                            random_state=args.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        )
    )

    best = None

    for model_name, pipe in candidate_models:
        print("\nTraining:", model_name)

        pipe.fit(X_train, y_train)

        val_pred_model = pipe.predict(X_val)
        test_pred_model = pipe.predict(X_test)

        val_pred = inverse_target(val_pred_model, args.target_mode)
        test_pred = inverse_target(test_pred_model, args.target_mode)

        val_metrics = compute_metrics(y_val_raw, val_pred)
        test_metrics = compute_metrics(y_test_raw, test_pred)

        row = {
            "model": model_name,
            "target_mode": args.target_mode,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }

        results.append(row)

        print(
            f"{model_name} | "
            f"VAL RMSE={val_metrics['RMSE']:.3f}, Spearman={val_metrics['Spearman']:.3f} | "
            f"TEST RMSE={test_metrics['RMSE']:.3f}, Spearman={test_metrics['Spearman']:.3f}"
        )

        if best is None or val_metrics["RMSE"] < best["val_RMSE"]:
            best = {
                "model_name": model_name,
                "pipeline": pipe,
                "val_RMSE": val_metrics["RMSE"],
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "val_pred": val_pred,
                "test_pred": test_pred,
            }

    if best is None:
        raise RuntimeError("No model was trained.")

    final_metrics = {
        "baseline": baseline,
        "best_model": best["model_name"],
        "val": best["val_metrics"],
        "test": best["test_metrics"],
    }

    print("\nFINAL BEST TABULAR-ONLY MODEL:")
    print(json.dumps(final_metrics, indent=2))

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_path = out_dir / "tabular_context_only_results.csv"
    metrics_path = out_dir / "metrics_tabular_context_only.json"
    predictions_path = out_dir / "predictions_tabular_context_only.csv"
    config_path = out_dir / "config_tabular_context_only.json"

    results_df.to_csv(results_path, index=False)
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    pred_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "sequence_id": val_df["sequence_id"].values,
                    "target_row_id": val_df["target_row_id"].values,
                    "split": "val",
                    "actual_PM25": y_val_raw,
                    "predicted_PM25": best["val_pred"],
                    "model": best["model_name"],
                }
            ),
            pd.DataFrame(
                {
                    "sequence_id": test_df["sequence_id"].values,
                    "target_row_id": test_df["target_row_id"].values,
                    "split": "test",
                    "actual_PM25": y_test_raw,
                    "predicted_PM25": best["test_pred"],
                    "model": best["model_name"],
                }
            ),
        ],
        ignore_index=True,
    )

    pred_df.to_csv(predictions_path, index=False)

    config = {
        "sequence_manifest": str(sequence_manifest),
        "rows": {
            "all": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_mode": args.target_mode,
        "models_tested": [name for name, _ in candidate_models],
        "best_model": best["model_name"],
        "warning": "Uses only non-target context features. PM10, AQI, and AQI category are intentionally excluded to avoid label leakage.",
    }

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    save_scatter(
        y_val_raw,
        best["val_pred"],
        fig_dir / "scatter_val_tabular_context_only.png",
        f"TRAQID Tabular-only VAL: {best['model_name']}",
    )

    save_scatter(
        y_test_raw,
        best["test_pred"],
        fig_dir / "scatter_test_tabular_context_only.png",
        f"TRAQID Tabular-only TEST: {best['model_name']}",
    )

    print("\nSaved:")
    print(" -", results_path)
    print(" -", metrics_path)
    print(" -", predictions_path)
    print(" -", config_path)
    print(" - figures:", fig_dir)


if __name__ == "__main__":
    main()