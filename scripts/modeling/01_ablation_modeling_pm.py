from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_CSV = Path(
    "outputs/features/particle_density_modeling_table_idd_finetuned_osm_v2.csv"
)

DEFAULT_OUTPUT_DIR = Path("outputs/modeling/ablation_pm2_idd_finetuned_osm_v2")


def is_pm_column(col: str) -> bool:
    c = col.upper()
    return "PM" in c or "NPM" in c


def is_bad_feature_column(col: str) -> bool:
    c = col.lower()

    bad_terms = [
        "path",
        "key",
        "status",
        "error",
        "model",
        "timestamp",
        "time",
        "date",
        "sample_index",
        "sample_id",
        "run_id",
        "unix",
        "offset",
        "latitude",
        "longitude",
        "location_key",
        "available",
    ]

    return any(term in c for term in bad_terms)


def unique_keep_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def rmse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def safe_spearman(y_true, y_pred):
    try:
        corr, _ = spearmanr(y_true, y_pred)
        if np.isnan(corr):
            return np.nan
        return float(corr)
    except Exception:
        return np.nan


def build_feature_groups(df, target):
    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()

    # Do not use PM columns as predictors.
    # Example: do not use PM10 to predict PM2.5.
    candidate_cols = [
        c for c in numeric_cols
        if c != target
        and not is_pm_column(c)
        and not is_bad_feature_column(c)
    ]

    env_cols = [
        c for c in candidate_cols
        if (
            "rh" == c.lower()
            or "humidity" in c.lower()
            or "temp" in c.lower()
            or "dry_air" in c.lower()
        )
    ]

    vehicle_cols = [
        c for c in candidate_cols
        if c.startswith("idd_")
    ]

    road_cols = [
        c for c in candidate_cols
        if c.startswith("road_")
    ]

    osm_cols = [
        c for c in candidate_cols
        if (
            c.startswith("osm_")
            or c.endswith("_250m")
            or "road_segment_count" in c
            or "total_road_length" in c
            or "road_count" in c
        )
    ]

    interaction_cols = [
        c for c in candidate_cols
        if (
            "_x_" in c
            or c.startswith("vehicle_resuspension")
            or c.startswith("resuspension_x")
            or c.startswith("exhaust_x")
        )
    ]

    groups = {
        "environment_only": env_cols,
        "vehicle_only": vehicle_cols,
        "road_only": road_cols,
        "osm_only": osm_cols,

        "vehicle_plus_road": unique_keep_order(
            vehicle_cols + road_cols
        ),

        "vehicle_road_interactions": unique_keep_order(
            vehicle_cols + road_cols + interaction_cols
        ),

        "full_without_osm": unique_keep_order(
            env_cols + vehicle_cols + road_cols + interaction_cols
        ),

        "full_with_osm": unique_keep_order(
            env_cols + vehicle_cols + road_cols + osm_cols + interaction_cols
        ),
    }

    # Remove empty groups.
    groups = {k: v for k, v in groups.items() if len(v) > 0}

    return groups


def make_models(random_state=42):
    models = {
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        random_state=random_state,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_dummy(df, y, splits):
    rows = []
    pred_rows = []

    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(np.zeros((len(y_train), 1)), y_train)

        pred = dummy.predict(np.zeros((len(y_test), 1)))

        rows.append({
            "feature_group": "dummy_baseline",
            "model": "dummy_mean",
            "split": split_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_features": 0,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": rmse(y_test, pred),
            "r2": r2_score(y_test, pred),
            "spearman_pred_vs_true": safe_spearman(y_test, pred),
        })

        for i, p in zip(test_idx, pred):
            pred_rows.append({
                "feature_group": "dummy_baseline",
                "model": "dummy_mean",
                "split": split_id,
                "row_index": int(i),
                "y_true": float(y.iloc[i]),
                "y_pred": float(p),
            })

    return rows, pred_rows


def evaluate_group(df, y, feature_group_name, feature_cols, models, splits):
    rows = []
    pred_rows = []

    X = df[feature_cols].copy()

    for model_name, model in models.items():
        for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            pred = model.predict(X_test)

            rows.append({
                "feature_group": feature_group_name,
                "model": model_name,
                "split": split_id,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features": len(feature_cols),
                "mae": mean_absolute_error(y_test, pred),
                "rmse": rmse(y_test, pred),
                "r2": r2_score(y_test, pred),
                "spearman_pred_vs_true": safe_spearman(y_test, pred),
            })

            for i, p in zip(test_idx, pred):
                pred_rows.append({
                    "feature_group": feature_group_name,
                    "model": model_name,
                    "split": split_id,
                    "row_index": int(i),
                    "y_true": float(y.iloc[i]),
                    "y_pred": float(p),
                })

    return rows, pred_rows


def fit_full_rf_importance(df, y, feature_cols, output_path):
    X = df[feature_cols].copy()

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=800,
                    random_state=42,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X, y)

    rf = model.named_steps["model"]

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    imp.to_csv(output_path, index=False)
    return imp


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--target", default="value.sPM2")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--require-image", action="store_true")
    parser.add_argument("--require-osm", action="store_true")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    print("\nAblation modeling")
    print("=" * 70)
    print("Input CSV:", input_csv)
    print("Target:", args.target)

    # Optional filtering
    if args.require_image and "image_features_available" in df.columns:
        before = len(df)
        df = df[df["image_features_available"] == True].copy()
        print(f"Filtered image_features_available: {before} -> {len(df)}")

    if args.require_osm and "osm_features_available" in df.columns:
        before = len(df)
        df = df[df["osm_features_available"] == True].copy()
        print(f"Filtered osm_features_available: {before} -> {len(df)}")

    # Sort chronologically to avoid random leakage
    if "sample_index" in df.columns:
        df = df.sort_values("sample_index").reset_index(drop=True)
    elif "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp_parsed").reset_index(drop=True)

    # Keep rows with valid target
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df[df[args.target].notna()].copy().reset_index(drop=True)

    y = df[args.target]

    print("Rows used:", len(df))
    print("Target stats:")
    print(y.describe())

    if len(df) < 40:
        raise ValueError("Too few rows for ablation modeling.")

    n_splits = min(args.n_splits, max(2, len(df) // 40))
    print("TimeSeriesSplit n_splits:", n_splits)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(df))

    feature_groups = build_feature_groups(df, args.target)

    print("\nFeature groups:")
    for group_name, cols in feature_groups.items():
        print(f"{group_name}: {len(cols)} features")

    with open(output_dir / "ablation_feature_groups.json", "w") as f:
        json.dump(feature_groups, f, indent=2)

    models = make_models(random_state=42)

    all_rows = []
    all_pred_rows = []

    dummy_rows, dummy_pred_rows = evaluate_dummy(df, y, splits)
    all_rows.extend(dummy_rows)
    all_pred_rows.extend(dummy_pred_rows)

    for group_name, cols in feature_groups.items():
        print(f"\nEvaluating group: {group_name} ({len(cols)} features)")
        rows, pred_rows = evaluate_group(
            df=df,
            y=y,
            feature_group_name=group_name,
            feature_cols=cols,
            models=models,
            splits=splits,
        )

        all_rows.extend(rows)
        all_pred_rows.extend(pred_rows)

    results = pd.DataFrame(all_rows)
    predictions = pd.DataFrame(all_pred_rows)

    results_csv = output_dir / "ablation_results_by_split.csv"
    predictions_csv = output_dir / "ablation_predictions.csv"

    results.to_csv(results_csv, index=False)
    predictions.to_csv(predictions_csv, index=False)

    summary = (
        results
        .groupby(["feature_group", "model"])
        .agg(
            n_features=("n_features", "max"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            spearman_mean=("spearman_pred_vs_true", "mean"),
            spearman_std=("spearman_pred_vs_true", "std"),
        )
        .reset_index()
        .sort_values(["rmse_mean", "mae_mean"], ascending=[True, True])
    )

    summary_csv = output_dir / "ablation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\nSaved:")
    print("Results by split:", results_csv)
    print("Summary:", summary_csv)
    print("Predictions:", predictions_csv)

    print("\nAblation summary sorted by RMSE:")
    print(summary.to_string(index=False))

    # Feature importance for full model if available
    if "full_with_osm" in feature_groups:
        importance_path = output_dir / "rf_feature_importance_full_with_osm.csv"
        imp = fit_full_rf_importance(
            df=df,
            y=y,
            feature_cols=feature_groups["full_with_osm"],
            output_path=importance_path,
        )
        print("\nSaved RF feature importance:", importance_path)
        print("\nTop RF importances:")
        print(imp.head(30).to_string(index=False))

    elif "full_without_osm" in feature_groups:
        importance_path = output_dir / "rf_feature_importance_full_without_osm.csv"
        imp = fit_full_rf_importance(
            df=df,
            y=y,
            feature_cols=feature_groups["full_without_osm"],
            output_path=importance_path,
        )
        print("\nSaved RF feature importance:", importance_path)
        print("\nTop RF importances:")
        print(imp.head(30).to_string(index=False))


if __name__ == "__main__":
    main()