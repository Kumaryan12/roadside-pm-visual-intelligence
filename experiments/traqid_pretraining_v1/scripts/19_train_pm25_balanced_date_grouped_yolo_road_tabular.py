from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "PM2.5"


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def add_time_features(df):
    df = df.copy()
    if "created_at" not in df.columns:
        raise ValueError("created_at missing from manifest.")
    dt = pd.to_datetime(df["created_at"], errors="coerce")
    hour = dt.dt.hour.fillna(0).astype(float)
    dayofweek = dt.dt.dayofweek.fillna(0).astype(float)
    month = dt.dt.month.fillna(1).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["date"] = dt.dt.date.astype(str)
    return df


def normalize_row_id(df, fallback_name=None):
    df = df.copy()
    if "row_id" not in df.columns:
        if fallback_name and fallback_name in df.columns:
            df = df.rename(columns={fallback_name: "row_id"})
        else:
            raise ValueError("row_id column missing.")
    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
    df = df.dropna(subset=["row_id"]).copy()
    df["row_id"] = df["row_id"].astype(int)
    return df


def forbidden_feature(column):
    name = column.lower()
    forbidden_tokens = [
        "pm2.5", "pm25", "pm_2_5", "pm10", "aqi", "target", "actual",
        "predicted", "residual", "error", "split", "date_chrono", "front_path",
        "rear_path", "image_path", "filename", "filepath", "status",
    ]
    return any(token in name for token in forbidden_tokens)


def prepare_dataset(manifest_path, yolo_path, road_path, fold_assignment_path):
    manifest = normalize_row_id(pd.read_csv(manifest_path))
    yolo = normalize_row_id(pd.read_csv(yolo_path))
    road = normalize_row_id(pd.read_csv(road_path), fallback_name="sample_index")
    folds = pd.read_csv(fold_assignment_path)

    if TARGET not in manifest.columns:
        raise ValueError(f"{TARGET} missing from manifest.")

    manifest = add_time_features(manifest)
    manifest = manifest.dropna(subset=["created_at", TARGET, "date"]).copy()

    folds["date"] = folds["date"].astype(str)
    folds["group_fold"] = pd.to_numeric(folds["group_fold"], errors="raise").astype(int)

    manifest = manifest.merge(
        folds[["date", "group_fold"]], on="date", how="inner", validate="many_to_one"
    )

    yolo_keep = [
        c for c in yolo.columns
        if c == "row_id" or (pd.api.types.is_numeric_dtype(yolo[c]) and not forbidden_feature(c))
    ]
    road_keep = [
        c for c in road.columns
        if c == "row_id" or (pd.api.types.is_numeric_dtype(road[c]) and not forbidden_feature(c))
    ]

    yolo = yolo[yolo_keep].drop_duplicates("row_id", keep="first")
    road = road[road_keep].drop_duplicates("row_id", keep="first")

    base_cols = [
        "row_id", "created_at", "date", "group_fold", TARGET,
        "Temperature", "Humidity", "Season", "Day_or_Night",
        "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
        "month_sin", "month_cos",
    ]
    base_cols = [c for c in base_cols if c in manifest.columns]

    merged = manifest[base_cols].merge(yolo, on="row_id", how="left", suffixes=("", "_yolo"))
    merged = merged.merge(road, on="row_id", how="left", suffixes=("", "_road"))
    return merged.sort_values(["created_at", "row_id"]).reset_index(drop=True)


def split_fold(df, test_group, n_folds):
    val_group = (test_group % n_folds) + 1
    train_groups = [f for f in range(1, n_folds + 1) if f not in {test_group, val_group}]
    return (
        df[df["group_fold"].isin(train_groups)].copy(),
        df[df["group_fold"] == val_group].copy(),
        df[df["group_fold"] == test_group].copy(),
        train_groups,
        val_group,
    )


def build_preprocessor(feature_cols):
    categorical_cols = [c for c in ["Season", "Day_or_Night"] if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot()),
            ]),
            categorical_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models(seed):
    return {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=700, random_state=seed, n_jobs=-1,
            max_features=0.5, min_samples_leaf=3,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=900, random_state=seed, n_jobs=-1,
            max_features=0.5, min_samples_leaf=2,
        ),
        "extra_trees_conservative": ExtraTreesRegressor(
            n_estimators=900, random_state=seed, n_jobs=-1,
            max_features=0.35, min_samples_leaf=4,
        ),
        "hgb": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.035, max_leaf_nodes=31,
            l2_regularization=0.05, random_state=seed,
        ),
    }


def save_feature_importance(model, feature_names, out_path):
    if not hasattr(model, "feature_importances_"):
        return
    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )
    parser.add_argument(
        "--yolo-features",
        default="experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_advanced_features_front.csv",
    )
    parser.add_argument(
        "--road-features",
        default="experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_road_features_front.csv",
    )
    parser.add_argument(
        "--fold-assignment",
        default="experiments/traqid_pretraining_v1/data/processed/balanced_date_grouped_T7_cv/balanced_date_fold_assignment.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/pm25_balanced_date_grouped_yolo_road_tabular_cv",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(
        args.manifest, args.yolo_features, args.road_features, args.fold_assignment
    )

    excluded = {"row_id", "created_at", "date", "group_fold", TARGET}
    feature_cols = [
        c for c in df.columns
        if c not in excluded
        and not forbidden_feature(c)
        and (pd.api.types.is_numeric_dtype(df[c]) or c in {"Season", "Day_or_Night"})
    ]

    if not feature_cols:
        raise RuntimeError("No usable features were found.")

    print("=" * 100)
    print("BALANCED DATE-GROUPED YOLO + ROAD + TABULAR PM2.5 CV")
    print("=" * 100)
    print("Rows:", len(df))
    print("Features:", len(feature_cols))
    print("Dates:", df["date"].nunique())
    print("Folds:", sorted(df["group_fold"].unique().tolist()))

    metric_rows = []
    prediction_parts = []

    for fold in range(1, args.folds + 1):
        fold_seed = args.seed + fold
        train_df, val_df, test_df, train_groups, val_group = split_fold(
            df, test_group=fold, n_folds=args.folds
        )

        if min(len(train_df), len(val_df), len(test_df)) == 0:
            raise RuntimeError(
                f"Fold {fold} has an empty split: train={len(train_df)}, "
                f"val={len(val_df)}, test={len(test_df)}"
            )

        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 100)
        print(f"FOLD {fold}")
        print("=" * 100)
        print("Train groups:", train_groups)
        print("Validation group:", val_group)
        print("Test group:", fold)
        print("Train rows:", len(train_df))
        print("Validation rows:", len(val_df))
        print("Test rows:", len(test_df))
        print("Train dates:", sorted(train_df["date"].unique().tolist()))
        print("Validation dates:", sorted(val_df["date"].unique().tolist()))
        print("Test dates:", sorted(test_df["date"].unique().tolist()))

        preprocessor = build_preprocessor(feature_cols)
        x_train = np.asarray(preprocessor.fit_transform(train_df[feature_cols]), dtype=np.float32)
        x_val = np.asarray(preprocessor.transform(val_df[feature_cols]), dtype=np.float32)
        x_test = np.asarray(preprocessor.transform(test_df[feature_cols]), dtype=np.float32)

        y_train = train_df[TARGET].to_numpy(dtype=float)
        y_val = val_df[TARGET].to_numpy(dtype=float)
        y_test = test_df[TARGET].to_numpy(dtype=float)

        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]

        joblib.dump(preprocessor, fold_dir / "preprocessor.joblib")

        for split_name, y_true, y_pred in [
            ("val", y_val, np.full(len(y_val), y_train.mean())),
            ("test", y_test, np.full(len(y_test), y_train.mean())),
        ]:
            metric_rows.append({
                "cv_fold": fold,
                "model": "train_mean_baseline",
                "split": split_name,
                **compute_metrics(y_true, y_pred),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "n_raw_features": len(feature_cols),
                "n_processed_features": x_train.shape[1],
                "train_PM2.5_mean": float(y_train.mean()),
                "val_PM2.5_mean": float(y_val.mean()),
                "test_PM2.5_mean": float(y_test.mean()),
            })

        for model_name, model in make_models(fold_seed).items():
            print("Training:", model_name)
            model.fit(x_train, y_train)
            val_pred = model.predict(x_val)
            test_pred = model.predict(x_test)

            joblib.dump(model, fold_dir / f"{model_name}.joblib")
            save_feature_importance(
                model, feature_names, fold_dir / f"{model_name}_feature_importance.csv"
            )

            for split_name, split_df, y_true, y_pred in [
                ("val", val_df, y_val, val_pred),
                ("test", test_df, y_test, test_pred),
            ]:
                metric_rows.append({
                    "cv_fold": fold,
                    "model": model_name,
                    "split": split_name,
                    **compute_metrics(y_true, y_pred),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "n_raw_features": len(feature_cols),
                    "n_processed_features": x_train.shape[1],
                    "train_PM2.5_mean": float(y_train.mean()),
                    "val_PM2.5_mean": float(y_val.mean()),
                    "test_PM2.5_mean": float(y_test.mean()),
                })

                pred_df = split_df[["row_id", "created_at", "date", "group_fold", TARGET]].copy()
                pred_df = pred_df.rename(columns={TARGET: "actual_PM2.5"})
                pred_df["predicted_PM2.5"] = y_pred
                pred_df["prediction_error"] = y_pred - y_true
                pred_df["absolute_error"] = np.abs(y_pred - y_true)
                pred_df["cv_fold"] = fold
                pred_df["split"] = split_name
                pred_df["model"] = model_name
                prediction_parts.append(pred_df)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    metrics_df.to_csv(out_dir / "all_fold_metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "all_fold_predictions.csv", index=False)

    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    aggregate_rows = []
    for model_name, group in test_metrics.groupby("model"):
        aggregate_rows.append({
            "model": model_name,
            "mean_R2": float(group["R2"].mean()),
            "std_R2": float(group["R2"].std(ddof=1)),
            "mean_RMSE": float(group["RMSE"].mean()),
            "std_RMSE": float(group["RMSE"].std(ddof=1)),
            "mean_MAE": float(group["MAE"].mean()),
            "std_MAE": float(group["MAE"].std(ddof=1)),
            "folds": int(len(group)),
        })

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        ["mean_R2", "mean_RMSE"], ascending=[False, True]
    )
    aggregate_df.to_csv(out_dir / "aggregate_test_metrics.csv", index=False)

    pooled_rows = []
    test_predictions = predictions_df[predictions_df["split"] == "test"]
    for model_name, group in test_predictions.groupby("model"):
        pooled_rows.append({
            "model": model_name,
            "rows": len(group),
            "unique_dates": group["date"].nunique(),
            **compute_metrics(
                group["actual_PM2.5"].to_numpy(),
                group["predicted_PM2.5"].to_numpy(),
            ),
        })

    pooled_df = pd.DataFrame(pooled_rows).sort_values(
        ["R2", "RMSE"], ascending=[False, True]
    )
    pooled_df.to_csv(out_dir / "pooled_test_metrics.csv", index=False)

    (out_dir / "raw_feature_columns.txt").write_text(
        "\n".join(feature_cols), encoding="utf-8"
    )

    (out_dir / "config.json").write_text(
        json.dumps({
            "manifest": args.manifest,
            "yolo_features": args.yolo_features,
            "road_features": args.road_features,
            "fold_assignment": args.fold_assignment,
            "folds": args.folds,
            "seed": args.seed,
            "rows": len(df),
            "dates": int(df["date"].nunique()),
            "raw_features": len(feature_cols),
            "feature_columns": feature_cols,
        }, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("BALANCED DATE-GROUPED YOLO + ROAD + TABULAR CV COMPLETE")
    print("=" * 100)
    print("\nAGGREGATE TEST METRICS")
    print(aggregate_df.to_string(index=False))
    print("\nPOOLED TEST METRICS")
    print(pooled_df.to_string(index=False))


if __name__ == "__main__":
    main()
