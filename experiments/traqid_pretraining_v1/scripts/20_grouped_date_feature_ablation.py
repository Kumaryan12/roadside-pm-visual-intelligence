from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "PM2.5"


# ============================================================
# Utilities
# ============================================================

def make_onehot():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
    }


def normalize_row_id(df, fallback=None):
    df = df.copy()

    if "row_id" not in df.columns:
        if fallback and fallback in df.columns:
            df = df.rename(columns={fallback: "row_id"})
        else:
            raise ValueError("row_id column missing.")

    df["row_id"] = pd.to_numeric(
        df["row_id"],
        errors="coerce",
    )

    df = df.dropna(subset=["row_id"]).copy()
    df["row_id"] = df["row_id"].astype(int)

    return df


def add_time_features(df):
    df = df.copy()

    if "created_at" not in df.columns:
        raise ValueError("created_at missing.")

    dt = pd.to_datetime(
        df["created_at"],
        errors="coerce",
    )

    df["date"] = dt.dt.date.astype(str)

    hour = dt.dt.hour.fillna(0).astype(float)
    dayofweek = dt.dt.dayofweek.fillna(0).astype(float)
    month = dt.dt.month.fillna(1).astype(float)

    df["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )
    df["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    df["dayofweek_sin"] = np.sin(
        2 * np.pi * dayofweek / 7
    )
    df["dayofweek_cos"] = np.cos(
        2 * np.pi * dayofweek / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )
    df["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    return df


def forbidden_feature(column):
    c = column.lower()

    forbidden_tokens = [
        "pm2.5",
        "pm25",
        "pm_2_5",
        "pm10",
        "aqi",
        "target",
        "actual",
        "predicted",
        "residual",
        "error",
        "split",
        "path",
        "file",
        "filename",
        "image",
        "status",
    ]

    return any(
        token in c
        for token in forbidden_tokens
    )


# ============================================================
# Dataset preparation
# ============================================================

def prepare_dataset(
    manifest_path,
    yolo_path,
    road_path,
    fold_assignment_path,
):
    manifest = pd.read_csv(manifest_path)
    yolo = pd.read_csv(yolo_path)
    road = pd.read_csv(road_path)
    folds = pd.read_csv(fold_assignment_path)

    manifest = normalize_row_id(manifest)
    yolo = normalize_row_id(yolo)
    road = normalize_row_id(
        road,
        fallback="sample_index",
    )

    if TARGET not in manifest.columns:
        raise ValueError(
            f"{TARGET} missing from manifest."
        )

    manifest = add_time_features(manifest)

    manifest = manifest.dropna(
        subset=[
            "created_at",
            "date",
            TARGET,
        ]
    ).copy()

    folds["date"] = folds["date"].astype(str)

    folds["group_fold"] = pd.to_numeric(
        folds["group_fold"],
        errors="raise",
    ).astype(int)

    manifest = manifest.merge(
        folds[
            [
                "date",
                "group_fold",
            ]
        ],
        on="date",
        how="inner",
        validate="many_to_one",
    )

    yolo_numeric = [
        c
        for c in yolo.columns
        if c == "row_id"
        or (
            pd.api.types.is_numeric_dtype(
                yolo[c]
            )
            and not forbidden_feature(c)
        )
    ]

    road_numeric = [
        c
        for c in road.columns
        if c == "row_id"
        or (
            pd.api.types.is_numeric_dtype(
                road[c]
            )
            and not forbidden_feature(c)
        )
    ]

    yolo = yolo[yolo_numeric].copy()
    road = road[road_numeric].copy()

    yolo = yolo.drop_duplicates(
        "row_id",
        keep="first",
    )

    road = road.drop_duplicates(
        "row_id",
        keep="first",
    )

    base_cols = [
        "row_id",
        "created_at",
        "date",
        "group_fold",
        TARGET,
        "Temperature",
        "Humidity",
        "Season",
        "Day_or_Night",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
    ]

    base_cols = [
        c
        for c in base_cols
        if c in manifest.columns
    ]

    merged = manifest[
        base_cols
    ].merge(
        yolo,
        on="row_id",
        how="left",
        suffixes=("", "_yolo"),
    )

    merged = merged.merge(
        road,
        on="row_id",
        how="left",
        suffixes=("", "_road"),
    )

    return merged.sort_values(
        [
            "created_at",
            "row_id",
        ]
    ).reset_index(drop=True)


# ============================================================
# Feature family detection
# ============================================================

def detect_feature_groups(df):
    metadata = {
        "row_id",
        "created_at",
        "date",
        "group_fold",
        TARGET,
    }

    tabular_core = [
        c
        for c in [
            "Temperature",
            "Humidity",
            "hour_sin",
            "hour_cos",
        ]
        if c in df.columns
    ]

    calendar_context = [
        c
        for c in [
            "Season",
            "Day_or_Night",
            "dayofweek_sin",
            "dayofweek_cos",
            "month_sin",
            "month_cos",
        ]
        if c in df.columns
    ]

    yolo_features = []
    road_features = []

    for column in df.columns:
        if column in metadata:
            continue

        if column in tabular_core:
            continue

        if column in calendar_context:
            continue

        if forbidden_feature(column):
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        c = column.lower()

        road_tokens = [
            "road",
            "lane",
            "area",
            "depth",
            "polygon",
            "mask",
            "surface",
            "texture",
            "edge",
            "visibility",
        ]

        if any(
            token in c
            for token in road_tokens
        ):
            road_features.append(column)
        else:
            yolo_features.append(column)

    groups = {
        "tabular_core_only": tabular_core,

        "calendar_context_only": calendar_context,

        "tabular_plus_calendar": sorted(
            set(
                tabular_core
                + calendar_context
            )
        ),

        "yolo_only": yolo_features,

        "road_only": road_features,

        "yolo_plus_road": sorted(
            set(
                yolo_features
                + road_features
            )
        ),

        "yolo_plus_tabular": sorted(
            set(
                yolo_features
                + tabular_core
            )
        ),

        "road_plus_tabular": sorted(
            set(
                road_features
                + tabular_core
            )
        ),

        "yolo_road_tabular_no_calendar": sorted(
            set(
                yolo_features
                + road_features
                + tabular_core
            )
        ),

        "full": sorted(
            set(
                yolo_features
                + road_features
                + tabular_core
                + calendar_context
            )
        ),
    }

    groups = {
        name: cols
        for name, cols in groups.items()
        if len(cols) > 0
    }

    return groups


# ============================================================
# Fold splitting
# ============================================================

def split_fold(
    df,
    test_group,
    n_folds,
):
    val_group = (
        test_group % n_folds
    ) + 1

    train_groups = [
        fold
        for fold in range(
            1,
            n_folds + 1,
        )
        if fold not in {
            test_group,
            val_group,
        }
    ]

    train_df = df[
        df["group_fold"].isin(
            train_groups
        )
    ].copy()

    val_df = df[
        df["group_fold"] == val_group
    ].copy()

    test_df = df[
        df["group_fold"] == test_group
    ].copy()

    return (
        train_df,
        val_df,
        test_df,
        train_groups,
        val_group,
    )


# ============================================================
# Preprocessing
# ============================================================

def build_preprocessor(
    feature_cols,
):
    categorical_cols = [
        c
        for c in [
            "Season",
            "Day_or_Night",
        ]
        if c in feature_cols
    ]

    numeric_cols = [
        c
        for c in feature_cols
        if c not in categorical_cols
    ]

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            make_onehot(),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# Models
# ============================================================

def make_models(seed):
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=600,
            random_state=seed,
            n_jobs=-1,
            max_features=0.5,
            min_samples_leaf=3,
        ),

        "extra_trees": ExtraTreesRegressor(
            n_estimators=700,
            random_state=seed,
            n_jobs=-1,
            max_features=0.5,
            min_samples_leaf=3,
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            "experiments/"
            "traqid_pretraining_v1/"
            "data/processed/"
            "traqid_paired_manifest_with_splits.csv"
        ),
    )

    parser.add_argument(
        "--yolo-features",
        default=(
            "experiments/"
            "traqid_pretraining_v1/"
            "pipeline_validation/"
            "outputs/"
            "traqid_yolo_advanced_features_front.csv"
        ),
    )

    parser.add_argument(
        "--road-features",
        default=(
            "experiments/"
            "traqid_pretraining_v1/"
            "pipeline_validation/"
            "outputs/"
            "traqid_road_features_front.csv"
        ),
    )

    parser.add_argument(
        "--fold-assignment",
        default=(
            "experiments/"
            "traqid_pretraining_v1/"
            "data/processed/"
            "balanced_date_grouped_T7_cv/"
            "balanced_date_fold_assignment.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "experiments/"
            "traqid_pretraining_v1/"
            "reports/"
            "grouped_date_feature_ablation"
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = prepare_dataset(
        manifest_path=args.manifest,
        yolo_path=args.yolo_features,
        road_path=args.road_features,
        fold_assignment_path=args.fold_assignment,
    )

    feature_groups = detect_feature_groups(
        df
    )

    print("=" * 100)
    print(
        "GROUPED DATE FEATURE-FAMILY ABLATION"
    )
    print("=" * 100)

    print("Rows:", len(df))
    print("Dates:", df["date"].nunique())

    print("\nFeature groups:")

    for group_name, columns in feature_groups.items():
        print(
            f"{group_name}: "
            f"{len(columns)} features"
        )

    all_results = []
    all_predictions = []

    for fold in range(
        1,
        args.folds + 1,
    ):
        (
            train_df,
            val_df,
            test_df,
            train_groups,
            val_group,
        ) = split_fold(
            df,
            test_group=fold,
            n_folds=args.folds,
        )

        print("\n" + "=" * 100)
        print(f"FOLD {fold}")
        print("=" * 100)

        print(
            "Train:",
            len(train_df),
            "Val:",
            len(val_df),
            "Test:",
            len(test_df),
        )

        y_train = train_df[
            TARGET
        ].to_numpy(dtype=float)

        y_val = val_df[
            TARGET
        ].to_numpy(dtype=float)

        y_test = test_df[
            TARGET
        ].to_numpy(dtype=float)

        baseline_val = np.full(
            len(y_val),
            y_train.mean(),
        )

        baseline_test = np.full(
            len(y_test),
            y_train.mean(),
        )

        for split_name, y_true, y_pred in [
            (
                "val",
                y_val,
                baseline_val,
            ),
            (
                "test",
                y_test,
                baseline_test,
            ),
        ]:
            metrics = compute_metrics(
                y_true,
                y_pred,
            )

            all_results.append(
                {
                    "cv_fold": fold,
                    "feature_group": "mean_baseline",
                    "model": "mean_baseline",
                    "split": split_name,
                    "n_features": 0,
                    **metrics,
                }
            )

        for (
            feature_group_name,
            feature_cols,
        ) in feature_groups.items():

            print(
                f"\nFeature group: "
                f"{feature_group_name} "
                f"({len(feature_cols)})"
            )

            preprocessor = build_preprocessor(
                feature_cols
            )

            X_train = preprocessor.fit_transform(
                train_df[feature_cols]
            )

            X_val = preprocessor.transform(
                val_df[feature_cols]
            )

            X_test = preprocessor.transform(
                test_df[feature_cols]
            )

            X_train = np.asarray(
                X_train,
                dtype=np.float32,
            )

            X_val = np.asarray(
                X_val,
                dtype=np.float32,
            )

            X_test = np.asarray(
                X_test,
                dtype=np.float32,
            )

            models = make_models(
                args.seed + fold
            )

            for model_name, model in models.items():
                print(
                    "Training:",
                    model_name
                )

                model.fit(
                    X_train,
                    y_train,
                )

                val_pred = model.predict(
                    X_val
                )

                test_pred = model.predict(
                    X_test
                )

                for (
                    split_name,
                    split_df,
                    y_true,
                    y_pred,
                ) in [
                    (
                        "val",
                        val_df,
                        y_val,
                        val_pred,
                    ),
                    (
                        "test",
                        test_df,
                        y_test,
                        test_pred,
                    ),
                ]:
                    metrics = compute_metrics(
                        y_true,
                        y_pred,
                    )

                    all_results.append(
                        {
                            "cv_fold": fold,
                            "feature_group": feature_group_name,
                            "model": model_name,
                            "split": split_name,
                            "n_features": len(
                                feature_cols
                            ),
                            **metrics,
                        }
                    )

                    prediction_df = split_df[
                        [
                            "row_id",
                            "created_at",
                            "date",
                            "group_fold",
                            TARGET,
                        ]
                    ].copy()

                    prediction_df = (
                        prediction_df.rename(
                            columns={
                                TARGET:
                                "actual_PM2.5"
                            }
                        )
                    )

                    prediction_df[
                        "predicted_PM2.5"
                    ] = y_pred

                    prediction_df[
                        "cv_fold"
                    ] = fold

                    prediction_df[
                        "split"
                    ] = split_name

                    prediction_df[
                        "feature_group"
                    ] = feature_group_name

                    prediction_df[
                        "model"
                    ] = model_name

                    all_predictions.append(
                        prediction_df
                    )

    results_df = pd.DataFrame(
        all_results
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    results_df.to_csv(
        out_dir
        / "all_fold_feature_ablation_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        out_dir
        / "all_fold_feature_ablation_predictions.csv",
        index=False,
    )

    test_results = results_df[
        results_df["split"] == "test"
    ].copy()

    aggregate_rows = []

    for (
        feature_group,
        model_name,
    ), group in test_results.groupby(
        [
            "feature_group",
            "model",
        ]
    ):
        aggregate_rows.append(
            {
                "feature_group":
                feature_group,

                "model":
                model_name,

                "n_features":
                int(
                    group[
                        "n_features"
                    ].iloc[0]
                ),

                "mean_R2":
                float(
                    group["R2"].mean()
                ),

                "std_R2":
                float(
                    group["R2"].std(
                        ddof=1
                    )
                ),

                "mean_RMSE":
                float(
                    group["RMSE"].mean()
                ),

                "std_RMSE":
                float(
                    group["RMSE"].std(
                        ddof=1
                    )
                ),

                "mean_MAE":
                float(
                    group["MAE"].mean()
                ),

                "std_MAE":
                float(
                    group["MAE"].std(
                        ddof=1
                    )
                ),

                "folds":
                int(len(group)),
            }
        )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    ).sort_values(
        [
            "mean_R2",
            "mean_RMSE",
        ],
        ascending=[
            False,
            True,
        ],
    )

    aggregate_df.to_csv(
        out_dir
        / "aggregate_feature_ablation_metrics.csv",
        index=False,
    )

    pooled_rows = []

    test_predictions = predictions_df[
        predictions_df["split"] == "test"
    ].copy()

    for (
        feature_group,
        model_name,
    ), group in test_predictions.groupby(
        [
            "feature_group",
            "model",
        ]
    ):
        metrics = compute_metrics(
            group[
                "actual_PM2.5"
            ].to_numpy(),

            group[
                "predicted_PM2.5"
            ].to_numpy(),
        )

        pooled_rows.append(
            {
                "feature_group":
                feature_group,

                "model":
                model_name,

                "rows":
                len(group),

                "unique_dates":
                group[
                    "date"
                ].nunique(),

                **metrics,
            }
        )

    pooled_df = pd.DataFrame(
        pooled_rows
    ).sort_values(
        [
            "R2",
            "RMSE",
        ],
        ascending=[
            False,
            True,
        ],
    )

    pooled_df.to_csv(
        out_dir
        / "pooled_feature_ablation_metrics.csv",
        index=False,
    )

    config = {
        "manifest":
        args.manifest,

        "yolo_features":
        args.yolo_features,

        "road_features":
        args.road_features,

        "fold_assignment":
        args.fold_assignment,

        "rows":
        len(df),

        "dates":
        int(
            df["date"].nunique()
        ),

        "folds":
        args.folds,

        "feature_groups":
        feature_groups,
    }

    (
        out_dir / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print(
        "GROUPED DATE FEATURE ABLATION COMPLETE"
    )
    print("=" * 100)

    print(
        "\nAGGREGATE TEST RESULTS"
    )

    print(
        aggregate_df.to_string(
            index=False
        )
    )

    print(
        "\nPOOLED TEST RESULTS"
    )

    print(
        pooled_df.to_string(
            index=False
        )
    )

    print("\nSaved:")

    print(
        out_dir
        / "all_fold_feature_ablation_metrics.csv"
    )

    print(
        out_dir
        / "aggregate_feature_ablation_metrics.csv"
    )

    print(
        out_dir
        / "pooled_feature_ablation_metrics.csv"
    )


if __name__ == "__main__":
    main()