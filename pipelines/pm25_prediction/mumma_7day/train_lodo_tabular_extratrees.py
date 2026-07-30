"""Leave-one-day-out evaluation of the TRAQID-style tabular ExtraTrees model.

For each outer test day, affine calibration is fitted from leave-one-day-out
predictions generated exclusively within the remaining training days. The
outer test day is never used for preprocessing, fitting, or calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from pipelines.pm25_prediction.mumma_7day.train_random_tabular_extratrees import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    make_features,
    regression_metrics,
)


def prepare_matrix(
    features: pd.DataFrame,
    fit_idx: np.ndarray,
    apply_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str], dict[str, float]]:
    local = features.copy()
    medians = local.iloc[fit_idx][["Temperature", "Humidity"]].median()
    local[["Temperature", "Humidity"]] = local[["Temperature", "Humidity"]].fillna(
        medians
    )
    categories = pd.get_dummies(
        local.iloc[fit_idx][CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
    ).columns.tolist()
    categorical = pd.get_dummies(
        local[CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
    ).reindex(columns=categories, fill_value=False)
    matrix = np.hstack(
        [
            local[NUMERIC_FEATURES].to_numpy(dtype=float),
            categorical.to_numpy(dtype=float),
        ]
    )
    scaler = StandardScaler()
    train = scaler.fit_transform(matrix[fit_idx])
    applied = scaler.transform(matrix[apply_idx])
    return train, applied, scaler, NUMERIC_FEATURES + categories, medians.to_dict()


def new_model(trees: int, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=trees,
        max_depth=None,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamp-col", default="sample_timestamp")
    parser.add_argument("--temperature-col", default="temp")
    parser.add_argument("--humidity-col", default="rh")
    parser.add_argument("--target-col", default="sPM2")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--trees", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist all five outer-fold estimators (requires substantial disk space).",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.modeling_table)
    required = {
        args.timestamp_col,
        args.temperature_col,
        args.humidity_col,
        args.target_col,
        args.date_col,
        "sample_id",
        "run_id",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    target_series = pd.to_numeric(source[args.target_col], errors="coerce")
    source = source.loc[target_series.notna()].reset_index(drop=True)
    target = pd.to_numeric(source[args.target_col], errors="raise").to_numpy(dtype=float)
    features = make_features(
        source, args.timestamp_col, args.temperature_col, args.humidity_col
    )
    dates = source[args.date_col].astype(str).to_numpy()
    unique_dates = sorted(np.unique(dates).tolist())
    if len(unique_dates) < 3:
        raise ValueError("At least three dates are required for nested calibration")

    fold_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    if args.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    for fold, test_date in enumerate(unique_dates, start=1):
        outer_test = np.flatnonzero(dates == test_date)
        outer_train = np.flatnonzero(dates != test_date)
        train_dates = sorted(np.unique(dates[outer_train]).tolist())

        oof_prediction = np.full(len(outer_train), np.nan, dtype=float)
        outer_position = {index: pos for pos, index in enumerate(outer_train)}
        for inner_date in train_dates:
            inner_val = np.flatnonzero((dates == inner_date) & (dates != test_date))
            inner_train = np.flatnonzero((dates != inner_date) & (dates != test_date))
            x_train, x_val, _, _, _ = prepare_matrix(features, inner_train, inner_val)
            inner_model = new_model(args.trees, args.seed)
            inner_model.fit(x_train, target[inner_train])
            inner_pred = np.clip(inner_model.predict(x_val), 0, None)
            for index, prediction in zip(inner_val, inner_pred):
                oof_prediction[outer_position[index]] = prediction
        if np.isnan(oof_prediction).any():
            raise RuntimeError(f"Incomplete inner OOF predictions for outer fold {fold}")

        calibrator = LinearRegression().fit(
            oof_prediction.reshape(-1, 1), target[outer_train]
        )
        x_train, x_test, scaler, feature_names, medians = prepare_matrix(
            features, outer_train, outer_test
        )
        model = new_model(args.trees, args.seed)
        model.fit(x_train, target[outer_train])
        raw = np.clip(model.predict(x_test), 0, None)
        calibrated = np.clip(calibrator.predict(raw.reshape(-1, 1)), 0, None)

        raw_metrics = regression_metrics(target[outer_test], raw)
        calibrated_metrics = regression_metrics(target[outer_test], calibrated)
        fold_rows.append(
            {
                "fold": fold,
                "test_date": test_date,
                "n_train": int(len(outer_train)),
                "n_test": int(len(outer_test)),
                "calibration_slope": float(calibrator.coef_[0]),
                "calibration_intercept": float(calibrator.intercept_),
                **{f"raw_{key}": value for key, value in raw_metrics.items()},
                **{
                    f"calibrated_{key}": value
                    for key, value in calibrated_metrics.items()
                },
            }
        )
        predictions = source.iloc[outer_test][
            ["sample_id", "run_id", args.date_col, args.timestamp_col, args.target_col]
        ].copy()
        predictions["fold"] = fold
        predictions["prediction_raw"] = raw
        predictions["prediction_calibrated"] = calibrated
        prediction_frames.append(predictions)

        if args.save_models:
            joblib.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "calibrator": calibrator,
                    "feature_names": feature_names,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "encoded_categories": feature_names[len(NUMERIC_FEATURES) :],
                    "imputation_medians": medians,
                    "outer_test_date": test_date,
                },
                model_dir / f"fold_{fold}_test_{test_date}.joblib",
                compress=("xz", 3),
            )
        print(
            f"fold={fold} test_date={test_date} "
            f"raw_R2={raw_metrics['R2']:.4f} calibrated_R2={calibrated_metrics['R2']:.4f}",
            flush=True,
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    actual = predictions[args.target_col].to_numpy(dtype=float)
    pooled_raw = regression_metrics(actual, predictions["prediction_raw"].to_numpy())
    pooled_calibrated = regression_metrics(
        actual, predictions["prediction_calibrated"].to_numpy()
    )
    summary = {
        "protocol": "nested_leave_one_date_out",
        "reportable_as_unseen_date_generalization": True,
        "target": args.target_col,
        "outer_dates": unique_dates,
        "outer_folds": len(unique_dates),
        "base_model": "ExtraTreesRegressor",
        "features": "TRAQID-compatible meteorology_and_time_only",
        "pm_or_opc_predictors_used": False,
        "calibration": (
            "Affine calibration fitted on inner leave-one-date-out OOF predictions "
            "from outer-training dates only"
        ),
        "pooled_test_metrics": {
            "raw": pooled_raw,
            "calibrated": pooled_calibrated,
        },
        "mean_fold_metrics": {
            column: float(folds[column].mean())
            for column in folds.columns
            if column.startswith("raw_") or column.startswith("calibrated_")
        },
        "models_saved": args.save_models,
    }
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
