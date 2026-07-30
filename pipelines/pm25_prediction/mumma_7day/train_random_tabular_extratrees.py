"""Exploratory random-row MUMMA retraining of the TRAQID tabular ExtraTrees model.

The feature contract matches the historical 13-feature TRAQID experiment. An
optional affine calibration is fitted only on validation predictions, and the
locked test partition is evaluated once. Random rows from the same days/runs
make this unsuitable as evidence of temporal or unseen-day generalisation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "Temperature",
    "Humidity",
    "hour_num",
    "month_num",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]
CATEGORICAL_FEATURES = ["Season", "Day_or_Night"]


def season_from_month(month: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5, 6])],
            ["Winter", "Summer"],
            default="Monsoon",
        ),
        index=month.index,
    )


def make_features(
    source: pd.DataFrame,
    timestamp_col: str,
    temperature_col: str,
    humidity_col: str,
) -> pd.DataFrame:
    timestamp = pd.to_datetime(source[timestamp_col], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"{int(timestamp.isna().sum())} timestamps could not be parsed")
    frame = pd.DataFrame(index=source.index)
    frame["Temperature"] = pd.to_numeric(source[temperature_col], errors="coerce")
    frame["Humidity"] = pd.to_numeric(source[humidity_col], errors="coerce")
    frame["hour_num"] = timestamp.dt.hour
    frame["month_num"] = timestamp.dt.month
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour_num"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour_num"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month_num"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month_num"] / 12)
    frame["Season"] = season_from_month(frame["month_num"])
    frame["Day_or_Night"] = np.where(frame["hour_num"] < 18, "Day", "Night")
    return frame


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
        "bias": float(np.mean(y_pred - y_true)),
    }


def adjacency_audit(source: pd.DataFrame, split: np.ndarray) -> dict[str, float | int]:
    audit = source[["run_id", "sample_timestamp"]].copy()
    audit["split"] = split
    audit["original_index"] = np.arange(len(audit))
    audit = audit.sort_values(["run_id", "sample_timestamp"])
    prev_split = audit.groupby("run_id")["split"].shift(1)
    next_split = audit.groupby("run_id")["split"].shift(-1)
    test = audit["split"].eq("test")
    adjacent_train = test & (prev_split.eq("train") | next_split.eq("train"))
    date_splits = pd.DataFrame(
        {"date": source["date"].to_numpy(), "split": split}
    ).groupby("date")["split"].nunique()
    return {
        "test_rows": int(test.sum()),
        "test_rows_adjacent_to_train_row": int(adjacent_train.sum()),
        "test_fraction_adjacent_to_train_row": float(adjacent_train.sum() / test.sum()),
        "dates_shared_across_splits": int(date_splits.ge(2).sum()),
        "warning": "Ten-second neighbours and collection dates cross random partitions.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamp-col", default="sample_timestamp")
    parser.add_argument("--temperature-col", default="temp")
    parser.add_argument("--humidity-col", default="rh")
    parser.add_argument("--target-col", default="sPM2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=700)
    args = parser.parse_args()

    source = pd.read_csv(args.modeling_table)
    required = {
        args.timestamp_col,
        args.temperature_col,
        args.humidity_col,
        args.target_col,
        "run_id",
        "date",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    target = pd.to_numeric(source[args.target_col], errors="coerce")
    keep = target.notna()
    source = source.loc[keep].reset_index(drop=True)
    target = target.loc[keep].reset_index(drop=True).to_numpy(dtype=float)
    features = make_features(
        source, args.timestamp_col, args.temperature_col, args.humidity_col
    )

    bins = pd.qcut(target, q=10, labels=False, duplicates="drop")
    all_indices = np.arange(len(source))
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=args.seed,
        stratify=bins,
    )
    train_bins = np.asarray(bins)[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=args.seed,
        stratify=train_bins,
    )

    medians = features.iloc[train_idx][["Temperature", "Humidity"]].median()
    features[["Temperature", "Humidity"]] = features[["Temperature", "Humidity"]].fillna(
        medians
    )
    categories = pd.get_dummies(
        features.iloc[train_idx][CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
    ).columns.tolist()
    categorical = pd.get_dummies(
        features[CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
    ).reindex(columns=categories, fill_value=False)
    raw_matrix = np.hstack(
        [
            features[NUMERIC_FEATURES].to_numpy(dtype=float),
            categorical.to_numpy(dtype=float),
        ]
    )
    feature_names = NUMERIC_FEATURES + categories
    scaler = StandardScaler()
    x_train = scaler.fit_transform(raw_matrix[train_idx])
    x_val = scaler.transform(raw_matrix[val_idx])
    x_test = scaler.transform(raw_matrix[test_idx])

    model = ExtraTreesRegressor(
        n_estimators=args.trees,
        max_depth=None,
        min_samples_leaf=1,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x_train, target[train_idx])
    val_raw = np.clip(model.predict(x_val), 0, None)
    test_raw = np.clip(model.predict(x_test), 0, None)

    calibrator = LinearRegression().fit(val_raw.reshape(-1, 1), target[val_idx])
    val_calibrated = np.clip(calibrator.predict(val_raw.reshape(-1, 1)), 0, None)
    test_calibrated = np.clip(calibrator.predict(test_raw.reshape(-1, 1)), 0, None)
    val_raw_metrics = regression_metrics(target[val_idx], val_raw)
    val_calibrated_metrics = regression_metrics(target[val_idx], val_calibrated)
    selected = (
        "affine_validation_calibration"
        if val_calibrated_metrics["RMSE"] < val_raw_metrics["RMSE"]
        else "raw_model"
    )
    test_selected = test_calibrated if selected.startswith("affine") else test_raw

    split = np.full(len(source), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    audit = adjacency_audit(source, split)
    results = {
        "protocol": "random_rows_stratified_by_target_decile",
        "reportable_as_generalization": False,
        "exploratory_only": True,
        "target": args.target_col,
        "features": feature_names,
        "pm_or_opc_predictors_used": False,
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "model": {
            "type": "ExtraTreesRegressor",
            "trees": args.trees,
            "max_depth": None,
            "min_samples_leaf": 1,
            "seed": args.seed,
        },
        "calibration": {
            "type": "affine_linear_regression",
            "fit_partition": "validation only",
            "slope": float(calibrator.coef_[0]),
            "intercept": float(calibrator.intercept_),
            "selected_by_validation_rmse": selected,
        },
        "metrics": {
            "val_raw": val_raw_metrics,
            "val_calibrated": val_calibrated_metrics,
            "test_raw": regression_metrics(target[test_idx], test_raw),
            "test_calibrated": regression_metrics(target[test_idx], test_calibrated),
            "test_selected": regression_metrics(target[test_idx], test_selected),
        },
        "leakage_audit": audit,
        "warning": (
            "This random-row score is affected by temporal autocorrelation and must not be "
            "reported as unseen-day or deployment performance."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": 1,
        "model": model,
        "scaler": scaler,
        "calibrator": calibrator,
        "selected_prediction": selected,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoded_categories": categories,
        "feature_names": feature_names,
        "imputation_medians": medians.to_dict(),
        "training_dataset": "MUMMA five-day",
        "results": results,
    }
    joblib.dump(bundle, args.output_dir / "model.joblib", compress=("xz", 3))
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    prediction_table = source.iloc[test_idx][
        ["sample_id", "run_id", "date", args.timestamp_col, args.target_col]
    ].copy()
    prediction_table["prediction_raw"] = test_raw
    prediction_table["prediction_calibrated"] = test_calibrated
    prediction_table["prediction_selected"] = test_selected
    prediction_table.to_csv(args.output_dir / "test_predictions.csv", index=False)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
