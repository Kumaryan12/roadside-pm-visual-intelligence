"""Export the historical TRAQID tabular-only ExtraTrees model.

This intentionally reproduces the row-random, AQI-stratified experiment that
reported R2=0.81364. It is not an unseen-date validation protocol.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_YOLO = Path(
    "experiments/traqid_pretraining_v1/pipeline_validation/outputs/"
    "traqid_yolo_vehicle_features_front.csv"
)
DEFAULT_ROAD = Path(
    "experiments/traqid_pretraining_v1/pipeline_validation/outputs/"
    "traqid_road_features_front.csv"
)
DEFAULT_OUTPUT = Path(
    "experiments/traqid_pretraining_v1/pipeline_validation/models/"
    "traqid_tabular_extratrees_random_stratified_seed42.joblib"
)

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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    timestamp = pd.to_datetime(out["created_at"], errors="raise")
    out["hour_num"] = timestamp.dt.hour
    out["month_num"] = timestamp.dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_num"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_num"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12)
    return out


def encode_features(
    df: pd.DataFrame,
    *,
    encoded_categories: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    numeric = df[NUMERIC_FEATURES].to_numpy(dtype=float)
    categorical = pd.get_dummies(
        df[CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=False
    )
    if encoded_categories is not None:
        categorical = categorical.reindex(columns=encoded_categories, fill_value=False)
    names = NUMERIC_FEATURES + categorical.columns.tolist()
    return np.hstack([numeric, categorical.to_numpy(dtype=float)]), names


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-csv", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--road-csv", type=Path, default=DEFAULT_ROAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    yolo = pd.read_csv(args.yolo_csv)
    yolo = yolo.loc[yolo["yolo_status"].eq("success")].copy()
    road = pd.read_csv(args.road_csv, usecols=["sample_index", "road_condition_status"])
    road = road.loc[road["road_condition_status"].eq("success")].rename(
        columns={"sample_index": "row_id"}
    )
    df = yolo.merge(road[["row_id"]], on="row_id", how="inner").reset_index(drop=True)
    df = add_time_features(df)
    df[["Temperature", "Humidity"]] = df[["Temperature", "Humidity"]].apply(
        pd.to_numeric, errors="coerce"
    )
    medians = df[["Temperature", "Humidity"]].median(numeric_only=True)
    df[["Temperature", "Humidity"]] = df[["Temperature", "Humidity"]].fillna(medians)

    train_val, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["aqi_cat"]
    )
    train, val = train_test_split(
        train_val, test_size=0.20, random_state=42, stratify=train_val["aqi_cat"]
    )

    # Match the historical implementation: categories were formed from train+test.
    combined = pd.concat([train, test], axis=0)
    _, feature_names = encode_features(combined)
    encoded_categories = feature_names[len(NUMERIC_FEATURES) :]
    x_train_raw, _ = encode_features(train, encoded_categories=encoded_categories)
    x_test_raw, _ = encode_features(test, encoded_categories=encoded_categories)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)
    y_train = train["PM2.5"].to_numpy(dtype=float)
    y_test = test["PM2.5"].to_numpy(dtype=float)

    model = ExtraTreesRegressor(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    prediction = np.clip(model.predict(x_test), 0, None)
    test_metrics = metrics(y_test, prediction)

    bundle = {
        "format_version": 1,
        "model": model,
        "scaler": scaler,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoded_categories": encoded_categories,
        "feature_names": feature_names,
        "imputation_medians": medians.to_dict(),
        "target": "PM2.5",
        "target_units": "ug/m3",
        "training_dataset": "TRAQID",
        "protocol": "random_row_split_stratified_by_aqi_after_feature_merge",
        "reportable_as_unseen_date_generalization": False,
        "random_state": 42,
        "split_counts": {
            "train": int(len(train)),
            "validation_unused": int(len(val)),
            "test": int(len(test)),
        },
        "test_metrics": test_metrics,
        "training_ranges": {
            name: [float(df[name].min()), float(df[name].max())]
            for name in ["Temperature", "Humidity"]
        },
        "source_script": str(Path(__file__)),
        "warning": (
            "Historical random-row TRAQID result; nearby observations and dates are "
            "shared across partitions. External MUMMA performance must be reported separately."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # ExtraTrees contains many large arrays; xz keeps the portable artifact
    # substantially smaller than joblib's default zlib compression.
    joblib.dump(bundle, args.output, compress=("xz", 3))

    loaded = joblib.load(args.output)
    reload_prediction = np.clip(loaded["model"].predict(x_test), 0, None)
    if not np.allclose(prediction, reload_prediction):
        raise RuntimeError("Reloaded model predictions do not match the exported model")

    print(json.dumps({"output": str(args.output), **bundle["split_counts"], **test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
