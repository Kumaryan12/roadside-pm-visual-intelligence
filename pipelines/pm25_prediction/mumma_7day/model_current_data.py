"""Exploratory PM2.5 baselines for the incomplete two-day MUMMA delivery.

The primary protocol trains on one complete day and tests on the other, then
reverses the direction. Random-row and OPC target-proxy experiments are emitted
only as explicitly non-reportable diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "sPM2"
RANDOM_STATE = 42

MET_GAS = [
    "temp", "rh", "k30Co2", "sTemp", "sRh", "sVocI", "sNoxI",
    "co_ppb", "no2_ppb", "so2_ppb", "o3_ppb_compensated",
    "ch4_ratio", "nh3_ppm",
]
TIME = ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]
MOBILITY = ["lat", "long", "alt", "sog_clean", "cog_sin", "cog_cos", "hdop"]
OPC_TARGET_PROXIES = [
    "sPM1", "sPM4", "sPM10", "sNPMp5", "sNPM1", "sNPM2",
    "sNPM4", "sNPM10", "sTPS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="artifacts/runs/mumma_3day_ingestion_v1/sensor_10s.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/runs/mumma_current_2day_baselines_v1",
    )
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = pd.to_datetime(result["sample_timestamp"], errors="raise")
    minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
    result["hour_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    result["hour_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))
    result["minute_sin"] = np.sin(2 * np.pi * timestamp.dt.minute / 60)
    result["minute_cos"] = np.cos(2 * np.pi * timestamp.dt.minute / 60)

    course = np.deg2rad(pd.to_numeric(result["cog"], errors="coerce"))
    result["cog_sin"] = np.sin(course)
    result["cog_cos"] = np.cos(course)

    speed = pd.to_numeric(result["sog"], errors="coerce")
    result["sog_clean"] = speed.mask((speed < 0) | (speed > 60))
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.date.astype(str)
    return result


def available(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return columns


def feature_groups(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    groups = {
        "met_gas": {
            "columns": available(frame, MET_GAS),
            "reportable": True,
            "note": "Contemporaneous meteorology and non-PM gas channels.",
        },
        "met_gas_time": {
            "columns": available(frame, MET_GAS + TIME),
            "reportable": True,
            "note": "Meteorology/gases plus cyclic time of day.",
        },
        "met_gas_time_mobility": {
            "columns": available(frame, MET_GAS + TIME + MOBILITY),
            "reportable": True,
            "note": "Adds location, motion, course and GPS quality.",
        },
        "opc_target_proxy_diagnostic": {
            "columns": available(frame, MET_GAS + OPC_TARGET_PROXIES),
            "reportable": False,
            "note": "NON-REPORTABLE: OPC mass/number channels share the PM2.5 measurement process.",
        },
    }
    return groups


def estimators(seed: int) -> dict[str, object]:
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=400, min_samples_leaf=5, max_features=0.7,
                random_state=seed, n_jobs=-1,
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=400, min_samples_leaf=5, max_features=0.8,
                random_state=seed, n_jobs=-1,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=seed,
            )),
        ]),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    prediction_std = float(np.std(y_pred))
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": spearman if prediction_std > 0 else float("nan"),
        "bias": float(np.mean(y_pred - y_true)),
    }


def evaluate_fold(
    frame: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    protocol: str,
    fold: str,
    groups: dict[str, dict[str, object]],
    seed: int,
    target: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    train = frame.loc[train_index]
    test = frame.loc[test_index]
    y_train = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    metric_rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []

    base_pred = np.repeat(float(np.mean(y_train)), len(test))
    base_metrics = metrics(y_test, base_pred)
    metric_rows.append({
        "protocol": protocol, "fold": fold, "feature_set": "train_mean",
        "model": "train_mean", "reportable": protocol == "leave_one_day_out",
        "n_train": len(train), "n_test": len(test), **base_metrics,
    })
    prediction_tables.append(prediction_frame(test, y_test, base_pred, protocol, fold, "train_mean", "train_mean", protocol == "leave_one_day_out"))

    for group_name, specification in groups.items():
        columns = specification["columns"]
        reportable = bool(specification["reportable"]) and protocol == "leave_one_day_out"
        for model_name, model in estimators(seed).items():
            model.fit(train[columns], y_train)
            prediction = model.predict(test[columns])
            metric_rows.append({
                "protocol": protocol, "fold": fold, "feature_set": group_name,
                "model": model_name, "reportable": reportable,
                "n_train": len(train), "n_test": len(test),
                **metrics(y_test, prediction),
            })
            prediction_tables.append(prediction_frame(
                test, y_test, prediction, protocol, fold, group_name,
                model_name, reportable,
            ))
    return metric_rows, prediction_tables


def prediction_frame(
    test: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
    protocol: str,
    fold: str,
    feature_set: str,
    model: str,
    reportable: bool,
) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": test["sample_id"].to_numpy(),
        "run_id": test["run_id"].to_numpy(),
        "date": test["date"].to_numpy(),
        "sample_timestamp": test["sample_timestamp"].to_numpy(),
        "protocol": protocol,
        "fold": fold,
        "feature_set": feature_set,
        "model": model,
        "reportable": reportable,
        "y_true": truth,
        "y_pred": prediction,
        "residual": truth - prediction,
    })


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["protocol", "feature_set", "model", "reportable"]
    for key, group in predictions.groupby(keys, sort=True):
        rows.append(dict(zip(keys, key)) | {"n_test": len(group)} | metrics(
            group["y_true"].to_numpy(), group["y_pred"].to_numpy()
        ))
    return pd.DataFrame(rows).sort_values(["protocol", "reportable", "rmse"], ascending=[True, False, True])


def write_summary(output: Path, frame: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    reportable = aggregate[(aggregate["protocol"] == "leave_one_day_out") & aggregate["reportable"]]
    best = reportable.sort_values("rmse").iloc[0]
    diagnostic = aggregate[(aggregate["protocol"] == "random_row") | (~aggregate["reportable"])]
    lines = [
        "# Current MUMMA two-day modeling summary",
        "",
        "Exploratory only: two dates are insufficient for model selection or a generalization claim.",
        "",
        f"- Rows: {len(frame)}",
        f"- Dates: {', '.join(sorted(frame['date'].unique()))}",
        "- Primary protocol: train on one whole date, test on the other, then reverse.",
        "- PM/OPC mass and number channels are excluded from reportable feature sets.",
        f"- Best exploratory reportable result: `{best['feature_set']} / {best['model']}` "
        f"(pooled RMSE {best['rmse']:.3f}, MAE {best['mae']:.3f}, R2 {best['r2']:.3f}).",
        "",
        "Random-row results are temporally overlap-contaminated. OPC target-proxy results are "
        "measurement-proxy diagnostics. Neither should be presented as fair model performance.",
        "",
        f"Diagnostic configurations evaluated: {len(diagnostic)}.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = engineer_features(pd.read_csv(input_path))
    if args.target not in frame:
        raise ValueError(f"Target column not found: {args.target}")
    if frame[args.target].isna().any():
        raise ValueError("Target contains missing values")
    dates = sorted(frame["date"].unique())
    if len(dates) != 2:
        raise ValueError(f"This exploratory runner requires exactly two dates; found {dates}")
    groups = feature_groups(frame)

    metric_rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []
    for test_date in dates:
        test_index = frame.index[frame["date"] == test_date].to_numpy()
        train_index = frame.index[frame["date"] != test_date].to_numpy()
        rows, tables = evaluate_fold(
            frame, train_index, test_index, protocol="leave_one_day_out",
            fold=f"test_{test_date}", groups=groups, seed=args.random_state,
            target=args.target,
        )
        metric_rows.extend(rows)
        prediction_tables.extend(tables)

    train_index, test_index = train_test_split(
        frame.index.to_numpy(), test_size=0.2, random_state=args.random_state,
    )
    rows, tables = evaluate_fold(
        frame, train_index, test_index, protocol="random_row",
        fold=f"seed_{args.random_state}", groups=groups,
        seed=args.random_state, target=args.target,
    )
    metric_rows.extend(rows)
    prediction_tables.extend(tables)

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    aggregate = aggregate_metrics(predictions)
    fold_metrics.to_csv(output / "metrics_by_fold.csv", index=False)
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "feature_groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    run = {
        "input_csv": str(input_path), "output_dir": str(output),
        "rows": len(frame), "dates": dates, "target": args.target,
        "random_state": args.random_state,
        "primary_protocol": "leave_one_day_out_two_direction",
        "random_row_reportable": False,
        "opc_target_proxy_reportable": False,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    write_summary(output, frame, aggregate)
    print(aggregate.to_string(index=False))
    print(f"Saved results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
