"""Apply the exported TRAQID tabular ExtraTrees bundle to another sensor table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def first_present(columns: pd.Index, candidates: list[str]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def season_from_month(month: pd.Series) -> pd.Series:
    # Reproduce the labels represented in TRAQID: Dec-Feb winter, Mar-Jun summer,
    # and Jul-Nov monsoon. This is a transfer convention, not a learned feature.
    return pd.Series(
        np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5, 6])],
            ["Winter", "Summer"],
            default="Monsoon",
        ),
        index=month.index,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--timestamp-col")
    parser.add_argument("--temperature-col")
    parser.add_argument("--humidity-col")
    parser.add_argument("--target-col", help="Optional PM2.5 column for external metrics")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    source = pd.read_csv(args.input_csv)
    timestamp_col = args.timestamp_col or first_present(
        source.columns, ["sample_timestamp", "timestamp", "created_at"]
    )
    temperature_col = args.temperature_col or first_present(
        source.columns, ["temp", "Temperature", "sTemp"]
    )
    humidity_col = args.humidity_col or first_present(
        source.columns, ["rh", "Humidity", "sRh"]
    )
    if not all([timestamp_col, temperature_col, humidity_col]):
        raise ValueError(
            "Could not infer timestamp/temperature/humidity columns; provide the three column options."
        )

    timestamp = pd.to_datetime(source[timestamp_col], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"{int(timestamp.isna().sum())} timestamps could not be parsed")
    frame = pd.DataFrame(index=source.index)
    frame["Temperature"] = pd.to_numeric(source[temperature_col], errors="coerce")
    frame["Humidity"] = pd.to_numeric(source[humidity_col], errors="coerce")
    for name in ["Temperature", "Humidity"]:
        frame[name] = frame[name].fillna(bundle["imputation_medians"][name])
    frame["hour_num"] = timestamp.dt.hour
    frame["month_num"] = timestamp.dt.month
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour_num"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour_num"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month_num"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month_num"] / 12)
    frame["Season"] = season_from_month(frame["month_num"])
    frame["Day_or_Night"] = np.where(frame["hour_num"] < 18, "Day", "Night")

    categorical = pd.get_dummies(
        frame[bundle["categorical_features"]],
        columns=bundle["categorical_features"],
        drop_first=False,
    ).reindex(columns=bundle["encoded_categories"], fill_value=False)
    raw = np.hstack(
        [
            frame[bundle["numeric_features"]].to_numpy(dtype=float),
            categorical.to_numpy(dtype=float),
        ]
    )
    prediction = np.clip(
        bundle["model"].predict(bundle["scaler"].transform(raw)), 0, None
    )
    output = source.copy()
    output["predicted_pm25_traqid_tabular_extratrees"] = prediction

    metrics = None
    if args.target_col:
        valid = pd.to_numeric(source[args.target_col], errors="coerce").notna()
        actual = pd.to_numeric(source.loc[valid, args.target_col], errors="raise").to_numpy()
        predicted = prediction[valid.to_numpy()]
        metrics = {
            "rows": int(valid.sum()),
            "MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted)),
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(
        json.dumps(
            {
                "output": str(args.output_csv),
                "rows": int(len(output)),
                "column_mapping": {
                    "timestamp": timestamp_col,
                    "temperature": temperature_col,
                    "humidity": humidity_col,
                    "target": args.target_col,
                },
                "external_metrics": metrics,
                "warning": "MUMMA metrics are external domain-transfer results, not TRAQID validation.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
