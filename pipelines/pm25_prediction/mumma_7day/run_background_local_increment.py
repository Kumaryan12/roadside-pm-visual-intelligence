"""Evaluate an external background plus learned local PM2.5 increments.

The selected PM2.5 field is a fixed external background. Local models are
trained only on ``observed PM2.5 - external background`` using non-PM feature
groups. Whole-date holdout is primary; random-row results are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipelines.pm25_prediction.mumma_7day.model_current_data import estimators
from pipelines.pm25_prediction.mumma_7day.run_residual_fusion_current_data import metric_row


BACKGROUND = "background_cams_pm25_ug_m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", required=True)
    parser.add_argument("--feature-groups", required=True)
    parser.add_argument("--background-csv", required=True)
    parser.add_argument("--background-col", default=BACKGROUND)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--local-feature-sets", nargs="+", default=[
        "visual_yolo_road", "sensor_plus_visual", "sensor_plus_alphaearth",
        "visual_yolo_road_alphaearth", "sensor_plus_visual_alphaearth",
    ])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(
    frame: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    feature_sets: dict[str, list[str]],
    *,
    protocol: str,
    fold: str,
    target: str,
    background_col: str,
    background_name: str,
    seed: int,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    train = frame.loc[train_index]
    test = frame.loc[test_index]
    truth = test[target].to_numpy(dtype=float)
    raw_background = test[background_col].to_numpy(dtype=float)
    rows = [{
        "protocol": protocol, "fold": fold, "background": background_name,
        "local_feature_set": "none", "local_model": "none", "n_test": len(test),
        **metric_row(truth, raw_background),
    }]
    predictions = [pd.DataFrame({
        "sample_id": test["sample_id"], "date": test["date"], "protocol": protocol,
        "fold": fold, "background": background_name, "local_feature_set": "none",
        "local_model": "none", "actual": truth, "background_prediction": raw_background,
        "local_increment_prediction": 0.0, "prediction": raw_background,
    })]
    local_target = (
        train[target].to_numpy(dtype=float)
        - train[background_col].to_numpy(dtype=float)
    )
    for feature_set, columns in feature_sets.items():
        for model_name, model in estimators(seed).items():
            model.fit(train[columns], local_target)
            local_prediction = model.predict(test[columns])
            prediction = raw_background + local_prediction
            rows.append({
                "protocol": protocol, "fold": fold, "background": background_name,
                "local_feature_set": feature_set, "local_model": model_name,
                "n_test": len(test), **metric_row(truth, prediction),
            })
            predictions.append(pd.DataFrame({
                "sample_id": test["sample_id"], "date": test["date"], "protocol": protocol,
                "fold": fold, "background": background_name,
                "local_feature_set": feature_set,
                "local_model": model_name, "actual": truth,
                "background_prediction": raw_background,
                "local_increment_prediction": local_prediction, "prediction": prediction,
            }))
    return rows, predictions


def main() -> int:
    args = parse_args()
    table = pd.read_csv(args.modeling_table)
    table["sample_id"] = table["sample_id"].astype(str)
    background = pd.read_csv(args.background_csv)
    background["sample_id"] = background["sample_id"].astype(str)
    if background["sample_id"].duplicated().any():
        raise ValueError("Background table sample_id must be unique")
    if not background["background_status"].eq("success").all():
        raise ValueError("Background table contains unsuccessful rows")
    if args.background_col not in background.columns:
        raise ValueError(
            f"Background table is missing requested field: {args.background_col}"
        )
    background_columns = [
        column for column in background.columns
        if column.startswith("background_") and column not in table.columns
    ]
    frame = table.merge(
        background[["sample_id", *background_columns]], on="sample_id", validate="one_to_one"
    )
    if len(frame) != len(table):
        raise ValueError("Background sample coverage mismatch")
    groups = json.loads(Path(args.feature_groups).read_text())
    missing_groups = sorted(set(args.local_feature_sets) - set(groups))
    if missing_groups:
        raise ValueError(f"Unknown local feature groups: {missing_groups}")
    feature_sets = {
        name: list(groups[name]["columns"]) for name in args.local_feature_sets
    }
    forbidden = {args.target, "sPM1", "sPM4", "sPM10", "sNPMp5", "sNPM1", "sNPM2", "sNPM4", "sNPM10", "sTPS"}
    leaked = {name: sorted(set(columns) & forbidden) for name, columns in feature_sets.items()}
    leaked = {name: values for name, values in leaked.items() if values}
    if leaked:
        raise ValueError(f"Local feature sets contain PM/OPC target proxies: {leaked}")

    frame["date"] = frame["date"].astype(str)
    background_name = {
        "background_cams_pm25_ug_m3": "raw_cams",
        "background_merra2_pm25_ug_m3": "raw_merra2",
        "background_geos_cf_pm25_ug_m3": "raw_geos_cf",
    }.get(args.background_col, args.background_col)
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for date in sorted(frame["date"].unique()):
        test_index = frame.index[frame["date"].eq(date)].to_numpy()
        train_index = frame.index[~frame["date"].eq(date)].to_numpy()
        fold_rows, fold_predictions = evaluate(
            frame, train_index, test_index, feature_sets,
            protocol="leave_one_day_out", fold=f"test_{date}",
            target=args.target, background_col=args.background_col,
            background_name=background_name, seed=args.seed,
        )
        rows.extend(fold_rows); predictions.extend(fold_predictions)
    train_index, test_index = train_test_split(
        frame.index.to_numpy(), test_size=0.2, random_state=args.seed, shuffle=True
    )
    fold_rows, fold_predictions = evaluate(
        frame, train_index, test_index, feature_sets,
        protocol="random_row", fold=f"seed_{args.seed}", target=args.target,
        background_col=args.background_col, background_name=background_name,
        seed=args.seed,
    )
    rows.extend(fold_rows); predictions.extend(fold_predictions)

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    prediction = pd.concat(predictions, ignore_index=True)
    aggregate_rows = []
    keys = ["protocol", "background", "local_feature_set", "local_model"]
    for key, part in prediction.groupby(keys, sort=False):
        aggregate_rows.append({
            **dict(zip(keys, key, strict=True)), "n_test": len(part),
            **metric_row(part["actual"].to_numpy(), part["prediction"].to_numpy()),
        })
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["protocol", "rmse"])
    metrics.to_csv(output / "metrics_by_fold.csv", index=False)
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    prediction.to_csv(output / "predictions.csv", index=False)
    run = {
        "decomposition": (
            f"PM2.5 = external {args.background_col} + learned local increment"
        ),
        "background_target_fitted": False,
        "background_column": args.background_col,
        "local_pm_opc_inputs_used": False,
        "primary_protocol": "leave_one_day_out",
        "random_row_reportable": False,
        "background_csv": args.background_csv,
        "local_feature_sets": list(feature_sets),
        "rows": len(frame),
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
