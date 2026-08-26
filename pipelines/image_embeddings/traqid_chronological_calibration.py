"""Evaluate short chronological calibration on frozen TRAQID predictions.

This runner does not refit the underlying PM2.5 model.  For every complete
held-out date it takes the earliest fraction of predictions and measurements
as a deployment-calibration segment, removes every later T=7 sequence sharing
any raw row with that segment, and evaluates calibration on the remaining
chronologically later sequences.

The resulting protocol is calibration-assisted deployment, not zero-shot
unseen-date generalisation.  Only the adaptation segment may be used to fit the
small post-hoc calibrators; the later evaluation targets remain untouched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ACTUAL = "actual_PM2.5"
PREDICTED = "predicted_PM2.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--sequence-table", required=True)
    parser.add_argument("--method", default="reference_context_local")
    parser.add_argument("--adaptation-fraction", type=float, default=0.10)
    parser.add_argument("--min-adaptation-rows", type=int, default=30)
    parser.add_argument("--min-test-rows", type=int, default=30)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "bias": float(np.mean(prediction - truth)),
    }


def parse_row_ids(value: object) -> set[int]:
    values = np.fromstring(str(value), sep="|", dtype=np.int64)
    if not len(values):
        raise ValueError(f"Could not parse seq_row_ids={value!r}")
    return set(values.tolist())


def fit_calibrators(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Return y_cal = intercept + slope * prediction calibrators."""
    residual = truth - prediction
    mean_offset = float(np.mean(residual))
    median_offset = float(np.median(residual))

    centered_prediction = prediction - prediction.mean()
    centered_truth = truth - truth.mean()
    denominator = float(centered_prediction @ centered_prediction)
    if denominator <= np.finfo(float).eps:
        affine_slope = 1.0
    else:
        affine_slope = float(
            (centered_prediction @ centered_truth) / denominator
        )
    affine_intercept = float(truth.mean() - affine_slope * prediction.mean())

    calibrators = {
        "zero_shot": (0.0, 1.0),
        "mean_offset": (mean_offset, 1.0),
        "median_offset": (median_offset, 1.0),
        "affine_ols": (affine_intercept, affine_slope),
    }
    diagnostics = {
        "adaptation_truth_mean": float(truth.mean()),
        "adaptation_prediction_mean": float(prediction.mean()),
        "adaptation_mean_residual": mean_offset,
        "adaptation_median_residual": median_offset,
        "affine_intercept": affine_intercept,
        "affine_slope": affine_slope,
    }
    return calibrators, diagnostics


def aggregate(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_date_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for calibration, group in predictions.groupby("calibration", sort=False):
        for date, date_group in group.groupby("date", sort=True):
            by_date_rows.append(
                {
                    "calibration": calibration,
                    "date": date,
                    "n_test": len(date_group),
                    **metrics(
                        date_group[ACTUAL].to_numpy(dtype=float),
                        date_group["calibrated_PM2.5"].to_numpy(dtype=float),
                    ),
                }
            )
        date_metrics = pd.DataFrame(
            [row for row in by_date_rows if row["calibration"] == calibration]
        )
        pooled = metrics(
            group[ACTUAL].to_numpy(dtype=float),
            group["calibrated_PM2.5"].to_numpy(dtype=float),
        )
        aggregate_rows.append(
            {
                "protocol": "chronological_first_fraction_calibration_then_purged_test",
                "base_method": group["base_method"].iloc[0],
                "calibration": calibration,
                "dates_evaluated": group["date"].nunique(),
                "n_test": len(group),
                **pooled,
                "mean_date_r2": float(date_metrics["r2"].mean()),
                "median_date_r2": float(date_metrics["r2"].median()),
                "positive_date_r2_count": int((date_metrics["r2"] > 0).sum()),
                "mean_date_rmse": float(date_metrics["rmse"].mean()),
                "worst_date_rmse": float(date_metrics["rmse"].max()),
            }
        )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(by_date_rows)


def add_outer_safe_meta_offsets(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn early-to-late offset mapping without using the target date's future.

    Each target date is excluded while a one-feature ridge mapping is learned
    from the other dates. Ridge strength is selected by leave-one-date-out CV
    inside those remaining dates. This is a small meta-calibrator, not a refit
    of the underlying PM model.
    """
    zero = predictions.loc[predictions["calibration"].eq("zero_shot")].copy()
    zero["future_residual"] = zero[ACTUAL] - zero[PREDICTED]
    late = zero.groupby("date", as_index=False)["future_residual"].mean()
    early = parameters.loc[
        parameters["calibration"].eq("zero_shot"),
        ["date", "adaptation_mean_residual", "adaptation_median_residual"],
    ].drop_duplicates("date")
    summary = early.merge(late, on="date", validate="one_to_one")
    if len(summary) < 4:
        raise ValueError("Meta-calibration requires at least four dates")

    output_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    for feature, name in [
        ("adaptation_mean_residual", "meta_ridge_mean_offset"),
        ("adaptation_median_residual", "meta_ridge_median_offset"),
    ]:
        for date in summary["date"]:
            fit_mask = summary["date"].ne(date)
            x_train = summary.loc[fit_mask, [feature]].to_numpy(dtype=float)
            y_train = summary.loc[fit_mask, "future_residual"].to_numpy(dtype=float)
            model = GridSearchCV(
                make_pipeline(StandardScaler(), Ridge()),
                {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
                cv=LeaveOneOut(),
                scoring="neg_mean_squared_error",
            )
            model.fit(x_train, y_train)
            x_test = summary.loc[summary["date"].eq(date), [feature]].to_numpy(
                dtype=float
            )
            offset = float(model.predict(x_test)[0])
            date_rows = zero.loc[zero["date"].eq(date)].copy()
            date_rows["calibration"] = name
            date_rows["calibrated_PM2.5"] = date_rows[PREDICTED] + offset
            date_rows = date_rows.drop(columns="future_residual")
            output_rows.append(date_rows)
            parameter_rows.append(
                {
                    "date": date,
                    "base_method": date_rows["base_method"].iloc[0],
                    "calibration": name,
                    "intercept": offset,
                    "slope": 1.0,
                    "n_adaptation": int(
                        parameters.loc[
                            parameters["date"].eq(date), "n_adaptation"
                        ].iloc[0]
                    ),
                    "n_purged": int(
                        parameters.loc[parameters["date"].eq(date), "n_purged"].iloc[0]
                    ),
                    "n_test": len(date_rows),
                    "meta_feature": feature,
                    "meta_training_dates": int(fit_mask.sum()),
                    "meta_ridge_alpha": float(model.best_params_["ridge__alpha"]),
                    "target_date_future_labels_used": False,
                }
            )
    return pd.concat(output_rows, ignore_index=True), pd.DataFrame(parameter_rows)


def main() -> int:
    args = parse_args()
    if not 0 < args.adaptation_fraction < 1:
        raise ValueError("--adaptation-fraction must be strictly between 0 and 1")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions, low_memory=False)
    required_prediction = {"sequence_id", "date", "method", ACTUAL, PREDICTED}
    if missing := sorted(required_prediction - set(predictions.columns)):
        raise ValueError(f"Predictions are missing columns: {missing}")
    predictions = predictions.loc[predictions["method"].eq(args.method)].copy()
    if predictions.empty:
        available = sorted(pd.read_csv(args.predictions, usecols=["method"])["method"].unique())
        raise ValueError(f"No rows for method={args.method!r}; available={available}")
    if predictions["sequence_id"].duplicated().any():
        raise ValueError("Selected prediction method has duplicated sequence IDs")

    sequences = pd.read_csv(
        args.sequence_table,
        usecols=["sequence_id", "target_created_at", "seq_row_ids"],
        low_memory=False,
    )
    if sequences["sequence_id"].duplicated().any():
        raise ValueError("Sequence table has duplicated sequence IDs")
    frame = predictions.merge(sequences, on="sequence_id", validate="one_to_one")
    if len(frame) != len(predictions):
        raise ValueError("Sequence metadata did not cover all predictions")
    frame["target_time"] = pd.to_datetime(
        frame["target_created_at"], errors="raise", utc=True
    )
    frame[ACTUAL] = pd.to_numeric(frame[ACTUAL], errors="raise")
    frame[PREDICTED] = pd.to_numeric(frame[PREDICTED], errors="raise")

    prediction_outputs: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for date, date_frame in frame.groupby("date", sort=True):
        date_frame = date_frame.sort_values(["target_time", "sequence_id"]).reset_index(
            drop=True
        )
        adaptation_rows = max(
            args.min_adaptation_rows,
            int(math.ceil(args.adaptation_fraction * len(date_frame))),
        )
        if adaptation_rows >= len(date_frame):
            raise ValueError(f"Date {date} has no rows left after adaptation")
        # Keep identical timestamps on one side of the boundary.  Splitting a
        # timestamp between adaptation and test would not be chronological.
        cutoff_time = date_frame["target_time"].iloc[adaptation_rows - 1]
        adaptation = date_frame.loc[date_frame["target_time"] <= cutoff_time].copy()
        candidates = date_frame.loc[date_frame["target_time"] > cutoff_time].copy()

        adaptation_raw_rows: set[int] = set()
        for value in adaptation["seq_row_ids"]:
            adaptation_raw_rows.update(parse_row_ids(value))
        overlap_mask = candidates["seq_row_ids"].map(
            lambda value: bool(parse_row_ids(value) & adaptation_raw_rows)
        )
        test = candidates.loc[~overlap_mask].copy()
        if len(test) < args.min_test_rows:
            raise ValueError(
                f"Date {date} retained only {len(test)} test rows after purging"
            )
        if not (test["target_time"] > adaptation["target_time"].max()).all():
            raise RuntimeError(f"Chronological boundary failed for date {date}")

        calibrators, diagnostics = fit_calibrators(
            adaptation[ACTUAL].to_numpy(dtype=float),
            adaptation[PREDICTED].to_numpy(dtype=float),
        )
        for calibration, (intercept, slope) in calibrators.items():
            result = test[
                ["sequence_id", "date", "target_created_at", ACTUAL, PREDICTED]
            ].copy()
            result.insert(
                0,
                "protocol",
                "chronological_first_fraction_adaptation_purged_test",
            )
            result["base_method"] = args.method
            result["calibration"] = calibration
            result["calibrated_PM2.5"] = intercept + slope * result[PREDICTED]
            prediction_outputs.append(result)
            parameter_rows.append(
                {
                    "date": date,
                    "base_method": args.method,
                    "calibration": calibration,
                    "intercept": intercept,
                    "slope": slope,
                    "n_adaptation": len(adaptation),
                    "n_purged": int(overlap_mask.sum()),
                    "n_test": len(test),
                    **diagnostics,
                }
            )
        audit_rows.append(
            {
                "date": date,
                "rows_total": len(date_frame),
                "rows_adaptation": len(adaptation),
                "rows_purged_for_raw_overlap": int(overlap_mask.sum()),
                "rows_test": len(test),
                "adaptation_start": adaptation["target_created_at"].iloc[0],
                "adaptation_end": adaptation["target_created_at"].iloc[-1],
                "test_start": test["target_created_at"].iloc[0],
                "test_end": test["target_created_at"].iloc[-1],
                "raw_overlap_after_purge": 0,
            }
        )

    parameters = pd.DataFrame(parameter_rows)
    all_predictions = pd.concat(prediction_outputs, ignore_index=True)
    meta_predictions, meta_parameters = add_outer_safe_meta_offsets(
        all_predictions, parameters
    )
    all_predictions = pd.concat(
        [all_predictions, meta_predictions], ignore_index=True
    )
    parameters = pd.concat([parameters, meta_parameters], ignore_index=True)
    aggregate_metrics, date_metrics = aggregate(all_predictions)
    audit = pd.DataFrame(audit_rows)

    all_predictions.to_csv(output / "predictions.csv", index=False)
    aggregate_metrics.to_csv(output / "metrics_aggregate.csv", index=False)
    date_metrics.to_csv(output / "metrics_by_date.csv", index=False)
    parameters.to_csv(output / "calibration_parameters.csv", index=False)
    audit.to_csv(output / "split_audit.csv", index=False)
    run = {
        "dataset": "TRAQID reference-covered complete dates",
        "protocol": "first chronological fraction adapts frozen predictions; raw-overlap purge; remainder is test",
        "base_predictions": args.predictions,
        "sequence_table": args.sequence_table,
        "base_method": args.method,
        "adaptation_fraction": args.adaptation_fraction,
        "calibrations": [
            "zero_shot",
            "mean_offset",
            "median_offset",
            "affine_ols",
            "meta_ridge_mean_offset",
            "meta_ridge_median_offset",
        ],
        "dates_evaluated": int(frame["date"].nunique()),
        "outer_test_targets_used_to_fit_base_model": False,
        "adaptation_targets_used_for_calibration": True,
        "post_adaptation_test_targets_used_for_selection": False,
        "raw_sequence_overlap_after_purge": 0,
        "reporting_label": "calibration-assisted deployment; not zero-shot generalisation",
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    print("\nTRAQID CHRONOLOGICAL CALIBRATION RESULTS")
    print(aggregate_metrics.sort_values("rmse").to_string(index=False))
    print("\nSPLIT AUDIT")
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
