"""Evaluate tabular residual correction on a validation-selected image model.

The image configuration is selected only by mean outer-validation RMSE. For
each outer fold, the correction model learns residuals only from that fold's
validation predictions and is evaluated on the untouched held-out date. Results
remain exploratory until correction selection is nested or pre-registered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.pm25_prediction.mumma_7day.model_current_data import estimators


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-dir", required=True)
    parser.add_argument("--tabular-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--base-view")
    parser.add_argument("--base-sequence-length", type=int)
    parser.add_argument("--base-cell", choices=["gru", "lstm"])
    return parser.parse_args()


def metric_row(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "bias": float(np.mean(prediction - truth)),
    }


def attach_samples(prediction_path: Path, sequence_path: Path, table: pd.DataFrame) -> pd.DataFrame:
    prediction = pd.read_csv(prediction_path)
    sequence = pd.read_csv(sequence_path, usecols=["sequence_id", "target_sample_id"])
    result = prediction.merge(sequence, on="sequence_id", validate="one_to_one")
    return result.merge(
        table, left_on="target_sample_id", right_on="sample_id",
        how="left", validate="one_to_one",
    )


def main() -> int:
    args = parse_args()
    ladder = Path(args.ladder_dir)
    tabular = Path(args.tabular_run_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ladder_results = pd.read_csv(ladder / "model_ladder_results.csv")
    validation_rank = (
        ladder_results.groupby(["view", "sequence_length", "cell"], as_index=False)
        .agg(validation_rmse_mean=("val_rmse", "mean"), validation_rmse_worst=("val_rmse", "max"))
        .sort_values(["validation_rmse_mean", "validation_rmse_worst"], kind="stable")
    )
    explicit_base = [args.base_view, args.base_sequence_length, args.base_cell]
    if any(value is not None for value in explicit_base) and not all(value is not None for value in explicit_base):
        raise ValueError("Provide --base-view, --base-sequence-length, and --base-cell together")
    if all(value is not None for value in explicit_base):
        candidates = validation_rank[
            validation_rank["view"].eq(args.base_view)
            & validation_rank["sequence_length"].eq(args.base_sequence_length)
            & validation_rank["cell"].eq(args.base_cell)
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Requested image base is unavailable: "
                f"view={args.base_view}, T={args.base_sequence_length}, cell={args.base_cell}"
            )
        selected = candidates.iloc[0]
        selection_rule = "explicit base configuration supplied by command line"
    else:
        selected = validation_rank.iloc[0]
        selection_rule = "minimum mean outer-validation RMSE; test metrics not used"
    view = str(selected["view"])
    length = int(selected["sequence_length"])
    cell = str(selected["cell"])

    table = pd.read_csv(tabular / "modeling_table.csv")
    if not table["sample_id"].is_unique:
        raise ValueError("Tabular modeling table sample_id is not unique")
    groups = json.loads((tabular / "feature_groups.json").read_text())

    rows = []
    predictions_out = []
    fold_lookup = ladder_results[["fold", "test_date"]].drop_duplicates().sort_values("fold")
    for fold_record in fold_lookup.itertuples(index=False):
        fold = int(fold_record.fold)
        model_dir = ladder / "models" / f"fold_{fold}" / view / f"T{length}" / cell
        sequence_path = model_dir.parent / "sequences.csv"
        validation = attach_samples(model_dir / "predictions_val.csv", sequence_path, table)
        test = attach_samples(model_dir / "predictions_test.csv", sequence_path, table)
        actual_column = f"actual_{args.target}"
        predicted_column = f"predicted_{args.target}"
        val_residual = validation[actual_column].to_numpy() - validation[predicted_column].to_numpy()
        test_truth = test[actual_column].to_numpy()
        base_prediction = test[predicted_column].to_numpy()

        rows.append({
            "fold": fold,
            "test_date": str(fold_record.test_date),
            "feature_set": "none",
            "correction_model": "image_base",
            "n_correction_train": len(validation),
            "n_test": len(test),
            **metric_row(test_truth, base_prediction),
        })
        base_table = test[["target_sample_id", actual_column]].copy()
        base_table["fold"] = fold
        base_table["feature_set"] = "none"
        base_table["correction_model"] = "image_base"
        base_table["prediction"] = base_prediction
        predictions_out.append(base_table)

        mean_corrected = base_prediction + float(val_residual.mean())
        rows.append({
            "fold": fold,
            "test_date": str(fold_record.test_date),
            "feature_set": "validation_residual_mean",
            "correction_model": "constant",
            "n_correction_train": len(validation),
            "n_test": len(test),
            **metric_row(test_truth, mean_corrected),
        })

        for group_name, specification in groups.items():
            columns = list(specification["columns"])
            if validation[columns].isna().all(axis=0).any():
                missing = validation[columns].columns[validation[columns].isna().all(axis=0)].tolist()
                raise ValueError(f"All-missing correction columns in {group_name}: {missing}")
            for model_name, model in estimators(args.random_state).items():
                model.fit(validation[columns], val_residual)
                corrected = base_prediction + model.predict(test[columns])
                rows.append({
                    "fold": fold,
                    "test_date": str(fold_record.test_date),
                    "feature_set": group_name,
                    "correction_model": model_name,
                    "n_correction_train": len(validation),
                    "n_test": len(test),
                    **metric_row(test_truth, corrected),
                })
                prediction_table = test[["target_sample_id", actual_column]].copy()
                prediction_table["fold"] = fold
                prediction_table["feature_set"] = group_name
                prediction_table["correction_model"] = model_name
                prediction_table["prediction"] = corrected
                predictions_out.append(prediction_table)

    metrics_by_fold = pd.DataFrame(rows)
    predictions = pd.concat(predictions_out, ignore_index=True)
    aggregate_rows = []
    for (feature_set, correction_model), group in predictions.groupby(
        ["feature_set", "correction_model"], sort=True,
    ):
        truth = group[f"actual_{args.target}"].to_numpy()
        aggregate_rows.append({
            "feature_set": feature_set,
            "correction_model": correction_model,
            "n_test": len(group),
            **metric_row(truth, group["prediction"].to_numpy()),
        })
    aggregate = pd.DataFrame(aggregate_rows).sort_values("rmse")
    fold_summary = (
        metrics_by_fold.groupby(["feature_set", "correction_model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_worst=("rmse", "max"),
            r2_mean=("r2", "mean"),
            bias_mean=("bias", "mean"),
        )
        .sort_values(["rmse_mean", "rmse_worst"])
    )
    metrics_by_fold.to_csv(output / "metrics_by_fold.csv", index=False)
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    fold_summary.to_csv(output / "metrics_fold_summary.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    validation_rank.to_csv(output / "image_validation_ranking.csv", index=False)
    run = {
        "selected_image_model": {"view": view, "sequence_length": length, "cell": cell},
        "selection_rule": selection_rule,
        "selected_base_validation_rmse_mean": float(selected["validation_rmse_mean"]),
        "selected_base_validation_rmse_worst": float(selected["validation_rmse_worst"]),
        "correction_training": "outer-validation residuals only",
        "outer_folds": int(fold_lookup["fold"].nunique()),
        "exploratory_only": True,
        "reason": "Correction candidates are compared on outer-test outputs; use a pre-registered candidate or nested correction selection for a confirmatory claim.",
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    print(aggregate.to_string(index=False))
    print("\nMean across held-out-date folds:")
    print(fold_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
