"""Fair matched TRAQID direct/CAMS/MERRA-2 whole-date comparison.

The experiment holds out complete acquisition dates in five outer folds.  The
direct branch predicts PM2.5, while the atmospheric branches predict:

    local increment = PM2.5 - external background

The fixed external background is added back before scoring.  A pre-specified
residual model is fitted on each fold's outer-validation residuals and applied
once to the untouched outer-test dates.  Residual predictors explicitly
exclude PM, AQI, row identifiers, and date-fold indicators.

This is an exploratory fair evaluation: correction-model identities were
chosen before this run from the earlier random diagnostic, but those earlier
diagnostics used the same dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.image_embeddings.traqid_random_background_comparison import (
    residual_columns,
)
from pipelines.pm25_prediction.mumma_7day.model_current_data import estimators
from roadside_pm.validation.outer_folds import make_grouped_outer_folds


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "direct": {
        "background_column": None,
        "correction_model": "random_forest",
    },
    "cams": {
        "background_column": "background_cams_pm25_ug_m3",
        "correction_model": "extra_trees",
    },
    "merra2": {
        "background_column": "background_merra2_pm25_ug_m3",
        "correction_model": "random_forest",
    },
    "local_reference": {
        "background_column": "reference_pm25_ug_m3",
        "correction_model": "random_forest",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-manifest",
        default=(
            "experiments/traqid_pretraining_v1/data/processed/"
            "paper_style_sequences/"
            "traqid_paper_style_T7_front_sequence_manifest_time_balanced_purged.csv"
        ),
    )
    parser.add_argument(
        "--embeddings",
        default=(
            "experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/"
            "traqid_paper_resnet50_front_rear_concat_gap_features.npy"
        ),
    )
    parser.add_argument(
        "--engineered-table",
        default=(
            "experiments/traqid_pretraining_v1/data/processed/"
            "traqid_engineered_T7_sequence_table.csv"
        ),
    )
    parser.add_argument("--background-csv", required=True)
    parser.add_argument(
        "--local-reference-csv",
        help=(
            "Aligned monitor table containing sample_id and "
            "reference_pm25_ug_m3; required for local_reference."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=["direct", "cams", "merra2"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-direct-run",
        help=(
            "Optional existing matched date-grouped direct run. Its fold "
            "manifests and val/test predictions are verified and reused."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def canonicalize(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sequence_id",
        "date",
        "seq_row_ids",
        "target_row_id",
        "target_PM2.5",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"TRAQID sequence manifest is missing: {missing}")
    result = source.copy()
    result["embedding_rows"] = result["seq_row_ids"]
    result["target_sample_id"] = result["target_row_id"].astype(str)
    result["date"] = result["date"].astype(str)
    return result


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "bias": float(np.mean(prediction - truth)),
    }


def attach_predictions(
    path: Path,
    manifest: pd.DataFrame,
    engineered: pd.DataFrame,
    features: list[str],
    *,
    target: str,
) -> pd.DataFrame:
    prediction = pd.read_csv(path)
    required = {
        "sequence_id",
        f"actual_{target}",
        f"predicted_{target}",
    }
    missing = sorted(required - set(prediction.columns))
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {missing}")
    result = prediction.merge(
        manifest[
            [
                "sequence_id",
                "date",
                "target_sample_id",
                "_background",
            ]
        ],
        on="sequence_id",
        validate="one_to_one",
    )
    return result.merge(
        engineered[["sequence_id", *features]],
        on="sequence_id",
        validate="one_to_one",
    )


def verify_reused_fold(
    expected: pd.DataFrame,
    existing_path: Path,
) -> pd.DataFrame:
    existing = pd.read_csv(existing_path)
    expected_assignment = expected.set_index("sequence_id")["outer_split"].sort_index()
    existing_assignment = (
        existing.set_index("sequence_id")["outer_split"].sort_index()
    )
    if not expected_assignment.index.equals(existing_assignment.index):
        raise ValueError(f"Reused fold has different sequence IDs: {existing_path}")
    if not expected_assignment.equals(existing_assignment):
        raise ValueError(f"Reused fold has different date assignments: {existing_path}")
    return existing


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    canonical = canonicalize(pd.read_csv(args.sequence_manifest))
    folds = make_grouped_outer_folds(
        canonical,
        group_column="date",
        n_splits=args.outer_folds,
        seed=args.seed,
    )
    if args.max_folds:
        folds = folds[: args.max_folds]

    background = pd.read_csv(args.background_csv)
    background["sample_id"] = background["sample_id"].astype(str)
    required_background = {"sample_id", "background_status"}
    required_background.update(
        str(VARIANTS[name]["background_column"])
        for name in args.variants
        if VARIANTS[name]["background_column"] is not None
        and name != "local_reference"
    )
    missing_background = sorted(required_background - set(background.columns))
    if missing_background:
        raise ValueError(f"Background table is missing: {missing_background}")
    if background["sample_id"].duplicated().any():
        raise ValueError("Background sample_id must be unique")
    if not background["background_status"].eq("success").all():
        raise ValueError("Background table contains unsuccessful rows")
    if "local_reference" in args.variants:
        if not args.local_reference_csv:
            raise ValueError(
                "--local-reference-csv is required for local_reference"
            )
        reference = pd.read_csv(args.local_reference_csv)
        reference["sample_id"] = reference["sample_id"].astype(str)
        required_reference = {"sample_id", "reference_pm25_ug_m3"}
        missing_reference = sorted(required_reference - set(reference.columns))
        if missing_reference:
            raise ValueError(
                f"Local-reference table is missing: {missing_reference}"
            )
        if reference["sample_id"].duplicated().any():
            raise ValueError("Local-reference sample_id must be unique")
        background = background.merge(
            reference[["sample_id", "reference_pm25_ug_m3"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if background["reference_pm25_ug_m3"].isna().any():
            raise ValueError(
                "Local-reference coverage is incomplete. Do not impute missing "
                "monitor PM2.5 in this matched experiment."
            )
    background = background.set_index("sample_id")

    engineered = pd.read_csv(args.engineered_table)
    if engineered["sequence_id"].duplicated().any():
        raise ValueError("Engineered table sequence_id must be unique")
    features = residual_columns(engineered)
    forbidden = [
        column
        for column in features
        if any(token in column.lower() for token in ("pm2.5", "pm10", "aqi"))
    ]
    if forbidden:
        raise AssertionError(f"Target-proxy residual features selected: {forbidden}")
    (output / "residual_feature_columns.txt").write_text(
        "\n".join(features) + "\n"
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(ROOT / "src")
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    reuse_direct = Path(args.reuse_direct_run) if args.reuse_direct_run else None
    fold_metrics: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []

    for variant in args.variants:
        specification = VARIANTS[variant]
        background_column = specification["background_column"]
        correction_name = str(specification["correction_model"])
        for fold_number, base_fold in enumerate(folds, 1):
            fold_dir = output / variant / f"fold_{fold_number}"
            model_dir = fold_dir / args.cell
            fold_dir.mkdir(parents=True, exist_ok=True)

            manifest = base_fold.copy()
            if background_column is None:
                manifest["_background"] = 0.0
                target = "target_PM2.5"
            else:
                values = manifest["target_sample_id"].map(
                    background[background_column]
                )
                if values.isna().any():
                    raise ValueError(
                        f"{variant} fold {fold_number}: "
                        f"{int(values.isna().sum())} targets lack background"
                    )
                manifest["_background"] = values.to_numpy(dtype=float)
                target = f"target_PM2.5_{variant}_local_increment"
                manifest[target] = (
                    manifest["target_PM2.5"].to_numpy(dtype=float)
                    - manifest["_background"].to_numpy(dtype=float)
                )
            manifest_path = fold_dir / "sequence_manifest.csv"
            manifest.to_csv(manifest_path, index=False)

            reused = variant == "direct" and reuse_direct is not None
            if reused:
                existing_fold = reuse_direct / f"fold_{fold_number}"
                verify_reused_fold(
                    manifest,
                    existing_fold / "sequence_manifest.csv",
                )
                val_path = existing_fold / args.cell / "predictions_val.csv"
                test_path = existing_fold / args.cell / "predictions_test.csv"
            else:
                command = [
                    sys.executable,
                    str(ROOT / "pipelines/image_embeddings/train_rnn.py"),
                    "--sequence-manifest",
                    str(manifest_path),
                    "--embeddings",
                    args.embeddings,
                    "--split-col",
                    "outer_split",
                    "--target-cols",
                    target,
                    "--cell",
                    args.cell,
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--seed",
                    str(args.seed + fold_number),
                    "--device",
                    args.device,
                    "--output-dir",
                    str(model_dir),
                ]
                print(" ".join(command), flush=True)
                if args.dry_run:
                    continue
                if not (
                    args.resume
                    and (model_dir / "predictions_val.csv").is_file()
                    and (model_dir / "predictions_test.csv").is_file()
                ):
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        check=True,
                    )
                val_path = model_dir / "predictions_val.csv"
                test_path = model_dir / "predictions_test.csv"

            if args.dry_run:
                continue
            validation = attach_predictions(
                val_path,
                manifest,
                engineered,
                features,
                target=target,
            )
            test = attach_predictions(
                test_path,
                manifest,
                engineered,
                features,
                target=target,
            )

            actual_column = f"actual_{target}"
            predicted_column = f"predicted_{target}"
            validation_residual = (
                validation[actual_column].to_numpy(dtype=float)
                - validation[predicted_column].to_numpy(dtype=float)
            )
            test_truth = (
                test[actual_column].to_numpy(dtype=float)
                + test["_background"].to_numpy(dtype=float)
            )
            base_prediction = (
                test[predicted_column].to_numpy(dtype=float)
                + test["_background"].to_numpy(dtype=float)
            )

            correction = estimators(args.seed + fold_number)[correction_name]
            correction.fit(validation[features], validation_residual)
            corrected_prediction = (
                base_prediction + correction.predict(test[features])
            )

            for model_name, prediction in (
                ("image_base", base_prediction),
                (correction_name, corrected_prediction),
            ):
                fold_metrics.append(
                    {
                        "variant": variant,
                        "fold": fold_number,
                        "test_dates": "|".join(
                            sorted(test["date"].astype(str).unique())
                        ),
                        "model": model_name,
                        "n_validation": len(validation),
                        "n_test": len(test),
                        **metrics(test_truth, prediction),
                    }
                )
                prediction_tables.append(
                    pd.DataFrame(
                        {
                            "sequence_id": test["sequence_id"],
                            "target_sample_id": test["target_sample_id"],
                            "date": test["date"],
                            "variant": variant,
                            "fold": fold_number,
                            "model": model_name,
                            "actual_PM2.5": test_truth,
                            "predicted_PM2.5": prediction,
                            "background_PM2.5": test["_background"],
                        }
                    )
                )

    if args.dry_run:
        print(
            "Dry run complete. Commands for the requested variants are shown above."
        )
        return 0

    fold_frame = pd.DataFrame(fold_metrics)
    fold_frame.to_csv(output / "metrics_by_fold.csv", index=False)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for (variant, model_name), group in predictions.groupby(["variant", "model"]):
        values = metrics(
            group["actual_PM2.5"].to_numpy(dtype=float),
            group["predicted_PM2.5"].to_numpy(dtype=float),
        )
        matching_folds = fold_frame[
            fold_frame["variant"].eq(variant)
            & fold_frame["model"].eq(model_name)
        ]
        aggregate_rows.append(
            {
                "variant": variant,
                "model": model_name,
                "n_test": len(group),
                **values,
                "mean_fold_mae": float(matching_folds["mae"].mean()),
                "mean_fold_rmse": float(matching_folds["rmse"].mean()),
                "mean_fold_r2": float(matching_folds["r2"].mean()),
                "worst_fold_rmse": float(matching_folds["rmse"].max()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["r2", "rmse"],
        ascending=[False, True],
    )
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    selected = aggregate[
        aggregate.apply(
            lambda row: row["model"]
            == VARIANTS[str(row["variant"])]["correction_model"],
            axis=1,
        )
    ].sort_values("r2", ascending=False)
    selected.to_csv(output / "matched_selected_comparison.csv", index=False)

    run = {
        "dataset": "TRAQID",
        "protocol": "five-fold complete-date outer holdout",
        "reportable_as_unseen_date_generalization": True,
        "exploratory": True,
        "reason": (
            "Correction identities were pre-specified from earlier random "
            "diagnostics on the same dataset; confirm on future dates."
        ),
        "variants": {name: VARIANTS[name] for name in args.variants},
        "local_reference_csv": args.local_reference_csv,
        "local_reference_is_inference_time_pm_input": (
            "local_reference" in args.variants
        ),
        "background_target_fitted": False,
        "residual_training": (
            "outer-validation residuals only; applied once to outer-test dates"
        ),
        "pm_aqi_residual_predictors_used": False,
        "residual_feature_count": len(features),
        "outer_folds": len(folds),
        "seed": args.seed,
        "direct_run_reused": str(reuse_direct) if reuse_direct else None,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print("\nPOOLED FAIR COMPARISON\n" + aggregate.to_string(index=False))
    print("\nMATCHED PRE-SPECIFIED CORRECTIONS\n" + selected.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
