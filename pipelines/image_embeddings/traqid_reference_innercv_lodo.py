"""Strict TRAQID reference-assisted LODO with inner date-grouped selection.

For each outer held-out date, all remaining dates form the outer-training set.
The external local-reference PM2.5 signal is forced as the decomposition
background; it is never selected using the outer-test target.  GroupKFold over
outer-training dates generates honest inner out-of-fold predictions for
fusion-weight selection.  The final branch models are then refit on all outer
training dates and evaluated once on the untouched outer date.

The runner reports matched nonvisual, image-augmented, structured-visual and
full branches, plus fixed and inner-CV-selected three-branch fusions.  Results
are reference-assisted and must not be described as camera-only prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pipelines.image_embeddings.traqid_final_architecture_random_diagnostic import (
    build_multiscale_features,
)
from pipelines.image_embeddings.traqid_lodo_multisource_hybrid import (
    CAMS,
    MERRA,
    NONVISUAL_PREFIXES,
    TARGET,
    aggregate_predictions,
    derive_season,
    fit_local_model,
    image_sequence_features,
    metrics,
    prepare_matrix,
)


REFERENCE_CONTEXT_COLUMNS = [
    "reference_context__level",
    "reference_context__mean_t7",
    "reference_context__std_t7",
    "reference_context__delta_t7",
    "reference_context__slope_t7",
    "reference_context__minus_cams",
    "reference_context__minus_merra2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engineered-table",
        default=(
            "artifacts/runs/traqid_local_reference_openaq_50km_v2/"
            "traqid_engineered_T7_reference_complete.csv"
        ),
    )
    parser.add_argument(
        "--background-csv",
        default="artifacts/runs/traqid_atmospheric_background_v1/background_hourly.csv",
    )
    parser.add_argument("--local-reference-csv", required=True)
    parser.add_argument(
        "--reference-column",
        choices=[
            "reference_median_pm25_ug_m3",
            "reference_nearest_pm25_ug_m3",
            "reference_idw_pm25_ug_m3",
        ],
        required=True,
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=48)
    parser.add_argument("--simplex-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def mean_date_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    dates: np.ndarray,
) -> float:
    frame = pd.DataFrame(
        {"truth": truth, "prediction": prediction, "date": dates.astype(str)}
    )
    values = [
        metrics(group["truth"].to_numpy(), group["prediction"].to_numpy())["rmse"]
        for _, group in frame.groupby("date")
    ]
    return float(np.mean(values))


def select_hybrid_by_date(
    truth: np.ndarray,
    branch_predictions: list[np.ndarray],
    dates: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select simplex weights by mean held-out-date RMSE."""
    # Reuse the validated simplex enumeration by passing the OOF stack as both
    # validation and test, then explicitly rescore every returned candidate is
    # not possible.  Enumerate through the public helper's underlying behavior
    # with one-hot-generated candidates using a local import.
    from pipelines.image_embeddings.traqid_lodo_multisource_hybrid import (
        simplex_weights,
    )

    stack = np.column_stack(branch_predictions)
    best_weight: np.ndarray | None = None
    best_score = np.inf
    for weight in simplex_weights(step):
        prediction = stack @ weight
        score = mean_date_rmse(truth, prediction, dates)
        if score < best_score:
            best_score = score
            best_weight = weight
    assert best_weight is not None
    return stack @ best_weight, best_weight


def feature_families(
    train: pd.DataFrame,
    candidate_features: list[str],
) -> tuple[list[str], list[str]]:
    usable = [
        column
        for column in candidate_features
        if train[column].nunique(dropna=True) > 1
    ]
    nonvisual = [
        column for column in usable if column.startswith(NONVISUAL_PREFIXES)
    ]
    structured = [column for column in usable if column not in nonvisual]
    if not nonvisual or not structured:
        raise ValueError("Feature family separation produced an empty group")
    return usable, nonvisual


def add_reference_context_features(
    frame: pd.DataFrame,
    reference_values: pd.Series,
) -> pd.DataFrame:
    """Add inference-time reference level, short trend and product contrasts."""
    sequence_values = []
    for value in frame["seq_row_ids"].astype(str):
        row_ids = np.fromstring(value, sep="|", dtype=int).astype(str)
        sequence_values.append(reference_values.reindex(row_ids).to_numpy(dtype=float))
    lengths = {len(values) for values in sequence_values}
    if len(lengths) != 1:
        raise ValueError(f"Reference-context sequences have mixed lengths: {lengths}")
    values = np.vstack(sequence_values)
    if not np.isfinite(values).all():
        raise ValueError(
            "Reference context is incomplete inside at least one retained sequence"
        )
    x = np.arange(values.shape[1], dtype=float)
    centered_x = x - x.mean()
    slope = ((values - values.mean(axis=1, keepdims=True)) @ centered_x) / np.sum(
        centered_x**2
    )
    result = frame.copy()
    result["reference_context__level"] = result[
        "forced_reference_background"
    ].to_numpy(dtype=float)
    result["reference_context__mean_t7"] = values.mean(axis=1)
    result["reference_context__std_t7"] = values.std(axis=1)
    result["reference_context__delta_t7"] = values[:, -1] - values[:, 0]
    result["reference_context__slope_t7"] = slope
    result["reference_context__minus_cams"] = (
        result["forced_reference_background"].to_numpy(dtype=float)
        - pd.to_numeric(result[CAMS], errors="coerce").to_numpy(dtype=float)
    )
    result["reference_context__minus_merra2"] = (
        result["forced_reference_background"].to_numpy(dtype=float)
        - pd.to_numeric(result[MERRA], errors="coerce").to_numpy(dtype=float)
    )
    if result[REFERENCE_CONTEXT_COLUMNS].isna().any().any():
        raise ValueError("Non-finite reference-context feature generated")
    return result


def branch_matrices(
    train: pd.DataFrame,
    held: pd.DataFrame,
    usable: list[str],
    nonvisual: list[str],
    reference_context: list[str],
    image_train: np.ndarray,
    image_held: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    nonvisual_context = nonvisual + reference_context
    usable_context = usable + reference_context
    return {
        "reference_nonvisual_local": (
            prepare_matrix(train, nonvisual),
            prepare_matrix(held, nonvisual),
        ),
        "reference_nonvisual_images_local": (
            prepare_matrix(train, nonvisual, image_train),
            prepare_matrix(held, nonvisual, image_held),
        ),
        "reference_nonvisual_structured_local": (
            prepare_matrix(train, usable),
            prepare_matrix(held, usable),
        ),
        "reference_full_local": (
            prepare_matrix(train, usable, image_train),
            prepare_matrix(held, usable, image_held),
        ),
        "reference_context_local": (
            prepare_matrix(train, nonvisual_context),
            prepare_matrix(held, nonvisual_context),
        ),
        "reference_context_images_local": (
            prepare_matrix(train, nonvisual_context, image_train),
            prepare_matrix(held, nonvisual_context, image_held),
        ),
        "reference_context_structured_local": (
            prepare_matrix(train, usable_context),
            prepare_matrix(held, usable_context),
        ),
        "reference_context_full_local": (
            prepare_matrix(train, usable_context, image_train),
            prepare_matrix(held, usable_context, image_held),
        ),
    }


def fit_branches(
    train: pd.DataFrame,
    held: pd.DataFrame,
    embeddings: np.ndarray,
    usable: list[str],
    nonvisual: list[str],
    reference_context: list[str],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, np.ndarray]:
    image_train, image_held = image_sequence_features(
        embeddings,
        train,
        [train, held],
        args.pca_components,
        seed + 500,
    )
    matrices = branch_matrices(
        train,
        held,
        usable,
        nonvisual,
        reference_context,
        image_train,
        image_held,
    )
    local_target = (
        train[TARGET].to_numpy(dtype=float)
        - train["forced_reference_background"].to_numpy(dtype=float)
    )
    predictions: dict[str, np.ndarray] = {}
    for index, (name, (x_train, x_held)) in enumerate(matrices.items(), start=1):
        held_increment, _ = fit_local_model(
            x_train,
            local_target,
            x_held,
            x_held,
            train["date"].to_numpy(),
            args,
            seed + index,
        )
        predictions[name] = (
            held["forced_reference_background"].to_numpy(dtype=float)
            + held_increment
        )
    return predictions


def run_outer_fold(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    candidate_features: list[str],
    test_date: str,
    args: argparse.Namespace,
    fold_number: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    train = frame.loc[frame["date"].ne(test_date)].reset_index(drop=True)
    test = frame.loc[frame["date"].eq(test_date)].reset_index(drop=True)
    usable, nonvisual = feature_families(train, candidate_features)
    reference_context = [
        column
        for column in REFERENCE_CONTEXT_COLUMNS
        if train[column].nunique(dropna=True) > 1
    ]
    if not reference_context:
        raise ValueError("All reference-context features are constant")
    groups = train["date"].astype(str).to_numpy()
    unique_dates = np.unique(groups)
    inner_folds = min(args.inner_folds, len(unique_dates))
    if inner_folds < 2:
        raise ValueError("Inner grouped selection requires at least two dates")

    branch_names = [
        "reference_nonvisual_local",
        "reference_nonvisual_images_local",
        "reference_nonvisual_structured_local",
        "reference_full_local",
        "reference_context_local",
        "reference_context_images_local",
        "reference_context_structured_local",
        "reference_context_full_local",
    ]
    oof = {name: np.full(len(train), np.nan, dtype=float) for name in branch_names}
    inner = GroupKFold(n_splits=inner_folds)
    for inner_number, (fit_indices, held_indices) in enumerate(
        inner.split(train, groups=groups), start=1
    ):
        print(
            f"  inner {inner_number}/{inner_folds} "
            f"held_dates={sorted(train.iloc[held_indices]['date'].unique())}",
            flush=True,
        )
        predictions = fit_branches(
            train.iloc[fit_indices].reset_index(drop=True),
            train.iloc[held_indices].reset_index(drop=True),
            embeddings,
            usable,
            nonvisual,
            reference_context,
            args,
            args.seed + fold_number * 10000 + inner_number * 100,
        )
        for name in branch_names:
            oof[name][held_indices] = predictions[name]
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise RuntimeError("Inner OOF predictions are incomplete")

    base_fusion_branches = [
        "reference_nonvisual_local",
        "reference_nonvisual_images_local",
        "reference_nonvisual_structured_local",
    ]
    context_fusion_branches = [
        "reference_context_local",
        "reference_context_images_local",
        "reference_context_structured_local",
    ]
    y_train = train[TARGET].to_numpy(dtype=float)
    _, base_weights = select_hybrid_by_date(
        y_train,
        [oof[name] for name in base_fusion_branches],
        groups,
        args.simplex_step,
    )
    _, context_weights = select_hybrid_by_date(
        y_train,
        [oof[name] for name in context_fusion_branches],
        groups,
        args.simplex_step,
    )
    oof_rows = []
    for name in branch_names:
        oof_rows.append(
            {
                "outer_test_date": test_date,
                "method": name,
                **metrics(y_train, oof[name]),
                "mean_date_rmse": mean_date_rmse(y_train, oof[name], groups),
            }
        )

    final_predictions = fit_branches(
        train,
        test,
        embeddings,
        usable,
        nonvisual,
        reference_context,
        args,
        args.seed + fold_number * 10000 + 9000,
    )
    final_predictions["reference_only"] = test[
        "forced_reference_background"
    ].to_numpy(dtype=float)
    final_predictions["fixed_equal_three_branch_diagnostic"] = np.mean(
        np.column_stack([final_predictions[name] for name in base_fusion_branches]),
        axis=1,
    )
    final_predictions["inner_cv_selected_three_branch"] = (
        np.column_stack([final_predictions[name] for name in base_fusion_branches])
        @ base_weights
    )
    final_predictions["reference_context_fixed_equal_three_branch_diagnostic"] = np.mean(
        np.column_stack(
            [final_predictions[name] for name in context_fusion_branches]
        ),
        axis=1,
    )
    final_predictions["reference_context_inner_cv_selected_three_branch"] = (
        np.column_stack(
            [final_predictions[name] for name in context_fusion_branches]
        )
        @ context_weights
    )

    prediction_rows = []
    metric_rows = []
    y_test = test[TARGET].to_numpy(dtype=float)
    for method, prediction in final_predictions.items():
        prediction_rows.append(
            pd.DataFrame(
                {
                    "protocol": "reference_assisted_lodo_inner_group_cv",
                    "split_id": test_date,
                    "sequence_id": test["sequence_id"].to_numpy(),
                    "date": test["date"].to_numpy(),
                    "season": test["season"].to_numpy(),
                    "method": method,
                    "actual_PM2.5": y_test,
                    "predicted_PM2.5": prediction,
                    "reference_column": args.reference_column,
                }
            )
        )
        metric_rows.append(
            {
                "protocol": "reference_assisted_lodo_inner_group_cv",
                "split_id": test_date,
                "test_groups": test_date,
                "validation_groups": f"{inner_folds}-fold grouped inner OOF",
                "method": method,
                "n_train": len(train),
                "n_validation": len(train),
                "n_test": len(test),
                **metrics(y_test, prediction),
            }
        )
    metadata = {
        "test_date": test_date,
        "train_dates": sorted(train["date"].unique()),
        "inner_folds": inner_folds,
        "base_fusion_branch_order": base_fusion_branches,
        "base_fusion_weights": base_weights.tolist(),
        "context_fusion_branch_order": context_fusion_branches,
        "context_fusion_weights": context_weights.tolist(),
        "reference_column": args.reference_column,
        "reference_is_inference_time_pm_input": True,
        "outer_test_target_used_for_selection": False,
        "usable_features": len(usable),
        "nonvisual_features": len(nonvisual),
        "reference_context_features": reference_context,
    }
    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        metadata,
        pd.DataFrame(oof_rows),
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    engineered = pd.read_csv(args.engineered_table, low_memory=False)
    engineered = pd.concat(
        [
            engineered,
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        engineered["target_created_at"], errors="raise"
                    ).dt.date.astype(str),
                    "season": derive_season(engineered),
                },
                index=engineered.index,
            ),
        ],
        axis=1,
    ).copy()
    if engineered["sequence_id"].duplicated().any():
        raise ValueError("Engineered sequence IDs must be unique")

    background = pd.read_csv(args.background_csv, low_memory=False)
    background["sample_id"] = background["sample_id"].astype(str)
    background_indexed = background.drop_duplicates("sample_id").set_index("sample_id")
    multiscale, candidate_features, feature_audit = build_multiscale_features(
        engineered,
        background_indexed,
    )
    frame = engineered.merge(multiscale, on="sequence_id", validate="one_to_one")

    reference = pd.read_csv(args.local_reference_csv, low_memory=False)
    required = {"sample_id", args.reference_column}
    if missing := sorted(required - set(reference.columns)):
        raise ValueError(f"Reference table is missing columns: {missing}")
    reference["sample_id"] = reference["sample_id"].astype(str)
    reference_values = (
        reference.drop_duplicates("sample_id")
        .set_index("sample_id")[args.reference_column]
        .pipe(pd.to_numeric, errors="coerce")
    )
    frame["forced_reference_background"] = (
        frame["target_row_id"].astype(str).map(reference_values)
    )
    if frame["forced_reference_background"].isna().any():
        missing_dates = sorted(
            frame.loc[frame["forced_reference_background"].isna(), "date"].unique()
        )
        raise ValueError(
            "The forced-reference table is incomplete for this engineered table; "
            f"missing dates={missing_dates}"
        )
    frame["seq_row_ids"] = frame["seq_row_ids"].astype(str)
    frame = add_reference_context_features(frame, reference_values)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
    embeddings = np.load(args.embeddings, mmap_mode="r")
    if frame["target_row_id"].max() >= len(embeddings):
        raise ValueError("Embedding array does not cover every target row")

    dates = sorted(frame["date"].unique())
    if args.max_folds:
        dates = dates[: args.max_folds]
    design = {
        "dataset": "TRAQID",
        "protocol": "complete-date outer LODO with inner date-grouped CV",
        "dates_available": sorted(frame["date"].unique()),
        "planned_outer_folds": len(dates),
        "reference_column": args.reference_column,
        "reference_is_inference_time_pm_input": True,
        "background_forced": True,
        "reference_context_features": REFERENCE_CONTEXT_COLUMNS,
        "outer_test_targets_used_for_selection": False,
        "random_or_overlapping_split_used": False,
        "inner_folds": args.inner_folds,
        "feature_audit": feature_audit,
        "seed": args.seed,
    }
    (output / "study_design.json").write_text(json.dumps(design, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(design, indent=2))
        return 0

    prediction_tables = []
    metric_tables = []
    oof_tables = []
    for fold_number, test_date in enumerate(dates, start=1):
        fold_dir = output / "folds" / test_date
        fold_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = fold_dir / "predictions.csv"
        metrics_path = fold_dir / "metrics.csv"
        oof_path = fold_dir / "inner_oof_metrics.csv"
        metadata_path = fold_dir / "metadata.json"
        if (
            args.resume
            and prediction_path.exists()
            and metrics_path.exists()
            and oof_path.exists()
            and metadata_path.exists()
        ):
            print(f"Resume outer {fold_number}/{len(dates)}: {test_date}")
            prediction_tables.append(pd.read_csv(prediction_path))
            metric_tables.append(pd.read_csv(metrics_path))
            oof_tables.append(pd.read_csv(oof_path))
            continue
        print(f"TRAQID reference outer {fold_number}/{len(dates)} test={test_date}")
        predictions, fold_metrics, metadata, oof_metrics = run_outer_fold(
            frame,
            embeddings,
            candidate_features,
            test_date,
            args,
            fold_number,
        )
        predictions.to_csv(prediction_path, index=False)
        fold_metrics.to_csv(metrics_path, index=False)
        oof_metrics.to_csv(oof_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        prediction_tables.append(predictions)
        metric_tables.append(fold_metrics)
        oof_tables.append(oof_metrics)

    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    all_metrics = pd.concat(metric_tables, ignore_index=True)
    all_oof = pd.concat(oof_tables, ignore_index=True)
    aggregate, by_date = aggregate_predictions(all_predictions)
    all_predictions.to_csv(output / "predictions.csv", index=False)
    all_metrics.to_csv(output / "metrics_by_split.csv", index=False)
    all_oof.to_csv(output / "inner_oof_metrics.csv", index=False)
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    by_date.to_csv(output / "metrics_by_date.csv", index=False)
    (output / "run.json").write_text(
        json.dumps({**design, "completed_outer_folds": len(dates)}, indent=2) + "\n"
    )
    print("\nTRAQID FORCED-REFERENCE INNER-CV LODO RESULTS")
    print(aggregate.sort_values("rmse").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
