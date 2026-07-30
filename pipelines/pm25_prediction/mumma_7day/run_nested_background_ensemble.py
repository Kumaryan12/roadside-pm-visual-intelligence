"""Validation-select an ensemble of temporal and tabular PM2.5 branches.

For every whole-date outer fold, the temporal residual correction is
cross-fitted by run within the outer-validation date.  A tabular local-
increment model is fitted on outer-training rows and evaluated on the same
validation sequence targets.  A single convex weight is selected by validation
RMSE, frozen, and applied to the untouched outer-test date.

The final temporal test branch may use a correction fitted on all outer-
validation rows, and the final tabular branch may be refitted on outer train +
validation after weight selection.  No outer-test target is used for fitting or
weight selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pipelines.pm25_prediction.mumma_7day.model_current_data import estimators
from pipelines.pm25_prediction.mumma_7day.run_residual_fusion_current_data import metric_row


BACKGROUND = "background_cams_pm25_ug_m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-ladder", required=True)
    parser.add_argument("--temporal-residual-run", required=True)
    parser.add_argument("--tabular-local-run", required=True)
    parser.add_argument("--modeling-table", required=True)
    parser.add_argument("--feature-groups", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--background-col", default=BACKGROUND)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--view", default="lens6")
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--temporal-feature-set", default="sensor_plus_visual")
    parser.add_argument("--temporal-correction-model", default="random_forest")
    parser.add_argument("--tabular-feature-set", default="visual_yolo_road_alphaearth")
    parser.add_argument("--tabular-model", default="extra_trees")
    parser.add_argument("--inner-group-folds", type=int, default=5)
    parser.add_argument(
        "--interval-alpha",
        type=float,
        default=0.10,
        help="Miscoverage level for validation-calibrated symmetric conformal intervals.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def optimal_convex_weight(
    truth: np.ndarray, temporal: np.ndarray, tabular: np.ndarray,
) -> float:
    """Return the MSE-optimal temporal weight constrained to [0, 1]."""
    difference = np.asarray(temporal, dtype=float) - np.asarray(tabular, dtype=float)
    denominator = float(np.dot(difference, difference))
    if denominator <= np.finfo(float).eps:
        return 0.5
    weight = float(np.dot(difference, np.asarray(truth, dtype=float) - tabular) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def group_crossfit_convex_predictions(
    truth: np.ndarray,
    temporal: np.ndarray,
    tabular: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Predict each validation group with a weight selected on other groups."""
    truth = np.asarray(truth, dtype=float)
    temporal = np.asarray(temporal, dtype=float)
    tabular = np.asarray(tabular, dtype=float)
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Weight cross-fitting requires at least two validation groups")
    prediction = np.full(len(truth), np.nan, dtype=float)
    for group in unique_groups:
        held_out = groups == group
        weight = optimal_convex_weight(
            truth[~held_out], temporal[~held_out], tabular[~held_out],
        )
        prediction[held_out] = (
            weight * temporal[held_out] + (1.0 - weight) * tabular[held_out]
        )
    if np.isnan(prediction).any():
        raise RuntimeError("Weight cross-fitting left unpredicted validation rows")
    return prediction


def conformal_radius(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    alpha: float,
) -> float:
    """Return the finite-sample split-conformal absolute-residual radius."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("interval alpha must be strictly between zero and one")
    scores = np.abs(
        np.asarray(truth, dtype=float) - np.asarray(prediction, dtype=float)
    )
    if not len(scores):
        raise ValueError("Conformal calibration requires at least one residual")
    probability = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    return float(np.quantile(scores, probability, method="higher"))


def crossfit_temporal_correction(
    validation: pd.DataFrame,
    columns: list[str],
    *,
    target: str,
    model_name: str,
    folds: int,
    seed: int,
) -> np.ndarray:
    groups = validation["run_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Temporal correction cross-fitting requires at least two validation runs")
    actual = validation[f"actual_{target}"].to_numpy(dtype=float)
    base = validation[f"predicted_{target}"].to_numpy(dtype=float)
    residual = actual - base
    corrected = np.full(len(validation), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    for inner_fold, (train_index, test_index) in enumerate(
        splitter.split(validation, groups=groups), start=1,
    ):
        candidates = estimators(seed + inner_fold - 1)
        if model_name not in candidates:
            raise ValueError(f"Unknown temporal correction model: {model_name}")
        model = candidates[model_name]
        model.fit(validation.iloc[train_index][columns], residual[train_index])
        corrected[test_index] = base[test_index] + model.predict(validation.iloc[test_index][columns])
    if np.isnan(corrected).any():
        raise RuntimeError("Temporal validation cross-fitting left unpredicted rows")
    return corrected


def main() -> int:
    args = parse_args()
    if not 0.0 < args.interval_alpha < 1.0:
        raise ValueError("--interval-alpha must be strictly between zero and one")
    ladder = Path(args.temporal_ladder)
    residual_run = Path(args.temporal_residual_run)
    tabular_run = Path(args.tabular_local_run)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.modeling_table)
    table["sample_id"] = table["sample_id"].astype(str)
    if not table["sample_id"].is_unique:
        raise ValueError("Modeling table sample_id must be unique")
    feature_groups = json.loads(Path(args.feature_groups).read_text())
    for name in (args.temporal_feature_set, args.tabular_feature_set):
        if name not in feature_groups:
            raise ValueError(f"Unknown feature group: {name}")
    temporal_columns = list(feature_groups[args.temporal_feature_set]["columns"])
    tabular_columns = list(feature_groups[args.tabular_feature_set]["columns"])

    forbidden = {
        args.target, "sPM1", "sPM4", "sPM10", "sNPMp5", "sNPM1", "sNPM2",
        "sNPM4", "sNPM10", "sTPS",
    }
    for name, columns in {
        args.temporal_feature_set: temporal_columns,
        args.tabular_feature_set: tabular_columns,
    }.items():
        leaked = sorted(set(columns) & forbidden)
        if leaked:
            raise ValueError(f"Feature set {name} contains PM/OPC target proxies: {leaked}")

    residual_predictions = pd.read_csv(residual_run / "predictions.csv")
    residual_predictions["target_sample_id"] = residual_predictions["target_sample_id"].astype(str)
    temporal_test_all = residual_predictions[
        residual_predictions["feature_set"].eq(args.temporal_feature_set)
        & residual_predictions["correction_model"].eq(args.temporal_correction_model)
    ].copy()
    tabular_predictions = pd.read_csv(tabular_run / "predictions.csv")
    tabular_predictions["sample_id"] = tabular_predictions["sample_id"].astype(str)
    tabular_test_all = tabular_predictions[
        tabular_predictions["protocol"].eq("leave_one_day_out")
        & tabular_predictions["local_feature_set"].eq(args.tabular_feature_set)
        & tabular_predictions["local_model"].eq(args.tabular_model)
    ].copy()

    ladder_results = pd.read_csv(ladder / "model_ladder_results.csv")
    fold_lookup = ladder_results[["fold", "test_date"]].drop_duplicates().sort_values("fold")
    rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []
    validation_audit: list[dict[str, object]] = []
    uncertainty_rows: list[dict[str, object]] = []

    for record in fold_lookup.itertuples(index=False):
        fold = int(record.fold)
        test_date = str(record.test_date)
        model_root = ladder / "models" / f"fold_{fold}" / args.view / f"T{args.sequence_length}"
        model_dir = model_root / args.cell
        validation = pd.read_csv(model_dir / "predictions_val_decomposition.csv")
        validation["target_sample_id"] = validation["target_sample_id"].astype(str)
        validation = validation.merge(
            table[["sample_id", "run_id", *temporal_columns, *[c for c in tabular_columns if c not in temporal_columns]]],
            left_on="target_sample_id", right_on="sample_id", how="left", validate="one_to_one",
        )
        if validation["sample_id"].isna().any():
            raise ValueError(f"Fold {fold}: validation targets missing from modeling table")

        temporal_validation = crossfit_temporal_correction(
            validation, temporal_columns, target=args.target,
            model_name=args.temporal_correction_model, folds=args.inner_group_folds,
            seed=args.seed + 100 * fold,
        )

        split_candidates = sorted((ladder / "splits").glob(f"fold_{fold}_test_*.csv"))
        if len(split_candidates) != 1:
            raise ValueError(f"Fold {fold}: expected one adjusted split manifest")
        split = pd.read_csv(split_candidates[0])
        split["sample_id"] = split["sample_id"].astype(str)
        train_ids = set(split.loc[split["split_outer_day"].eq("train"), "sample_id"])
        training = table[table["sample_id"].isin(train_ids)].copy()
        if args.background_col not in training.columns:
            training = training.merge(
                split[["sample_id", args.background_col]],
                on="sample_id",
                validate="one_to_one",
            )
        local_target = (
            training[args.target].to_numpy(dtype=float)
            - training[args.background_col].to_numpy(dtype=float)
        )
        tabular_candidates = estimators(args.seed + fold - 1)
        if args.tabular_model not in tabular_candidates:
            raise ValueError(f"Unknown tabular model: {args.tabular_model}")
        tabular_model = tabular_candidates[args.tabular_model]
        tabular_model.fit(training[tabular_columns], local_target)
        validation_background = split.set_index("sample_id").loc[
            validation["target_sample_id"], args.background_col
        ].to_numpy(dtype=float)
        tabular_validation = validation_background + tabular_model.predict(validation[tabular_columns])
        validation_truth = validation[f"actual_{args.target}"].to_numpy(dtype=float)
        weight = optimal_convex_weight(validation_truth, temporal_validation, tabular_validation)
        validation_ensemble = weight * temporal_validation + (1.0 - weight) * tabular_validation
        validation_selected_crossfit = group_crossfit_convex_predictions(
            validation_truth,
            temporal_validation,
            tabular_validation,
            validation["run_id"].astype(str).to_numpy(),
        )
        calibration_predictions = {
            "validation_selected_weight": validation_selected_crossfit,
            "fixed_equal_weight_diagnostic": (
                0.5 * temporal_validation + 0.5 * tabular_validation
            ),
            "temporal_branch": temporal_validation,
            "tabular_branch": tabular_validation,
        }
        interval_radii = {
            method: conformal_radius(
                validation_truth, calibration_prediction,
                alpha=args.interval_alpha,
            )
            for method, calibration_prediction in calibration_predictions.items()
        }
        validation_audit.append({
            "fold": fold, "test_date": test_date, "n_validation": len(validation),
            "validation_runs": int(validation["run_id"].nunique()),
            "temporal_weight": weight,
            **{
                f"temporal_{key}": value
                for key, value in metric_row(validation_truth, temporal_validation).items()
            },
            **{
                f"tabular_{key}": value
                for key, value in metric_row(validation_truth, tabular_validation).items()
            },
            **{
                f"ensemble_{key}": value
                for key, value in metric_row(validation_truth, validation_ensemble).items()
            },
        })

        temporal_test = temporal_test_all[temporal_test_all["fold"].eq(fold)].copy()
        tabular_test = tabular_test_all[tabular_test_all["date"].astype(str).eq(test_date)].copy()
        test = temporal_test[["target_sample_id", f"actual_{args.target}", "prediction"]].rename(
            columns={"prediction": "temporal_prediction"}
        ).merge(
            tabular_test[["sample_id", "actual", "prediction"]].rename(
                columns={"prediction": "tabular_prediction"}
            ),
            left_on="target_sample_id", right_on="sample_id", validate="one_to_one",
        )
        validation_ids = set(validation["target_sample_id"])
        test_ids = set(test["target_sample_id"])
        if validation_ids & test_ids:
            raise ValueError(f"Fold {fold}: validation and outer-test target IDs overlap")
        if train_ids & validation_ids or train_ids & test_ids:
            raise ValueError(f"Fold {fold}: outer train IDs overlap validation/test targets")
        if not np.allclose(test[f"actual_{args.target}"], test["actual"], rtol=1e-5, atol=1e-5):
            raise ValueError(f"Fold {fold}: branch targets disagree")
        truth = test[f"actual_{args.target}"].to_numpy(dtype=float)
        selected_prediction = (
            weight * test["temporal_prediction"].to_numpy(dtype=float)
            + (1.0 - weight) * test["tabular_prediction"].to_numpy(dtype=float)
        )
        fixed_prediction = 0.5 * (
            test["temporal_prediction"].to_numpy(dtype=float)
            + test["tabular_prediction"].to_numpy(dtype=float)
        )
        for method, prediction in {
            "validation_selected_weight": selected_prediction,
            "fixed_equal_weight_diagnostic": fixed_prediction,
            "temporal_branch": test["temporal_prediction"].to_numpy(dtype=float),
            "tabular_branch": test["tabular_prediction"].to_numpy(dtype=float),
        }.items():
            radius = interval_radii[method]
            interval_lower = prediction - radius
            interval_upper = prediction + radius
            covered = (truth >= interval_lower) & (truth <= interval_upper)
            rows.append({
                "fold": fold, "test_date": test_date, "method": method,
                "temporal_weight": weight if method == "validation_selected_weight" else (
                    0.5 if method == "fixed_equal_weight_diagnostic" else (1.0 if method == "temporal_branch" else 0.0)
                ),
                "n_validation": len(validation), "n_test": len(test),
                **metric_row(truth, prediction),
            })
            uncertainty_rows.append({
                "fold": fold,
                "test_date": test_date,
                "method": method,
                "alpha": args.interval_alpha,
                "nominal_coverage": 1.0 - args.interval_alpha,
                "n_calibration": len(validation_truth),
                "calibration_groups": int(validation["run_id"].nunique()),
                "interval_radius": radius,
                "interval_width": 2.0 * radius,
                "n_test": len(truth),
                "test_coverage": float(np.mean(covered)),
            })
            part = test[["target_sample_id", f"actual_{args.target}"]].copy()
            part["fold"] = fold; part["test_date"] = test_date; part["method"] = method
            part["temporal_weight"] = rows[-1]["temporal_weight"]
            part["temporal_prediction"] = test["temporal_prediction"].to_numpy()
            part["tabular_prediction"] = test["tabular_prediction"].to_numpy()
            part["prediction"] = prediction
            part["interval_alpha"] = args.interval_alpha
            part["interval_radius"] = radius
            part["interval_lower"] = interval_lower
            part["interval_upper"] = interval_upper
            part["interval_covered"] = covered
            prediction_tables.append(part)

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    aggregate_rows = []
    for method, part in predictions.groupby("method", sort=False):
        aggregate_rows.append({
            "method": method, "n_test": len(part),
            **metric_row(part[f"actual_{args.target}"].to_numpy(), part["prediction"].to_numpy()),
        })
    aggregate = pd.DataFrame(aggregate_rows).sort_values("rmse")
    fold_summary = metrics.groupby("method", as_index=False).agg(
        folds=("fold", "nunique"), mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"), rmse_worst=("rmse", "max"),
        r2_mean=("r2", "mean"), bias_mean=("bias", "mean"),
        temporal_weight_mean=("temporal_weight", "mean"),
        temporal_weight_min=("temporal_weight", "min"),
        temporal_weight_max=("temporal_weight", "max"),
    ).sort_values("rmse_mean")
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty_aggregate = (
        predictions.groupby("method", as_index=False)
        .agg(
            n_test=("interval_covered", "size"),
            empirical_coverage=("interval_covered", "mean"),
            mean_interval_width=("interval_radius", lambda value: float(2.0 * value.mean())),
            worst_fold_radius=("interval_radius", "max"),
        )
        .sort_values("empirical_coverage", ascending=False)
    )
    metrics.to_csv(output / "metrics_by_fold.csv", index=False)
    pd.DataFrame(validation_audit).to_csv(output / "validation_weight_audit.csv", index=False)
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    fold_summary.to_csv(output / "metrics_fold_summary.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    uncertainty.to_csv(output / "uncertainty_by_fold.csv", index=False)
    uncertainty_aggregate.to_csv(output / "uncertainty_aggregate.csv", index=False)
    run = {
        "protocol": "whole-date outer holdout with run-grouped cross-fitted validation weight selection",
        "outer_test_targets_used_for_weight_selection": False,
        "weight_objective": "validation RMSE",
        "weight_constraint": "convex temporal weight in [0, 1]",
        "uncertainty": {
            "method": "symmetric absolute-residual conformal interval",
            "alpha": args.interval_alpha,
            "nominal_coverage": 1.0 - args.interval_alpha,
            "calibration_partition": "outer-validation targets only",
            "validation_selected_weight_calibration": (
                "leave-one-run-out weight cross-fitted predictions"
            ),
            "outer_test_targets_used_for_interval_calibration": False,
            "caveat": (
                "Coverage on a new date is diagnostic because dates are not "
                "guaranteed exchangeable and only five dates are available."
            ),
        },
        "temporal_branch": {
            "model": (
                f"{args.view} {args.cell.upper()} T={args.sequence_length} "
                f"with {args.background_col} background"
            ),
            "background_column": args.background_col,
            "correction": f"{args.temporal_feature_set} {args.temporal_correction_model}",
            "validation_correction": "GroupKFold out-of-run predictions within outer validation",
        },
        "tabular_branch": {
            "feature_set": args.tabular_feature_set, "model": args.tabular_model,
            "background_column": args.background_col,
            "validation_fit": "outer train only", "test_fit": "outer train + validation (saved branch)",
        },
        "confirmatory": False,
        "reason": "Branch architectures were chosen after earlier outer-test exploration; pre-register this complete protocol for future dates.",
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    print("\nPooled outer-test metrics:")
    print(aggregate.to_string(index=False))
    print("\nMean across held-out dates:")
    print(fold_summary.to_string(index=False))
    print("\nSelected weights by fold:")
    print(metrics[metrics["method"].eq("validation_selected_weight")][
        ["fold", "test_date", "temporal_weight", "mae", "rmse", "r2"]
    ].to_string(index=False))
    print("\nValidation-calibrated uncertainty:")
    print(uncertainty_aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
