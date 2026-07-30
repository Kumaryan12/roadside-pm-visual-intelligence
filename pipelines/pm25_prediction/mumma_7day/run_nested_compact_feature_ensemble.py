"""Compare a compact feature budget with the full CAMS nested ensemble.

Feature ranking is repeated inside every outer-training partition using
held-out training dates.  The outer validation date is reserved for ensemble
weight selection and the outer test date is untouched until final evaluation.

The candidate pool is the union of non-PM sensor, YOLO, road, OSM, and
AlphaEarth predictors.  A compact selection is shared across the architecture:
the temporal residual correction receives its compatible sensor/visual subset,
while the tabular local-increment branch receives its compatible
visual/OSM/AlphaEarth subset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from pipelines.pm25_prediction.mumma_7day.model_current_data import estimators
from pipelines.pm25_prediction.mumma_7day.run_nested_background_ensemble import (
    BACKGROUND,
    crossfit_temporal_correction,
    optimal_convex_weight,
)
from pipelines.pm25_prediction.mumma_7day.run_residual_fusion_current_data import (
    metric_row,
)


FORBIDDEN_PM_OPC = {
    "sPM1", "sPM2", "sPM4", "sPM10", "sNPMp5", "sNPM1", "sNPM2",
    "sNPM4", "sNPM10", "sTPS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-ladder", type=Path, required=True)
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--feature-groups", type=Path, required=True)
    parser.add_argument("--background-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--view", default="lens6")
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument(
        "--candidate-feature-set",
        default="sensor_plus_visual_osm_alphaearth",
    )
    parser.add_argument(
        "--temporal-feature-pool",
        default="sensor_plus_visual",
    )
    parser.add_argument(
        "--tabular-feature-pool",
        default="visual_yolo_road_osm_alphaearth",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[90, 216],
        help="Total selected-feature budgets. Include the full pool for comparison.",
    )
    parser.add_argument("--minimum-osm-features", type=int, default=5)
    parser.add_argument("--inner-permutation-repeats", type=int, default=2)
    parser.add_argument("--inner-group-folds", type=int, default=5)
    parser.add_argument("--temporal-correction-model", default="random_forest")
    parser.add_argument("--tabular-model", default="extra_trees")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def rank_features_inside_outer_train(
    training: pd.DataFrame,
    columns: list[str],
    *,
    local_target_col: str,
    date_col: str,
    model_name: str,
    repeats: int,
    seed: int,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank predictors using inner held-out dates within outer training only."""
    dates = sorted(training[date_col].astype(str).unique())
    if len(dates) < 2:
        raise ValueError("Feature ranking requires at least two outer-training dates")
    rows: list[dict[str, object]] = []
    for inner_fold, inner_date in enumerate(dates, start=1):
        inner_train = training[training[date_col].astype(str) != inner_date]
        inner_validation = training[training[date_col].astype(str) == inner_date]
        candidates = estimators(seed + inner_fold - 1)
        if model_name not in candidates:
            raise ValueError(f"Unknown ranking model: {model_name}")
        model = candidates[model_name]
        model.fit(
            inner_train[columns],
            inner_train[local_target_col].to_numpy(dtype=float),
        )
        result = permutation_importance(
            model,
            inner_validation[columns],
            inner_validation[local_target_col].to_numpy(dtype=float),
            scoring="neg_root_mean_squared_error",
            n_repeats=repeats,
            random_state=seed + inner_fold - 1,
            n_jobs=jobs,
        )
        for feature, importance, importance_std in zip(
            columns,
            result.importances_mean,
            result.importances_std,
            strict=True,
        ):
            rows.append(
                {
                    "inner_validation_date": inner_date,
                    "feature": feature,
                    "rmse_increase_when_permuted": float(importance),
                    "permutation_std": float(importance_std),
                }
            )
    per_inner = pd.DataFrame(rows)
    minimum_positive = math.ceil(0.67 * len(dates))
    ranked = (
        per_inner.groupby("feature", as_index=False)
        .agg(
            importance_mean=("rmse_increase_when_permuted", "mean"),
            importance_median=("rmse_increase_when_permuted", "median"),
            importance_min=("rmse_increase_when_permuted", "min"),
            positive_inner_folds=(
                "rmse_increase_when_permuted",
                lambda values: int((values > 0).sum()),
            ),
        )
    )
    ranked["stable_inner"] = (
        (ranked["positive_inner_folds"] >= minimum_positive)
        & (ranked["importance_mean"] > 0)
        & (ranked["importance_median"] > 0)
    )
    ranked = ranked.sort_values(
        ["stable_inner", "importance_mean", "positive_inner_folds"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return ranked, per_inner


def select_budget(
    ranked: pd.DataFrame,
    *,
    budget: int,
    all_columns: list[str],
    osm_columns: set[str],
    minimum_osm_features: int,
) -> list[str]:
    """Select a ranked budget while guaranteeing a small OSM representation."""
    if budget >= len(all_columns):
        return list(all_columns)
    if budget <= 0:
        raise ValueError("Feature budgets must be positive")
    osm_quota = min(minimum_osm_features, budget, len(osm_columns))
    ranked_features = ranked["feature"].tolist()
    selected = [
        feature for feature in ranked_features if feature in osm_columns
    ][:osm_quota]
    for feature in ranked_features:
        if feature not in selected:
            selected.append(feature)
        if len(selected) == budget:
            break
    if len(selected) != budget:
        raise RuntimeError(f"Could select only {len(selected)} of {budget} features")
    return selected


def prepare_prediction_table(
    path: Path,
    table: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    result = pd.read_csv(path)
    result["target_sample_id"] = result["target_sample_id"].astype(str)
    if BACKGROUND in result.columns:
        background_by_sample = table.set_index("sample_id")[BACKGROUND]
        expected_background = result["target_sample_id"].map(background_by_sample)
        if expected_background.isna().any():
            raise ValueError(
                f"Prediction targets in {path} are missing background rows"
            )
        if not np.allclose(
            result[BACKGROUND].to_numpy(dtype=float),
            expected_background.to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ValueError(
                f"Background values in {path} disagree with the supplied "
                "background table"
            )
        # The model-table join below is the authoritative source. Dropping this
        # existing copy prevents pandas from creating BACKGROUND_x/BACKGROUND_y.
        result = result.drop(columns=[BACKGROUND])
    result = result.merge(
        table[["sample_id", "run_id", BACKGROUND, *feature_columns]],
        left_on="target_sample_id",
        right_on="sample_id",
        validate="one_to_one",
    )
    if result["sample_id"].isna().any():
        raise ValueError(f"Prediction targets in {path} are missing model-table rows")
    return result


def fit_temporal_test_correction(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    *,
    target: str,
    model_name: str,
    seed: int,
) -> np.ndarray:
    if not columns:
        return test[f"predicted_{target}"].to_numpy(dtype=float)
    validation_actual = validation[f"actual_{target}"].to_numpy(dtype=float)
    validation_base = validation[f"predicted_{target}"].to_numpy(dtype=float)
    residual = validation_actual - validation_base
    candidates = estimators(seed)
    if model_name not in candidates:
        raise ValueError(f"Unknown temporal correction model: {model_name}")
    model = candidates[model_name]
    model.fit(validation[columns], residual)
    return (
        test[f"predicted_{target}"].to_numpy(dtype=float)
        + model.predict(test[columns])
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = json.loads(args.feature_groups.read_text())
    requested_groups = {
        args.candidate_feature_set,
        args.temporal_feature_pool,
        args.tabular_feature_pool,
    }
    missing_groups = sorted(requested_groups - set(groups))
    if missing_groups:
        raise ValueError(f"Unknown feature groups: {missing_groups}")
    candidate_columns = list(
        dict.fromkeys(groups[args.candidate_feature_set]["columns"])
    )
    temporal_pool = set(groups[args.temporal_feature_pool]["columns"])
    tabular_pool = set(groups[args.tabular_feature_pool]["columns"])
    leaked = sorted(set(candidate_columns) & FORBIDDEN_PM_OPC)
    if leaked:
        raise ValueError(f"Candidate pool contains forbidden PM/OPC fields: {leaked}")
    budgets = sorted(set(args.budgets))
    if any(budget > len(candidate_columns) for budget in budgets):
        raise ValueError(
            f"Budgets cannot exceed the {len(candidate_columns)}-feature pool"
        )
    osm_reference = set(groups["visual_yolo_road_osm"]["columns"])
    osm_reference -= set(groups["visual_yolo_road"]["columns"])
    osm_columns = set(candidate_columns) & osm_reference
    if args.minimum_osm_features > len(osm_columns):
        raise ValueError(
            f"Requested {args.minimum_osm_features} OSM features but only "
            f"{len(osm_columns)} are available"
        )

    table = pd.read_csv(args.modeling_table)
    table["sample_id"] = table["sample_id"].astype(str)
    if not table["sample_id"].is_unique:
        raise ValueError("Modeling table sample_id must be unique")
    background = pd.read_csv(
        args.background_csv,
        usecols=["sample_id", BACKGROUND],
    )
    background["sample_id"] = background["sample_id"].astype(str)
    if not background["sample_id"].is_unique:
        raise ValueError("Background table sample_id must be unique")
    table = table.merge(background, on="sample_id", validate="one_to_one").copy()
    table["_local_target"] = (
        table[args.target].to_numpy(dtype=float)
        - table[BACKGROUND].to_numpy(dtype=float)
    )
    required = {
        args.target, args.date_col, "run_id", BACKGROUND, *candidate_columns,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Modeling table is missing required columns: {missing}")

    ladder_results = pd.read_csv(args.temporal_ladder / "model_ladder_results.csv")
    fold_lookup = (
        ladder_results[["fold", "test_date"]]
        .drop_duplicates()
        .sort_values("fold")
    )
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    ranking_rows: list[pd.DataFrame] = []
    inner_importance_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []

    for record in fold_lookup.itertuples(index=False):
        fold = int(record.fold)
        test_date = str(record.test_date)
        split_candidates = sorted(
            (args.temporal_ladder / "splits").glob(
                f"fold_{fold}_test_*.csv"
            )
        )
        if len(split_candidates) != 1:
            raise ValueError(f"Fold {fold}: expected exactly one split manifest")
        split = pd.read_csv(split_candidates[0])
        split["sample_id"] = split["sample_id"].astype(str)
        train_ids = set(
            split.loc[split["split_outer_day"].eq("train"), "sample_id"]
        )
        validation_ids_all = set(
            split.loc[split["split_outer_day"].eq("val"), "sample_id"]
        )
        test_ids_all = set(
            split.loc[split["split_outer_day"].eq("test"), "sample_id"]
        )
        if (
            train_ids & validation_ids_all
            or train_ids & test_ids_all
            or validation_ids_all & test_ids_all
        ):
            raise ValueError(f"Fold {fold}: split sample IDs overlap")
        outer_train = table[table["sample_id"].isin(train_ids)].copy()
        if outer_train.empty:
            raise ValueError(f"Fold {fold}: outer-training partition is empty")

        ranked, per_inner = rank_features_inside_outer_train(
            outer_train,
            candidate_columns,
            local_target_col="_local_target",
            date_col=args.date_col,
            model_name=args.tabular_model,
            repeats=args.inner_permutation_repeats,
            seed=args.seed + 1000 * fold,
            jobs=args.jobs,
        )
        ranked.insert(0, "fold", fold)
        ranked.insert(1, "test_date", test_date)
        ranking_rows.append(ranked)
        per_inner.insert(0, "fold", fold)
        per_inner.insert(1, "test_date", test_date)
        inner_importance_rows.append(per_inner)

        model_dir = (
            args.temporal_ladder
            / "models"
            / f"fold_{fold}"
            / args.view
            / f"T{args.sequence_length}"
            / args.cell
        )
        validation = prepare_prediction_table(
            model_dir / "predictions_val_decomposition.csv",
            table,
            candidate_columns,
        )
        test = prepare_prediction_table(
            model_dir / "predictions_test_decomposition.csv",
            table,
            candidate_columns,
        )
        validation_target_ids = set(validation["target_sample_id"])
        test_target_ids = set(test["target_sample_id"])
        if not validation_target_ids <= validation_ids_all:
            raise ValueError(f"Fold {fold}: validation sequence targets violate split")
        if not test_target_ids <= test_ids_all:
            raise ValueError(f"Fold {fold}: test sequence targets violate split")
        development = table[
            table["sample_id"].isin(train_ids | validation_ids_all)
        ].copy()

        for budget in budgets:
            selected = select_budget(
                ranked,
                budget=budget,
                all_columns=candidate_columns,
                osm_columns=osm_columns,
                minimum_osm_features=args.minimum_osm_features,
            )
            selected_temporal = [
                feature for feature in selected if feature in temporal_pool
            ]
            selected_tabular = [
                feature for feature in selected if feature in tabular_pool
            ]
            if not selected_temporal:
                raise ValueError(
                    f"Fold {fold}, budget {budget}: no temporal features selected"
                )
            if not selected_tabular:
                raise ValueError(
                    f"Fold {fold}, budget {budget}: no tabular features selected"
                )
            selected_osm = [feature for feature in selected if feature in osm_columns]
            selection_rows.extend(
                {
                    "fold": fold,
                    "test_date": test_date,
                    "budget": budget,
                    "feature": feature,
                    "used_by_temporal": feature in temporal_pool,
                    "used_by_tabular": feature in tabular_pool,
                    "is_osm": feature in osm_columns,
                    "rank": selected.index(feature) + 1,
                }
                for feature in selected
            )

            temporal_validation = crossfit_temporal_correction(
                validation,
                selected_temporal,
                target=args.target,
                model_name=args.temporal_correction_model,
                folds=args.inner_group_folds,
                seed=args.seed + 100 * fold,
            )
            temporal_test = fit_temporal_test_correction(
                validation,
                test,
                selected_temporal,
                target=args.target,
                model_name=args.temporal_correction_model,
                seed=args.seed + 100 * fold,
            )

            tabular_candidates = estimators(args.seed + fold)
            if args.tabular_model not in tabular_candidates:
                raise ValueError(f"Unknown tabular model: {args.tabular_model}")
            tabular_validation_model = tabular_candidates[args.tabular_model]
            tabular_validation_model.fit(
                outer_train[selected_tabular],
                outer_train["_local_target"].to_numpy(dtype=float),
            )
            tabular_validation = (
                validation[BACKGROUND].to_numpy(dtype=float)
                + tabular_validation_model.predict(
                    validation[selected_tabular]
                )
            )

            tabular_test_candidates = estimators(args.seed + 10_000 + fold)
            tabular_test_model = tabular_test_candidates[args.tabular_model]
            tabular_test_model.fit(
                development[selected_tabular],
                development["_local_target"].to_numpy(dtype=float),
            )
            tabular_test = (
                test[BACKGROUND].to_numpy(dtype=float)
                + tabular_test_model.predict(test[selected_tabular])
            )

            validation_truth = validation[f"actual_{args.target}"].to_numpy(
                dtype=float
            )
            test_truth = test[f"actual_{args.target}"].to_numpy(dtype=float)
            weight = optimal_convex_weight(
                validation_truth,
                temporal_validation,
                tabular_validation,
            )
            ensemble_test = weight * temporal_test + (1.0 - weight) * tabular_test
            for method, prediction, method_weight in (
                ("validation_selected_weight", ensemble_test, weight),
                ("fixed_equal_weight_diagnostic", 0.5 * (temporal_test + tabular_test), 0.5),
                ("temporal_branch", temporal_test, 1.0),
                ("tabular_branch", tabular_test, 0.0),
            ):
                metric_rows.append(
                    {
                        "fold": fold,
                        "test_date": test_date,
                        "budget": budget,
                        "selected_features_total": len(selected),
                        "temporal_features": len(selected_temporal),
                        "tabular_features": len(selected_tabular),
                        "osm_features": len(selected_osm),
                        "method": method,
                        "temporal_weight": method_weight,
                        "n_test": len(test),
                        **metric_row(test_truth, prediction),
                    }
                )
                part = test[
                    ["target_sample_id", f"actual_{args.target}"]
                ].copy()
                part["fold"] = fold
                part["test_date"] = test_date
                part["budget"] = budget
                part["method"] = method
                part["temporal_weight"] = method_weight
                part["temporal_prediction"] = temporal_test
                part["tabular_prediction"] = tabular_test
                part["prediction"] = prediction
                prediction_rows.append(part)
            print(
                f"completed fold={fold} test_date={test_date} budget={budget} "
                f"temporal={len(selected_temporal)} "
                f"tabular={len(selected_tabular)} osm={len(selected_osm)}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    aggregate_rows: list[dict[str, object]] = []
    for (budget, method), part in predictions.groupby(
        ["budget", "method"],
        sort=False,
    ):
        fold_part = metrics[
            metrics["budget"].eq(budget) & metrics["method"].eq(method)
        ]
        aggregate_rows.append(
            {
                "budget": int(budget),
                "method": method,
                "n_test": len(part),
                **metric_row(
                    part[f"actual_{args.target}"].to_numpy(dtype=float),
                    part["prediction"].to_numpy(dtype=float),
                ),
                "mean_fold_r2": float(fold_part["r2"].mean()),
                "mean_fold_rmse": float(fold_part["rmse"].mean()),
                "worst_fold_rmse": float(fold_part["rmse"].max()),
                "temporal_features_mean": float(
                    fold_part["temporal_features"].mean()
                ),
                "tabular_features_mean": float(
                    fold_part["tabular_features"].mean()
                ),
                "osm_features_mean": float(fold_part["osm_features"].mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["method", "rmse"])
    selections = pd.DataFrame(selection_rows)
    selection_frequency = (
        selections.groupby(["budget", "feature"], as_index=False)
        .agg(
            selected_folds=("fold", "nunique"),
            used_by_temporal=("used_by_temporal", "max"),
            used_by_tabular=("used_by_tabular", "max"),
            is_osm=("is_osm", "max"),
            mean_rank=("rank", "mean"),
        )
        .sort_values(
            ["budget", "selected_folds", "mean_rank"],
            ascending=[True, False, True],
        )
    )

    metrics.to_csv(args.output_dir / "metrics_by_fold.csv", index=False)
    aggregate.to_csv(args.output_dir / "metrics_aggregate.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.concat(ranking_rows, ignore_index=True).to_csv(
        args.output_dir / "outer_train_feature_rankings.csv",
        index=False,
    )
    pd.concat(inner_importance_rows, ignore_index=True).to_csv(
        args.output_dir / "inner_date_permutation_importance.csv",
        index=False,
    )
    selections.to_csv(args.output_dir / "selected_features_by_fold.csv", index=False)
    selection_frequency.to_csv(
        args.output_dir / "selection_frequency.csv",
        index=False,
    )
    summary = {
        "protocol": (
            "whole-date outer holdout; feature ranking inside outer train; "
            "outer validation reserved for convex ensemble weight selection"
        ),
        "candidate_feature_set": args.candidate_feature_set,
        "candidate_features": len(candidate_columns),
        "budgets": budgets,
        "minimum_osm_features_per_compact_fold": args.minimum_osm_features,
        "ranking": (
            f"inner held-out-date permutation RMSE using {args.tabular_model}"
        ),
        "temporal_feature_pool": args.temporal_feature_pool,
        "tabular_feature_pool": args.tabular_feature_pool,
        "pm_opc_target_proxies_used": False,
        "outer_test_targets_used_for_selection": False,
        "confirmatory": False,
        "reason": (
            "The compact budget was proposed after inspecting earlier five-day "
            "results; use untouched future dates for confirmation."
        ),
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("\nPooled feature-budget comparison:")
    print(aggregate.to_string(index=False))
    print("\nCompact features most frequently selected:")
    compact_budget = min(budgets)
    print(
        selection_frequency[
            selection_frequency["budget"].eq(compact_budget)
        ].head(100).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
