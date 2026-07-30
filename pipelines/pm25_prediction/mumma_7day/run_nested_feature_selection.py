"""Nested leave-one-date-out feature selection for MUMMA PM2.5.

Feature selection is repeated inside every outer fold using permutation
importance on inner held-out dates. The outer test date is not used to rank or
select features. A reduced HGB model is paired with the full feature model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


FORBIDDEN_PM_OPC = {
    "sPM1", "sPM4", "sPM10", "sNPMp5", "sNPM1", "sNPM2",
    "sNPM4", "sNPM10", "sTPS",
}


def new_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
        "bias": float(np.mean(y_pred - y_true)),
    }


def rank_inside_outer_train(
    frame: pd.DataFrame,
    columns: list[str],
    target: str,
    date_col: str,
    outer_test_date: str,
    repeats: int,
    seed: int,
    jobs: int,
) -> pd.DataFrame:
    training_dates = sorted(
        frame.loc[frame[date_col].astype(str) != outer_test_date, date_col]
        .astype(str)
        .unique()
    )
    rows: list[dict] = []
    for inner_date in training_dates:
        inner_train = frame[
            (frame[date_col].astype(str) != outer_test_date)
            & (frame[date_col].astype(str) != inner_date)
        ]
        inner_val = frame[frame[date_col].astype(str) == inner_date]
        model = new_model(seed)
        model.fit(inner_train[columns], inner_train[target].to_numpy(dtype=float))
        result = permutation_importance(
            model,
            inner_val[columns],
            inner_val[target].to_numpy(dtype=float),
            scoring="neg_root_mean_squared_error",
            n_repeats=repeats,
            random_state=seed,
            n_jobs=jobs,
        )
        for feature, mean, std in zip(
            columns, result.importances_mean, result.importances_std, strict=True
        ):
            rows.append(
                {
                    "inner_validation_date": inner_date,
                    "feature": feature,
                    "rmse_increase_when_permuted": float(mean),
                    "permutation_std": float(std),
                }
            )
    per_inner = pd.DataFrame(rows)
    minimum_positive = math.ceil(0.75 * len(training_dates))
    ranked = (
        per_inner.groupby("feature", as_index=False)
        .agg(
            importance_mean=("rmse_increase_when_permuted", "mean"),
            importance_median=("rmse_increase_when_permuted", "median"),
            importance_min=("rmse_increase_when_permuted", "min"),
            positive_inner_folds=(
                "rmse_increase_when_permuted", lambda values: int((values > 0).sum())
            ),
        )
    )
    ranked["stable_inner"] = (
        (ranked["positive_inner_folds"] >= minimum_positive)
        & (ranked["importance_mean"] > 0)
        & (ranked["importance_median"] > 0)
    )
    return ranked.sort_values(
        ["stable_inner", "importance_mean", "positive_inner_folds"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--feature-groups", type=Path, required=True)
    parser.add_argument("--feature-set", default="visual_yolo_road_osm_alphaearth")
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--permutation-repeats", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.modeling_table)
    groups = json.loads(args.feature_groups.read_text())
    if args.feature_set not in groups:
        raise ValueError(f"Unknown feature set: {args.feature_set}")
    if not groups[args.feature_set].get("reportable", False):
        raise ValueError("The requested candidate feature group is non-reportable")
    columns = list(dict.fromkeys(groups[args.feature_set]["columns"]))
    forbidden = sorted(set(columns) & FORBIDDEN_PM_OPC)
    if forbidden:
        raise ValueError(f"PM/OPC proxy predictors are forbidden: {forbidden}")
    if args.top_k <= 0 or args.top_k > len(columns):
        raise ValueError(f"top-k must be between 1 and {len(columns)}")
    missing = sorted(set(columns + [args.target, args.date_col]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    dates = sorted(frame[args.date_col].astype(str).unique())
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    for fold, outer_date in enumerate(dates, start=1):
        ranked = rank_inside_outer_train(
            frame,
            columns,
            args.target,
            args.date_col,
            outer_date,
            args.permutation_repeats,
            args.seed,
            args.jobs,
        )
        selected = ranked.head(args.top_k)["feature"].tolist()
        ranked = ranked.copy()
        ranked.insert(0, "outer_fold", fold)
        ranked.insert(1, "outer_test_date", outer_date)
        ranked["selected"] = ranked["feature"].isin(selected)
        selected_rows.append(ranked)

        train = frame[frame[args.date_col].astype(str) != outer_date]
        test = frame[frame[args.date_col].astype(str) == outer_date]
        y_train = train[args.target].to_numpy(dtype=float)
        y_test = test[args.target].to_numpy(dtype=float)
        for label, model_columns in [("nested_top_k", selected), ("all_features", columns)]:
            model = new_model(args.seed)
            model.fit(train[model_columns], y_train)
            prediction = model.predict(test[model_columns])
            metric_rows.append(
                {
                    "outer_fold": fold,
                    "outer_test_date": outer_date,
                    "model_variant": label,
                    "n_features": len(model_columns),
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics(y_test, prediction),
                }
            )
            table = test[["sample_id", "run_id", args.date_col, "sample_timestamp"]].copy()
            table["outer_fold"] = fold
            table["model_variant"] = label
            table["y_true"] = y_test
            table["y_pred"] = prediction
            prediction_rows.append(table)
        print(
            f"completed outer fold {fold}/{len(dates)} test_date={outer_date} "
            f"selected_features={len(selected)}",
            flush=True,
        )

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    rankings = pd.concat(selected_rows, ignore_index=True)
    aggregate_rows = []
    for variant, group in predictions.groupby("model_variant"):
        pooled = metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        folds = fold_metrics[fold_metrics["model_variant"] == variant]
        aggregate_rows.append(
            {
                "model_variant": variant,
                "n_features": args.top_k if variant == "nested_top_k" else len(columns),
                **{f"pooled_{key}": value for key, value in pooled.items()},
                **{
                    f"mean_fold_{key}": float(folds[key].mean())
                    for key in ["MAE", "RMSE", "R2", "Spearman", "bias"]
                },
                "worst_fold_RMSE": float(folds["RMSE"].max()),
                "worst_fold_R2": float(folds["R2"].min()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values("pooled_RMSE")
    frequency = (
        rankings[rankings["selected"]]
        .groupby("feature", as_index=False)
        .agg(
            selected_outer_folds=("outer_fold", "nunique"),
            mean_inner_importance=("importance_mean", "mean"),
            median_inner_importance=("importance_median", "median"),
        )
        .sort_values(["selected_outer_folds", "mean_inner_importance"], ascending=False)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "metrics_aggregate.csv", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    rankings.to_csv(args.output_dir / "inner_feature_rankings.csv", index=False)
    frequency.to_csv(args.output_dir / "selection_frequency.csv", index=False)
    summary = {
        "protocol": "nested_leave_one_date_out_feature_selection",
        "reportable_as_unseen_date_generalization": True,
        "candidate_feature_set": args.feature_set,
        "candidate_features": len(columns),
        "selected_per_outer_fold": args.top_k,
        "outer_dates": dates,
        "inner_selection": "held-out-date permutation importance within outer training only",
        "pm_opc_target_proxies_used": False,
        "comparison": "nested top-k versus all candidate features",
    }
    (args.output_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nAggregate results:")
    print(aggregate.to_string(index=False))
    print("\nMost consistently selected features:")
    print(frequency.head(40).to_string(index=False))
    print(f"\nSaved results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
