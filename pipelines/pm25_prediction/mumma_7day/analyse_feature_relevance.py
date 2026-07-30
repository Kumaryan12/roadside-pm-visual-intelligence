"""Leakage-aware feature relevance analysis for the five-day MUMMA table.

Produces descriptive global/within-day correlations and held-out-date
permutation importance. Correlations are not used as causal evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def correlations(frame: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
    numeric = frame[columns + [target]].apply(pd.to_numeric, errors="coerce")
    centered = numeric.groupby(frame["date"]).transform(lambda values: values - values.mean())
    rows = []
    for column in columns:
        global_constant = numeric[column].nunique(dropna=True) <= 1
        within_constant = centered[column].nunique(dropna=True) <= 1
        rows.append(
            {
                "feature": column,
                "global_pearson": np.nan if global_constant else numeric[column].corr(numeric[target], method="pearson"),
                "global_spearman": np.nan if global_constant else numeric[column].corr(numeric[target], method="spearman"),
                "within_day_pearson": np.nan if within_constant else centered[column].corr(centered[target], method="pearson"),
                "within_day_spearman": np.nan if within_constant else centered[column].corr(centered[target], method="spearman"),
                "non_null_rows": int(numeric[column].notna().sum()),
                "unique_values": int(numeric[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--feature-groups", type=Path, required=True)
    parser.add_argument("--feature-set", default="visual_yolo_road_osm_alphaearth")
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    frame = pd.read_csv(args.modeling_table)
    groups = json.loads(args.feature_groups.read_text())
    if args.feature_set not in groups:
        raise ValueError(f"Unknown feature set {args.feature_set!r}")
    if not groups[args.feature_set].get("reportable", False):
        raise ValueError("Feature relevance requested for a non-reportable feature group")
    columns = list(dict.fromkeys(groups[args.feature_set]["columns"]))
    forbidden = [
        column
        for column in columns
        if column in {"sPM1", "sPM4", "sPM10", "sNPMp5", "sNPM1", "sNPM2", "sNPM4", "sNPM10", "sTPS"}
    ]
    if forbidden:
        raise ValueError(f"PM/OPC proxy predictors are forbidden: {forbidden}")

    missing = sorted(set(columns + [args.target, "date"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    correlation_table = correlations(frame, columns, args.target)

    dates = sorted(frame["date"].astype(str).unique())
    importance_rows = []
    fold_rows = []
    for fold, test_date in enumerate(dates, start=1):
        train = frame[frame["date"].astype(str) != test_date]
        test = frame[frame["date"].astype(str) == test_date]
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=250,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        model.fit(train[columns], train[args.target].to_numpy(dtype=float))
        prediction = model.predict(test[columns])
        baseline_rmse = float(
            np.sqrt(mean_squared_error(test[args.target].to_numpy(dtype=float), prediction))
        )
        baseline_r2 = float(r2_score(test[args.target].to_numpy(dtype=float), prediction))
        result = permutation_importance(
            model,
            test[columns],
            test[args.target].to_numpy(dtype=float),
            scoring="neg_root_mean_squared_error",
            n_repeats=args.repeats,
            random_state=args.seed,
            n_jobs=args.jobs,
        )
        fold_rows.append(
            {
                "fold": fold,
                "test_date": test_date,
                "n_train": len(train),
                "n_test": len(test),
                "baseline_rmse": baseline_rmse,
                "baseline_r2": baseline_r2,
            }
        )
        for column, mean, std in zip(
            columns, result.importances_mean, result.importances_std, strict=True
        ):
            importance_rows.append(
                {
                    "fold": fold,
                    "test_date": test_date,
                    "feature": column,
                    "rmse_increase_when_permuted": float(mean),
                    "permutation_std": float(std),
                    "baseline_rmse": baseline_rmse,
                    "baseline_r2": baseline_r2,
                }
            )
        print(f"completed fold {fold}/{len(dates)} test_date={test_date}", flush=True)

    per_fold = pd.DataFrame(importance_rows)
    aggregate = (
        per_fold.groupby("feature", as_index=False)
        .agg(
            permutation_rmse_increase_mean=("rmse_increase_when_permuted", "mean"),
            permutation_rmse_increase_median=("rmse_increase_when_permuted", "median"),
            permutation_rmse_increase_min=("rmse_increase_when_permuted", "min"),
            positive_folds=("rmse_increase_when_permuted", lambda values: int((values > 0).sum())),
            negative_folds=("rmse_increase_when_permuted", lambda values: int((values < 0).sum())),
        )
        .merge(correlation_table, on="feature", how="left", validate="one_to_one")
    )
    aggregate["stable_transfer_feature"] = (
        (aggregate["positive_folds"] >= 4)
        & (aggregate["permutation_rmse_increase_mean"] > 0)
        & (aggregate["permutation_rmse_increase_median"] > 0)
    )
    aggregate = aggregate.sort_values(
        ["stable_transfer_feature", "permutation_rmse_increase_mean"],
        ascending=[False, False],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_dir / "feature_relevance.csv", index=False)
    per_fold.to_csv(args.output_dir / "permutation_importance_by_fold.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary = {
        "feature_set": args.feature_set,
        "target": args.target,
        "features_evaluated": len(columns),
        "protocol": "leave_one_date_out_test_permutation_importance",
        "permutation_repeats": args.repeats,
        "pm_opc_target_proxies_used": False,
        "stable_transfer_rule": "positive held-out-date importance in at least four of five folds",
        "stable_transfer_features": aggregate.loc[
            aggregate["stable_transfer_feature"], "feature"
        ].tolist(),
        "warnings": [
            "Correlation is descriptive and does not imply causation.",
            "Permutation importance is unreliable when predictors are strongly correlated.",
            "Only five dates are available, so stability estimates remain preliminary.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nTop held-out-date permutation features:")
    print(
        aggregate[
            [
                "feature",
                "permutation_rmse_increase_mean",
                "positive_folds",
                "within_day_spearman",
                "global_spearman",
                "stable_transfer_feature",
            ]
        ].head(30).to_string(index=False)
    )
    print(f"\nSaved feature relevance to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
