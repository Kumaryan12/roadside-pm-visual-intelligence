"""Select visual weights from inner held-site predictions and score outer sites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score(actual, predicted):
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
        "pearson": float(pearsonr(actual, predicted)[0]),
        "spearman": float(spearmanr(actual, predicted)[0]),
    }


def choose_alpha(data: pd.DataFrame, maximum: float, step: float):
    grid = np.arange(0.0, maximum + step / 2, step)
    errors = []
    for alpha in grid:
        prediction = (
            data["predicted_reference_context"]
            + alpha * data["predicted_conditioned_image_correction"]
        )
        errors.append(mean_squared_error(data["actual_pm25"], prediction) ** 0.5)
    index = int(np.argmin(errors))
    return float(grid[index]), float(errors[index])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha-max", type=float, default=0.30)
    parser.add_argument("--alpha-step", type=float, default=0.02)
    args = parser.parse_args()
    root = Path(args.benchmark_root) / "taiwan"
    outer_tables = []
    fold_reports = []
    for outer in sorted(root.glob("site_*")):
        inner_tables = []
        for inner in sorted((outer / "inner_alpha_cv").glob("held_*")):
            path = inner / "model" / "predictions_test.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing inner held-site prediction: {path}")
            table = pd.read_csv(path)
            table["inner_test_site"] = inner.name.removeprefix("held_")
            inner_tables.append(table)
        if not inner_tables:
            raise FileNotFoundError(f"No inner predictions below {outer / 'inner_alpha_cv'}")
        inner_data = pd.concat(inner_tables, ignore_index=True)
        alpha, inner_rmse = choose_alpha(inner_data, args.alpha_max, args.alpha_step)

        outer_path = outer / "models" / "conditioned_reference_gru_T7_pm25" / "predictions_test.csv"
        if not outer_path.exists():
            raise FileNotFoundError(f"Missing outer prediction: {outer_path}")
        test = pd.read_csv(outer_path)
        test["test_site"] = outer.name.removeprefix("site_")
        test["inner_cv_selected_alpha"] = alpha
        test["predicted_inner_cv_conservative_image"] = (
            test["predicted_reference_context"]
            + alpha * test["predicted_conditioned_image_correction"]
        )
        outer_tables.append(test)
        fold_reports.append(
            {
                "test_site": outer.name.removeprefix("site_"),
                "inner_cv_selected_alpha": alpha,
                "inner_cv_rmse": inner_rmse,
                "inner_sites": sorted(inner_data["inner_test_site"].astype(str).unique()),
                "outer_test": score(
                    test["actual_pm25"], test["predicted_inner_cv_conservative_image"]
                ),
            }
        )

    pooled = pd.concat(outer_tables, ignore_index=True)
    methods = {
        "reference_only": "predicted_reference_only",
        "reference_context": "predicted_reference_context",
        "inner_cv_conservative_conditioned_image": "predicted_inner_cv_conservative_image",
    }
    pooled_rows, site_rows = [], []
    for method, column in methods.items():
        pooled_rows.append({"method": method, **score(pooled.actual_pm25, pooled[column])})
        for site, group in pooled.groupby("test_site"):
            site_rows.append(
                {"method": method, "test_site": site, **score(group.actual_pm25, group[column])}
            )
    pooled_metrics = pd.DataFrame(pooled_rows)
    site_metrics = pd.DataFrame(site_rows)
    site_summary = (
        site_metrics.groupby("method")
        .agg(
            mean_site_r2=("r2", "mean"), median_site_r2=("r2", "median"),
            positive_site_r2_count=("r2", lambda x: int((x > 0).sum())),
            sites=("test_site", "nunique"), mean_site_rmse=("rmse", "mean"),
            worst_site_rmse=("rmse", "max"),
        )
        .reset_index()
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output / "predictions_pooled.csv", index=False)
    pooled_metrics.to_csv(output / "metrics_pooled.csv", index=False)
    site_metrics.to_csv(output / "metrics_by_site.csv", index=False)
    site_summary.to_csv(output / "metrics_site_summary.csv", index=False)
    pd.DataFrame(fold_reports).to_csv(output / "inner_selected_alphas.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "protocol": "outer_leave_one_site_out_inner_training_site_crossfit_alpha",
                "alpha_search": {"minimum": 0.0, "maximum": args.alpha_max, "step": args.alpha_step},
                "outer_test_used_for_alpha_selection": False,
                "folds": fold_reports,
                "pooled": pooled_metrics.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print("\nTAIWAN INNER-CV CONSERVATIVE VISUAL EXPERT\n")
    print(pooled_metrics.to_string(index=False))
    print("\nSITE SUMMARY\n")
    print(site_summary.to_string(index=False))
    print("\nINNER-CV SELECTED ALPHAS\n")
    print(pd.DataFrame(fold_reports)[["test_site", "inner_cv_selected_alpha", "inner_cv_rmse"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
