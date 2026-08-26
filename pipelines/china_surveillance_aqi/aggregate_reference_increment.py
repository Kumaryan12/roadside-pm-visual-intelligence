"""Aggregate Taiwan site-held-out reference/increment predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
        "pearson": float(pearsonr(actual, predicted)[0]),
        "spearman": float(spearmanr(actual, predicted)[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.benchmark_root) / "taiwan"
    tables = []
    for fold in sorted(root.glob("site_*")):
        path = fold / "models" / "reference_image_gru_T7_pm25" / "predictions_test_final.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed fold prediction: {path}")
        table = pd.read_csv(path)
        table["test_site"] = fold.name.removeprefix("site_")
        tables.append(table)
    pooled = pd.concat(tables, ignore_index=True)
    methods = {
        "reference_only": "predicted_pm25_reference_only",
        "reference_plus_image_increment": "predicted_pm25_reference_plus_image_increment",
    }
    pooled_rows = []
    site_rows = []
    for method, column in methods.items():
        pooled_rows.append({"method": method, **score(pooled["actual_pm25"], pooled[column])})
        for site, group in pooled.groupby("test_site"):
            site_rows.append(
                {"method": method, "test_site": site, **score(group["actual_pm25"], group[column])}
            )
    pooled_metrics = pd.DataFrame(pooled_rows)
    site_metrics = pd.DataFrame(site_rows)
    summary_rows = []
    for method, group in site_metrics.groupby("method"):
        summary_rows.append(
            {
                "method": method,
                "mean_site_r2": float(group["r2"].mean()),
                "median_site_r2": float(group["r2"].median()),
                "positive_site_r2_count": int(group["r2"].gt(0).sum()),
                "sites": int(len(group)),
                "mean_site_rmse": float(group["rmse"].mean()),
                "worst_site_rmse": float(group["rmse"].max()),
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output / "predictions_pooled.csv", index=False)
    pooled_metrics.to_csv(output / "metrics_pooled.csv", index=False)
    site_metrics.to_csv(output / "metrics_by_site.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "metrics_site_summary.csv", index=False)
    report = {
        "protocol": "six_fold_leave_one_site_out_synchronized_reference",
        "pooled": pooled_metrics.to_dict(orient="records"),
        "site_summary": summary_rows,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print("\nTAIWAN REFERENCE-INCREMENT POOLED RESULTS\n")
    print(pooled_metrics.to_string(index=False))
    print("\nSITE SUMMARY\n")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
