"""Aggregate six Taiwan folds for the conditioned reference-context GRU."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.benchmark_root) / "taiwan"
    tables = []
    alphas = []
    for fold in sorted(root.glob("site_*")):
        model = fold / "models" / "conditioned_reference_gru_T7_pm25"
        path = model / "predictions_test.csv"
        metrics_path = model / "metrics.json"
        if not path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Missing completed fold: {model}")
        table = pd.read_csv(path)
        table["test_site"] = fold.name.removeprefix("site_")
        tables.append(table)
        alphas.append(
            {"test_site": fold.name.removeprefix("site_"), "alpha": json.loads(metrics_path.read_text())["validation_selected_alpha"]}
        )
    data = pd.concat(tables, ignore_index=True)
    methods = {
        "reference_only": "predicted_reference_only",
        "reference_context": "predicted_reference_context",
        "reference_context_plus_conditioned_image": "predicted_reference_context_plus_conditioned_image",
    }
    pooled_rows, site_rows = [], []
    for method, column in methods.items():
        pooled_rows.append({"method": method, **score(data.actual_pm25, data[column])})
        for site, group in data.groupby("test_site"):
            site_rows.append({"method": method, "test_site": site, **score(group.actual_pm25, group[column])})
    pooled = pd.DataFrame(pooled_rows)
    by_site = pd.DataFrame(site_rows)
    summary = (
        by_site.groupby("method")
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
    data.to_csv(output / "predictions_pooled.csv", index=False)
    pooled.to_csv(output / "metrics_pooled.csv", index=False)
    by_site.to_csv(output / "metrics_by_site.csv", index=False)
    summary.to_csv(output / "metrics_site_summary.csv", index=False)
    pd.DataFrame(alphas).to_csv(output / "validation_selected_alphas.csv", index=False)
    print("\nTAIWAN CONDITIONED REFERENCE MODEL\n")
    print(pooled.to_string(index=False))
    print("\nSITE SUMMARY\n")
    print(summary.to_string(index=False))
    print("\nVALIDATION-SELECTED IMAGE WEIGHTS\n")
    print(pd.DataFrame(alphas).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
