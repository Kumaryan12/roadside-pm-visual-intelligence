"""Paired site-cluster bootstrap for the conservative visual expert gain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXT = "predicted_reference_context"
VISUAL = "predicted_inner_cv_conservative_image"


def cluster_summaries(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, group in data.groupby("test_site"):
        actual = group["actual_pm25"].to_numpy(dtype=float)
        row = {
            "site": str(site),
            "n": len(group),
            "sum_y": actual.sum(),
            "sum_y2": np.square(actual).sum(),
        }
        for name, column in (("context", CONTEXT), ("visual", VISUAL)):
            error = group[column].to_numpy(dtype=float) - actual
            row[f"sae_{name}"] = np.abs(error).sum()
            row[f"sse_{name}"] = np.square(error).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(summary: pd.DataFrame, counts: np.ndarray, prefix: str):
    n = float(np.dot(counts, summary["n"]))
    sae = float(np.dot(counts, summary[f"sae_{prefix}"]))
    sse = float(np.dot(counts, summary[f"sse_{prefix}"]))
    total = float(np.dot(counts, summary["sum_y2"])) - float(
        np.dot(counts, summary["sum_y"])
    ) ** 2 / n
    return sae / n, np.sqrt(sse / n), 1.0 - sse / total


def interval(values: np.ndarray):
    return {
        "estimate": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = pd.read_csv(args.predictions)
    required = {"test_site", "actual_pm25", CONTEXT, VISUAL}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")
    summary = cluster_summaries(data)
    sites = len(summary)
    if sites < 5:
        raise ValueError(f"Site-cluster bootstrap needs at least five sites, found {sites}")
    rng = np.random.default_rng(args.seed)
    draws = np.zeros((args.replicates, 3), dtype=float)
    for index in range(args.replicates):
        selected = rng.integers(0, sites, size=sites)
        counts = np.bincount(selected, minlength=sites)
        context = metrics(summary, counts, "context")
        visual = metrics(summary, counts, "visual")
        draws[index] = [
            visual[0] - context[0],
            visual[1] - context[1],
            visual[2] - context[2],
        ]
    columns = ["delta_mae_visual_minus_context", "delta_rmse_visual_minus_context", "delta_r2_visual_minus_context"]
    samples = pd.DataFrame(draws, columns=columns)
    result = {
        "protocol": "paired_nonparametric_site_cluster_bootstrap",
        "clusters": sites,
        "replicates": args.replicates,
        "seed": args.seed,
        "interpretation": "Negative MAE/RMSE deltas and positive R2 delta favor the visual expert.",
        "delta_mae": interval(samples[columns[0]].to_numpy()),
        "delta_rmse": interval(samples[columns[1]].to_numpy()),
        "delta_r2": interval(samples[columns[2]].to_numpy()),
        "bootstrap_probability_visual_improves": {
            "mae": float((samples[columns[0]] < 0).mean()),
            "rmse": float((samples[columns[1]] < 0).mean()),
            "r2": float((samples[columns[2]] > 0).mean()),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "site_sufficient_statistics.csv", index=False)
    samples.to_csv(output / "bootstrap_samples.csv", index=False)
    (output / "bootstrap_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
