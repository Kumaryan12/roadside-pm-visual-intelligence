"""Evaluate reference-only and reference-plus-learned-increment predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
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
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    sequences = pd.read_csv(args.sequence_manifest)
    sequences = sequences.loc[sequences["split"].eq("test")].copy()
    predictions = pd.read_csv(args.predictions)
    table = sequences.merge(predictions, on="sequence_id", how="inner", validate="one_to_one")
    if len(table) != len(sequences):
        raise ValueError(
            f"Prediction coverage mismatch: expected {len(sequences)} test rows, found {len(table)}"
        )
    actual_increment = table["actual_target_local_increment"].to_numpy(dtype=float)
    expected_increment = table["target_local_increment"].to_numpy(dtype=float)
    if not np.allclose(actual_increment, expected_increment, atol=1e-4):
        raise ValueError("Prediction targets do not match the reference sequence manifest")
    background = table["background_reference_pm25"].to_numpy(dtype=float)
    actual_pm25 = table["target_pm25"].to_numpy(dtype=float)
    reconstructed_actual = background + actual_increment
    if not np.allclose(actual_pm25, reconstructed_actual, atol=1e-4):
        raise ValueError("PM2.5 decomposition identity failed")
    predicted_increment = table["predicted_target_local_increment"].to_numpy(dtype=float)
    table["actual_pm25"] = actual_pm25
    table["predicted_pm25_reference_only"] = background
    table["predicted_pm25_reference_plus_image_increment"] = background + predicted_increment

    report = {
        "protocol": "leave_one_site_out_synchronized_reference",
        "reference_only": metrics(actual_pm25, background),
        "reference_plus_image_increment": metrics(
            actual_pm25,
            table["predicted_pm25_reference_plus_image_increment"].to_numpy(dtype=float),
        ),
        "local_increment": metrics(actual_increment, predicted_increment),
        "target_site_or_target_pm25_used_in_background": False,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "predictions_test_final.csv", index=False)
    (output / "metrics_final.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
