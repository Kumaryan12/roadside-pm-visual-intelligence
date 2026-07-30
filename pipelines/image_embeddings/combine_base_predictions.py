"""Combine disjoint safe image-base prediction files for residual fusion."""

import argparse
from pathlib import Path

import pandas as pd

from roadside_pm.modeling.residual_fusion import SAFE_PREDICTION_ORIGINS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    combined = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    if combined["sequence_id"].duplicated().any(): raise ValueError("Prediction files contain duplicate sequence_id values")
    unsafe = set(combined["prediction_origin"].astype(str)) - SAFE_PREDICTION_ORIGINS
    if unsafe: raise ValueError(f"Unsafe prediction origins: {sorted(unsafe)}")
    output = Path(args.output_csv); output.parent.mkdir(parents=True, exist_ok=True); combined.to_csv(output, index=False)
    print(f"Saved {len(combined)} safe base predictions: {output}")


if __name__ == "__main__": main()

