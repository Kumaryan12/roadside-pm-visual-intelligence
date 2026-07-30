"""Merge a pre-extracted AlphaEarth feature table into the modeling table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from roadside_pm.features.geospatial.merge import merge_sample_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--alphaearth-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--key", default="sample_index")
    args = parser.parse_args()

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_sample_features(
        pd.read_csv(args.input_table),
        pd.read_csv(args.alphaearth_csv),
        key=args.key,
        feature_prefix="aef_",
    )
    merged.to_csv(output, index=False)
    print(f"Saved modeling table with AlphaEarth features: {output}")


if __name__ == "__main__":
    main()

