"""Normalize historical road-area v3 output to the canonical schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from roadside_pm.features.road.area import normalize_depth_gated_v3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_depth_gated_v3(pd.read_csv(args.input_csv))
    normalized.to_csv(output, index=False)
    print(f"Saved canonical road-area features: {output}")


if __name__ == "__main__":
    main()

