"""Merge canonical metric-depth road-area features into a modeling table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from roadside_pm.features.geospatial.merge import merge_sample_features


CANONICAL_COLUMNS = [
    "matched_run_id",
    "sample_index",
    "road_area_segmented_unoccluded_m2",
    "road_area_vehicle_occluded_m2",
    "road_area_effective_visible_m2",
    "road_area_vehicle_occlusion_fraction",
    "road_area_occlusion_quality",
    "road_area_feature_version",
    "road_area_features_available",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--road-area-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    base = pd.read_csv(args.input_table)
    road = pd.read_csv(args.road_area_csv)
    keys = ["matched_run_id", "sample_index"]
    if "matched_run_id" not in base.columns or "matched_run_id" not in road.columns:
        keys = ["sample_index"]
    selected = road[[column for column in CANONICAL_COLUMNS if column in road.columns]].copy()
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    merge_sample_features(base, selected, key=keys).to_csv(output, index=False)
    print(f"Saved modeling table with canonical road-area features: {output}")


if __name__ == "__main__":
    main()
