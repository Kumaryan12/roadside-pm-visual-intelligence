"""Assemble leakage-safe MUMMA pilot features by stable sample identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VEHICLE_FEATURES = [
    "idd_total_vehicle_count",
    "idd_heavy_vehicle_count",
    "idd_motor_vehicle_count",
    "idd_exhaust_proxy_initial",
    "idd_resuspension_vehicle_proxy_initial",
    "idd_auto_rickshaw_count",
    "idd_bicycle_count",
    "idd_bus_count",
    "idd_car_count",
    "idd_motorcycle_count",
    "idd_truck_count",
    "idd_unknown_vehicle_count",
    "idd_vehicle_box_area_ratio",
    "idd_average_confidence",
    "idd_max_confidence",
]

ROAD_FEATURES = [
    "road_area_ratio",
    "road_mean_brightness",
    "road_mean_saturation",
    "road_contrast_std",
    "road_shadow_ratio",
    "road_glare_ratio",
    "road_brown_pixel_ratio",
    "road_gray_dry_pixel_ratio",
    "road_edge_density",
    "road_laplacian_std",
    "road_haze_flatness_proxy",
]

ROAD_AREA_FEATURES = [
    "road_area_segmented_unoccluded_m2",
    "road_area_vehicle_occluded_m2",
    "road_area_effective_visible_m2",
    "road_area_vehicle_occlusion_fraction",
    "road_depth_p5_m",
    "road_depth_p25_m",
    "road_depth_p50_m",
    "road_depth_p75_m",
    "road_depth_p95_m",
    "road_depth_std_m",
    "road_area_occlusion_quality",
    "road_area_feature_version",
    "area_quality_flag",
    "final_road_area_feature_status_v3",
]


def _lens_pivot(frame: pd.DataFrame, features: list[str], prefix: str) -> pd.DataFrame:
    if frame.duplicated(["sample_id", "lens_id"]).any():
        raise ValueError(f"Duplicate sample/lens rows in {prefix} features")
    available = [column for column in features if column in frame.columns]
    numeric = frame[["sample_id", "lens_id", *available]].copy()
    for column in available:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    wide = numeric.set_index(["sample_id", "lens_id"])[available].unstack("lens_id")
    wide.columns = [f"{prefix}_lens{int(lens)}_{feature}" for feature, lens in wide.columns]
    return wide.reset_index()


def aggregate_vehicle_features(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["idd_detection_status"].eq("success")].copy()
    wide = _lens_pivot(valid, VEHICLE_FEATURES, "vehicle")
    lens_count = valid.groupby("sample_id")["lens_id"].nunique().rename("vehicle_valid_lens_count")
    wide = wide.merge(lens_count, on="sample_id", how="left", validate="one_to_one")
    for feature in VEHICLE_FEATURES:
        columns = [column for column in wide if column.endswith(f"_{feature}")]
        if columns:
            # Overlapping cameras can observe the same object. Max is an
            # explicit conservative proxy, not a claim of a deduplicated count.
            wide[f"vehicle_cross_view_max_{feature}"] = wide[columns].max(
                axis=1, skipna=True
            )
    wide["vehicle_features_available"] = wide["vehicle_valid_lens_count"].gt(0)
    return wide


def aggregate_road_features(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in frame.columns:
        lookup = manifest[["processed_frame_key", "sample_id"]].drop_duplicates()
        frame = frame.merge(
            lookup,
            on="processed_frame_key",
            how="left",
            validate="many_to_one",
        )
    if frame["sample_id"].isna().any():
        raise ValueError("Some road rows could not be mapped to sample_id")
    all_samples = frame[["sample_id"]].drop_duplicates()
    expected_lenses = sorted(frame["lens_id"].dropna().astype(int).unique())
    valid = frame[frame["road_condition_status"].eq("success")].copy()
    wide = _lens_pivot(valid, ROAD_FEATURES, "road")
    lens_count = valid.groupby("sample_id")["lens_id"].nunique().rename("road_valid_lens_count")
    wide = all_samples.merge(wide, on="sample_id", how="left", validate="one_to_one")
    wide = wide.merge(lens_count, on="sample_id", how="left", validate="one_to_one")
    wide["road_valid_lens_count"] = wide["road_valid_lens_count"].fillna(0).astype(int)
    for feature in ROAD_FEATURES:
        for lens in expected_lenses:
            column = f"road_lens{lens}_{feature}"
            if column not in wide.columns:
                wide[column] = float("nan")
    for feature in ROAD_FEATURES:
        columns = [column for column in wide if column.endswith(f"_{feature}")]
        if columns:
            wide[f"road_cross_view_mean_{feature}"] = wide[columns].mean(
                axis=1, skipna=True
            )
    wide["road_features_available"] = wide["road_valid_lens_count"].gt(0)
    return wide


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csv", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--vehicle-csv", required=True)
    parser.add_argument("--road-csv", required=True)
    parser.add_argument("--road-area-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    sensor = pd.read_csv(args.sensor_csv)
    manifest = pd.read_csv(args.frame_manifest)
    sample_ids = manifest["sample_id"].drop_duplicates()
    sensor = sensor[sensor["sample_id"].isin(sample_ids)].copy()
    if sensor.empty:
        raise ValueError("No sensor rows match the frame manifest")
    if sensor["sample_id"].duplicated().any():
        raise ValueError("Sensor sample_id must be unique")
    if "sample_index" not in sensor.columns:
        sample_index = manifest[["sample_id", "sample_index"]].drop_duplicates()
        if sample_index["sample_id"].duplicated().any():
            raise ValueError("Manifest maps one sample_id to multiple sample_index values")
        sensor = sensor.merge(
            sample_index, on="sample_id", how="left", validate="one_to_one"
        )

    vehicle = aggregate_vehicle_features(pd.read_csv(args.vehicle_csv))
    road = aggregate_road_features(pd.read_csv(args.road_csv), manifest)
    area_raw = pd.read_csv(args.road_area_csv)
    area_columns = [
        "sample_id",
        *[column for column in ROAD_AREA_FEATURES if column in area_raw.columns],
    ]
    area = area_raw[area_columns].drop_duplicates("sample_id")
    area["road_area_usable"] = area.get(
        "final_road_area_feature_status_v3", pd.Series(index=area.index, dtype="object")
    ).eq("use_primary")

    result = sensor.merge(vehicle, on="sample_id", how="left", validate="one_to_one")
    result = result.merge(road, on="sample_id", how="left", validate="one_to_one")
    result = result.merge(area, on="sample_id", how="left", validate="one_to_one")
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = {
        "output_csv": str(output),
        "rows": int(len(result)),
        "columns": int(len(result.columns)),
        "unique_sample_ids": int(result["sample_id"].nunique()),
        "vehicle_fusion_policy": "retain_per_lens_and_use_cross_view_max_not_sum",
        "road_fusion_policy": "mean_available_road_lenses_preserve_missing_as_nan",
        "road_area_policy": "lens1_historical_intrinsics_validation_required",
        "vehicle_lens_coverage": {
            str(key): int(value)
            for key, value in result["vehicle_valid_lens_count"].value_counts().items()
        },
        "road_lens_coverage": {
            str(key): int(value)
            for key, value in result["road_valid_lens_count"].value_counts().items()
        },
        "usable_road_area": {
            str(key): int(value)
            for key, value in result["road_area_usable"].value_counts(dropna=False).items()
        },
        "inputs": {
            "sensor_csv": args.sensor_csv,
            "frame_manifest": args.frame_manifest,
            "vehicle_csv": args.vehicle_csv,
            "road_csv": args.road_csv,
            "road_area_csv": args.road_area_csv,
        },
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved canonical pilot feature table: {output}")
    print(f"Saved assembly summary: {summary_path}")
    print(f"Rows: {len(result)}; columns: {len(result.columns)}")
    print("Vehicle lens coverage:", result["vehicle_valid_lens_count"].value_counts().to_dict())
    print("Road lens coverage:", result["road_valid_lens_count"].value_counts().to_dict())
    print("Usable road area:", result["road_area_usable"].value_counts(dropna=False).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
