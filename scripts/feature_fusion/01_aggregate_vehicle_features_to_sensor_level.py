from pathlib import Path
import argparse
import pandas as pd


DEFAULT_SENSOR_CSV = Path("data/sensor/MC1S_window_115430_124150.csv")
DEFAULT_DETECTION_CSV = Path("outputs/features/idd_vehicle_detections_frame_level_v2.csv")
DEFAULT_FRAME_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")
DEFAULT_OUTPUT_CSV = Path("outputs/features/idd_vehicle_features_sensor_level_v2.csv")

COUNT_COLS = [
    "idd_auto_rickshaw_count",
    "idd_bicycle_count",
    "idd_bus_count",
    "idd_car_count",
    "idd_motorcycle_count",
    "idd_truck_count",
    "idd_unknown_vehicle_count",
]

SUM_COLS = COUNT_COLS + [
    "idd_heavy_vehicle_count",
    "idd_motor_vehicle_count",
    "idd_exhaust_proxy_initial",
    "idd_resuspension_vehicle_proxy_initial",
]

MEAN_MAX_COLS = [
    "idd_vehicle_box_area_ratio",
    "idd_average_confidence",
    "idd_max_confidence",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sensor-csv", default=str(DEFAULT_SENSOR_CSV))
    parser.add_argument("--detection-csv", default=str(DEFAULT_DETECTION_CSV))
    parser.add_argument("--frame-manifest", default=str(DEFAULT_FRAME_MANIFEST))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))

    args = parser.parse_args()

    sensor_csv = Path(args.sensor_csv)
    detection_csv = Path(args.detection_csv)
    frame_manifest = Path(args.frame_manifest)
    output_csv = Path(args.output_csv)

    if not sensor_csv.exists():
        raise FileNotFoundError(f"Sensor CSV not found: {sensor_csv}")

    if not detection_csv.exists():
        raise FileNotFoundError(f"Detection CSV not found: {detection_csv}")

    if not frame_manifest.exists():
        raise FileNotFoundError(f"Frame manifest not found: {frame_manifest}")

    sensor_df = pd.read_csv(sensor_csv)
    det_df = pd.read_csv(detection_csv)
    manifest_df = pd.read_csv(frame_manifest)
    manifest_df = pd.read_csv(frame_manifest)
    print("Sensor shape:", sensor_df.shape)
    print("Detection shape:", det_df.shape)
    print("Frame manifest shape:", manifest_df.shape)

    if "sample_index" not in det_df.columns:

        print("\nDetection CSV does not contain sample_index.")
        print("Recovering sample_index by merging with frame manifest...")

        needed_manifest_cols = [
            "processed_frame_key",
            "sample_index",
            "sensor_timestamp",
            "sample_unix",
        ]

        available_manifest_cols = [
            c for c in needed_manifest_cols
            if c in manifest_df.columns
        ]

        if "processed_frame_key" not in available_manifest_cols:
            raise ValueError("Frame manifest must contain processed_frame_key.")

        if "sample_index" not in available_manifest_cols:
            raise ValueError("Frame manifest must contain sample_index.")

        det_df = det_df.merge(
            manifest_df[available_manifest_cols],
            on="processed_frame_key",
            how="left",
            validate="many_to_one",
        )

        missing_sample = det_df["sample_index"].isna().sum()
        print("Missing sample_index after merge:", missing_sample)

        if missing_sample > 0:
            raise ValueError("Some detection rows could not be matched to frame manifest.")

    det_df = det_df[det_df["idd_detection_status"] == "success"].copy()

    for col in SUM_COLS + MEAN_MAX_COLS:
        if col not in det_df.columns:
            print(f"Missing column, filling with 0: {col}")
            det_df[col] = 0
        det_df[col] = pd.to_numeric(det_df[col], errors="coerce").fillna(0)

    agg_spec = {}

    for col in SUM_COLS:
        agg_spec[f"{col}_sum"] = (col, "sum")
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_max"] = (col, "max")

    for col in MEAN_MAX_COLS:
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_max"] = (col, "max")

    agg_spec["vehicle_lens_count"] = ("lens_id", "nunique")
    agg_spec["vehicle_frame_count"] = ("processed_frame_key", "count")
    agg_spec["matched_run_id"] = ("matched_run_id", lambda x: ",".join(sorted(set(map(str, x)))))
    agg_spec["vehicle_min_offset_sec"] = ("video_offset_sec", "min")
    agg_spec["vehicle_max_offset_sec"] = ("video_offset_sec", "max")

    vehicle_agg = (
        det_df
        .groupby("sample_index")
        .agg(**agg_spec)
        .reset_index()
    )

    sensor_df = sensor_df.copy()
    sensor_df["sample_index"] = range(len(sensor_df))

    merged = sensor_df.merge(
        vehicle_agg,
        on="sample_index",
        how="left",
        validate="one_to_one",
    )

    merged["vehicle_features_available"] = merged["vehicle_frame_count"].fillna(0) > 0
    merged["vehicle_all_3_lenses_available"] = merged["vehicle_lens_count"].fillna(0) == 3

    # Fill missing vehicle numeric features with 0.
    for col in merged.columns:
        if (
            col.startswith("idd_")
            or col.startswith("vehicle_")
        ):
            if merged[col].dtype != "bool":
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    if "matched_run_id" in merged.columns:
        merged["matched_run_id"] = merged["matched_run_id"].fillna("")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("\nSaved:", output_csv)
    print("Output shape:", merged.shape)

    print("\nVehicle features available:")
    print(merged["vehicle_features_available"].value_counts(dropna=False))

    print("\nAll 3 lenses available:")
    print(merged["vehicle_all_3_lenses_available"].value_counts(dropna=False))

    print("\nAggregated vehicle count totals:")
    for col in [
        "idd_auto_rickshaw_count_sum",
        "idd_bicycle_count_sum",
        "idd_bus_count_sum",
        "idd_car_count_sum",
        "idd_motorcycle_count_sum",
        "idd_truck_count_sum",
        "idd_unknown_vehicle_count_sum",
    ]:
        if col in merged.columns:
            print(col, int(merged[col].sum()))

    show_cols = [
        "sample_index",
        "timestamp",
        "vehicle_all_3_lenses_available",
        "idd_auto_rickshaw_count_sum",
        "idd_car_count_sum",
        "idd_motorcycle_count_sum",
        "idd_bus_count_sum",
        "idd_truck_count_sum",
        "idd_exhaust_proxy_initial_sum",
        "idd_resuspension_vehicle_proxy_initial_sum",
    ]
    show_cols = [c for c in show_cols if c in merged.columns]

    print("\nPreview:")
    print(merged[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()