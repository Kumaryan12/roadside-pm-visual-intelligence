from pathlib import Path
import argparse
import pandas as pd


DEFAULT_SENSOR_CSV = "data/sensor/MC1S_best_3min_window.csv"
DEFAULT_FEATURES_CSV = "outputs/pipeline_1s/features/sensor_level_1s_aggregated_features_best3min.csv"
DEFAULT_OUTPUT_CSV = "outputs/pipeline_1s/modeling/modeling_table_1s_features_best3min.csv"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sensor-csv", default=DEFAULT_SENSOR_CSV)
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)

    args = parser.parse_args()

    sensor_path = Path(args.sensor_csv)
    features_path = Path(args.features_csv)
    output_path = Path(args.output_csv)

    if not sensor_path.exists():
        raise FileNotFoundError(f"Sensor CSV not found: {sensor_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_path}")

    sensor = pd.read_csv(sensor_path)
    features = pd.read_csv(features_path)

    print("\nSensor shape:", sensor.shape)
    print("Features shape:", features.shape)

    if "sensor_row_id" not in sensor.columns:
        sensor = sensor.copy()
        sensor["sensor_row_id"] = range(len(sensor))

    required_feature_cols = ["sensor_row_id"]
    missing = [c for c in required_feature_cols if c not in features.columns]
    if missing:
        raise ValueError(f"Features CSV missing columns: {missing}")

    merged = sensor.merge(
        features,
        on="sensor_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print("\nSaved:", output_path)
    print("Merged shape:", merged.shape)

    print("\nFeature availability:")
    if "image_1s_features_available" in merged.columns:
        print(merged["image_1s_features_available"].value_counts(dropna=False))

    print("\nMissing visual features rows:")
    feature_cols = [c for c in features.columns if c not in ["sensor_row_id"]]
    missing_rows = merged[feature_cols].isna().all(axis=1).sum()
    print(missing_rows, "/", len(merged))

    print("\nHead:")
    preview_cols = [
        "sensor_row_id",
        "timestamp",
        "value.sPM2",
        "value.sPM10",
        "nPM2",
        "veh1s_all_lenses_idd_total_vehicle_count_mean",
        "veh1s_all_lenses_idd_exhaust_proxy_initial_sum",
        "veh1s_all_lenses_idd_resuspension_vehicle_proxy_initial_sum",
        "road1s_lens1_road_area_m2_occlusion_adjusted_conservative_v3_mean",
        "road1s_lens1_road_area_vehicle_occlusion_fraction_v3_mean",
    ]
    preview_cols = [c for c in preview_cols if c in merged.columns]
    print(merged[preview_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()