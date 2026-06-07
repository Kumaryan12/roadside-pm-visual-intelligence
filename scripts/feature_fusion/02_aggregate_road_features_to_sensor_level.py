from pathlib import Path
import argparse
import pandas as pd


DEFAULT_SENSOR_CSV = Path("data/sensor/MC1S_window_115430_124150.csv")
DEFAULT_ROAD_CSV = Path("outputs/features/segformer_road_condition_features_frame_level_v2.csv")
DEFAULT_OUTPUT_CSV = Path("outputs/features/segformer_road_condition_features_sensor_level_v2.csv")


ROAD_FEATURE_COLS = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-csv", default=str(DEFAULT_SENSOR_CSV))
    parser.add_argument("--road-csv", default=str(DEFAULT_ROAD_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()

    sensor = pd.read_csv(args.sensor_csv)
    road = pd.read_csv(args.road_csv)

    road = road[road["road_condition_status"] == "success"].copy()

    for col in ROAD_FEATURE_COLS:
        road[col] = pd.to_numeric(road[col], errors="coerce")

    agg_spec = {}

    for col in ROAD_FEATURE_COLS:
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_max"] = (col, "max")
        agg_spec[f"{col}_min"] = (col, "min")
        agg_spec[f"{col}_std"] = (col, "std")

    agg_spec["road_lens_count"] = ("lens_id", "nunique")
    agg_spec["road_frame_count"] = ("processed_frame_key", "count")

    road_agg = (
        road
        .groupby("sample_index")
        .agg(**agg_spec)
        .reset_index()
    )

    sensor = sensor.copy()
    sensor["sample_index"] = range(len(sensor))

    merged = sensor.merge(
        road_agg,
        on="sample_index",
        how="left",
        validate="one_to_one",
    )

    merged["road_features_available"] = merged["road_frame_count"].fillna(0) > 0
    merged["road_all_2_lenses_available"] = merged["road_lens_count"].fillna(0) == 2

    for col in merged.columns:
        if col.startswith("road_") and merged[col].dtype != "bool":
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Shape:", merged.shape)

    print("\nRoad features available:")
    print(merged["road_features_available"].value_counts(dropna=False))

    print("\nAll 2 lenses available:")
    print(merged["road_all_2_lenses_available"].value_counts(dropna=False))

    show_cols = [
        "sample_index",
        "timestamp",
        "road_all_2_lenses_available",
        "road_area_ratio_mean",
        "road_brown_pixel_ratio_mean",
        "road_gray_dry_pixel_ratio_mean",
        "road_edge_density_mean",
        "road_haze_flatness_proxy_mean",
    ]

    print("\nPreview:")
    print(merged[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()