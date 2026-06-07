from pathlib import Path
import argparse
import numpy as np
import pandas as pd


DEFAULT_VEHICLE_CSV = Path("outputs/features/idd_vehicle_features_sensor_level_v2.csv")

# Default kept as original/pretrained road file.
# For fine-tuned model, pass --road-csv manually in the command.
DEFAULT_ROAD_CSV = Path("outputs/features/segformer_road_condition_features_sensor_level_v2.csv")

DEFAULT_OUTPUT_CSV = Path("outputs/features/particle_density_modeling_table_v2.csv")


def to_bool_series(series):
    """
    Robust conversion for boolean-like columns that may appear as:
    True/False, 1/0, 'True'/'False', 'true'/'false', yes/no.
    """
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def make_numeric(df, col, fill_value=None):
    """
    Convert a column to numeric if it exists.
    Optionally fill NaN values.
    """
    if col not in df.columns:
        return

    df[col] = pd.to_numeric(df[col], errors="coerce")

    if fill_value is not None:
        df[col] = df[col].fillna(fill_value)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vehicle-csv", default=str(DEFAULT_VEHICLE_CSV))
    parser.add_argument("--road-csv", default=str(DEFAULT_ROAD_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))

    args = parser.parse_args()

    vehicle_csv = Path(args.vehicle_csv)
    road_csv = Path(args.road_csv)
    output_csv = Path(args.output_csv)

    if not vehicle_csv.exists():
        raise FileNotFoundError(f"Vehicle CSV not found: {vehicle_csv}")

    if not road_csv.exists():
        raise FileNotFoundError(f"Road CSV not found: {road_csv}")

    vehicle = pd.read_csv(vehicle_csv)
    road = pd.read_csv(road_csv)

    print("Vehicle shape:", vehicle.shape)
    print("Road shape:", road.shape)

    if "sample_index" not in vehicle.columns:
        raise ValueError("Vehicle CSV missing sample_index")

    if "sample_index" not in road.columns:
        raise ValueError("Road CSV missing sample_index")

    vehicle["sample_index"] = pd.to_numeric(
        vehicle["sample_index"],
        errors="coerce",
    ).astype("Int64")

    road["sample_index"] = pd.to_numeric(
        road["sample_index"],
        errors="coerce",
    ).astype("Int64")

    # Keep only road-related columns from road CSV.
    # Vehicle CSV already contains original sensor columns.
    road_feature_cols = [
        c for c in road.columns
        if c == "sample_index" or c.startswith("road_")
    ]

    road_small = road[road_feature_cols].copy()

    fused = vehicle.merge(
        road_small,
        on="sample_index",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------
    # Availability flags
    # ------------------------------------------------------------

    if "vehicle_all_3_lenses_available" in fused.columns:
        vehicle_available = to_bool_series(fused["vehicle_all_3_lenses_available"])
    elif "vehicle_features_available" in fused.columns:
        vehicle_available = to_bool_series(fused["vehicle_features_available"])
    else:
        vehicle_available = pd.Series(False, index=fused.index)

    if "road_all_2_lenses_available" in fused.columns:
        road_available = to_bool_series(fused["road_all_2_lenses_available"])
    elif "road_features_available" in fused.columns:
        road_available = to_bool_series(fused["road_features_available"])
    else:
        road_available = pd.Series(False, index=fused.index)

    fused["image_features_available"] = vehicle_available & road_available

    # ------------------------------------------------------------
    # Vehicle count cleanup + total vehicle count
    # ------------------------------------------------------------

    count_cols = [
        "idd_auto_rickshaw_count_sum",
        "idd_bicycle_count_sum",
        "idd_bus_count_sum",
        "idd_car_count_sum",
        "idd_motorcycle_count_sum",
        "idd_truck_count_sum",
        "idd_unknown_vehicle_count_sum",
    ]

    existing_count_cols = [c for c in count_cols if c in fused.columns]

    for col in existing_count_cols:
        make_numeric(fused, col, fill_value=0)

    if existing_count_cols:
        fused["idd_total_vehicle_count_sum"] = fused[existing_count_cols].sum(axis=1)

    # ------------------------------------------------------------
    # Numeric cleanup for important vehicle proxy columns
    # ------------------------------------------------------------

    vehicle_numeric_cols = [
        "idd_heavy_vehicle_count_sum",
        "idd_motor_vehicle_count_sum",
        "idd_exhaust_proxy_initial_sum",
        "idd_resuspension_vehicle_proxy_initial_sum",
        "idd_vehicle_box_area_ratio_mean",
        "idd_average_confidence_mean",
        "idd_max_confidence_max",
    ]

    for col in vehicle_numeric_cols:
        make_numeric(fused, col)

    # ------------------------------------------------------------
    # Environmental helper features
    # ------------------------------------------------------------

    if "rh" in fused.columns:
        make_numeric(fused, "rh")
        fused["rh_fraction"] = fused["rh"] / 100.0
        fused["dry_air_fraction"] = 1.0 - fused["rh_fraction"]

    if "temp" in fused.columns:
        make_numeric(fused, "temp")

    # ------------------------------------------------------------
    # PM target helper features
    # These are derived from sensor values.
    # Use them for analysis, not as predictors while training PM models.
    # ------------------------------------------------------------

    if "value.sPM1" in fused.columns:
        make_numeric(fused, "value.sPM1")

    if "value.sPM2" in fused.columns:
        make_numeric(fused, "value.sPM2")

    if "value.sPM4" in fused.columns:
        make_numeric(fused, "value.sPM4")

    if "value.sPM10" in fused.columns:
        make_numeric(fused, "value.sPM10")

    if "value.sPM10" in fused.columns and "value.sPM2" in fused.columns:
        fused["pm10_minus_pm2"] = fused["value.sPM10"] - fused["value.sPM2"]
        fused["pm2_fraction_of_pm10"] = np.where(
            fused["value.sPM10"] > 0,
            fused["value.sPM2"] / fused["value.sPM10"],
            np.nan,
        )

    if "value.sPM4" in fused.columns and "value.sPM2" in fused.columns:
        fused["pm4_minus_pm2"] = fused["value.sPM4"] - fused["value.sPM2"]

    # ------------------------------------------------------------
    # Road numeric cleanup
    # ------------------------------------------------------------

    road_numeric_cols = [
        c for c in fused.columns
        if c.startswith("road_")
        and not c.endswith("_available")
    ]

    for col in road_numeric_cols:
        make_numeric(fused, col)

    # ------------------------------------------------------------
    # Interaction features
    # These are candidate predictors, not final physical dust scores.
    # ------------------------------------------------------------

    resusp_col = "idd_resuspension_vehicle_proxy_initial_sum"
    exhaust_col = "idd_exhaust_proxy_initial_sum"

    # Keep old interaction names for compatibility with your earlier scripts.
    if resusp_col in fused.columns and "road_brown_pixel_ratio_mean" in fused.columns:
        fused["vehicle_resuspension_x_road_brown_mean"] = (
            fused[resusp_col] * fused["road_brown_pixel_ratio_mean"]
        )

    if resusp_col in fused.columns and "road_gray_dry_pixel_ratio_mean" in fused.columns:
        fused["vehicle_resuspension_x_road_gray_dry_mean"] = (
            fused[resusp_col] * fused["road_gray_dry_pixel_ratio_mean"]
        )

    if resusp_col in fused.columns and "dry_air_fraction" in fused.columns:
        fused["vehicle_resuspension_x_dry_air"] = (
            fused[resusp_col] * fused["dry_air_fraction"]
        )

    if exhaust_col in fused.columns and "dry_air_fraction" in fused.columns:
        fused["vehicle_exhaust_x_dry_air"] = (
            fused[exhaust_col] * fused["dry_air_fraction"]
        )

    # Additional road × vehicle interactions.
    interaction_road_cols = [
        "road_area_ratio_mean",
        "road_brown_pixel_ratio_mean",
        "road_gray_dry_pixel_ratio_mean",
        "road_edge_density_mean",
        "road_laplacian_std_mean",
        "road_contrast_std_mean",
        "road_haze_flatness_proxy_mean",
    ]

    for road_col in interaction_road_cols:
        if resusp_col in fused.columns and road_col in fused.columns:
            clean_name = (
                road_col
                .replace("road_", "")
                .replace("_mean", "")
            )
            fused[f"vehicle_resuspension_x_{clean_name}"] = (
                fused[resusp_col] * fused[road_col]
            )

    # Road-only composite descriptors.
    if (
        "road_brown_pixel_ratio_mean" in fused.columns
        and "road_edge_density_mean" in fused.columns
    ):
        fused["road_brown_x_edge_density"] = (
            fused["road_brown_pixel_ratio_mean"] * fused["road_edge_density_mean"]
        )

    if (
        "road_gray_dry_pixel_ratio_mean" in fused.columns
        and "road_laplacian_std_mean" in fused.columns
    ):
        fused["road_gray_dry_x_texture"] = (
            fused["road_gray_dry_pixel_ratio_mean"] * fused["road_laplacian_std_mean"]
        )

    if (
        "road_brown_pixel_ratio_mean" in fused.columns
        and "road_laplacian_std_mean" in fused.columns
    ):
        fused["road_brown_x_texture"] = (
            fused["road_brown_pixel_ratio_mean"] * fused["road_laplacian_std_mean"]
        )

    # ------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(output_csv, index=False)

    print("\nSaved:", output_csv)
    print("Output shape:", fused.shape)

    print("\nImage feature availability:")
    print(fused["image_features_available"].value_counts(dropna=False))

    if "vehicle_all_3_lenses_available" in fused.columns:
        print("\nVehicle all 3 lenses available:")
        print(
            to_bool_series(
                fused["vehicle_all_3_lenses_available"]
            ).value_counts(dropna=False)
        )

    if "road_all_2_lenses_available" in fused.columns:
        print("\nRoad all 2 lenses available:")
        print(
            to_bool_series(
                fused["road_all_2_lenses_available"]
            ).value_counts(dropna=False)
        )

    show_cols = [
        "sample_index",
        "timestamp",
        "image_features_available",
        "value.sPM2",
        "value.sPM10",
        "pm10_minus_pm2",
        "idd_total_vehicle_count_sum",
        "idd_auto_rickshaw_count_sum",
        "idd_car_count_sum",
        "idd_motorcycle_count_sum",
        "idd_truck_count_sum",
        "idd_exhaust_proxy_initial_sum",
        "idd_resuspension_vehicle_proxy_initial_sum",
        "road_area_ratio_mean",
        "road_brown_pixel_ratio_mean",
        "road_gray_dry_pixel_ratio_mean",
        "road_edge_density_mean",
        "vehicle_resuspension_x_road_brown_mean",
        "vehicle_resuspension_x_road_gray_dry_mean",
        "vehicle_resuspension_x_dry_air",
    ]

    show_cols = [c for c in show_cols if c in fused.columns]

    print("\nPreview:")
    print(fused[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()