from pathlib import Path
import argparse
import pandas as pd
import numpy as np


DEFAULT_INPUT_TABLE = Path("outputs/features/particle_density_modeling_table_idd_finetuned_v2.csv")

DEFAULT_OSM_CSV = Path(
    "/Users/aryansatyendrakumar/Projects/pm_density_image_pipeline/"
    "outputs/features/Sensor_data_with_osm.csv"
)

DEFAULT_OUTPUT_CSV = Path(
    "outputs/features/particle_density_modeling_table_idd_finetuned_osm_v2.csv"
)


OSM_FEATURE_COLS = [
    "osm_location_key",
    "osm_latitude_rounded",
    "osm_longitude_rounded",

    "fuel_station_count_250m",
    "restaurant_count_250m",
    "bus_stop_count_250m",
    "parking_count_250m",
    "construction_count_250m",
    "industrial_count_250m",
    "factory_count_250m",
    "warehouse_count_250m",
    "marketplace_count_250m",
    "park_count_250m",
    "school_college_count_250m",
    "hospital_count_250m",
    "commercial_count_250m",
    "retail_count_250m",

    "road_segment_count_250m",
    "total_road_length_250m",
    "motorway_count_250m",
    "trunk_road_count_250m",
    "primary_road_count_250m",
    "secondary_road_count_250m",
    "tertiary_road_count_250m",
    "residential_road_count_250m",
    "service_road_count_250m",
    "living_street_count_250m",
    "unclassified_road_count_250m",

    "osm_status",
    "osm_error",
]


def normalize_timestamp(series):
    return pd.to_datetime(series, errors="coerce").dt.floor("s")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-table", default=str(DEFAULT_INPUT_TABLE))
    parser.add_argument("--osm-csv", default=str(DEFAULT_OSM_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))

    args = parser.parse_args()

    input_table = Path(args.input_table)
    osm_csv = Path(args.osm_csv)
    output_csv = Path(args.output_csv)

    if not input_table.exists():
        raise FileNotFoundError(f"Input modeling table not found: {input_table}")

    if not osm_csv.exists():
        raise FileNotFoundError(f"OSM CSV not found: {osm_csv}")

    main_df = pd.read_csv(input_table)
    osm_df = pd.read_csv(osm_csv)

    print("Main table shape:", main_df.shape)
    print("OSM table shape:", osm_df.shape)

    if "sample_index" not in main_df.columns:
        raise ValueError("Main table missing sample_index")

    if "sample_id" not in osm_df.columns:
        raise ValueError("OSM CSV missing sample_id")

    main_df["sample_index"] = pd.to_numeric(
        main_df["sample_index"],
        errors="coerce",
    ).astype("Int64")

    osm_df["sample_id"] = pd.to_numeric(
        osm_df["sample_id"],
        errors="coerce",
    ).astype("Int64")

    # Keep only clean OSM columns from old project.
    available_osm_cols = ["sample_id"] + [
        c for c in OSM_FEATURE_COLS
        if c in osm_df.columns
    ]

    osm_small = osm_df[available_osm_cols].copy()

    # Rename old sample_id to new sample_index.
    osm_small = osm_small.rename(columns={"sample_id": "sample_index"})

    # Drop duplicate sample IDs if any.
    duplicate_count = osm_small["sample_index"].duplicated().sum()
    if duplicate_count:
        print("Warning: duplicate sample_index rows in OSM:", duplicate_count)
        osm_small = osm_small.drop_duplicates("sample_index", keep="first")

    fused = main_df.merge(
        osm_small,
        on="sample_index",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------
    # OSM availability flag
    # ------------------------------------------------------------
    if "osm_status" in fused.columns:
        fused["osm_features_available"] = (
            fused["osm_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("success")
        )
    else:
        osm_count_cols = [
            c for c in fused.columns
            if c.endswith("_250m") or c == "total_road_length_250m"
        ]
        fused["osm_features_available"] = fused[osm_count_cols].notna().any(axis=1)

    # ------------------------------------------------------------
    # Numeric cleanup
    # ------------------------------------------------------------
    osm_numeric_cols = [
        c for c in fused.columns
        if (
            c.endswith("_count_250m")
            or c == "total_road_length_250m"
            or c in ["osm_latitude_rounded", "osm_longitude_rounded"]
        )
    ]

    for col in osm_numeric_cols:
        fused[col] = pd.to_numeric(fused[col], errors="coerce")

    # Fill missing OSM count features with 0 only if OSM status failed/missing.
    # For successful OSM rows, 0 is a real count.
    count_cols = [c for c in osm_numeric_cols if c.endswith("_count_250m")]
    for col in count_cols:
        fused[col] = fused[col].fillna(0)

    if "total_road_length_250m" in fused.columns:
        fused["total_road_length_250m"] = fused["total_road_length_250m"].fillna(0)

    # ------------------------------------------------------------
    # Derived OSM context features
    # ------------------------------------------------------------
    if "commercial_count_250m" in fused.columns and "retail_count_250m" in fused.columns:
        fused["osm_commercial_retail_intensity_250m"] = (
            fused["commercial_count_250m"] + fused["retail_count_250m"]
        )

    industrial_cols = [
        "industrial_count_250m",
        "factory_count_250m",
        "warehouse_count_250m",
        "construction_count_250m",
    ]
    existing_industrial_cols = [c for c in industrial_cols if c in fused.columns]
    if existing_industrial_cols:
        fused["osm_industrial_construction_intensity_250m"] = (
            fused[existing_industrial_cols].sum(axis=1)
        )

    activity_cols = [
        "restaurant_count_250m",
        "marketplace_count_250m",
        "commercial_count_250m",
        "retail_count_250m",
        "parking_count_250m",
        "bus_stop_count_250m",
        "school_college_count_250m",
        "hospital_count_250m",
    ]
    existing_activity_cols = [c for c in activity_cols if c in fused.columns]
    if existing_activity_cols:
        fused["osm_human_activity_intensity_250m"] = (
            fused[existing_activity_cols].sum(axis=1)
        )

    major_road_cols = [
        "motorway_count_250m",
        "trunk_road_count_250m",
        "primary_road_count_250m",
        "secondary_road_count_250m",
    ]
    existing_major_road_cols = [c for c in major_road_cols if c in fused.columns]
    if existing_major_road_cols:
        fused["osm_major_road_intensity_250m"] = (
            fused[existing_major_road_cols].sum(axis=1)
        )

    minor_road_cols = [
        "tertiary_road_count_250m",
        "residential_road_count_250m",
        "service_road_count_250m",
        "living_street_count_250m",
        "unclassified_road_count_250m",
    ]
    existing_minor_road_cols = [c for c in minor_road_cols if c in fused.columns]
    if existing_minor_road_cols:
        fused["osm_minor_road_intensity_250m"] = (
            fused[existing_minor_road_cols].sum(axis=1)
        )

    if (
        "osm_major_road_intensity_250m" in fused.columns
        and "road_segment_count_250m" in fused.columns
    ):
        fused["osm_major_road_fraction_250m"] = np.where(
            fused["road_segment_count_250m"] > 0,
            fused["osm_major_road_intensity_250m"] / fused["road_segment_count_250m"],
            np.nan,
        )

    if (
        "total_road_length_250m" in fused.columns
        and "road_segment_count_250m" in fused.columns
    ):
        fused["osm_avg_road_segment_length_250m"] = np.where(
            fused["road_segment_count_250m"] > 0,
            fused["total_road_length_250m"] / fused["road_segment_count_250m"],
            np.nan,
        )

    # Vehicle-road-network interaction
    if (
        "idd_total_vehicle_count_sum" in fused.columns
        and "total_road_length_250m" in fused.columns
    ):
        fused["vehicle_count_x_osm_road_length"] = (
            pd.to_numeric(fused["idd_total_vehicle_count_sum"], errors="coerce")
            * fused["total_road_length_250m"]
        )

    if (
        "idd_resuspension_vehicle_proxy_initial_sum" in fused.columns
        and "osm_major_road_intensity_250m" in fused.columns
    ):
        fused["resuspension_x_osm_major_road_intensity"] = (
            pd.to_numeric(
                fused["idd_resuspension_vehicle_proxy_initial_sum"],
                errors="coerce",
            )
            * fused["osm_major_road_intensity_250m"]
        )

    if (
        "idd_exhaust_proxy_initial_sum" in fused.columns
        and "osm_human_activity_intensity_250m" in fused.columns
    ):
        fused["exhaust_x_osm_human_activity_intensity"] = (
            pd.to_numeric(fused["idd_exhaust_proxy_initial_sum"], errors="coerce")
            * fused["osm_human_activity_intensity_250m"]
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(output_csv, index=False)

    print("\nSaved:", output_csv)
    print("Output shape:", fused.shape)

    print("\nOSM availability:")
    print(fused["osm_features_available"].value_counts(dropna=False))

    print("\nOSM status:")
    if "osm_status" in fused.columns:
        print(fused["osm_status"].value_counts(dropna=False))

    show_cols = [
        "sample_index",
        "timestamp",
        "image_features_available",
        "osm_features_available",
        "value.sPM2",
        "value.sPM10",
        "idd_total_vehicle_count_sum",
        "idd_resuspension_vehicle_proxy_initial_sum",
        "road_brown_pixel_ratio_mean",
        "road_gray_dry_pixel_ratio_mean",
        "bus_stop_count_250m",
        "parking_count_250m",
        "commercial_count_250m",
        "road_segment_count_250m",
        "total_road_length_250m",
        "primary_road_count_250m",
        "secondary_road_count_250m",
        "osm_human_activity_intensity_250m",
        "osm_major_road_intensity_250m",
        "resuspension_x_osm_major_road_intensity",
    ]

    show_cols = [c for c in show_cols if c in fused.columns]

    print("\nPreview:")
    print(fused[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()