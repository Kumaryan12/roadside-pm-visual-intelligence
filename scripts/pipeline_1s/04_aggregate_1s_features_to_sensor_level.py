from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_VEHICLE_FRAME_CSV = (
    "outputs/pipeline_1s/vehicle_detections/"
    "idd_vehicle_detections_frame_level_1s_v1_best3min.csv"
)

DEFAULT_ROAD_AREA_V3_CSV = (
    "outputs/pipeline_1s/road_area_lens1_best3min_v1/"
    "lens1_depth_gated_occlusion_adjusted_road_area_v3_1s_best3min.csv"
)

DEFAULT_OUTPUT_CSV = (
    "outputs/pipeline_1s/features/"
    "sensor_level_1s_aggregated_features_best3min.csv"
)


# ============================================================
# HELPERS
# ============================================================

KEY_COLS = [
    "sensor_row_id",
    "sample_index",
    "sensor_timestamp",
    "sensor_unix",
]


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def flatten_columns(df):
    df.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        if isinstance(col, tuple)
        else str(col)
        for col in df.columns
    ]
    return df


def add_availability_summary(df, prefix):
    """
    One row per sensor_row_id summarizing frame availability.
    """
    out = (
        df.groupby(KEY_COLS, dropna=False)
        .agg(
            **{
                f"{prefix}_frame_count": ("processed_frame_key", "nunique"),
                f"{prefix}_relative_second_count": ("relative_time_sec", "nunique"),
                f"{prefix}_min_relative_time_sec": ("relative_time_sec", "min"),
                f"{prefix}_max_relative_time_sec": ("relative_time_sec", "max"),
            }
        )
        .reset_index()
    )
    return out


def aggregate_numeric(df, group_cols, numeric_cols, prefix, aggs):
    if not numeric_cols:
        return df[group_cols].drop_duplicates().copy()

    agg_dict = {c: aggs for c in numeric_cols}

    out = df.groupby(group_cols, dropna=False)[numeric_cols].agg(agg_dict)
    out = flatten_columns(out.reset_index())

    rename = {}
    for c in out.columns:
        if c in group_cols:
            continue
        rename[c] = f"{prefix}_{c}"

    out = out.rename(columns=rename)
    return out


def safe_merge(left, right, on, how="left"):
    return left.merge(right, on=on, how=how, validate="one_to_one")


# ============================================================
# VEHICLE AGGREGATION
# ============================================================

def aggregate_vehicle_features(vehicle_df):
    require_columns(
        vehicle_df,
        KEY_COLS
        + [
            "lens_id",
            "relative_time_sec",
            "processed_frame_key",
            "idd_detection_status",
        ],
        "vehicle frame CSV",
    )

    ok = vehicle_df[vehicle_df["idd_detection_status"] == "success"].copy()

    if len(ok) == 0:
        raise ValueError("No successful vehicle detection rows found.")

    # These are frame-level numeric vehicle features from YOLO script.
    base_vehicle_cols = [
        "idd_total_detections_raw",
        "idd_total_vehicle_count",
        "idd_total_pm_relevant_objects",
        "idd_vehicle_box_area_ratio",
        "idd_average_confidence",
        "idd_max_confidence",
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
    ]

    vehicle_cols = [c for c in base_vehicle_cols if c in ok.columns]

    for c in vehicle_cols:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")

    # ------------------------------------------------------------------
    # A) All-lens aggregation directly across all frame rows
    # This treats every camera-frame as an exposure observation.
    # ------------------------------------------------------------------
    all_lens_aggs = ["sum", "mean", "max", "std"]

    veh_all = aggregate_numeric(
        df=ok,
        group_cols=KEY_COLS,
        numeric_cols=vehicle_cols,
        prefix="veh1s_all_lenses",
        aggs=all_lens_aggs,
    )

    # ------------------------------------------------------------------
    # B) Lens-normalized aggregation
    # First aggregate per sensor_row_id + lens_id, then average/max over lenses.
    # This avoids treating 3 lenses as simply triple exposure.
    # ------------------------------------------------------------------
    per_lens = (
        ok.groupby(KEY_COLS + ["lens_id"], dropna=False)[vehicle_cols]
        .agg(["sum", "mean", "max"])
        .reset_index()
    )
    per_lens = flatten_columns(per_lens)

    per_lens_numeric = [
        c for c in per_lens.columns
        if c not in KEY_COLS + ["lens_id"]
    ]

    veh_lens_norm = aggregate_numeric(
        df=per_lens,
        group_cols=KEY_COLS,
        numeric_cols=per_lens_numeric,
        prefix="veh1s_lensnorm",
        aggs=["mean", "max", "std"],
    )

    # ------------------------------------------------------------------
    # C) Availability / completeness summary
    # ------------------------------------------------------------------
    veh_avail = (
        ok.groupby(KEY_COLS, dropna=False)
        .agg(
            veh1s_frame_count=("processed_frame_key", "nunique"),
            veh1s_lens_count=("lens_id", "nunique"),
            veh1s_relative_second_count=("relative_time_sec", "nunique"),
            veh1s_min_relative_time_sec=("relative_time_sec", "min"),
            veh1s_max_relative_time_sec=("relative_time_sec", "max"),
        )
        .reset_index()
    )

    expected_vehicle_frames = 61 * 3
    veh_avail["veh1s_expected_frame_count"] = expected_vehicle_frames
    veh_avail["veh1s_frame_completeness"] = (
        veh_avail["veh1s_frame_count"] / expected_vehicle_frames
    )

    # ------------------------------------------------------------------
    # Merge vehicle blocks
    # ------------------------------------------------------------------
    out = veh_avail.copy()
    out = safe_merge(out, veh_all, on=KEY_COLS)
    out = safe_merge(out, veh_lens_norm, on=KEY_COLS)

    return out


# ============================================================
# ROAD AREA AGGREGATION
# ============================================================

def aggregate_road_area_features(road_df):
    require_columns(
        road_df,
        KEY_COLS
        + [
            "lens_id",
            "relative_time_sec",
            "processed_frame_key",
            "area_status",
        ],
        "road-area v3 CSV",
    )

    ok = road_df[road_df["area_status"] == "success"].copy()

    if len(ok) == 0:
        raise ValueError("No successful road-area rows found.")

    # Lens 1 only by construction, but enforce it.
    if "lens_id" in ok.columns:
        ok = ok[ok["lens_id"].astype(int) == 1].copy()

    road_numeric_cols = [
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_full_footprint_v3",
        "vehicle_occluded_road_area_m2_conservative_v3",
        "road_area_m2_occlusion_adjusted_conservative_v3",
        "road_area_vehicle_occlusion_fraction_v3",

        "vehicle_count_detected_for_occlusion_v3",
        "vehicle_count_used_for_occlusion_v3",
        "vehicle_count_rejected_by_depth_v3",
        "vehicle_count_missing_depth_v3",

        "road_depth_p5_m",
        "road_depth_p25_m",
        "road_depth_p50_m",
        "road_depth_p75_m",
        "road_depth_p95_m",
        "road_depth_std_m",

        "road_area_ratio_px",
        "road_area_percent_px",
        "raw_road_pixels",
        "road_pixels_used_for_area",
        "valid_road_quads_used",
    ]

    road_numeric_cols = [c for c in road_numeric_cols if c in ok.columns]

    for c in road_numeric_cols:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")

    road_aggs = ["mean", "median", "std", "min", "max"]

    road_num = aggregate_numeric(
        df=ok,
        group_cols=KEY_COLS,
        numeric_cols=road_numeric_cols,
        prefix="road1s_lens1",
        aggs=road_aggs,
    )

    road_avail = (
        ok.groupby(KEY_COLS, dropna=False)
        .agg(
            road1s_frame_count=("processed_frame_key", "nunique"),
            road1s_relative_second_count=("relative_time_sec", "nunique"),
            road1s_min_relative_time_sec=("relative_time_sec", "min"),
            road1s_max_relative_time_sec=("relative_time_sec", "max"),
        )
        .reset_index()
    )

    expected_road_frames = 61
    road_avail["road1s_expected_frame_count"] = expected_road_frames
    road_avail["road1s_frame_completeness"] = (
        road_avail["road1s_frame_count"] / expected_road_frames
    )

    # Status/quality proportions
    status_blocks = []

    categorical_cols = [
        "final_road_area_feature_status_v3",
        "vehicle_occlusion_quality_v3",
        "road_mask_area_quality_v3",
    ]

    for col in categorical_cols:
        if col not in ok.columns:
            continue

        tmp = (
            ok.groupby(KEY_COLS + [col], dropna=False)
            .size()
            .reset_index(name="count")
        )

        pivot = tmp.pivot_table(
            index=KEY_COLS,
            columns=col,
            values="count",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()

        pivot.columns = [
            str(c) if not isinstance(c, tuple) else "_".join(map(str, c))
            for c in pivot.columns
        ]

        category_cols = [c for c in pivot.columns if c not in KEY_COLS]

        total = pivot[category_cols].sum(axis=1).replace(0, np.nan)

        rename = {}
        for c in category_cols:
            clean = str(c).replace(" ", "_").replace("/", "_").lower()
            new_c = f"road1s_{col}_{clean}_count"
            pivot[f"road1s_{col}_{clean}_fraction"] = pivot[c] / total
            rename[c] = new_c

        pivot = pivot.rename(columns=rename)
        status_blocks.append(pivot)

    out = road_avail.copy()
    out = safe_merge(out, road_num, on=KEY_COLS)

    for block in status_blocks:
        out = safe_merge(out, block, on=KEY_COLS)

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vehicle-frame-csv",
        default=DEFAULT_VEHICLE_FRAME_CSV,
    )

    parser.add_argument(
        "--road-area-v3-csv",
        default=DEFAULT_ROAD_AREA_V3_CSV,
    )

    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
    )

    args = parser.parse_args()

    vehicle_path = Path(args.vehicle_frame_csv)
    road_path = Path(args.road_area_v3_csv)
    output_path = Path(args.output_csv)

    if not vehicle_path.exists():
        raise FileNotFoundError(f"Vehicle frame CSV not found: {vehicle_path}")

    if not road_path.exists():
        raise FileNotFoundError(f"Road-area v3 CSV not found: {road_path}")

    vehicle_df = pd.read_csv(vehicle_path)
    road_df = pd.read_csv(road_path)

    print("\nInput vehicle frame CSV:", vehicle_path)
    print("Vehicle shape:", vehicle_df.shape)

    print("\nInput road-area v3 CSV:", road_path)
    print("Road shape:", road_df.shape)

    print("\nAggregating vehicle features...")
    vehicle_agg = aggregate_vehicle_features(vehicle_df)
    print("Vehicle aggregated shape:", vehicle_agg.shape)

    print("\nAggregating road-area features...")
    road_agg = aggregate_road_area_features(road_df)
    print("Road aggregated shape:", road_agg.shape)

    print("\nMerging vehicle + road features...")
    merged = vehicle_agg.merge(
        road_agg,
        on=KEY_COLS,
        how="outer",
        validate="one_to_one",
    )

    merged = merged.sort_values("sensor_row_id").reset_index(drop=True)

    merged["image_1s_features_available"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print("\nSaved:")
    print(output_path)
    print("Final shape:", merged.shape)

    print("\nSensor row coverage:")
    print("Rows:", len(merged))
    print("sensor_row_id min/max:", merged["sensor_row_id"].min(), merged["sensor_row_id"].max())

    print("\nCompleteness summary:")
    cols = [
        "veh1s_frame_completeness",
        "road1s_frame_completeness",
        "veh1s_frame_count",
        "road1s_frame_count",
    ]
    cols = [c for c in cols if c in merged.columns]
    print(merged[cols].describe().T)

    print("\nHead:")
    show_cols = [
        "sensor_row_id",
        "sample_index",
        "sensor_timestamp",
        "veh1s_frame_count",
        "road1s_frame_count",
        "veh1s_all_lenses_idd_total_vehicle_count_sum",
        "veh1s_all_lenses_idd_total_vehicle_count_mean",
        "road1s_lens1_road_area_m2_occlusion_adjusted_conservative_v3_mean",
        "road1s_lens1_road_area_vehicle_occlusion_fraction_v3_mean",
    ]
    show_cols = [c for c in show_cols if c in merged.columns]
    print(merged[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()