from pathlib import Path
import argparse
import pandas as pd
import numpy as np


CLASS_CONSERVATIVE_FACTOR = {
    "car": 0.60,
    "auto_rickshaw": 0.60,
    "autorickshaw": 0.60,
    "motorcycle": 0.50,
    "bus": 0.50,
    "truck": 0.50,
    "bicycle": 0.50,
    "unknown_vehicle": 0.50,
    "vehicle fallback": 0.50,
}


def norm_class(x):
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def quality_from_fraction(frac):
    if not np.isfinite(frac):
        return "unknown"
    if frac <= 0.30:
        return "good"
    if frac <= 0.50:
        return "medium"
    if frac <= 0.70:
        return "low"
    return "very_low"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frame-csv",
        default="outputs/road_area_lens1_batch/lens1_bbox_nms_footprint_occlusion_adjusted_road_area.csv",
    )

    parser.add_argument(
        "--vehicle-detail-csv",
        default="outputs/road_area_lens1_batch/lens1_bbox_nms_vehicle_occlusion_details.csv",
    )

    parser.add_argument(
        "--output-frame-csv",
        default="outputs/road_area_lens1_batch/lens1_bbox_nms_conservative_occlusion_adjusted_road_area.csv",
    )

    parser.add_argument(
        "--output-vehicle-detail-csv",
        default="outputs/road_area_lens1_batch/lens1_bbox_nms_conservative_vehicle_occlusion_details.csv",
    )

    args = parser.parse_args()

    frame_df = pd.read_csv(args.frame_csv)
    veh_df = pd.read_csv(args.vehicle_detail_csv)

    veh_df["class_norm"] = veh_df["class_name"].apply(norm_class)
    veh_df["conservative_factor"] = veh_df["class_norm"].map(CLASS_CONSERVATIVE_FACTOR).fillna(0.50)

    veh_df["hidden_road_area_m2_est_full_footprint"] = veh_df["hidden_road_area_m2_est"]
    veh_df["hidden_road_area_m2_est_conservative"] = (
        veh_df["hidden_road_area_m2_est_full_footprint"]
        * veh_df["conservative_factor"]
    )

    agg = (
        veh_df
        .groupby(["sample_index", "lens_id"], as_index=False)
        .agg(
            vehicle_occluded_road_area_m2_est_conservative=(
                "hidden_road_area_m2_est_conservative", "sum"
            ),
            vehicle_count_used_for_occlusion_conservative=(
                "hidden_road_area_m2_est_conservative",
                lambda s: int((s > 0).sum())
            ),
        )
    )

    out = frame_df.copy()

    out = out.merge(
        agg,
        on=["sample_index", "lens_id"],
        how="left",
    )

    out["vehicle_occluded_road_area_m2_est_conservative"] = (
        out["vehicle_occluded_road_area_m2_est_conservative"].fillna(0.0)
    )

    visible_col = "visible_road_area_m2_depth_est"

    out["occlusion_adjusted_road_area_m2_est_conservative"] = (
        out[visible_col]
        + out["vehicle_occluded_road_area_m2_est_conservative"]
    )

    out["vehicle_occlusion_fraction_conservative"] = (
        out["vehicle_occluded_road_area_m2_est_conservative"]
        / out["occlusion_adjusted_road_area_m2_est_conservative"].replace(0, np.nan)
    )

    out["occlusion_adjustment_quality_conservative"] = (
        out["vehicle_occlusion_fraction_conservative"].apply(quality_from_fraction)
    )

    Path(args.output_frame_csv).parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(args.output_frame_csv, index=False)
    veh_df.to_csv(args.output_vehicle_detail_csv, index=False)

    print("Saved frame:", args.output_frame_csv)
    print("Saved vehicle:", args.output_vehicle_detail_csv)

    cols = [
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_est",
        "vehicle_occluded_road_area_m2_est_conservative",
        "occlusion_adjusted_road_area_m2_est",
        "occlusion_adjusted_road_area_m2_est_conservative",
        "vehicle_occlusion_fraction_of_adjusted_area",
        "vehicle_occlusion_fraction_conservative",
    ]

    print("\nSummary:")
    print(out[cols].describe().T)

    print("\nQuality counts:")
    print(out["occlusion_adjustment_quality_conservative"].value_counts(dropna=False))

    print("\nWorst conservative occlusion fractions:")
    print(
        out[
            [
                "sample_index",
                "visible_road_area_m2_depth_est",
                "vehicle_occluded_road_area_m2_est_conservative",
                "occlusion_adjusted_road_area_m2_est_conservative",
                "vehicle_occlusion_fraction_conservative",
                "occlusion_adjustment_quality_conservative",
            ]
        ]
        .sort_values("vehicle_occlusion_fraction_conservative", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()