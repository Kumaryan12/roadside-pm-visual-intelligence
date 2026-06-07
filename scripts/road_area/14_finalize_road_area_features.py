from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def road_mask_quality(visible_area):
    if not np.isfinite(visible_area):
        return "unknown"
    if visible_area < 15:
        return "very_low_visible_road"
    if visible_area < 30:
        return "low_visible_road"
    return "acceptable_visible_road"


def final_status(row):
    occ_q = str(row.get("occlusion_adjustment_quality_conservative", "unknown"))
    road_q = str(row.get("road_mask_area_quality", "unknown"))

    if occ_q == "very_low":
        return "exclude_or_manual_review"

    if road_q == "very_low_visible_road":
        return "exclude_or_manual_review"

    if occ_q == "low":
        return "use_sensitivity_only"

    if road_q == "low_visible_road":
        return "use_sensitivity_only"

    if occ_q in ["good", "medium"] and road_q == "acceptable_visible_road":
        return "use_primary"

    return "manual_review"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-csv",
        default="outputs/road_area_lens1_batch/lens1_bbox_nms_conservative_occlusion_adjusted_road_area.csv",
    )

    parser.add_argument(
        "--output-csv",
        default="outputs/road_area_lens1_batch/lens1_final_road_area_features_v1.csv",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required = [
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_est_conservative",
        "occlusion_adjusted_road_area_m2_est_conservative",
        "vehicle_occlusion_fraction_conservative",
        "occlusion_adjustment_quality_conservative",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["road_mask_area_quality"] = df["visible_road_area_m2_depth_est"].apply(road_mask_quality)

    df["final_road_area_feature_status"] = df.apply(final_status, axis=1)

    df["road_area_occlusion_added_m2"] = df["vehicle_occluded_road_area_m2_est_conservative"]

    df["road_area_occlusion_added_ratio_vs_visible"] = (
        df["road_area_occlusion_added_m2"]
        / df["visible_road_area_m2_depth_est"].replace(0, np.nan)
    )

    df["road_area_feature_version"] = "v1_depth_visible_plus_nms_conservative_vehicle_footprint"

    # Clean alias columns for modeling/reporting
    df["road_area_m2_depth_est_visible_v1"] = df["visible_road_area_m2_depth_est"]
    df["road_area_m2_vehicle_occluded_conservative_v1"] = df[
        "vehicle_occluded_road_area_m2_est_conservative"
    ]
    df["road_area_m2_occlusion_adjusted_conservative_v1"] = df[
        "occlusion_adjusted_road_area_m2_est_conservative"
    ]
    df["road_area_vehicle_occlusion_fraction_v1"] = df[
        "vehicle_occlusion_fraction_conservative"
    ]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nSaved final road-area features:")
    print(output_csv)
    print("Shape:", df.shape)

    print("\nFinal status counts:")
    print(df["final_road_area_feature_status"].value_counts(dropna=False))

    print("\nRoad mask quality counts:")
    print(df["road_mask_area_quality"].value_counts(dropna=False))

    print("\nOcclusion quality counts:")
    print(df["occlusion_adjustment_quality_conservative"].value_counts(dropna=False))

    summary_cols = [
        "road_area_m2_depth_est_visible_v1",
        "road_area_m2_vehicle_occluded_conservative_v1",
        "road_area_m2_occlusion_adjusted_conservative_v1",
        "road_area_vehicle_occlusion_fraction_v1",
        "road_area_occlusion_added_ratio_vs_visible",
    ]

    print("\nSummary:")
    print(df[summary_cols].describe().T)

    print("\nFrames needing review/exclusion:")
    review = df[df["final_road_area_feature_status"] != "use_primary"]
    cols = [
        "sample_index",
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_est_conservative",
        "occlusion_adjusted_road_area_m2_est_conservative",
        "vehicle_occlusion_fraction_conservative",
        "occlusion_adjustment_quality_conservative",
        "road_mask_area_quality",
        "final_road_area_feature_status",
    ]
    cols = [c for c in cols if c in review.columns]
    print(review[cols].sort_values("vehicle_occlusion_fraction_conservative", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()