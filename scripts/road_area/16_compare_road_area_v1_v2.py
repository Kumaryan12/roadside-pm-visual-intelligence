from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v1-csv",
        default="outputs/road_area_lens1_batch/lens1_final_road_area_features_v1.csv",
    )

    parser.add_argument(
        "--v2-csv",
        default="outputs/road_area_lens1_batch/lens1_bbox_visibility_only_occlusion_adjusted_road_area_v2.csv",
    )

    parser.add_argument(
        "--output-csv",
        default="outputs/road_area_lens1_batch/lens1_road_area_v1_v2_comparison.csv",
    )

    args = parser.parse_args()

    v1 = pd.read_csv(args.v1_csv)
    v2 = pd.read_csv(args.v2_csv)

    key_cols = ["matched_run_id", "sample_index", "lens_id"]

    v1_cols = key_cols + [
        "road_area_m2_depth_est_visible_v1",
        "road_area_m2_vehicle_occluded_conservative_v1",
        "road_area_m2_occlusion_adjusted_conservative_v1",
        "road_area_vehicle_occlusion_fraction_v1",
        "final_road_area_feature_status",
    ]

    v1_cols = [c for c in v1_cols if c in v1.columns]

    v2_cols = key_cols + [
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_conservative_v2",
        "road_area_m2_occlusion_adjusted_conservative_v2",
        "road_area_vehicle_occlusion_fraction_v2",
        "final_road_area_feature_status_v2",
    ]

    v2_cols = [c for c in v2_cols if c in v2.columns]

    merged = v1[v1_cols].merge(
        v2[v2_cols],
        on=key_cols,
        how="outer",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )

    if "road_area_m2_occlusion_adjusted_conservative_v1" in merged.columns:
        merged["final_area_v2_minus_v1_m2"] = (
            merged["road_area_m2_occlusion_adjusted_conservative_v2"]
            - merged["road_area_m2_occlusion_adjusted_conservative_v1"]
        )

        merged["final_area_v2_vs_v1_ratio"] = (
            merged["road_area_m2_occlusion_adjusted_conservative_v2"]
            / merged["road_area_m2_occlusion_adjusted_conservative_v1"]
        )

    if "road_area_m2_vehicle_occluded_conservative_v1" in merged.columns:
        merged["vehicle_hidden_v2_minus_v1_m2"] = (
            merged["vehicle_occluded_road_area_m2_conservative_v2"]
            - merged["road_area_m2_vehicle_occluded_conservative_v1"]
        )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print("\nSaved comparison:")
    print(out)
    print("Shape:", merged.shape)

    print("\nStatus v1:")
    if "final_road_area_feature_status" in merged.columns:
        print(merged["final_road_area_feature_status"].value_counts(dropna=False))

    print("\nStatus v2:")
    if "final_road_area_feature_status_v2" in merged.columns:
        print(merged["final_road_area_feature_status_v2"].value_counts(dropna=False))

    print("\nComparison summary:")
    cols = [
        "road_area_m2_occlusion_adjusted_conservative_v1",
        "road_area_m2_occlusion_adjusted_conservative_v2",
        "final_area_v2_minus_v1_m2",
        "final_area_v2_vs_v1_ratio",
        "road_area_vehicle_occlusion_fraction_v1",
        "road_area_vehicle_occlusion_fraction_v2",
        "vehicle_hidden_v2_minus_v1_m2",
    ]

    cols = [c for c in cols if c in merged.columns]
    print(merged[cols].describe().T)

    print("\nLargest v2 increases:")
    show_cols = key_cols + cols
    print(
        merged[show_cols]
        .sort_values("final_area_v2_minus_v1_m2", ascending=False)
        .head(25)
        .to_string(index=False)
    )

    print("\nLargest v2 decreases:")
    print(
        merged[show_cols]
        .sort_values("final_area_v2_minus_v1_m2", ascending=True)
        .head(25)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()