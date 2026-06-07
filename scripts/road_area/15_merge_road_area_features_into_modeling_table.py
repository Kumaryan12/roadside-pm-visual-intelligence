from pathlib import Path
import argparse
import pandas as pd


ROAD_AREA_COLUMNS = [
    # Possible keys / metadata
    "processed_frame_key",
    "matched_run_id",
    "sample_index",
    "lens_id",
    "video_offset_sec",

    # Clean v1 feature aliases
    "road_area_m2_depth_est_visible_v1",
    "road_area_m2_vehicle_occluded_conservative_v1",
    "road_area_m2_occlusion_adjusted_conservative_v1",
    "road_area_vehicle_occlusion_fraction_v1",

    # Quality / status columns
    "road_mask_area_quality",
    "occlusion_adjustment_quality_conservative",
    "final_road_area_feature_status",
    "road_area_feature_version",

    # Original/intermediate columns for traceability
    "visible_road_area_m2_depth_est",
    "vehicle_occluded_road_area_m2_est_conservative",
    "occlusion_adjusted_road_area_m2_est_conservative",
    "vehicle_occlusion_fraction_conservative",
    "road_area_occlusion_added_ratio_vs_visible",
]


def choose_merge_keys(model_df: pd.DataFrame, road_df: pd.DataFrame):
    """
    Choose the safest available merge key.

    Your modeling table is sensor-level, so it usually does not have
    processed_frame_key or lens_id. For the current project, the correct
    key is expected to be:

        matched_run_id + sample_index

    because the road-area table is lens-1-only, one row per sample.
    """

    candidates = [
        ["processed_frame_key"],
        ["matched_run_id", "sample_index", "lens_id"],
        ["matched_run_id", "sample_index"],
        ["sample_index", "lens_id"],
    ]

    for keys in candidates:
        if all(k in model_df.columns for k in keys) and all(k in road_df.columns for k in keys):
            return keys

    common = sorted(set(model_df.columns).intersection(set(road_df.columns)))

    raise ValueError(
        "No safe merge key found.\n"
        f"Common columns are: {common}\n"
        "Expected at least matched_run_id + sample_index."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modeling-table",
        default="outputs/features/particle_density_modeling_table_idd_finetuned_osm_v2.csv",
        help="Sensor-level modeling table.",
    )

    parser.add_argument(
        "--road-area-csv",
        default="outputs/road_area_lens1_batch/lens1_final_road_area_features_v1.csv",
        help="Final lens-1 road-area feature CSV.",
    )

    parser.add_argument(
        "--output-csv",
        default="outputs/features/particle_density_modeling_table_idd_finetuned_osm_roadarea_v1.csv",
        help="Output modeling table with road-area features merged.",
    )

    parser.add_argument(
        "--primary-only-output-csv",
        default="outputs/features/particle_density_modeling_table_idd_finetuned_osm_roadarea_v1_primary_only.csv",
        help="Output table retaining only rows with primary-quality road-area features.",
    )

    args = parser.parse_args()

    modeling_path = Path(args.modeling_table)
    road_area_path = Path(args.road_area_csv)

    if not modeling_path.exists():
        raise FileNotFoundError(f"Modeling table not found: {modeling_path}")

    if not road_area_path.exists():
        raise FileNotFoundError(f"Road-area CSV not found: {road_area_path}")

    model_df = pd.read_csv(modeling_path)
    road_df = pd.read_csv(road_area_path)

    print("\nInput modeling table:", modeling_path)
    print("Modeling shape:", model_df.shape)

    print("\nRoad-area table:", road_area_path)
    print("Road-area shape:", road_df.shape)

    merge_keys = choose_merge_keys(model_df, road_df)
    print("\nUsing merge keys:", merge_keys)

    # Since road-area table is lens-1 calibrated, make sure we only merge lens 1
    # if the lens_id column exists in road_df.
    if "lens_id" in road_df.columns:
        print("\nRoad-area lens counts before filtering:")
        print(road_df["lens_id"].value_counts(dropna=False))

        road_df = road_df[road_df["lens_id"] == 1].copy()

        print("\nRoad-area shape after keeping lens_id == 1:", road_df.shape)

    # Keep only relevant road-area columns that exist.
    keep_cols = [c for c in ROAD_AREA_COLUMNS if c in road_df.columns]

    # Ensure merge keys are retained.
    for key in merge_keys:
        if key not in keep_cols:
            keep_cols.insert(0, key)

    road_small = road_df[keep_cols].copy()

    # Check duplicates on road side. The right side of a many_to_one merge must
    # have unique keys.
    duplicate_count = road_small.duplicated(subset=merge_keys).sum()

    print("\nRoad-area duplicate key rows:", duplicate_count)

    if duplicate_count > 0:
        print("WARNING: duplicate road-area keys found. Keeping first occurrence.")
        road_small = road_small.drop_duplicates(subset=merge_keys, keep="first")

    # Avoid suffix chaos: remove non-key columns from road_small if they already
    # exist in the modeling table.
    overlapping_non_key_cols = [
        c for c in road_small.columns
        if c in model_df.columns and c not in merge_keys
    ]

    if overlapping_non_key_cols:
        print("\nDropping overlapping non-key columns from road-area table:")
        print(overlapping_non_key_cols)
        road_small = road_small.drop(columns=overlapping_non_key_cols)

    merged = model_df.merge(
        road_small,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print("\nSaved merged table:")
    print(output_csv)
    print("Merged shape:", merged.shape)

    road_feature_col = "road_area_m2_occlusion_adjusted_conservative_v1"

    if road_feature_col in merged.columns:
        non_missing = merged[road_feature_col].notna().sum()

        print(f"\nRows with road-area features: {non_missing} / {len(merged)}")

        print("\nRoad-area adjusted feature summary:")
        print(merged[road_feature_col].describe())

    else:
        print(f"\nWARNING: {road_feature_col} not found after merge.")

    if "final_road_area_feature_status" in merged.columns:
        print("\nFinal road-area feature status counts:")
        print(merged["final_road_area_feature_status"].value_counts(dropna=False))

        primary = merged[
            merged["final_road_area_feature_status"] == "use_primary"
        ].copy()

        primary_csv = Path(args.primary_only_output_csv)
        primary.to_csv(primary_csv, index=False)

        print("\nSaved primary-only table:")
        print(primary_csv)
        print("Primary-only shape:", primary.shape)

    print("\nDone.")


if __name__ == "__main__":
    main()