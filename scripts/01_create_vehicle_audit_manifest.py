from pathlib import Path

import pandas as pd


# Old project file location
OLD_PROJECT_ROOT = Path("/Users/aryansatyendrakumar/Projects/pm_density_image_pipeline")

INPUT_MANIFEST = OLD_PROJECT_ROOT / "outputs/features/processed_frame_manifest.csv"

OUTPUT_DIR = Path("audits")
OUTPUT_CSV = OUTPUT_DIR / "vehicle_count_audit_v1.csv"

RANDOM_STATE = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_MANIFEST.exists():
        print(f"Input manifest not found: {INPUT_MANIFEST}")
        return

    df = pd.read_csv(INPUT_MANIFEST)

    print("\nLoaded processed frame manifest:")
    print(df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    required_cols = [
        "processed_frame_key",
        "lens_id",
        "processed_frame_path",
        "preprocess_status",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print("\nMissing required columns:")
        print(missing)
        return

    df = df[df["preprocess_status"] == "success"].copy()

    print("\nAfter keeping successful processed frames:")
    print(df.shape)

    # We want a balanced audit across lenses.
    # Choose up to 40 frames from each lens.
    audit_parts = []

    for lens_id in sorted(df["lens_id"].unique()):
        lens_df = df[df["lens_id"] == lens_id].copy()

        n = min(40, len(lens_df))

        sampled = lens_df.sample(
            n=n,
            random_state=RANDOM_STATE + int(lens_id),
        )

        audit_parts.append(sampled)

    audit_df = pd.concat(audit_parts, axis=0).reset_index(drop=True)

    # Keep only useful metadata
    keep_cols = [
        "processed_frame_key",
        "source_frame_key",
        "matched_run_id",
        "video_offset_sec",
        "lens_id",
        "processed_frame_path",
    ]

    keep_cols = [c for c in keep_cols if c in audit_df.columns]

    audit_df = audit_df[keep_cols].copy()

    # Manual annotation columns
    audit_df["manual_car_count"] = ""
    audit_df["manual_motorcycle_count"] = ""
    audit_df["manual_auto_rickshaw_count"] = ""
    audit_df["manual_bus_count"] = ""
    audit_df["manual_truck_count"] = ""
    audit_df["manual_bicycle_count"] = ""
    audit_df["manual_person_count"] = ""

    # Optional qualitative columns
    audit_df["traffic_density_label"] = ""
    audit_df["visibility_label"] = ""
    audit_df["occlusion_level"] = ""
    audit_df["notes"] = ""

    audit_df = audit_df.sort_values(
        ["lens_id", "video_offset_sec"]
    ).reset_index(drop=True)

    audit_df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved audit manifest to: {OUTPUT_CSV}")
    print("Shape:", audit_df.shape)

    print("\nRows per lens:")
    print(audit_df["lens_id"].value_counts().sort_index())

    print("\nManual count columns to fill:")
    print([
        "manual_car_count",
        "manual_motorcycle_count",
        "manual_auto_rickshaw_count",
        "manual_bus_count",
        "manual_truck_count",
        "manual_bicycle_count",
        "manual_person_count",
    ])


if __name__ == "__main__":
    main()