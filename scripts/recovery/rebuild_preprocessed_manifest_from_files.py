from pathlib import Path
import re
import pandas as pd


PREPROCESSED_ROOT = Path("outputs/preprocessed_frames_v2")
SOURCE_MANIFEST = Path("outputs/features/processed_frame_manifest_v2.csv")
OUTPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")


def main():
    if not PREPROCESSED_ROOT.exists():
        raise FileNotFoundError(f"Preprocessed root not found: {PREPROCESSED_ROOT}")

    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(f"Source manifest not found: {SOURCE_MANIFEST}")

    source = pd.read_csv(SOURCE_MANIFEST)

    source_success = source[source["preprocess_status"] == "success"].copy()

    files = sorted(PREPROCESSED_ROOT.rglob("*.jpg"))

    print("Preprocessed image files found:", len(files))
    print("Source successful extracted rows:", len(source_success))

    rows = []

    for file_path in files:
        processed_frame_key = file_path.stem

        # The preprocessed key was created as:
        # {source_frame_key}_preprocessed_lens{lens_id}
        match = re.match(r"(.+)_preprocessed_lens(\d+)$", processed_frame_key)

        if not match:
            print("Skipping unrecognized filename:", file_path.name)
            continue

        source_frame_key = match.group(1)
        lens_id = int(match.group(2))

        source_match = source_success[
            (source_success["processed_frame_key"].astype(str) == source_frame_key)
            & (pd.to_numeric(source_success["lens_id"], errors="coerce") == lens_id)
        ]

        if len(source_match) == 0:
            print("No source manifest match for:", processed_frame_key)
            continue

        if len(source_match) > 1:
            print("Multiple source matches for:", processed_frame_key)

        src = source_match.iloc[0].to_dict()

        rows.append({
            "sample_index": src.get("sample_index", None),
            "sensor_timestamp": src.get("sensor_timestamp", ""),
            "sample_unix": src.get("sample_unix", None),
            "lens_id": lens_id,
            "matched_run_id": src.get("matched_run_id", ""),
            "video_offset_sec": src.get("video_offset_sec", None),
            "source_video_path": src.get("source_video_path", ""),

            "source_frame_key": source_frame_key,
            "source_frame_path": src.get("processed_frame_path", ""),

            "processed_frame_key": processed_frame_key,
            "processed_frame_path": str(file_path),

            # We do not need exact crop/mask pixels for downstream road/vehicle features.
            # Mark unavailable if manifest was recovered from files.
            "crop_x1": None,
            "crop_y1": None,
            "crop_x2": None,
            "crop_y2": None,
            "crop_x1_ratio": None,
            "crop_y1_ratio": None,
            "crop_x2_ratio": None,
            "crop_y2_ratio": None,
            "rotation_name": "recovered_existing_file",
            "mask_applied": True,
            "mask_x1": None,
            "mask_y1": None,
            "mask_x2": None,
            "mask_y2": None,
            "mask_x1_ratio": None,
            "mask_y1_ratio": None,
            "mask_x2_ratio": None,
            "mask_y2_ratio": None,

            "preprocess_status": "success",
            "preprocess_error": "",
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["sample_index", "lens_id"]).reset_index(drop=True)

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_MANIFEST, index=False)

    print("\nSaved:", OUTPUT_MANIFEST)
    print("Shape:", out.shape)

    print("\nStatus:")
    print(out["preprocess_status"].value_counts(dropna=False))

    print("\nRows by lens:")
    print(out.groupby("lens_id")["preprocess_status"].value_counts(dropna=False))

    print("\nUnique samples:", out["sample_index"].nunique())

    missing_paths = [p for p in out["processed_frame_path"] if not Path(p).exists()]
    print("Missing processed_frame_path files:", len(missing_paths))


if __name__ == "__main__":
    main()