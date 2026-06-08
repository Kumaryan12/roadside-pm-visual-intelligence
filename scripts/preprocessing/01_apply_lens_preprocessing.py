from pathlib import Path
import argparse

import cv2
import pandas as pd


INPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_v2.csv")
OUTPUT_ROOT = Path("outputs/preprocessed_frames_v2")
OUTPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")


# Crop ratios selected from previous manual preview inspection.
# Format: lens_id: (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
LENS_CROP_CONFIG = {
    1: (0.25, 0.10, 0.70, 0.88),
    4: (0.40, 0.10, 0.90, 0.88),
    6: (0.15, 0.10, 0.85, 0.88),
}


# Rotation chosen from previous manual preview inspection.
LENS_ROTATION_CONFIG = {
    1: "rot90_counterclockwise",
    4: "rot90_counterclockwise",
    6: "rot90_counterclockwise",
}


# Platform mask after crop + rotation.
# Format: lens_id: (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
PLATFORM_MASK_CONFIG = {
    1: (0.00, 0.82, 1.00, 1.00),
    4: (0.00, 0.95, 1.00, 1.00),
    6: (0.00, 0.75, 1.00, 1.00),
}


def resolve_path(path_value: str) -> Path:
    """
    Resolve paths relative to the current project root.
    """
    path = Path(str(path_value))

    if path.is_absolute():
        return path

    return Path.cwd() / path


def crop_image(image, crop_ratios):
    h, w = image.shape[:2]

    x1r, y1r, x2r, y2r = crop_ratios

    x1 = int(x1r * w)
    y1 = int(y1r * h)
    x2 = int(x2r * w)
    y2 = int(y2r * h)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    cropped = image[y1:y2, x1:x2]
    return cropped, x1, y1, x2, y2


def rotate_image(image, rotation_name):
    if rotation_name == "rot0":
        return image

    if rotation_name == "rot90_clockwise":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if rotation_name == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)

    if rotation_name == "rot90_counterclockwise":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError(f"Unknown rotation name: {rotation_name}")


def apply_platform_mask(image, mask_ratios):
    """
    Applies a black rectangular mask on the processed frame.
    This hides the fixed camera roof/platform so YOLO does not detect it.
    """
    h, w = image.shape[:2]

    x1r, y1r, x2r, y2r = mask_ratios

    x1 = int(x1r * w)
    y1 = int(y1r * h)
    x2 = int(x2r * w)
    y2 = int(y2r * h)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    masked = image.copy()
    masked[y1:y2, x1:x2] = (0, 0, 0)

    return masked, x1, y1, x2, y2


def preprocess_one_row(row, no_mask: bool, output_root: Path):
    # Keep the original extracted-frame identity before overwriting processed fields
    original_row = row.to_dict()

    source_frame_key = str(row["processed_frame_key"])
    lens_id = int(row["lens_id"])
    source_frame_path = resolve_path(row["processed_frame_path"])

    if lens_id not in LENS_CROP_CONFIG:
        result = original_row.copy()
        result.update({
            "source_frame_key": source_frame_key,
            "source_frame_path": str(source_frame_path),
            "processed_frame_key": "",
            "processed_frame_path": "",
            "preprocess_status": "failed",
            "preprocess_error": f"no_crop_config_for_lens_{lens_id}",
        })
        return result

    if lens_id not in LENS_ROTATION_CONFIG:
        result = original_row.copy()
        result.update({
            "source_frame_key": source_frame_key,
            "source_frame_path": str(source_frame_path),
            "processed_frame_key": "",
            "processed_frame_path": "",
            "preprocess_status": "failed",
            "preprocess_error": f"no_rotation_config_for_lens_{lens_id}",
        })
        return result

    if not source_frame_path.exists():
        result = original_row.copy()
        result.update({
            "source_frame_key": source_frame_key,
            "source_frame_path": str(source_frame_path),
            "processed_frame_key": "",
            "processed_frame_path": "",
            "preprocess_status": "failed",
            "preprocess_error": f"source_frame_not_found: {source_frame_path}",
        })
        return result

    image = cv2.imread(str(source_frame_path))

    if image is None:
        result = original_row.copy()
        result.update({
            "source_frame_key": source_frame_key,
            "source_frame_path": str(source_frame_path),
            "processed_frame_key": "",
            "processed_frame_path": "",
            "preprocess_status": "failed",
            "preprocess_error": f"could_not_read_source_frame: {source_frame_path}",
        })
        return result

    crop_ratios = LENS_CROP_CONFIG[lens_id]
    rotation_name = LENS_ROTATION_CONFIG[lens_id]

    cropped, crop_x1, crop_y1, crop_x2, crop_y2 = crop_image(
        image=image,
        crop_ratios=crop_ratios,
    )

    rotated = rotate_image(cropped, rotation_name)

    mask_applied = False
    mask_x1 = mask_y1 = mask_x2 = mask_y2 = None
    mask_ratios = (None, None, None, None)

    if not no_mask and lens_id in PLATFORM_MASK_CONFIG:
        mask_ratios = PLATFORM_MASK_CONFIG[lens_id]
        processed, mask_x1, mask_y1, mask_x2, mask_y2 = apply_platform_mask(
            image=rotated,
            mask_ratios=mask_ratios,
        )
        mask_applied = True
    else:
        processed = rotated

    processed_frame_key = f"{source_frame_key}_preprocessed_lens{lens_id}"

    matched_run_id = str(row.get("matched_run_id", "unknown_run"))

    # For 1-second pipeline, include sensor_row_id in the folder if available.
    # This prevents too many files being dumped into only run/lens folders.
    sensor_row_id = row.get("sensor_row_id", None)

    if pd.notna(sensor_row_id):
        output_path = (
            output_root
            / matched_run_id
            / f"sensor_{int(sensor_row_id):05d}"
            / f"lens{lens_id}"
            / f"{processed_frame_key}.jpg"
        )
    else:
        output_path = (
            output_root
            / matched_run_id
            / f"lens{lens_id}"
            / f"{processed_frame_key}.jpg"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), processed)

    if not ok:
        result = original_row.copy()
        result.update({
            "source_frame_key": source_frame_key,
            "source_frame_path": str(source_frame_path),
            "processed_frame_key": processed_frame_key,
            "processed_frame_path": str(output_path),
            "preprocess_status": "failed",
            "preprocess_error": "cv2_imwrite_failed",
        })
        return result

    # Important:
    # Start from original row so 1-second metadata is preserved:
    # sensor_row_id, sensor_unix, frame_sample_unix, relative_time_sec, etc.
    result = original_row.copy()

    result.update({
        # Source frame info
        "source_frame_key": source_frame_key,
        "source_frame_path": str(source_frame_path),

        # New processed frame info
        "processed_frame_key": processed_frame_key,
        "processed_frame_path": str(output_path),

        # Keep normalized lens/run values
        "lens_id": lens_id,
        "matched_run_id": matched_run_id,

        # Crop metadata
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "crop_x2": crop_x2,
        "crop_y2": crop_y2,
        "crop_x1_ratio": crop_ratios[0],
        "crop_y1_ratio": crop_ratios[1],
        "crop_x2_ratio": crop_ratios[2],
        "crop_y2_ratio": crop_ratios[3],

        # Rotation metadata
        "rotation_name": rotation_name,

        # Mask metadata
        "mask_applied": mask_applied,
        "mask_x1": mask_x1,
        "mask_y1": mask_y1,
        "mask_x2": mask_x2,
        "mask_y2": mask_y2,
        "mask_x1_ratio": mask_ratios[0],
        "mask_y1_ratio": mask_ratios[1],
        "mask_x2_ratio": mask_ratios[2],
        "mask_y2_ratio": mask_ratios[3],

        # Status
        "preprocess_status": "success",
        "preprocess_error": "",
    })

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-manifest",
        type=str,
        default=str(INPUT_MANIFEST),
        help="Input extracted-frame manifest.",
    )

    parser.add_argument(
        "--output-manifest",
        type=str,
        default=str(OUTPUT_MANIFEST),
        help="Output preprocessed-frame manifest.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT),
        help="Output root folder for preprocessed frames.",
    )

    parser.add_argument(
        "--lenses",
        nargs="+",
        type=int,
        default=[1, 4, 6],
        help="Lens IDs to preprocess. Example: --lenses 1 4 6",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional test limit. Example: --limit 10",
    )

    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable platform mask for debugging.",
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)
    input_manifest = Path(args.input_manifest)
    output_manifest = Path(args.output_manifest)

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    df = pd.read_csv(input_manifest)

    if "preprocess_status" in df.columns:
        df = df[df["preprocess_status"] == "success"].copy()

    df = df[df["lens_id"].isin(args.lenses)].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    if df.empty:
        print("No extracted frames found for selected lenses.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    print("\nLens preprocessing setup")
    print("=" * 60)
    print("Input manifest:", input_manifest)
    print("Output manifest:", output_manifest)
    print("Output root:", output_root)
    print("Frames selected:", len(df))
    print("Lenses selected:", args.lenses)
    print("Apply platform mask:", not args.no_mask)

    output_rows = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"\n[{i}/{len(df)}] Preprocessing: {row['processed_frame_key']}")

        result = preprocess_one_row(row, no_mask=args.no_mask, output_root=output_root)
        output_rows.append(result)

        print("  Lens:", result.get("lens_id"))
        print("  Status:", result.get("preprocess_status"))
        print("  Source:", result.get("source_frame_path"))
        print("  Output:", result.get("processed_frame_path"))
        print("  Rotation:", result.get("rotation_name"))
        print("  Mask:", result.get("mask_applied"))

        if result.get("preprocess_error"):
            print("  Error:", result.get("preprocess_error"))

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_manifest, index=False)

    print("\nDone.")
    print("Saved manifest:", output_manifest)
    print("Shape:", out_df.shape)

    print("\nStatus counts:")
    print(out_df["preprocess_status"].value_counts(dropna=False))

    print("\nRows by lens:")
    print(out_df.groupby("lens_id")["preprocess_status"].value_counts())


if __name__ == "__main__":
    main()