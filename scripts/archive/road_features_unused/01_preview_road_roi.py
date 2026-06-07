from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


INPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")
OUTPUT_CSV = Path("outputs/features/road_dust_features_frame_level_v2.csv")


# Use only lenses with meaningful visible road surface.
ROAD_ROI_POLYGONS = {
    1: [
        (0.65, 0.70),
        (1.00, 0.70),
        (1.00, 0.82),
        (0.55, 0.82),
    ],
    6: [
        (0.00, 0.67),
        (0.35, 0.67),
        (0.45, 0.75),
        (0.00, 0.75),
    ],
}


def resolve_path(path_value: str) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return Path.cwd() / p


def norm_poly_to_pixels(poly, w, h):
    return np.array(
        [[int(x * w), int(y * h)] for x, y in poly],
        dtype=np.int32,
    )


def build_roi_mask(image_shape, lens_id: int):
    h, w = image_shape[:2]

    if lens_id not in ROAD_ROI_POLYGONS:
        return None

    pts = norm_poly_to_pixels(ROAD_ROI_POLYGONS[lens_id], w, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    return mask


def extract_features_from_roi(image, mask):
    """
    Extract interpretable visual road-dust/dryness features from ROI.

    Important:
    These are visual proxy features, not ground-truth PM/dust measurements.
    """

    roi_pixels = image[mask > 0]

    if roi_pixels.size == 0:
        return None

    roi_area_pixels = int(len(roi_pixels))

    # Convert to HSV and grayscale.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    roi_hsv = hsv[mask > 0]
    roi_gray = gray[mask > 0]

    h_vals = roi_hsv[:, 0].astype(np.float32)
    s_vals = roi_hsv[:, 1].astype(np.float32)
    v_vals = roi_hsv[:, 2].astype(np.float32)

    # Normalize saturation/value to [0, 1].
    s_norm = s_vals / 255.0
    v_norm = v_vals / 255.0

    mean_brightness = float(v_norm.mean())
    mean_saturation = float(s_norm.mean())
    contrast_std = float(roi_gray.std() / 255.0)

    # Brown/dust-like HSV range.
    # OpenCV hue: 0-179.
    # Dust/soil/road-brown often lies around yellow-orange/brown hue,
    # moderate saturation, moderate-to-high brightness.
    brown_mask = (
        (h_vals >= 8) & (h_vals <= 35) &
        (s_norm >= 0.12) & (s_norm <= 0.75) &
        (v_norm >= 0.20) & (v_norm <= 0.95)
    )
    brown_pixel_ratio = float(brown_mask.mean())

    # Gray/dry road-like pixels:
    # low saturation, medium brightness.
    gray_dry_mask = (
        (s_norm <= 0.25) &
        (v_norm >= 0.25) & (v_norm <= 0.85)
    )
    gray_dry_pixel_ratio = float(gray_dry_mask.mean())

    # Very dark pixels can be shadows/black mask contamination.
    dark_pixel_ratio = float((v_norm < 0.15).mean())

    # Very bright pixels can be glare/white vehicles/lane markings.
    bright_pixel_ratio = float((v_norm > 0.90).mean())

    # Edge density within ROI.
    roi_gray_full = gray.copy()
    edges = cv2.Canny(roi_gray_full, 80, 160)
    edge_density = float((edges[mask > 0] > 0).mean())

    # Haze/flatness proxy:
    # Low contrast + high brightness can indicate visual haze/dust,
    # but also overexposure, so this is only a weak proxy.
    haze_score = float(mean_brightness * (1.0 - contrast_std))

    # Dustiness proxy:
    # brown/gray-dry pixels increase score,
    # excessive dark/bright contamination decreases confidence.
    road_dust_score_initial = (
        0.45 * brown_pixel_ratio +
        0.25 * gray_dry_pixel_ratio +
        0.20 * haze_score +
        0.10 * (1.0 - edge_density)
    )

    # Penalize if ROI is dominated by dark masked/shadow regions or glare.
    contamination_penalty = 0.5 * dark_pixel_ratio + 0.3 * bright_pixel_ratio
    road_dust_score_initial = road_dust_score_initial * (1.0 - contamination_penalty)

    road_dust_score_initial = float(np.clip(road_dust_score_initial, 0.0, 1.0))

    return {
        "road_roi_area_pixels": roi_area_pixels,
        "road_brown_pixel_ratio": brown_pixel_ratio,
        "road_gray_dry_pixel_ratio": gray_dry_pixel_ratio,
        "road_dark_pixel_ratio": dark_pixel_ratio,
        "road_bright_pixel_ratio": bright_pixel_ratio,
        "road_mean_brightness": mean_brightness,
        "road_mean_saturation": mean_saturation,
        "road_contrast_std": contrast_std,
        "road_edge_density": edge_density,
        "road_haze_score": haze_score,
        "road_dust_score_initial": road_dust_score_initial,
    }


def process_row(row):
    lens_id = int(row["lens_id"])

    result = {
        "sample_index": row.get("sample_index", None),
        "sensor_timestamp": row.get("sensor_timestamp", ""),
        "sample_unix": row.get("sample_unix", None),
        "lens_id": lens_id,
        "matched_run_id": row.get("matched_run_id", ""),
        "video_offset_sec": row.get("video_offset_sec", None),
        "processed_frame_key": row.get("processed_frame_key", ""),
        "processed_frame_path": row.get("processed_frame_path", ""),
        "road_feature_status": "failed",
        "road_feature_error": "",
    }

    if lens_id not in ROAD_ROI_POLYGONS:
        result["road_feature_error"] = "lens_not_used_for_road_features"
        return result

    frame_path = resolve_path(row["processed_frame_path"])

    if not frame_path.exists():
        result["road_feature_error"] = f"frame_not_found: {frame_path}"
        return result

    image = cv2.imread(str(frame_path))

    if image is None:
        result["road_feature_error"] = f"could_not_read_frame: {frame_path}"
        return result

    mask = build_roi_mask(image.shape, lens_id)

    if mask is None:
        result["road_feature_error"] = "roi_mask_missing"
        return result

    features = extract_features_from_roi(image, mask)

    if features is None:
        result["road_feature_error"] = "empty_roi"
        return result

    result.update(features)
    result["road_feature_status"] = "success"
    result["road_feature_error"] = ""

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-manifest", default=str(INPUT_MANIFEST))
    parser.add_argument("--output-csv", default=str(OUTPUT_CSV))
    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 6])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    output_csv = Path(args.output_csv)

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    df = pd.read_csv(input_manifest)

    if "preprocess_status" in df.columns:
        df = df[df["preprocess_status"] == "success"].copy()

    df = df[df["lens_id"].isin(args.lenses)].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("\nRoad dust feature extraction")
    print("=" * 60)
    print("Input manifest:", input_manifest)
    print("Output CSV:", output_csv)
    print("Rows selected:", len(df))
    print("Lenses used:", args.lenses)

    rows = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"[{i}/{len(df)}] lens={row['lens_id']} frame={row['processed_frame_key']}")
        rows.append(process_row(row))

    out = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("\nSaved:", output_csv)
    print("Shape:", out.shape)

    print("\nStatus:")
    print(out["road_feature_status"].value_counts(dropna=False))

    print("\nRows by lens:")
    print(out.groupby("lens_id")["road_feature_status"].value_counts())

    if "road_dust_score_initial" in out.columns:
        print("\nRoad dust score stats:")
        print(out["road_dust_score_initial"].describe())


if __name__ == "__main__":
    main()