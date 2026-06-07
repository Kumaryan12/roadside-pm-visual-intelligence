from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


DEFAULT_INPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")
DEFAULT_OUTPUT_CSV = Path("outputs/features/segformer_road_condition_features_frame_level_v2.csv")


from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_NAME = "models/road_segmentation/best_segformer_b0_idd_binary_road/best_segformer_b0_idd_binary_road"

# Load original processor from Hugging Face
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768")

# Load your fine-tuned weights into model
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    ignore_mismatched_sizes=True  # if your output classes changed
)
# Same artificial platform masks that were applied during preprocessing.
# Coordinates are normalized: (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
PLATFORM_MASK_CONFIG = {
    1: (0.00, 0.82, 1.00, 1.00),
    4: (0.00, 0.95, 1.00, 1.00),
    6: (0.00, 0.75, 1.00, 1.00),
}


def resolve_path(path_value: str) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return Path.cwd() / p


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    id2label = model.config.id2label

    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("Labels:", id2label)

    return processor, model, id2label, device


def segment_image(image_bgr, processor, model, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )

    pred = upsampled_logits.argmax(dim=1)[0].detach().cpu().numpy()

    return pred


def make_road_mask(pred, id2label):
    road_ids = []

    for class_id, label in id2label.items():
        if str(label).lower() == "road":
            road_ids.append(int(class_id))

    if not road_ids:
        raise ValueError(f"No road class found in labels: {id2label}")

    mask = np.isin(pred, road_ids).astype(np.uint8) * 255

    # Clean tiny noise.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask



def build_platform_exclusion_mask(image_shape, lens_id: int):
    """
    Returns a binary mask where valid pixels are 255 and the artificial
    platform/black-mask region is 0.
    """
    h, w = image_shape[:2]
    valid_mask = np.ones((h, w), dtype=np.uint8) * 255

    if lens_id not in PLATFORM_MASK_CONFIG:
        return valid_mask

    x1r, y1r, x2r, y2r = PLATFORM_MASK_CONFIG[lens_id]

    x1 = int(x1r * w)
    y1 = int(y1r * h)
    x2 = int(x2r * w)
    y2 = int(y2r * h)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    valid_mask[y1:y2, x1:x2] = 0

    return valid_mask


def clean_road_mask(image_bgr, road_mask, lens_id: int):
    """
    Removes artificial black/platform areas from the SegFormer road mask.

    This keeps the pretrained model output, but prevents the artificial black
    region created during preprocessing from being counted as road-condition pixels.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]

    # Remove near-black pixels from artificial masks.
    # Threshold is conservative so normal shadows are mostly preserved.
    non_black_mask = (value_channel > 25).astype(np.uint8) * 255

    # Remove the known lens-specific platform-mask region.
    platform_valid_mask = build_platform_exclusion_mask(
        image_shape=image_bgr.shape,
        lens_id=lens_id,
    )

    cleaned = cv2.bitwise_and(road_mask, non_black_mask)
    cleaned = cv2.bitwise_and(cleaned, platform_valid_mask)

    return cleaned


def extract_road_features(image_bgr, road_mask):
    road_pixels = image_bgr[road_mask > 0]

    if road_pixels.size == 0:
        return None

    h, w = image_bgr.shape[:2]
    total_pixels = h * w
    road_area_pixels = int(len(road_pixels))
    road_area_ratio = float(road_area_pixels / total_pixels)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    roi_hsv = hsv[road_mask > 0]
    roi_gray = gray[road_mask > 0]

    h_vals = roi_hsv[:, 0].astype(np.float32)
    s_vals = roi_hsv[:, 1].astype(np.float32) / 255.0
    v_vals = roi_hsv[:, 2].astype(np.float32) / 255.0

    road_mean_brightness = float(v_vals.mean())
    road_mean_saturation = float(s_vals.mean())
    road_contrast_std = float(roi_gray.std() / 255.0)

    # Color/appearance descriptors, not dust labels.
    road_shadow_ratio = float((v_vals < 0.22).mean())
    road_glare_ratio = float((v_vals > 0.90).mean())

    # Brownish/yellowish road/soil-like pixels.
    road_brown_pixel_ratio = float((
        (h_vals >= 8) & (h_vals <= 35) &
        (s_vals >= 0.10) & (s_vals <= 0.80) &
        (v_vals >= 0.18) & (v_vals <= 0.95)
    ).mean())

    # Gray-dry asphalt-like low-saturation road pixels.
    road_gray_dry_pixel_ratio = float((
        (s_vals <= 0.28) &
        (v_vals >= 0.25) & (v_vals <= 0.85)
    ).mean())

    # Edge density inside road mask.
    edges = cv2.Canny(gray, 80, 160)
    road_edge_density = float((edges[road_mask > 0] > 0).mean())

    # Texture roughness proxy using Laplacian variance.
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    road_laplacian_std = float(lap[road_mask > 0].std() / 255.0)

    # Weak visibility/flatness descriptor, not a dust score.
    road_haze_flatness_proxy = float(road_mean_brightness * (1.0 - road_contrast_std))

    return {
        "road_area_pixels": road_area_pixels,
        "road_area_ratio": road_area_ratio,
        "road_mean_brightness": road_mean_brightness,
        "road_mean_saturation": road_mean_saturation,
        "road_contrast_std": road_contrast_std,
        "road_shadow_ratio": road_shadow_ratio,
        "road_glare_ratio": road_glare_ratio,
        "road_brown_pixel_ratio": road_brown_pixel_ratio,
        "road_gray_dry_pixel_ratio": road_gray_dry_pixel_ratio,
        "road_edge_density": road_edge_density,
        "road_laplacian_std": road_laplacian_std,
        "road_haze_flatness_proxy": road_haze_flatness_proxy,
    }


def process_row(row, processor, model, id2label, device):
    result = {
        "sample_index": row.get("sample_index", None),
        "sensor_timestamp": row.get("sensor_timestamp", ""),
        "sample_unix": row.get("sample_unix", None),
        "lens_id": int(row.get("lens_id")),
        "matched_run_id": row.get("matched_run_id", ""),
        "video_offset_sec": row.get("video_offset_sec", None),
        "processed_frame_key": row.get("processed_frame_key", ""),
        "processed_frame_path": row.get("processed_frame_path", ""),
        "road_condition_status": "failed",
        "road_condition_error": "",
        "segmentation_model": MODEL_NAME,
    }

    frame_path = resolve_path(row["processed_frame_path"])

    if not frame_path.exists():
        result["road_condition_error"] = f"frame_not_found: {frame_path}"
        return result

    image = cv2.imread(str(frame_path))

    if image is None:
        result["road_condition_error"] = f"could_not_read_frame: {frame_path}"
        return result

    try:
        pred = segment_image(image, processor, model, device)
        road_mask = make_road_mask(pred, id2label)
        road_mask = clean_road_mask(
            image_bgr=image,
            road_mask=road_mask,
            lens_id=int(row.get("lens_id")),
        )

        features = extract_road_features(image, road_mask)

        if features is None:
            result["road_condition_error"] = "empty_road_mask"
            return result

        result.update(features)
        result["road_condition_status"] = "success"
        return result

    except Exception as exc:
        result["road_condition_error"] = str(exc)
        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 6])
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    output_csv = Path(args.output_csv)

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    df = pd.read_csv(input_manifest)
    df = df[df["preprocess_status"] == "success"].copy()
    df = df[df["lens_id"].isin(args.lenses)].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("\nSegFormer road-condition feature extraction")
    print("=" * 60)
    print("Input manifest:", input_manifest)
    print("Output CSV:", output_csv)
    print("Rows selected:", len(df))
    print("Lenses used:", args.lenses)

    processor, model, id2label, device = load_model()

    rows = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"[{i}/{len(df)}] lens={row['lens_id']} frame={row['processed_frame_key']}")
        rows.append(process_row(row, processor, model, id2label, device))

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("\nSaved:", output_csv)
    print("Shape:", out.shape)

    print("\nStatus:")
    print(out["road_condition_status"].value_counts(dropna=False))

    print("\nRows by lens:")
    print(out.groupby("lens_id")["road_condition_status"].value_counts())

    if "road_area_ratio" in out.columns:
        print("\nRoad area ratio stats:")
        print(out["road_area_ratio"].describe())

    if "road_brown_pixel_ratio" in out.columns:
        print("\nRoad feature stats:")
        print(out[[
            "road_area_ratio",
            "road_mean_brightness",
            "road_mean_saturation",
            "road_contrast_std",
            "road_brown_pixel_ratio",
            "road_gray_dry_pixel_ratio",
            "road_edge_density",
            "road_laplacian_std",
            "road_haze_flatness_proxy",
        ]].describe().T)


if __name__ == "__main__":
    main()