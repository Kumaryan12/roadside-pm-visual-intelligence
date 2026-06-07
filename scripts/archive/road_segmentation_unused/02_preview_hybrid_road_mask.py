from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


DEFAULT_INPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/figures/hybrid_road_mask_previews")

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-768-768"


# Manual road ROI polygons on preprocessed frames.
# Coordinates are normalized: (x_ratio, y_ratio)
# We use only LENS1 and LENS6 for road condition.
MANUAL_ROI_POLYGONS = {
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


def build_manual_roi_mask(image_shape, lens_id: int):
    h, w = image_shape[:2]

    if lens_id not in MANUAL_ROI_POLYGONS:
        return None

    pts = norm_poly_to_pixels(MANUAL_ROI_POLYGONS[lens_id], w, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    return mask


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


def make_segformer_road_mask(pred, id2label):
    road_ids = []

    for class_id, label in id2label.items():
        if str(label).lower() == "road":
            road_ids.append(int(class_id))

    if not road_ids:
        raise ValueError(f"No road class found in labels: {id2label}")

    mask = np.isin(pred, road_ids).astype(np.uint8) * 255
    return mask


def make_hybrid_mask(manual_mask, segformer_mask):
    # Intersection: pixel must be inside manual ROI AND predicted road.
    hybrid = cv2.bitwise_and(manual_mask, segformer_mask)

    # Clean tiny speckles.
    kernel = np.ones((5, 5), np.uint8)
    hybrid = cv2.morphologyEx(hybrid, cv2.MORPH_OPEN, kernel)
    hybrid = cv2.morphologyEx(hybrid, cv2.MORPH_CLOSE, kernel)

    return hybrid


def overlay_mask(image_bgr, mask, color=(0, 255, 0), alpha=0.35):
    overlay = image_bgr.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return blended


def draw_manual_roi_boundary(image_bgr, lens_id):
    if lens_id not in MANUAL_ROI_POLYGONS:
        return image_bgr

    h, w = image_bgr.shape[:2]
    pts = norm_poly_to_pixels(MANUAL_ROI_POLYGONS[lens_id], w, h)

    out = image_bgr.copy()
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 255), thickness=3)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 6])
    parser.add_argument("--per-lens", type=int, default=5)

    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    df = pd.read_csv(input_manifest)
    df = df[df["preprocess_status"] == "success"].copy()
    df = df[df["lens_id"].isin(args.lenses)].copy()

    preview_rows = []

    for lens_id in args.lenses:
        lens_df = df[df["lens_id"] == lens_id].copy()

        if len(lens_df) == 0:
            continue

        indices = np.linspace(
            0,
            len(lens_df) - 1,
            num=min(args.per_lens, len(lens_df)),
            dtype=int,
        )

        preview_rows.append(lens_df.iloc[indices])

    preview_df = pd.concat(preview_rows, ignore_index=True)

    processor, model, id2label, device = load_model()

    for _, row in preview_df.iterrows():
        lens_id = int(row["lens_id"])
        frame_key = str(row["processed_frame_key"])
        frame_path = resolve_path(row["processed_frame_path"])

        image = cv2.imread(str(frame_path))

        if image is None:
            print("Could not read:", frame_path)
            continue

        manual_mask = build_manual_roi_mask(image.shape, lens_id)

        if manual_mask is None:
            print("No manual ROI for lens:", lens_id)
            continue

        pred = segment_image(image, processor, model, device)
        segformer_mask = make_segformer_road_mask(pred, id2label)
        hybrid_mask = make_hybrid_mask(manual_mask, segformer_mask)

        manual_ratio = float((manual_mask > 0).mean())
        seg_ratio = float((segformer_mask > 0).mean())
        hybrid_ratio = float((hybrid_mask > 0).mean())

        manual_preview = overlay_mask(image, manual_mask, color=(0, 255, 255), alpha=0.25)
        seg_preview = overlay_mask(image, segformer_mask, color=(0, 255, 0), alpha=0.35)
        hybrid_preview = overlay_mask(image, hybrid_mask, color=(0, 255, 0), alpha=0.45)
        hybrid_preview = draw_manual_roi_boundary(hybrid_preview, lens_id)

        base_name = f"{frame_key}_lens{lens_id}"

        cv2.imwrite(str(output_dir / f"{base_name}_manual_roi.jpg"), manual_preview)
        cv2.imwrite(str(output_dir / f"{base_name}_segformer_road.jpg"), seg_preview)
        cv2.imwrite(str(output_dir / f"{base_name}_hybrid_road.jpg"), hybrid_preview)
        cv2.imwrite(str(output_dir / f"{base_name}_hybrid_mask.png"), hybrid_mask)

        print(
            "Saved previews for:",
            base_name,
            "| manual_ratio:",
            round(manual_ratio, 4),
            "| seg_ratio:",
            round(seg_ratio, 4),
            "| hybrid_ratio:",
            round(hybrid_ratio, 4),
        )


if __name__ == "__main__":
    main()