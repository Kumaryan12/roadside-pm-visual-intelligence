from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


DEFAULT_INPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/figures/road_segmentation_previews_cleaned")

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
    path = Path(str(path_value))

    if path.is_absolute():
        return path

    return Path.cwd() / path


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
    inputs = {key: value.to(device) for key, value in inputs.items()}

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

    road_mask = np.isin(pred, road_ids).astype(np.uint8) * 255

    # Light cleanup for tiny speckles.
    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

    return road_mask


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


def cleaned_road_mask(image_bgr, road_mask, lens_id: int):
    """
    Removes artificial black/platform areas from the SegFormer road mask.

    This keeps the pretrained model output, but prevents the artificial black
    region created during preprocessing from being counted as road.
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


def overlay_mask(image_bgr, mask):
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 255, 0)

    blended = cv2.addWeighted(
        overlay,
        0.35,
        image_bgr,
        0.65,
        0,
    )

    return blended


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-manifest",
        default=str(DEFAULT_INPUT_MANIFEST),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--lenses",
        nargs="+",
        type=int,
        default=[1, 6],
    )

    parser.add_argument(
        "--per-lens",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    df = pd.read_csv(input_manifest)

    if "preprocess_status" not in df.columns:
        raise ValueError("Input manifest missing required column: preprocess_status")

    if "lens_id" not in df.columns:
        raise ValueError("Input manifest missing required column: lens_id")

    df = df[df["preprocess_status"] == "success"].copy()
    df["lens_id"] = pd.to_numeric(df["lens_id"], errors="coerce")
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

    if not preview_rows:
        print("No preview rows found for selected lenses.")
        print("Check whether the input manifest has successful rows for lenses:", args.lenses)
        return

    preview_df = pd.concat(preview_rows, ignore_index=True)

    print("\nSegFormer pretrained road preview")
    print("=" * 60)
    print("Input manifest:", input_manifest)
    print("Output dir:", output_dir)
    print("Rows selected:", len(preview_df))
    print("Lenses:", args.lenses)

    processor, model, id2label, device = load_model()

    for _, row in preview_df.iterrows():
        lens_id = int(row["lens_id"])
        frame_key = str(row["processed_frame_key"])
        frame_path = resolve_path(row["processed_frame_path"])

        image = cv2.imread(str(frame_path))

        if image is None:
            print("Could not read:", frame_path)
            continue

        pred = segment_image(
            image_bgr=image,
            processor=processor,
            model=model,
            device=device,
        )

        road_mask = make_road_mask(
            pred=pred,
            id2label=id2label,
        )

        cleaned_mask = cleaned_road_mask(
            image_bgr=image,
            road_mask=road_mask,
            lens_id=lens_id,
        )

        cleaned_preview = overlay_mask(
            image_bgr=image,
            mask=cleaned_mask,
        )

        cleaned_road_ratio = float((cleaned_mask > 0).mean())

        base_name = f"{frame_key}_lens{lens_id}"

        preview_path = output_dir / f"{base_name}_roadseg_cleaned_ratio_{cleaned_road_ratio:.3f}.jpg"
        mask_path = output_dir / f"{base_name}_roadmask_cleaned.png"

        cv2.imwrite(str(preview_path), cleaned_preview)
        cv2.imwrite(str(mask_path), cleaned_mask)

        print(
            "Saved:",
            preview_path,
            "cleaned_ratio:",
            round(cleaned_road_ratio, 4),
        )


if __name__ == "__main__":
    main()