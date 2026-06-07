from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


DEFAULT_INPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")

DEFAULT_MODEL_PATH = Path(
    "models/road_segmentation/"
    "best_segformer_b0_idd_binary_road/"
    "best_segformer_b0_idd_binary_road"
)

FALLBACK_PROCESSOR = "nvidia/segformer-b0-finetuned-cityscapes-768-768"

DEFAULT_OUTPUT_DIR = Path("outputs/validation/road_feature_validation_pack_v1")
DEFAULT_OUTPUT_CSV = Path("outputs/validation/road_feature_validation_pack_v1/manual_label_sheet.csv")


def resolve_path(path_value):
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return Path.cwd() / p


def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = SegformerImageProcessor.from_pretrained(str(model_path))
    except Exception:
        print("Could not load local processor. Using fallback processor:", FALLBACK_PROCESSOR)
        processor = SegformerImageProcessor.from_pretrained(FALLBACK_PROCESSOR)

    model = SegformerForSemanticSegmentation.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    id2label = model.config.id2label

    print("Device:", device)
    print("Model path:", model_path)
    print("num_labels:", model.config.num_labels)
    print("id2label:", id2label)

    return processor, model, id2label, device


def get_road_class_ids(id2label, num_labels):
    road_ids = []

    for class_id, label in id2label.items():
        if str(label).lower() == "road":
            road_ids.append(int(class_id))

    if not road_ids and num_labels == 2:
        road_ids = [1]

    if not road_ids:
        raise ValueError(f"Could not identify road class from id2label={id2label}")

    return road_ids


def segment_road(image_bgr, processor, model, id2label, device):
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

    road_ids = get_road_class_ids(
        id2label=id2label,
        num_labels=model.config.num_labels,
    )

    road_mask = np.isin(pred, road_ids).astype(np.uint8) * 255

    return road_mask


def clean_road_mask(image_bgr, road_mask, black_threshold=25, morph_kernel_size=5):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    non_black_mask = (value > black_threshold).astype(np.uint8) * 255
    cleaned = cv2.bitwise_and(road_mask, non_black_mask)

    if morph_kernel_size > 0:
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


def make_overlay(image_bgr, mask, color, alpha=0.50):
    overlay = image_bgr.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)
    return blended


def extract_features_and_masks(
    image_bgr,
    road_mask,
    brown_h_min=8,
    brown_h_max=35,
    brown_s_min=0.10,
    brown_s_max=0.80,
    brown_v_min=0.18,
    brown_v_max=0.95,
    gray_s_max=0.28,
    gray_v_min=0.25,
    gray_v_max=0.85,
):
    h_img, w_img = image_bgr.shape[:2]
    total_pixels = h_img * w_img

    road_region = road_mask > 0
    road_pixels = int(road_region.sum())

    if road_pixels == 0:
        return None, None, None, None, None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    roi_hsv = hsv[road_region]
    roi_gray = gray[road_region]

    h_vals = roi_hsv[:, 0].astype(np.float32)
    s_vals = roi_hsv[:, 1].astype(np.float32) / 255.0
    v_vals = roi_hsv[:, 2].astype(np.float32) / 255.0

    brown_condition = (
        (h_vals >= brown_h_min)
        & (h_vals <= brown_h_max)
        & (s_vals >= brown_s_min)
        & (s_vals <= brown_s_max)
        & (v_vals >= brown_v_min)
        & (v_vals <= brown_v_max)
    )

    gray_dry_condition = (
        (s_vals <= gray_s_max)
        & (v_vals >= gray_v_min)
        & (v_vals <= gray_v_max)
    )

    brown_mask = np.zeros_like(road_mask, dtype=np.uint8)
    gray_dry_mask = np.zeros_like(road_mask, dtype=np.uint8)

    road_indices = np.where(road_region)
    brown_mask[road_indices] = brown_condition.astype(np.uint8) * 255
    gray_dry_mask[road_indices] = gray_dry_condition.astype(np.uint8) * 255

    edges = cv2.Canny(gray, 80, 160)
    edge_mask = np.zeros_like(road_mask, dtype=np.uint8)
    edge_mask[road_region] = edges[road_region]

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    road_area_ratio = road_pixels / total_pixels
    road_area_percent = road_area_ratio * 100.0

    road_mean_brightness = float(v_vals.mean())
    road_mean_saturation = float(s_vals.mean())
    road_contrast_std = float(roi_gray.std() / 255.0)

    road_brown_pixel_ratio = float(brown_condition.mean())
    road_gray_dry_pixel_ratio = float(gray_dry_condition.mean())

    road_edge_density = float((edges[road_region] > 0).mean())
    road_laplacian_std = float(lap[road_region].std() / 255.0)

    road_haze_flatness_proxy = float(
        road_mean_brightness * (1.0 - road_contrast_std)
    )

    black_ratio_full = float((hsv[:, :, 2] < 25).mean())
    glare_ratio_full = float((hsv[:, :, 2] > 240).mean())

    road_shadow_ratio = float((v_vals < 0.22).mean())
    road_glare_ratio = float((v_vals > 0.90).mean())

    road_area_too_low = road_area_ratio < 0.02
    road_area_too_high = road_area_ratio > 0.60
    black_ratio_too_high = black_ratio_full > 0.35
    glare_ratio_too_high = glare_ratio_full > 0.10
    road_shadow_too_high = road_shadow_ratio > 0.50
    road_glare_too_high = road_glare_ratio > 0.25

    road_feature_reliable = not (
        road_area_too_low
        or road_area_too_high
        or black_ratio_too_high
        or glare_ratio_too_high
        or road_shadow_too_high
        or road_glare_too_high
    )

    texture_score = np.clip(
        0.5 * (road_edge_density / 0.06)
        + 0.5 * (road_laplacian_std / 0.08),
        0,
        1,
    )

    visual_dust_score = float(
        100.0
        * (
            0.45 * road_brown_pixel_ratio
            + 0.25 * road_gray_dry_pixel_ratio
            + 0.20 * texture_score
            + 0.10 * np.clip(road_haze_flatness_proxy / 0.60, 0, 1)
        )
    )

    features = {
        "road_area_pixels": road_pixels,
        "road_area_ratio": road_area_ratio,
        "road_area_percent": road_area_percent,

        "road_mean_brightness": road_mean_brightness,
        "road_mean_saturation": road_mean_saturation,
        "road_contrast_std": road_contrast_std,

        "road_brown_pixel_ratio": road_brown_pixel_ratio,
        "road_gray_dry_pixel_ratio": road_gray_dry_pixel_ratio,

        "road_edge_density": road_edge_density,
        "road_laplacian_std": road_laplacian_std,
        "road_haze_flatness_proxy": road_haze_flatness_proxy,

        "road_shadow_ratio": road_shadow_ratio,
        "road_glare_ratio": road_glare_ratio,

        "black_ratio_full": black_ratio_full,
        "glare_ratio_full": glare_ratio_full,

        "texture_score_0_1": float(texture_score),
        "visual_dust_score_0_100": visual_dust_score,

        "road_area_too_low": road_area_too_low,
        "road_area_too_high": road_area_too_high,
        "black_ratio_too_high": black_ratio_too_high,
        "glare_ratio_too_high": glare_ratio_too_high,
        "road_shadow_too_high": road_shadow_too_high,
        "road_glare_too_high": road_glare_too_high,
        "road_feature_reliable": road_feature_reliable,
    }

    return features, brown_mask, gray_dry_mask, edge_mask, road_region


def make_contact_sheet(image_bgr, road_overlay, brown_overlay, gray_overlay, edge_overlay):
    panels = [
        ("Original", image_bgr),
        ("Road Mask", road_overlay),
        ("Brown/Soil-like", brown_overlay),
        ("Gray-Dry", gray_overlay),
        ("Edges on Road", edge_overlay),
    ]

    resized_panels = []

    target_w = 360
    target_h = 240

    for title, img in panels:
        img_resized = cv2.resize(img, (target_w, target_h))

        canvas = np.ones((target_h + 35, target_w, 3), dtype=np.uint8) * 255
        canvas[35:, :, :] = img_resized

        cv2.putText(
            canvas,
            title,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        resized_panels.append(canvas)

    sheet = np.hstack(resized_panels)
    return sheet


def select_validation_rows(df, lenses, per_lens, strategy, random_seed):
    selected = []

    rng = np.random.default_rng(random_seed)

    for lens_id in lenses:
        lens_df = df[df["lens_id"] == lens_id].copy()

        if len(lens_df) == 0:
            print(f"No rows for lens {lens_id}")
            continue

        if strategy == "even":
            indices = np.linspace(
                0,
                len(lens_df) - 1,
                num=min(per_lens, len(lens_df)),
                dtype=int,
            )
            selected.append(lens_df.iloc[indices])

        elif strategy == "random":
            n = min(per_lens, len(lens_df))
            indices = rng.choice(len(lens_df), size=n, replace=False)
            selected.append(lens_df.iloc[indices])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    if not selected:
        return pd.DataFrame()

    return pd.concat(selected, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))

    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 6])
    parser.add_argument("--per-lens", type=int, default=50)
    parser.add_argument("--strategy", choices=["even", "random"], default="even")
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--black-threshold", type=int, default=25)
    parser.add_argument("--morph-kernel-size", type=int, default=5)

    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_csv = Path(args.output_csv)

    preview_dir = output_dir / "contact_sheets"
    mask_dir = output_dir / "masks"

    preview_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    df = pd.read_csv(input_manifest)

    if "preprocess_status" in df.columns:
        df = df[df["preprocess_status"] == "success"].copy()

    if "lens_id" not in df.columns:
        raise ValueError("Manifest missing lens_id")

    if "processed_frame_path" not in df.columns:
        raise ValueError("Manifest missing processed_frame_path")

    validation_df = select_validation_rows(
        df=df,
        lenses=args.lenses,
        per_lens=args.per_lens,
        strategy=args.strategy,
        random_seed=args.random_seed,
    )

    if len(validation_df) == 0:
        raise ValueError("No validation rows selected.")

    print("Validation rows selected:", len(validation_df))
    print("Rows by lens:")
    print(validation_df["lens_id"].value_counts())

    processor, model, id2label, device = load_model(model_path)

    rows = []

    for idx, row in validation_df.iterrows():
        sample_index = row.get("sample_index", idx)
        lens_id = int(row["lens_id"])

        frame_key = str(row.get("processed_frame_key", f"row_{idx}"))
        timestamp = str(row.get("sensor_timestamp", row.get("timestamp", "")))

        frame_path = resolve_path(row["processed_frame_path"])

        print(f"[{idx + 1}/{len(validation_df)}] sample={sample_index} lens={lens_id}")

        record = {
            "sample_index": sample_index,
            "timestamp": timestamp,
            "lens_id": lens_id,
            "processed_frame_key": frame_key,
            "processed_frame_path": str(frame_path),
            "validation_status": "failed",
            "validation_error": "",

            "manual_road_mask_quality": "",
            "manual_road_surface_class": "",
            "manual_brown_mask_correct": "",
            "manual_gray_dry_mask_correct": "",
            "manual_notes": "",
        }

        if not frame_path.exists():
            record["validation_error"] = f"frame_not_found: {frame_path}"
            rows.append(record)
            continue

        image = cv2.imread(str(frame_path))

        if image is None:
            record["validation_error"] = f"could_not_read: {frame_path}"
            rows.append(record)
            continue

        try:
            raw_road_mask = segment_road(
                image_bgr=image,
                processor=processor,
                model=model,
                id2label=id2label,
                device=device,
            )

            clean_mask = clean_road_mask(
                image_bgr=image,
                road_mask=raw_road_mask,
                black_threshold=args.black_threshold,
                morph_kernel_size=args.morph_kernel_size,
            )

            features, brown_mask, gray_dry_mask, edge_mask, road_region = extract_features_and_masks(
                image_bgr=image,
                road_mask=clean_mask,
            )

            if features is None:
                record["validation_status"] = "failed"
                record["validation_error"] = "empty_road_mask_after_cleaning"
                rows.append(record)
                continue

            road_overlay = make_overlay(image, clean_mask, color=(0, 255, 0), alpha=0.40)
            brown_overlay = make_overlay(image, brown_mask, color=(0, 140, 255), alpha=0.55)
            gray_overlay = make_overlay(image, gray_dry_mask, color=(180, 180, 180), alpha=0.55)
            edge_overlay = make_overlay(image, edge_mask, color=(255, 0, 255), alpha=0.65)

            contact_sheet = make_contact_sheet(
                image_bgr=image,
                road_overlay=road_overlay,
                brown_overlay=brown_overlay,
                gray_overlay=gray_overlay,
                edge_overlay=edge_overlay,
            )

            safe_name = f"sample_{int(sample_index):04d}_lens{lens_id}_{frame_key}"

            contact_path = preview_dir / f"{safe_name}_validation_sheet.jpg"
            road_mask_path = mask_dir / f"{safe_name}_road_mask.png"
            brown_mask_path = mask_dir / f"{safe_name}_brown_mask.png"
            gray_mask_path = mask_dir / f"{safe_name}_gray_dry_mask.png"
            edge_mask_path = mask_dir / f"{safe_name}_edge_mask.png"

            cv2.imwrite(str(contact_path), contact_sheet)
            cv2.imwrite(str(road_mask_path), clean_mask)
            cv2.imwrite(str(brown_mask_path), brown_mask)
            cv2.imwrite(str(gray_mask_path), gray_dry_mask)
            cv2.imwrite(str(edge_mask_path), edge_mask)

            record.update(features)

            record["contact_sheet_path"] = str(contact_path)
            record["road_mask_path"] = str(road_mask_path)
            record["brown_mask_path"] = str(brown_mask_path)
            record["gray_dry_mask_path"] = str(gray_mask_path)
            record["edge_mask_path"] = str(edge_mask_path)

            record["validation_status"] = "success"

        except Exception as exc:
            record["validation_status"] = "failed"
            record["validation_error"] = str(exc)

        rows.append(record)

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    print("\nSaved manual validation sheet:", output_csv)
    print("Saved contact sheets:", preview_dir)
    print("Saved masks:", mask_dir)

    print("\nStatus:")
    print(out["validation_status"].value_counts(dropna=False))

    if "road_feature_reliable" in out.columns:
        print("\nRoad feature reliability:")
        print(out["road_feature_reliable"].value_counts(dropna=False))

    show_cols = [
        "sample_index",
        "timestamp",
        "lens_id",
        "road_area_percent",
        "road_brown_pixel_ratio",
        "road_gray_dry_pixel_ratio",
        "road_edge_density",
        "road_laplacian_std",
        "road_feature_reliable",
        "contact_sheet_path",
    ]

    show_cols = [c for c in show_cols if c in out.columns]

    print("\nPreview:")
    print(out[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()