from pathlib import Path
import argparse
import traceback

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


def resolve_path(path_value, project_root):
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return project_root / p


def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)


def depth_to_points(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1)


def estimate_surface_area_from_depth(
    depth,
    road_mask,
    fx,
    fy,
    min_depth_m=0.5,
    max_depth_m=30.0,
    erode_mask=True,
):
    if depth.shape != road_mask.shape:
        raise ValueError(
            f"Depth shape {depth.shape} and mask shape {road_mask.shape} do not match."
        )

    h, w = depth.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    valid_depth = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    usable_mask = road_mask & valid_depth

    if erode_mask:
        kernel = np.ones((3, 3), np.uint8)
        usable_mask = cv2.erode(
            usable_mask.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)

    points = depth_to_points(depth, fx, fy, cx, cy)

    p00 = points[:-1, :-1]
    p01 = points[:-1, 1:]
    p10 = points[1:, :-1]
    p11 = points[1:, 1:]

    m00 = usable_mask[:-1, :-1]
    m01 = usable_mask[:-1, 1:]
    m10 = usable_mask[1:, :-1]
    m11 = usable_mask[1:, 1:]

    quad_valid = m00 & m01 & m10 & m11

    area_1 = triangle_area(p00, p10, p01)
    area_2 = triangle_area(p10, p11, p01)

    quad_area = area_1 + area_2
    total_area_m2 = float(np.sum(quad_area[quad_valid]))

    return {
        "estimated_road_area_m2": total_area_m2,
        "road_pixels_used_for_area": int(usable_mask.sum()),
        "valid_road_quads_used": int(quad_valid.sum()),
        "fx_px": float(fx),
        "fy_px": float(fy),
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
    }


def generate_road_mask(image_pil, road_processor, road_model, device, threshold=0.5):
    original_w, original_h = image_pil.size

    inputs = road_processor(images=image_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = road_model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    )

    if upsampled_logits.shape[1] == 1:
        probs = torch.sigmoid(upsampled_logits)[0, 0].detach().cpu().numpy()
        mask = probs >= threshold
    else:
        pred = upsampled_logits.argmax(dim=1)[0].detach().cpu().numpy()
        mask = pred == 1

    return mask


def generate_metric_depth(image_pil, depth_processor, depth_model, device):
    original_w, original_h = image_pil.size

    inputs = depth_processor(images=image_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = depth_model(**inputs)

    predicted_depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_h, original_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_m = depth.detach().cpu().numpy().astype(np.float32)

    return depth_m


def resize_for_area(depth, mask, target_width, fx, fy):
    if target_width is None:
        return depth, mask, fx, fy

    h, w = depth.shape

    if w <= target_width:
        return depth, mask, fx, fy

    scale = target_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    depth_resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(
        mask.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    fx_resized = fx * scale
    fy_resized = fy * scale

    return depth_resized, mask_resized, fx_resized, fy_resized


def save_depth_preview(depth_m, output_path):
    valid = np.isfinite(depth_m)
    if valid.sum() == 0:
        return

    lo = np.percentile(depth_m[valid], 2)
    hi = np.percentile(depth_m[valid], 98)

    depth_norm = (depth_m - lo) / max(hi - lo, 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)

    preview = (depth_norm * 255).astype(np.uint8)
    preview_color = cv2.applyColorMap(preview, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(output_path), preview_color)


def make_quality_flag(row):
    if row["road_area_ratio_px"] < 0.03:
        return "bad_low_road_mask_area"

    if row["road_pixels_used_for_area"] < 1000:
        return "bad_too_few_valid_road_pixels"

    if not np.isfinite(row["estimated_road_area_m2"]):
        return "bad_invalid_area"

    if row["road_depth_p95_m"] <= row["road_depth_p5_m"]:
        return "bad_depth_distribution"

    if row["road_depth_p50_m"] < 0.5:
        return "bad_unphysical_near_depth"

    return "needs_visual_review"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="outputs/features/processed_frame_manifest_preprocessed_v2.csv",
    )

    parser.add_argument(
        "--lens-id",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--road-model-path",
        default="models/road_segmentation/best_segformer_b0_idd_binary_road/best_segformer_b0_idd_binary_road",
    )

    parser.add_argument(
        "--depth-model-id",
        default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    )

    parser.add_argument(
        "--fx",
        type=float,
        required=True,
        help="Lens-specific calibrated fx in pixels.",
    )

    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="Lens-specific calibrated fy in pixels. If omitted, fy = fx.",
    )

    parser.add_argument("--min-depth-m", type=float, default=0.5)
    parser.add_argument("--max-depth-m", type=float, default=30.0)

    parser.add_argument(
        "--area-width",
        type=int,
        default=1024,
        help="Downscale width for area calculation. Use 0 for full resolution.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every nth lens-1 frame. Example: 5 means process every 5th row.",
    )

    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save road masks to disk.",
    )

    parser.add_argument(
        "--save-depth-npy",
        action="store_true",
        help="Save depth .npy files. Warning: can consume a lot of storage.",
    )

    parser.add_argument(
        "--save-depth-preview",
        action="store_true",
        help="Save colored depth preview images.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/road_area_lens1_batch",
    )

    parser.add_argument(
        "--output-csv",
        default="outputs/road_area_lens1_batch/lens1_depth_estimated_road_area.csv",
    )

    args = parser.parse_args()

    project_root = Path(".").resolve()
    output_dir = Path(args.output_dir)
    output_csv = Path(args.output_csv)

    mask_dir = output_dir / "masks"
    depth_dir = output_dir / "depth"
    preview_dir = output_dir / "depth_previews"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    if args.save_depth_npy:
        depth_dir.mkdir(parents=True, exist_ok=True)

    if args.save_depth_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    fy = args.fy if args.fy is not None else args.fx
    area_width = None if args.area_width == 0 else args.area_width

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Lens ID:", args.lens_id)
    print("fx:", args.fx)
    print("fy:", fy)
    print("Area calculation width:", area_width if area_width else "full resolution")

    print("\nLoading road segmentation model...")
    road_processor = SegformerImageProcessor.from_pretrained(args.road_model_path)
    road_model = SegformerForSemanticSegmentation.from_pretrained(args.road_model_path)
    road_model.to(device)
    road_model.eval()

    print("\nLoading depth model...")
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(args.depth_model_id)
    depth_model.to(device)
    depth_model.eval()

    manifest_path = project_root / args.manifest
    df = pd.read_csv(manifest_path)

    df = df[
        (df["lens_id"] == args.lens_id)
        & (df["preprocess_status"] == "success")
    ].copy()

    df = df.sort_values(["sample_index", "video_offset_sec"]).reset_index(drop=True)

    if args.stride > 1:
        df = df.iloc[::args.stride].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("\nFrames to process:", len(df))

    rows = []

    for i, row in df.iterrows():
        sample_index = int(row["sample_index"])
        lens_id = int(row["lens_id"])

        out_id = f"sample_{sample_index:05d}_lens_{lens_id}_row_{i:06d}"

        result = row.to_dict()
        result["batch_row_index"] = i
        result["area_status"] = "started"
        result["area_error"] = ""

        try:
            image_path = resolve_path(row["processed_frame_path"], project_root)

            if not image_path.exists():
                raise FileNotFoundError(f"Processed image not found: {image_path}")

            print(f"\n[{i + 1}/{len(df)}] {image_path}")

            image_pil = Image.open(image_path).convert("RGB")
            image_w, image_h = image_pil.size

            result["area_image_width_px"] = image_w
            result["area_image_height_px"] = image_h
            result["area_input_image_path"] = str(image_path)

            road_mask = generate_road_mask(
                image_pil=image_pil,
                road_processor=road_processor,
                road_model=road_model,
                device=device,
                threshold=0.5,
            )

            depth_m = generate_metric_depth(
                image_pil=image_pil,
                depth_processor=depth_processor,
                depth_model=depth_model,
                device=device,
            )

            if depth_m.shape != road_mask.shape:
                depth_m = cv2.resize(
                    depth_m,
                    (road_mask.shape[1], road_mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            total_pixels = road_mask.size
            road_pixels = int(road_mask.sum())
            road_area_ratio = road_pixels / total_pixels if total_pixels > 0 else np.nan

            road_depth = depth_m[road_mask]
            road_depth = road_depth[np.isfinite(road_depth)]

            if len(road_depth) > 0:
                result["road_depth_p5_m"] = float(np.percentile(road_depth, 5))
                result["road_depth_p25_m"] = float(np.percentile(road_depth, 25))
                result["road_depth_p50_m"] = float(np.percentile(road_depth, 50))
                result["road_depth_p75_m"] = float(np.percentile(road_depth, 75))
                result["road_depth_p95_m"] = float(np.percentile(road_depth, 95))
                result["road_depth_std_m"] = float(np.std(road_depth))
            else:
                result["road_depth_p5_m"] = np.nan
                result["road_depth_p25_m"] = np.nan
                result["road_depth_p50_m"] = np.nan
                result["road_depth_p75_m"] = np.nan
                result["road_depth_p95_m"] = np.nan
                result["road_depth_std_m"] = np.nan

            depth_for_area, mask_for_area, fx_for_area, fy_for_area = resize_for_area(
                depth=depth_m,
                mask=road_mask,
                target_width=area_width,
                fx=args.fx,
                fy=fy,
            )

            area_result = estimate_surface_area_from_depth(
                depth=depth_for_area,
                road_mask=mask_for_area,
                fx=fx_for_area,
                fy=fy_for_area,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                erode_mask=True,
            )

            result.update(area_result)

            result["road_area_ratio_px"] = float(road_area_ratio)
            result["road_area_percent_px"] = float(road_area_ratio * 100.0)
            result["raw_road_pixels"] = road_pixels
            result["raw_total_pixels"] = int(total_pixels)
            result["area_calc_width_px"] = int(depth_for_area.shape[1])
            result["area_calc_height_px"] = int(depth_for_area.shape[0])
            result["area_fx_after_resize_px"] = float(fx_for_area)
            result["area_fy_after_resize_px"] = float(fy_for_area)

            if args.save_masks:
                mask_path = mask_dir / f"{out_id}_road_mask.png"
                cv2.imwrite(str(mask_path), (road_mask.astype(np.uint8) * 255))
                result["saved_road_mask_path"] = str(mask_path)

            if args.save_depth_npy:
                depth_path = depth_dir / f"{out_id}_depth_m.npy"
                np.save(depth_path, depth_m.astype(np.float32))
                result["saved_depth_npy_path"] = str(depth_path)

            if args.save_depth_preview:
                preview_path = preview_dir / f"{out_id}_depth_preview.png"
                save_depth_preview(depth_m, preview_path)
                result["saved_depth_preview_path"] = str(preview_path)

            result["area_quality_flag"] = make_quality_flag(result)
            result["area_status"] = "success"

            print(
                f"Area = {result['estimated_road_area_m2']:.3f} m² | "
                f"road% = {result['road_area_percent_px']:.2f}% | "
                f"quality = {result['area_quality_flag']}"
            )

        except Exception as exc:
            result["area_status"] = "error"
            result["area_error"] = str(exc)
            result["area_traceback"] = traceback.format_exc()

            print("[ERROR]", exc)

        rows.append(result)

        pd.DataFrame(rows).to_csv(output_csv, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    print("\nSaved:")
    print(output_csv)

    print("\nStatus counts:")
    print(out["area_status"].value_counts(dropna=False))

    if "estimated_road_area_m2" in out.columns:
        ok = out[out["area_status"] == "success"].copy()
        if len(ok) > 0:
            print("\nEstimated road area summary:")
            print(ok["estimated_road_area_m2"].describe())

            print("\nQuality flags:")
            print(ok["area_quality_flag"].value_counts(dropna=False))


if __name__ == "__main__":
    main()