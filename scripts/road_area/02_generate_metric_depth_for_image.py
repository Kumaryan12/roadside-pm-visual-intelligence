from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def save_depth_preview(depth_m, output_path):
    depth = depth_m.copy()

    valid = np.isfinite(depth)
    if valid.sum() == 0:
        raise ValueError("No valid depth values for preview.")

    lo = np.percentile(depth[valid], 2)
    hi = np.percentile(depth[valid], 98)

    depth_norm = (depth - lo) / max(hi - lo, 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)

    preview = (depth_norm * 255).astype(np.uint8)
    preview_color = cv2.applyColorMap(preview, cv2.COLORMAP_INFERNO)

    cv2.imwrite(str(output_path), preview_color)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)

    parser.add_argument(
        "--model-id",
        default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    )

    parser.add_argument("--output-depth-npy", required=True)
    parser.add_argument("--output-preview", required=True)

    args = parser.parse_args()

    image_path = Path(args.image)
    output_depth_npy = Path(args.output_depth_npy)
    output_preview = Path(args.output_preview)

    output_depth_npy.parent.mkdir(parents=True, exist_ok=True)
    output_preview.parent.mkdir(parents=True, exist_ok=True)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Loading model:", args.model_id)

    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id)

    model.to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_h, original_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_m = prediction.cpu().numpy().astype(np.float32)

    np.save(output_depth_npy, depth_m)
    save_depth_preview(depth_m, output_preview)

    print("Saved depth map:", output_depth_npy)
    print("Saved preview:", output_preview)

    print("\nDepth stats:")
    print("min:", float(np.nanmin(depth_m)))
    print("median:", float(np.nanmedian(depth_m)))
    print("max:", float(np.nanmax(depth_m)))


if __name__ == "__main__":
    main()