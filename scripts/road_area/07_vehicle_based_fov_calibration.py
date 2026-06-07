from pathlib import Path
import argparse
import math
import json

import cv2
import numpy as np
import pandas as pd


def robust_depth_in_bbox(depth, x, y, w, h, center_fraction=0.5):
    """
    Uses the central part of the bbox to avoid background and edges.
    """
    h_img, w_img = depth.shape

    cx1 = int(x + (1 - center_fraction) * w / 2)
    cx2 = int(x + (1 + center_fraction) * w / 2)
    cy1 = int(y + (1 - center_fraction) * h / 2)
    cy2 = int(y + (1 + center_fraction) * h / 2)

    cx1 = max(0, min(cx1, w_img - 1))
    cx2 = max(0, min(cx2, w_img))
    cy1 = max(0, min(cy1, h_img - 1))
    cy2 = max(0, min(cy2, h_img))

    patch = depth[cy1:cy2, cx1:cx2]
    vals = patch[np.isfinite(patch)]

    if len(vals) == 0:
        raise ValueError("No valid depth values inside bbox.")

    return {
        "depth_median_m": float(np.median(vals)),
        "depth_p25_m": float(np.percentile(vals, 25)),
        "depth_p75_m": float(np.percentile(vals, 75)),
        "depth_std_m": float(np.std(vals)),
        "depth_count": int(len(vals)),
        "center_x1": cx1,
        "center_y1": cy1,
        "center_x2": cx2,
        "center_y2": cy2,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--depth-npy", required=True)

    parser.add_argument(
        "--object-name",
        default="car_rear",
        help="Example: car_rear, bus_rear, auto_rear",
    )

    parser.add_argument(
        "--real-object-width-m",
        type=float,
        required=True,
        help="Known/assumed real width of object in meters.",
    )

    parser.add_argument(
        "--output-csv",
        default="outputs/road_area_inputs/vehicle_fov_calibration.csv",
    )

    parser.add_argument(
        "--output-preview",
        default="outputs/road_area_inputs/vehicle_fov_calibration_preview.png",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    depth_path = Path(args.depth_npy)
    output_csv = Path(args.output_csv)
    output_preview = Path(args.output_preview)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_preview.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    depth = np.load(depth_path)

    if depth.ndim == 3:
        depth = depth.squeeze()

    h_img, w_img = image.shape[:2]

    if depth.shape[:2] != (h_img, w_img):
        depth = cv2.resize(depth, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    print("\nSelect bbox around a front/rear-facing vehicle.")
    print("Use the full visible vehicle width.")
    print("Press ENTER/SPACE after selecting. Press c to cancel.")

    roi = cv2.selectROI("Select calibration vehicle", image, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = [int(v) for v in roi]

    if w <= 0 or h <= 0:
        raise ValueError("Invalid bbox selection.")

    depth_stats = robust_depth_in_bbox(depth, x, y, w, h, center_fraction=0.5)

    object_depth_m = depth_stats["depth_median_m"]
    real_width_m = args.real_object_width_m

    fx_px = (w * object_depth_m) / real_width_m
    fy_px = fx_px

    hfov_rad = 2 * math.atan(w_img / (2 * fx_px))
    hfov_deg = math.degrees(hfov_rad)

    result = {
        "image_path": str(image_path),
        "depth_npy": str(depth_path),
        "object_name": args.object_name,
        "real_object_width_m": real_width_m,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_width_px": w,
        "bbox_height_px": h,
        "image_width_px": w_img,
        "image_height_px": h_img,
        "estimated_object_depth_m": object_depth_m,
        "estimated_fx_px": fx_px,
        "estimated_fy_px": fy_px,
        "estimated_hfov_deg": hfov_deg,
    }

    result.update(depth_stats)

    pd.DataFrame([result]).to_csv(output_csv, index=False)

    preview = image.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.rectangle(
        preview,
        (depth_stats["center_x1"], depth_stats["center_y1"]),
        (depth_stats["center_x2"], depth_stats["center_y2"]),
        (0, 0, 255),
        2,
    )

    label = (
        f"{args.object_name}: fx={fx_px:.1f}px, "
        f"HFOV={hfov_deg:.1f}deg, Z={object_depth_m:.2f}m"
    )

    cv2.putText(
        preview,
        label,
        (max(10, x), max(30, y - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_preview), preview)

    print("\nSaved calibration:")
    print(output_csv)

    print("\nSaved preview:")
    print(output_preview)

    print("\nCalibration result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()