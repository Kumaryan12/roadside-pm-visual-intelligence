from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
import math
import subprocess

def robust_depth_in_bbox(depth, x, y, w, h, center_fraction=0.5):
    h_img, w_img = depth.shape
    cx1 = int(x + (1 - center_fraction) * w / 2)
    cx2 = int(x + (1 + center_fraction) * w / 2)
    cy1 = int(y + (1 - center_fraction) * h / 2)
    cy2 = int(y + (1 + center_fraction) * h / 2)
    cx1, cx2 = max(0, cx1), min(w_img, cx2)
    cy1, cy2 = max(0, cy1), min(h_img, cy2)
    patch = depth[cy1:cy2, cx1:cx2]
    vals = patch[np.isfinite(patch)]
    if len(vals) == 0:
        raise ValueError("No valid depth values inside bbox.")
    return float(np.median(vals))

def select_vehicle_bbox(image_path):
    image = cv2.imread(str(image_path))
    roi = cv2.selectROI("Select vehicle rear/front", image, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise ValueError("Invalid bbox selection")
    return x, y, w, h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="+", required=True, help="Paths to processed frame images")
    parser.add_argument("--depths", nargs="+", required=True, help="Corresponding depth .npy files")
    parser.add_argument("--real-width-m", type=float, required=True, help="Assumed vehicle width in meters")
    parser.add_argument("--target-image", required=True, help="Target image to compute m²")
    parser.add_argument("--target-mask", required=True, help="Target road mask")
    parser.add_argument("--max-depth-m", type=float, default=30.0)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    fx_list = []
    for img_path, depth_path in zip(args.frames, args.depths):
        depth = np.load(depth_path)
        if depth.ndim == 3: depth = depth.squeeze()
        x, y, w, h = select_vehicle_bbox(img_path)
        median_depth = robust_depth_in_bbox(depth, x, y, w, h)
        fx = w * median_depth / args.real_width_m
        fx_list.append(fx)
        print(f"Frame: {img_path} → fx = {fx:.1f} px, vehicle depth = {median_depth:.2f} m")

    median_fx = np.median(fx_list)
    print(f"\nMedian fx across {len(fx_list)} frames: {median_fx:.1f} px")

    # Run road area estimation for target image using median fx
    cmd = [
        "python", "scripts/road_validation/05_estimate_road_area_from_metric_depth.py",
        "--image", args.target_image,
        "--mask", args.target_mask,
        "--depth-npy", args.depths[0],  # use depth map corresponding to target image
        "--fx", str(median_fx),
        "--fy", str(median_fx),
        "--max-depth-m", str(args.max_depth_m),
        "--output-csv", args.output_csv
    ]
    subprocess.run(cmd, check=True)
    print(f"Vehicle-calibrated road area saved to {args.output_csv}")

if __name__ == "__main__":
    main()