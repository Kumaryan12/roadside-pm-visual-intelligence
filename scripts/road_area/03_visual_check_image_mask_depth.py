from pathlib import Path
import argparse
import cv2
import numpy as np


def normalize_depth(depth):
    valid = np.isfinite(depth)

    lo = np.percentile(depth[valid], 2)
    hi = np.percentile(depth[valid], 98)

    depth_norm = (depth - lo) / max(hi - lo, 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)

    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    return depth_color


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--depth-npy", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    depth = np.load(args.depth_npy)

    if image is None:
        raise FileNotFoundError(args.image)
    if mask is None:
        raise FileNotFoundError(args.mask)

    if depth.ndim == 3:
        depth = depth.squeeze()

    h, w = image.shape[:2]

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    road = mask > 0

    depth_color = normalize_depth(depth)

    image_overlay = image.copy()
    image_overlay[road] = (
        0.6 * image_overlay[road] + 0.4 * np.array([0, 255, 0])
    ).astype(np.uint8)

    depth_road_only = np.zeros_like(depth_color)
    depth_road_only[road] = depth_color[road]

    combined = np.hstack([
        cv2.resize(image, (640, 360)),
        cv2.resize(image_overlay, (640, 360)),
        cv2.resize(depth_color, (640, 360)),
        cv2.resize(depth_road_only, (640, 360)),
    ])

    cv2.imwrite(args.output, combined)

    road_depth = depth[road]
    road_depth = road_depth[np.isfinite(road_depth)]

    print("Saved:", args.output)

    print("\nRoad depth stats:")
    print("road pixels:", len(road_depth))
    print("min:", float(np.min(road_depth)))
    print("p25:", float(np.percentile(road_depth, 25)))
    print("median:", float(np.median(road_depth)))
    print("p75:", float(np.percentile(road_depth, 75)))
    print("max:", float(np.max(road_depth)))


if __name__ == "__main__":
    main()