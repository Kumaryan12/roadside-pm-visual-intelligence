from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


def load_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return mask > 0


def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)


def depth_to_points(depth, fx, fy, cx, cy):
    h, w = depth.shape

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    return points


def estimate_surface_area_from_depth(
    depth,
    road_mask,
    fx,
    fy,
    cx=None,
    cy=None,
    min_depth_m=0.5,
    max_depth_m=80.0,
    erode_mask=True,
):
    if depth.shape != road_mask.shape:
        raise ValueError(
            f"Depth shape {depth.shape} and mask shape {road_mask.shape} do not match."
        )

    h, w = depth.shape

    if cx is None:
        cx = (w - 1) / 2.0
    if cy is None:
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

    road_pixels_used = int(usable_mask.sum())
    valid_quads_used = int(quad_valid.sum())

    return {
        "estimated_road_area_m2": total_area_m2,
        "road_pixels_used_for_area": road_pixels_used,
        "valid_road_quads_used": valid_quads_used,
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": cx,
        "cy_px": cy,
        "min_depth_m": min_depth_m,
        "max_depth_m": max_depth_m,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--depth-npy", required=True)

    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)

    parser.add_argument("--min-depth-m", type=float, default=0.5)
    parser.add_argument("--max-depth-m", type=float, default=80.0)

    parser.add_argument(
        "--output-csv",
        default="outputs/validation/road_feature_validation_pack_v1/depth_road_area_estimate_single.csv",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    depth_path = Path(args.depth_npy)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    road_mask = load_mask(mask_path)
    depth = np.load(depth_path)

    if depth.ndim == 3:
        depth = depth.squeeze()

    if depth.shape != road_mask.shape:
        depth = cv2.resize(
            depth,
            (road_mask.shape[1], road_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    fy = args.fy if args.fy is not None else args.fx

    result = estimate_surface_area_from_depth(
        depth=depth,
        road_mask=road_mask,
        fx=args.fx,
        fy=fy,
        cx=args.cx,
        cy=args.cy,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )

    result["image_path"] = str(image_path)
    result["mask_path"] = str(mask_path)
    result["depth_npy"] = str(depth_path)
    result["image_width"] = image.shape[1]
    result["image_height"] = image.shape[0]

    pd.DataFrame([result]).to_csv(output_csv, index=False)

    print("\nEstimated road area:")
    print(f"{result['estimated_road_area_m2']:.3f} m²")

    print("\nSaved:")
    print(output_csv)


if __name__ == "__main__":
    main()