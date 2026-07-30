from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


def polygon_mask(shape_hw, points_xy):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-area-csv", required=True)
    parser.add_argument("--polygon-csv", required=True)
    parser.add_argument("--ground-truth-area-m2", type=float, required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-overlay", required=True)
    args = parser.parse_args()

    road = pd.read_csv(args.road_area_csv)
    poly = pd.read_csv(args.polygon_csv)

    if len(road) != 1:
        print(f"[WARN] road-area CSV has {len(road)} rows. Using first row.")

    r = road.iloc[0]

    image_path = Path(str(r["processed_frame_path"]))
    depth_path = Path(str(r["saved_depth_npy_path"]))
    mask_path = Path(str(r["saved_road_mask_path"]))

    if not image_path.exists():
        image_path = Path.cwd() / image_path
    if not depth_path.exists():
        depth_path = Path.cwd() / depth_path
    if not mask_path.exists():
        mask_path = Path.cwd() / mask_path

    img = cv2.imread(str(image_path))
    depth = np.load(str(depth_path))
    road_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")
    if depth is None:
        raise SystemExit(f"Could not read depth: {depth_path}")
    if road_mask is None:
        raise SystemExit(f"Could not read road mask: {mask_path}")

    road_mask = road_mask > 0

    h, w = depth.shape[:2]

    # The road-area script may resize area calculation to area_calc_width/height.
    # We need polygon coordinates in the same coordinate system as depth/mask.
    image_h, image_w = img.shape[:2]

    points_orig = poly[["x_original", "y_original"]].values.astype(float)

    scale_x = w / image_w
    scale_y = h / image_h

    points_calc = [(float(x) * scale_x, float(y) * scale_y) for x, y in points_orig]

    poly_mask = polygon_mask((h, w), points_calc)

    valid_depth = np.isfinite(depth) & (depth > 0)

    fx = float(r["area_fx_after_resize_px"]) if "area_fx_after_resize_px" in r else float(r["fx_px"])
    fy = float(r["area_fy_after_resize_px"]) if "area_fy_after_resize_px" in r else float(r["fy_px"])

    # Approx metric area per pixel for a pinhole camera:
    # dA ≈ Z^2 / (fx * fy)
    pixel_area_m2 = (depth ** 2) / (fx * fy)

    # Area inside chalk polygon, using road mask
    selected = poly_mask & road_mask & valid_depth

    predicted_area_m2 = float(pixel_area_m2[selected].sum())

    gt = float(args.ground_truth_area_m2)
    signed_error = predicted_area_m2 - gt
    abs_error = abs(signed_error)
    percent_error = abs_error / gt * 100 if gt > 0 else np.nan

    result = {
        "image_path": str(image_path),
        "depth_path": str(depth_path),
        "road_mask_path": str(mask_path),
        "ground_truth_area_m2": gt,
        "predicted_polygon_road_area_m2": predicted_area_m2,
        "signed_error_m2": signed_error,
        "absolute_error_m2": abs_error,
        "absolute_percent_error": percent_error,
        "polygon_pixel_count": int(poly_mask.sum()),
        "road_pixels_inside_polygon": int((poly_mask & road_mask).sum()),
        "valid_depth_road_pixels_inside_polygon": int(selected.sum()),
        "fx_used_px": fx,
        "fy_used_px": fy,
        "depth_mask_height": h,
        "depth_mask_width": w,
        "original_image_height": image_h,
        "original_image_width": image_w,
        "scale_x_original_to_calc": scale_x,
        "scale_y_original_to_calc": scale_y,
    }

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out_csv, index=False)

    # Overlay polygon on original image
    overlay = img.copy()
    pts = points_orig.astype(np.int32).reshape((-1, 1, 2))

    cv2.polylines(overlay, [pts], True, (0, 255, 255), 5)

    for i, (x, y) in enumerate(points_orig.astype(int)):
        cv2.circle(overlay, (x, y), 12, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            f"P{i+1}",
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    text_lines = [
        f"GT area: {gt:.2f} m2",
        f"Predicted: {predicted_area_m2:.2f} m2",
        f"Abs error: {abs_error:.2f} m2",
        f"Percent error: {percent_error:.1f}%",
    ]

    y0 = 60
    for line in text_lines:
        cv2.putText(
            overlay,
            line,
            (40, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        y0 += 55

    out_img = Path(args.output_overlay)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), overlay)

    print("Saved:", out_csv)
    print("Saved:", out_img)
    print()
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
