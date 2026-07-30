from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


def make_polygon_mask(shape_hw, points_xy):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def resolve_path(x):
    p = Path(str(x))
    if p.exists():
        return p
    p2 = Path.cwd() / p
    if p2.exists():
        return p2
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-area-csv", required=True)
    parser.add_argument("--polygon-dir", required=True)
    parser.add_argument("--ground-truth-area-m2", type=float, required=True)
    parser.add_argument("--fx-original", type=float, required=True)
    parser.add_argument("--fy-original", type=float, required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--overlay-dir", required=True)
    args = parser.parse_args()

    road = pd.read_csv(args.road_area_csv)
    polygon_dir = Path(args.polygon_dir)
    overlay_dir = Path(args.overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, r in road.iterrows():
        sample = int(r["sample_index"])

        poly_path = polygon_dir / f"sample_{sample:05d}_polygon_points.csv"

        if not poly_path.exists():
            print(f"[WARN] polygon missing for sample {sample}: {poly_path}")
            continue

        image_path = resolve_path(r["processed_frame_path"])
        depth_path = resolve_path(r["saved_depth_npy_path"])
        mask_path = resolve_path(r["saved_road_mask_path"])

        img = cv2.imread(str(image_path))
        depth = np.load(str(depth_path))
        road_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or road_mask is None:
            print(f"[WARN] missing image/mask for sample {sample}")
            continue

        road_mask = road_mask > 0

        h, w = depth.shape[:2]
        img_h, img_w = img.shape[:2]

        poly = pd.read_csv(poly_path)
        points_orig = poly[["x_original", "y_original"]].values.astype(float)

        scale_x = w / img_w
        scale_y = h / img_h

        points_calc = np.array(
            [[x * scale_x, y * scale_y] for x, y in points_orig],
            dtype=np.int32
        )

        poly_mask = make_polygon_mask((h, w), points_calc)

        valid_depth = np.isfinite(depth) & (depth > 0)

        fx = args.fx_original
        fy = args.fy_original

        pixel_area_m2 = (depth ** 2) / (fx * fy)

        selected = poly_mask & road_mask & valid_depth
        pred = float(pixel_area_m2[selected].sum())

        gt = float(args.ground_truth_area_m2)
        signed_error = pred - gt
        abs_error = abs(signed_error)
        pct_error = abs_error / gt * 100 if gt > 0 else np.nan

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
            f"sample {sample}",
            f"GT: {gt:.2f} m2",
            f"Pred: {pred:.2f} m2",
            f"APE: {pct_error:.1f}%",
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

        overlay_path = overlay_dir / f"sample_{sample:05d}_polygon_validation_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        rows.append({
            "sample_index": sample,
            "image_path": str(image_path),
            "ground_truth_area_m2": gt,
            "predicted_polygon_road_area_m2": pred,
            "signed_error_m2": signed_error,
            "absolute_error_m2": abs_error,
            "absolute_percent_error": pct_error,
            "polygon_pixel_count": int(poly_mask.sum()),
            "road_pixels_inside_polygon": int((poly_mask & road_mask).sum()),
            "valid_road_depth_pixels_inside_polygon": int(selected.sum()),
            "fx_used_px": fx,
            "fy_used_px": fy,
            "overlay_path": str(overlay_path),
        })

        print(f"sample {sample}: pred={pred:.3f}, GT={gt:.3f}, APE={pct_error:.2f}%")

    out = pd.DataFrame(rows)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\nSaved:", out_path)

    if len(out):
        print("\nSummary:")
        print(out["absolute_percent_error"].describe())
        print("\nMean APE:", out["absolute_percent_error"].mean())
        print("RMSE:", np.sqrt(np.mean(out["signed_error_m2"] ** 2)))


if __name__ == "__main__":
    main()
