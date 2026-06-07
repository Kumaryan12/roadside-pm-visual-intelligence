from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def nms_group(df, iou_thresh):
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    keep_rows = []
    suppressed = np.zeros(len(df), dtype=bool)

    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)

    for i in range(len(df)):
        if suppressed[i]:
            continue

        keep_rows.append(i)

        for j in range(i + 1, len(df)):
            if suppressed[j]:
                continue

            iou = box_iou(boxes[i], boxes[j])

            if iou >= iou_thresh:
                suppressed[j] = True

    return df.iloc[keep_rows].copy()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-csv",
        default="outputs/features/idd_vehicle_detections_object_level_v2.csv",
    )

    parser.add_argument(
        "--output-csv",
        default="outputs/features/idd_vehicle_detections_object_level_v2_nms.csv",
    )

    parser.add_argument("--lens-id", type=int, default=1)
    parser.add_argument("--iou-thresh", type=float, default=0.60)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    before_all = len(df)

    df = df[df["lens_id"] == args.lens_id].copy()
    df = df[df["is_pm_relevant_vehicle"] == True].copy()
    df = df[pd.to_numeric(df["confidence"], errors="coerce") >= args.confidence_threshold].copy()

    before = len(df)

    kept = []

    for key, g in df.groupby(["processed_frame_key", "lens_id"], sort=False):
        kept.append(nms_group(g, args.iou_thresh))

    out = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("Input object rows all lenses:", before_all)
    print("Lens filtered rows before NMS:", before)
    print("Rows after NMS:", len(out))
    print("Removed:", before - len(out))
    print("Saved:", output_csv)

    if len(out) > 0:
        print("\nClass counts after NMS:")
        print(out["pm_class_name"].value_counts(dropna=False))


if __name__ == "__main__":
    main()