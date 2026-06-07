from pathlib import Path
import argparse
import cv2
import pandas as pd
import numpy as np


def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img


def resize_to_height(img, target_h=420):
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def draw_vehicle_boxes(img, veh_df):
    out = img.copy()

    for _, r in veh_df.iterrows():
        x1, y1, x2, y2 = map(int, [r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]])
        cls = str(r["class_name"])
        hidden = float(r.get("hidden_road_area_m2_est_conservative", r.get("hidden_road_area_m2_est", 0.0)))
        contact = float(r.get("road_contact_confidence", 0.0))
        vis = float(r.get("visibility_factor", 0.0))

        color = (0, 255, 0) if hidden > 0 else (0, 0, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} | hidden={hidden:.1f} | rc={contact:.2f} | vis={vis:.2f}"
        y_text = max(20, y1 - 8)
        cv2.putText(
            out,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return out


def make_text_panel(width, height, lines):
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255

    y = 35
    for line in lines:
        cv2.putText(
            panel,
            str(line),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 32

    return panel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frame-csv",
        default="outputs/road_area_lens1_batch/lens1_bbox_nms_conservative_occlusion_adjusted_road_area.csv",
    )

    parser.add_argument(
        "--vehicle-detail-csv",
        default="outputs/road_area_lens1_batch/lens1_bbox_nms_conservative_vehicle_occlusion_details.csv",
    )

    parser.add_argument(
        "--mask-dir",
        default="outputs/road_area_lens1_batch/masks",
    )

    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/road_area_lens1_batch/occlusion_audit_pack",
    )

    args = parser.parse_args()

    frame_df = pd.read_csv(args.frame_csv)
    veh_df = pd.read_csv(args.vehicle_detail_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = Path(args.mask_dir)

    for sample in args.samples:
        f = frame_df[frame_df["sample_index"] == sample].copy()

        if len(f) == 0:
            print(f"[WARN] sample {sample} not found in frame CSV")
            continue

        f = f.iloc[0]

        image_path = Path(str(f["processed_frame_path"]))
        if not image_path.exists():
            print(f"[WARN] image missing: {image_path}")
            continue

        img = read_image(image_path)
        if img is None:
            print(f"[WARN] cannot read image: {image_path}")
            continue

        # Road mask path
        if "saved_road_mask_path" in f and pd.notna(f["saved_road_mask_path"]):
            mask_path = Path(str(f["saved_road_mask_path"]))
        else:
            pattern = f"sample_{sample:05d}_lens_1_*_road_mask.png"
            matches = list(mask_dir.glob(pattern))
            mask_path = matches[0] if matches else None

        if mask_path is not None and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_rgb = np.zeros_like(img)

        sample_veh = veh_df[
            (veh_df["sample_index"] == sample)
            & (veh_df["lens_id"] == 1)
        ].copy()

        img_boxes = draw_vehicle_boxes(img, sample_veh)

        visible = float(f["visible_road_area_m2_depth_est"])
        hidden_full = float(f.get("vehicle_occluded_road_area_m2_est", np.nan))
        hidden_cons = float(f.get("vehicle_occluded_road_area_m2_est_conservative", np.nan))
        adjusted_cons = float(f.get("occlusion_adjusted_road_area_m2_est_conservative", np.nan))
        frac_cons = float(f.get("vehicle_occlusion_fraction_conservative", np.nan))
        quality = str(f.get("occlusion_adjustment_quality_conservative", ""))

        used_count = int((sample_veh.get("hidden_road_area_m2_est_conservative", sample_veh.get("hidden_road_area_m2_est", 0)) > 0).sum())
        det_count = len(sample_veh)

        lines = [
            f"sample_index: {sample}",
            f"visible road area: {visible:.2f} m2",
            f"hidden vehicle area full: {hidden_full:.2f} m2",
            f"hidden vehicle area conservative: {hidden_cons:.2f} m2",
            f"adjusted conservative area: {adjusted_cons:.2f} m2",
            f"conservative occlusion fraction: {frac_cons:.3f}",
            f"quality: {quality}",
            f"vehicle detections: {det_count}",
            f"vehicles used: {used_count}",
            "",
            "Green box = contributed hidden area",
            "Red box = rejected/no hidden-area contribution",
        ]

        img_r = resize_to_height(img, 420)
        mask_r = resize_to_height(mask_rgb, 420)
        boxes_r = resize_to_height(img_boxes, 420)

        text_panel = make_text_panel(
            width=max(img_r.shape[1], mask_r.shape[1], boxes_r.shape[1]),
            height=420,
            lines=lines,
        )

        # Make widths equal by padding
        panels = [img_r, mask_r, boxes_r, text_panel]
        max_h = max(p.shape[0] for p in panels)
        max_w = max(p.shape[1] for p in panels)

        padded = []
        for p in panels:
            canvas = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
            canvas[:p.shape[0], :p.shape[1]] = p
            padded.append(canvas)

        top = np.hstack([padded[0], padded[1]])
        bottom = np.hstack([padded[2], padded[3]])
        sheet = np.vstack([top, bottom])

        out_path = output_dir / f"sample_{sample:05d}_occlusion_audit.jpg"
        cv2.imwrite(str(out_path), sheet)

        print("Saved:", out_path)


if __name__ == "__main__":
    main()