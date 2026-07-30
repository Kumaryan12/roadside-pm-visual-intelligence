from pathlib import Path
import argparse
import cv2
import pandas as pd
import numpy as np


def read_image(path):
    img = cv2.imread(str(path))
    return img


def resolve_path(path_value):
    if path_value is None or pd.isna(path_value):
        return None

    p = Path(str(path_value))

    if p.is_absolute() and p.exists():
        return p

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return p


def resize_to_height(img, target_h=420):
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def get_first(row, candidates, default=np.nan):
    for c in candidates:
        if c in row.index and not pd.isna(row[c]):
            return row[c]
    return default


def get_hidden_area(row):
    return float(get_first(
        row,
        [
            "hidden_road_area_m2_est_conservative",
            "hidden_road_area_m2_conservative_v3",
            "hidden_road_area_m2_conservative_v2",
            "hidden_road_area_m2_est",
            "hidden_road_area_m2_full_footprint_v3",
            "hidden_road_area_m2_full_footprint_v2",
        ],
        0.0,
    ))


def get_used_flag(row):
    for c in [
        "used_for_occlusion_v3",
        "used_for_occlusion_v2",
    ]:
        if c in row.index and not pd.isna(row[c]):
            val = row[c]
            if isinstance(val, str):
                return val.strip().lower() in ["true", "1", "yes"]
            return bool(val)

    return get_hidden_area(row) > 0


def get_reason(row):
    for c in [
        "vehicle_depth_factor_reason_v3",
        "visibility_reason_v3",
        "visibility_reason_v2",
        "visibility_reason",
        "road_contact_reason",
        "depth_reason_v3",
        "depth_reason",
    ]:
        if c in row.index and not pd.isna(row[c]):
            return str(row[c])
    return ""


def draw_vehicle_boxes(img, veh_df):
    out = img.copy()

    for _, r in veh_df.iterrows():
        x1, y1, x2, y2 = map(int, [r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]])
        cls = str(r["class_name"])

        hidden = get_hidden_area(r)
        used = get_used_flag(r)

        contact = get_first(r, ["road_contact_confidence"], np.nan)
        vis = get_first(r, ["visibility_factor", "bbox_visibility_factor_v3", "bbox_visibility_factor_v2"], np.nan)
        depth = get_first(r, ["vehicle_depth_m_v3", "vehicle_depth_m"], np.nan)
        reason = get_reason(r)

        # Green = used for occlusion / contributes hidden road area
        # Red = rejected / zero contribution
        color = (0, 180, 0) if used and hidden > 0 else (0, 0, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label_parts = [cls, f"hidden={hidden:.1f}"]

        if pd.notna(depth):
            label_parts.append(f"d={float(depth):.1f}m")

        if pd.notna(vis):
            label_parts.append(f"vis={float(vis):.2f}")

        if pd.notna(contact):
            label_parts.append(f"rc={float(contact):.2f}")

        if reason:
            label_parts.append(reason[:24])

        label = " | ".join(label_parts)

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

    y = 32
    for line in lines:
        cv2.putText(
            panel,
            str(line),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return panel


def get_frame_image_path(frame_row):
    for c in [
        "processed_frame_path",
        "area_input_image_path",
        "source_frame_path",
    ]:
        if c in frame_row.index and not pd.isna(frame_row[c]):
            p = resolve_path(frame_row[c])
            if p is not None and p.exists():
                return p
    return None


def get_mask_path(frame_row, sample, mask_dir):
    if "saved_road_mask_path" in frame_row.index and pd.notna(frame_row["saved_road_mask_path"]):
        p = resolve_path(frame_row["saved_road_mask_path"])
        if p is not None and p.exists():
            return p

    patterns = [
        f"sample_{sample:05d}_lens_1_*_road_mask.png",
        f"sample_{sample:05d}_*_road_mask.png",
    ]

    for pattern in patterns:
        matches = list(mask_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def get_frame_metric(frame_row, candidates, default=np.nan):
    return get_first(frame_row, candidates, default)


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
        f = frame_df[frame_df["sample_index"].astype(int) == int(sample)].copy()

        if len(f) == 0:
            print(f"[WARN] sample {sample} not found in frame CSV")
            continue

        f = f.iloc[0]

        image_path = get_frame_image_path(f)

        if image_path is None or not image_path.exists():
            print(f"[WARN] image missing for sample {sample}")
            continue

        img = read_image(image_path)

        if img is None:
            print(f"[WARN] cannot read image: {image_path}")
            continue

        mask_path = get_mask_path(f, sample, mask_dir)

        if mask_path is not None and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_rgb = np.zeros_like(img)

        sample_veh = veh_df[
            (veh_df["sample_index"].astype(int) == int(sample))
            & (veh_df["lens_id"].astype(int) == 1)
        ].copy()

        img_boxes = draw_vehicle_boxes(img, sample_veh)

        visible = get_frame_metric(f, [
            "visible_road_area_m2_depth_est",
            "estimated_road_area_m2",
        ])

        hidden_full = get_frame_metric(f, [
            "vehicle_occluded_road_area_m2_est",
            "vehicle_occluded_road_area_m2_full_footprint_v3",
            "vehicle_occluded_road_area_m2_full_footprint_v2",
        ])

        hidden_cons = get_frame_metric(f, [
            "vehicle_occluded_road_area_m2_est_conservative",
            "vehicle_occluded_road_area_m2_conservative_v3",
            "vehicle_occluded_road_area_m2_conservative_v2",
            "vehicle_occluded_road_area_m2_conservative",
        ])

        adjusted_cons = get_frame_metric(f, [
            "occlusion_adjusted_road_area_m2_est_conservative",
            "road_area_m2_occlusion_adjusted_conservative_v3",
            "road_area_m2_occlusion_adjusted_conservative_v2",
            "road_area_m2_occlusion_adjusted_conservative",
        ])

        frac_cons = get_frame_metric(f, [
            "vehicle_occlusion_fraction_conservative",
            "road_area_vehicle_occlusion_fraction_v3",
            "road_area_vehicle_occlusion_fraction_v2",
            "road_area_vehicle_occlusion_fraction",
        ])

        quality = get_frame_metric(f, [
            "occlusion_adjustment_quality_conservative",
            "vehicle_occlusion_quality_v3",
            "vehicle_occlusion_quality_v2",
            "vehicle_occlusion_quality",
        ], "")

        status = get_frame_metric(f, [
            "final_road_area_feature_status_v3",
            "final_road_area_feature_status_v2",
            "final_road_area_feature_status",
            "area_status",
        ], "")

        used_count = int(sample_veh.apply(lambda r: get_used_flag(r) and get_hidden_area(r) > 0, axis=1).sum())
        det_count = len(sample_veh)

        rejected_count = det_count - used_count

        if "vehicle_count_rejected_by_depth_v3" in f.index and not pd.isna(f["vehicle_count_rejected_by_depth_v3"]):
            rejected_count = int(f["vehicle_count_rejected_by_depth_v3"])

        def fmt(x):
            try:
                if pd.isna(x):
                    return "NA"
                return f"{float(x):.2f}"
            except Exception:
                return str(x)

        def fmt3(x):
            try:
                if pd.isna(x):
                    return "NA"
                return f"{float(x):.3f}"
            except Exception:
                return str(x)

        lines = [
            f"sample_index: {sample}",
            f"visible road area: {fmt(visible)} m2",
            f"hidden vehicle area full: {fmt(hidden_full)} m2",
            f"hidden vehicle area conservative: {fmt(hidden_cons)} m2",
            f"adjusted conservative area: {fmt(adjusted_cons)} m2",
            f"conservative occlusion fraction: {fmt3(frac_cons)}",
            f"quality: {quality}",
            f"status: {status}",
            f"vehicle detections: {det_count}",
            f"vehicles used: {used_count}",
            f"vehicles rejected: {rejected_count}",
            "",
            "Green box = contributed hidden area",
            "Red box = rejected/no contribution",
        ]

        img_r = resize_to_height(img, 420)
        mask_r = resize_to_height(mask_rgb, 420)
        boxes_r = resize_to_height(img_boxes, 420)

        text_panel = make_text_panel(
            width=max(img_r.shape[1], mask_r.shape[1], boxes_r.shape[1]),
            height=420,
            lines=lines,
        )

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
