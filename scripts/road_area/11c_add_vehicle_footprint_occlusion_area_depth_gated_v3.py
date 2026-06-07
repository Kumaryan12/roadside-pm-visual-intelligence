from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd


# ============================================================
# VEHICLE FOOTPRINT PRIORS
# ============================================================

VEHICLE_DIMS_M = {
    "car": {"width": 1.75, "length": 4.30},
    "auto_rickshaw": {"width": 1.40, "length": 2.70},
    "autorickshaw": {"width": 1.40, "length": 2.70},
    "motorcycle": {"width": 0.80, "length": 2.00},
    "bicycle": {"width": 0.60, "length": 1.80},
    "bus": {"width": 2.50, "length": 10.50},
    "truck": {"width": 2.50, "length": 8.00},
    "unknown_vehicle": {"width": 1.75, "length": 4.30},
    "vehicle_fallback": {"width": 1.75, "length": 4.30},
    "vehicle fallback": {"width": 1.75, "length": 4.30},
}


CONSERVATIVE_FACTOR = {
    "car": 0.60,
    "auto_rickshaw": 0.60,
    "autorickshaw": 0.60,
    "motorcycle": 0.50,
    "bicycle": 0.50,
    "bus": 0.50,
    "truck": 0.50,
    "unknown_vehicle": 0.50,
    "vehicle_fallback": 0.50,
    "vehicle fallback": 0.50,
}


# ============================================================
# EFFECTIVE VALID FRAME CONFIG
# ============================================================
# Same artificial bottom/platform mask used during preprocessing.
# Visibility should be computed only within the valid image area,
# not including the artificial black region.

PLATFORM_MASK_CONFIG = {
    1: (0.00, 0.82, 1.00, 1.00),
    4: (0.00, 0.95, 1.00, 1.00),
    6: (0.00, 0.75, 1.00, 1.00),
}


def get_effective_frame_bounds(image_width, image_height, lens_id):
    """
    Returns effective valid frame bounds:
    x_min, y_min, x_max, y_max

    For lens 1:
    artificial black/platform region starts at 0.82H,
    so the effective valid bottom is 0.82H.
    """

    image_width = float(image_width)
    image_height = float(image_height)

    x_min = 0.0
    y_min = 0.0
    x_max = image_width
    y_max = image_height

    try:
        lens_id = int(lens_id)
    except Exception:
        lens_id = None

    if lens_id in PLATFORM_MASK_CONFIG:
        _, y1r, _, _ = PLATFORM_MASK_CONFIG[lens_id]
        y_max = float(y1r * image_height)

    return x_min, y_min, x_max, y_max


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_class_name(x):
    return str(x).strip().lower().replace("-", "_")


def resolve_path(path_value, project_root):
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return project_root / p


def get_vehicle_dims(class_name):
    cls = normalize_class_name(class_name)
    return VEHICLE_DIMS_M.get(cls, VEHICLE_DIMS_M["unknown_vehicle"])


def get_conservative_factor(class_name):
    cls = normalize_class_name(class_name)
    return float(CONSERVATIVE_FACTOR.get(cls, 0.50))


# ============================================================
# BBOX VISIBILITY FACTOR
# ============================================================

def compute_bbox_visibility_factor(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
    lens_id,
    edge_margin_px=8,
    min_box_area_ratio=0.00005,
    tiny_box_factor=0.30,
    edge_factor=0.65,
    two_edge_factor=0.45,
    three_edge_factor=0.30,
    huge_box_area_ratio=0.35,
    huge_box_factor=0.50,
):
    """
    Estimate how much of the physical vehicle is visible.

    Important:
    This uses the effective valid frame, not the full image height.
    The artificial black bottom mask is excluded from frame-boundary logic.
    """

    image_width = float(image_width)
    image_height = float(image_height)

    eff_x1, eff_y1, eff_x2, eff_y2 = get_effective_frame_bounds(
        image_width=image_width,
        image_height=image_height,
        lens_id=lens_id,
    )

    effective_width = max(eff_x2 - eff_x1, 1.0)
    effective_height = max(eff_y2 - eff_y1, 1.0)
    effective_area = effective_width * effective_height

    raw_x1 = float(x1)
    raw_y1 = float(y1)
    raw_x2 = float(x2)
    raw_y2 = float(y2)

    # Clip bbox to effective valid frame for area-ratio calculation.
    clip_x1 = max(eff_x1, min(eff_x2, raw_x1))
    clip_y1 = max(eff_y1, min(eff_y2, raw_y1))
    clip_x2 = max(eff_x1, min(eff_x2, raw_x2))
    clip_y2 = max(eff_y1, min(eff_y2, raw_y2))

    clipped_bw = max(0.0, clip_x2 - clip_x1)
    clipped_bh = max(0.0, clip_y2 - clip_y1)
    clipped_bbox_area = clipped_bw * clipped_bh

    bbox_area_ratio = clipped_bbox_area / max(effective_area, 1.0)

    if clipped_bbox_area <= 0:
        return (
            0.0,
            "bbox_outside_effective_valid_frame",
            0,
            bbox_area_ratio,
        )

    touches_left = raw_x1 <= eff_x1 + edge_margin_px
    touches_right = raw_x2 >= eff_x2 - edge_margin_px
    touches_top = raw_y1 <= eff_y1 + edge_margin_px

    # Critical correction:
    # bottom means effective valid bottom, not full image bottom.
    touches_bottom = raw_y2 >= eff_y2 - edge_margin_px

    touches_count = (
        int(touches_left)
        + int(touches_right)
        + int(touches_top)
        + int(touches_bottom)
    )

    if bbox_area_ratio < min_box_area_ratio:
        return (
            tiny_box_factor,
            "tiny_bbox_uncertain_effective_frame",
            touches_count,
            bbox_area_ratio,
        )

    if bbox_area_ratio >= huge_box_area_ratio:
        return (
            huge_box_factor,
            "huge_bbox_heavy_occlusion_downweighted_effective_frame",
            touches_count,
            bbox_area_ratio,
        )

    if touches_count >= 3:
        return (
            three_edge_factor,
            "touches_three_or_more_effective_frame_edges",
            touches_count,
            bbox_area_ratio,
        )

    if touches_count == 2:
        return (
            two_edge_factor,
            "touches_two_effective_frame_edges",
            touches_count,
            bbox_area_ratio,
        )

    if touches_count == 1:
        return (
            edge_factor,
            "touches_one_effective_frame_edge",
            touches_count,
            bbox_area_ratio,
        )

    return (
        1.0,
        "fully_inside_effective_valid_frame",
        touches_count,
        bbox_area_ratio,
    )


# ============================================================
# VEHICLE DEPTH ESTIMATION
# ============================================================

def estimate_vehicle_depth_from_bbox(
    depth_m,
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
    bottom_fraction=0.30,
    min_valid_depth_m=0.5,
    max_valid_depth_m=80.0,
    min_valid_pixels=20,
):
    """
    Estimate vehicle distance using bottom portion of bbox.

    Bottom part is used because it is closer to road-contact / occlusion region.
    """

    if depth_m is None:
        return {
            "vehicle_depth_m": np.nan,
            "vehicle_depth_p25_m": np.nan,
            "vehicle_depth_p75_m": np.nan,
            "depth_valid_pixel_count": 0,
            "depth_reason": "depth_map_missing",
        }

    dh, dw = depth_m.shape

    if int(dw) != int(image_width) or int(dh) != int(image_height):
        depth_resized = cv2.resize(
            depth_m,
            (int(image_width), int(image_height)),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        depth_resized = depth_m

    h, w = depth_resized.shape

    x1 = int(max(0, min(w - 1, round(float(x1)))))
    x2 = int(max(0, min(w, round(float(x2)))))
    y1 = int(max(0, min(h - 1, round(float(y1)))))
    y2 = int(max(0, min(h, round(float(y2)))))

    if x2 <= x1 or y2 <= y1:
        return {
            "vehicle_depth_m": np.nan,
            "vehicle_depth_p25_m": np.nan,
            "vehicle_depth_p75_m": np.nan,
            "depth_valid_pixel_count": 0,
            "depth_reason": "invalid_bbox",
        }

    bbox_h = y2 - y1
    crop_y1 = int(y1 + (1.0 - bottom_fraction) * bbox_h)
    crop_y1 = max(y1, min(crop_y1, y2 - 1))

    crop = depth_resized[crop_y1:y2, x1:x2]

    valid = (
        np.isfinite(crop)
        & (crop >= min_valid_depth_m)
        & (crop <= max_valid_depth_m)
    )

    valid_depths = crop[valid]

    if valid_depths.size < min_valid_pixels:
        return {
            "vehicle_depth_m": np.nan,
            "vehicle_depth_p25_m": np.nan,
            "vehicle_depth_p75_m": np.nan,
            "depth_valid_pixel_count": int(valid_depths.size),
            "depth_reason": "insufficient_valid_depth_pixels",
        }

    return {
        "vehicle_depth_m": float(np.median(valid_depths)),
        "vehicle_depth_p25_m": float(np.percentile(valid_depths, 25)),
        "vehicle_depth_p75_m": float(np.percentile(valid_depths, 75)),
        "depth_valid_pixel_count": int(valid_depths.size),
        "depth_reason": "depth_from_bbox_bottom_crop",
    }


def depth_factor_from_vehicle_depth(
    vehicle_depth_m,
    near_full_m=10.0,
    mid_m=15.0,
    cutoff_m=20.0,
    missing_depth_factor=0.50,
):
    """
    Smooth depth gate.

    <=10 m: full contribution
    10–15 m: partial
    15–20 m: reduced
    >20 m: rejected
    """

    if not np.isfinite(vehicle_depth_m):
        return missing_depth_factor, "missing_depth_downweighted"

    if vehicle_depth_m <= near_full_m:
        return 1.0, "depth_le_10m_full"

    if vehicle_depth_m <= mid_m:
        return 0.75, "depth_10_15m_partial"

    if vehicle_depth_m <= cutoff_m:
        return 0.50, "depth_15_20m_partial"

    return 0.0, "depth_gt_20m_rejected"


# ============================================================
# QUALITY FLAGS
# ============================================================

def occlusion_quality_from_fraction(frac):
    if not np.isfinite(frac):
        return "unknown"
    if frac <= 0.30:
        return "good"
    if frac <= 0.50:
        return "medium"
    if frac <= 0.70:
        return "low"
    return "very_low"


def road_mask_quality_from_visible_area(visible):
    try:
        visible = float(visible)
    except Exception:
        return "unknown"

    if not np.isfinite(visible):
        return "unknown"
    if visible < 15:
        return "very_low_visible_road"
    if visible < 30:
        return "low_visible_road"
    return "acceptable_visible_road"


def final_status_v3(row):
    occ_q = str(row.get("vehicle_occlusion_quality_v3", "unknown"))

    try:
        visible = float(row.get("visible_road_area_m2_depth_est", np.nan))
    except Exception:
        visible = np.nan

    if not np.isfinite(visible):
        return "exclude_or_manual_review"

    if visible < 15:
        return "exclude_or_manual_review"

    if visible < 30:
        return "use_sensitivity_only"

    if occ_q == "very_low":
        return "exclude_or_manual_review"

    if occ_q == "low":
        return "use_sensitivity_only"

    if occ_q in ["good", "medium"]:
        return "use_primary"

    return "manual_review"


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--road-area-csv",
        default="outputs/road_area_lens1_batch/lens1_depth_estimated_road_area.csv",
    )

    parser.add_argument(
        "--detections-csv",
        default="outputs/features/idd_vehicle_detections_object_level_lens1_nms_v2.csv",
    )

    parser.add_argument(
        "--output-frame-csv",
        default="outputs/road_area_lens1_batch/lens1_depth_gated_occlusion_adjusted_road_area_v3.csv",
    )

    parser.add_argument(
        "--output-vehicle-detail-csv",
        default="outputs/road_area_lens1_batch/lens1_depth_gated_vehicle_occlusion_details_v3.csv",
    )

    parser.add_argument("--lens-id", type=int, default=1)

    parser.add_argument("--class-col", default="pm_class_name")
    parser.add_argument("--x1-col", default="x1")
    parser.add_argument("--y1-col", default="y1")
    parser.add_argument("--x2-col", default="x2")
    parser.add_argument("--y2-col", default="y2")
    parser.add_argument("--conf-col", default="confidence")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)

    parser.add_argument("--edge-margin-px", type=float, default=8.0)
    parser.add_argument("--vehicle-depth-cutoff-m", type=float, default=20.0)
    parser.add_argument("--missing-depth-factor", type=float, default=0.50)

    args = parser.parse_args()

    project_root = Path(".").resolve()

    road_df = pd.read_csv(args.road_area_csv)
    det_df = pd.read_csv(args.detections_csv)

    if "lens_id" in road_df.columns:
        road_df = road_df[road_df["lens_id"] == args.lens_id].copy()

    if "lens_id" in det_df.columns:
        det_df = det_df[det_df["lens_id"] == args.lens_id].copy()

    if args.conf_col in det_df.columns:
        det_df[args.conf_col] = pd.to_numeric(det_df[args.conf_col], errors="coerce")
        det_df = det_df[det_df[args.conf_col] >= args.confidence_threshold].copy()

    if "is_pm_relevant_vehicle" in det_df.columns:
        det_df = det_df[det_df["is_pm_relevant_vehicle"] == True].copy()

    if "visible_road_area_m2_depth_est" not in road_df.columns:
        if "estimated_road_area_m2" in road_df.columns:
            road_df["visible_road_area_m2_depth_est"] = road_df["estimated_road_area_m2"]
        else:
            raise ValueError("No visible road area column found.")

    if "saved_depth_npy_path" not in road_df.columns:
        raise ValueError(
            "saved_depth_npy_path not found in road-area CSV. "
            "Rerun 10_batch_lens1_road_area_depth.py with --save-depth-npy."
        )

    if "processed_frame_key" in road_df.columns and "processed_frame_key" in det_df.columns:
        key_cols = ["processed_frame_key", "lens_id"]
    else:
        key_cols = ["matched_run_id", "sample_index", "lens_id"]

    for k in key_cols:
        if k not in road_df.columns:
            raise ValueError(f"Road-area CSV missing key column: {k}")
        if k not in det_df.columns:
            raise ValueError(f"Detections CSV missing key column: {k}")

    grouped = det_df.groupby(key_cols, dropna=False)

    depth_cache = {}
    vehicle_rows = []
    frame_summaries = []

    for _, road_row in road_df.iterrows():
        key = tuple(road_row[k] for k in key_cols)

        visible_area = float(road_row["visible_road_area_m2_depth_est"])

        if key in grouped.groups:
            frame_det = grouped.get_group(key).copy()
        else:
            frame_det = pd.DataFrame(columns=det_df.columns)

        depth_path_value = road_row.get("saved_depth_npy_path", "")
        depth_path = resolve_path(depth_path_value, project_root)

        depth_m = None
        if depth_path.exists():
            if str(depth_path) not in depth_cache:
                depth_cache[str(depth_path)] = np.load(depth_path)
            depth_m = depth_cache[str(depth_path)]

        total_hidden_full = 0.0
        total_hidden_conservative = 0.0
        used_count = 0
        rejected_depth_count = 0
        missing_depth_count = 0

        for _, d in frame_det.iterrows():
            class_name = d.get(args.class_col, "unknown_vehicle")
            class_norm = normalize_class_name(class_name)

            x1 = float(d[args.x1_col])
            y1 = float(d[args.y1_col])
            x2 = float(d[args.x2_col])
            y2 = float(d[args.y2_col])

            image_width = float(d.get("image_width", road_row.get("area_image_width_px", np.nan)))
            image_height = float(d.get("image_height", road_row.get("area_image_height_px", np.nan)))

            if not np.isfinite(image_width) or not np.isfinite(image_height):
                image_width = float(road_row.get("area_image_width_px", 0))
                image_height = float(road_row.get("area_image_height_px", 0))

            lens_id_value = road_row.get("lens_id", args.lens_id)

            eff_x1, eff_y1, eff_x2, eff_y2 = get_effective_frame_bounds(
                image_width=image_width,
                image_height=image_height,
                lens_id=lens_id_value,
            )

            dims = get_vehicle_dims(class_norm)
            standard_width_m = dims["width"]
            standard_length_m = dims["length"]
            standard_footprint_m2 = standard_width_m * standard_length_m

            bbox_visibility_factor, visibility_reason, edge_count, bbox_area_ratio = (
                compute_bbox_visibility_factor(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    image_width=image_width,
                    image_height=image_height,
                    lens_id=lens_id_value,
                    edge_margin_px=args.edge_margin_px,
                )
            )

            depth_info = estimate_vehicle_depth_from_bbox(
                depth_m=depth_m,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                image_width=image_width,
                image_height=image_height,
            )

            vehicle_depth_m = depth_info["vehicle_depth_m"]

            depth_factor, depth_factor_reason = depth_factor_from_vehicle_depth(
                vehicle_depth_m=vehicle_depth_m,
                cutoff_m=args.vehicle_depth_cutoff_m,
                missing_depth_factor=args.missing_depth_factor,
            )

            if depth_factor == 0.0:
                rejected_depth_count += 1

            if not np.isfinite(vehicle_depth_m):
                missing_depth_count += 1

            conservative_factor = get_conservative_factor(class_norm)

            hidden_full = (
                standard_footprint_m2
                * bbox_visibility_factor
                * depth_factor
            )

            hidden_conservative = hidden_full * conservative_factor

            used_for_occlusion = hidden_conservative > 0

            if used_for_occlusion:
                used_count += 1

            total_hidden_full += hidden_full
            total_hidden_conservative += hidden_conservative

            vehicle_rows.append({
                "sample_index": road_row.get("sample_index"),
                "matched_run_id": road_row.get("matched_run_id", ""),
                "lens_id": lens_id_value,
                "processed_frame_key": road_row.get("processed_frame_key", ""),
                "processed_frame_path": road_row.get("processed_frame_path", ""),

                "class_name": class_norm,
                "detection_confidence": float(d.get(args.conf_col, np.nan)),

                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_width_px": max(0.0, x2 - x1),
                "bbox_height_px": max(0.0, y2 - y1),
                "bbox_area_ratio": bbox_area_ratio,
                "image_width": image_width,
                "image_height": image_height,

                "effective_frame_x1_v3": eff_x1,
                "effective_frame_y1_v3": eff_y1,
                "effective_frame_x2_v3": eff_x2,
                "effective_frame_y2_v3": eff_y2,
                "effective_frame_height_v3": eff_y2 - eff_y1,

                "standard_width_m": standard_width_m,
                "standard_length_m": standard_length_m,
                "standard_footprint_m2": standard_footprint_m2,

                "bbox_visibility_factor_v3": bbox_visibility_factor,
                "visibility_reason_v3": visibility_reason,
                "bbox_touches_edge_count_v3": edge_count,

                "vehicle_depth_m_v3": vehicle_depth_m,
                "vehicle_depth_p25_m_v3": depth_info["vehicle_depth_p25_m"],
                "vehicle_depth_p75_m_v3": depth_info["vehicle_depth_p75_m"],
                "depth_valid_pixel_count_v3": depth_info["depth_valid_pixel_count"],
                "depth_reason_v3": depth_info["depth_reason"],

                "vehicle_depth_factor_v3": depth_factor,
                "vehicle_depth_factor_reason_v3": depth_factor_reason,

                "conservative_factor_v3": conservative_factor,
                "hidden_road_area_m2_full_footprint_v3": hidden_full,
                "hidden_road_area_m2_conservative_v3": hidden_conservative,
                "used_for_occlusion_v3": bool(used_for_occlusion),

                "method_version": "v3_effective_frame_visibility_plus_vehicle_depth_gate",
            })

        final_area = visible_area + total_hidden_conservative
        occ_frac = total_hidden_conservative / final_area if final_area > 0 else np.nan
        occ_q = occlusion_quality_from_fraction(occ_frac)

        summary = road_row.to_dict()
        summary["visible_road_area_m2_depth_est"] = visible_area

        summary["vehicle_occluded_road_area_m2_full_footprint_v3"] = total_hidden_full
        summary["vehicle_occluded_road_area_m2_conservative_v3"] = total_hidden_conservative
        summary["road_area_m2_occlusion_adjusted_conservative_v3"] = final_area
        summary["road_area_vehicle_occlusion_fraction_v3"] = occ_frac
        summary["vehicle_occlusion_quality_v3"] = occ_q

        summary["vehicle_count_detected_for_occlusion_v3"] = int(len(frame_det))
        summary["vehicle_count_used_for_occlusion_v3"] = int(used_count)
        summary["vehicle_count_rejected_by_depth_v3"] = int(rejected_depth_count)
        summary["vehicle_count_missing_depth_v3"] = int(missing_depth_count)

        summary["road_mask_area_quality_v3"] = road_mask_quality_from_visible_area(visible_area)
        summary["final_road_area_feature_status_v3"] = final_status_v3(summary)

        summary["road_area_feature_version_v3"] = (
            "v3_effective_frame_visibility_plus_vehicle_depth_gate"
        )

        frame_summaries.append(summary)

    frame_out = pd.DataFrame(frame_summaries)
    vehicle_out = pd.DataFrame(vehicle_rows)

    output_frame_csv = Path(args.output_frame_csv)
    output_vehicle_csv = Path(args.output_vehicle_detail_csv)

    output_frame_csv.parent.mkdir(parents=True, exist_ok=True)
    output_vehicle_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_out.to_csv(output_frame_csv, index=False)
    vehicle_out.to_csv(output_vehicle_csv, index=False)

    print("\nSaved frame-level v3 road-area output:")
    print(output_frame_csv)
    print("Shape:", frame_out.shape)

    print("\nSaved vehicle-level v3 occlusion details:")
    print(output_vehicle_csv)
    print("Shape:", vehicle_out.shape)

    print("\nV3 final status counts:")
    print(frame_out["final_road_area_feature_status_v3"].value_counts(dropna=False))

    print("\nV3 occlusion quality counts:")
    print(frame_out["vehicle_occlusion_quality_v3"].value_counts(dropna=False))

    cols = [
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_conservative_v3",
        "road_area_m2_occlusion_adjusted_conservative_v3",
        "road_area_vehicle_occlusion_fraction_v3",
        "vehicle_count_detected_for_occlusion_v3",
        "vehicle_count_used_for_occlusion_v3",
        "vehicle_count_rejected_by_depth_v3",
        "vehicle_count_missing_depth_v3",
    ]

    print("\nV3 summary:")
    print(frame_out[cols].describe().T)

    print("\nWorst V3 occlusion frames:")
    show_cols = [
        "sample_index",
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_conservative_v3",
        "road_area_m2_occlusion_adjusted_conservative_v3",
        "road_area_vehicle_occlusion_fraction_v3",
        "vehicle_occlusion_quality_v3",
        "final_road_area_feature_status_v3",
        "vehicle_count_detected_for_occlusion_v3",
        "vehicle_count_used_for_occlusion_v3",
        "vehicle_count_rejected_by_depth_v3",
    ]

    print(
        frame_out[show_cols]
        .sort_values("road_area_vehicle_occlusion_fraction_v3", ascending=False)
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()