from pathlib import Path
import argparse
import pandas as pd
import numpy as np


# ============================================================
# VEHICLE FOOTPRINT PRIORS
# ============================================================
# Approximate real-world footprint = width × length.
# These are not ground-truth per vehicle; they are class-level priors.

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


# Conservative factors to avoid assuming full footprint is recoverable
# from a single front-facing 2D frame.
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


def normalize_class_name(x):
    return str(x).strip().lower().replace("-", "_")


def get_vehicle_dims(class_name):
    cls = normalize_class_name(class_name)
    return VEHICLE_DIMS_M.get(cls, VEHICLE_DIMS_M["unknown_vehicle"])


def get_conservative_factor(class_name):
    cls = normalize_class_name(class_name)
    return float(CONSERVATIVE_FACTOR.get(cls, 0.50))


# ============================================================
# VISIBILITY FACTOR
# ============================================================

def compute_bbox_visibility_factor(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
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
    YOLO boxes are usually clipped inside the image, so we infer truncation
    by checking whether the bbox touches image boundaries.

    Returns:
    - bbox_visibility_factor
    - visibility_reason
    - touches_edge_count
    - bbox_area_ratio
    """

    image_width = float(image_width)
    image_height = float(image_height)

    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    image_area = max(image_width * image_height, 1.0)
    bbox_area = bw * bh
    bbox_area_ratio = bbox_area / image_area

    touches_left = x1 <= edge_margin_px
    touches_right = x2 >= image_width - edge_margin_px
    touches_top = y1 <= edge_margin_px
    touches_bottom = y2 >= image_height - edge_margin_px

    touches_count = int(touches_left) + int(touches_right) + int(touches_top) + int(touches_bottom)

    # Tiny/far boxes are very uncertain for footprint area.
    if bbox_area_ratio < min_box_area_ratio:
        return tiny_box_factor, "tiny_bbox_uncertain", touches_count, bbox_area_ratio

    # Very huge boxes are often near/partial/heavy-occlusion cases.
    # We do not reject them, but down-weight them.
    if bbox_area_ratio >= huge_box_area_ratio:
        return huge_box_factor, "huge_bbox_heavy_occlusion_downweighted", touches_count, bbox_area_ratio

    if touches_count >= 3:
        return three_edge_factor, "touches_three_or_more_edges", touches_count, bbox_area_ratio

    if touches_count == 2:
        return two_edge_factor, "touches_two_edges", touches_count, bbox_area_ratio

    if touches_count == 1:
        return edge_factor, "touches_one_edge", touches_count, bbox_area_ratio

    return 1.0, "fully_inside_frame", touches_count, bbox_area_ratio


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


def final_status_v2(row):
    occ_q = str(row.get("vehicle_occlusion_quality_v2", "unknown"))
    visible = row.get("visible_road_area_m2_depth_est", np.nan)

    try:
        visible = float(visible)
    except Exception:
        visible = np.nan

    # Keep same visible-road adequacy logic as v1
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
        default="outputs/road_area_lens1_batch/lens1_bbox_visibility_only_occlusion_adjusted_road_area_v2.csv",
    )

    parser.add_argument(
        "--output-vehicle-detail-csv",
        default="outputs/road_area_lens1_batch/lens1_bbox_visibility_only_vehicle_occlusion_details_v2.csv",
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

    args = parser.parse_args()

    road_df = pd.read_csv(args.road_area_csv)
    det_df = pd.read_csv(args.detections_csv)

    # Keep lens 1 only
    if "lens_id" in road_df.columns:
        road_df = road_df[road_df["lens_id"] == args.lens_id].copy()

    if "lens_id" in det_df.columns:
        det_df = det_df[det_df["lens_id"] == args.lens_id].copy()

    # Filter detections
    if args.conf_col in det_df.columns:
        det_df[args.conf_col] = pd.to_numeric(det_df[args.conf_col], errors="coerce")
        det_df = det_df[det_df[args.conf_col] >= args.confidence_threshold].copy()

    if "is_pm_relevant_vehicle" in det_df.columns:
        det_df = det_df[det_df["is_pm_relevant_vehicle"] == True].copy()

    # Choose merge keys
    if "processed_frame_key" in road_df.columns and "processed_frame_key" in det_df.columns:
        key_cols = ["processed_frame_key", "lens_id"]
    else:
        key_cols = ["matched_run_id", "sample_index", "lens_id"]

    for k in key_cols:
        if k not in road_df.columns:
            raise ValueError(f"Road-area CSV missing key column: {k}")
        if k not in det_df.columns:
            raise ValueError(f"Detections CSV missing key column: {k}")

    # Determine visible road area column
    if "visible_road_area_m2_depth_est" not in road_df.columns:
        if "estimated_road_area_m2" in road_df.columns:
            road_df["visible_road_area_m2_depth_est"] = road_df["estimated_road_area_m2"]
        else:
            raise ValueError("No visible road area column found.")

    vehicle_rows = []
    frame_summaries = []

    grouped = det_df.groupby(key_cols, dropna=False)

    for _, road_row in road_df.iterrows():
        key = tuple(road_row[k] for k in key_cols)

        visible_area = float(road_row["visible_road_area_m2_depth_est"])

        if key in grouped.groups:
            frame_det = grouped.get_group(key).copy()
        else:
            frame_det = pd.DataFrame(columns=det_df.columns)

        total_hidden_full = 0.0
        total_hidden_conservative = 0.0
        used_count = 0

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
                    edge_margin_px=args.edge_margin_px,
                )
            )

            conservative_factor = get_conservative_factor(class_norm)

            hidden_full = standard_footprint_m2 * bbox_visibility_factor
            hidden_conservative = hidden_full * conservative_factor

            # We assume valid detected road-facing vehicles are on the road.
            # No road-contact confidence is used in v2.
            used_for_occlusion = hidden_conservative > 0

            if used_for_occlusion:
                used_count += 1

            total_hidden_full += hidden_full
            total_hidden_conservative += hidden_conservative

            vehicle_record = {
                "sample_index": road_row.get("sample_index"),
                "matched_run_id": road_row.get("matched_run_id", ""),
                "lens_id": road_row.get("lens_id", args.lens_id),
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

                "standard_width_m": standard_width_m,
                "standard_length_m": standard_length_m,
                "standard_footprint_m2": standard_footprint_m2,

                "bbox_visibility_factor_v2": bbox_visibility_factor,
                "visibility_reason_v2": visibility_reason,
                "bbox_touches_edge_count_v2": edge_count,

                "conservative_factor_v2": conservative_factor,
                "hidden_road_area_m2_full_footprint_v2": hidden_full,
                "hidden_road_area_m2_conservative_v2": hidden_conservative,
                "used_for_occlusion_v2": bool(used_for_occlusion),

                "method_version": "v2_bbox_visibility_only_no_road_contact",
            }

            vehicle_rows.append(vehicle_record)

        final_area = visible_area + total_hidden_conservative
        occ_frac = total_hidden_conservative / final_area if final_area > 0 else np.nan
        occ_q = occlusion_quality_from_fraction(occ_frac)

        summary = road_row.to_dict()
        summary["visible_road_area_m2_depth_est"] = visible_area

        summary["vehicle_occluded_road_area_m2_full_footprint_v2"] = total_hidden_full
        summary["vehicle_occluded_road_area_m2_conservative_v2"] = total_hidden_conservative
        summary["road_area_m2_occlusion_adjusted_conservative_v2"] = final_area
        summary["road_area_vehicle_occlusion_fraction_v2"] = occ_frac
        summary["vehicle_occlusion_quality_v2"] = occ_q

        summary["vehicle_count_detected_for_occlusion_v2"] = int(len(frame_det))
        summary["vehicle_count_used_for_occlusion_v2"] = int(used_count)

        summary["road_mask_area_quality_v2"] = road_mask_quality_from_visible_area(visible_area)
        summary["final_road_area_feature_status_v2"] = final_status_v2(summary)

        summary["road_area_feature_version_v2"] = "v2_bbox_visibility_only_no_road_contact"

        frame_summaries.append(summary)

    frame_out = pd.DataFrame(frame_summaries)
    vehicle_out = pd.DataFrame(vehicle_rows)

    output_frame_csv = Path(args.output_frame_csv)
    output_vehicle_csv = Path(args.output_vehicle_detail_csv)

    output_frame_csv.parent.mkdir(parents=True, exist_ok=True)
    output_vehicle_csv.parent.mkdir(parents=True, exist_ok=True)

    frame_out.to_csv(output_frame_csv, index=False)
    vehicle_out.to_csv(output_vehicle_csv, index=False)

    print("\nSaved frame-level v2 road-area output:")
    print(output_frame_csv)
    print("Shape:", frame_out.shape)

    print("\nSaved vehicle-level v2 occlusion details:")
    print(output_vehicle_csv)
    print("Shape:", vehicle_out.shape)

    print("\nV2 final status counts:")
    print(frame_out["final_road_area_feature_status_v2"].value_counts(dropna=False))

    print("\nV2 occlusion quality counts:")
    print(frame_out["vehicle_occlusion_quality_v2"].value_counts(dropna=False))

    cols = [
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_conservative_v2",
        "road_area_m2_occlusion_adjusted_conservative_v2",
        "road_area_vehicle_occlusion_fraction_v2",
        "vehicle_count_detected_for_occlusion_v2",
        "vehicle_count_used_for_occlusion_v2",
    ]

    print("\nV2 summary:")
    print(frame_out[cols].describe().T)

    print("\nWorst V2 occlusion frames:")
    show_cols = [
        "sample_index",
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_conservative_v2",
        "road_area_m2_occlusion_adjusted_conservative_v2",
        "road_area_vehicle_occlusion_fraction_v2",
        "vehicle_occlusion_quality_v2",
        "final_road_area_feature_status_v2",
    ]
    print(
        frame_out[show_cols]
        .sort_values("road_area_vehicle_occlusion_fraction_v2", ascending=False)
        .head(25)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()