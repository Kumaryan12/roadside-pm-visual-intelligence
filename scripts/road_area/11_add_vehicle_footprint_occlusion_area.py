from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import cv2


# ------------------------------------------------------------
# 1. Standard vehicle footprint assumptions
# ------------------------------------------------------------

VEHICLE_DIMS_M = {
    # class_name: width_m, length_m
    "car": {"width_m": 1.75, "length_m": 4.30},
    "auto": {"width_m": 1.40, "length_m": 2.70},
    "auto-rickshaw": {"width_m": 1.40, "length_m": 2.70},
    "autorickshaw": {"width_m": 1.40, "length_m": 2.70},
    "motorcycle": {"width_m": 0.80, "length_m": 2.00},
    "bike": {"width_m": 0.80, "length_m": 2.00},
    "bus": {"width_m": 2.50, "length_m": 10.50},
    "truck": {"width_m": 2.50, "length_m": 8.00},
    "van": {"width_m": 1.80, "length_m": 4.80},
    "bicycle": {"width_m": 0.60, "length_m": 1.80},
}


def normalize_class_name(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower().replace("_", "-").replace(" ", "-")


def get_vehicle_dims(class_name):
    c = normalize_class_name(class_name)

    # direct match
    if c in VEHICLE_DIMS_M:
        return VEHICLE_DIMS_M[c]

    # loose matching
    if "auto" in c or "rickshaw" in c:
        return VEHICLE_DIMS_M["auto-rickshaw"]
    if "motor" in c or "bike" in c:
        return VEHICLE_DIMS_M["motorcycle"]
    if "car" in c:
        return VEHICLE_DIMS_M["car"]
    if "bus" in c:
        return VEHICLE_DIMS_M["bus"]
    if "truck" in c or "lorry" in c:
        return VEHICLE_DIMS_M["truck"]
    if "van" in c:
        return VEHICLE_DIMS_M["van"]
    if "cycle" in c:
        return VEHICLE_DIMS_M["bicycle"]

    return None


# ------------------------------------------------------------
# 2. Geometry helpers
# ------------------------------------------------------------

def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w))
    x2 = max(0, min(int(round(x2)), w))
    y1 = max(0, min(int(round(y1)), h))
    y2 = max(0, min(int(round(y2)), h))
    return x1, y1, x2, y2


def region_fraction(mask_bool, box):
    h, w = mask_bool.shape
    x1, y1, x2, y2 = clip_box(*box, w=w, h=h)

    if x2 <= x1 or y2 <= y1:
        return np.nan

    region = mask_bool[y1:y2, x1:x2]
    if region.size == 0:
        return np.nan

    return float(region.mean())


def region_median_depth(depth, box):
    h, w = depth.shape
    x1, y1, x2, y2 = clip_box(*box, w=w, h=h)

    if x2 <= x1 or y2 <= y1:
        return np.nan

    region = depth[y1:y2, x1:x2]
    vals = region[np.isfinite(region)]

    if len(vals) == 0:
        return np.nan

    return float(np.median(vals))


# ------------------------------------------------------------
# 3. Road-contact confidence
# ------------------------------------------------------------

def compute_road_contact_confidence(road_mask, bbox):
    """
    road_mask: bool array, True = road
    bbox: x1, y1, x2, y2 in pixel coordinates

    Returns rule-based confidence that vehicle is on road.
    """
    h, w = road_mask.shape
    x1, y1, x2, y2 = map(float, bbox)

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return {
            "road_contact_confidence": 0.0,
            "below_road_fraction": np.nan,
            "bottom_band_road_fraction": np.nan,
            "surrounding_road_fraction": np.nan,
            "road_contact_reason": "invalid_bbox",
        }

    strip_h = int(np.clip(0.10 * bh, 5, 50))

    # Below vehicle
    below_box = (x1, y2, x2, y2 + strip_h)

    # Bottom 25% inside bbox
    bottom_y1 = y1 + 0.75 * bh
    bottom_band_box = (x1, bottom_y1, x2, y2)

    # Surrounding lower region
    pad_x = 0.15 * bw
    pad_y = 0.15 * bh
    surrounding_box = (
        x1 - pad_x,
        bottom_y1 - pad_y,
        x2 + pad_x,
        y2 + strip_h,
    )

    below_frac = region_fraction(road_mask, below_box)
    bottom_frac = region_fraction(road_mask, bottom_band_box)
    surround_frac = region_fraction(road_mask, surrounding_box)

    b = 0.0 if np.isnan(below_frac) else below_frac
    bb = 0.0 if np.isnan(bottom_frac) else bottom_frac
    s = 0.0 if np.isnan(surround_frac) else surround_frac

    if b >= 0.30:
        conf = 1.0
        reason = "road_visible_below_vehicle"
    elif s >= 0.30:
        conf = 0.75
        reason = "road_visible_around_vehicle"
    elif bb >= 0.15:
        conf = 0.50
        reason = "some_road_in_bottom_band"
    else:
        conf = 0.0
        reason = "no_road_contact_evidence"

    return {
        "road_contact_confidence": conf,
        "below_road_fraction": below_frac,
        "bottom_band_road_fraction": bottom_frac,
        "surrounding_road_fraction": surround_frac,
        "road_contact_reason": reason,
    }


# ------------------------------------------------------------
# 4. Visibility factor
# ------------------------------------------------------------

def compute_visibility_factor(bbox, image_w, image_h):
    """
    Simple visibility approximation from bbox truncation.
    """
    x1, y1, x2, y2 = map(float, bbox)
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return 0.0, "invalid_bbox"

    margin = 3

    touches_left = x1 <= margin
    touches_right = x2 >= image_w - margin
    touches_top = y1 <= margin
    touches_bottom = y2 >= image_h - margin

    edge_touches = sum([touches_left, touches_right, touches_top, touches_bottom])

    if edge_touches == 0:
        return 1.0, "fully_inside_frame"

    if edge_touches == 1:
        return 0.65, "touches_one_frame_edge"

    if edge_touches == 2:
        return 0.45, "touches_two_frame_edges"

    return 0.25, "heavily_truncated"


# ------------------------------------------------------------
# 5. Optional pixel-derived side-view length refinement
# ------------------------------------------------------------

def estimate_side_view_length_if_applicable(
    class_name,
    bbox,
    vehicle_depth_m,
    fx_px,
):
    """
    Uses bbox width as approximate vehicle length only if bbox is very wide
    and class is likely side-view/angled.

    This is optional refinement, not the default.
    """
    dims = get_vehicle_dims(class_name)
    if dims is None or not np.isfinite(vehicle_depth_m):
        return np.nan, False, "no_dims_or_depth"

    x1, y1, x2, y2 = map(float, bbox)
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return np.nan, False, "invalid_bbox"

    aspect = bw / bh

    # Very rough side-view heuristic.
    # Front/rear cars are usually not extremely wide;
    # side-view vehicles tend to have larger width/height.
    is_side_like = aspect >= 2.0

    if not is_side_like:
        return np.nan, False, "not_side_view_like"

    pixel_length_m = bw * vehicle_depth_m / fx_px

    standard_length = dims["length_m"]

    # Clip to plausible range to prevent bad depth/bbox explosions.
    min_len = 0.60 * standard_length
    max_len = 1.40 * standard_length
    clipped_length = float(np.clip(pixel_length_m, min_len, max_len))

    return clipped_length, True, "side_view_pixel_length_used"


# ------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--road-area-csv",
        required=True,
        help="CSV from depth-based road area batch.",
    )

    parser.add_argument(
        "--detections-csv",
        required=True,
        help="Vehicle detection CSV with bbox columns.",
    )

    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Directory containing saved road masks from road-area batch.",
    )

    parser.add_argument(
        "--depth-dir",
        default=None,
        help="Optional directory containing saved depth npy files. If unavailable, vehicle depth will be skipped.",
    )

    parser.add_argument(
        "--fx",
        type=float,
        default=2212.7,
    )

    parser.add_argument(
        "--depth-cutoff-m",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Minimum detection confidence to include.",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
    )

    parser.add_argument(
        "--output-vehicle-detail-csv",
        required=True,
    )

    parser.add_argument(
        "--class-col",
        default="class_name",
    )

    parser.add_argument("--x1-col", default="x1")
    parser.add_argument("--y1-col", default="y1")
    parser.add_argument("--x2-col", default="x2")
    parser.add_argument("--y2-col", default="y2")
    parser.add_argument("--conf-col", default="confidence")

    args = parser.parse_args()

    road_df = pd.read_csv(args.road_area_csv)
    det_df = pd.read_csv(args.detections_csv)

    output_csv = Path(args.output_csv)
    output_detail_csv = Path(args.output_vehicle_detail_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_detail_csv.parent.mkdir(parents=True, exist_ok=True)

    mask_dir = Path(args.mask_dir)
    depth_dir = Path(args.depth_dir) if args.depth_dir else None

    required_det_cols = [
        "sample_index",
        "lens_id",
        args.class_col,
        args.x1_col,
        args.y1_col,
        args.x2_col,
        args.y2_col,
    ]

    for c in required_det_cols:
        if c not in det_df.columns:
            raise ValueError(f"Missing detection column: {c}")

    if args.conf_col in det_df.columns:
        det_df = det_df[pd.to_numeric(det_df[args.conf_col], errors="coerce") >= args.confidence_threshold].copy()

    frame_rows = []
    vehicle_rows = []

    # Use lens_id + sample_index as key.
    for _, frame in road_df.iterrows():
        sample_index = int(frame["sample_index"])
        lens_id = int(frame["lens_id"])

        frame_dets = det_df[
            (det_df["sample_index"] == sample_index)
            & (det_df["lens_id"] == lens_id)
        ].copy()

        visible_area = float(frame["estimated_road_area_m2"])

        # Resolve mask path.
        # The road-area batch script saved mask paths if save-masks was enabled.
        if "saved_road_mask_path" in frame and pd.notna(frame["saved_road_mask_path"]):
            mask_path = Path(str(frame["saved_road_mask_path"]))
        else:
            # fallback by searching sample/lens pattern
            pattern = f"sample_{sample_index:05d}_lens_{lens_id}_*_road_mask.png"
            matches = list(mask_dir.glob(pattern))
            mask_path = matches[0] if matches else None

        if mask_path is None or not mask_path.exists():
            print(f"[WARN] No mask for sample {sample_index}, lens {lens_id}")
            road_mask = None
            image_h = int(frame.get("area_image_height_px", 0))
            image_w = int(frame.get("area_image_width_px", 0))
        else:
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            road_mask = mask_img > 0
            image_h, image_w = road_mask.shape

        # Optional depth map.
        depth = None
        if depth_dir is not None:
            # Try saved depth path from road batch first.
            if "saved_depth_npy_path" in frame and pd.notna(frame["saved_depth_npy_path"]):
                dp = Path(str(frame["saved_depth_npy_path"]))
                if dp.exists():
                    depth = np.load(dp)

            if depth is None:
                pattern = f"sample_{sample_index:05d}_lens_{lens_id}_*_depth_m.npy"
                matches = list(depth_dir.glob(pattern))
                if matches:
                    depth = np.load(matches[0])

            if depth is not None and road_mask is not None and depth.shape[:2] != road_mask.shape:
                depth = cv2.resize(
                    depth.squeeze(),
                    (road_mask.shape[1], road_mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

        vehicle_occluded_area = 0.0
        vehicle_count_used = 0
        vehicle_count_detected = len(frame_dets)

        class_area_sums = {}

        for _, det in frame_dets.iterrows():
            class_name = det[args.class_col]
            dims = get_vehicle_dims(class_name)

            if dims is None:
                continue

            bbox = (
                det[args.x1_col],
                det[args.y1_col],
                det[args.x2_col],
                det[args.y2_col],
            )

            x1, y1, x2, y2 = map(float, bbox)

            if image_w <= 0 or image_h <= 0:
                continue

            visibility_factor, visibility_reason = compute_visibility_factor(
                bbox=bbox,
                image_w=image_w,
                image_h=image_h,
            )

            if road_mask is not None:
                contact = compute_road_contact_confidence(road_mask, bbox)
            else:
                contact = {
                    "road_contact_confidence": 0.5,
                    "below_road_fraction": np.nan,
                    "bottom_band_road_fraction": np.nan,
                    "surrounding_road_fraction": np.nan,
                    "road_contact_reason": "no_mask_default_uncertain",
                }

            # vehicle depth from bottom-center/lower band if depth is available
            vehicle_depth_m = np.nan
            depth_validity_factor = 1.0
            depth_reason = "depth_not_used"

            if depth is not None:
                bh = y2 - y1
                lower_y1 = y1 + 0.65 * bh
                lower_band_box = (x1, lower_y1, x2, y2)
                vehicle_depth_m = region_median_depth(depth, lower_band_box)

                if not np.isfinite(vehicle_depth_m):
                    depth_validity_factor = 0.0
                    depth_reason = "invalid_vehicle_depth"
                elif vehicle_depth_m > args.depth_cutoff_m:
                    depth_validity_factor = 0.0
                    depth_reason = "vehicle_beyond_depth_cutoff"
                else:
                    depth_validity_factor = 1.0
                    depth_reason = "vehicle_within_depth_cutoff"

            width_m = dims["width_m"]
            standard_length_m = dims["length_m"]
            standard_footprint_m2 = width_m * standard_length_m

            pixel_length_m, used_pixel_length, pixel_length_reason = (
                estimate_side_view_length_if_applicable(
                    class_name=class_name,
                    bbox=bbox,
                    vehicle_depth_m=vehicle_depth_m,
                    fx_px=args.fx,
                )
            )

            if used_pixel_length:
                length_used_m = pixel_length_m
                length_source = "pixel_estimated_side_view_length"
            else:
                length_used_m = standard_length_m
                length_source = "standard_class_length"

            footprint_used_m2 = width_m * length_used_m

            road_contact_conf = float(contact["road_contact_confidence"])

            hidden_area = (
                footprint_used_m2
                * visibility_factor
                * road_contact_conf
                * depth_validity_factor
            )

            vehicle_occluded_area += hidden_area

            if hidden_area > 0:
                vehicle_count_used += 1

            norm_class = normalize_class_name(class_name)
            class_area_sums[norm_class] = class_area_sums.get(norm_class, 0.0) + hidden_area

            detail = {
                "sample_index": sample_index,
                "lens_id": lens_id,
                "class_name": class_name,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_width_px": x2 - x1,
                "bbox_height_px": y2 - y1,
                "standard_width_m": width_m,
                "standard_length_m": standard_length_m,
                "standard_footprint_m2": standard_footprint_m2,
                "length_used_m": length_used_m,
                "length_source": length_source,
                "footprint_used_m2": footprint_used_m2,
                "visibility_factor": visibility_factor,
                "visibility_reason": visibility_reason,
                "road_contact_confidence": road_contact_conf,
                "road_contact_reason": contact["road_contact_reason"],
                "below_road_fraction": contact["below_road_fraction"],
                "bottom_band_road_fraction": contact["bottom_band_road_fraction"],
                "surrounding_road_fraction": contact["surrounding_road_fraction"],
                "vehicle_depth_m": vehicle_depth_m,
                "depth_validity_factor": depth_validity_factor,
                "depth_reason": depth_reason,
                "pixel_length_m": pixel_length_m,
                "pixel_length_reason": pixel_length_reason,
                "hidden_road_area_m2_est": hidden_area,
            }

            if args.conf_col in det.index:
                detail["detection_confidence"] = det[args.conf_col]

            vehicle_rows.append(detail)

        adjusted_area = visible_area + vehicle_occluded_area
        occlusion_fraction = (
            vehicle_occluded_area / adjusted_area if adjusted_area > 0 else np.nan
        )

        out_frame = frame.to_dict()
        out_frame["visible_road_area_m2_depth_est"] = visible_area
        out_frame["vehicle_occluded_road_area_m2_est"] = vehicle_occluded_area
        out_frame["occlusion_adjusted_road_area_m2_est"] = adjusted_area
        out_frame["vehicle_occlusion_fraction_of_adjusted_area"] = occlusion_fraction
        out_frame["vehicle_count_detected_for_occlusion"] = vehicle_count_detected
        out_frame["vehicle_count_used_for_occlusion"] = vehicle_count_used

        for cls, val in class_area_sums.items():
            safe_cls = cls.replace("-", "_")
            out_frame[f"occluded_area_m2_{safe_cls}"] = val

        frame_rows.append(out_frame)

    final_df = pd.DataFrame(frame_rows)
    detail_df = pd.DataFrame(vehicle_rows)

    final_df.to_csv(output_csv, index=False)
    detail_df.to_csv(output_detail_csv, index=False)

    print("\nSaved frame-level occlusion-adjusted road area:")
    print(output_csv)
    print("Shape:", final_df.shape)

    print("\nSaved vehicle-level details:")
    print(output_detail_csv)
    print("Shape:", detail_df.shape)

    print("\nFrame-level summary:")
    cols = [
        "visible_road_area_m2_depth_est",
        "vehicle_occluded_road_area_m2_est",
        "occlusion_adjusted_road_area_m2_est",
        "vehicle_occlusion_fraction_of_adjusted_area",
        "vehicle_count_detected_for_occlusion",
        "vehicle_count_used_for_occlusion",
    ]
    existing_cols = [c for c in cols if c in final_df.columns]
    print(final_df[existing_cols].describe().T)


if __name__ == "__main__":
    main()