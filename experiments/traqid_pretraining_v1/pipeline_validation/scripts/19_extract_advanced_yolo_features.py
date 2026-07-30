from pathlib import Path
import argparse
import time

import numpy as np
import pandas as pd
from PIL import Image

from ultralytics import YOLO


# ---------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------
DEFAULT_MANIFEST = Path(
    "experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv"
)

DEFAULT_MODEL = Path("models/detectors/yolo11m_idd15_v1_best.pt")

DEFAULT_OUT = Path(
    "experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_advanced_features_front.csv"
)


# ---------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------
def map_vehicle_class(class_name):
    """
    Map detector-specific class names to broader interpretable groups.
    This is intentionally flexible because IDD/YOLO class names may vary.
    """

    if class_name is None:
        return None

    name = str(class_name).lower().strip()

    # Heavy / public transport
    if "bus" in name:
        return "bus"
    if "truck" in name or "lorry" in name:
        return "truck"

    # Indian traffic classes
    if "auto" in name or "rickshaw" in name or "autorickshaw" in name:
        return "auto"

    # Two-wheelers
    if "motor" in name or "bike" in name or "scooter" in name or "two" in name:
        return "motorcycle"
    if "bicycle" in name or "cycle" in name:
        return "bicycle"

    # Common objects
    if "car" in name or "jeep" in name or "van" in name:
        return "car"
    if "person" in name or "pedestrian" in name or "rider" in name:
        return "person"

    return None


GROUPS = [
    "car",
    "bus",
    "truck",
    "auto",
    "motorcycle",
    "bicycle",
    "person",
]

VEHICLE_GROUPS = [
    "car",
    "bus",
    "truck",
    "auto",
    "motorcycle",
    "bicycle",
]

HEAVY_GROUPS = [
    "bus",
    "truck",
]

TWO_WHEELER_GROUPS = [
    "motorcycle",
    "bicycle",
]


def empty_feature_dict():
    d = {
        "yolo_adv_status": "success",
        "yolo_adv_error": "",
        "image_width": np.nan,
        "image_height": np.nan,
        "image_area": np.nan,
        "detections_total_raw": 0,
    }

    for g in GROUPS:
        d[f"{g}_count"] = 0
        d[f"{g}_area_ratio"] = 0.0
        d[f"{g}_conf_sum"] = 0.0
        d[f"{g}_conf_area_ratio"] = 0.0
        d[f"{g}_near_count"] = 0
        d[f"{g}_near_area_ratio"] = 0.0

    aggregate = {
        "total_vehicle_count": 0,
        "total_vehicle_area_ratio": 0.0,
        "total_vehicle_conf_sum": 0.0,
        "total_vehicle_conf_area_ratio": 0.0,
        "heavy_vehicle_count": 0,
        "heavy_vehicle_area_ratio": 0.0,
        "heavy_vehicle_conf_area_ratio": 0.0,
        "two_wheeler_count": 0,
        "two_wheeler_area_ratio": 0.0,
        "two_wheeler_conf_area_ratio": 0.0,
        "near_vehicle_count": 0,
        "near_vehicle_area_ratio": 0.0,
        "near_heavy_vehicle_count": 0,
        "near_heavy_vehicle_area_ratio": 0.0,
        "near_two_wheeler_count": 0,
        "near_two_wheeler_area_ratio": 0.0,
        "mean_vehicle_box_area_ratio": 0.0,
        "max_vehicle_box_area_ratio": 0.0,
        "mean_vehicle_confidence": 0.0,
        "max_vehicle_confidence": 0.0,
        "vehicle_area_share_heavy": 0.0,
        "vehicle_area_share_two_wheeler": 0.0,
        "vehicle_area_share_auto": 0.0,
        "vehicle_count_share_heavy": 0.0,
        "vehicle_count_share_two_wheeler": 0.0,
        "vehicle_count_share_auto": 0.0,
        "traffic_mix_count_index": 0.0,
        "traffic_mix_area_index": 0.0,
        "near_traffic_mix_count_index": 0.0,
        "near_traffic_mix_area_index": 0.0,
        "bottom_half_vehicle_count": 0,
        "bottom_half_vehicle_area_ratio": 0.0,
        "center_lane_vehicle_count": 0,
        "center_lane_vehicle_area_ratio": 0.0,
        "left_half_vehicle_count": 0,
        "right_half_vehicle_count": 0,
    }

    d.update(aggregate)
    return d


def extract_features_from_result(result, image_path, near_y_threshold=0.60):
    """
    Extract count, area, near-field, and confidence-weighted features from one YOLO result.
    """

    feat = empty_feature_dict()

    try:
        with Image.open(image_path) as im:
            width, height = im.size

        image_area = float(width * height)
        feat["image_width"] = width
        feat["image_height"] = height
        feat["image_area"] = image_area

        if result.boxes is None or len(result.boxes) == 0:
            return feat

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        names = result.names
        vehicle_box_areas = []
        vehicle_confs = []

        for i in range(len(xyxy)):
            class_id = int(cls[i])
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            group = map_vehicle_class(class_name)

            if group is None:
                continue

            x1, y1, x2, y2 = xyxy[i]
            c = float(conf[i])

            bw = max(0.0, float(x2 - x1))
            bh = max(0.0, float(y2 - y1))
            box_area = bw * bh
            area_ratio = box_area / image_area if image_area > 0 else 0.0

            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            cy_norm = cy / height if height > 0 else 0.0
            cx_norm = cx / width if width > 0 else 0.0

            is_near = cy_norm >= near_y_threshold
            is_bottom_half = cy_norm >= 0.50
            is_center_lane = 0.33 <= cx_norm <= 0.67
            is_left = cx_norm < 0.50
            is_right = cx_norm >= 0.50

            feat["detections_total_raw"] += 1

            feat[f"{group}_count"] += 1
            feat[f"{group}_area_ratio"] += area_ratio
            feat[f"{group}_conf_sum"] += c
            feat[f"{group}_conf_area_ratio"] += c * area_ratio

            if is_near:
                feat[f"{group}_near_count"] += 1
                feat[f"{group}_near_area_ratio"] += area_ratio

            if group in VEHICLE_GROUPS:
                vehicle_box_areas.append(area_ratio)
                vehicle_confs.append(c)

                feat["total_vehicle_count"] += 1
                feat["total_vehicle_area_ratio"] += area_ratio
                feat["total_vehicle_conf_sum"] += c
                feat["total_vehicle_conf_area_ratio"] += c * area_ratio

                if is_near:
                    feat["near_vehicle_count"] += 1
                    feat["near_vehicle_area_ratio"] += area_ratio

                if is_bottom_half:
                    feat["bottom_half_vehicle_count"] += 1
                    feat["bottom_half_vehicle_area_ratio"] += area_ratio

                if is_center_lane:
                    feat["center_lane_vehicle_count"] += 1
                    feat["center_lane_vehicle_area_ratio"] += area_ratio

                if is_left:
                    feat["left_half_vehicle_count"] += 1
                if is_right:
                    feat["right_half_vehicle_count"] += 1

            if group in HEAVY_GROUPS:
                feat["heavy_vehicle_count"] += 1
                feat["heavy_vehicle_area_ratio"] += area_ratio
                feat["heavy_vehicle_conf_area_ratio"] += c * area_ratio

                if is_near:
                    feat["near_heavy_vehicle_count"] += 1
                    feat["near_heavy_vehicle_area_ratio"] += area_ratio

            if group in TWO_WHEELER_GROUPS:
                feat["two_wheeler_count"] += 1
                feat["two_wheeler_area_ratio"] += area_ratio
                feat["two_wheeler_conf_area_ratio"] += c * area_ratio

                if is_near:
                    feat["near_two_wheeler_count"] += 1
                    feat["near_two_wheeler_area_ratio"] += area_ratio

        total_count = feat["total_vehicle_count"]
        total_area = feat["total_vehicle_area_ratio"]
        eps = 1e-6

        if vehicle_box_areas:
            feat["mean_vehicle_box_area_ratio"] = float(np.mean(vehicle_box_areas))
            feat["max_vehicle_box_area_ratio"] = float(np.max(vehicle_box_areas))

        if vehicle_confs:
            feat["mean_vehicle_confidence"] = float(np.mean(vehicle_confs))
            feat["max_vehicle_confidence"] = float(np.max(vehicle_confs))

        feat["vehicle_area_share_heavy"] = feat["heavy_vehicle_area_ratio"] / (total_area + eps)
        feat["vehicle_area_share_two_wheeler"] = feat["two_wheeler_area_ratio"] / (total_area + eps)
        feat["vehicle_area_share_auto"] = feat["auto_area_ratio"] / (total_area + eps)

        feat["vehicle_count_share_heavy"] = feat["heavy_vehicle_count"] / (total_count + eps)
        feat["vehicle_count_share_two_wheeler"] = feat["two_wheeler_count"] / (total_count + eps)
        feat["vehicle_count_share_auto"] = feat["auto_count"] / (total_count + eps)

        # Count-based traffic mix
        feat["traffic_mix_count_index"] = (
            2.0 * feat["heavy_vehicle_count"]
            + 1.5 * feat["auto_count"]
            + 1.0 * feat["car_count"]
            + 0.8 * feat["two_wheeler_count"]
        )

        # Area-based traffic mix
        feat["traffic_mix_area_index"] = (
            2.0 * feat["heavy_vehicle_area_ratio"]
            + 1.5 * feat["auto_area_ratio"]
            + 1.0 * feat["car_area_ratio"]
            + 0.8 * feat["two_wheeler_area_ratio"]
        )

        feat["near_traffic_mix_count_index"] = (
            2.0 * feat["near_heavy_vehicle_count"]
            + 1.5 * feat["auto_near_count"]
            + 1.0 * feat["car_near_count"]
            + 0.8 * feat["near_two_wheeler_count"]
        )

        feat["near_traffic_mix_area_index"] = (
            2.0 * feat["near_heavy_vehicle_area_ratio"]
            + 1.5 * feat["auto_near_area_ratio"]
            + 1.0 * feat["car_near_area_ratio"]
            + 0.8 * feat["near_two_wheeler_area_ratio"]
        )

        return feat

    except Exception as e:
        feat = empty_feature_dict()
        feat["yolo_adv_status"] = "failed"
        feat["yolo_adv_error"] = str(e)
        return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--image-col", type=str, default="front_path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--near-y", type=float, default=0.60)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    model_path = Path(args.model)
    out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading manifest:", manifest_path)
    df = pd.read_csv(manifest_path)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("Rows:", len(df))
    print("Loading YOLO model:", model_path)

    model = YOLO(str(model_path))
    print("Model names:", model.names)

    rows = []
    start = time.time()

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"Processing {idx}/{len(df)} elapsed={time.time() - start:.1f}s")

        image_path = Path(str(row[args.image_col]))

        base = {
            "row_id": row.get("row_id", idx),
            "image_id": row.get("image_id", row.get("Image", np.nan)),
            "created_at": row.get("created_at", ""),
            "front_path": row.get("front_path", ""),
            "PM2.5": row.get("PM2.5", np.nan),
            "PM10": row.get("PM10", np.nan),
            "Temperature": row.get("Temperature", np.nan),
            "Humidity": row.get("Humidity", np.nan),
            "Season": row.get("Season", ""),
            "Day_or_Night": row.get("Day_or_Night", ""),
            "aqi_cat": row.get("aqi_cat", ""),
            "split_date_chrono": row.get("split_date_chrono", ""),
        }

        if not image_path.exists():
            feat = empty_feature_dict()
            feat["yolo_adv_status"] = "failed"
            feat["yolo_adv_error"] = f"image_not_found: {image_path}"
        else:
            try:
                results = model.predict(
                    source=str(image_path),
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
                feat = extract_features_from_result(
                    results[0],
                    image_path=image_path,
                    near_y_threshold=args.near_y,
                )
            except Exception as e:
                feat = empty_feature_dict()
                feat["yolo_adv_status"] = "failed"
                feat["yolo_adv_error"] = str(e)

        base.update(feat)
        rows.append(base)

    out = pd.DataFrame(rows)

    print("\nShape:", out.shape)
    print("\nStatus:")
    print(out["yolo_adv_status"].value_counts(dropna=False))

    count_cols = [
        "car_count",
        "bus_count",
        "truck_count",
        "auto_count",
        "motorcycle_count",
        "bicycle_count",
        "person_count",
        "total_vehicle_count",
        "heavy_vehicle_count",
        "two_wheeler_count",
        "near_vehicle_count",
        "near_heavy_vehicle_count",
    ]

    print("\nCount summary:")
    print(out[count_cols].describe().to_string())

    print("\nArea summary:")
    area_cols = [
        "total_vehicle_area_ratio",
        "heavy_vehicle_area_ratio",
        "two_wheeler_area_ratio",
        "near_vehicle_area_ratio",
        "traffic_mix_area_index",
        "near_traffic_mix_area_index",
    ]
    print(out[area_cols].describe().to_string())

    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()