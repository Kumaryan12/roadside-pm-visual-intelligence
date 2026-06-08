"""
Run fine-tuned IDD YOLO11m vehicle detector on preprocessed PM-density frames.

Purpose:
- Use Indian-road-specific YOLO detector trained/fine-tuned on IDD 15-class dataset.
- Extract vehicle-class counts from preprocessed camera frames.
- Save frame-level vehicle features for later particle-density / PM modeling.
- Also save object-level detection boxes for vehicle-footprint occlusion adjustment.

Important:
- This script preserves all metadata from the input manifest.
- This is required for the 1-second pipeline, where fields like:
  sensor_row_id, sensor_unix, frame_sample_unix, relative_time_sec
  must survive from extraction → preprocessing → detection → aggregation.

Input:
- Preprocessed frame manifest CSV.

Outputs:
- Frame-level vehicle features CSV.
- Object-level detection boxes CSV.
- Optional annotated images.
"""

from pathlib import Path
import argparse

import cv2
import pandas as pd
from ultralytics import YOLO


# ============================================================
# DEFAULT PATHS
# ============================================================

PROJECT_ROOT = Path.cwd()

DEFAULT_INPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")
DEFAULT_MODEL_PATH = Path("models/detectors/yolo11m_idd15_v1_best.pt")
DEFAULT_OUTPUT_CSV = Path("outputs/features/idd_vehicle_detections_frame_level_v2.csv")
DEFAULT_OBJECT_OUTPUT_CSV = Path("outputs/features/idd_vehicle_detections_object_level_v2.csv")
DEFAULT_ANNOTATED_DIR = Path("outputs/figures/idd_detector_annotations")


# ============================================================
# IDD CLASS CONFIG
# ============================================================

IDD_CLASS_NAMES = [
    "animal",
    "autorickshaw",
    "bicycle",
    "bus",
    "car",
    "caravan",
    "motorcycle",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "trailer",
    "train",
    "truck",
    "vehicle fallback",
]


PM_CLASS_MAP = {
    "autorickshaw": "auto_rickshaw",
    "bicycle": "bicycle",
    "bus": "bus",
    "car": "car",
    "caravan": "unknown_vehicle",
    "motorcycle": "motorcycle",
    "trailer": "truck",
    "truck": "truck",
    "vehicle fallback": "unknown_vehicle",

    # Ignored classes for PM/traffic-composition features
    "person": "ignore",
    "rider": "ignore",
    "animal": "ignore",
    "traffic light": "ignore",
    "traffic sign": "ignore",
    "train": "ignore",
}


PM_CLASSES = [
    "auto_rickshaw",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "truck",
    "unknown_vehicle",
]


# Initial hypothesis weights only. These are not calibrated emission factors.
EXHAUST_WEIGHTS_INITIAL = {
    "bicycle": 0.00,
    "motorcycle": 0.50,
    "auto_rickshaw": 0.90,
    "car": 1.00,
    "bus": 3.00,
    "truck": 4.00,
    "unknown_vehicle": 1.00,
}


# Initial hypothesis weights only. These are not calibrated resuspension factors.
RESUSPENSION_WEIGHTS_INITIAL = {
    "bicycle": 0.05,
    "motorcycle": 0.30,
    "auto_rickshaw": 0.60,
    "car": 1.00,
    "bus": 3.50,
    "truck": 4.50,
    "unknown_vehicle": 1.00,
}


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_frame_path(path_value: str) -> Path:
    """
    Resolve frame paths relative to the current project root.

    Supports:
    - absolute paths
    - relative paths like outputs/preprocessed_frames_v2/...
    """
    raw_path = Path(str(path_value))

    if raw_path.is_absolute():
        return raw_path

    return PROJECT_ROOT / raw_path


# ============================================================
# FRAME-LEVEL ROW INITIALIZATION
# ============================================================

def initialize_feature_row(row) -> dict:
    """
    Initialize frame-level detection row.

    Critical:
    Start from row.to_dict() so all upstream metadata is preserved.

    This preserves old-pipeline fields:
    - sample_index
    - sensor_timestamp
    - sample_unix
    - lens_id
    - matched_run_id
    - video_offset_sec

    And also 1-second pipeline fields:
    - sensor_row_id
    - sensor_unix
    - frame_sample_unix
    - relative_time_sec
    - frame_extraction_status
    - frame_extraction_error
    """
    resolved_frame_path = resolve_frame_path(row.get("processed_frame_path", ""))

    result = row.to_dict()

    result.update({
        "processed_frame_key": str(row.get("processed_frame_key", "")),
        "source_frame_key": row.get("source_frame_key", ""),
        "matched_run_id": row.get("matched_run_id", ""),
        "video_offset_sec": row.get("video_offset_sec", None),
        "lens_id": row.get("lens_id", ""),
        "processed_frame_path": str(resolved_frame_path),

        # Detection status
        "idd_detection_status": "success",
        "idd_detection_error": "",
        "idd_annotated_image_path": "",

        # Detection summary
        "idd_total_detections_raw": 0,
        "idd_total_vehicle_count": 0,
        "idd_total_pm_relevant_objects": 0,
        "idd_vehicle_box_area_ratio": 0.0,
        "idd_average_confidence": 0.0,
        "idd_max_confidence": 0.0,

        # Engineered vehicle groups
        "idd_heavy_vehicle_count": 0,
        "idd_motor_vehicle_count": 0,
        "idd_exhaust_proxy_initial": 0.0,
        "idd_resuspension_vehicle_proxy_initial": 0.0,
    })

    for cls in PM_CLASSES:
        result[f"idd_{cls}_count"] = 0

    for cls in IDD_CLASS_NAMES:
        result[f"idd_raw_{cls.replace(' ', '_')}_count"] = 0

    return result


def failed_feature_row(row, error: str) -> dict:
    result = initialize_feature_row(row)
    result["idd_detection_status"] = "failed"
    result["idd_detection_error"] = str(error)
    return result


# ============================================================
# ANNOTATION HELPER
# ============================================================

def save_annotated_prediction(pred, row, annotated_dir: Path) -> str:
    annotated_dir.mkdir(parents=True, exist_ok=True)

    key = str(row.get("processed_frame_key", "unknown"))
    lens_id = str(row.get("lens_id", "unknown"))

    out_path = annotated_dir / f"{key}_lens{lens_id}_idd_pred.jpg"

    annotated_image = pred.plot()
    cv2.imwrite(str(out_path), annotated_image)

    return str(out_path)


# ============================================================
# OBJECT-LEVEL ROW HELPER
# ============================================================

def make_object_row(
    row,
    det_index,
    cls_id,
    idd_class,
    pm_class,
    is_pm_relevant,
    conf_score,
    x1,
    y1,
    x2,
    y2,
    bbox_width,
    bbox_height,
    box_area,
    box_area_ratio,
    image_width,
    image_height,
    image_area,
):
    """
    Create one object-level detection row.

    Critical:
    Include upstream 1-second metadata so NMS and road-area scripts
    can preserve the sensor/window relationship.
    """
    return {
        # 1-second pipeline metadata
        "sensor_row_id": row.get("sensor_row_id", None),
        "sensor_unix": row.get("sensor_unix", None),
        "frame_sample_unix": row.get("frame_sample_unix", None),
        "relative_time_sec": row.get("relative_time_sec", None),
        "frame_extraction_status": row.get("frame_extraction_status", ""),
        "frame_extraction_error": row.get("frame_extraction_error", ""),

        # Original project metadata
        "sample_index": row.get("sample_index", None),
        "sensor_timestamp": row.get("sensor_timestamp", ""),
        "sample_unix": row.get("sample_unix", None),

        "processed_frame_key": str(row.get("processed_frame_key", "")),
        "source_frame_key": row.get("source_frame_key", ""),
        "matched_run_id": row.get("matched_run_id", ""),
        "video_offset_sec": row.get("video_offset_sec", None),
        "lens_id": row.get("lens_id", ""),
        "processed_frame_path": str(resolve_frame_path(row.get("processed_frame_path", ""))),

        # Object detection metadata
        "detection_index": det_index,
        "idd_class_id": cls_id,
        "idd_class_name": idd_class,
        "pm_class_name": pm_class,
        "is_pm_relevant_vehicle": bool(is_pm_relevant),

        "confidence": float(conf_score),

        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),

        "bbox_width": float(bbox_width),
        "bbox_height": float(bbox_height),
        "bbox_area": float(box_area),
        "bbox_area_ratio": float(box_area_ratio),

        "image_width": int(image_width),
        "image_height": int(image_height),
        "image_area": float(image_area),
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features_and_objects_from_prediction(
    pred,
    row,
    save_annotated: bool,
    annotated_dir: Path,
):
    """
    Returns:
    - frame-level feature row
    - object-level detection rows
    """
    result = initialize_feature_row(row)
    object_rows = []

    h, w = pred.orig_shape
    image_area = float(h * w)

    raw_counts = {cls: 0 for cls in IDD_CLASS_NAMES}
    pm_counts = {cls: 0 for cls in PM_CLASSES}

    total_box_area = 0.0
    confidences = []

    if save_annotated:
        result["idd_annotated_image_path"] = save_annotated_prediction(
            pred=pred,
            row=row,
            annotated_dir=annotated_dir,
        )

    if pred.boxes is not None and len(pred.boxes) > 0:
        for det_index, box in enumerate(pred.boxes):
            cls_id = int(box.cls.item())
            conf_score = float(box.conf.item())

            if cls_id < 0 or cls_id >= len(IDD_CLASS_NAMES):
                continue

            idd_class = IDD_CLASS_NAMES[cls_id]
            pm_class = PM_CLASS_MAP.get(idd_class, "ignore")

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bbox_width = max(0.0, x2 - x1)
            bbox_height = max(0.0, y2 - y1)
            box_area = bbox_width * bbox_height
            box_area_ratio = float(box_area / image_area) if image_area > 0 else 0.0

            raw_counts[idd_class] += 1

            is_pm_relevant = pm_class != "ignore"

            object_rows.append(
                make_object_row(
                    row=row,
                    det_index=det_index,
                    cls_id=cls_id,
                    idd_class=idd_class,
                    pm_class=pm_class,
                    is_pm_relevant=is_pm_relevant,
                    conf_score=conf_score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    bbox_width=bbox_width,
                    bbox_height=bbox_height,
                    box_area=box_area,
                    box_area_ratio=box_area_ratio,
                    image_width=w,
                    image_height=h,
                    image_area=image_area,
                )
            )

            if not is_pm_relevant:
                continue

            pm_counts[pm_class] += 1

            total_box_area += box_area
            confidences.append(conf_score)

    total_pm_objects = sum(pm_counts.values())

    result["idd_total_detections_raw"] = int(sum(raw_counts.values()))
    result["idd_total_vehicle_count"] = int(total_pm_objects)
    result["idd_total_pm_relevant_objects"] = int(total_pm_objects)

    result["idd_vehicle_box_area_ratio"] = (
        float(total_box_area / image_area) if image_area > 0 else 0.0
    )

    result["idd_average_confidence"] = (
        float(sum(confidences) / len(confidences)) if confidences else 0.0
    )

    result["idd_max_confidence"] = (
        float(max(confidences)) if confidences else 0.0
    )

    for cls, count in pm_counts.items():
        result[f"idd_{cls}_count"] = int(count)

    for cls, count in raw_counts.items():
        result[f"idd_raw_{cls.replace(' ', '_')}_count"] = int(count)

    result["idd_heavy_vehicle_count"] = int(
        pm_counts["bus"] + pm_counts["truck"]
    )

    result["idd_motor_vehicle_count"] = int(
        pm_counts["auto_rickshaw"]
        + pm_counts["motorcycle"]
        + pm_counts["car"]
        + pm_counts["bus"]
        + pm_counts["truck"]
        + pm_counts["unknown_vehicle"]
    )

    exhaust_proxy = 0.0
    for cls, weight in EXHAUST_WEIGHTS_INITIAL.items():
        exhaust_proxy += weight * pm_counts.get(cls, 0)

    resuspension_proxy = 0.0
    for cls, weight in RESUSPENSION_WEIGHTS_INITIAL.items():
        resuspension_proxy += weight * pm_counts.get(cls, 0)

    result["idd_exhaust_proxy_initial"] = float(exhaust_proxy)
    result["idd_resuspension_vehicle_proxy_initial"] = float(resuspension_proxy)

    return result, object_rows


# ============================================================
# FRAME PROCESSING
# ============================================================

def process_frame(
    model: YOLO,
    row,
    conf: float,
    imgsz: int,
    save_annotated: bool,
    annotated_dir: Path,
):
    frame_path = resolve_frame_path(row.get("processed_frame_path", ""))

    if not frame_path.exists():
        return failed_feature_row(row, f"frame_not_found: {frame_path}"), []

    # IDD class IDs relevant for vehicle/PM feature extraction:
    # 1 autorickshaw, 2 bicycle, 3 bus, 4 car, 5 caravan,
    # 6 motorcycle, 11 trailer, 13 truck, 14 vehicle fallback
    vehicle_class_ids = [1, 2, 3, 4, 5, 6, 11, 13, 14]

    predictions = model.predict(
        source=str(frame_path),
        conf=conf,
        imgsz=imgsz,
        classes=vehicle_class_ids,
        max_det=500,
        verbose=False,
    )

    if not predictions:
        return failed_feature_row(row, "no_prediction_returned"), []

    return extract_features_and_objects_from_prediction(
        pred=predictions[0],
        row=row,
        save_annotated=save_annotated,
        annotated_dir=annotated_dir,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--object-output-csv", default=str(DEFAULT_OBJECT_OUTPUT_CSV))

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)

    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--annotated-dir", default=str(DEFAULT_ANNOTATED_DIR))
    parser.add_argument("--annotated-limit", type=int, default=100)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing frame/object outputs and rerun all selected frames.",
    )

    args = parser.parse_args()

    input_manifest = Path(args.input_manifest)
    model_path = Path(args.model_path)
    output_csv = Path(args.output_csv)
    object_output_csv = Path(args.object_output_csv)
    annotated_dir = Path(args.annotated_dir)

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    object_output_csv.parent.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(input_manifest)

    required_columns = [
        "processed_frame_key",
        "processed_frame_path",
        "lens_id",
        "preprocess_status",
    ]

    missing_columns = [c for c in required_columns if c not in manifest_df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    frame_df = manifest_df[manifest_df["preprocess_status"] == "success"].copy()

    if args.limit is not None:
        frame_df = frame_df.head(args.limit).copy()

    print("\nIDD detector inference")
    print("=" * 60)
    print("Input manifest:", input_manifest)
    print("Model path:", model_path)
    print("Frame-level output CSV:", output_csv)
    print("Object-level output CSV:", object_output_csv)
    print("Frames:", len(frame_df))
    print("Confidence:", args.conf)
    print("Image size:", args.imgsz)
    print("Save annotated:", args.save_annotated)
    print("Annotated limit:", args.annotated_limit)
    print("Overwrite:", args.overwrite)

    model = YOLO(str(model_path))

    output_rows = []
    object_rows = []
    completed_keys = set()

    if output_csv.exists() and not args.overwrite:
        old_df = pd.read_csv(output_csv)

        if "processed_frame_key" in old_df.columns:
            completed_keys = set(old_df["processed_frame_key"].astype(str))
            output_rows = old_df.to_dict(orient="records")

        print("\nExisting frame-level output found.")
        print("Already processed:", len(completed_keys))

    if object_output_csv.exists() and not args.overwrite:
        old_obj_df = pd.read_csv(object_output_csv)
        object_rows = old_obj_df.to_dict(orient="records")

        print("Existing object-level output found.")
        print("Existing object rows:", len(object_rows))

    annotated_saved = sum(
        1
        for old_row in output_rows
        if str(old_row.get("idd_annotated_image_path", "")).strip()
    )

    processed_since_save = 0

    for _, row in frame_df.iterrows():
        key = str(row["processed_frame_key"])

        if key in completed_keys:
            continue

        save_this_annotated = (
            args.save_annotated
            and annotated_saved < args.annotated_limit
        )

        print(f"\nProcessing {len(completed_keys) + 1}/{len(frame_df)}: {key}")

        try:
            features, objects = process_frame(
                model=model,
                row=row,
                conf=args.conf,
                imgsz=args.imgsz,
                save_annotated=save_this_annotated,
                annotated_dir=annotated_dir,
            )
        except Exception as exc:
            features = failed_feature_row(row, str(exc))
            objects = []

        output_rows.append(features)
        object_rows.extend(objects)
        completed_keys.add(key)

        if str(features.get("idd_annotated_image_path", "")).strip():
            annotated_saved += 1

        print("  status:", features["idd_detection_status"])
        print("  auto:", features["idd_auto_rickshaw_count"])
        print("  car:", features["idd_car_count"])
        print("  motorcycle:", features["idd_motorcycle_count"])
        print("  bus:", features["idd_bus_count"])
        print("  truck:", features["idd_truck_count"])
        print("  objects saved:", len(objects))
        print("  exhaust proxy:", features["idd_exhaust_proxy_initial"])

        processed_since_save += 1

        if processed_since_save >= args.checkpoint_every:
            pd.DataFrame(output_rows).to_csv(output_csv, index=False)
            pd.DataFrame(object_rows).to_csv(object_output_csv, index=False)
            print("Checkpoint saved:", output_csv)
            print("Checkpoint object saved:", object_output_csv)
            processed_since_save = 0

    out_df = pd.DataFrame(output_rows)
    obj_df = pd.DataFrame(object_rows)

    out_df.to_csv(output_csv, index=False)
    obj_df.to_csv(object_output_csv, index=False)

    print("\nDone.")
    print("Saved frame-level:", output_csv)
    print("Frame shape:", out_df.shape)

    print("Saved object-level:", object_output_csv)
    print("Object shape:", obj_df.shape)

    print("\nDetection status:")
    print(out_df["idd_detection_status"].value_counts(dropna=False))

    print("\nTotal detected PM-relevant counts:")
    summary_cols = [
        "idd_auto_rickshaw_count",
        "idd_bicycle_count",
        "idd_bus_count",
        "idd_car_count",
        "idd_motorcycle_count",
        "idd_truck_count",
        "idd_unknown_vehicle_count",
    ]

    for col in summary_cols:
        if col in out_df.columns:
            print(col, int(out_df[col].sum()))

    if len(obj_df) > 0:
        print("\nObject-level class counts:")
        print(obj_df["idd_class_name"].value_counts(dropna=False))

        print("\nPM-relevant object counts:")
        print(obj_df["pm_class_name"].value_counts(dropna=False))


if __name__ == "__main__":
    main()