from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "Ultralytics is not installed. Install using: pip install ultralytics"
    ) from e


MANIFEST = Path("experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv")
MODEL_PATH = Path("models/detectors/yolo11m_idd15_v1_best.pt")
OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)


def normalize_class_name(name: str) -> str:
    return str(name).lower().replace("_", " ").replace("-", " ").strip()


def map_vehicle_class(class_name: str) -> str | None:
    """
    Flexible mapping because IDD/custom YOLO class names can differ slightly.
    We only keep traffic-relevant classes for source-proxy modelling.
    """
    name = normalize_class_name(class_name)

    if "car" in name:
        return "car"

    if "bus" in name:
        return "bus"

    if "truck" in name or "lorry" in name:
        return "truck"

    if "auto" in name or "rickshaw" in name or "autorickshaw" in name:
        return "auto"

    if "motorcycle" in name or "motorbike" in name or "bike" in name or "two wheeler" in name:
        return "motorcycle"

    if "bicycle" in name or "cycle" in name:
        return "bicycle"

    if "person" in name or "pedestrian" in name:
        return "person"

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=OUTDIR / "traqid_yolo_vehicle_features_front.csv")
    parser.add_argument("--img-col", type=str, default="front_path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="Use 'cpu', 'mps', or '0'. Leave empty for auto.")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    df = df[
        (df["front_exists"] == True)
        & (df["rear_exists"] == True)
        & (df["env_plausible"] == True)
    ].copy().reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("Rows to process:", len(df))
    print("Model:", args.model)
    print("Output:", args.output)

    model = YOLO(str(args.model))

    print("\nModel class names:")
    print(model.names)

    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        image_path = Path(r[args.img_col])

        base_row = {
            "row_id": int(r["row_id"]),
            "image_id": int(r["image_id"]),
            "created_at": r["created_at"],
            "front_path": str(image_path),
            "PM2.5": r["PM2.5"],
            "PM10": r["PM10"],
            "Temperature": r["Temperature"],
            "Humidity": r["Humidity"],
            "Season": r["Season"],
            "Day_or_Night": r["Day_or_Night"],
            "aqi_cat": r["aqi_cat"],
            "split_date_chrono": r["split_date_chrono"],
            "yolo_status": "success",
            "yolo_error": "",
            "car_count": 0,
            "bus_count": 0,
            "truck_count": 0,
            "auto_count": 0,
            "motorcycle_count": 0,
            "bicycle_count": 0,
            "person_count": 0,
            "total_vehicle_count": 0,
            "heavy_vehicle_count": 0,
            "two_wheeler_count": 0,
            "detections_total_raw": 0,
        }

        if not image_path.exists():
            base_row["yolo_status"] = "missing_image"
            base_row["yolo_error"] = "image_path_not_found"
            rows.append(base_row)
            continue

        try:
            kwargs = dict(
                source=str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )
            if args.device is not None:
                kwargs["device"] = args.device

            result = model.predict(**kwargs)[0]

            if result.boxes is None or len(result.boxes) == 0:
                rows.append(base_row)
                continue

            cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
            base_row["detections_total_raw"] = len(cls_ids)

            for cls_id in cls_ids:
                raw_name = model.names.get(cls_id, str(cls_id))
                mapped = map_vehicle_class(raw_name)

                if mapped is None:
                    continue

                col = f"{mapped}_count"
                if col in base_row:
                    base_row[col] += 1

            base_row["total_vehicle_count"] = (
                base_row["car_count"]
                + base_row["bus_count"]
                + base_row["truck_count"]
                + base_row["auto_count"]
                + base_row["motorcycle_count"]
                + base_row["bicycle_count"]
            )

            base_row["heavy_vehicle_count"] = (
                base_row["bus_count"] + base_row["truck_count"]
            )

            base_row["two_wheeler_count"] = (
                base_row["motorcycle_count"] + base_row["bicycle_count"]
            )

        except Exception as e:
            base_row["yolo_status"] = "error"
            base_row["yolo_error"] = repr(e)

        rows.append(base_row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    print("\nSaved:", args.output)
    print("Shape:", out.shape)

    print("\nYOLO status:")
    print(out["yolo_status"].value_counts(dropna=False))

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
    ]

    print("\nVehicle count summary:")
    print(out[count_cols].describe().T)

    print("\nPreview:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()