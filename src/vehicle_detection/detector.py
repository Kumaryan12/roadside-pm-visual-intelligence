from typing import Optional
import cv2
from ultralytics import YOLO

from src.config import YOLO_MODEL_NAME, VEHICLE_CLASSES, VEHICLE_CONF_THRESHOLD


class VehicleDetector:
    def __init__(self, model_name: str = YOLO_MODEL_NAME):
        self.model = YOLO(model_name)

    def _is_platform_false_detection(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int,
        class_name: str,
    ) -> bool:
        

        if image_width <= 0 or image_height <= 0:
            return False

        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)

        box_area = float(box_width * box_height)
        image_area = float(image_width * image_height)

        box_area_ratio = box_area / image_area
        box_width_ratio = box_width / float(image_width)

        box_center_y = (y1 + y2) / 2.0
        bottom_touch_ratio = y2 / float(image_height)

        # Platform false detections are usually car-like.
        # Do not apply this aggressive filter to motorcycle/bicycle,
        # because nearby two-wheelers may naturally appear low in the image.
        car_like_classes = {"car", "truck", "bus"}

        if class_name not in car_like_classes:
            return False

        # Main platform rule:
        # large/wide box in the lower region
        if box_center_y > 0.72 * image_height and (
            box_area_ratio > 0.10 or box_width_ratio > 0.50
        ):
            return True

        # Additional rule:
        # box touches the bottom and is very wide
        if bottom_touch_ratio > 0.93 and box_width_ratio > 0.40:
            return True

        return False

    def detect(self, image_path: str, save_annotated_path: Optional[str] = None) -> dict:
        image = cv2.imread(image_path)

        image_area = 1.0
        height = 0
        width = 0

        if image is not None:
            height, width = image.shape[:2]
            image_area = float(height * width)

        results = self.model(
            image_path,
            conf=VEHICLE_CONF_THRESHOLD
        )

        result = results[0]

        counts = {
            "car_count": 0,
            "motorcycle_count": 0,
            "bus_count": 0,
            "truck_count": 0,
            "bicycle_count": 0
        }

        detections = []
        filtered_detections = []

        total_vehicle_box_area = 0.0
        confidence_sum = 0.0

        small_vehicle_count = 0
        medium_vehicle_count = 0
        large_vehicle_count = 0

        raw_vehicle_detection_count = 0
        platform_filtered_count = 0

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls_id]

                if class_name not in VEHICLE_CLASSES:
                    continue

                raw_vehicle_detection_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                is_platform_false_detection = False

                if is_platform_false_detection:
                    platform_filtered_count += 1

                    filtered_detections.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "filter_reason": "platform_or_camera_roof_false_detection",
                        "source": "full"
                    })

                    # Draw filtered boxes in a different color for debugging.
                    if image is not None and save_annotated_path is not None:
                        label = f"FILTERED {class_name} {conf:.2f}"

                        cv2.rectangle(
                            image,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            image,
                            label,
                            (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2
                        )

                    continue

                key = f"{class_name}_count"

                if key not in counts:
                    # Safety fallback in case VEHICLE_CLASSES contains a class
                    # that does not have a corresponding count key.
                    continue

                counts[key] += 1

                box_width = max(0, x2 - x1)
                box_height = max(0, y2 - y1)
                box_area = float(box_width * box_height)
                box_area_ratio = box_area / image_area

                total_vehicle_box_area += box_area
                confidence_sum += conf

                if box_area_ratio < 0.005:
                    small_vehicle_count += 1
                elif box_area_ratio < 0.03:
                    medium_vehicle_count += 1
                else:
                    large_vehicle_count += 1

                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "box_area": box_area,
                    "box_area_ratio": box_area_ratio,
                    "source": "full"
                })

                if image is not None and save_annotated_path is not None:
                    label = f"{class_name} {conf:.2f}"

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        image,
                        label,
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

        total_vehicles = sum(counts.values())

        if total_vehicles > 0:
            average_vehicle_confidence = confidence_sum / total_vehicles
        else:
            average_vehicle_confidence = 0.0

        vehicle_box_area_ratio = total_vehicle_box_area / image_area

        if image is not None and save_annotated_path is not None:
            cv2.imwrite(save_annotated_path, image)

        return {
            "total_vehicles": total_vehicles,
            **counts,

            "vehicle_box_area_ratio": vehicle_box_area_ratio,
            "average_vehicle_confidence": average_vehicle_confidence,
            "small_vehicle_count": small_vehicle_count,
            "medium_vehicle_count": medium_vehicle_count,
            "large_vehicle_count": large_vehicle_count,

            # Detection diagnostics
            "raw_vehicle_detection_count": raw_vehicle_detection_count,
            "platform_filtered_count": platform_filtered_count,

            # For compatibility with pipeline.py
            "full_source_detection_count": total_vehicles,
            "tile_source_detection_count": 0,
            "raw_detection_count_before_merge": raw_vehicle_detection_count,
            "final_detection_count_after_merge": total_vehicles,

            "vehicle_detections": detections,
            "filtered_vehicle_detections": filtered_detections
        }