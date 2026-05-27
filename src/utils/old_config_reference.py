from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"

OUTPUT_DIR = BASE_DIR / "outputs"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
FEATURES_DIR = OUTPUT_DIR / "features"

# YOLO model
YOLO_MODEL_NAME = "yolo11s.pt"

# Vehicle classes available in COCO-style YOLO models
VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle"
}

# Road condition labels for zero-shot CLIP classification
ROAD_LABELS = [
    "clean paved asphalt road",
    "dusty paved road",
    "unpaved dirt road",
    "wet road",
    "road with construction dust",
    "road with heavy loose dust"
]

# Mapping road label to numerical dust score
ROAD_DUST_SCORE_MAP = {
    "clean paved asphalt road": 0.10,
    "wet road": 0.15,
    "dusty paved road": 0.60,
    "unpaved dirt road": 0.80,
    "road with construction dust": 0.90,
    "road with heavy loose dust": 1.00
}

# Confidence threshold for YOLO vehicle detections
VEHICLE_CONF_THRESHOLD = 0.25

# Sliced detection settings
USE_SLICED_DETECTION = True
TILE_SIZE = 640
TILE_OVERLAP = 0.20
IOU_MERGE_THRESHOLD = 0.35