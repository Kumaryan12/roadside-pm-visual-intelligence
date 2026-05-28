# Scripts

This folder contains only reproducible pipeline scripts.

## vehicle_detection

Scripts related to running vehicle detectors on processed PM-density frames.

### 01_run_idd_detector_on_processed_frames.py

Runs the fine-tuned YOLO11m detector trained on the Indian Driving Dataset detection subset.

Main output:

outputs/features/idd_vehicle_detections_processed_frames.csv

Optional visual verification output:

outputs/figures/idd_detector_annotations/

### 02_preview_idd_detections.py

Creates annotated prediction images for a small number of processed frames so detector quality can be visually verified before running full inference.
