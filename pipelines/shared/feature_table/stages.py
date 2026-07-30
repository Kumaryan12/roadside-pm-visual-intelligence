"""Stage contracts for the canonical roadside feature-table pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    dependencies: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    arguments: tuple[str, ...]
    description: str


STAGES = (
    Stage("extract_frames", "scripts/frame_extraction/01_extract_frames_from_sensor_timestamps.py", (), ("sensor_csv", "video_root"), ("extracted_manifest",), ("--sensor-csv", "{sensor_csv}", "--video-root", "{video_root}", "--output-root", "{extracted_frames}", "--output-manifest", "{extracted_manifest}", "--lenses", "{lenses}", "--skip-existing"), "Extract frames at aligned sensor timestamps."),
    Stage("preprocess_frames", "scripts/preprocessing/01_apply_lens_preprocessing.py", ("extract_frames",), ("extracted_manifest",), ("preprocessed_manifest",), ("--input-manifest", "{extracted_manifest}", "--output-manifest", "{preprocessed_manifest}", "--output-root", "{preprocessed_frames}", "--lenses", "{lenses}"), "Apply lens-specific preprocessing."),
    Stage("detect_vehicles", "scripts/vehicle_detection/01_run_idd_detector_on_processed_frames.py", ("preprocess_frames",), ("preprocessed_manifest", "yolo_model"), ("vehicle_frame", "vehicle_object"), ("--input-manifest", "{preprocessed_manifest}", "--model-path", "{yolo_model}", "--output-csv", "{vehicle_frame}", "--object-output-csv", "{vehicle_object}", "--conf", "{yolo_conf}", "--imgsz", "{yolo_imgsz}"), "Run fine-tuned IDD YOLO inference."),
    Stage("segment_road_features", "scripts/road_segmentation/03_extract_segformer_road_condition_features.py", ("preprocess_frames",), ("preprocessed_manifest", "road_model"), ("road_frame",), ("--input-manifest", "{preprocessed_manifest}", "--output-csv", "{road_frame}", "--lenses", "{road_lenses}"), "Extract SegFormer mask-restricted road features."),
    Stage("estimate_road_area", "scripts/road_area/10_batch_lens1_road_area_depth.py", ("preprocess_frames",), ("preprocessed_manifest", "road_model"), ("road_area_base",), ("--manifest", "{preprocessed_manifest}", "--lens-id", "{area_lens}", "--road-model-path", "{road_model}", "--depth-model-id", "{depth_model}", "--fx", "{fx}", "--fy", "{fy}", "--min-depth-m", "{min_depth}", "--max-depth-m", "{max_depth}", "--area-width", "{area_width}", "--output-dir", "{road_area_dir}", "--output-csv", "{road_area_base}", "{save_masks}", "{save_depth}"), "Estimate SegFormer + metric-depth road surface area."),
    Stage("nms_vehicle_objects", "scripts/road_area/10_filter_object_detections_nms.py", ("detect_vehicles",), ("vehicle_object",), ("vehicle_object_nms",), ("--input-csv", "{vehicle_object}", "--output-csv", "{vehicle_object_nms}", "--lens-id", "{area_lens}", "--confidence-threshold", "{yolo_conf}"), "Filter vehicle objects for road occlusion."),
    Stage("depth_gated_occlusion", "scripts/road_area/11c_add_vehicle_footprint_occlusion_area_depth_gated_v3.py", ("estimate_road_area", "nms_vehicle_objects"), ("road_area_base", "vehicle_object_nms"), ("road_area_v3", "road_area_vehicle_detail"), ("--road-area-csv", "{road_area_base}", "--detections-csv", "{vehicle_object_nms}", "--output-frame-csv", "{road_area_v3}", "--output-vehicle-detail-csv", "{road_area_vehicle_detail}", "--lens-id", "{area_lens}"), "Apply depth-gated vehicle-footprint occlusion v3."),
    Stage("normalize_road_area", "pipelines/shared/normalize_road_area_v3.py", ("depth_gated_occlusion",), ("road_area_v3",), ("road_area_canonical",), ("--input-csv", "{road_area_v3}", "--output-csv", "{road_area_canonical}"), "Normalize v3 road-area columns."),
    Stage("aggregate_vehicles", "scripts/feature_fusion/01_aggregate_vehicle_features_to_sensor_level.py", ("detect_vehicles",), ("sensor_csv", "vehicle_frame", "preprocessed_manifest"), ("vehicle_sensor",), ("--sensor-csv", "{sensor_csv}", "--detection-csv", "{vehicle_frame}", "--frame-manifest", "{preprocessed_manifest}", "--output-csv", "{vehicle_sensor}"), "Aggregate vehicle features to sensor rows."),
    Stage("aggregate_road", "scripts/feature_fusion/02_aggregate_road_features_to_sensor_level.py", ("segment_road_features",), ("sensor_csv", "road_frame"), ("road_sensor",), ("--sensor-csv", "{sensor_csv}", "--road-csv", "{road_frame}", "--output-csv", "{road_sensor}"), "Aggregate road-surface features to sensor rows."),
    Stage("fuse_visual", "scripts/feature_fusion/03_create_particle_density_modeling_table.py", ("aggregate_vehicles", "aggregate_road"), ("vehicle_sensor", "road_sensor"), ("visual_table",), ("--vehicle-csv", "{vehicle_sensor}", "--road-csv", "{road_sensor}", "--output-csv", "{visual_table}"), "Fuse sensor, vehicle, and road features."),
    Stage("extract_osm", "pipelines/shared/extract_osm_features.py", ("fuse_visual",), ("sensor_csv",), ("osm_csv",), ("--sensor-csv", "{sensor_csv}", "--output-csv", "{osm_csv}", "--lat-col", "{lat_col}", "--lon-col", "{lon_col}", "--sample-col", "{sample_col}", "--radius-m", "{osm_radius}", "--endpoint", "{osm_endpoint}"), "Extract cached OSM context from Overpass."),
    Stage("merge_osm", "scripts/feature_fusion/04_add_osm_features_to_modeling_table.py", ("fuse_visual", "extract_osm"), ("visual_table", "osm_csv"), ("osm_table",), ("--input-table", "{visual_table}", "--osm-csv", "{osm_csv}", "--output-csv", "{osm_table}"), "Merge extracted OSM features."),
    Stage("merge_road_area", "pipelines/shared/merge_road_area_features.py", ("merge_osm", "normalize_road_area"), ("osm_table", "road_area_canonical"), ("road_area_table",), ("--input-table", "{osm_table}", "--road-area-csv", "{road_area_canonical}", "--output-csv", "{road_area_table}"), "Merge canonical metric road-area features."),
    Stage("extract_alphaearth", "pipelines/shared/extract_alphaearth_features.py", ("merge_road_area",), ("sensor_csv",), ("alphaearth_csv",), ("--sensor-csv", "{sensor_csv}", "--output-csv", "{alphaearth_csv}", "--lat-col", "{lat_col}", "--lon-col", "{lon_col}", "--timestamp-col", "{timestamp_col}", "--sample-col", "{sample_col}", "--max-available-year", "{aef_max_year}", "{ee_project_flag}", "{ee_project}", "--buffers-m", "{aef_buffers}"), "Extract latest non-future AlphaEarth A00-A63 embeddings."),
    Stage("merge_alphaearth", "pipelines/shared/merge_alphaearth_features.py", ("merge_road_area", "extract_alphaearth"), ("road_area_table", "alphaearth_csv"), ("final_table",), ("--input-table", "{road_area_table}", "--alphaearth-csv", "{alphaearth_csv}", "--output-csv", "{final_table}"), "Merge AlphaEarth A00-A63 features."),
)


STAGE_BY_NAME = {stage.name: stage for stage in STAGES}
