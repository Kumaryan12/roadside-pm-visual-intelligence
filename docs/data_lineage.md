# Data lineage

This document describes lineage recoverable from static code, CLI defaults, filenames, and committed reports as of 2026-07-11. Arrows mean “is read to produce.” Optional or uncertain edges are marked `(?)`. See `experiment_registry.csv` for per-script imports, inputs, outputs, and static callers.

## Shared roadside/MUMMA feature lineage

The canonical executable registry for this lineage is
`pipelines/shared/feature_table/run.py`, configured by
`configs/pipelines/feature_table_v1.yaml`. Each run writes isolated frames,
features, tables, commands, checksums, and stage statuses under
`artifacts/runs/<run_id>/`; historical scripts remain the underlying stage
implementations during migration.

```text
sensor CSV + source videos
  -> scripts/frame_extraction/01_extract_frames_from_sensor_timestamps.py
  -> extracted-frame images + manifest
  -> scripts/preprocessing/01_apply_lens_preprocessing.py
  -> preprocessed images + manifest

preprocessed images + IDD YOLO weights
  -> scripts/vehicle_detection/01_run_idd_detector_on_processed_frames.py
  -> outputs/features/idd_vehicle_detections_processed_frames.csv
  -> scripts/feature_fusion/01_aggregate_vehicle_features_to_sensor_level.py
  -> sensor-level vehicle features

preprocessed images + SegFormer road weights
  -> scripts/road_segmentation/03_extract_segformer_road_condition_features.py
  -> road masks/condition features
  -> scripts/feature_fusion/02_aggregate_road_features_to_sensor_level.py
  -> sensor-level road features

lens-1 images + road mask + metric depth + vehicle detections
  -> scripts/road_area/10_batch_lens1_road_area_depth.py
  -> scripts/road_area/10_filter_object_detections_nms.py
  -> 11 / 11b / 11c occlusion alternatives
  -> scripts/road_area/12_make_conservative_occlusion_adjustment.py
  -> scripts/road_area/14_finalize_road_area_features.py
  -> scripts/road_area/15_merge_road_area_features_into_modeling_table.py

sensor + vehicle + road
  -> scripts/feature_fusion/03_create_particle_density_modeling_table.py
  -> scripts/feature_fusion/04_add_osm_features_to_modeling_table.py
  -> scripts/feature_fusion/05_add_effective_density_to_csv.py
  -> final modeling table
```

The canonical completion order is now: extract timestamp-aligned frames,
preprocess lenses, run IDD YOLO, extract SegFormer road-surface features,
estimate metric-depth road area, apply NMS and depth-gated vehicle occlusion,
aggregate frame features to sensor rows, fuse visual features, merge OSM,
merge canonical road area, and merge AlphaEarth A00-A63 embeddings.

OSM is extracted for cached unique rounded coordinates through Overpass, then
expanded back to sensor rows. AlphaEarth is extracted from the annual Earth
Engine collection using the measurement year, producing A00-A63 point values
and configured buffer mean/standard-deviation features. AlphaEarth requires an
authenticated Earth Engine account/project; OSM requires network access to the
configured Overpass endpoint.

The exact manifest filenames vary by CLI defaults and run IDs. The road-area `11`, `11b`, and `11c` scripts are alternative versions (original footprint, bbox-visibility v2, depth-gated v3), not a strict sequential chain. **Uncertainty: medium** on which output was used in each downstream report.

## One-second pipeline

```text
sensor table
  -> 00_audit_runs_for_best_window.py / 00_make_sensor_window_subset.py
  -> selected sensor windows
  -> 01_extract_1s_frames_from_sensor_windows.py
  -> 1-second frame manifests/images
  -> [vehicle, road, depth feature extraction not present as numbered 02/03 scripts] (?)
  -> 04_aggregate_1s_features_to_sensor_level.py
  -> 05_merge_1s_features_with_sensor_data.py
  -> 06_analyze_1s_extra_information.py
  -> 07_lag_window_analysis.py
  -> 08_generate_lag_window_report_artifacts.py
```

The missing numbered feature stages are a concrete reproducibility gap. Existing `outputs/pipeline_1s/vehicle_detections`, `road_features`, `road_segmentation`, and `depth` directories imply those stages ran, but their exact commands are not recoverable from the repository. **Uncertainty: high.**

## MUMMA 281-sample lineage

```text
data/processed/final_feature_table.csv (281 rows)
  -> 01_pm25_model_from_final_feature_table.py
     -> ablation results/predictions + feature groups
  -> 02_engineered_fair_pm25_models.py
     -> fair engineered results + importances/groups
  -> 03_residual_fusion_pm25.py
     -> residual-fusion results/predictions/groups
  -> 04_fair_model_optimization.py
     -> optimized results/predictions/selected features

final feature table
  -> 05_temporal_t7_fair_pm25.py
  -> 05b_temporal_t7_random2fold_fair_pm25.py
     -> temporal comparison reports

final feature table + referenced frame images
  -> 06_extract_resnet50_embeddings.py
     -> embeddings/resnet50_avgpool_embeddings.npy + index/summary
  -> 07_resnet_t7_random2fold_lstm_fair_pm25.py
     -> ResNet-T7 results/predictions

final feature table
  -> 08_analyse_actual_pm25_data.py
  -> duplicate/spike/lag/summary diagnostics
```

Best fair reported candidate is the optimized residual top-30 source model from stage 04 (R² 0.6499). The ResNet branch is a comparison (best R² 0.4250), not an improvement. The near-perfect `upper_bound_all_sensor_visual_osm` result includes an intentionally unfair feature set and is excluded from the recommendation.

## TRAQID core and paper-replication lineage

```text
TRAQID.csv + front/rear images
  -> 01_audit_traqid_full.py / 01_traqid_sanity_report.py
  -> 02_build_traqid_paired_manifest.py
  -> paired manifest
     -> 03 EDA
     -> 04 split creation
     -> 05 tabular baseline
     -> 06 generic image embeddings
     -> 07 embedding baselines
     -> 09 supervised MobileNet PM2.5 -> 10 supervised embeddings

paired/split manifest + embeddings
  -> 11 T7 sequence builders
  -> 12 embedding GRU / split-first ResNet50-GRU
  -> 13 purged split and single-image MLP
  -> 15 front/rear fusion and YOLO-road-tabular model
  -> 16/17/18 grouped-date and tabular-fusion branches

paired manifest
  -> 19_build_paper_style_T_sequence_manifest.py
  -> T2..T9 sequence manifests
  -> 20_extract_paper_cnn_features.py
  -> VGG16/ResNet50 front and rear feature arrays/indexes
  -> 22_fuse_paper_front_rear_features.py
  -> concat/mean fused embeddings
  -> 21_train_paper_cnn_lstm_multitarget.py
  -> 23_train_paper_cnn_lstm_tabular_fusion.py
  -> model checkpoints + predictions + metrics
  -> 24/25 paper assets and corrected tables
```

### Leakage-corrected validation

```text
paper T7 sequence manifest
  -> 28_analyze_sequence_overlap_leakage.py / 31_sequence_overlap_diagnostic.py
  -> overlap reports
  -> 29_check_purged_block_time_confounds.py
  -> 30_build_time_balanced_purged_split.py
  -> time-balanced, zero-frame-overlap split
  -> 23_train_paper_cnn_lstm_tabular_fusion.py
  -> time-balanced purged metrics
  -> 31_make_naive_baseline_split_table.py
  -> 32_make_final_validation_ladder_table.py
  -> reports/paper_final/*
```

The high random and two-fold results are downstream of overlapping sliding windows. For defensible estimation, the lineage terminates at the time-balanced purged and chronological reports. The random split remains useful only as paper replication.

## Canonical ResNet50 temporal branch

```text
preprocessed frame manifest + one-row-per-sample table with fixed splits
  -> pipelines/image_embeddings/extract_resnet50.py
  -> resnet50_embeddings.npy + embedding_index.csv
  -> pipelines/image_embeddings/build_sequences.py
  -> windows contained within one trip/group and one preassigned split
  -> pipelines/image_embeddings/train_rnn.py
  -> GRU or LSTM checkpoint + history + val/test predictions + metrics
```

The image manifest and sensor-level sample table are deliberately separate:
multiple lens/view rows may exist per sensor sample, while targets and split
assignments must be unique per sample. A lens/view is selected explicitly at
embedding extraction. Image arrays join engineered features only inside a
modeling run using stable sample/sequence IDs. The canonical trainer establishes
the image-only benchmark; engineered features are then connected through the
strict residual-correction stage below. The shared model class also retains a
late-fusion architecture for controlled future comparison.

### Partial residual fusion

```text
safe OOF/held-out GRU or LSTM predictions
  + sequence_id -> target_sample_id mapping
  + canonical engineered feature table
  -> grouped cross-fitted tabular residual model
  -> corrected PM2.5 = image base + predicted residual
```

This preserves the image model as the primary estimator while allowing YOLO,
road, metric area, meteorology, OSM, and AlphaEarth features to explain its
remaining error. Base and corrected metrics are always reported together.

For TRAQID, `pipelines/image_embeddings/traqid_outer_crossfit.py` applies this
protocol directly to the existing T7 manifest, 4096-dimensional front/rear
ResNet50 concatenated embeddings, and engineered T7 feature table. Complete
dates are held out in each outer fold; validation dates are selected only from
the remaining training dates. Concatenated outer-test predictions form the
safe OOF image-base table used by grouped residual correction.

### Forecasting and sequence audit

```text
TRAQID base manifest
  -> sequence_leakage_audit/build_hourly_base_manifest.py
  -> build_forecasting_manifest.py (T12/H12 variants)
  -> overlap_diagnostic.py
  -> train_forecast_numerical_baselines.py / train_forecast_numerical_lstm.py
  -> combine_overlap_summaries.py + bootstrap_metric_ci.py
  -> make_forecast_* tables
  -> experiments/traqid_pretraining_v1/reports/paper_final/
```

Chronological date-wise forecasting has no direct input/target overlap; random and two-fold forecasting manifests have severe overlap.

## TRAQID pipeline-validation branch

This branch starts from TRAQID raw/paired data but is operationally separate from the paper CNN-LSTM branch:

```text
sanity/split diagnostics
  -> tabular and embedding baselines
  -> YOLO vehicle feature extraction
  -> vehicle/full-fusion/engineered models
  -> road manifest + road-dust score/ResNet50 fusion
  -> advanced YOLO and image-dominant models
  -> purged/multifold/lag-assisted validation
  -> final summary tables
```

Number 10, 18, and 22 scripts are absent. Static references do not establish that every earlier stage feeds every later stage. **Uncertainty: medium-high.**

## HVAQ lineage

```text
HVAQ raw images/labels
  -> 01_build_hvaq_manifest.py
  -> image-label manifest
  -> 02_build_hvaq_T7_sequences.py
  -> T7 sequence manifest

image-label manifest
  -> 03_extract_hvaq_vgg16_embeddings.py
  -> VGG16 embedding array + index

T7 manifest + embedding array/index
  -> 04_train_hvaq_T7_vgg16_lstm.py
  -> random / chronological-date / purged-block checkpoints, figures, metrics
```

## Model artifact lineage

- `models/detectors/yolo11m_idd15_v1_best.pt`: externally trained on IDD15 (Kaggle per experiment log); consumed by the root IDD inference script. Producer is not in this repository.
- SegFormer `model.safetensors` plus config/preprocessor JSON: externally fine-tuned IDD binary-road model; consumed by road segmentation and validator apps. Producer is not present.
- TRAQID `.pt` artifacts: produced by the corresponding `train_*` scripts; folder names encode backbone, fusion, split, and T. Exact producer-to-file edges are often dynamic, so the registry marks them uncertain.
- TRAQID `.joblib` artifacts: classical estimators/preprocessors from grouped-date YOLO-road-tabular and embedding-baseline runs. They are stored under both `models/` and `reports/`, mixing model and report concerns.
- HVAQ `.pt` artifacts: outputs of the HVAQ trainer for three split protocols.
- No serialized model artifacts were found under MUMMA-281; its reports contain predictions/metrics but the trained estimators were not persisted. **Confidence: high.**

## Missing or external lineage

- Raw data are largely ignored and provenance checksums/licenses are absent.
- Detector and SegFormer training code/datasets are external.
- FTIR/source-attribution inputs and processing are absent.
- CLI command history, environment lockfiles, and a canonical run manifest are absent.
- Static analysis cannot recover paths created through string formatting, glob expansion, environment variables, or external notebooks/job runners.
