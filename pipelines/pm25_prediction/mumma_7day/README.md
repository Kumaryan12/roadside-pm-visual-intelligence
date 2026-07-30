# MUMMA seven-day, 10-second pipeline preparation

This package validates the incoming collection before expensive image or
geospatial processing. Raw sensor and video files are read-only inputs.

## Before data arrival

Copy `data/manifests/mumma_7day_collections.template.csv` to
`data/manifests/mumma_7day_collections.csv`. One row represents one indivisible
collection/run. Fill the real dates, sensor paths, video paths, lens identifiers,
timezone, and known video start times. Then update column names and OPC metadata
in `configs/datasets/mumma_7day.yaml`.

## First commands after arrival

```bash
python -m pipelines.pm25_prediction.mumma_7day.audit_collection \
  --config configs/datasets/mumma_7day.yaml \
  --output-dir artifacts/runs/mumma_7day_ingestion_v1
```

Do not start YOLO, SegFormer, depth, ResNet, OSM, or AlphaEarth until this audit
passes and a 50--100 sample alignment preview has been visually checked.

## Current ELICIUS three-day delivery

Build the canonical, metadata-bounded 10-second sensor table without modifying
the SSD:

```bash
python -m pipelines.pm25_prediction.mumma_7day.prepare_elicius \
  --ssd-root /Volumes/ELICIUS \
  --output-dir artifacts/runs/mumma_3day_ingestion_v1
```

The current delivery contains 14 complete runs across 2026-02-01 and
2026-02-03. The 2026-02-02 folders currently have no AQI exports, and one
2026-02-03 run has no camera folder, so those runs are retained in the audit but
excluded from the canonical table. This status is specific to the files seen in
this delivery and must be regenerated when the SSD is updated.

Extract the 49-sample, three-lens alignment pilot:

```bash
python -m pipelines.pm25_prediction.mumma_7day.extract_frames \
  --run-id run_20260201_104356_8677 \
  --output-root artifacts/runs/mumma_3day_frame_pilot_v1 \
  --skip-existing
```

This seeks to the start of each 10-second bin. Use
`--frame-offset-seconds 5` only for a separately named bin-centre sensitivity
run. Never mix the two alignment definitions in one artifact directory.

Apply the established per-lens rotation, crop, and platform mask to the pilot:

```bash
python scripts/preprocessing/01_apply_lens_preprocessing.py \
  --input-manifest artifacts/runs/mumma_3day_frame_pilot_v1/manifests/extracted_frames.csv \
  --output-manifest artifacts/runs/mumma_3day_frame_pilot_v1/manifests/preprocessed_frames.csv \
  --output-root artifacts/runs/mumma_3day_frame_pilot_v1/frames/preprocessed \
  --lenses 1 2 6
```

The original pilot used lenses 1, 4, and 6 and produced 147/147 extracted and
147/147 preprocessed frames. A subsequent six-lens audit found that Lens 4 is
dominated by fixed instrument hardware. The canonical set is therefore now
1, 2, and 6: Lens 2 supplies an unobstructed lateral roadside/context view,
while lenses 1 and 6 supply the road-facing features. Lens 2 uses a calibrated
counter-clockwise rotation and physical crop, with no synthetic platform mask.
Only two usable collection days are currently available, so this delivery can
validate feature extraction and table assembly but cannot support the planned
seven-day generalization evaluation.

## Pilot visual features

Run fine-tuned IDD YOLO on all three canonical lenses (1, 2, and 6):

```bash
python scripts/vehicle_detection/01_run_idd_detector_on_processed_frames.py \
  --input-manifest artifacts/runs/mumma_3day_frame_pilot_v1/manifests/preprocessed_frames.csv \
  --model-path models/detectors/yolo11m_idd15_v1_best.pt \
  --output-csv artifacts/runs/mumma_3day_feature_pilot_v1/features/yolo_frame_features.csv \
  --object-output-csv artifacts/runs/mumma_3day_feature_pilot_v1/features/yolo_objects.csv \
  --conf 0.25 --imgsz 960 --checkpoint-every 25
```

Extract road-condition features only from road-facing lenses 1 and 6:

```bash
python scripts/road_segmentation/03_extract_segformer_road_condition_features.py \
  --input-manifest artifacts/runs/mumma_3day_frame_pilot_v1/manifests/preprocessed_frames.csv \
  --output-csv artifacts/runs/mumma_3day_feature_pilot_v1/features/road_features_lenses_1_6.csv \
  --lenses 1 6
```

Estimate provisional lens-1 metric road area. The focal lengths below are
historical values, not a substitute for calibration of the delivered camera:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python \
  scripts/road_area/10_batch_lens1_road_area_depth.py \
  --manifest artifacts/runs/mumma_3day_frame_pilot_v1/manifests/preprocessed_frames.csv \
  --lens-id 1 \
  --road-model-path models/road_segmentation/best_segformer_b0_idd_binary_road/best_segformer_b0_idd_binary_road \
  --depth-model-id depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf \
  --fx 1345.4897862232779 --fy 1345.4897862232779 \
  --min-depth-m 0.5 --max-depth-m 30 --area-width 1024 \
  --save-masks --save-depth-npy --save-depth-preview \
  --output-dir artifacts/runs/mumma_3day_feature_pilot_v1/road_area/lens1 \
  --output-csv artifacts/runs/mumma_3day_feature_pilot_v1/road_area/lens1_depth_estimated_road_area.csv
```

Then run lens-1 NMS, depth-gated occlusion v3, canonical normalization, and the
stable-identity table assembler. Exact commands and current paths are retained
in shell history and the table summary. The assembler keeps per-lens vehicle
features and uses a cross-view maximum instead of summing overlapping cameras;
failed road masks remain missing rather than becoming false zeroes.

```bash
python -m pipelines.pm25_prediction.mumma_7day.assemble_feature_table \
  --sensor-csv artifacts/runs/mumma_3day_ingestion_v1/sensor_10s.csv \
  --frame-manifest artifacts/runs/mumma_3day_frame_pilot_v1/manifests/preprocessed_frames.csv \
  --vehicle-csv artifacts/runs/mumma_3day_feature_pilot_v1/features/yolo_frame_features.csv \
  --road-csv artifacts/runs/mumma_3day_feature_pilot_v1/features/road_features_lenses_1_6.csv \
  --road-area-csv artifacts/runs/mumma_3day_feature_pilot_v1/road_area/lens1_road_area_canonical.csv \
  --output-csv artifacts/runs/mumma_3day_feature_pilot_v1/tables/pilot_feature_table.csv
```

Current pilot: 49 rows, 174 columns, complete three-lens YOLO coverage, 47 rows
with both road lenses, two rows with only lens 6, and 43 rows with provisionally
usable road-area estimates. Metric area remains validation-required.

## Current two-day modeling baseline

The incomplete delivery supports an engineering baseline, not model selection
or a generalization claim. Run the fixed-hyperparameter comparison with:

```bash
python -m pipelines.pm25_prediction.mumma_7day.model_current_data \
  --input-csv artifacts/runs/mumma_3day_ingestion_v1/sensor_10s.csv \
  --output-dir artifacts/runs/mumma_current_2day_baselines_v1
```

The primary protocol trains on one complete day and tests on the other, then
reverses the direction. PM/OPC mass and number channels are excluded from the
reportable feature sets. Random-row results are overlap-contaminated diagnostics,
and the OPC target-proxy feature set is explicitly non-reportable.

## Pilot geospatial features

AlphaEarth uses the latest annual layer not after the measurement year. The
current 2026 samples therefore use the 2025 layer with an explicit one-year lag:

```bash
python -m pipelines.shared.extract_alphaearth_features \
  --sensor-csv artifacts/runs/mumma_3day_feature_pilot_v1/tables/pilot_feature_table.csv \
  --output-csv artifacts/runs/mumma_3day_feature_pilot_v1/geospatial/alphaearth_2025.csv \
  --cache-json artifacts/cache/alphaearth/annual_embeddings.json \
  --lat-col lat --lon-col long --timestamp-col sample_timestamp \
  --sample-col sample_id --max-available-year 2025 \
  --buffers-m 50 100 250 --scale-m 10 --round-decimals 5

python -m pipelines.shared.merge_alphaearth_features \
  --input-table artifacts/runs/mumma_3day_feature_pilot_v1/tables/pilot_feature_table.csv \
  --alphaearth-csv artifacts/runs/mumma_3day_feature_pilot_v1/geospatial/alphaearth_2025.csv \
  --output-csv artifacts/runs/mumma_3day_feature_pilot_v1/tables/pilot_feature_table_alphaearth.csv \
  --key sample_id
```

All 49 pilot rows currently have 64 point embedding bands and mean/std values
for each band at 50, 100, and 250 m: 448 AlphaEarth features and no missing
values. The merged table has 49 rows and 629 columns. Dimensionality reduction,
if used, must be fitted inside each training fold.

The cached Overpass extractor passed a real three-location OSM smoke test:

```bash
python -m pipelines.shared.extract_osm_features \
  --sensor-csv artifacts/runs/mumma_3day_feature_pilot_v1/tables/pilot_feature_table.csv \
  --output-csv artifacts/runs/mumma_3day_feature_pilot_v1/geospatial/osm_smoke_3.csv \
  --cache-json artifacts/cache/osm/overpass_250m.json \
  --lat-col lat --lon-col long --sample-col sample_id \
  --radius-m 250 --round-decimals 5 --delay-seconds 10 --limit 3
```

Do not scale this point-by-point Overpass command to the full moving route. Both
public endpoints were too slow and one returned HTTP 429. The production OSM
stage should download the route bounding region once, or use a local `.pbf`, and
compute all 250 m joins locally. Failed API requests are never cached as genuine
zero-valued features.

For a moving route when a local `.pbf` is not yet available, use the buffered
route-tile extractor. It requests only occupied 0.02-degree tiles (42 tiles for
the current five-day route), caches each successful tile independently, and
performs the exact 250 m filtering locally:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv_torch/bin/python \
  -m pipelines.shared.extract_osm_corridor_features \
  --sensor-csv artifacts/runs/mumma_5day_combined_lenses_1_2_6_v1/sensor_10s.csv \
  --output-csv artifacts/runs/mumma_5day_geospatial_corridor_v1/osm_250m.csv \
  --cache-dir artifacts/cache/osm/mumma_5day_corridor_tiles_002deg \
  --lat-col lat --lon-col long --sample-col sample_id \
  --radius-m 250 --round-decimals 3 --tile-degrees 0.02 \
  --endpoint https://overpass.private.coffee/api/interpreter \
  --request-timeout-seconds 180 --request-retries 2 \
  --delay-seconds 3
```

An interrupted command can be rerun unchanged. Only missing tiles are fetched.
The final CSV is withheld until all occupied tiles are cached, so an incomplete
geospatial table cannot accidentally enter modeling.

After producing the canonical feature table, assign evaluation protocols:

```bash
python -m pipelines.pm25_prediction.mumma_7day.make_splits \
  --config configs/datasets/mumma_7day.yaml \
  --input-csv artifacts/runs/mumma_7day_feature_table_v1/tables/final_feature_table.csv \
  --output-csv artifacts/runs/mumma_7day_feature_table_v1/tables/final_feature_table_with_splits.csv
```

The output includes `logo_day_fold`, `split_chronological_day`,
`split_grouped_trip`, `spatial_block_id`, and `split_grouped_spatial`.
Sequences must be constructed only after these columns exist.

## Acceptance checks

- Seven distinct collection dates are present.
- Sampling gaps and duplicate timestamps are explained.
- Sensor and video coverage agree after timezone normalization.
- PM2.5 and OPC units are documented.
- Complete trips/days/spatial blocks remain in one partition.
- Random row splits are diagnostic only.
