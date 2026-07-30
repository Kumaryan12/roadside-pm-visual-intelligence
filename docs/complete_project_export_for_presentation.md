# Roadside PM Visual Intelligence: complete project export

Snapshot date: 2026-07-15  
Purpose: presentation-building source document  
Repository: `roadside-pm-visual-intelligence`

This document consolidates the project goals, datasets, code architecture,
feature pipelines, experiments, results, negative findings, scientific
limitations, and artifact locations. Numbers are copied from repository
manifests and result files. Results are labelled as fair, exploratory,
leakage-contaminated, provisional, or incomplete where appropriate.

## 1. Executive summary

The project has two connected research objectives.

1. Predict roadside PM2.5 using moving-platform imagery and non-PM context:
   timestamp-aligned multi-lens images, IDD YOLO traffic, segmented-road
   appearance, road exposure, meteorology/gases, mobility, OSM, AlphaEarth,
   atmospheric background, and CNN temporal embeddings.
2. Model particle-size regimes and eventually estimate source percentages:
   use PM mass channels and OPC-derived number/size information to construct
   physically interpretable particle targets, predict them from context, and
   constrain source-related components. Until chemical/FTIR reference evidence
   is available, these outputs are **source proxies**, not validated chemical
   source apportionment.

The main modeling architecture developed is:

```text
timestamped video frames
  -> CNN embedding sequence (ResNet50/ConvNeXt/MobileNet)
  -> GRU or LSTM image-base PM2.5 estimate

YOLO + segmented-road + met/gas + mobility + OSM + AlphaEarth
  -> tabular model of the held-out image-base residual

final PM2.5 = image-base prediction + tabular residual correction
```

The strongest current five-day fair aggregate is an exploratory nested
background ensemble with RMSE 38.67, MAE 25.28, and pooled R² 0.457. It is not
confirmatory because architecture choices were influenced by earlier outer-test
exploration. The best clean tabular-only fair pooled result without OSM has
RMSE 48.40 and R² 0.140. Random-window image scores reach R² around 0.85, and a
smoothed-target diagnostic reached R² 0.958, but those runs have severe temporal
overlap and are not evidence of unseen-day generalization.

## 2. Scientific principles

- Images are not treated as direct particulate sensors. They provide traffic,
  road, scene, and temporal context.
- PM/OPC channels other than the target are excluded from reportable PM2.5
  prediction unless an experiment is explicitly labelled an upper-bound or
  target-proxy diagnostic.
- Random rows or overlapping sliding windows are diagnostic only.
- Whole-date holdout is the primary current five-day generalization test.
- Splits are assigned before sequence construction for fair temporal models.
- A sequence must stay within one run/trip and one split; sequences do not cross
  between dates or video runs.
- Preprocessing, imputation, scaling, PCA, feature selection, and residual fitting
  must be fitted on training partitions only.
- Residual correction must use out-of-fold or otherwise held-out base predictions.
- Chemical source names require chemical/FTIR evidence. NMF components alone do
  not identify vehicle exhaust, dust, biomass burning, or secondary aerosol.

## 3. Repository structure established during restructuring

```text
configs/                 dataset, feature, validation, and pipeline configuration
data/                    schemas/manifests and small canonical inputs
src/roadside_pm/         reusable package code
pipelines/               executable pipeline entry points
scripts/                 historical and specialized extraction scripts
experiments/             dataset-specific hypotheses, configs, and compact reports
artifacts/runs/           isolated generated runs, models, predictions, and manifests
docs/                    architecture, lineage, inventory, cleanup, and this export
requirements/            environment groups
```

The migration is intentionally non-destructive. Historical scripts and reports
remain evidence until their results are reproduced by canonical pipelines.

## 4. Dataset families

### 4.1 Five-day MUMMA moving-platform dataset

Current canonical sensor table:

`artifacts/runs/mumma_5day_combined_lenses_1_2_6_v1/sensor_10s.csv`

It contains 6,873 ten-second samples, 63 source columns, 55 video runs, five
dates, and one aligned frame for each of lenses 1, 2, and 6.

| Date | Rows | First timestamp | Last timestamp | Runs |
|---|---:|---|---|---:|
| 2026-02-01 | 1,265 | 08:27:06 | 14:57:47 | 10 |
| 2026-02-02 | 1,736 | 09:12:52 | 17:15:34 | 15 |
| 2026-02-03 | 902 | 10:58:02 | 14:51:19 | 4 |
| 2026-02-04 | 1,542 | 08:58:49 | 14:06:01 | 16 |
| 2026-02-05 | 1,428 | 08:33:40 | 13:31:57 | 10 |

The timestamps above are overall daily coverage bounds, not continuous recording
claims; individual runs and gaps occur inside them. There is no evening coverage
after the listed daily end times in the current combined dataset.

### 4.2 Earlier two-day/current-data subset

The first complete modeling subset used February 1 and February 3, with 2,167
rows. It supported early image ladders, T=3/7/9 diagnostics, YOLO/road residual
fusion, source-proxy experiments, and initial AlphaEarth/OSM smoke tests.

### 4.3 MUMMA-281

This is the earlier 281-row final-feature-table pipeline. It contains tabular,
visual, road, OSM, temporal T=7, residual-fusion, and ResNet comparisons. It is
kept as a separate experiment family under
`experiments/mumma_281_pipeline_v1/`.

### 4.4 TRAQID

TRAQID supports paper-style front/rear CNN-LSTM reproduction, split diagnostics,
ResNet50-GRU outer cross-fitting, grouped-date feature ablations, and leakage
analysis. High paper-style random/two-fold results are kept as reproduction
benchmarks; chronological/grouped results are the defensible branch.

### 4.5 HVAQ

HVAQ has a VGG16 embedding + T=7 LSTM/sequence family with random,
chronological-date, and purged-block protocols. It is historical/comparative,
not the current five-day MUMMA production branch.

## 5. Five-day ingestion and alignment

### 5.1 External SSD discovery

Raw data were read from `/Volumes/ELICIUS/<date-folder>`. The ingestion runner
discovers run metadata, sensor files, lens videos, timestamps, and GPS without
modifying the SSD. Deliveries were processed separately and then combined.

Relevant entry points:

- `pipelines/pm25_prediction/mumma_7day/prepare_elicius.py`
- `pipelines/pm25_prediction/mumma_7day/extract_frames.py`
- `pipelines/pm25_prediction/mumma_7day/combine_deliveries.py`
- `pipelines/pm25_prediction/mumma_7day/audit_collection.py`
- `pipelines/pm25_prediction/mumma_7day/audit_sensor_quality.py`

### 5.2 Lens selection

The final selected views are lenses 1, 2, and 6. Lens 4 was excluded after its
orientation changed. Lens audits and calibration pilots were created before
settling on the three-view combination. Road-surface feature extraction uses
lenses 1 and 6; depth/metric road-area estimation currently uses lens 1.

### 5.3 Frame extraction and preprocessing

Each sensor sample requests one frame per selected lens at the sensor timestamp.
All 6,873 samples have three preprocessed image rows:

- Expected and available frames: 20,619.
- Preprocessed manifest shape: 20,619 rows × 106 columns.
- Identity contract: unique `sample_id`, exactly one frame for each required lens.
- Canonical manifest:
  `artifacts/runs/mumma_5day_combined_lenses_1_2_6_v1/manifests/preprocessed_frames.csv`.

Lens correction is configured by `configs/lens_preprocessing.yaml` and applied
by `scripts/preprocessing/01_apply_lens_preprocessing.py`.

## 6. Visual feature extraction

### 6.1 IDD YOLO vehicle detector

The detector is a YOLO11m model fine-tuned on IDD15 Indian road scenes. The
local artifact is `models/detectors/yolo11m_idd15_v1_best.pt`.

IDD validation summary from the experiment log:

- Overall precision: 0.677.
- Overall recall: 0.488.
- Overall mAP50: 0.497.
- Overall mAP50-95: 0.330.
- Auto-rickshaw: precision 0.779, recall 0.702, mAP50 0.743.
- Bus mAP50 0.722; motorcycle 0.698; truck 0.694; car 0.693.

Five-day extraction outputs:

- Frame-level table: 20,619 rows × 141 columns.
- Object-level table: 136,534 rows × 32 columns.
- All 20,619 frame inferences succeeded.

| PM-relevant class | Detected objects |
|---|---:|
| Motorcycle | 50,196 |
| Car | 45,193 |
| Auto-rickshaw | 20,621 |
| Truck | 9,355 |
| Bus | 7,826 |
| Bicycle | 2,382 |
| Unknown/fallback vehicle | 961 |

Counts are multi-view detection intensity, not de-duplicated real-world vehicle
counts. Features include per-class counts, heavy/motor vehicle counts,
confidence, box-area ratios, exhaust proxy, and resuspension proxy, with sums,
means/maxima, and lens-specific values.

Key files:

- `artifacts/runs/mumma_5day_visual_features_v1/yolo_frame_features.csv`
- `artifacts/runs/mumma_5day_visual_features_v1/yolo_objects.csv`
- `scripts/vehicle_detection/01_run_idd_detector_on_processed_frames.py`

### 6.2 SegFormer road features

Road pixels are obtained from the fine-tuned binary-road SegFormer. Feature
measurements are restricted to the segmented road rather than the entire image.

Features include:

- road area ratio;
- mean brightness and saturation;
- contrast and Laplacian texture;
- brown and gray/dry pixel ratios;
- shadow and glare ratios;
- edge density;
- haze/flatness proxy.

Five-day extraction status:

- Input rows for lenses 1 and 6: 13,746.
- Successful rows: 13,041.
- Failed rows: 705.
- Lens 1: 6,503 success, 370 failed.
- Lens 6: 6,538 success, 335 failed.
- Mean successful road-area ratio: 0.1163.

Output:
`artifacts/runs/mumma_5day_visual_features_v1/road_features_lenses_1_6.csv`.

### 6.3 Metric depth and road-area branch

Lens-1 segmented road masks were combined with metric depth and camera geometry
to estimate visible road surface area.

- Rows: 6,873, all with `area_status=success`.
- Mean estimated area: 89.48 m².
- Median estimated area: 82.44 m².
- Range: 0 to 254.40 m².
- Quality flags: 5,866 `needs_visual_review`; 1,007
  `bad_low_road_mask_area`.
- Mean fraction removed during cleaning: 0.551.

Because every row has a cautionary quality flag and calibration remains
provisional, these features are retained for ablation but are not part of the
primary reportable feature set.

Output:
`artifacts/runs/mumma_5day_visual_features_v1/road_area_lens1/lens1_depth_estimated_road_area.csv`.

## 7. Geospatial and atmospheric context

### 7.1 AlphaEarth

The Earth Engine AlphaEarth annual embedding was sampled at each GPS point using
the latest available year (2025). The output contains A00-A63, 64 fixed
pretrained embedding dimensions.

- Rows: 6,873.
- Successful rows: 6,873.
- Cached unique locations fetched: 6,602.
- Output:
  `artifacts/runs/mumma_5day_geospatial_alphaearth_v1/alphaearth_2025_point.csv`.

The dimensions are learned Earth-observation representation bands rather than
named variables. They encode land cover, built form, vegetation, surface, and
spatial context implicitly. They improve some fair visual/tabular comparisons,
but must not be interpreted individually as direct pollutant measurements.

### 7.2 OSM

The original point-by-point 250 m Overpass extraction was too slow and produced
timeouts/HTTP errors. A route-tile extractor was implemented:

- rounds to 1,064 unique locations;
- covers the route with 42 occupied 0.02-degree tiles;
- caches every successful tile;
- performs exact 250 m filtering locally;
- refuses to write a partial feature CSV.

At this snapshot, 37 of 42 tiles are cached; the remaining five failed with
Overpass HTTP 504/read timeouts. The final
`artifacts/runs/mumma_5day_geospatial_corridor_v1/osm_250m.csv` does not yet
exist. OSM is therefore absent from current five-day final benchmark results.

Entry point: `pipelines/shared/extract_osm_corridor_features.py`.

### 7.3 Atmospheric background

The five-day background table has 6,873 rows × 32 columns and includes:

- CAMS PM1, PM2.5, PM10, AOD550, dust AOD, and black-carbon AOD;
- MERRA-2 PM2.5, dust, sea salt, black carbon, organic carbon, and sulphate;
- ERA5 temperature, dewpoint, RH, wind components/speed, pressure,
  precipitation, and solar radiation.

Mean CAMS background PM2.5 is 29.35 µg/m³ in the extracted table. CAMS is used
as external background; it is not fitted to the mobile PM2.5 target.

Output:
`artifacts/runs/mumma_5day_atmospheric_background_v1/background_hourly.csv`.

## 8. Modeling tables and feature groups

The current no-OSM five-day modeling table is:

`artifacts/runs/mumma_5day_tabular_no_osm_benchmark_v1/modeling_table.csv`

Primary feature families:

1. Meteorology/gases: temperature, RH, CO2, VOC/NOx indices, CO, NO2, SO2,
   compensated O3, CH4, NH3.
2. Time: hour/minute sine and cosine.
3. Mobility/GPS: latitude, longitude, altitude, speed, course sine/cosine, HDOP.
4. YOLO: class counts, heavy/motor totals, box ratios, confidence, exhaust and
   resuspension proxies, lens-specific and aggregated views.
5. Road: segmented road appearance/texture/dust/haze features.
6. AlphaEarth: A00-A63.
7. Optional/provisional: depth-derived road area.
8. Pending: validated OSM 250 m context.
9. External-background branch: CAMS/MERRA-2/ERA5.

PM1, PM4, PM10, OPC number channels, and sTPS are not used as fair PM2.5
predictors. They are reserved for particle-regime/source-proxy work.

## 9. Evaluation protocols

### 9.1 Fair current protocol

Five leave-one-day-out folds. Each fold trains on four complete dates and tests
on the fifth. For temporal models, validation is selected only from the outer
training dates and sequences remain inside runs/splits.

### 9.2 Random-row diagnostic

Rows are randomly assigned 80/20. Adjacent 10-second samples from the same run
can appear in both partitions, so temporal/spatial autocorrelation inflates
performance. This is not unseen-route or unseen-day generalization.

### 9.3 Random-window diagnostic

Sliding windows are constructed and then randomly split. Raw frames are shared
between train/validation/test contexts. For five-day T=7 random runs, 98.32% of
test target frames appeared in training context. These results are explicitly
leakage-contaminated.

### 9.4 Smoothed-target upper-bound diagnostic

A causal trailing three-row PM2.5 mean and overlapping T=7 windows produced
test R² 0.958, MAE 3.87, and RMSE 6.03. However, 99.27% of test target frames
were seen as training context. This result explains how 95% R² was reached; it
must never be presented as fair generalization.

## 10. Five-day tabular-only PM2.5 results without OSM

Models tested: Ridge, Random Forest, Extra Trees, and Histogram Gradient
Boosting.

### 10.1 Random-row results

| Feature set/model | MAE | RMSE | R² | Spearman |
|---|---:|---:|---:|---:|
| Sensor/met/gas/time/mobility, Extra Trees | 8.43 | 29.83 | 0.730 | 0.960 |
| Sensor + AlphaEarth, Extra Trees | 8.91 | 30.07 | 0.726 | 0.956 |
| Sensor/met/gas/time/mobility, Random Forest | 8.96 | 30.16 | 0.724 | 0.953 |
| Sensor + YOLO + road + AlphaEarth, Extra Trees | 9.50 | 30.51 | 0.718 | 0.952 |
| Visual YOLO + road + AlphaEarth, HGB | 22.38 | 39.88 | 0.518 | 0.789 |

Sensor/time/mobility is strongest under random rows. Adding visual features does
not improve that diagnostic, which suggests strong same-day/same-route temporal
structure.

### 10.2 Fair pooled leave-one-day-out results

| Feature set/model | MAE | RMSE | Pooled R² | Spearman |
|---|---:|---:|---:|---:|
| Visual YOLO + road + AlphaEarth, Ridge | 35.42 | 48.40 | 0.140 | 0.484 |
| Visual YOLO + road + AlphaEarth, HGB | 36.12 | 48.69 | 0.130 | 0.472 |
| Visual YOLO + road + AlphaEarth, Random Forest | 37.04 | 49.11 | 0.115 | 0.458 |
| Sensor + visual + AlphaEarth, HGB | 37.00 | 49.23 | 0.110 | 0.568 |
| Sensor-only, Extra Trees | 38.83 | 51.39 | 0.031 | 0.517 |

The best pooled fair model still fails on most individual days:

| Held-out date | RMSE | R² | Bias |
|---|---:|---:|---:|
| Feb 1 | 35.23 | -2.115 | -4.47 |
| Feb 2 | 66.67 | -2.069 | -50.29 |
| Feb 3 | 59.04 | -2.127 | +46.71 |
| Feb 4 | 31.11 | 0.074 | +4.62 |
| Feb 5 | 39.08 | -0.001 | +4.81 |

Main finding: day-specific PM2.5 baselines dominate. A pooled R² hides opposing
Feb 2 and Feb 3 biases. This motivated the background + local-increment branch.

## 11. Image temporal experiments

### 11.1 Sequence design

For T=7, each example uses seven consecutive sample-aligned embeddings inside
one run. Earlier discussions described the first six as context and the seventh
as target time; the current sequence manifests retain the seven aligned rows and
target the final row. Sliding windows advance by one sample. Fair builders keep
every raw row in one split; random builders deliberately retain overlap for
diagnostic reproduction.

### 11.2 Two-day random T experiments

Lens-6 ResNet50-GRU image base:

| T | Image-base test MAE | RMSE | R² | Leakage status |
|---:|---:|---:|---:|---|
| 3 | 10.40 | 19.53 | 0.679 | contaminated |
| 7 | 8.43 | 14.73 | 0.797 | contaminated |
| 9 | 6.38 | 12.47 | 0.828 | 99.27% target-context leakage |

At T=7, random-forest YOLO/road correction changed R² from 0.797 to 0.797
approximately; at T=9 correction did not improve the image base. Longer windows
increased overlap and apparent performance.

### 11.3 Five-day random temporal results

| Backbone/view/cell | T | Test MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| ConvNeXt-tiny concat, GRU | 7 | 9.66 | 19.80 | 0.849 |
| ResNet50 concat 1/2/6, GRU | 7 | 9.68 | 20.17 | 0.843 |
| ResNet50 lens 6, GRU | 7 | 10.19 | 20.60 | 0.836 |
| ResNet50 concat 1/2/6, GRU | 9 | 8.07 | 25.51 | 0.771 |
| CAMS lens-6 ResNet50-GRU | 7 | 10.20 | 21.07 | 0.733 |

All are non-reportable random-window diagnostics with approximately 98-99%
target-context overlap.

### 11.4 Fair backbone comparison

Mean outer-fold image-only performance for concat 1/2/6 T=7:

| Backbone/cell | Mean MAE | Mean RMSE | Mean R² | Worst RMSE |
|---|---:|---:|---:|---:|
| ConvNeXt-tiny + GRU | 39.63 | 47.74 | -1.329 | 82.16 |
| ResNet50 + LSTM | 39.95 | 48.46 | -1.394 | 85.26 |
| Robust ResNet50 + GRU | 40.43 | 48.53 | -1.471 | 87.19 |
| MobileNetV2 + GRU | 43.04 | 50.89 | -1.742 | 92.77 |

The broader ResNet50 view ladder found lens 6 strongest among individual views,
but all image-only unseen-day averages were poor. ConvNeXt improved slightly
over ResNet50, not enough to solve day shift.

## 12. Residual fusion and background decomposition

### 12.1 Direct fair residual correction

Selected five-day outer-test aggregates:

| Base/correction | MAE | RMSE | R² | Status |
|---|---:|---:|---:|---|
| ResNet T7 + sensor/AlphaEarth RF correction | 32.82 | 47.33 | 0.187 | exploratory fair outer predictions |
| ResNet T7 + sensor/visual RF correction | 33.05 | 47.69 | 0.175 | exploratory fair outer predictions |
| ConvNeXt T7 + sensor/met/gas RF correction | 33.72 | 48.78 | 0.136 | exploratory |
| Robust GRU T7 + sensor/visual RF correction | 36.07 | 51.65 | 0.032 | exploratory |

Depth/road-area features generally worsened residual correction and remain
provisional.

### 12.2 Background + local increment

The target is decomposed as:

```text
mobile PM2.5 = external CAMS background PM2.5 + local increment
```

CAMS is not target-fitted. Image and tabular branches model local variation.
The background-increment ResNet-GRU temporal branch with sensor/visual RF
correction achieved MAE 26.26, RMSE 39.89, pooled R² 0.423 over 6,543 fair
outer-test sequences.

### 12.3 Nested background ensemble

The temporal background-increment branch was combined with a tabular
visual/YOLO/road/AlphaEarth Extra Trees branch. Ensemble weights were selected
using run-grouped cross-fitted validation within each outer fold, without using
outer-test targets.

| Method | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|
| Fixed 50/50 diagnostic | 25.28 | 38.29 | 0.468 | -5.28 |
| Validation-selected convex weight | 25.28 | 38.67 | 0.457 | -7.01 |
| Temporal branch | 26.26 | 39.89 | 0.423 | -6.72 |
| Tabular branch | 27.82 | 40.08 | 0.417 | -3.85 |

This is the strongest current five-day fair aggregate, but is labelled
**confirmatory=false** because branch architectures were selected after prior
outer-test exploration. The complete procedure should be preregistered and
evaluated on future unseen dates.

## 13. Spatial and event analyses

### 13.1 PM2.5 spatial variogram

- Complete rows: 6,873 across five dates.
- PM2.5 mean: 96.39 µg/m³.
- Population variance: 2,724.00.
- Range: 26.7 to 1,108.64 µg/m³.
- Distance: Haversine latitude/longitude distance.
- Bin width: 0.5 km; maximum 50 km.
- First within-date bin: mean distance 0.246 km, 356,053 pairs,
  classical semivariance 924.13, robust Cressie-Hawkins 296.32.

Interpretation remains confounded by temporal autocorrelation, repeated routes,
nonstationary day effects, and unequal spatial sampling.

### 13.2 Sudden PM2.5 changes

Consecutive ten-second changes were scanned within valid runs.

- Valid transitions: 6,818.
- Threshold: absolute change at or above empirical 99.5th percentile.
- Threshold value: 119.736 µg/m³.
- Events: 35; all have a T=7 sequence.
- Rises: 20; falls: 15.
- Immediate reversals: 12.

Outputs provide event timestamps, video-relative timestamps, and requested raw
one-second sensor windows for all five days. They are candidate events, not
validated pollution episodes; reversals may reflect sensor spikes.

Key files:

- `artifacts/runs/mumma_5day_sudden_pm25_t7_v1/all_5day_event_video_timestamps.csv`
- `artifacts/runs/mumma_5day_sudden_pm25_t7_v1/all_5day_raw_1s_request_windows.csv`
- `pipelines/analysis/find_sudden_pm25_sequences.py`
- `pipelines/analysis/build_raw_1s_request_windows.py`

## 14. Particle regimes and source-proxy work

### 14.1 Particle inputs currently available

The two-day source experiments used cumulative mass/number-like channels:
`sPM1`, `sPM2`, `sPM4`, `sPM10`, `sNPMp5`, `sNPM1`, `sNPM2`, `sNPM4`,
`sNPM10`, and `sTPS`.

The repository does not yet have confirmed raw differential OPC-bin counts with
fully verified units and bin edges. Cumulative channels can be differenced to
form size regimes, but this is not equivalent to having instrument-native raw
differential bins. Effective density remains provisional until sNPM2 and sTPS
units are confirmed.

### 14.2 NMF latent components

`run_source_proxy.py` fits non-negative matrix factorization on training-only
PM/OPC rows and predicts inferred component fractions from non-PM context.

Four-component random diagnostic:

- Rows: 2,167; train 1,734; test 433.
- Macro fraction MAE: 0.0897.
- Component R²: 0.508, -3.757, -2.382, 0.604.
- Reconstruction RMSE: 0.000649.
- Mass closure numerical MAE: approximately zero.

Two-component random diagnostic:

- Macro fraction MAE: 0.0385.
- Both reported component R² values: 0.884.
- Reconstruction RMSE: 0.0404.

However, fair date-transfer collapsed:

- Test Feb 1 two-component macro MAE 0.322; R² -41.49.
- Test Feb 3 two-component macro MAE 0.303; R² -3.96.
- Four-component date-transfer results were also unstable/strongly negative.

Conclusion: NMF components reconstruct the same particle matrix but are not
stable source identities across days.

### 14.3 Assumption-conditioned named scenarios

Illustrative profiles were created for vehicle exhaust, road dust, secondary
background, and unresolved material in
`configs/source_attribution/assumed_profiles_v1.yaml`.

The solver returned median fractions of approximately 100% vehicle exhaust and
0% for the other sources. Diagnostics showed:

- profile matrix rank 4;
- condition number 126.91;
- maximum profile cosine similarity 0.970;
- degenerate dominant solution = true.

The apparently high context-prediction scores only reproduce the assumed
solver output. They do not validate vehicle exhaust as the true source. This
experiment is an assumption-sensitivity failure and must not be reported as
measured source contribution.

### 14.4 Direct particle-size regimes

The scientifically preferred current-data target family directly derives:

- mass fraction 0-1 µm of PM10;
- mass fraction 1-2.5 µm;
- mass fraction 2.5-4 µm;
- mass fraction 4-10 µm;
- effective diameter;
- provisional effective density;
- total number proxy.

Random diagnostic results with tabular context + 32 PCA ResNet dimensions:

- four mass-fraction R² values approximately 0.744;
- effective diameter R² 0.648;
- effective density R² 0.544;
- number proxy R² 0.844.

Fair date tests were poor. On Feb 3, mass-fraction R² values were about -0.085;
on Feb 1, they were around -72 due severe distribution shift. These are
measured particle-regime targets but are not yet generalized.

### 14.5 Requirements for real source apportionment

1. Raw differential OPC bins with bin edges, flow/number units, and calibration.
2. Confirmed PM mass-channel and sTPS definitions/units.
3. FTIR or other chemical marker measurements aligned to the same samples.
4. Reference/source profiles for traffic exhaust, brake/tire wear, road dust,
   biomass burning, sea salt, and secondary aerosol as appropriate.
5. Meteorology/background and uncertainty propagation.
6. Identifiability, condition-number, mass-closure, and unresolved-component
   diagnostics.
7. Whole-date or future-campaign validation.

Until these exist, the correct terminology is particle-regime prediction and
source-proxy attribution.

## 15. MUMMA-281 results

The 281-row pipeline remains valuable as a compact historical benchmark.

| Experiment | MAE | RMSE | R² | Interpretation |
|---|---:|---:|---:|---|
| Fair met/gas Extra Trees | 6.09 | 10.63 | 0.582 | strongest simple fair ablation |
| Fair vehicle/road/OSM Extra Trees | 6.79 | 11.19 | 0.536 | visual/geospatial comparison |
| Optimized residual raw-source log1p | 5.84 | 9.91 | 0.636 | best current verified optimization row |
| Earlier residual Extra Trees | 6.33 | 10.38 | 0.601 | residual fusion |
| Fair temporal T7 best row | 6.26 | 10.78 | 0.575 | temporal residual model |
| ResNet50 T7 random-2fold ET | 8.76 | 13.21 | 0.425 | image comparison |

The `upper_bound_all_sensor_visual_osm` ridge result R² 0.998 includes unfair
target-related sensor features and is excluded from scientific recommendations.
The optimized MUMMA-281 model family does not currently persist serialized
estimators; reports/predictions support evaluation but retraining is required.

## 16. TRAQID results and lessons

The new canonical outer-crossfit ResNet50-GRU run has five date-held-out folds:

| Fold | MAE | RMSE | R² |
|---:|---:|---:|---:|
| 1 | 23.81 | 48.47 | 0.249 |
| 2 | 33.47 | 40.25 | -0.079 |
| 3 | 35.94 | 51.54 | -0.883 |
| 4 | 33.83 | 49.21 | -0.313 |
| 5 | 35.86 | 58.66 | -0.307 |

The initial smoke fold gave test R² 0.233. High historical TRAQID paper-style
scores arose from random/two-fold sliding-window overlap. The project now keeps
three interpretation tiers:

1. Paper replication/random windows: leakage-contaminated upper bound.
2. Time-balanced purged split: zero-frame-overlap within-date estimate.
3. Chronological/grouped date holdout: unseen-date stress test.

## 17. Negative findings and corrections made

- OpenCV whole-image and CLIP road models were removed from the active design;
  SegFormer road-mask features are canonical.
- Lens 4 orientation changed; final multi-view work uses lenses 1, 2, and 6.
- More overlapping frames do not guarantee fair improvement; T=9 inflated the
  two-day random score while contaminating 99.27% of targets.
- Concatenating all views did not consistently beat lens 6 under fair validation.
- ConvNeXt improved image-only fair RMSE slightly but still had negative mean R².
- LSTM was marginally better than GRU in one fair concat comparison, but neither
  solved unseen-day distribution shift.
- YOLO/road residual correction often made strong random image bases worse or
  only slightly better.
- Provisional road depth/area generally degraded models.
- AlphaEarth gave modest fair improvements in some branches, not a decisive gain.
- Tabular random scores are much better than whole-date scores because adjacent
  route observations leak temporal/spatial structure.
- Raw PM2.5 has extreme spikes; RMSE is much larger than MAE. Event review and
  one-second data are needed.
- NMF reconstruction and mass closure do not prove source identifiability.
- Assumed source profiles produced a degenerate 100% vehicle solution.
- OSM public endpoints are slow/unreliable; route-tile caching and local radius
  filtering replaced point-by-point querying.
- The strongest nested ensemble is exploratory because the same five dates have
  been repeatedly inspected. New dates are required for confirmation.

## 18. Current best/relevant pipelines

### Five-day MUMMA PM2.5

- Ingestion: `pipelines/pm25_prediction/mumma_7day/prepare_elicius.py`.
- Frame extraction: `pipelines/pm25_prediction/mumma_7day/extract_frames.py`.
- Combine deliveries: `pipelines/pm25_prediction/mumma_7day/combine_deliveries.py`.
- Tabular benchmark: `pipelines/pm25_prediction/mumma_7day/model_multimodal_current_data.py`.
- Image ladder: `pipelines/image_embeddings/run_two_day_ladder.py` and canonical
  `extract_*`, `build_sequences.py`, `train_rnn.py` stages.
- Residual fusion: `pipelines/pm25_prediction/mumma_7day/run_residual_fusion_current_data.py`.
- Background increment: `pipelines/image_embeddings/run_background_increment_fair.py`.
- Nested ensemble: `pipelines/pm25_prediction/mumma_7day/run_nested_background_ensemble.py`.

### Road/vehicle/geospatial

- YOLO: `scripts/vehicle_detection/01_run_idd_detector_on_processed_frames.py`.
- Road: `scripts/road_segmentation/03_extract_segformer_road_condition_features.py`.
- Road area: `scripts/road_area/10_batch_lens1_road_area_depth.py` and related
  NMS/occlusion/finalization scripts.
- AlphaEarth: `pipelines/shared/extract_alphaearth_features_batched.py`.
- OSM: `pipelines/shared/extract_osm_corridor_features.py`.
- Atmospheric background: `pipelines/shared/extract_atmospheric_background.py`.

### Source proxies

- Latent NMF: `pipelines/particle_source_attribution/run_source_proxy.py`.
- Assumed profiles: `pipelines/particle_source_attribution/run_assumption_scenarios.py`.
- Direct regimes: `pipelines/particle_source_attribution/run_particle_regimes.py`.

### TRAQID

- Presentation reproduction:
  `pipelines/image_embeddings/traqid_presentation_reproduction.py`.
- Fair outer cross-fit: `pipelines/image_embeddings/traqid_outer_crossfit.py`.

### MUMMA-281

- Canonical wrapper: `pipelines/pm25_prediction/mumma_281/run.py`.

## 19. Current best results by reporting category

| Category | Current leader | Result | Reporting status |
|---|---|---|---|
| Five-day fair exploratory | Nested background ensemble | RMSE 38.67, R² 0.457 | promising; not confirmatory |
| Five-day fair image-only | ConvNeXt-tiny GRU T7 | mean RMSE 47.74, mean R² -1.329 | fair but poor |
| Five-day fair tabular no OSM | Visual/road/AlphaEarth Ridge | RMSE 48.40, pooled R² 0.140 | fair pooled; unstable per day |
| Five-day random temporal | ConvNeXt GRU T7 | RMSE 19.80, R² 0.849 | leakage-contaminated |
| Five-day random tabular | Sensor-context Extra Trees | RMSE 29.83, R² 0.730 | autocorrelation-contaminated |
| Smoothed upper-bound | T7 trailing-3-row model | RMSE 6.03, R² 0.958 | 99.27% overlap; non-reportable |
| MUMMA-281 fair optimized | Residual raw-source log1p | RMSE 9.91, R² 0.636 | historical fair CV |
| Particle-regime random | Number proxy | R² 0.844 | random diagnostic |
| Chemical source percentages | None | not identifiable | awaiting chemical/FTIR evidence |

## 20. Work still required

### Immediate

1. Complete the remaining OSM route tiles and create the validated 6,873-row
   OSM table.
2. Re-run the complete tabular ablation with OSM and AlphaEarth.
3. Audit the 705 failed road rows and the 35 sudden PM events visually.
4. Obtain one-second sensor data for the daily coverage intervals or prioritized
   event windows.
5. Confirm PM/OPC/sTPS units, OPC bin edges, flow basis, calibration, and whether
   channels are cumulative or differential.

### Scientific confirmation

1. Freeze/preregister the background-increment nested ensemble.
2. Collect new dates with different routes, weather, traffic, and pollution
   baselines.
3. Evaluate once on untouched future dates.
4. Add uncertainty intervals, per-day metrics, calibration plots, and extreme
   event metrics.
5. Compare against persistence, daily mean, CAMS-only, sensor-only, and spatial
   baseline models.

### Source attribution

1. Acquire FTIR/chemical measurements and reference profiles.
2. Use raw differential OPC bins rather than inferred cumulative differences.
3. Fit constrained non-negative receptor/source-proxy models with an unresolved
   component and uncertainty.
4. Validate identities chemically and across dates before reporting percentages.

## 21. Suggested PPT storyline

1. Motivation: why moving roadside PM2.5 and source context matter.
2. Two project objectives: PM2.5 prediction and particle/source attribution.
3. Dataset: five dates, 6,873 rows, 20,619 images, lenses 1/2/6.
4. End-to-end sensor/video alignment pipeline.
5. Visual features: IDD YOLO and SegFormer road masks.
6. Geospatial/background context: AlphaEarth, OSM, CAMS/MERRA-2/ERA5.
7. Model architecture: CNN-RNN base + tabular residual correction.
8. Validation: random diagnostics versus whole-date fair testing.
9. Random results and why they are inflated.
10. Fair backbone and tabular results.
11. Background + local increment improvement.
12. Current nested ensemble result and confirmatory caveat.
13. Spatial variogram and sudden-event analysis.
14. Particle-regime/source-proxy approaches.
15. Why named source attribution is not yet identifiable.
16. Limitations, requested one-second/chemical data, and next campaign.

## 22. Artifact registry by experiment family

### Ingestion, calibration, and feature construction

- `mumma_3day_ingestion_v1`: first SSD discovery/sensor audit pilot.
- `mumma_3day_frame_pilot_v1`: initial frame extraction pilot.
- `mumma_3day_all_lens_audit_v1`: all-lens roadside-view audit.
- `mumma_lens_orientation_audit_v1`: initial orientation comparison.
- `mumma_lens_orientation_audit_day3_v1`: day-3 orientation comparison.
- `mumma_lens2_calibration_v1`: lens-2 preprocessing/calibration outputs.
- `mumma_2day_ingestion_lenses_1_2_6_v1`: Feb 1/3 ingestion.
- `mumma_2day_frames_lenses_1_2_6_v1`: Feb 1/3 frames/preprocessing.
- `mumma_feb02_feb04_ingestion_lenses_1_2_6_v1`: Feb 2/4 ingestion.
- `mumma_feb02_feb04_frames_lenses_1_2_6_v1`: Feb 2/4 frames/preprocessing.
- `mumma_feb05_ingestion_lenses_1_2_6_v1`: Feb 5 ingestion.
- `mumma_feb05_frames_lenses_1_2_6_v1`: Feb 5 frames/preprocessing.
- `mumma_5day_combined_lenses_1_2_6_v1`: canonical combined sensor/frame manifests.
- `mumma_2day_visual_features_v1`: early two-day visual features.
- `mumma_5day_visual_features_v1`: five-day YOLO, road, and road-area features.
- `20260711T095645Z_feature_table_v1`: canonical feature-table orchestration pilot.

### Geospatial and atmospheric

- `mumma_2day_geospatial_v1`: early OSM/AlphaEarth tests.
- `mumma_3day_feature_pilot_v1`: early road/geospatial smoke outputs.
- `mumma_5day_geospatial_v1`: earlier five-day point OSM work.
- `mumma_5day_geospatial_alphaearth_v1`: complete five-day AlphaEarth table.
- `mumma_5day_geospatial_corridor_v1`: resumable OSM corridor extraction; incomplete.
- `mumma_5day_atmospheric_background_v1`: CAMS/MERRA-2/ERA5 background table.

### Two-day PM2.5 model development

- `mumma_current_2day_baselines_v1`: initial sensor baselines.
- `mumma_2day_yolo_models_v1`: sensor + YOLO tabular ablation.
- `mumma_2day_yolo_road_models_v1`: sensor + YOLO + road table/models.
- `mumma_2day_model_comparison_v1`: classical comparison.
- `mumma_partial_image_gru_smoke_v1`: early GRU/LSTM smoke test.
- `mumma_2day_image_ladder_v1`: lenses/views, T=1/3/7/13/31, GRU/LSTM ladder.
- `mumma_2day_image_yolo_residual_fusion_v1`: image + YOLO correction.
- `mumma_2day_image_yolo_road_residual_fusion_v1`: image + YOLO/road correction.
- `mumma_2day_gru_yolo_road_residual_safe_v1`: fair two-date correction.
- `mumma_2day_gru_T7_yolo_road_residual_safe_v1`: fair T7 variant.
- `mumma_2day_gru_yolo_road_residual_random_v1`: random T3 diagnostic.
- `mumma_2day_gru_T7_yolo_road_residual_random_v1`: random T7 diagnostic.
- `mumma_2day_gru_T9_yolo_road_residual_random_v1`: random T9 diagnostic.
- `mumma_2day_gru_T7_yolo_tabular_residual_random_v1`: random tabular correction.
- `mumma_2day_concat126_gru_T7_yolo_tabular_residual_random_v1`: multi-lens random fusion.
- `mumma_2day_gated_multimodal_T7_30s_random_v1`: smoothed-target R² 0.958 diagnostic.

### Five-day temporal/backbone development

- `mumma_5day_image_ladder_T7_gru_v1`: fair ResNet view comparison.
- `mumma_5day_concat_1_2_6_gru_T7_robust_fair_v1`: robust ResNet-GRU fair run.
- `mumma_5day_concat_1_2_6_lstm_T7_fair_v1`: ResNet-LSTM fair run.
- `mumma_5day_convnext_tiny_gru_T7_fair_v1`: ConvNeXt fair run.
- `mumma_5day_mobilenetv2_gru_T7_fair_v1`: MobileNetV2 fair run.
- `mumma_5day_lens6_resnet_gru_T7_sensor_visual_random_v1`: lens-6 random run.
- `mumma_5day_concat_1_2_6_gru_T7_sensor_visual_random_v1`: concat T7 random run.
- `mumma_5day_concat_1_2_6_gru_T9_sensor_visual_random_v1`: concat T9 random run.
- `mumma_5day_concat_1_2_6_lstm_T7_sensor_visual_random_v1`: LSTM random run.
- `mumma_5day_convnext_concat_T7_sensor_visual_random_v1`: ConvNeXt random run.
- `mumma_5day_convnext_T7_sensor_alphaearth_random_v1`: ConvNeXt/AlphaEarth random run.
- `mumma_5day_cams_lens6_resnet_gru_T7_sensor_visual_alphaearth_random_v1`: CAMS random diagnostic.

### Five-day tabular, residual, and background models

- `mumma_5day_yolo_road_area_models_v1`: tabular with provisional area.
- `mumma_5day_yolo_road_alphaearth_models_v1`: complete no-OSM AlphaEarth table/models.
- `mumma_5day_tabular_no_osm_benchmark_v1`: clean repeated no-OSM benchmark.
- `mumma_5day_gru_T7_yolo_road_area_residual_safe_v1`: fair ResNet correction ablation.
- `mumma_5day_convnext_T7_yolo_road_area_residual_fair_v1`: fair ConvNeXt correction.
- `mumma_5day_resnet_T7_alphaearth_residual_fair_v1`: fair AlphaEarth correction.
- `mumma_5day_robust_gru_T7_sensor_visual_residual_fair_v1`: robust residual run.
- `mumma_5day_background_local_increment_v1`: tabular CAMS + local increment.
- `mumma_5day_resnet_gru_T7_background_increment_fair_v1`: image background increment.
- `mumma_5day_resnet_gru_T7_background_increment_alphaearth_residual_fair_v1`: corrected background increment.
- `mumma_5day_nested_background_ensemble_fair_v1`: current best exploratory fair ensemble.

### Analysis and source proxies

- `mumma_5day_sensor_quality_audit_v1`: sensor quality diagnostics.
- `mumma_5day_pm25_spatial_variogram_v1`: spatial semivariance analysis.
- `mumma_5day_sudden_pm25_t7_v1`: sudden events and one-second request windows.
- `mumma_2day_source_proxy_random_v1`: four-component NMF random diagnostic.
- `mumma_2day_source_proxy_2component_random_v1`: two-component NMF diagnostic.
- `mumma_source_proxy_fair_test_2026-02-01_v1`: four-component Feb 1 test.
- `mumma_source_proxy_fair_test_2026-02-03_v1`: four-component Feb 3 test.
- `mumma_source_proxy_2component_fair_test_2026-02-01_v1`: two-component Feb 1 test.
- `mumma_source_proxy_2component_fair_test_2026-02-03_v1`: two-component Feb 3 test.
- `mumma_2day_assumed_source_scenarios_random_v1`: assumed named-source sensitivity test.
- `mumma_2day_particle_regimes_random_v1`: direct particle-regime random diagnostic.
- `mumma_particle_regimes_fair_test_2026-02-01_v1`: particle regimes Feb 1 test.
- `mumma_particle_regimes_fair_test_2026-02-03_v1`: particle regimes Feb 3 test.

### TRAQID canonical runs

- `traqid_gru_outercv_smoke`: one-fold smoke test.
- `traqid_resnet50_gru_outercv_v1`: five-fold date-held-out ResNet50-GRU run.

## 23. Traceability documents

- `docs/architecture.md`: target package/pipeline architecture.
- `docs/data_lineage.md`: producers, consumers, and experiment lineage.
- `docs/codebase_inventory.md`: repository inventory and uncertainty.
- `docs/experiment_registry.csv`: per-script inventory and static callers.
- `docs/experiment_log.md`: historical IDD and extraction notes.
- `docs/repository_cleanup_plan.md`: non-destructive cleanup/migration plan.
- `configs/pipeline_registry.yaml`: canonical family registry.
- `pipelines/pm25_prediction/mumma_7day/README.md`: five-day commands.
- `pipelines/image_embeddings/README.md`: sequence/embedding contract.
- `pipelines/particle_source_attribution/README.md`: source-proxy terminology.

## 24. Final presentation claims that are safe today

- A reproducible multi-lens moving-platform feature pipeline has been built from
  external video/sensor data through aligned frames, IDD traffic, segmented-road
  features, AlphaEarth, and atmospheric context.
- Five dates provide 6,873 aligned ten-second samples and 20,619 preprocessed
  lens images.
- The IDD detector extracted 136,534 PM-relevant vehicle objects, including an
  explicit auto-rickshaw class.
- Random temporal scores are high but heavily overlap-contaminated; the project
  now quantifies that leakage rather than presenting it as generalization.
- Whole-date performance is substantially harder because daily pollution
  baselines shift.
- External-background/local-increment modeling and a nested temporal-tabular
  ensemble materially improve the current fair aggregate.
- The current best ensemble is promising but exploratory and requires untouched
  future-date confirmation.
- Particle-size regimes can be modeled on random splits, but date transfer is
  poor.
- Named chemical source percentages are not yet scientifically identifiable
  without raw differential OPC metadata and FTIR/chemical reference evidence.
