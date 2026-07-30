# PM2.5 model experiment consolidation

Last audited: 2026-07-16

This document consolidates the PM2.5 prediction work represented by the
historical TRAQID presentation and the experiments performed afterward. It is
intended to be the technical source for a presentation. Exact run-level metrics
are indexed in `docs/pm25_model_registry.csv`. Distinct neural, classical,
fusion, feature-extraction, forecasting, and particle/source architectures are
mapped in `docs/model_architecture_catalog.md` and
`docs/model_architecture_registry.csv`.

## 1. Executive conclusion

There are two fundamentally different result categories and they must not be
mixed.

| Category | Current representative | MAE | RMSE | R2 | Interpretation |
|---|---|---:|---:|---:|---|
| Historical TRAQID random-window benchmark | 3-fold OOF ResNet50-GRU + clipped ExtraTrees | 4.212 | 8.717 | 0.966 | High overlap-contaminated benchmark; not future-date generalization. |
| MUMMA two-day smoothed random upper bound | T7 trailing-3-row target model | 3.875 | 6.033 | 0.958 | Smoothed target and 99.27% target/context leakage; not reportable as generalization. |
| MUMMA five-day random sequence | ConvNeXt-tiny GRU T7 + sensor/visual RF | 8.951 | 18.537 | 0.867 | Best current five-day random sequence row; 98.3% target/context leakage. |
| MUMMA five-day random sequence with OSM | CAMS + lens-6 ResNet50-GRU T7 + 216-feature ExtraTrees correction | 9.787 | 19.897 | 0.847 | OSM adds a small gain within the same random protocol. |
| MUMMA five-day fair pooled exploratory | Nested CAMS temporal-tabular ensemble, validation-selected weight | 25.281 | 38.666 | 0.457 | Best selection-valid pooled five-day result, but architecture was chosen after repeated inspection. |
| MUMMA five-day fair diagnostic | Same ensemble, fixed 50:50 weight | 25.275 | 38.288 | 0.468 | Slightly better, but the fixed weight is a diagnostic rather than the primary selected result. |

The main scientific result is not that a model has already generalized. The
main result is that external-background decomposition and temporal-tabular
ensembling substantially improved pooled whole-date performance, while
per-date performance remains unstable. The current pipeline must be frozen and
tested once on untouched future dates.

## 2. Dataset and processing evolution

### 2.1 TRAQID historical benchmark

- 26,558 T=7 sequences.
- Random 80/20 sequence split: 21,246 train/validation sequences and 5,312 test
  sequences.
- Front and rear images.
- Target: PM2.5.
- Overlapping sliding windows were shuffled across partitions.

### 2.2 Initial MUMMA two-day development

- Dates: 2026-02-01 and 2026-02-03.
- 2,167 aligned ten-second observations were used in the main two-day table.
- Lenses 1, 2 and 6 became the retained views after orientation inspection.
- This phase established frame extraction, preprocessing, IDD YOLO traffic,
  SegFormer road features, ResNet50 embeddings, GRU/LSTM sequence construction,
  residual correction, and explicit leakage auditing.

### 2.3 Canonical five-day MUMMA dataset

- Dates: 2026-02-01 through 2026-02-05.
- 6,873 aligned ten-second sensor samples from 55 runs.
- 20,619 preprocessed images: exactly one image per sample for lenses 1, 2 and 6.
- 136,534 detected PM-relevant road users.
- 64-dimensional AlphaEarth annual embedding for every sample.
- 25 OSM predictors within a 250 m radius for every sample.
- External CAMS/background table aligned by `sample_id`.
- Target: `sPM2`; PM/OPC channels were excluded from reportable predictors.

Canonical modeling inputs:

- `artifacts/runs/mumma_5day_complete_nonpm_models_v1/modeling_table.csv`
- `artifacts/runs/mumma_5day_complete_nonpm_models_v1/feature_groups.json`
- `artifacts/runs/mumma_5day_combined_lenses_1_2_6_v1/sensor_10s.csv`
- `artifacts/runs/mumma_5day_atmospheric_background_v1/background_hourly.csv`
- `artifacts/runs/mumma_5day_geospatial_corridor_v1/osm_250m.csv`
- `artifacts/runs/mumma_5day_geospatial_alphaearth_v1/alphaearth_2025_point.csv`

## 3. Validation protocols

### 3.1 Random row split

Rows are shuffled across train and test. Dates and adjacent ten-second samples
occur in both partitions. This is useful for a same-distribution diagnostic but
does not measure new-day performance.

### 3.2 Random sequence split after window construction

T=7 or T=9 sliding windows are constructed first and shuffled second. Adjacent
windows share most of their raw frames. In the five-day T=7 runs:

- 6,451 of 6,873 raw samples occur in more than one partition;
- 5,142 raw samples overlap train and test;
- 1,287 of 1,309 test targets were already used as training context;
- target/context leakage fraction is 98.32%.

These results are explicitly non-reportable as unseen-date generalization.

### 3.3 Whole-date outer holdout

Each of the five dates is held out in turn. Sequences are grouped by `run_id`,
so a sequence cannot cross a run or day boundary. This is the principal fair
stress test used for MUMMA.

### 3.4 Nested ensemble validation

For each held-out date, the ensemble weight is selected using outer-validation
predictions only. Temporal residual correction is cross-fitted by run within
that validation date. The untouched outer-test date is not used to choose the
weight. This is methodologically stronger than selecting a correction model on
outer-test scores, but the overall architecture remains exploratory because the
same five dates informed earlier development decisions.

## 4. Historical slide experiment: exact reconstruction

The supplied slide is the TRAQID presentation benchmark, not a MUMMA model.

### Architecture

1. Per frame, front and rear images were encoded by ResNet50 global-average
   pooling: 2,048 dimensions per view.
2. Front and rear vectors were concatenated to 4,096 dimensions.
3. T=7 frame embeddings were normalized and projected to 512 dimensions.
4. A two-layer unidirectional GRU used hidden size 256 and dropout 0.25.
5. Three shuffled OOF GRU models generated train/validation OOF predictions and
   a three-model averaged test prediction.
6. The residual target was `actual PM2.5 - OOF base prediction`.
7. The residual table began with 112 row features. Last, mean, standard
   deviation, minimum and maximum aggregation over T=7 produced 555 usable
   engineered features. Base prediction, absolute base prediction and squared
   base prediction increased this to 558 residual predictors.
8. Conservative ExtraTrees used 900 trees, `max_features=0.35`,
   `min_samples_leaf=4`, no bootstrap, and seed 42.
9. Predicted residuals were clipped to +/-12.5 ug/m3 before being added to the
   base ensemble.

### TRAQID residual feature families

- Weather and time: temperature, humidity, season, day/night, cyclic hour,
  month and day-of-week.
- YOLO: counts, confidence sums, area ratios, near-field counts/areas, traffic
  mix, class shares and spatial subregions for cars, buses, trucks,
  motorcycles, bicycles, auto-rickshaws and people.
- Road mask appearance: area, brightness, saturation, contrast, shadow, glare,
  brown/dry pixels, edges, Laplacian texture and haze proxy.
- Base prediction features: prediction, absolute prediction and squared
  prediction.

Exact feature names are stored in
`experiments/traqid_pretraining_v1/reports/pm25_oof_residual_3fold_seed42/pm25_oof_residual_feature_columns.txt`.

### TRAQID random-window progression

| Variant | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Single ResNet50-GRU | 5.991 | 10.359 | 0.951 |
| Tuned residual using base prediction | 5.440 | 9.847 | 0.956 |
| Three-model base ensemble | 5.113 | 9.434 | 0.960 |
| OOF ExtraTrees residual, unclipped | 4.234 | 8.867 | 0.964 |
| OOF ExtraTrees residual, clipped +/-12.5 | 4.212 | 8.717 | 0.966 |

### Why the slide result is not the final scientific model

The random split was applied to overlapping sliding windows. Nearly identical
image context can therefore occur on both sides of the split. The later
TRAQID five-fold date-held-out ResNet50-GRU test R2 values were 0.249, -0.079,
-0.883, -0.313 and -0.307. The large gap is evidence of temporal/date shift,
not merely an implementation difference.

Primary provenance:

- `configs/pipelines/traqid_presentation_reproduction_v1.yaml`
- `pipelines/image_embeddings/traqid_presentation_reproduction.py`
- `experiments/traqid_pretraining_v1/scripts/10_pm25_oof_residual_correction.py`
- `experiments/traqid_pretraining_v1/reports/pm25_oof_residual_3fold_seed42/FINAL_PM25_RESULT_SUMMARY.md`

Uncertainty: the clipping outputs are preserved in the report folder, but a
separate committed clipping script was not found. The main OOF training script
produces the unclipped residual candidates; clipping appears to have been a
subsequent analysis step.

## 5. MUMMA feature dictionary

The authoritative feature membership is
`artifacts/runs/mumma_5day_complete_nonpm_models_v1/feature_groups.json`.

| Family | Count | Content |
|---|---:|---|
| Sensor/met/gas/time/mobility | 24 | Temperature, RH, CO2, VOC/NOx indices, gases, cyclic time, GPS, speed, course and HDOP. |
| YOLO traffic | 58 | Vehicle counts, class counts, heavy/motor totals, area occupancy, confidences, exhaust/resuspension proxies and per-lens versions. |
| Road appearance | 45 | SegFormer road area and appearance/texture metrics aggregated across lenses plus lens 1 and lens 6 values. |
| Road depth/metric area | 16 | Depth percentiles, cleaned mask diagnostics and estimated road area. Provisional and generally harmful. |
| OSM | 25 | POI/activity context and road-network hierarchy within 250 m. |
| AlphaEarth | 64 | Annual learned embedding dimensions A00 through A63. Semantic meanings are latent, not named land-cover variables. |

### 5.1 Sensor/met/gas/time/mobility: 24

`temp`, `rh`, `k30Co2`, `sTemp`, `sRh`, `sVocI`, `sNoxI`, `co_ppb`,
`no2_ppb`, `so2_ppb`, `o3_ppb_compensated`, `ch4_ratio`, `nh3_ppm`,
`hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`, `lat`, `long`, `alt`,
`sog_clean`, `cog_sin`, `cog_cos`, `hdop`.

### 5.2 YOLO traffic: 58

- Aggregate counts: total, heavy, motor, auto-rickshaw, bicycle, bus, car,
  motorcycle, truck and unknown vehicle.
- Aggregate proxies: initial exhaust and resuspension proxies.
- Aggregate geometry/confidence: mean/max vehicle box area, average confidence
  and maximum confidence.
- The same core counts/proxies/box-area features for lenses 1, 2 and 6.
- `yolo_lens_count` records view availability.

### 5.3 Road appearance: 45

Eleven road-mask measurements are used: area ratio, brightness, saturation,
contrast, shadow, glare, brown-pixel ratio, gray/dry-pixel ratio, edge density,
Laplacian texture and haze flatness. Each is represented by all-lens mean,
all-lens maximum, lens 1 and lens 6, plus `road_lens_count`.

Lens 2 is present in YOLO but absent from per-lens road-condition columns in the
canonical table. Road feature extraction used lenses 1 and 6 because those
views were the calibrated road-facing pair.

### 5.4 OSM: 25

POI/activity counts: fuel station, restaurant, bus stop, parking,
construction, industrial, factory, warehouse, marketplace, park,
school/college, hospital, commercial and retail.

Road context: road segment count, total road length, motorway, trunk, primary,
secondary, tertiary, residential, service, living-street and unclassified road
counts.

OSM location keys and rounded latitude/longitude are join metadata, not model
predictors.

### 5.5 Composite feature groups

| Feature group | Predictors |
|---|---:|
| `visual_yolo_road` | 103 |
| `sensor_plus_visual` | 127 |
| `visual_yolo_road_area_depth` | 119 |
| `sensor_plus_visual_area_depth` | 143 |
| `visual_yolo_road_osm` | 128 |
| `sensor_plus_visual_osm` | 152 |
| `alphaearth` | 64 |
| `sensor_plus_alphaearth` | 88 |
| `visual_yolo_road_alphaearth` | 167 |
| `sensor_plus_visual_alphaearth` | 191 |
| `visual_yolo_road_osm_alphaearth` | 192 |
| `sensor_plus_visual_osm_alphaearth` | 216 |

None of these reportable groups includes PM1, PM2.5, PM4, PM10, OPC number
channels or sTPS.

## 6. MUMMA temporal model architecture

### Image embeddings

- Backbones tested: ResNet50, ConvNeXt-tiny and MobileNetV2.
- Views tested: lens 1, lens 2, lens 6, mean of lenses 1/2/6, and concatenation
  of lenses 1/2/6.
- Recurrent cells tested: GRU and LSTM.
- Sequence lengths prominently tested: T=3, T=7 and T=9; the early ladder also
  included T=1, 13 and 31.
- Sequences are grouped by `run_id`; fair sequences do not cross runs/dates.

### Current generic GRU/LSTM trainer

- One-layer unidirectional GRU or LSTM.
- Hidden dimension 128.
- Dropout 0.3 in the prediction head.
- Last recurrent output is used as the temporal representation.
- Head: layer normalization, dropout, linear 128, ReLU, dropout, output layer.
- AdamW, learning rate 1e-4, weight decay 1e-4.
- SmoothL1/Huber loss with beta 1.0.
- Batch size 32, maximum 100 epochs, patience 15.
- Optional log1p target and date-balanced sampling were tested in robust runs.

### Classical residual candidates

- Ridge: median imputation, standardization, alpha 10.
- Random Forest: 400 trees, minimum leaf 5, `max_features=0.7`.
- ExtraTrees: 400 trees, minimum leaf 5, `max_features=0.8`.
- HistGradientBoosting: learning rate 0.05, 250 iterations, 15 leaf nodes,
  L2 regularization 1.0.

## 7. Experiment chronology and what changed

### 7.1 Two-day sequence prototyping

Random overlapping sequences improved as T increased: the image base moved
from R2 0.679 at T=3 to 0.797 at T=7 and 0.828 at T=9. Leakage simultaneously
increased; at T=9, 99.27% of test targets had already appeared as training
context. YOLO/road residual correction did not reliably improve these strong
image bases. A three-row trailing-mean target produced R2 0.958, demonstrating
how smoothing plus overlap can create a very high but non-deployment score.

Whole-date two-day tests were poor: the best T=7 visual correction had RMSE
41.00 and R2 -0.542. With only two dates, correction selection was exploratory.

### 7.2 Five-day backbone/view comparison

| Fair five-fold image model | Mean MAE | Mean RMSE | Mean R2 | Worst RMSE |
|---|---:|---:|---:|---:|
| ResNet50-GRU T7, lens 6 | 38.72 | 47.29 | -1.385 | 80.70 |
| ResNet50-GRU T7, mean 1/2/6 | 38.58 | 47.43 | -1.319 | 78.30 |
| ResNet50-GRU T7, concat 1/2/6 | 39.05 | 47.54 | -1.323 | 82.65 |
| ConvNeXt-tiny-GRU T7, concat | 39.63 | 47.74 | -1.329 | 82.16 |
| ResNet50-LSTM T7, concat | 39.95 | 48.46 | -1.394 | 85.26 |
| Robust ResNet50-GRU T7, concat | 40.43 | 48.53 | -1.471 | 87.19 |
| MobileNetV2-GRU T7, concat | 43.04 | 50.89 | -1.742 | 92.77 |

Conclusion: changing the backbone or recurrent cell did not solve new-day
distribution shift. Lens 6 was retained for the background temporal branch
because it was competitive, simpler and the best single view.

The attempted five-day random ResNet50-LSTM T7 run contains only sequences,
overlap audit and a log; no trained metrics were found, so its status is
incomplete.

### 7.3 Five-day random sequence diagnostics

| Base/correction | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Lens-6 ResNet50-GRU T7 + sensor/visual RF | 10.105 | 19.608 | 0.852 |
| Concat ResNet50-GRU T7 + sensor/visual RF | 9.328 | 19.308 | 0.856 |
| Concat ResNet50-GRU T9 + sensor/visual RF | 8.292 | 25.251 | 0.776 |
| ConvNeXt-tiny-GRU T7 + sensor/visual RF | 8.951 | 18.537 | 0.867 |
| ConvNeXt-tiny-GRU T7 + sensor/AlphaEarth ET | 8.825 | 18.592 | 0.867 |
| CAMS + lens-6 ResNet50-GRU T7 + sensor/visual/AlphaEarth ET | 9.856 | 20.032 | 0.845 |
| CAMS + same + OSM ET | 9.787 | 19.897 | 0.847 |

OSM improved the matched CAMS random run by only 0.134 RMSE and 0.0021 R2.
Extreme events remain the principal error source: five extreme test samples in
the final OSM random run had RMSE 188.90 versus 17.58 for inliers.

### 7.4 Five-day tabular models

The tabular benchmark compared Ridge, Random Forest, ExtraTrees and
HistGradientBoosting across feature groups.

- Best random row diagnostic: sensor/met/gas/time/mobility ExtraTrees, MAE
  8.43, RMSE 29.83, R2 0.730, Spearman 0.960.
- Best complete fair pooled visual/geospatial model: YOLO/road/OSM/AlphaEarth
  HistGradientBoosting, MAE 34.96, RMSE 47.74, R2 0.163, Spearman 0.509.
- Without OSM, visual/road/AlphaEarth Ridge obtained RMSE 48.40 and R2 0.140.
- Nested top-30 feature selection was worse than all 192 features: RMSE 48.87
  versus 47.59 and R2 0.123 versus 0.168.
- A TRAQID-style weather/time ExtraTrees model transferred poorly: nested
  five-date raw RMSE 61.44 and R2 -0.386; affine calibration worsened it.
- The serialized TRAQID weather/time ExtraTrees model also failed under direct
  zero-shot transfer to MUMMA: MAE 40.62, RMSE 60.23, R2 -0.332 and Spearman
  0.195 over all 6,873 rows. Retraining on MUMMA improved random rows but did
  not repair unseen-date performance.
- Provisional metric road area/depth generally degraded prediction and remains
  flagged for visual review.

Pooled fair R2 can be positive even when mean per-date R2 is negative because
between-day mean differences contribute to pooled variance. Both pooled and
per-date metrics must therefore be shown.

### 7.5 Fair image plus residual correction

| Temporal base and best correction | MAE | RMSE | Pooled R2 |
|---|---:|---:|---:|
| ResNet50-GRU T7 + sensor/visual RF | 33.05 | 47.69 | 0.175 |
| ResNet50-GRU T7 + sensor/AlphaEarth RF | 32.82 | 47.33 | 0.187 |
| ConvNeXt-tiny-GRU T7 + sensor/met/gas/time/mobility RF | 33.72 | 48.78 | 0.136 |
| Robust ResNet50-GRU T7 + sensor/visual RF | 36.07 | 51.65 | 0.032 |

These correction candidates were compared after outer-test evaluation and are
therefore exploratory rather than confirmatory selections.

### 7.6 CAMS background plus local increment

The target was decomposed as:

`sPM2 = external CAMS background + learned local increment`.

CAMS was never fitted to MUMMA target values. The local branch learned
`sPM2 - CAMS` from non-PM predictors.

- Tabular visual/road/AlphaEarth ExtraTrees: MAE 27.59, RMSE 39.59, R2 0.425.
- Adding OSM: MAE 27.52, RMSE 39.56, R2 0.425; a negligible improvement.
- Temporal lens-6 ResNet50-GRU T7 with sensor/visual RF correction: MAE 26.26,
  RMSE 39.89, R2 0.423.

This was the largest methodological improvement because it explicitly modeled
the daily/regional background shift that image-only models could not learn.

### 7.7 Nested temporal-tabular ensemble

Temporal branch:

- lens-6 ResNet50-GRU, T=7;
- target is local increment over external CAMS background;
- sensor plus YOLO/road Random Forest residual correction;
- temporal correction validation predictions cross-fitted by run.

Tabular branch:

- external CAMS plus visual/road/AlphaEarth ExtraTrees local increment;
- fitted on outer training rows;
- optional OSM ablation.

The final prediction is a convex combination of the temporal and tabular
branches. The primary weight is selected on validation RMSE and then frozen.

| Nested method | OSM | MAE | RMSE | Pooled R2 | Mean date R2 |
|---|---|---:|---:|---:|---:|
| Validation-selected weight | No | 25.281 | 38.666 | 0.457 | -0.427 |
| Fixed 50:50 diagnostic | No | 25.275 | 38.288 | 0.468 | -0.378 |
| Validation-selected weight | Yes | 25.456 | 38.780 | 0.454 | -0.440 |
| Fixed 50:50 diagnostic | Yes | 25.427 | 38.400 | 0.465 | -0.398 |

The no-OSM validation-selected model is the current frozen candidate. OSM
slightly improved the standalone tabular branch but slightly worsened the full
ensemble. It remains useful context and should be retained in the data table.

Per-date R2 for the selected no-OSM ensemble was -1.305, -0.912, -0.474,
0.368 and 0.188 for February 1 through 5. This is why the pooled R2 0.457 must
not be presented alone.

## 8. Feature relevance findings

Feature relevance was assessed with global correlations, within-date centered
correlations and held-out-date permutation importance over the 192-feature
visual/OSM/AlphaEarth group.

Stable preliminary signals included:

- AlphaEarth: A52, A51, A06, A49, A10, A14, A36, A39 and A46;
- road: lens-6 Laplacian/edge/glare, maximum road area, gray/dry ratio and
  saturation;
- traffic: lens-2 vehicle box area, lens-6 total vehicles, auto-rickshaws,
  buses, heavy vehicles and exhaust/resuspension proxies;
- OSM: commercial count, service roads, bus stops, total road length and
  hospital count.

Only AlphaEarth A36 was selected in every nested feature-selection fold. Hard
top-30 selection reduced performance, which suggests correlated feature groups
carry distributed information.

Global car/vehicle correlations were negative while within-day centered
correlations became positive. For example, total vehicle count had global
Spearman approximately -0.179 and within-day approximately +0.180. This is
route/day confounding and must not be interpreted as vehicles reducing PM2.5.

## 9. Important negative results retained

- Longer overlapping windows inflated some two-day random results but did not
  improve five-day generalization.
- Multi-lens concatenation did not consistently beat lens 6 fairly.
- ConvNeXt, LSTM and robust loss/date balancing did not solve date shift.
- YOLO/road residuals sometimes worsened already strong random image bases.
- Metric road area/depth was noisy: all 6,873 estimates were flagged for review
  or low mask area, and the feature family generally degraded models.
- AlphaEarth and OSM were helpful in some branches but neither was a decisive
  standalone improvement.
- Top-30 nested selection was worse than keeping all 192 visual/geospatial
  features.
- Weather/time-only TRAQID transfer failed on unseen MUMMA dates.
- Affine calibration did not repair cross-date distribution shift.
- Random scores remained dominated by temporal autocorrelation and overlapping
  context.

## 10. Supporting analyses

- Spatial variogram: 6,873 rows, Haversine distance, within-date first 0-0.5 km
  semivariance 924.13; interpretation is confounded by time and repeated routes.
- Sudden-event audit: 35 transitions above the 99.5th percentile absolute
  change threshold of 119.74 ug/m3; 12 were immediate reversals and require
  one-second sensor/image validation.
- Maximum observed sPM2 was 1,108.64 ug/m3. Extreme samples dominate RMSE.
- The requested one-second data should be used for timing, lag, aggregation and
  spike validation rather than randomly multiplying highly autocorrelated rows.

## 11. Current recommended pipelines

### Frozen candidate for future-date confirmation

1. Extract and preprocess lens 6 images at aligned sensor timestamps.
2. Encode with ResNet50 and build T=7 sequences within `run_id`.
3. Obtain external CAMS PM2.5 background without fitting the target.
4. Train the GRU on the local increment.
5. Correct temporal validation residuals with sensor + YOLO/road Random Forest,
   cross-fitted by run.
6. Train a CAMS + YOLO/road/AlphaEarth ExtraTrees local-increment branch.
7. Select one convex temporal/tabular weight on validation predictions only.
8. Freeze everything and evaluate once on untouched future dates.

Primary implementation:

- `pipelines/image_embeddings/run_background_increment_fair.py`
- `pipelines/pm25_prediction/mumma_7day/run_residual_fusion_current_data.py`
- `pipelines/pm25_prediction/mumma_7day/run_background_local_increment.py`
- `pipelines/pm25_prediction/mumma_7day/run_nested_background_ensemble.py`

### Random historical/comparison pipeline

For comparison with earlier work, use lens-6 ResNet50-GRU T7, CAMS background
and the 216-feature sensor/YOLO/road/OSM/AlphaEarth ExtraTrees correction. Label
the output as an overlap-contaminated random diagnostic.

## 12. Safe presentation claims

- The project built a reproducible moving-platform multimodal pipeline from
  videos and sensor files through aligned frames, traffic, road, geospatial and
  atmospheric context.
- High random-window R2 values are reproducible but are heavily affected by
  overlap and temporal autocorrelation.
- Whole-date testing is much harder and exposes large daily baseline shifts.
- CAMS background/local-increment modeling materially improved pooled fair
  performance.
- The current nested ensemble reaches pooled exploratory R2 0.457, but mean
  per-date R2 remains negative and new dates are required.
- OSM adds useful interpretable context and a small gain in a matched random
  run, but did not improve the current fair ensemble.
- Extreme PM2.5 transitions remain the dominant technical and data-quality
  challenge.

## 13. Claims that are not safe

- Do not describe R2 0.966, 0.958, 0.867 or 0.847 as future-time
  generalization.
- Do not call the pooled R2 0.457 a confirmed deployment score.
- Do not treat AlphaEarth dimensions as directly interpretable land-cover
  variables.
- Do not infer causality or source percentages from feature correlation.
- Do not use PM/OPC channels as PM2.5 predictors in a reportable model.
- Do not claim chemical source apportionment without differential OPC bins and
  FTIR/chemical reference evidence.

## 14. Remaining uncertainty and next evidence required

1. Collect untouched future dates spanning new routes, weather, traffic and
   background pollution.
2. Freeze the no-OSM nested candidate before those targets are inspected.
3. Validate sudden peaks using the raw one-second sensor stream.
4. Report pooled and per-date metrics, uncertainty intervals, extreme-event
   metrics and calibration together.
5. Confirm sensor/OPC units and differential bin definitions before restarting
   source attribution.
