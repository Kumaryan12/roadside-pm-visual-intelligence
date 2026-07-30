# Model architecture catalog

Last audited: 2026-07-16

This document identifies the distinct model architectures implemented or used
in this repository. It complements `docs/pm25_experiment_consolidation.md`,
which focuses on PM2.5 results. The machine-readable companion is
`docs/model_architecture_registry.csv`.

## 1. How to read this catalog

The repository contains four different kinds of architecture:

1. **Feature extractors** convert images or coordinates into features. They do
   not predict PM2.5 by themselves.
2. **Predictive architectures** learn PM2.5, PM10, AQI, forecasts, or particle
   properties.
3. **Fusion architectures** combine a base prediction with tabular,
   geospatial, atmospheric, or residual information.
4. **Source/particle architectures** infer latent particle components or model
   particle-size targets. These are not chemically validated source
   apportionment.

Status labels used below:

- **canonical candidate**: retained for the next untouched-date evaluation;
- **active experiment**: implemented and supported by run artifacts;
- **historical**: useful for reconstruction or comparison, but superseded;
- **diagnostic only**: intentionally leakage-contaminated or otherwise not a
  generalization claim;
- **implemented, evidence incomplete**: code exists but a complete verified
  result was not found;
- **feature extractor**: upstream representation model, not a PM predictor.

Architecture variants that differ only by seed, fold, feature subset, tree
count, or output directory are grouped together. They remain individually
indexed at the run level in `docs/pm25_model_registry.csv`.

## 2. System-level architecture map

```text
videos + sensor timestamps
        |
        +--> aligned frames (lenses 1, 2, 6)
        |       |
        |       +--> ImageNet CNN embeddings
        |       |       ResNet50 / ConvNeXt-tiny / MobileNetV2 /
        |       |       EfficientNet-B0 / VGG16
        |       |
        |       +--> IDD YOLO11m --> traffic and occupancy features
        |       |
        |       +--> IDD SegFormer-B0 --> road mask/appearance features
        |               |
        |               +--> Depth Anything V2 --> provisional metric area
        |
        +--> sensor/met/gas/time/mobility features
        +--> coordinates --> OSM 250 m + AlphaEarth 64-D
        +--> timestamp/location --> external CAMS background
                                |
                                v
                 PM2.5 prediction architectures
        image GRU/LSTM | tabular trees | gated fusion | residual fusion
                                |
                                v
          current candidate: CAMS local-increment temporal/tabular ensemble

PM/OPC channels --> NMF proxy / assumed-profile NNLS / particle-regime models
```

## 3. Feature-extractor architectures

### FE-01: frozen ImageNet CNN embedding family

**Purpose.** Convert each frame into a fixed vector before temporal modeling.

**Implemented backbones.** ResNet50 (2,048 dimensions), ConvNeXt-tiny (768),
MobileNetV2 (1,280), EfficientNet-B0 (1,280), and VGG16 GAP (512). The canonical
shared extractor uses torchvision ImageNet weights. The HVAQ VGG16 extractor
uses the convolutional feature stack followed by adaptive global average
pooling.

**Fusion variants.** Single view, front/rear mean, front/rear concatenation,
lens 1/2/6 mean, and lens 1/2/6 concatenation.

**Code.**

- `src/roadside_pm/features/images/resnet.py`
- `pipelines/image_embeddings/extract_torchvision.py`
- `experiments/traqid_pretraining_v1/scripts/20_extract_paper_cnn_features.py`
- `experiments/hvaq_vgg16_lstm_v1/scripts/03_extract_hvaq_vgg16_embeddings.py`

**Evidence.** Extensively used in TRAQID, HVAQ, MUMMA-281, and five-day MUMMA
runs. Backbone changes did not solve held-out-date shift. This is an upstream
encoder family, not an end-to-end PM architecture.

### FE-02: supervised single-frame CNN regressor/encoder

**Purpose.** Fine-tune MobileNetV2, EfficientNet-B0, or ResNet50 directly on
single-image PM2.5, then optionally export its learned embedding.

**Design.** ImageNet backbone plus a scalar regression head; the training code
supports a frozen phase followed by partial unfreezing. A matching extraction
script reloads the PM-trained checkpoint.

**Code.**

- `experiments/traqid_pretraining_v1/scripts/09_train_traqid_supervised_cnn_pm25.py`
- `experiments/traqid_pretraining_v1/scripts/10_extract_supervised_cnn_embeddings.py`

**Evidence/status.** Historical TRAQID experiment with checkpoints for front
and rear MobileNetV2. Random single-image R2 was about 0.142; it did not replace
the ResNet50 sequence architecture.

### FE-03: IDD YOLO11m traffic detector

**Purpose.** Detect Indian-road vehicles and convert detections into class
counts, confidence, box area, occupancy, near-field, exhaust, and
resuspension-proxy features.

**Design.** Fine-tuned Ultralytics YOLO11m with 15 IDD classes. PM-relevant
classes include auto-rickshaw, bicycle, bus, car, motorcycle, truck, and a
vehicle fallback. It produces frame-level features and object-level boxes.

**Code/model.**

- `scripts/vehicle_detection/01_run_idd_detector_on_processed_frames.py`
- `models/detectors/yolo11m_idd15_v1_best.pt`

**Evidence/status.** Active feature extractor. It produced 136,534 relevant
objects for the five-day MUMMA table. The exact original detector-training
pipeline is not present, so training provenance remains partly uncertain.

### FE-04: IDD SegFormer-B0 binary road segmenter

**Purpose.** Estimate road masks and road-only appearance features.

**Design.** SegFormer-B0 fine-tuned for binary road segmentation. Derived
features include road area ratio, brightness, saturation, contrast, shadows,
glare, brown/dry pixels, edges, Laplacian texture, and haze flatness.

**Code/model.**

- `scripts/road_area/01_generate_road_mask_for_image.py`
- `scripts/road_area/10_batch_lens1_road_area_depth.py`
- `models/road_segmentation/best_segformer_b0_idd_binary_road/best_segformer_b0_idd_binary_road`

**Evidence/status.** Active feature extractor. Canonical per-lens road features
use lenses 1 and 6. The original segmenter training code/dataset provenance is
not fully integrated into the current pipeline.

### FE-05: metric-depth road-area estimator

**Purpose.** Convert a SegFormer road mask into approximate visible road area
and depth statistics.

**Design.** `depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf`, pinhole
back-projection, triangulated surface area, platform/mask cleaning, and
occlusion adjustments.

**Code/config.**

- `scripts/road_area/10_batch_lens1_road_area_depth.py`
- `configs/features/road_area_depth.yaml`

**Evidence/status.** Provisional extractor. All 6,873 five-day estimates were
flagged for review or low road-mask area, and this feature family generally
degraded PM2.5 models. Retain for calibration research, not as a current core
predictor.

### FE-06: AlphaEarth annual geospatial embedding

**Purpose.** Add learned satellite/geospatial context at each coordinate.

**Design.** A 64-dimensional annual embedding (`A00`-`A63`) sampled at the
moving observations. It is a latent representation; individual dimensions are
not named land-cover measurements.

**Code.**

- `src/roadside_pm/features/geospatial/alphaearth.py`
- `pipelines/shared/extract_alphaearth_features_batched.py`

**Evidence/status.** Active feature extractor. All 6,873 five-day samples have
features. It produced small gains in some branches, but was not decisive alone.

### FE-07: OSM corridor/radius context

**Purpose.** Add interpretable static context around each GPS point.

**Design.** Local 250 m counts for 14 POI/activity classes and 11 road-network
features, computed from cached Overpass tiles followed by local radius
filtering.

**Code.**

- `pipelines/shared/extract_osm_corridor_features.py`
- `src/roadside_pm/features/geospatial/osm.py`

**Evidence/status.** Active feature extractor. OSM gave only small matched
gains and slightly worsened the current nested ensemble, but it remains useful
for interpretation and future route coverage.

## 4. Direct prediction architectures

### PM-01: classical tabular benchmark suite

**Purpose.** Predict PM2.5 from engineered row-level features without an image
sequence network.

**Models.** Ridge, Random Forest, ExtraTrees, and HistGradientBoosting with
median imputation; Ridge also uses scaling. Multi-output wrappers are used for
PM2.5/PM10/AQI variants.

**Inputs.** Any registered feature group: sensor/met/gas/time, YOLO, road,
depth/area, OSM, or AlphaEarth. Reportable models explicitly exclude PM/OPC
predictors.

**Canonical code.**

- `pipelines/pm25_prediction/mumma_7day/model_current_data.py`
- `experiments/traqid_pretraining_v1/pipeline_validation/scripts/12_best_fair_traqid_model.py`
- `experiments/mumma_281_pipeline_v1/scripts/01_pm25_model_from_final_feature_table.py`

**Evidence/status.** Active benchmark family. It includes the historical
TRAQID tabular-only ExtraTrees fair result (R2 0.814 in its original test) and
the five-day MUMMA tabular experiments. Direct transfer from TRAQID to MUMMA
failed, showing that an architecture artifact is not automatically portable
across datasets.

### PM-02: CNN embedding plus PCA plus classical regressor

**Purpose.** Test whether a fixed image embedding can be modeled without an
RNN.

**Design.** Frame or flattened T-window embedding, train-only PCA, then Ridge,
Random Forest, ExtraTrees, or HistGradientBoosting.

**Code.**

- `experiments/traqid_pretraining_v1/scripts/33_train_embedding_classical_baselines.py`
- `experiments/mumma_281_pipeline_v1/scripts/07_resnet_t7_random2fold_lstm_fair_pm25.py`

**Evidence/status.** Historical/benchmark. In MUMMA-281, flattened ResNet50
PCA-8 T7 ExtraTrees was the best row in this family: MAE 8.760, RMSE 13.211,
R2 0.425 under random two-fold sequence CV.

### PM-03: single-frame embedding MLP

**Purpose.** Establish a temporal-free image baseline.

**Design.** Front/rear ResNet50 concatenation (4,096 dimensions), LayerNorm,
512-unit ReLU, 256-unit ReLU, dropout, and one PM2.5 output.

**Code/report.**

- `experiments/traqid_pretraining_v1/scripts/13_train_pm25_single_image_resnet_mlp.py`
- `experiments/traqid_pretraining_v1/reports/single_image_pm25_resnet50_mlp_random_seed42/`
- `experiments/traqid_pretraining_v1/reports/single_image_pm25_resnet50_mlp_chrono_seed42/`

**Status.** Historical diagnostic with both random and chronological runs.

### PM-04: generic image-sequence GRU/LSTM

**Purpose.** The reusable CNN-RNN base used throughout current MUMMA work.

**Design.** A precomputed embedding sequence enters a one-layer GRU or LSTM
(hidden 128). The final recurrent output passes through LayerNorm, dropout,
128-unit ReLU, dropout, and the output layer. Training uses AdamW and
SmoothL1/Huber loss. Optional late tabular input is supported.

**Backbones/views/T.** ResNet50, ConvNeXt-tiny, MobileNetV2; lens 1, 2, 6,
mean/concat 1-2-6; most prominently T=3, 7, and 9.

**Code.**

- `src/roadside_pm/modeling/image_temporal.py`
- `pipelines/image_embeddings/train_rnn.py`
- `pipelines/image_embeddings/build_sequences.py`

**Evidence/status.** Active core architecture. ConvNeXt-tiny-GRU T7 was the
best five-day random sequence base/correction family, while lens-6 ResNet50-GRU
T7 was retained for the fair CAMS branch. Backbone and cell changes did not
solve held-out-date shift.

### PM-05: HVAQ VGG16-GAP plus LSTM

**Purpose.** Dataset-specific T=7 PM2.5 temporal model for HVAQ.

**Design.** Frozen VGG16 convolutional features, 512-D GAP embeddings,
train-only standardization, one-layer LSTM hidden 64, dropout 0.30, 128-unit
ReLU head, scalar output, Huber loss, and log1p target by default.

**Code/artifacts.**

- `experiments/hvaq_vgg16_lstm_v1/scripts/04_train_hvaq_T7_vgg16_lstm.py`
- `experiments/hvaq_vgg16_lstm_v1/models/`

**Evidence/status.** Historical but fully trained. Random test R2 0.765; purged
block R2 -1.253; chronological-date R2 -39.566. The validation gap is a strong
example of temporal leakage/distribution shift.

### PM-06: TRAQID paper-style CNN-LSTM multitarget

**Purpose.** Reproduce the paper-style PM2.5, PM10, and AQI sequence model.

**Design.** VGG16 or ResNet50 GAP embeddings; front, rear, mean, or concat;
unidirectional LSTM; final hidden state; dropout-MLP; three sigmoid outputs
mapped back through train-derived target min/max.

**Code/artifacts.**

- `experiments/traqid_pretraining_v1/scripts/21_train_paper_cnn_lstm_multitarget.py`
- `experiments/traqid_pretraining_v1/models/paper_cnn_lstm/`
- `experiments/traqid_pretraining_v1/reports/paper_cnn_lstm/`

**Status.** Historical reproduction family with multiple trained checkpoints.

### PM-07: early-concatenation multimodal RNN

**Purpose.** Put image and engineered features into the same temporal stream.

**Design.** At every timestep, concatenate CNN embedding with weather/time,
YOLO, and road features; LayerNorm and project to 512; feed a two-layer GRU or
LSTM (hidden 256, optional bidirectionality); 128-unit head; output PM2.5,
PM10, and AQI.

**Code/reports.**

- `experiments/traqid_pretraining_v1/scripts/05_train_traqid_T7_multimodal_rnn_no_pm_inputs.py`
- `experiments/traqid_pretraining_v1/reports/t7_multimodal_*`

**Status.** Historical TRAQID architecture with many seeds and ensemble runs.

### PM-08: late CNN-LSTM plus tabular fusion

**Purpose.** Keep image temporal modeling separate from the tabular branch and
fuse only their final representations.

**Design.** CNN embedding LSTM final state plus a two-layer tabular MLP;
concatenate; fusion MLP; three sigmoid-scaled outputs for PM2.5, PM10, and AQI.
T=2 through T=9, random, two-fold, chronological, purged, and time-balanced
variants were run.

**Code/artifacts.**

- `experiments/traqid_pretraining_v1/scripts/23_train_paper_cnn_lstm_tabular_fusion.py`
- `experiments/traqid_pretraining_v1/models/paper_cnn_lstm_tabular_fusion*/`
- `experiments/traqid_pretraining_v1/reports/paper_cnn_lstm_tabular_fusion*/`

**Status.** Historical, extensively trained.

### PM-09: late GRU plus tabular fusion

**Purpose.** PM2.5-only late fusion with a GRU image branch.

**Design.** GRU final state (hidden 128 by default), two-layer tabular MLP
(hidden 64), concatenate, 128-unit fusion head, scalar output. Bidirectional
GRU is optional.

**Code/artifacts.**

- `experiments/traqid_pretraining_v1/scripts/16_train_T7_gru_tabular_fusion.py`
- `experiments/traqid_pretraining_v1/scripts/18_train_balanced_date_grouped_cv_gru_tabular.py`
- `experiments/traqid_pretraining_v1/models/pm25_balanced_date_grouped_T7_gru_tabular_cv/`

**Status.** Historical/fair-validation TRAQID family. The grouped-date version
has five saved fold checkpoints and preprocessors.

### PM-10: gated dual-branch temporal fusion

**Purpose.** Learn how much to trust the image sequence versus the engineered
sequence.

**Design.** Independent image and engineered projections; independent GRU or
LSTM branches; both final states projected to 256 dimensions; a learned
sigmoid gate mixes them elementwise; 128-unit head; three outputs. Defaults are
image hidden 256, engineered hidden 128, two layers, and dropout 0.25.

**Code/report.**

- `experiments/traqid_pretraining_v1/scripts/06_train_traqid_T7_gated_multimodal_fusion_no_pm_inputs.py`
- `experiments/traqid_pretraining_v1/reports/gated_resnet50_concat_gru_seed11/`

**Status.** Historical TRAQID experiment. Do not confuse this architecture
with the later multi-lens attention model.

### PM-11: gated multi-view temporal GRU

**Purpose.** Learn a lens weight at every timestep, rather than manually
concatenating or averaging lenses.

**Design.** Each lens has its own LayerNorm-linear-GELU projection. A softmax
gate assigns attention across lenses. The weighted image vector is concatenated
with a projected tabular vector at every timestep, then modeled by a GRU. Lens
attention is saved for inspection.

**Code/artifact.**

- `src/roadside_pm/modeling/multimodal_temporal.py`
- `pipelines/pm25_prediction/mumma_7day/run_multimodal_temporal.py`
- `artifacts/runs/mumma_2day_gated_multimodal_T7_30s_random_v1/`

**Evidence/status.** Diagnostic only. The recorded R2 0.958 used a causal
trailing-three-row smoothed target and 99.27% target/context overlap. It is a
useful architecture experiment, not a deployment score.

### PM-12: historical OOF GRU plus clipped ExtraTrees residual

**Purpose.** The architecture shown in the supplied TRAQID presentation.

**Design.** T=7 front/rear ResNet50 concat; 512 projection; two-layer GRU
hidden 256; three random OOF base models; 558 aggregated residual predictors;
conservative 900-tree ExtraTrees; predicted residual clipped to +/-12.5
ug/m3; final prediction is base ensemble plus clipped residual.

**Code/config.**

- `experiments/traqid_pretraining_v1/scripts/10_pm25_oof_residual_correction.py`
- `pipelines/image_embeddings/traqid_presentation_reproduction.py`
- `configs/pipelines/traqid_presentation_reproduction_v1.yaml`

**Evidence/status.** Historical random-window benchmark: MAE 4.212, RMSE
8.717, R2 0.966. It is overlap-contaminated and not future-date evidence. A
standalone committed clipping script was not found; the clipping output is
preserved in the report artifacts.

### PM-13: strict image-base plus tabular residual correction

**Purpose.** General reusable implementation of the project's base-plus-
correction idea.

**Design.** Train a CNN-RNN base; require OOF/held-out predictions for residual
training; form `actual - base`; predict the residual from non-PM tabular
features using Ridge, ExtraTrees, Random Forest, or HistGradientBoosting; add
the correction to the base. The utility rejects unsafe prediction origins and
PM/OPC/target-like feature names.

**Code.**

- `src/roadside_pm/modeling/residual_fusion.py`
- `pipelines/image_embeddings/residual_correct.py`
- `pipelines/pm25_prediction/mumma_7day/run_residual_fusion_current_data.py`

**Evidence/status.** Active architecture used in both random diagnostics and
outer-date experiments. Correction helps some branches but can over-correct an
already strong random base.

### PM-14: external-background plus learned local increment

**Purpose.** Separate regional/daily pollution background from local roadside
variation.

**Design.**

```text
final PM2.5 = external CAMS PM2.5 + learned local increment
local target = observed sPM2 - CAMS
```

The local increment can be predicted by an image GRU, a tabular tree model, or
an image model followed by residual correction. CAMS is never fitted to MUMMA
targets.

**Code/artifacts.**

- `pipelines/image_embeddings/run_background_increment_fair.py`
- `pipelines/pm25_prediction/mumma_7day/run_background_local_increment.py`
- `artifacts/runs/mumma_5day_background_local_increment_v1/`

**Evidence/status.** Active and scientifically important. It was the largest
five-day fair improvement, raising pooled R2 to roughly 0.425 in standalone
branches.

### PM-15: nested CAMS temporal-tabular ensemble

**Purpose.** Combine complementary temporal-image and tabular-geospatial local
increment predictors without choosing weights on the held-out date.

**Temporal branch.** Lens-6 ResNet50-GRU T7 local increment plus
sensor/YOLO/road Random Forest residual correction, cross-fitted by run on the
validation data.

**Tabular branch.** CAMS plus YOLO/road/AlphaEarth ExtraTrees local increment;
OSM is an ablation.

**Fusion.** Convex temporal/tabular weight selected on outer-validation RMSE,
then frozen for the outer-test date. A fixed 50:50 blend is retained only as a
diagnostic.

**Code/artifacts.**

- `pipelines/pm25_prediction/mumma_7day/run_nested_background_ensemble.py`
- `artifacts/runs/mumma_5day_nested_background_ensemble_fair_v1/`

**Evidence/status.** Current canonical candidate. No-OSM validation-selected
pooled MAE 25.281, RMSE 38.666, R2 0.457; mean per-date R2 remains negative.
It must be frozen and tested on untouched future dates before a confirmatory
claim.

### PM-16: weighted, greedy, and stacking ensembles

**Purpose.** Combine heterogeneous models after individual training.

**Variants.** Equal averaging, validation-weighted averaging, random Dirichlet
weight search, greedy ensemble construction, and a classical stacking
regressor whose meta-model is Ridge.

**Code.**

- `experiments/traqid_pretraining_v1/scripts/08_greedy_weighted_ensemble_search.py`
- `experiments/traqid_pretraining_v1/pipeline_validation/scripts/12_best_fair_traqid_model.py`
- `pipelines/pm25_prediction/mumma_7day/run_nested_background_ensemble.py`

**Status.** Mixed. Historical TRAQID searches can overfit the validation/test
setting; the nested MUMMA weight-selection implementation is the relevant safe
version.

### PM-17: numerical multi-horizon LSTM forecaster

**Purpose.** Forecast future PM2.5 and PM10 rather than estimate the current
timestamp.

**Design.** T=12 historical numerical sequence plus static categorical/time
features; multi-layer LSTM; concatenate final hidden state with static inputs;
200-unit ReLU/dropout head; output `12 horizons x 2 pollutants`.

**Code/report.**

- `sequence_leakage_audit/train_forecast_numerical_lstm.py`
- `experiments/traqid_pretraining_v1/reports/forecast_numerical_lstm_T12_H12_timestamp/`

**Status.** Historical forecasting branch, separate from the current-image
estimation task.

### PM-18: numerical forecast baselines

**Purpose.** Benchmark the LSTM forecaster against simpler approaches.

**Design.** Persistence, Ridge, Random Forest, and ExtraTrees predict all 12
future PM2.5 and PM10 steps from lagged numerical and contextual features.

**Code/report.**

- `sequence_leakage_audit/train_forecast_numerical_baselines.py`
- `experiments/traqid_pretraining_v1/reports/forecast_numerical_baselines_T12_H12_timestamp/`

**Status.** Historical forecasting benchmark. Models using recent PM history
answer a forecasting question and must not be compared directly with the
reportable image/context-only current-PM task.

## 5. Particle and source-related architectures

### SA-01: train-partition NMF plus context ExtraTrees

**Purpose.** Discover neutral latent particle profiles from cumulative PM/OPC
channels, then predict their inferred PM2.5 fractions from non-PM context.

**Design.** Normalize ten PM/OPC channels; fit 2-4 component NMF on training
rows only; transform test rows; convert component weights to PM2.5 fractions;
fit multi-output ExtraTrees on sensor/visual context; enforce normalized
fractions and mass closure.

**Code/artifacts.**

- `pipelines/particle_source_attribution/run_source_proxy.py`
- `artifacts/runs/mumma_2day_source_proxy_random_v1/`
- `artifacts/runs/mumma_source_proxy_fair_test_*/`

**Status.** Exploratory source-proxy architecture. Components 1-4 are latent
mathematical factors, not named sources. FTIR/reference evidence is required
before chemical labels can be assigned.

### SA-02: assumed-profile Monte Carlo NNLS plus context ExtraTrees

**Purpose.** Test explicit source scenarios conditional on stated, uncertain
mass-size profiles.

**Design.** Convert cumulative PM1/PM2/PM4/PM10 to four differential mass bins;
sample source profiles from Dirichlet assumptions; solve nonnegative least
squares for each row and draw; retain median and 90% intervals; predict median
fractions from non-PM context with ExtraTrees.

**Code/config.**

- `pipelines/particle_source_attribution/run_assumption_scenarios.py`
- `configs/source_attribution/assumed_profiles_v1.yaml`

**Evidence/status.** Assumption-sensitivity diagnostic only. The observed
near-100% vehicle-exhaust solution was degenerate because assumed profiles
were highly similar; it is not measured attribution.

### SA-03: directly derived particle-regime prediction

**Purpose.** Model observable particle-size behavior without assigning source
names.

**Targets.** Four PM10 mass fractions, effective diameter, provisional
effective density, and total-number proxy.

**Design.** Non-PM context plus optional ResNet embeddings reduced by
train-only PCA; multi-output ExtraTrees for compositional fractions; separate
ExtraTrees models for scalar targets, with log transforms for density/number.

**Code/artifacts.**

- `pipelines/particle_source_attribution/run_particle_regimes.py`
- `artifacts/runs/mumma_2day_particle_regimes_random_v1/`
- `artifacts/runs/mumma_particle_regimes_fair_test_*/`

**Status.** Scientifically preferred current particle analysis. It describes
particle-size regimes, not chemical sources. Effective-density claims remain
provisional until OPC number and diameter units are confirmed.

## 6. Architecture relationships and supersession

| Earlier design | Later/relevant design | Relationship |
|---|---|---|
| Single-frame CNN/MLP | T7 image GRU/LSTM | Adds temporal visual context. |
| Manual lens mean/concat | Gated multi-view GRU | Learns time-varying lens weights, but only a contaminated two-day run exists. |
| Early concatenation multimodal RNN | Late fusion and residual correction | Separates modalities and makes leakage auditing easier. |
| Random OOF clipped TRAQID residual | Strict held-out residual fusion | Preserves the base-plus-correction idea with prediction-origin checks. |
| Direct PM2.5 image model | CAMS background/local increment | Explicitly handles large daily baseline shifts. |
| One temporal branch | Nested temporal-tabular ensemble | Combines temporal and static/geospatial strengths using validation-only weighting. |
| NMF named by inspection | Neutral components / direct regimes | Avoids unsupported source labels. |
| Assumed-profile NNLS | Future measured FTIR/OPC profiles | Current profiles are illustrative and must be replaced, not tuned to fit desired answers. |

## 7. What to explore in a presentation

A clear architecture-focused presentation can use this order:

1. Show FE-03 through FE-07 as the multimodal feature-extraction system.
2. Show PM-03, PM-04, PM-07, PM-08, PM-10, and PM-11 as the progression of
   CNN-RNN fusion ideas.
3. Show PM-12 as the historical high-score random architecture, with the
   overlap warning on the same slide.
4. Show PM-13 and PM-14 as the methodological corrections: safe residuals and
   external-background decomposition.
5. End the PM2.5 section with PM-15, the frozen future-date candidate.
6. Present SA-01, SA-02, and SA-03 as three different scientific questions,
   not three interchangeable source-apportionment models.

For every predictive slide, include four labels: **target**, **input
modalities**, **split protocol**, and **claim status**. This prevents random
window scores from being visually mixed with held-out-date results.

## 8. Remaining uncertainty

- The repository contains many legacy scripts that repeat the same estimator
  family with different feature groups. They are experiment variants, not new
  neural architectures.
- Some TRAQID report folders contain results without a single canonical run
  manifest; their architecture can be reconstructed from code/config, but
  exact invocation provenance is incomplete.
- The IDD YOLO and SegFormer artifacts are present, but full original training
  provenance is absent.
- The five-day random ResNet50-LSTM T7 directory lacks a completed metrics
  artifact; the fair LSTM run is complete.
- CLIP/OpenCV road classifiers remain in legacy source, but the declared
  canonical project does not use them. They are intentionally excluded from
  the active architecture path.
- There is no validated FTIR/chemical source-attribution architecture yet.
  Existing SA models are proxy, sensitivity, or particle-regime analyses.
