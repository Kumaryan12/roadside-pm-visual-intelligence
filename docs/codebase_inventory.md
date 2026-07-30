# Codebase inventory

Audit date: 2026-07-11. Scope: project files under the repository root, excluding `.git`, `.venv`, `.venv_torch`, bytecode, and caches. No model was deserialized and no pipeline was executed. Detailed per-file findings are in `docs/experiment_registry.csv`; lineage is in `docs/data_lineage.md`.

## Confidence convention

- **High**: stated by code/config/report or a direct filename reference.
- **Medium**: inferred from CLI defaults, directory sequence, or matching output names.
- **Low**: inferred only from naming or timestamps. All claims about “current” status are provisional because the repository has no run orchestrator, release tag, or canonical-pipeline manifest.

## Scope summary

| Asset | Count found | Notes |
|---|---:|---|
| Python files | 197 | 195 tracked plus two untracked/ignored app files present locally; includes package modules and executable scripts. |
| Notebooks | 1 | `experiments/traqid_pretraining_v1/notebooks/traqid_pm25_monthwise_plots.ipynb`; ignored/untracked. |
| Shell scripts | 1 | `sequence_leakage_audit/examples/traqid_t7_random.sh`. |
| Serialized model/preprocessor artifacts | 79 | 42 `.pt`, 36 `.joblib`, one `.safetensors`; ignored/untracked. |
| Top-level experiment families | 4 | HVAQ, MUMMA-281, legacy MUMMA, TRAQID. |
| Report roots | 8 | Four experiment report roots, TRAQID pipeline-validation reports, two `outputs` report roots, one report-artifact subtree. |
| Tracked files | 898 | 773 are under `experiments/`; generated CSV/JSON/Markdown dominate. |

The worktree was clean before this audit. `outputs/`, `models/`, most large experiment artifacts, and the notebook are ignored rather than tracked. In contrast, the repository tracks 434 experiment CSVs, 150 experiment JSONs, 58 experiment Markdown reports, 35 experiment text files, and two `.save` source backups.

## Source and pipeline families

### Shared modules (`src/`)

- `alignment`: package placeholder only.
- `evaluation/metrics.py`: common regression metrics.
- `modeling/train.py`: generic tabular training/evaluation helper.
- `pollution_proxy/traffic_proxy.py`: emission-weighted traffic proxies.
- `road_features`: CLIP and OpenCV road-condition features.
- `road_roi/cropper.py`: configured road-region cropping.
- `temporal_features/builder.py`: lag/rolling feature construction.
- `utils/io.py`: YAML and filesystem I/O; `old_config_reference.py` is explicitly legacy.
- `vehicle_detection`: detector wrapper and audit helpers.

Most executable scripts do not import these modules consistently; several reimplement local equivalents. **Confidence: high** from the import graph.

`src/vehicle_detection/detector.py` imports `src.config`, but no `src/config.py` or `src/config/` package exists. The module is therefore likely broken or orphaned unless an external path injects that module. **Confidence: high** for the missing in-repository import; **medium** for runtime impact.

### Shared acquisition and feature pipeline (`scripts/`)

- Frame extraction/preprocessing: `frame_extraction/01_*` then `preprocessing/01_*`; the recovery script rebuilds a missing manifest from existing images.
- Vehicle extraction: `vehicle_detection/01_run_idd_detector_on_processed_frames.py` using `models/detectors/yolo11m_idd15_v1_best.pt`.
- Road extraction: SegFormer mask/condition extraction, metric-depth road area, NMS, three occlusion variants, finalization and merge.
- Feature fusion: aggregate vehicle and road features, build the PM modeling table, add OSM, then effective density.
- Modeling: MUMMA and TRAQID engineered/ResNet experiments predating the self-contained experiment folders.
- `pipeline_1s`: sensor-window selection, 1-second frames, aggregation, merge, lag analysis, and report generation. Missing script numbers 02/03 indicate external or removed feature-extraction stages. **Confidence: high that the gap exists; low on why.**
- Analysis/diagnostics/dashboard: downstream inspection and visual QA rather than canonical producers.
- `scripts/archive/*_unused`: explicitly archived and likely orphaned.

### TRAQID (`experiments/traqid_pretraining_v1`)

This is the largest and most evolved family. It contains at least six overlapping branches:

1. Paired-manifest, tabular, generic embeddings, and supervised MobileNet experiments (`01`–`10`).
2. T7 sequence and GRU experiments, including random and purged variants (`11`–`18`).
3. Paper-replication ResNet50/VGG16 CNN-LSTM branch (`19`–`26`).
4. Leakage diagnostics and corrected validation ladder (`28`–`33` plus `sequence_leakage_audit/`).
5. YOLO/road/tabular grouped-date PM2.5 branch (`17`, `19`, `20_grouped_date_feature_ablation.py`).
6. `pipeline_validation/`: a separate tabular/embedding/YOLO/road/fusion validation branch through purged-date and forecasting experiments.

The paper-style random split result (average R² 0.9537) is not the best relevant evidence because its sliding windows overlap heavily. The current defensible estimation result is the **time-balanced purged T7 ResNet50 front/rear-concat LSTM + tabular fusion** (average R² 0.3675; PM2.5 R² 0.2125), with chronological date-wise evaluation retained as the unseen-date stress test (average R² -1.0944). For forecasting, use the chronological date-wise T12/H12 numerical baselines (best ExtraTrees average R² 0.2921) rather than random/two-fold results. **Confidence: high** from `reports/paper_final/`.

### MUMMA 281 (`experiments/mumma_281_pipeline_v1`)

The self-contained 281-row family supersedes `experiments/mumma_v1` and the older `scripts/modeling/03`–`06` sequence for current analysis. The stages are static ablation, fair engineered features, residual fusion, hyperparameter/feature optimization, temporal T7, ResNet50 extraction, ResNet T7 tests, and actual-PM diagnostics.

Best reported fair static model: script `04_fair_model_optimization.py`, residual top-30 source features, ExtraTrees leaf-2 base plus leaf-1 residual, log1p target, R² 0.6499/RMSE 9.7187. This is the current strongest reported MUMMA-281 candidate. The “upper_bound_all_sensor_visual_osm” R² 0.9981 is explicitly an upper bound and should not be treated as a fair result. The ResNet T7 branch peaks at R² 0.4250 and does not supersede the optimized static residual model. **Confidence: high** from result CSVs; **medium** on deployment relevance because no held-out external validation is present.

### HVAQ VGG16-LSTM (`experiments/hvaq_vgg16_lstm_v1`)

Four-stage self-contained pipeline: image/label manifest, T7 sequences, VGG16 embeddings, LSTM training under random, chronological-date, and purged-block protocols. It is a separate dataset/replication family, not the current TRAQID or MUMMA pipeline.

### Legacy MUMMA (`experiments/mumma_v1`)

Results-only tree with no local scripts. It corresponds to older root `scripts/modeling/03`–`06` runs. Several reports have names `fixed`, `random`, and `chronological`; treat them as historical because `mumma_281_pipeline_v1` is newer and more complete. One metrics Markdown table appears malformed (array-like RMSE cells repeated across rows), so it should not be used without regenerating. **Confidence: high.**

### FTIR/source attribution

No Python script, notebook, configuration, input file, model, or report explicitly mentioning FTIR was found. “Source” in MUMMA-281 refers to engineered visual/road/vehicle source-proxy feature groups, not spectroscopy. Therefore, there is **no identifiable FTIR/source-attribution pipeline in this repository**. It may live outside the repository or be represented only by unnamed columns in `final_feature_table.csv`; that latter possibility is unverified. **Confidence: high for no explicit FTIR assets; low for external provenance.**

## Current pipeline pointers

| Need | Most relevant path | Status |
|---|---|---|
| TRAQID defensible estimation | `19_build_paper_style_T_sequence_manifest.py` → `20_extract_paper_cnn_features.py` → `22_fuse_paper_front_rear_features.py` → `23_train_paper_cnn_lstm_tabular_fusion.py`; use time-balanced split from `30_build_time_balanced_purged_split.py` and tables from `32_make_final_validation_ladder_table.py` | Current evidence; random/two-fold scores are leakage-contaminated. |
| TRAQID forecasting | `sequence_leakage_audit/build_forecasting_manifest.py` → numerical baseline/LSTM trainers → forecast tables | Use chronological date-wise result. |
| MUMMA 281 PM2.5 | `01` → `02` → `03` → `04_fair_model_optimization.py`; `05`–`07` are temporal/ResNet comparisons; `08` audits target quality | Current self-contained family. |
| Road feature extraction | preprocessing → SegFormer `03_extract_segformer_road_condition_features.py`; road-area `10_batch` → `10_filter` → choose/compare `11`, `11b`, `11c` → `12` → `14` → `15` | v3 depth-gated occlusion is newest by naming; validation required before declaring it best. |
| YOLO/IDD vehicles | `scripts/vehicle_detection/01_run_idd_detector_on_processed_frames.py` with `models/detectors/yolo11m_idd15_v1_best.pt`, then feature-fusion aggregation | Current Indian-vehicle detector. TRAQID has separate COCO/advanced-YOLO feature branches. |
| ResNet temporal | TRAQID paper T7 branch above; MUMMA scripts `06_extract_resnet50_embeddings.py` → `07_resnet_t7_random2fold_lstm_fair_pm25.py`; HVAQ uses VGG16, not ResNet | TRAQID is mature but split-sensitive; MUMMA ResNet result is comparative, not best overall. |
| FTIR/source attribution | None found | Blocked by absent explicit assets/provenance. |

## Duplicates, superseded versions, and orphans

- Exact/near role duplicates: two TRAQID `01` sanity/audit scripts; two `05` training scripts; three `07` scripts; several `08`/`09`/`10` branches; two sequence builders (`11_*`); two `12` trainers; duplicated numeric prefixes throughout. These are branches, not necessarily byte-identical duplicates.
- `pipeline_validation/01_sanity_report.py.save` and `scripts/31_sequence_overlap_diagnostic.py.save` are committed source backups and are superseded by their `.py` counterparts.
- `scripts/archive/road_features_unused` and `scripts/archive/road_segmentation_unused` are explicit orphans.
- `src/utils/old_config_reference.py` is superseded by YAML configuration.
- `experiments/mumma_v1` is superseded by `mumma_281_pipeline_v1` for current 281-sample work.
- Root MUMMA scripts overlap with the self-contained MUMMA-281 family; preserve as provenance until outputs are reconciled.
- Root TRAQID modeling scripts overlap with the larger experiment directory; the latter has the corrected leakage analysis and final tables.
- Empty directories and missing numeric stages are likely scaffolding or removed stages, but cannot be called unused with certainty.
- Almost every executable has no static caller. That does **not** prove it was unused: CLI invocation, notebooks, IDE runs, and external job systems leave no call edge.

## Hard-coded paths

Two explicit user-machine paths were found:

- `scripts/feature_fusion/04_add_osm_features_to_modeling_table.py`: old `/Users/.../Projects/pm_density_image_pipeline/` root.
- TRAQID monthwise notebook: absolute path to its local `TRAQID.csv`.

Many scripts also hard-code repository-relative default paths. These are portable only when run from the repository root; all are listed in the registry input/output columns. Dynamic paths assembled from CLI arguments may not be captured by static extraction.

## Generated outputs beside source

The experiment roots mix source with manifests, embeddings, plots, reports, predictions, and checkpoints. TRAQID locally contains about 53,956 PNGs, 550 CSVs, 138 JSONs, 38 `.pt` files, 36 `.joblib` files, and multiple NumPy archives. HVAQ includes raw JPGs, generated CSV/JSON/PNG, embeddings, and checkpoints. MUMMA-281 mixes scripts with generated data/embeddings/reports. Most binary assets are ignored, but hundreds of report tables are tracked.

This layout preserves provenance but obscures canonical outputs and makes “source versus run product” ambiguous. Cleanup recommendations deliberately avoid deletion or movement and are recorded in `docs/repository_cleanup_plan.md`.
