# Target architecture and compatibility policy

The repository is migrating incrementally from script-oriented research code
to a package-and-pipeline structure. Existing scripts, experiment directories,
reports, and artifacts remain authoritative historical evidence until a new
pipeline reproduces them.

## Layers

1. `data/schemas`: canonical identities and table contracts.
2. `src/roadside_pm`: reusable feature, target, model, validation, and reporting logic.
3. `pipelines`: small ordered entry points that call package code.
4. `experiments`: configs, hypotheses, and compact conclusions.
5. `artifacts/runs`: run-specific models, predictions, figures, and provenance.
6. `reports`: reviewed cross-run conclusions.

## Scientific tracks

- PM2.5 prediction: images + IDD vehicles + road + OSM + AlphaEarth + tabular context under leakage-free validation.
- Particle density and source proxies: measured PM/OPC -> physical targets -> constrained component estimates and percentages.

Image embeddings form a separate modeling branch. ResNet50 embeddings and their
GRU/LSTM sequences consume the canonical manifest and split assignments, but
are stored as arrays plus ID indexes. They join the engineered table only inside
a model run; this keeps the base table interpretable and ensures PCA/scaling are
fit within training folds.

The preferred partial fusion treats the image GRU/LSTM as the base PM2.5
estimator and the engineered table as a residual corrector. Residual targets are
computed only from out-of-fold or otherwise held-out image predictions. The
tabular residual model is itself grouped-cross-fitted, and final predictions are
the non-negative sum of base and predicted residual. Direct in-sample base
predictions are rejected by the pipeline contract.

## Canonical road features

Road features use the fine-tuned binary-road SegFormer mask. Brown/gray color,
brightness, saturation, edge, texture, and dust-proxy measurements are computed
inside segmented road pixels. Haze and visibility may additionally use broader
scene context, but must remain separate from road-surface measurements.

Physical road exposure uses a second geometry branch: the SegFormer mask is
combined with metric Depth Anything output and lens-specific camera intrinsics
to reconstruct a triangulated road surface area. NMS-filtered IDD vehicle
detections then support depth-gated vehicle-footprint occlusion, producing raw
segmented area, vehicle-occluded area, and conservative effective visible road
area as separate features. These are model-derived estimates and must carry
calibration, depth-validity, segmentation, and occlusion-quality metadata.

CLIP zero-shot labels and whole-image OpenCV dust scores are retained only as
historical code. They are excluded from the active PM2.5 and source-proxy
pipelines because they are less spatially specific and can confuse buildings,
vehicles, sky, and vegetation with the road surface.

## Non-destructive migration rules

- Do not move or delete existing files during the foundation phase.
- Preserve existing commands until compatibility wrappers pass equivalence tests.
- Fit imputers, scalers, feature selection, PCA, and models inside training folds.
- Register random-window TRAQID results as paper replications, not generalized performance.
- Require checksums, producer/consumer mapping, and a reproduced result before marking an implementation superseded.
- Use “source-proxy attribution” unless chemical/receptor-model labels support stronger wording.

## Migration progress

- Road OpenCV and CLIP implementations remain preserved under
  `src/roadside_pm/features/road` with historical compatibility shims, but are
  explicitly non-canonical. New work uses SegFormer mask-restricted features.
- MUMMA-281 has a canonical stage registry and runner under
  `pipelines/pm25_prediction/mumma_281`. The runner delegates to historical
  scripts so existing outputs and defaults remain unchanged.
- The next extraction target is shared MUMMA feature classification and model
  evaluation logic, after exact historical metric reproduction is captured.
- The complete sensor/video-to-feature-table stage graph is registered under
  `pipelines/shared/feature_table`. It isolates every run under
  `artifacts/runs/<run_id>` and records commands, config checksum, input
  checksums, stage status, and the final-table path.
- Historical road-area v3 output now has a canonical schema adapter, and strict
  one-to-one mergers are used for metric road-area and AlphaEarth tables.
