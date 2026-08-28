#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="v1.0.3-paper"
SOURCE="$ROOT/docs/environmental_modelling_software_submission_v1"
STAGING="$ROOT/output/release/$VERSION"
ARCHIVE="$ROOT/output/release/roadside-pm-visual-intelligence-$VERSION.zip"

rm -rf "$STAGING"
mkdir -p "$STAGING/manuscript/figures_submission_final"
mkdir -p "$STAGING/outputs/taiwan_chronological_reference"
mkdir -p "$STAGING/outputs/taiwan_network_ablation"
mkdir -p "$STAGING/outputs/taiwan_reviewer_audit"
mkdir -p "$STAGING/outputs/traqid_chronological_calibration"
mkdir -p "$STAGING/outputs/traqid_reviewer_audit"

cp "$SOURCE/main.tex" "$SOURCE/references.bib" "$SOURCE/main.bbl" \
  "$SOURCE/main.pdf" "$SOURCE/graphical_abstract.tex" \
  "$SOURCE/graphical_abstract.pdf" "$SOURCE/graphical_abstract.png" \
  "$SOURCE/highlights.txt" "$SOURCE/cover_letter.md" \
  "$SOURCE/figure_captions.txt" "$SOURCE/declaration_of_competing_interest.docx" \
  "$SOURCE/dataset_scene_figure_sources.md" "$SOURCE/FIGURE_REBUILD_AUDIT.md" \
  "$SOURCE/README_submission.md" "$SOURCE/submission_checklist.md" \
  "$SOURCE/RELEASE_MANIFEST_v1.0.3.md" "$STAGING/manuscript/"
cp "$SOURCE"/figures_submission_final/* "$STAGING/manuscript/figures_submission_final/"

cp "$ROOT/artifacts/runs/taiwan_chronological_reference_v1/split_audit.json" \
  "$ROOT/artifacts/runs/taiwan_chronological_reference_v1/models/conservative_conditioned_resnet50_gru_T7/metrics.json" \
  "$ROOT/artifacts/runs/taiwan_chronological_reference_v1/models/conservative_conditioned_resnet50_gru_T7/history.csv" \
  "$ROOT/artifacts/runs/taiwan_chronological_reference_v1/models/conservative_conditioned_resnet50_gru_T7/predictions_val.csv" \
  "$ROOT/artifacts/runs/taiwan_chronological_reference_v1/models/conservative_conditioned_resnet50_gru_T7/predictions_test.csv" \
  "$STAGING/outputs/taiwan_chronological_reference/"

cp "$ROOT/artifacts/runs/taiwan_chronological_network_ablation_v1/comparison.csv" \
  "$ROOT/artifacts/runs/taiwan_chronological_network_ablation_v1/metrics.json" \
  "$ROOT/artifacts/runs/taiwan_chronological_network_ablation_v1/predictions_test.csv" \
  "$STAGING/outputs/taiwan_network_ablation/"

cp "$ROOT/artifacts/runs/taiwan_reviewer_requested_audit_submission_v1"/* \
  "$STAGING/outputs/taiwan_reviewer_audit/"

cp "$ROOT/artifacts/runs/traqid_T7_reference_context_chrono10_calibration_v1/metrics_aggregate.csv" \
  "$ROOT/artifacts/runs/traqid_T7_reference_context_chrono10_calibration_v1/metrics_by_date.csv" \
  "$ROOT/artifacts/runs/traqid_T7_reference_context_chrono10_calibration_v1/calibration_parameters.csv" \
  "$ROOT/artifacts/runs/traqid_T7_reference_context_chrono10_calibration_v1/split_audit.csv" \
  "$ROOT/artifacts/runs/traqid_T7_reference_context_chrono10_calibration_v1/run.json" \
  "$ROOT/artifacts/runs/traqid_T7_reference_context_chrono10_calibration_v1/predictions.csv" \
  "$STAGING/outputs/traqid_chronological_calibration/"

cp "$ROOT/artifacts/runs/traqid_reviewer_requested_audit_v1"/* \
  "$STAGING/outputs/traqid_reviewer_audit/"

cp "$ROOT/CITATION.cff" "$ROOT/LICENSE" "$ROOT/README.md" "$STAGING/"

find "$STAGING" -type f \
  \( -name '.DS_Store' -o -name '*.aux' -o -name '*.log' -o -name '*.blg' -o -name '*.out' \) \
  -delete

(
  cd "$STAGING"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS
)

rm -f "$ARCHIVE"
(
  cd "$(dirname "$STAGING")"
  zip -q -r -X "$ARCHIVE" "$VERSION"
)

echo "$STAGING"
echo "$ARCHIVE"
