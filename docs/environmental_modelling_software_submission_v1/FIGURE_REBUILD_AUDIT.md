# Deterministic figure rebuild and comparison audit

Audit date: 2026-08-28

## Outcome

Eleven of the twelve numbered manuscript figures were rebuilt from preserved
observations, split manifests, prediction tables and bootstrap summaries using
`scripts/manuscript/rebuild_submission_figures.py`. The rebuild uses Python,
NumPy, pandas, scikit-learn and Matplotlib only. It does not call a generative
image model.

The graphical abstract was independently recompiled from
`graphical_abstract.tex` with Tectonic. The resulting 1328 x 531 PNG is
pixel-for-pixel identical to the current submission PNG. Both files have
SHA-256:

`9266ee27989d27ee10e807c01eed4a71af2f721782d407a437dc0e630a9121ae`

This demonstrates that the graphical abstract is reproducible from explicit
TikZ geometry and mathematical text.

## Figure lineage

| Manuscript figure | Deterministic construction | Principal source |
|---|---|---|
| 1. TRAQID coverage and distribution | Counts per complete date; per-date box plots of measured PM2.5 | `traqid_paired_manifest_with_splits.csv` |
| 2. Dataset scene examples | Observational image montage; grayscale conversion, crop and uniform contrast adjustment | Exact filenames and licences are recorded in `dataset_scene_figure_sources.md`; raw datasets are not redistributed |
| 3. Framework architecture | Matplotlib boxes/arrows; equation `y_hat=B+Delta_hat_local` | Declared architecture and equations |
| 4. Taiwan protocol audit | Unique dates and rows by retained split | `sequences_reference.csv` and split audit |
| 5. Measured versus predicted | Hexbin density and identity line; MAE/RMSE/R2 recomputed | Frozen Taiwan test predictions |
| 6. Baselines and outages | RMSE bars and one-support-site removal audit | Submission-matched baseline audit |
| 7. Date-wise robustness | Per-date R2 distributions and paired RMSE reductions | `datewise_metrics.csv` |
| 8. Pollution-range performance | MAE/RMSE recomputed in fixed measured-PM bins | Frozen Taiwan test predictions |
| 9. Date-cluster bootstrap gain | Medians and percentile intervals from 5,000 date resamples | Submission-matched bootstrap summary |
| 10a--b. TRAQID calibration diagnostics | Early/future residual correlation and paired date RMSE | Calibration parameters and predictions |
| 11. TRAQID grouped sensitivity | Fixed concentration strata and paired date effects | Preserved reviewer-audit tables |
| 12. TRAQID examples | Dates selected mechanically by maximum, lower-median and minimum RMSE gain | Frozen zero-shot and calibrated predictions |

Every source and generated file checksum is recorded in
`figures_rebuilt_code_v1/rebuild_report.json`.

## Numerically recovered headline values

- Taiwan network background: n=7,180, MAE=4.703900, RMSE=6.613178,
  R2=0.690112.
- Taiwan final conditioned model: n=7,180, MAE=4.579935,
  RMSE=6.442474, R2=0.705904.
- Complete-date RMSE improved on 41 of 54 retained Taiwan test dates.
- Five-thousand-date-bootstrap median changes versus the network median:
  MAE reduction 0.124349 (95% interval 0.043420--0.204258), RMSE reduction
  0.172373 (0.082008--0.251269), and R2 gain 0.015991
  (0.007083--0.025610).
- TRAQID first-10%-to-future-90% residual correlation: r=0.896131 across
  16 dates.

## Comparison with the legacy/AI-assisted candidate artwork

`scripts/manuscript/compare_submission_figures.py` renders the legacy and
deterministic versions at a common size and reports pixel correlation, pixel
MAE and edge intersection-over-union. These are appearance diagnostics only;
they are not tests of numerical correctness. Changes in fonts, margins and
panel geometry can produce low similarity even when values agree exactly.

The side-by-side comparison is:

`figure_comparison_v1/legacy_vs_deterministic_contact_sheet.png`

The deterministic figures reproduce the same mathematical content while
providing traceable inputs, vector PDFs and stable regeneration commands. The
legacy versions are sometimes more compact, but they do not provide stronger
numerical evidence.

## Issues revealed by the rebuild

1. The manuscript correctly says that the Taiwan date bootstrap uses 5,000
   resamples, but the legacy Figure 8 and `figure_captions.txt` still say
   20,000. The submission-matched rerun now uses exactly 5,000.
2. With 5,000 resamples, the RMSE-reduction interval is 0.082--0.251, not the
   legacy 20,000-resample value 0.082--0.254. The manuscript and caption should
   be synchronized before submission.
3. Resolved for `v1.0.3-paper`: the separate artwork package and caption file
   now use Figures 1--12, with the calibration panels named 10a and 10b.
4. Resolved for `v1.0.3-paper`: montage transformations, exact source filenames,
   attribution and the MIT/CC BY 4.0 licences are stated explicitly. The raw
   image datasets remain external and are not redistributed.

## Commands

```bash
PYTHONPATH=src .venv_torch/bin/python \
  -m pipelines.china_surveillance_aqi.analyze_reviewer_baselines \
  --raw-manifest artifacts/runs/china_surveillance_aqi_paper_reproduction_v1/processed/china_surveillance_manifest.csv \
  --sequence-manifest artifacts/runs/taiwan_chronological_reference_v1/sequences_reference.csv \
  --frozen-test-predictions artifacts/runs/taiwan_chronological_reference_v1/models/conservative_conditioned_resnet50_gru_T7/predictions_test.csv \
  --exploratory-test-predictions artifacts/runs/taiwan_chronological_network_ablation_v1/predictions_test.csv \
  --output-dir artifacts/runs/taiwan_reviewer_requested_audit_submission_v1 \
  --bootstrap-replicates 5000 \
  --seed 42

.venv_torch/bin/python scripts/manuscript/rebuild_submission_figures.py
.venv_torch/bin/python scripts/manuscript/compare_submission_figures.py

mkdir -p docs/environmental_modelling_software_submission_v1/graphical_abstract_rebuilt_code_v1
tectonic -X compile \
  docs/environmental_modelling_software_submission_v1/graphical_abstract.tex \
  --outdir docs/environmental_modelling_software_submission_v1/graphical_abstract_rebuilt_code_v1
pdftoppm -png -singlefile -scale-to-x 1328 -scale-to-y 531 \
  docs/environmental_modelling_software_submission_v1/graphical_abstract_rebuilt_code_v1/graphical_abstract.pdf \
  docs/environmental_modelling_software_submission_v1/graphical_abstract_rebuilt_code_v1/graphical_abstract
```
