# Repository cleanup plan

This is a non-destructive plan only. Nothing was deleted, moved, renamed, or rewritten during the audit.

## Priority 0: establish canonical truth before cleanup

1. Add a machine-readable `configs/pipeline_registry.yaml` that names the canonical entry point, input contract, output root, validation protocol, and status for TRAQID, MUMMA-281, road extraction, IDD vehicle extraction, ResNet temporal work, and FTIR/source attribution.
2. Record exact commands, Git commit, environment, random seeds, input checksums, and outputs for every new run. A lightweight `runs/<run_id>/manifest.json` is sufficient.
3. Declare validation tiers. For TRAQID, label random/two-fold sliding-window results “paper replication / leakage-contaminated,” time-balanced purged “within-date estimation,” and chronological date-wise “unseen-date stress test.”
4. Resolve whether FTIR/source attribution belongs here. If yes, add its data contract and pipeline; if no, state the external repository/location in the registry. Current explicit coverage is absent.

## Priority 1: reproducibility gaps

- Reconstruct or document the missing one-second feature stages between scripts 01 and 04. Existing outputs imply vehicle/road/depth extraction occurred, but no exact numbered commands remain.
- Add producers or provenance records for the IDD YOLO and IDD SegFormer artifacts. Include dataset version, class mapping, training code/notebook location, and checksums.
- Persist MUMMA-281 trained estimators or explicitly document that reports are evaluation-only and retraining is required.
- Replace the two absolute user paths with repository-relative/configured inputs: the OSM fusion script and TRAQID monthwise notebook.
- Repair or formally retire `src/vehicle_detection/detector.py`, whose `src.config` import has no in-repository provider.
- Add a locked environment (`requirements` hashes, `uv.lock`, Conda lock, or equivalent). `requirements.txt` alone does not establish the environment used for historical runs.
- Regenerate or quarantine by status the malformed legacy MUMMA metrics report before it is cited.

## Priority 2: separate source from generated products

Without moving existing files yet, adopt these boundaries for future work:

```text
experiments/<family>/scripts/       source entry points
experiments/<family>/configs/       immutable run configs
artifacts/<family>/<run_id>/        models, embeddings, predictions, figures
reports/<family>/<run_id>/          compact metrics and narrative reports
data/manifests/                      versioned small manifests
```

Then:

- Keep compact, reviewable final tables and provenance manifests in Git.
- Store tens of thousands of PNGs, embeddings, checkpoints, and repeated fold estimators in artifact storage or Git LFS/DVC.
- Do not store `.joblib` estimators under `reports/`; use an artifact/model area.
- Add `.DS_Store`, notebook checkpoints, matplotlib caches, `.save` backups, and run logs to ignore rules if not already covered.
- Add a retention policy: canonical run, latest validated run, and explicitly cited paper runs; mark all others historical before any later deletion decision.

## Priority 3: consolidate version families

Create deprecation metadata first; do not remove code until reproduction checks pass.

| Candidate | Proposed canonical replacement | Required verification |
|---|---|---|
| `experiments/mumma_v1` and root MUMMA scripts `03`–`06` | `experiments/mumma_281_pipeline_v1` | Match source table, row IDs, feature definitions, split protocol, and metrics. |
| TRAQID `.save` files | corresponding `.py` scripts | Confirm no unique diff worth preserving; commit history should carry old versions. |
| `src/utils/old_config_reference.py` | YAML configs | Verify every still-used value exists in active configs. |
| `scripts/archive/*_unused` | active road scripts/modules | Confirm no docs, external runners, or unpublished figures rely on them. |
| Root TRAQID modeling scripts | experiment-local paper/leakage-corrected branches | Map each historical report to its producer and preserve cited runs. |
| Road occlusion `11`, `11b`, `11c` | likely `11c` + conservative `12` | Validate v3 against polygon/visual audit packs; naming alone is insufficient. |
| TRAQID duplicate numeric prefixes | unique semantic stage IDs | Add stable IDs before renaming; many same-number scripts are legitimate branches. |

## Priority 4: introduce orchestration and tests

- Add a single CLI or workflow file per canonical family. Today, almost no scripts call other scripts; ordering exists only in names/default paths.
- Add smoke tests for manifest schemas, unique row IDs, image-path existence, split disjointness, and output-column contracts.
- Make leakage checks mandatory before model reporting: raw-frame overlap, timestamp/date overlap, target-window overlap, and grouped split assertions.
- Add small fixture data so feature extraction and model-table assembly can run without private/raw datasets.
- Add artifact integrity tests (SHA-256, expected architecture/class mapping, load test) for published weights.
- Fail fast when invoked outside the repository root or resolve paths relative to a discovered project root.

## Priority 5: documentation and naming

- Expand `scripts/README.md`; it currently documents an absent `02_preview_idd_detections.py` and omits most scripts.
- For each entry point, add a module docstring with purpose, required inputs, outputs, split assumptions, and an example command.
- Use one numbering scheme per linear workflow; use branch labels (`train_tabular`, `train_image`, `validate_purged`) rather than duplicate numbers.
- Add dataset cards for TRAQID, MUMMA-281, HVAQ, IDD detector data, and road-segmentation data, including licenses and provenance.
- Link final reports to the exact run manifest and checkpoint rather than relying on long folder names.

## Orphan review queue

These are candidates for human review, not deletion:

- explicit archives under `scripts/archive/`;
- `.save` source backups;
- empty experiment/report/model/figure directories;
- the empty root `notebooks/` directory;
- `src/alignment` package placeholder;
- older `mumma_v1` results-only tree;
- diagnostics/visual packs with regenerated or numbered variants;
- scripts with no output references and no documentation;
- `source` and `project_file_inventory.txt`, whose generation/ownership is not integrated into a workflow.

A script having no static caller is not enough to call it unused: nearly all entry points are intended for direct CLI execution. Require at least command-history evidence, report provenance, or maintainer confirmation before declaring an orphan.

## Safe execution order

1. Approve canonical pipeline/status registry.
2. Capture checksums and provenance for all currently cited outputs/models.
3. Reproduce canonical results from clean environments.
4. Label files/folders as canonical, historical, archived, or unknown.
5. Update docs and ignores for future runs.
6. Only then propose moves/deletions in a separate reviewed change.

## Acceptance criteria

- Every final metric links to one config, one input manifest checksum, one code commit, and one output directory.
- TRAQID reports cannot present overlap-contaminated results without an explicit warning.
- MUMMA-281 has one documented canonical modeling command and persisted/reproducible estimator behavior.
- Road and vehicle features have versioned schemas and model checksums.
- No active code contains a user-specific absolute path.
- FTIR/source attribution is either reproducible here or explicitly documented as external/absent.
