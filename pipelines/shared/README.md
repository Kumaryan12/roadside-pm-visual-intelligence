# Shared feature pipeline

Planned stages: canonical sample manifest, image preprocessing, IDD vehicle
features, road features, OSM, AlphaEarth, OPC harmonization, and modeling-table
assembly. Current producers remain documented in `docs/data_lineage.md`.

## Canonical feature-table runner

The raw video and sensor-table workflow is registered under `feature_table/`:

```bash
python -m pipelines.shared.feature_table.run --preflight
python -m pipelines.shared.feature_table.run --dry-run --to-stage segment_road_features
python -m pipelines.shared.feature_table.run --run-id <id> --resume
```

Before the full run, set `camera.fx_px`, verify the video root, and supply the
registered OSM and AlphaEarth feature tables in
`configs/pipelines/feature_table_v1.yaml`.
