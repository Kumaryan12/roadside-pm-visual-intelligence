# Canonical pipelines

This directory is the home for new, ordered pipeline entry points. It is
additive: existing scripts remain in place until a canonical pipeline has
reproduced their outputs and a compatibility wrapper has been validated.

- `shared/`: manifest building and reusable feature extraction.
- `pm25_prediction/`: leakage-free multimodal PM2.5 prediction.
- `particle_source_attribution/`: OPC/density modeling and constrained
  source-proxy attribution.

Pipeline files should orchestrate code from `src/roadside_pm`; they should not
contain large reusable implementations.

