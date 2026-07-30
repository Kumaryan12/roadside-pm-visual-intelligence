# MUMMA-281 PM2.5 experiment family

This directory is the canonical index for the existing 281-sample experiment
family. Historical source, reports, embeddings, and outputs remain under
`experiments/mumma_281_pipeline_v1` until equivalence and leakage-safe
generalization runs are complete.

Use the canonical compatibility runner:

```bash
python -m pipelines.pm25_prediction.mumma_281.run --list
python -m pipelines.pm25_prediction.mumma_281.run fair_optimization --dry-run
```

The runner does not alter historical behavior. In particular, historical
KFold/random-two-fold results must not be relabeled as generalized results.

## Migration order

1. Audit PM2.5 target and canonical identities.
2. Reproduce static and engineered historical metrics.
3. Replace row-level KFold with collection/date/spatially grouped outer splits.
4. Add OSM and AlphaEarth through fold-safe transformations.
5. Compare engineered, image, and fused models on identical splits.

