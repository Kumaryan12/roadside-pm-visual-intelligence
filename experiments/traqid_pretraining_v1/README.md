# TRAQID experiment family

The canonical comparison specification is `configs/benchmark_matrix_v1.yaml`.
It retains historical random splits while separating them from valid
generalization evidence.

Rebuild the consolidated evidence table after any run:

```bash
python -m pipelines.image_embeddings.aggregate_traqid_benchmarks
```

Run the primary image models with unseen-date outer folds:

```bash
python -m pipelines.image_embeddings.traqid_outer_crossfit --cell gru
python -m pipelines.image_embeddings.traqid_outer_crossfit --cell lstm \
  --output-dir artifacts/runs/traqid_resnet50_lstm_residual_outercv_v1
```

Random two-fold results are historical/debug comparisons only. Overlapping T7
windows make them unsuitable for claims about deployment on new dates or routes.

## Presentation architecture reproduction

The Keynote result (`R2=0.965556`) is reproduced through a compatibility wrapper
around the original 3-fold OOF ResNet50-GRU plus clipped ExtraTrees experiment:

```bash
python -m pipelines.image_embeddings.traqid_presentation_reproduction --dry-run
python -m pipelines.image_embeddings.traqid_presentation_reproduction
```

This deliberately recreates the random overlapping-window benchmark. It is the
first track for result equivalence, not the fair-model validation protocol.
