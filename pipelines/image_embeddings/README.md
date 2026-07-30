# Image embedding and temporal-model pipeline

This pipeline is intentionally separate from the canonical engineered feature
table. It consumes the same preprocessed manifest and stable sample IDs, then:

1. extracts front/rear or lens-specific ResNet50 embeddings;
2. stores embeddings as arrays plus an ID index;
3. constructs sequences only after assigning leakage-safe groups/splits;
4. trains GRU and LSTM variants on identical folds;
5. optionally fuses tabular features inside each training fold.

Embeddings are not written as hundreds of columns into the canonical feature
table. They are joined by `sample_id`/`sample_index` or `sequence_id` inside a
modeling run. PCA, scaling, and feature selection are fitted on training folds
only.

Historical implementations remain:

- MUMMA: `06_extract_resnet50_embeddings.py` and
  `07_resnet_t7_random2fold_lstm_fair_pm25.py`.
- TRAQID: paper-style sequence/ResNet extraction and CNN-LSTM/GRU scripts.

The random/two-fold historical models are benchmarks, not generalized models.

## Canonical command

```bash
python -m pipelines.image_embeddings.run \
  --image-manifest artifacts/runs/<feature-run>/manifests/preprocessed_frames.csv \
  --sample-table artifacts/runs/<feature-run>/tables/final_feature_table_with_splits.csv \
  --image-col processed_frame_path \
  --filter-col lens_id --filter-value 1 \
  --id-col sample_index \
  --timestamp-col sensor_timestamp \
  --split-col split_grouped \
  --group-cols matched_run_id date spatial_block_id \
  --target-cols value.sPM2 \
  --sequence-length 7 \
  --cell gru \
  --run-dir artifacts/runs/<image-run>
```

Run the same manifest and split with `--cell lstm` for a controlled comparison.
