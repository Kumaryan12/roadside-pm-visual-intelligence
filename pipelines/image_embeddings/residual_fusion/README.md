# Image-base plus tabular residual correction

The image GRU/LSTM is the base estimator. The tabular model predicts only its
remaining error:

```text
base_pm25 = ResNet50 sequence -> GRU/LSTM
residual = actual_pm25 - out_of_fold_base_pm25
predicted_residual = tabular_model(engineered features)
corrected_pm25 = max(0, base_pm25 + predicted_residual)
```

Residual training refuses in-sample image predictions. Accepted prediction
origins are `oof`, `outer_val`, `outer_test`, and `heldout`.

```bash
python -m pipelines.image_embeddings.combine_base_predictions \
  --inputs <image-run>/gru_T7/predictions_val.csv <image-run>/gru_T7/predictions_test.csv \
  --output-csv <fusion-run>/safe_base_predictions.csv

python -m pipelines.image_embeddings.residual_correct \
  --base-predictions <fusion-run>/safe_base_predictions.csv \
  --sequence-manifest <image-run>/sequences_T7.csv \
  --feature-table <feature-run>/tables/final_feature_table.csv \
  --target value.sPM2 \
  --sample-key sample_index \
  --group-col matched_run_id \
  --model extra_trees \
  --output-dir <fusion-run>/extra_trees
```

For the final paper protocol, generate base predictions by outer grouped
cross-fitting for every evaluated row. The val/test combination above is useful
for integration testing and exploratory grouped correction, but it does not
replace an untouched external test set.

