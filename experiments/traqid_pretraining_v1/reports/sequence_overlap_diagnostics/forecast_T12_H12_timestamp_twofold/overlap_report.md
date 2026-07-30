# Sequence Overlap Diagnostic Report

- Manifest: `experiments/traqid_pretraining_v1/data/processed/forecasting_sequences/traqid_forecast_T12_H12_timestamp_manifest.csv`
- Split column: `split_twofold_forecast`
- Train split: `fold1_train`
- Eval splits: `fold1_test`
- Input column: `input_row_ids`
- Target column: `target_row_ids`
- Target id column: ``

## Summary

|   max_input_overlap_mean |   max_input_overlap_median |   max_input_overlap_max |   max_input_overlap_frac_ge_1 |   max_input_overlap_frac_ge_6 |   max_input_overlap_frac_ge_11 |   max_input_overlap_frac_ge_12 |   max_input_overlap_frac_ge_23 |   max_input_overlap_frac_eq_0 |   max_target_overlap_mean |   max_target_overlap_median |   max_target_overlap_max |   max_target_overlap_frac_ge_1 |   max_target_overlap_frac_ge_6 |   max_target_overlap_frac_ge_11 |   max_target_overlap_frac_ge_12 |   max_target_overlap_frac_ge_23 |   max_target_overlap_frac_eq_0 |   max_combined_overlap_mean |   max_combined_overlap_median |   max_combined_overlap_max |   max_combined_overlap_frac_ge_1 |   max_combined_overlap_frac_ge_6 |   max_combined_overlap_frac_ge_11 |   max_combined_overlap_frac_ge_12 |   max_combined_overlap_frac_ge_23 |   max_combined_overlap_frac_eq_0 | manifest                                                                                                              | split_col              | train_name   | eval_name   |   train_sequences |   eval_sequences | input_col     | target_col     | target_id_col   |
|-------------------------:|---------------------------:|------------------------:|------------------------------:|------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|------------------------------:|--------------------------:|----------------------------:|-------------------------:|-------------------------------:|-------------------------------:|--------------------------------:|--------------------------------:|--------------------------------:|-------------------------------:|----------------------------:|------------------------------:|---------------------------:|---------------------------------:|---------------------------------:|----------------------------------:|----------------------------------:|----------------------------------:|---------------------------------:|:----------------------------------------------------------------------------------------------------------------------|:-----------------------|:-------------|:------------|------------------:|-----------------:|:--------------|:---------------|:----------------|
|                  10.6646 |                         11 |                      11 |                             1 |                      0.999273 |                       0.754178 |                              0 |                              0 |                             0 |                   10.6646 |                          11 |                       11 |                              1 |                       0.999273 |                        0.754178 |                               0 |                               0 |                              0 |                     22.6646 |                            23 |                         23 |                                1 |                                1 |                                 1 |                                 1 |                          0.754178 |                                0 | experiments/traqid_pretraining_v1/data/processed/forecasting_sequences/traqid_forecast_T12_H12_timestamp_manifest.csv | split_twofold_forecast | fold1_train  | fold1_test  |              4128 |             4129 | input_row_ids | target_row_ids |                 |

## Interpretation guide

- `max_input_overlap`: maximum number of input frames/rows shared with any training sequence.
- `max_target_overlap`: maximum number of target-window rows shared with any training sequence. This matters for forecasting.
- `max_combined_overlap`: maximum overlap across input and target ids combined.
- High fractions near the sequence length indicate severe sliding-window leakage risk.
- Zero overlap does not guarantee full statistical independence; it only rules out direct row/frame overlap.