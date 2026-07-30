# Sequence Overlap Diagnostic Report

- Manifest: `experiments/traqid_pretraining_v1/data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest.csv`
- Split column: `split_twofold`
- Train split: `fold1_train`
- Eval splits: `fold1_test`
- Input column: `seq_row_ids`
- Target column: ``
- Target id column: `target_row_id`

## Summary

|   max_input_overlap_mean |   max_input_overlap_median |   max_input_overlap_max |   max_input_overlap_frac_ge_1 |   max_input_overlap_frac_ge_3 |   max_input_overlap_frac_ge_5 |   max_input_overlap_frac_ge_6 |   max_input_overlap_frac_eq_0 |   max_target_overlap_mean |   max_target_overlap_median |   max_target_overlap_max |   max_target_overlap_frac_ge_1 |   max_target_overlap_frac_ge_3 |   max_target_overlap_frac_ge_5 |   max_target_overlap_frac_ge_6 |   max_target_overlap_frac_eq_0 |   max_combined_overlap_mean |   max_combined_overlap_median |   max_combined_overlap_max |   max_combined_overlap_frac_ge_1 |   max_combined_overlap_frac_ge_3 |   max_combined_overlap_frac_ge_5 |   max_combined_overlap_frac_ge_6 |   max_combined_overlap_frac_eq_0 | manifest                                                                                                                 | split_col     | train_name   | eval_name   |   train_sequences |   eval_sequences | input_col   | target_col   | target_id_col   |
|-------------------------:|---------------------------:|------------------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------:|--------------------------:|----------------------------:|-------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|----------------------------:|------------------------------:|---------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|---------------------------------:|:-------------------------------------------------------------------------------------------------------------------------|:--------------|:-------------|:------------|------------------:|-----------------:|:------------|:-------------|:----------------|
|                  5.66662 |                          6 |                       6 |                      0.999925 |                      0.996009 |                      0.937947 |                      0.748099 |                   7.53069e-05 |                         0 |                           0 |                        0 |                              0 |                              0 |                              0 |                              0 |                              1 |                     5.66662 |                             6 |                          6 |                         0.999925 |                         0.996009 |                         0.937947 |                         0.748099 |                      7.53069e-05 | experiments/traqid_pretraining_v1/data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest.csv | split_twofold | fold1_train  | fold1_test  |             13279 |            13279 | seq_row_ids |              | target_row_id   |

## Interpretation guide

- `max_input_overlap`: maximum number of input frames/rows shared with any training sequence.
- `max_target_overlap`: maximum number of target-window rows shared with any training sequence. This matters for forecasting.
- `max_combined_overlap`: maximum overlap across input and target ids combined.
- High fractions near the sequence length indicate severe sliding-window leakage risk.
- Zero overlap does not guarantee full statistical independence; it only rules out direct row/frame overlap.