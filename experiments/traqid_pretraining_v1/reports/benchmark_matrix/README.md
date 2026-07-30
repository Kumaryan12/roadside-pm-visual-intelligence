# TRAQID benchmark matrix

Random sequence results are diagnostic replications, not generalization estimates. The primary evidence is date-grouped outer CV; chronological date-wise evaluation is the hardest stress test.

| protocol | evidence_status | method | fold | R2 | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| random_70_15_15 | contaminated | ResNet50(front+rear)-LSTM + tabular fusion | aggregate | 0.9444 | 10.8752 | nan |
| random_70_15_15 | contaminated | previous_step_persistence | test | 0.9434 | 10.9745 | 4.6933 |
| random_70_15_15 | contaminated | train_mean | test | -0.0001 | 46.1185 | 31.3344 |
| random_twofold | contaminated | ResNet50(front+rear)-LSTM + tabular fusion | aggregate | 0.9055 | 14.3772 | nan |
| random_twofold | contaminated | previous_step_persistence | fold1_test | 0.9451 | 10.9725 | 4.6873 |
| random_twofold | contaminated | train_mean | fold1_test | -0.0000 | 46.8400 | 31.6970 |
| old_purged_block | limited | ResNet50(front+rear)-LSTM + tabular fusion | aggregate | -0.0226 | 43.9824 | nan |
| time_balanced_purged | limited | ResNet50(front+rear)-LSTM + tabular fusion | aggregate | 0.2125 | 39.5613 | nan |
| time_balanced_purged | limited | pooled ResNet50 + extratrees | aggregate | 0.0805 | 42.7488 | 28.3060 |
| time_balanced_purged | limited | pooled ResNet50 + extratrees + tabular | aggregate | 0.4375 | 33.4352 | 21.7659 |
| time_balanced_purged | limited | pooled ResNet50 + rf + tabular | aggregate | 0.5010 | 31.4905 | 20.6462 |
| time_balanced_purged | limited | previous_step_persistence | test | 0.9353 | 11.3425 | 4.8505 |
| time_balanced_purged | limited | train_mean | test | -0.0000 | 44.5797 | 30.4609 |
| date_grouped_outer_cv | primary | ResNet50-GRU | fold_1 | 0.2330 | 48.9680 | 24.0812 |
| chronological_date | robust_stress_test | ResNet50(front+rear)-LSTM + tabular fusion | aggregate | -0.8300 | 56.0062 | nan |
| chronological_date | robust_stress_test | previous_step_persistence | test | 0.8394 | 16.5916 | 7.3406 |
| chronological_date | robust_stress_test | train_mean | test | -0.1355 | 44.1173 | 35.0894 |

## Interpretation

In random two-fold CV, previous-step persistence reaches R2=0.9451; the historical CNN-LSTM fusion reaches R2=0.9055. This indicates that temporal adjacency and overlapping windows dominate the random-split score.

## Decision rule

Select models using mean, dispersion, and worst-fold error across date-grouped outer folds. Use random splits only to reproduce earlier experiments and debug optimization.
