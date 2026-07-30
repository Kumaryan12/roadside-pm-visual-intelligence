| model                     | split_col     | train_name   | eval_name   |   train_sequences |   eval_sequences | target   |           R2 |     RMSE |      MAE |
|:--------------------------|:--------------|:-------------|:------------|------------------:|-----------------:|:---------|-------------:|---------:|---------:|
| train_mean                | split_twofold | fold1_train  | fold1_test  |             13279 |            13279 | PM2.5    | -4.4745e-05  |  46.84   | 31.697   |
| train_mean                | split_twofold | fold1_train  | fold1_test  |             13279 |            13279 | PM10     | -6.32165e-05 | 113.594  | 81.3489  |
| train_mean                | split_twofold | fold1_train  | fold1_test  |             13279 |            13279 | aqi      | -8.28923e-05 | 109.096  | 80.7344  |
| train_mean                | split_twofold | fold1_train  | fold1_test  |             13279 |            13279 | Average  | -6.36179e-05 |  89.8433 | 64.5934  |
| previous_step_persistence | split_twofold | fold1_train  | fold1_test  |             13279 |            13279 | PM2.5    |  0.945123    |  10.9725 |  4.68735 |
| previous_step_persistence | split_twofold | fold1_train  | fold1_test  |             13279 |            13279 | PM10     |  0.959089    |  22.9754 |  9.02937 |
| previous_step_persistence | split_twofold | fold1_train  | fold1_test  |             13279 |            13279 | aqi      |  0.940778    |  26.5479 | 11.496   |
| previous_step_persistence | split_twofold | fold1_train  | fold1_test  |             13279 |            13279 | Average  |  0.94833     |  20.1653 |  8.40423 |