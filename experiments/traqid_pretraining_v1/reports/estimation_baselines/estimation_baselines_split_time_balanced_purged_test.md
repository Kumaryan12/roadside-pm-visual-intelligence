| model                     | split_col                  | train_name   | eval_name   |   train_sequences |   eval_sequences | target   |           R2 |     RMSE |      MAE |
|:--------------------------|:---------------------------|:-------------|:------------|------------------:|-----------------:|:---------|-------------:|---------:|---------:|
| train_mean                | split_time_balanced_purged | train        | test        |             15076 |             6691 | PM2.5    | -5.29545e-06 |  44.5797 | 30.4609  |
| train_mean                | split_time_balanced_purged | train        | test        |             15076 |             6691 | PM10     | -0.000286211 | 108.735  | 79.9592  |
| train_mean                | split_time_balanced_purged | train        | test        |             15076 |             6691 | aqi      | -2.30238e-06 | 105.152  | 79.7744  |
| train_mean                | split_time_balanced_purged | train        | test        |             15076 |             6691 | Average  | -9.79362e-05 |  86.1555 | 63.3982  |
| previous_step_persistence | split_time_balanced_purged | train        | test        |             15076 |             6691 | PM2.5    |  0.935264    |  11.3425 |  4.85045 |
| previous_step_persistence | split_time_balanced_purged | train        | test        |             15076 |             6691 | PM10     |  0.959616    |  21.8479 |  8.89751 |
| previous_step_persistence | split_time_balanced_purged | train        | test        |             15076 |             6691 | aqi      |  0.932946    |  27.229  | 11.8921  |
| previous_step_persistence | split_time_balanced_purged | train        | test        |             15076 |             6691 | Average  |  0.942609    |  20.1398 |  8.54667 |