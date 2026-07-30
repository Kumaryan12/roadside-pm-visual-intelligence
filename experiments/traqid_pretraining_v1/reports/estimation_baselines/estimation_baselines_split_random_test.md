| model                     | split_col    | train_name   | eval_name   |   train_sequences |   eval_sequences | target   |           R2 |     RMSE |      MAE |
|:--------------------------|:-------------|:-------------|:------------|------------------:|-----------------:|:---------|-------------:|---------:|---------:|
| train_mean                | split_random | train        | test        |             18591 |             3983 | PM2.5    | -8.77696e-05 |  46.1185 | 31.3344  |
| train_mean                | split_random | train        | test        |             18591 |             3983 | PM10     | -0.000429165 | 115.04   | 82.0358  |
| train_mean                | split_random | train        | test        |             18591 |             3983 | aqi      | -0.000395043 | 109.341  | 80.9409  |
| train_mean                | split_random | train        | test        |             18591 |             3983 | Average  | -0.000303993 |  90.1665 | 64.7704  |
| previous_step_persistence | split_random | train        | test        |             18591 |             3983 | PM2.5    |  0.943369    |  10.9745 |  4.69331 |
| previous_step_persistence | split_random | train        | test        |             18591 |             3983 | PM10     |  0.955707    |  24.206  |  9.30987 |
| previous_step_persistence | split_random | train        | test        |             18591 |             3983 | aqi      |  0.935174    |  27.8337 | 11.7219  |
| previous_step_persistence | split_random | train        | test        |             18591 |             3983 | Average  |  0.94475     |  21.0047 |  8.57503 |