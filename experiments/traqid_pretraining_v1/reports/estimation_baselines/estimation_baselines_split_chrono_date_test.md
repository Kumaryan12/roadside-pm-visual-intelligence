| model                     | split_col         | train_name   | eval_name   |   train_sequences |   eval_sequences | target   |         R2 |    RMSE |      MAE |
|:--------------------------|:------------------|:-------------|:------------|------------------:|-----------------:|:---------|-----------:|--------:|---------:|
| train_mean                | split_chrono_date | train        | test        |             16825 |             5462 | PM2.5    | -0.13554   | 44.1173 | 35.0894  |
| train_mean                | split_chrono_date | train        | test        |             16825 |             5462 | PM10     | -0.0241696 | 80.6839 | 53.6456  |
| train_mean                | split_chrono_date | train        | test        |             16825 |             5462 | aqi      | -0.0133515 | 86.5108 | 66.4731  |
| train_mean                | split_chrono_date | train        | test        |             16825 |             5462 | Average  | -0.0576871 | 70.4373 | 51.736   |
| previous_step_persistence | split_chrono_date | train        | test        |             16825 |             5462 | PM2.5    |  0.839394  | 16.5916 |  7.34064 |
| previous_step_persistence | split_chrono_date | train        | test        |             16825 |             5462 | PM10     |  0.907346  | 24.2679 | 10.129   |
| previous_step_persistence | split_chrono_date | train        | test        |             16825 |             5462 | aqi      |  0.847147  | 33.599  | 14.6197  |
| previous_step_persistence | split_chrono_date | train        | test        |             16825 |             5462 | Average  |  0.864629  | 24.8195 | 10.6965  |