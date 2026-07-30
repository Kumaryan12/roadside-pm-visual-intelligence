| model       | split_col              | train_name   | eval_name   |     PM2.5_R2 |       PM10_R2 |       Avg_R2 |   PM2.5_RMSE |   PM10_RMSE |   Avg_RMSE |
|:------------|:-----------------------|:-------------|:------------|-------------:|--------------:|-------------:|-------------:|------------:|-----------:|
| train_mean  | split_random_forecast  | train        | test        | -1.70469e-05 |  -1.69277e-05 | -1.69873e-05 |      60.6619 |    149.005  |   104.833  |
| persistence | split_random_forecast  | train        | test        |  0.767808    |   0.871717    |  0.819763    |      29.2305 |     53.3679 |    41.2992 |
| ridge       | split_random_forecast  | train        | test        |  0.805479    |   0.901817    |  0.853648    |      26.7544 |     46.6889 |    36.7217 |
| extratrees  | split_random_forecast  | train        | test        |  0.874317    |   0.942656    |  0.908487    |      21.5055 |     35.6813 |    28.5934 |
| train_mean  | split_twofold_forecast | fold1_train  | fold1_test  | -0.000397325 |  -0.000243425 | -0.000320375 |      60.1589 |    146.689  |   103.424  |
| persistence | split_twofold_forecast | fold1_train  | fold1_test  |  0.739092    |   0.860129    |  0.79961     |      30.7226 |     54.8541 |    42.7883 |
| ridge       | split_twofold_forecast | fold1_train  | fold1_test  |  0.796896    |   0.892234    |  0.844565    |      27.1065 |     48.1488 |    37.6276 |
| extratrees  | split_twofold_forecast | fold1_train  | fold1_test  |  0.851855    |   0.930449    |  0.891152    |      23.1503 |     38.6807 |    30.9155 |
| train_mean  | split_chrono_forecast  | train        | test        | -0.151757    | -12.4062      | -6.27898     |      54.7361 |    114.184  |    84.46   |
| persistence | split_chrono_forecast  | train        | test        |  0.279243    |   0.212106    |  0.245674    |      43.3    |     27.6812 |    35.4906 |
| ridge       | split_chrono_forecast  | train        | test        |  0.446584    |   0.0972165   |  0.2719      |      37.942  |     29.6308 |    33.7864 |
| extratrees  | split_chrono_forecast  | train        | test        |  0.32269     |   0.261581    |  0.292135    |      41.9747 |     26.798  |    34.3864 |