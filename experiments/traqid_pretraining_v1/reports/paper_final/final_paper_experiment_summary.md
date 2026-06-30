# Final Paper Experiment Summary

## Best random-split model

ResNet50 front/rear concat LSTM + tabular fusion achieved:

- PM2.5 R² = 0.9444, RMSE = 10.8752
- PM10 R² = 0.9660, RMSE = 21.2207
- AQI R² = 0.9508, RMSE = 24.2424
- Average R² = 0.9537, RMSE = 18.7795

## Two-fold CV average

- PM2.5 R² = 0.9055, RMSE = 14.3772
- PM10 R² = 0.9430, RMSE = 27.0318
- AQI R² = 0.9161, RMSE = 31.5126
- Average R² = 0.9215, RMSE = 24.3072

## Chronological date-wise evaluation

- PM2.5 R² = -0.8300, RMSE = 56.0062
- PM10 R² = -1.4386, RMSE = 124.5012
- AQI R² = -1.0144, RMSE = 121.9729
- Average R² = -1.0944, RMSE = 100.8268

## Sequence length

The T-sweep from T=2 to T=9 selected T=7 as the best overall sequence length by average R².

## Core conclusion

The enhanced CNN-LSTM model reaches paper-level random-split performance on TRAQID, but chronological date-wise testing fails, showing that high random-split R² does not guarantee unseen-date generalization.
