# TRAQID Manifest EDA Report

## Basic summary

- **manifest_path**: experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv
- **rows**: 26678
- **columns**: 35
- **timestamp_col**: created_at_parsed
- **valid_timestamps**: 26678
- **unique_timestamps**: 8717
- **unique_dates**: 20
- **start_time**: 2022-10-22 18:00:00
- **end_time**: 2024-07-09 17:59:50


## Key timestamp findings

- Total image rows: **26678**
- Unique timestamps: **8717**
- Unique dates: **20**
- Start time: **2022-10-22 18:00:00**
- End time: **2024-07-09 17:59:50**


## Images per timestamp

|       |   num_rows_at_timestamp |
|:------|------------------------:|
| count |              8717       |
| mean  |                 3.06046 |
| std   |                 3.34458 |
| min   |                 1       |
| 25%   |                 1       |
| 50%   |                 1       |
| 75%   |                 7       |
| max   |                11       |


## Timestamp gaps, minutes

|       |    gap_minutes |
|:------|---------------:|
| count |   8716         |
| mean  |    103.424     |
| std   |   5495.87      |
| min   |      0.0833333 |
| 25%   |      0.133333  |
| 50%   |      0.166667  |
| 75%   |      1         |
| max   | 454414         |


## Numeric summary

|             |   count |     mean |       std |      min |     25% |     50% |      75% |     max |
|:------------|--------:|---------:|----------:|---------:|--------:|--------:|---------:|--------:|
| PM2.5       |   26678 |  66.5743 |  46.8281  |  12.9166 | 39.095  | 52.37   |  73.4426 | 400.863 |
| PM10        |   26678 | 126.178  | 113.606   |  32.1766 | 57.7676 | 72.731  | 157.018  | 669.851 |
| aqi         |   26678 | 143.601  | 109.035   |  36.545  | 69.78   | 97.1167 | 171.315  | 599.947 |
| Temperature |   26678 |  31.0102 |   5.64383 | -93.47   | 28.17   | 30.94   |  33.74   | 181.37  |
| Humidity    |   26678 |  38.6015 |  13.6111  |   9.4082 | 29.6301 | 38.177  |  46.527  |  76.167 |


## Correlations

|             |      PM2.5 |      PM10 |        aqi |   Temperature |   Humidity |
|:------------|-----------:|----------:|-----------:|--------------:|-----------:|
| PM2.5       |  1         |  0.751137 |  0.922037  |    -0.0511728 |  -0.143339 |
| PM10        |  0.751137  |  1        |  0.893632  |     0.11375   |  -0.363527 |
| aqi         |  0.922037  |  0.893632 |  1         |     0.0261065 |  -0.234534 |
| Temperature | -0.0511728 |  0.11375  |  0.0261065 |     1         |  -0.429566 |
| Humidity    | -0.143339  | -0.363527 | -0.234534  |    -0.429566  |   1        |


## Split columns detected

split_date_chrono


## Generated figures

- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/distribution_Humidity.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/distribution_PM10.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/distribution_PM2_5.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/distribution_Temperature.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/distribution_aqi.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/images_per_timestamp_hist.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/rows_by_hour.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/rows_per_date.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/scatter_PM10_vs_PM25.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/scatter_PM25_vs_Humidity.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/scatter_PM25_vs_Temperature.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/scatter_aqi_vs_PM25.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/timeseries_Humidity.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/timeseries_PM10.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/timeseries_PM2_5.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/timeseries_Temperature.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/timeseries_aqi.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/timestamp_gap_distribution_clipped.png`
- `experiments/traqid_pretraining_v1/reports/traqid_eda/figures/unique_timestamps_per_date.png`

