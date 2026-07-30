# Final PM2.5 Result Summary

## Final Best Model

3-fold OOF ResNet50-GRU ensemble + clipped ExtraTrees residual correction.

## Base Model

- Input: T=7 front + rear image sequence
- Feature extractor: ResNet50 GAP features
- Front embedding: 2048
- Rear embedding: 2048
- Concatenated embedding: 4096 per frame
- Temporal model: GRU
- Output: PM2.5 only

## Residual Correction

- Residual target: actual PM2.5 - base predicted PM2.5
- Residual input:
  - YOLO traffic features
  - road-condition features
  - tabular/weather/time features
  - base PM2.5 prediction features
- Aggregation over T=7:
  - last
  - mean
  - std
  - min
  - max
- Residual model: ExtraTreesRegressor conservative
- Residual clipping: ±12.5 PM2.5

## Final Test Result

| Model | R2 | RMSE | MAE |
|---|---:|---:|---:|
| Single PM2.5 ResNet50-GRU base | 0.951356 | 10.358972 | 5.991337 |
| Tuned residual + base prediction | 0.956042 | 9.847425 | 5.439865 |
| 3-fold ResNet50-GRU base ensemble | 0.959656 | 9.433944 | 5.113495 |
| OOF residual correction, unclipped | 0.964362 | 8.866612 | 4.233748 |
| OOF residual correction, clipped ±12.5 | 0.965556 | 8.716917 | 4.211589 |

## Interpretation

The final improvement comes from two components:

1. 3-fold base-model ensembling improves the ResNet50-GRU prediction.
2. OOF residual correction learns remaining error using engineered YOLO, road, tabular, and base-prediction features.

Residual clipping prevents rare over-correction and improves robustness.

## Important Limitation

This is a random-split benchmark result. Because T=7 sliding windows can overlap, this should not be claimed as proof of future-time deployment generalization. Chronological or purged validation remains the stricter test.