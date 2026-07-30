from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path("experiments/traqid_pretraining_v1")
OUT_DIR = ROOT / "reports/paper_diagnostics/naive_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("target_PM2.5", "PM2.5"),
    ("target_PM10", "PM10"),
    ("target_aqi", "AQI"),
]


SPLITS = [
    {
        "name": "time_balanced_purged",
        "manifest": ROOT / "data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest_time_balanced_purged.csv",
        "split_col": "split_time_balanced_purged",
        "train_name": "train",
        "test_name": "test",
        "model_metrics": ROOT / "reports/paper_cnn_lstm_tabular_fusion_time_balanced_purged/traqid_paper_resnet50_front_rear_concat_gap_traqid_paper_style_T7_front_sequence_manifest_time_balanced_purged_split_time_balanced_purged_tabular_fusion/metrics_paper_cnn_lstm_tabular_fusion.json",
    },
    {
        "name": "chrono_date",
        "manifest": ROOT / "data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest.csv",
        "split_col": "split_chrono_date",
        "train_name": "train",
        "test_name": "test",
        "model_metrics": ROOT / "reports/paper_cnn_lstm_tabular_fusion/traqid_paper_resnet50_front_rear_concat_gap_traqid_paper_style_T7_front_sequence_manifest_split_chrono_date_tabular_fusion/metrics_paper_cnn_lstm_tabular_fusion.json",
    },
]


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def metric_pack(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    rows = []

    for spec in SPLITS:
        print("=" * 90)
        print("Processing:", spec["name"])

        if not spec["manifest"].exists():
            raise FileNotFoundError(spec["manifest"])
        if not spec["model_metrics"].exists():
            raise FileNotFoundError(spec["model_metrics"])

        df = pd.read_csv(spec["manifest"])
        train = df[df[spec["split_col"]] == spec["train_name"]].copy()
        test = df[df[spec["split_col"]] == spec["test_name"]].copy()

        model_metrics = json.loads(spec["model_metrics"].read_text())["test"]

        for col, label in TARGETS:
            y_train = train[col].to_numpy(dtype=float)
            y_test = test[col].to_numpy(dtype=float)

            # Train-mean baseline: realistic no-test-label baseline.
            train_mean_pred = np.full_like(y_test, fill_value=np.nanmean(y_train), dtype=float)
            train_mean_metrics = metric_pack(y_test, train_mean_pred)

            # Test-mean oracle: R² exactly 0 by construction, useful as RMSE reference.
            test_mean_pred = np.full_like(y_test, fill_value=np.nanmean(y_test), dtype=float)
            test_mean_metrics = metric_pack(y_test, test_mean_pred)

            rows.append({
                "split": spec["name"],
                "target": label,

                "model_R2": model_metrics[label]["R2"],
                "model_RMSE": model_metrics[label]["RMSE"],
                "model_MAE": model_metrics[label]["MAE"],

                "train_mean_R2": train_mean_metrics["R2"],
                "train_mean_RMSE": train_mean_metrics["RMSE"],
                "train_mean_MAE": train_mean_metrics["MAE"],

                "test_mean_oracle_R2": test_mean_metrics["R2"],
                "test_mean_oracle_RMSE": test_mean_metrics["RMSE"],
                "test_mean_oracle_MAE": test_mean_metrics["MAE"],

                "train_target_mean": float(np.nanmean(y_train)),
                "test_target_mean": float(np.nanmean(y_test)),
                "train_target_std": float(np.nanstd(y_train)),
                "test_target_std": float(np.nanstd(y_test)),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            })

    out = pd.DataFrame(rows)

    csv_path = OUT_DIR / "naive_baseline_time_balanced_purged_vs_chrono.csv"
    md_path = OUT_DIR / "naive_baseline_time_balanced_purged_vs_chrono.md"

    out.to_csv(csv_path, index=False)

    pretty = out.copy()
    for c in pretty.columns:
        if pretty[c].dtype.kind in "fc":
            pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")

    md_path.write_text(pretty.to_markdown(index=False), encoding="utf-8")

    print("\nSaved:", csv_path)
    print("Saved:", md_path)
    print("\nSummary:")
    print(pretty[[
        "split", "target",
        "model_R2", "train_mean_R2", "model_RMSE", "train_mean_RMSE",
        "train_target_mean", "test_target_mean"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()