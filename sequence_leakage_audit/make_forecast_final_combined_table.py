#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path("experiments/traqid_pretraining_v1")
OVERLAP_ROOT = ROOT / "reports/sequence_overlap_diagnostics"

NUM_BASELINE_PATH = ROOT / "reports/forecast_numerical_baselines_T12_H12_timestamp/forecast_numerical_baseline_summary.csv"
LSTM_PATH = ROOT / "reports/forecast_numerical_lstm_T12_H12_timestamp/forecast_numerical_lstm_summary.csv"

OUT_DIR = ROOT / "reports/paper_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "Protocol": "Random 70/15/15",
        "overlap_path": OVERLAP_ROOT / "forecast_T12_H12_timestamp_random" / "overlap_summary.csv",
        "overlap_eval": "test",
        "split_col": "split_random_forecast",
        "eval_name": "test",
        "Interpretation": "High R2 under severe input and target-window overlap.",
    },
    {
        "Protocol": "Two-fold random CV",
        "overlap_path": OVERLAP_ROOT / "forecast_T12_H12_timestamp_twofold" / "overlap_summary.csv",
        "overlap_eval": "fold1_test",
        "split_col": "split_twofold_forecast",
        "eval_name": "fold1_test",
        "Interpretation": "High two-fold R2 under severe input and target-window overlap.",
    },
    {
        "Protocol": "Chronological date-wise",
        "overlap_path": OVERLAP_ROOT / "forecast_T12_H12_timestamp_chrono" / "overlap_summary.csv",
        "overlap_eval": "test",
        "split_col": "split_chrono_forecast",
        "eval_name": "test",
        "Interpretation": "No direct input/target overlap; future-window performance drops sharply.",
    },
]

num_df = pd.read_csv(NUM_BASELINE_PATH)
lstm_df = pd.read_csv(LSTM_PATH)

rows = []

for run in RUNS:
    overlap_df = pd.read_csv(run["overlap_path"])
    ov = overlap_df[overlap_df["eval_name"].astype(str) == run["overlap_eval"]].iloc[0]

    num_subset = num_df[
        (num_df["split_col"].astype(str) == run["split_col"]) &
        (num_df["eval_name"].astype(str) == run["eval_name"])
    ]

    extra = num_subset[num_subset["model"].astype(str) == "extratrees"].iloc[0]
    ridge = num_subset[num_subset["model"].astype(str) == "ridge"].iloc[0]
    persistence = num_subset[num_subset["model"].astype(str) == "persistence"].iloc[0]

    lstm_subset = lstm_df[
        (lstm_df["split_col"].astype(str) == run["split_col"]) &
        (lstm_df["eval_name"].astype(str) == run["eval_name"])
    ]

    if len(lstm_subset) == 0:
        raise ValueError(f"No LSTM row found for {run['split_col']} / {run['eval_name']}")

    lstm = lstm_subset.iloc[0]

    rows.append({
        "Protocol": run["Protocol"],
        "Input overlap >=11/12": round(float(ov["max_input_overlap_frac_ge_11"]), 4),
        "Target overlap >=11/12": round(float(ov["max_target_overlap_frac_ge_11"]), 4),
        "Max combined overlap": int(ov["max_combined_overlap_max"]),
        "Persistence Avg R2": round(float(persistence["Avg_R2"]), 4),
        "Ridge Avg R2": round(float(ridge["Avg_R2"]), 4),
        "ExtraTrees Avg R2": round(float(extra["Avg_R2"]), 4),
        "Numerical LSTM Avg R2": round(float(lstm["Avg_R2"]), 4),
        "ExtraTrees Avg RMSE": round(float(extra["Avg_RMSE"]), 4),
        "Numerical LSTM Avg RMSE": round(float(lstm["Avg_RMSE"]), 4),
        "Interpretation": run["Interpretation"],
    })

out = pd.DataFrame(rows)

csv_path = OUT_DIR / "final_forecasting_overlap_baseline_lstm_table.csv"
md_path = OUT_DIR / "final_forecasting_overlap_baseline_lstm_table.md"

out.to_csv(csv_path, index=False)
md_path.write_text(out.to_markdown(index=False))

print("Saved:", csv_path)
print("Saved:", md_path)
print()
print(out.to_string(index=False))
