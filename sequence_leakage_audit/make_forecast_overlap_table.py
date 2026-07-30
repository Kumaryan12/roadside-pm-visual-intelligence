#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path("experiments/traqid_pretraining_v1/reports/sequence_overlap_diagnostics")
OUT_DIR = Path("experiments/traqid_pretraining_v1/reports/paper_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "Protocol": "Random 70/15/15",
        "path": ROOT / "forecast_T12_H12_timestamp_random" / "overlap_summary.csv",
        "eval_name": "test",
        "Interpretation": "Random forecasting-window split; severe input and target-window overlap."
    },
    {
        "Protocol": "Two-fold random CV",
        "path": ROOT / "forecast_T12_H12_timestamp_twofold" / "overlap_summary.csv",
        "eval_name": "fold1_test",
        "Interpretation": "Two-fold forecasting-window split; severe input and target-window overlap."
    },
    {
        "Protocol": "Chronological date-wise",
        "path": ROOT / "forecast_T12_H12_timestamp_chrono" / "overlap_summary.csv",
        "eval_name": "test",
        "Interpretation": "Chronological test split; no direct input or target-window overlap."
    },
]

rows = []

for run in RUNS:
    df = pd.read_csv(run["path"])
    r = df[df["eval_name"].astype(str) == run["eval_name"]].iloc[0]

    rows.append({
        "Protocol": run["Protocol"],
        "Train sequences": int(r["train_sequences"]),
        "Eval sequences": int(r["eval_sequences"]),
        "Input overlap >=11/12": round(float(r["max_input_overlap_frac_ge_11"]), 4),
        "Target overlap >=11/12": round(float(r["max_target_overlap_frac_ge_11"]), 4),
        "Combined overlap >=23/24": round(float((pd.read_csv(run["path"]).loc[df["eval_name"].astype(str) == run["eval_name"], "max_combined_overlap_frac_ge_12"]).iloc[0]), 4),
        "Max input overlap": int(r["max_input_overlap_max"]),
        "Max target overlap": int(r["max_target_overlap_max"]),
        "Max combined overlap": int(r["max_combined_overlap_max"]),
        "Interpretation": run["Interpretation"],
    })

out = pd.DataFrame(rows)

csv_path = OUT_DIR / "final_forecasting_window_overlap_table.csv"
md_path = OUT_DIR / "final_forecasting_window_overlap_table.md"

out.to_csv(csv_path, index=False)
md_path.write_text(out.to_markdown(index=False))

print("Saved:", csv_path)
print("Saved:", md_path)
print()
print(out.to_string(index=False))
