#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

OUT_DIR = Path("experiments/traqid_pretraining_v1/reports/paper_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "Protocol": "Random 70/15/15",
        "Test overlap >=6/7": 0.9091,
        "PM2.5 R2": 0.9444,
        "PM10 R2": 0.9660,
        "AQI R2": 0.9508,
        "Average R2": 0.9537,
        "Average RMSE": 18.7795,
        "Interpretation": "Paper-style random sequence validation; severe sliding-window overlap leakage."
    },
    {
        "Protocol": "Two-fold random CV",
        "Test overlap >=6/7": 0.7481,
        "PM2.5 R2": 0.9055,
        "PM10 R2": 0.9430,
        "AQI R2": 0.9161,
        "Average R2": 0.9215,
        "Average RMSE": 24.3072,
        "Interpretation": "Paper-style two-fold CV; still highly overlap-contaminated."
    },
    {
        "Protocol": "Time-balanced purged",
        "Test overlap >=6/7": 0.0000,
        "PM2.5 R2": 0.2125,
        "PM10 R2": 0.5914,
        "AQI R2": 0.2987,
        "Average R2": 0.3675,
        "Average RMSE": 65.7048,
        "Interpretation": "No direct frame overlap; same dates and broadly balanced time/target distributions."
    },
    {
        "Protocol": "Chronological date-wise",
        "Test overlap >=6/7": 0.0000,
        "PM2.5 R2": -0.8300,
        "PM10 R2": -1.4386,
        "AQI R2": -1.0144,
        "Average R2": -1.0944,
        "Average RMSE": 100.8268,
        "Interpretation": "No direct frame overlap; future/unseen-date stress test."
    },
]

df = pd.DataFrame(rows)

csv_path = OUT_DIR / "final_validation_ladder_overlap_performance.csv"
md_path = OUT_DIR / "final_validation_ladder_overlap_performance.md"

df.to_csv(csv_path, index=False)
md_path.write_text(df.to_markdown(index=False))

print("Saved:", csv_path)
print("Saved:", md_path)
print()
print(df.to_string(index=False))
