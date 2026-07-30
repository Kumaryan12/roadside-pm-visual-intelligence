#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path("experiments/traqid_pretraining_v1/reports/sequence_overlap_diagnostics")

RUNS = [
    {
        "protocol": "Random 70/15/15",
        "path": ROOT / "random_split_T7" / "overlap_summary.csv",
        "eval_name": "test",
        "interpretation": "Paper-style random sequence split; severe overlap leakage.",
    },
    {
        "protocol": "Two-fold random CV",
        "path": ROOT / "twofold_T7" / "overlap_summary.csv",
        "eval_name": "fold1_test",
        "interpretation": "Paper-style two-fold CV; still heavily overlap-contaminated.",
    },
    {
        "protocol": "Time-balanced purged",
        "path": ROOT / "time_balanced_purged_T7" / "overlap_summary.csv",
        "eval_name": "test",
        "interpretation": "No direct frame overlap; same dates and broadly balanced time/target distributions.",
    },
    {
        "protocol": "Chronological date-wise",
        "path": ROOT / "chrono_date_T7" / "overlap_summary.csv",
        "eval_name": "test",
        "interpretation": "No direct frame overlap; future/unseen-date stress test.",
    },
]

OUT_DIR = Path("experiments/traqid_pretraining_v1/reports/paper_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for run in RUNS:
    df = pd.read_csv(run["path"])

    if "eval_name" in df.columns:
        match = df[df["eval_name"].astype(str) == run["eval_name"]]
        if len(match) == 0:
            raise ValueError(f"No eval_name={run['eval_name']} in {run['path']}")
        r = match.iloc[0]
    else:
        r = df.iloc[0]

    rows.append({
        "Protocol": run["protocol"],
        "Train sequences": int(r["train_sequences"]),
        "Eval sequences": int(r["eval_sequences"]),
        "Mean max input overlap": round(float(r["max_input_overlap_mean"]), 4),
        "Median max input overlap": round(float(r["max_input_overlap_median"]), 4),
        "Max input overlap": int(r["max_input_overlap_max"]),
        "Frac overlap >= 1": round(float(r["max_input_overlap_frac_ge_1"]), 4),
        "Frac overlap >= 5": round(float(r["max_input_overlap_frac_ge_5"]), 4),
        "Frac overlap >= 6": round(float(r["max_input_overlap_frac_ge_6"]), 4),
        "Frac overlap = 0": round(float(r["max_input_overlap_frac_eq_0"]), 4),
        "Interpretation": run["interpretation"],
    })

out = pd.DataFrame(rows)

csv_path = OUT_DIR / "final_overlap_diagnostic_table.csv"
md_path = OUT_DIR / "final_overlap_diagnostic_table.md"

out.to_csv(csv_path, index=False)
md_path.write_text(out.to_markdown(index=False))

print("Saved:", csv_path)
print("Saved:", md_path)
print("")
print(out.to_string(index=False))
