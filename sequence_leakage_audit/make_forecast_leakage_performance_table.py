#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path("experiments/traqid_pretraining_v1")
OVERLAP_ROOT = ROOT / "reports/sequence_overlap_diagnostics"
BASELINE_PATH = ROOT / "reports/forecast_numerical_baselines_T12_H12_timestamp/forecast_numerical_baseline_summary.csv"
OUT_DIR = ROOT / "reports/paper_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "Protocol": "Random 70/15/15",
        "overlap_path": OVERLAP_ROOT / "forecast_T12_H12_timestamp_random" / "overlap_summary.csv",
        "overlap_eval_name": "test",
        "split_col": "split_random_forecast",
        "eval_name": "test",
        "Interpretation": "High numerical forecasting score under severe input and target-window overlap."
    },
    {
        "Protocol": "Two-fold random CV",
        "overlap_path": OVERLAP_ROOT / "forecast_T12_H12_timestamp_twofold" / "overlap_summary.csv",
        "overlap_eval_name": "fold1_test",
        "split_col": "split_twofold_forecast",
        "eval_name": "fold1_test",
        "Interpretation": "High two-fold score under severe input and target-window overlap."
    },
    {
        "Protocol": "Chronological date-wise",
        "overlap_path": OVERLAP_ROOT / "forecast_T12_H12_timestamp_chrono" / "overlap_summary.csv",
        "overlap_eval_name": "test",
        "split_col": "split_chrono_forecast",
        "eval_name": "test",
        "Interpretation": "No direct input/target overlap; much lower future-window generalization."
    },
]

baseline = pd.read_csv(BASELINE_PATH)

rows = []

for run in RUNS:
    overlap = pd.read_csv(run["overlap_path"])
    ov = overlap[overlap["eval_name"].astype(str) == run["overlap_eval_name"]].iloc[0]

    b = baseline[
        (baseline["split_col"].astype(str) == run["split_col"]) &
        (baseline["eval_name"].astype(str) == run["eval_name"])
    ].copy()

    # exclude train_mean from "best" model selection
    b_models = b[b["model"] != "train_mean"].copy()
    best = b_models.sort_values("Avg_R2", ascending=False).iloc[0]

    persistence = b[b["model"] == "persistence"].iloc[0]
    ridge = b[b["model"] == "ridge"].iloc[0] if "ridge" in set(b["model"]) else None
    extratrees = b[b["model"] == "extratrees"].iloc[0] if "extratrees" in set(b["model"]) else None

    rows.append({
        "Protocol": run["Protocol"],
        "Input overlap >=11/12": round(float(ov["max_input_overlap_frac_ge_11"]), 4),
        "Target overlap >=11/12": round(float(ov["max_target_overlap_frac_ge_11"]), 4),
        "Max input overlap": int(ov["max_input_overlap_max"]),
        "Max target overlap": int(ov["max_target_overlap_max"]),
        "Max combined overlap": int(ov["max_combined_overlap_max"]),
        "Persistence Avg R2": round(float(persistence["Avg_R2"]), 4),
        "Ridge Avg R2": round(float(ridge["Avg_R2"]), 4) if ridge is not None else None,
        "ExtraTrees Avg R2": round(float(extratrees["Avg_R2"]), 4) if extratrees is not None else None,
        "Best numerical model": str(best["model"]),
        "Best numerical Avg R2": round(float(best["Avg_R2"]), 4),
        "Best numerical Avg RMSE": round(float(best["Avg_RMSE"]), 4),
        "Interpretation": run["Interpretation"],
    })

out = pd.DataFrame(rows)

csv_path = OUT_DIR / "final_forecasting_leakage_performance_table.csv"
md_path = OUT_DIR / "final_forecasting_leakage_performance_table.md"

out.to_csv(csv_path, index=False)
md_path.write_text(out.to_markdown(index=False))

print("Saved:", csv_path)
print("Saved:", md_path)
print()
print(out.to_string(index=False))
