"""Consolidate historical and current TRAQID PM2.5 benchmark evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/traqid_pretraining_v1"
PROTOCOLS = {
    "random_70_15_15": ("contaminated", "Random sequence split; sliding windows share frames across partitions."),
    "random_twofold": ("contaminated", "Random two-fold sequence CV; sliding windows share frames across folds."),
    "old_purged_block": ("limited", "No shared frames, but the test block is later/night-skewed within dates."),
    "time_balanced_purged": ("limited", "No shared frames, but train and test contain the same collection dates."),
    "chronological_date": ("robust_stress_test", "No shared frames; tests later, unseen collection dates."),
    "date_grouped_outer_cv": ("primary", "Outer test dates are absent from model fitting and early stopping."),
}


def number(row: dict, *names: str) -> float:
    for name in names:
        if name in row and pd.notna(row[name]): return float(row[name])
    return float("nan")


def add(rows, protocol, method, r2, rmse, source, mae=np.nan, fold="aggregate", notes=""):
    status, interpretation = PROTOCOLS[protocol]
    source = Path(source)
    rows.append({"protocol": protocol, "evidence_status": status, "method": method,
        "fold": fold, "target": "PM2.5", "R2": r2, "RMSE": rmse, "MAE": mae,
        "interpretation": interpretation, "notes": notes,
        "source": str(source.relative_to(ROOT) if source.is_absolute() else source)})


def collect_ladder(rows):
    path = EXP / "reports/paper_final/final_validation_ladder_with_overlap.csv"
    if not path.exists(): return
    names = {"Random 70/15/15": "random_70_15_15", "Two-fold random CV": "random_twofold",
        "Old purged-block": "old_purged_block", "Time-balanced purged": "time_balanced_purged",
        "Chronological date-wise": "chronological_date"}
    for _, row in pd.read_csv(path).iterrows():
        if row["Protocol"] in names:
            add(rows, names[row["Protocol"]], "ResNet50(front+rear)-LSTM + tabular fusion",
                float(row["PM2.5 R2"]), float(row["PM2.5 RMSE"]), path,
                notes=f"overlap_any={row['Test seq overlap >=1 frame']}; overlap_6of7={row['Test seq overlap >=6/7 frames']}")


def collect_estimation_baselines(rows):
    folder = EXP / "reports/estimation_baselines"
    files = {"random_70_15_15": folder / "estimation_baselines_split_random_test.csv",
        "random_twofold": folder / "estimation_baselines_split_twofold_fold1_test.csv",
        "time_balanced_purged": folder / "estimation_baselines_split_time_balanced_purged_test.csv",
        "chronological_date": folder / "estimation_baselines_split_chrono_date_test.csv"}
    for protocol, path in files.items():
        if not path.exists(): continue
        table = pd.read_csv(path)
        table = table[table.target.astype(str).str.lower().isin(["pm2.5", "target_pm2.5"])]
        for _, row in table.iterrows():
            add(rows, protocol, str(row.model), number(row, "R2"), number(row, "RMSE"), path,
                number(row, "MAE"), str(row.get("eval_name", "test")))


def collect_classical(rows):
    for path in (EXP / "reports/embedding_classical_baselines").glob("*/metrics_embedding_classical.json"):
        data = json.loads(path.read_text()); test = data.get("test", {}).get("PM2.5", {})
        if not test or data.get("split", {}).get("split_col") != "split_time_balanced_purged": continue
        method = f"pooled ResNet50 + {data.get('model')}" + (" + tabular" if data.get("include_tabular") else "")
        add(rows, "time_balanced_purged", method, number(test, "R2"), number(test, "RMSE"), path, number(test, "MAE"))


def collect_outer_cv(rows, roots):
    for root in roots:
        for path in root.glob("fold_*/**/metrics.json"):
            test = json.loads(path.read_text()).get("test", {}).get("target_PM2.5", {})
            if test:
                add(rows, "date_grouped_outer_cv", f"ResNet50-{path.parent.name.upper()}",
                    number(test, "R2"), number(test, "RMSE"), path, number(test, "MAE"), path.parts[-3])


def write_markdown(table, path):
    lines = ["# TRAQID benchmark matrix", "", "Random sequence results are diagnostic replications, not generalization estimates. The primary evidence is date-grouped outer CV; chronological date-wise evaluation is the hardest stress test.", ""]
    if not table.empty:
        columns = ["protocol", "evidence_status", "method", "fold", "R2", "RMSE", "MAE"]
        lines += ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for _, row in table[columns].iterrows():
            values = []
            for column in columns:
                value = row[column]
                values.append(f"{value:.4f}" if column in {"R2", "RMSE", "MAE"} and pd.notna(value) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
        random = table[table.protocol == "random_twofold"]
        persistence = random[random.method == "previous_step_persistence"]
        fusion = random[random.method.str.contains("LSTM", case=False)]
        if not persistence.empty and not fusion.empty:
            lines += ["## Interpretation", "", f"In random two-fold CV, previous-step persistence reaches R2={persistence.R2.iloc[0]:.4f}; the historical CNN-LSTM fusion reaches R2={fusion.R2.iloc[0]:.4f}. This indicates that temporal adjacency and overlapping windows dominate the random-split score.", ""]
    lines += ["## Decision rule", "", "Select models using mean, dispersion, and worst-fold error across date-grouped outer folds. Use random splits only to reproduce earlier experiments and debug optimization.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-run", action="append", default=["artifacts/runs/traqid_resnet50_rnn_residual_outercv_v1"])
    parser.add_argument("--output-dir", default="experiments/traqid_pretraining_v1/reports/benchmark_matrix")
    args = parser.parse_args(); rows = []
    collect_ladder(rows); collect_estimation_baselines(rows); collect_classical(rows)
    collect_outer_cv(rows, [ROOT / value for value in args.outer_run])
    table = pd.DataFrame(rows).drop_duplicates(subset=["protocol", "method", "fold", "source"])
    if not table.empty: table = table.sort_values(["evidence_status", "protocol", "method", "fold"])
    output = ROOT / args.output_dir; output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "traqid_benchmark_matrix.csv", index=False)
    pd.DataFrame([{"protocol": k, "evidence_status": v[0], "interpretation": v[1]} for k, v in PROTOCOLS.items()]).to_csv(output / "protocol_registry.csv", index=False)
    write_markdown(table, output / "README.md"); print(f"Wrote {len(table)} benchmark rows to {output}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
