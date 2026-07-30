"""Audit MUMMA sensor completeness, physical ordering, and extreme values."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="sPM2")
    args = parser.parse_args()

    frame = pd.read_csv(args.sensor_csv)
    required = {"sample_id", "date", "run_id", "sample_timestamp", args.target}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Sensor table missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values")

    target = pd.to_numeric(frame[args.target], errors="coerce")
    q1, q3 = target.quantile([0.25, 0.75])
    upper_extreme = float(q3 + 3.0 * (q3 - q1))
    flags = pd.DataFrame({
        "sample_id": frame["sample_id"],
        "date": frame["date"],
        "run_id": frame["run_id"],
        "sample_timestamp": frame["sample_timestamp"],
        args.target: target,
        "target_missing": target.isna(),
        "target_negative": target.lt(0),
        "target_extreme_3iqr": target.gt(upper_extreme),
    })

    mass = [column for column in ["sPM1", "sPM2", "sPM4", "sPM10"] if column in frame]
    number = [column for column in ["sNPMp5", "sNPM1", "sNPM2", "sNPM4", "sNPM10"] if column in frame]
    mass_violations = {}
    number_violations = {}
    for left, right in zip(mass, mass[1:]):
        key = f"{left}_gt_{right}"
        flags[key] = pd.to_numeric(frame[left], errors="coerce").gt(
            pd.to_numeric(frame[right], errors="coerce")
        )
        mass_violations[key] = int(flags[key].sum())
    for left, right in zip(number, number[1:]):
        key = f"{left}_gt_{right}"
        flags[key] = pd.to_numeric(frame[left], errors="coerce").gt(
            pd.to_numeric(frame[right], errors="coerce")
        )
        number_violations[key] = int(flags[key].sum())

    duplicate_pairs = {}
    for left, right in combinations(number, 2):
        duplicate_pairs[f"{left}_eq_{right}"] = int(
            pd.to_numeric(frame[left], errors="coerce").eq(
                pd.to_numeric(frame[right], errors="coerce")
            ).sum()
        )

    flag_columns = [column for column in flags if column not in {"sample_id", "date", "run_id", "sample_timestamp", args.target}]
    flagged = flags[flags[flag_columns].any(axis=1)].copy()
    by_date = (
        frame.assign(_target=target)
        .groupby("date")
        .agg(rows=("sample_id", "size"), runs=("run_id", "nunique"), target_mean=("_target", "mean"),
             target_std=("_target", "std"), target_min=("_target", "min"), target_max=("_target", "max"))
        .reset_index()
    )
    summary = {
        "sensor_csv": args.sensor_csv,
        "rows": int(len(frame)),
        "dates": sorted(frame["date"].astype(str).unique().tolist()),
        "runs": int(frame["run_id"].nunique()),
        "target": args.target,
        "target_missing": int(target.isna().sum()),
        "target_negative": int(target.lt(0).sum()),
        "target_extreme_threshold_3iqr": upper_extreme,
        "target_extreme_rows": int(target.gt(upper_extreme).sum()),
        "mass_cumulative_order_violations": mass_violations,
        "number_cumulative_order_violations": number_violations,
        "number_bin_equalities": duplicate_pairs,
        "raw_differential_opc_ready": not any(number_violations.values()),
        "warning": "Cumulative number-bin ordering violations must be resolved before differential OPC or chemical source-attribution claims.",
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sensor_quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    by_date.to_csv(output / "sensor_quality_by_date.csv", index=False)
    flagged.to_csv(output / "sensor_quality_flagged_rows.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
