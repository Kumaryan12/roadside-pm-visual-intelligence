"""Find extreme consecutive PM2.5 changes and attach temporal sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--expected-step-seconds", type=float, default=10.0)
    parser.add_argument("--absolute-threshold", type=float)
    parser.add_argument("--quantile", type=float, default=0.995)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns = ["sample_id", "run_id", "date", "sample_timestamp", args.target, "lat", "long"]
    frame = pd.read_csv(args.modeling_table, usecols=columns)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["sample_timestamp"] = pd.to_datetime(frame["sample_timestamp"], errors="raise")
    frame = frame.sort_values(["run_id", "sample_timestamp"]).reset_index(drop=True)
    grouped = frame.groupby("run_id", sort=False)
    frame["previous_sample_id"] = grouped["sample_id"].shift(1)
    frame["previous_timestamp"] = grouped["sample_timestamp"].shift(1)
    frame["previous_pm25"] = grouped[args.target].shift(1)
    frame["next_timestamp"] = grouped["sample_timestamp"].shift(-1)
    frame["next_pm25"] = grouped[args.target].shift(-1)
    frame["step_seconds"] = (frame["sample_timestamp"] - frame["previous_timestamp"]).dt.total_seconds()
    frame["delta_pm25"] = frame[args.target] - frame["previous_pm25"]
    frame["absolute_delta_pm25"] = frame["delta_pm25"].abs()
    frame["next_delta_pm25"] = frame["next_pm25"] - frame[args.target]
    valid = frame["step_seconds"].eq(args.expected_step_seconds) & frame["delta_pm25"].notna()
    distribution = frame.loc[valid, "absolute_delta_pm25"]
    if args.absolute_threshold is not None:
        threshold = float(args.absolute_threshold)
        threshold_rule = "explicit absolute threshold"
    else:
        if not 0 < args.quantile < 1:
            raise ValueError("quantile must be between zero and one")
        threshold = float(distribution.quantile(args.quantile))
        threshold_rule = f"absolute consecutive change >= empirical quantile {args.quantile}"
    events = frame[valid & frame["absolute_delta_pm25"].ge(threshold)].copy()
    events["direction"] = np.where(events["delta_pm25"] >= 0, "rise", "fall")
    events["immediate_reversal"] = (
        np.sign(events["delta_pm25"]) != np.sign(events["next_delta_pm25"])
    ) & events["next_delta_pm25"].abs().ge(0.5 * events["absolute_delta_pm25"])

    sequences = pd.read_csv(args.sequence_manifest)
    sequences["target_sample_id"] = sequences["target_sample_id"].astype(str)
    sequence_columns = [
        "sequence_id", "sequence_length", "sample_ids", "target_sample_id",
        "target_time", "split_outer_day", "run_id",
    ]
    events = events.merge(
        sequences[sequence_columns], left_on="sample_id", right_on="target_sample_id",
        how="left", suffixes=("", "_sequence"), validate="one_to_one",
    )
    value_lookup = frame.set_index("sample_id")[args.target].to_dict()
    time_lookup = frame.set_index("sample_id")["sample_timestamp"].astype(str).to_dict()
    events["sequence_pm25_values"] = events["sample_ids"].fillna("").map(
        lambda value: "|".join(f"{value_lookup[item]:.6g}" for item in value.split("|") if item)
    )
    events["sequence_timestamps"] = events["sample_ids"].fillna("").map(
        lambda value: "|".join(time_lookup[item] for item in value.split("|") if item)
    )
    events = events.sort_values("absolute_delta_pm25", ascending=False).reset_index(drop=True)
    events.insert(0, "event_rank", np.arange(1, len(events) + 1))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "event_rank", "date", "run_id", "previous_sample_id", "sample_id",
        "previous_timestamp", "sample_timestamp", "next_timestamp",
        "previous_pm25", args.target, "next_pm25", "delta_pm25",
        "absolute_delta_pm25", "next_delta_pm25", "direction", "immediate_reversal",
        "lat", "long", "sequence_id", "sequence_length", "split_outer_day",
        "sample_ids", "sequence_timestamps", "sequence_pm25_values",
    ]
    events[output_columns].to_csv(output / "sudden_pm25_t7_sequences.csv", index=False)
    summary = {
        "modeling_table": args.modeling_table,
        "sequence_manifest": args.sequence_manifest,
        "target": args.target,
        "valid_consecutive_transitions": int(valid.sum()),
        "expected_step_seconds": args.expected_step_seconds,
        "threshold_rule": threshold_rule,
        "absolute_change_threshold_ug_m3": threshold,
        "events": len(events),
        "events_with_t7_sequence": int(events["sequence_id"].notna().sum()),
        "rises": int(events["direction"].eq("rise").sum()),
        "falls": int(events["direction"].eq("fall").sum()),
        "immediate_reversals": int(events["immediate_reversal"].sum()),
        "warning": (
            "Sudden changes are detections, not validated pollution episodes. Immediate "
            "reversals and isolated extremes require sensor and image review."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(events[output_columns[:17]].head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
