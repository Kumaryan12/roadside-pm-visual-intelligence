"""Build leave-one-day-out image-model splits for the current delivery.

For each outer fold, one complete date is test-only. Validation consists of
the latest whole run(s) from the remaining dates; sequences must be built after this
assignment and remain within run boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def choose_validation_runs(training: pd.DataFrame, fraction: float) -> list[str]:
    stats = (
        training.groupby("run_id")
        .agg(first=("sample_timestamp", "min"), samples=("sample_id", "size"))
        .sort_values("first")
    )
    if len(stats) < 2:
        raise ValueError("Need at least two training-day runs for train/validation")
    target = max(1, int(round(len(training) * fraction)))
    chosen: list[str] = []
    count = 0
    for run_id, row in stats.iloc[::-1].iterrows():
        if len(chosen) >= len(stats) - 1:
            break
        chosen.append(str(run_id))
        count += int(row["samples"])
        if count >= target:
            break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lens-id", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")

    source = Path(args.preprocessed_manifest)
    output = Path(args.output_dir)
    frame = pd.read_csv(source)
    frame = frame[
        (frame["preprocess_status"] == "success")
        & (frame["lens_id"] == args.lens_id)
    ].copy()
    if frame["sample_id"].duplicated().any():
        raise ValueError("Expected exactly one selected-lens image per sample")
    frame["sample_timestamp"] = pd.to_datetime(frame["sample_timestamp"], errors="raise")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date.astype(str)
    dates = sorted(frame["date"].unique())
    if len(dates) < 2:
        raise ValueError(f"Expected at least two complete dates; found {dates}")

    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fold_number, test_date in enumerate(dates, start=1):
        fold = frame.copy()
        training_dates = [date for date in dates if date != test_date]
        training = fold[fold["date"].isin(training_dates)]
        validation_runs = choose_validation_runs(training, args.validation_fraction)
        fold["split_outer_day"] = "train"
        fold.loc[fold["run_id"].isin(validation_runs), "split_outer_day"] = "val"
        fold.loc[fold["date"] == test_date, "split_outer_day"] = "test"
        fold = fold.sort_values(["sample_timestamp", "sample_id"]).reset_index(drop=True)
        fold["sample_timestamp"] = fold["sample_timestamp"].astype(str)
        fold_path = output / f"fold_{fold_number}_test_{test_date}.csv"
        fold.to_csv(fold_path, index=False)
        summaries.append({
            "fold": fold_number,
            "test_date": test_date,
            "training_dates": training_dates,
            "validation_runs": validation_runs,
            "split_counts": fold["split_outer_day"].value_counts().to_dict(),
            "output_csv": str(fold_path),
        })

    summary = {
        "source_manifest": str(source),
        "lens_id": args.lens_id,
        "dates": dates,
        "rows": len(frame),
        "primary_protocol": "leave_one_day_out_outer_test",
        "folds": summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
