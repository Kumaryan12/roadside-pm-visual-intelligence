"""Create a chronological run split for a partial one-day image extraction.

This is an engineering smoke-test split only. It must not replace whole-day
evaluation once multiple complete collection days are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--lens-id", type=int, default=6)
    parser.add_argument("--min-eval-samples", type=int, default=30)
    args = parser.parse_args()

    source = Path(args.preprocessed_manifest)
    output = Path(args.output_csv)
    frame = pd.read_csv(source)
    frame = frame[
        (frame["preprocess_status"] == "success")
        & (frame["lens_id"] == args.lens_id)
    ].copy()
    if frame.empty:
        raise ValueError(f"No successful Lens {args.lens_id} frames")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Expected one selected-lens frame per sample_id")

    frame["sample_timestamp"] = pd.to_datetime(frame["sample_timestamp"], errors="raise")
    run_stats = (
        frame.groupby("run_id")
        .agg(first_timestamp=("sample_timestamp", "min"), samples=("sample_id", "size"))
        .sort_values("first_timestamp")
    )
    eligible_eval = run_stats[run_stats["samples"] >= args.min_eval_samples]
    if len(eligible_eval) < 3:
        raise ValueError(
            "Need at least three runs with enough samples for chronological "
            f"train/val/test; found {len(eligible_eval)}"
        )
    validation_run = eligible_eval.index[-2]
    test_run = eligible_eval.index[-1]
    frame["split_partial_chronological_run"] = "train"
    frame.loc[frame["run_id"] == validation_run, "split_partial_chronological_run"] = "val"
    frame.loc[frame["run_id"] == test_run, "split_partial_chronological_run"] = "test"
    frame = frame.sort_values(["sample_timestamp", "sample_id"]).reset_index(drop=True)
    frame["sample_timestamp"] = frame["sample_timestamp"].astype(str)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "source_manifest": str(source),
        "output_csv": str(output),
        "lens_id": args.lens_id,
        "rows": len(frame),
        "runs": int(frame["run_id"].nunique()),
        "dates": sorted(frame["date"].astype(str).unique()),
        "validation_run": validation_run,
        "test_run": test_run,
        "split_counts": frame["split_partial_chronological_run"].value_counts().to_dict(),
        "status": "engineering_smoke_only_single_day_partial_extraction",
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
