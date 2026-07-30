"""Run a robust leave-one-date-out image model from an existing ladder.

This preserves the source ladder's embeddings, folds, and sequence membership
while training new models with a log target, Smooth-L1 loss, inverse-frequency
training-date sampling, and train-derived extreme-value reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ladder-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", default="concat_1_2_6")
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="log1p")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--extreme-iqr-multiplier", type=float, default=3.0)
    parser.add_argument("--balance-by-date", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def embedding_path(source: Path, view: str) -> Path:
    mapping = {
        "mean_1_2_6": source / "embeddings/lenses_1_2_6_mean.npy",
        "concat_1_2_6": source / "embeddings/lenses_1_2_6_concat.npy",
    }
    if view.startswith("lens"):
        mapping[view] = source / f"embeddings/{view}_resnet50.npy"
    if view not in mapping:
        raise ValueError(f"Unsupported view {view!r}")
    path = mapping[view]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    args = parse_args()
    source = Path(args.source_ladder_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embeddings = embedding_path(source, args.view)
    source_results = pd.read_csv(source / "model_ladder_results.csv")
    selected = source_results[
        source_results["view"].eq(args.view)
        & source_results["sequence_length"].eq(args.sequence_length)
    ].sort_values("fold").drop_duplicates("fold")
    if selected.empty:
        raise ValueError(
            f"Source ladder has no reusable {args.view}/T{args.sequence_length} sequences"
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    rows = []
    for record in selected.itertuples(index=False):
        fold = int(record.fold)
        source_parent = source / "models" / f"fold_{fold}" / args.view / f"T{args.sequence_length}"
        source_sequences = source_parent / "sequences.csv"
        destination_parent = output / "models" / f"fold_{fold}" / args.view / f"T{args.sequence_length}"
        destination_sequences = destination_parent / "sequences.csv"
        model_dir = destination_parent / args.cell
        destination_parent.mkdir(parents=True, exist_ok=True)
        if not destination_sequences.is_file() or not args.resume:
            shutil.copy2(source_sequences, destination_sequences)
        metrics_path = model_dir / "metrics.json"
        if not (args.resume and metrics_path.is_file()):
            command = [
                sys.executable,
                str(ROOT / "pipelines/image_embeddings/train_rnn.py"),
                "--sequence-manifest", str(destination_sequences),
                "--embeddings", str(embeddings),
                "--split-col", "split_outer_day",
                "--target-cols", args.target,
                "--cell", args.cell,
                "--target-transform", args.target_transform,
                "--huber-beta", str(args.huber_beta),
                "--extreme-iqr-multiplier", str(args.extreme_iqr_multiplier),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--seed", str(args.seed),
                "--device", args.device,
                *(["--balance-by-date"] if args.balance_by_date else []),
                "--output-dir", str(model_dir),
            ]
            print(" ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
        report = json.loads(metrics_path.read_text())
        robustness = json.loads((model_dir / "metrics_robustness.json").read_text())
        target_robustness = robustness["targets"][args.target]
        row = {
            "fold": fold,
            "test_date": str(record.test_date),
            "view": args.view,
            "sequence_length": args.sequence_length,
            "cell": args.cell,
        }
        for split in ("val", "test"):
            for metric, value in report[split][args.target].items():
                row[f"{split}_{metric.lower()}"] = value
            split_robustness = target_robustness[split]
            row[f"{split}_extreme_threshold"] = target_robustness["upper_extreme_threshold"]
            row[f"{split}_extreme_rows"] = split_robustness["rows_extreme"]
            inlier_metrics = split_robustness["metrics_inlier"] or {}
            for metric, value in inlier_metrics.items():
                row[f"{split}_inlier_{metric.lower()}"] = value
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "model_ladder_results.csv", index=False)

    results = pd.DataFrame(rows)
    aggregate = (
        results.groupby(["view", "sequence_length", "cell"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            test_mae_mean=("test_mae", "mean"),
            test_rmse_mean=("test_rmse", "mean"),
            test_r2_mean=("test_r2", "mean"),
            test_rmse_worst=("test_rmse", "max"),
            test_inlier_mae_mean=("test_inlier_mae", "mean"),
            test_inlier_rmse_mean=("test_inlier_rmse", "mean"),
            test_extreme_rows=("test_extreme_rows", "sum"),
        )
    )
    aggregate.to_csv(output / "model_ladder_aggregate.csv", index=False)
    run = {
        "protocol": "leave_one_complete_date_out",
        "source_ladder_dir": str(source),
        "source_embeddings": str(embeddings),
        "view": args.view,
        "sequence_length": args.sequence_length,
        "cell": args.cell,
        "target": args.target,
        "target_transform": args.target_transform,
        "loss": "SmoothL1Loss",
        "huber_beta_standardized_target_units": args.huber_beta,
        "balance_by_training_date": args.balance_by_date,
        "extreme_threshold": f"training Q3 + {args.extreme_iqr_multiplier} * training IQR",
        "test_extremes_removed_from_primary_metrics": False,
        "reportable_as_unseen_date_generalization": True,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2), flush=True)
    print(aggregate.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
