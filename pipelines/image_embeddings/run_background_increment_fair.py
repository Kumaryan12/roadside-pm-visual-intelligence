"""Train a fair temporal image model on PM2.5 minus external background.

The runner reuses precomputed image embeddings and whole-date outer splits.
For every sequence target it predicts the local increment, then adds the fixed
external background back before reporting total-PM2.5 metrics. Its output
layout is compatible with the residual-fusion runner.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
LOCAL_TARGET = "sPM2_local_increment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ladder", required=True)
    parser.add_argument("--background-csv", required=True)
    parser.add_argument("--embedding-index", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", default="lens6")
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--background-col", default="background_cams_pm25_ug_m3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(command: list[str], environment: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def metric_values(truth, prediction) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(truth, prediction)),
        "RMSE": float(mean_squared_error(truth, prediction) ** 0.5),
        "R2": float(r2_score(truth, prediction)),
    }


def convert_predictions(
    model_dir: Path,
    sequence_csv: Path,
    background: pd.DataFrame,
    *,
    split: str,
    target: str,
    background_col: str,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    standard_path = model_dir / f"predictions_{split}.csv"
    local_path = model_dir / f"predictions_{split}_local_increment.csv"
    if not local_path.is_file():
        shutil.copy2(standard_path, local_path)
    local = pd.read_csv(local_path)
    sequences = pd.read_csv(
        sequence_csv, usecols=["sequence_id", "target_sample_id"]
    )
    result = local.merge(sequences, on="sequence_id", validate="one_to_one")
    result = result.merge(
        background[["sample_id", background_col]],
        left_on="target_sample_id", right_on="sample_id",
        how="left", validate="one_to_one",
    )
    if result[background_col].isna().any():
        raise ValueError(f"Missing background for {int(result[background_col].isna().sum())} targets")
    actual_local = result[f"actual_{LOCAL_TARGET}"].to_numpy()
    predicted_local = result[f"predicted_{LOCAL_TARGET}"].to_numpy()
    background_values = result[background_col].to_numpy()
    actual_total = actual_local + background_values
    predicted_total = predicted_local + background_values
    total = pd.DataFrame({
        "sequence_id": result["sequence_id"],
        "prediction_origin": result["prediction_origin"],
        f"actual_{target}": actual_total,
        f"predicted_{target}": predicted_total,
        "target_sample_id": result["target_sample_id"],
        background_col: background_values,
        f"actual_{LOCAL_TARGET}": actual_local,
        f"predicted_{LOCAL_TARGET}": predicted_local,
    })
    detailed_path = model_dir / f"predictions_{split}_decomposition.csv"
    total.to_csv(detailed_path, index=False)
    # Write the exact conventional schema so downstream residual fusion can
    # consume this model exactly like a total-PM image ladder.
    total[[
        "sequence_id", "prediction_origin", f"actual_{target}", f"predicted_{target}",
    ]].to_csv(standard_path, index=False)
    return (
        metric_values(actual_total, predicted_total),
        metric_values(actual_local, predicted_local),
        total,
    )


def main() -> int:
    args = parse_args()
    source = Path(args.source_ladder)
    output = Path(args.output_dir)
    background = pd.read_csv(args.background_csv)
    background["sample_id"] = background["sample_id"].astype(str)
    required_background = {"sample_id", args.background_col, "background_status"}
    missing = sorted(required_background - set(background.columns))
    if missing:
        raise ValueError(f"Background table is missing columns: {missing}")
    if background["sample_id"].duplicated().any():
        raise ValueError("Background sample_id must be unique")
    if not background["background_status"].eq("success").all():
        raise ValueError("Background table contains unsuccessful rows")

    split_summary = json.loads((source / "splits/summary.json").read_text())
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    python = sys.executable
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    for fold_info in split_summary["folds"]:
        fold = int(fold_info["fold"])
        fold_root = output / "models" / f"fold_{fold}" / args.view / f"T{args.sequence_length}"
        model_dir = fold_root / args.cell
        adjusted_manifest = output / "splits" / f"fold_{fold}_test_{fold_info['test_date']}.csv"
        sequence_csv = fold_root / "sequences.csv"
        total_metrics_path = model_dir / "metrics_total.json"
        if not (args.resume and total_metrics_path.is_file()):
            split = pd.read_csv(fold_info["output_csv"])
            split["sample_id"] = split["sample_id"].astype(str)
            split = split.merge(
                background[["sample_id", args.background_col]],
                on="sample_id", how="left", validate="one_to_one",
            )
            if split[args.background_col].isna().any():
                raise ValueError(
                    f"Fold {fold} has missing {args.background_col} rows"
                )
            split[LOCAL_TARGET] = split[args.target] - split[args.background_col]
            adjusted_manifest.parent.mkdir(parents=True, exist_ok=True)
            split.to_csv(adjusted_manifest, index=False)

            run([
                python, str(ROOT / "pipelines/image_embeddings/build_sequences.py"),
                "--manifest", str(adjusted_manifest),
                "--embedding-index", args.embedding_index,
                "--output-csv", str(sequence_csv),
                "--id-col", "sample_id", "--timestamp-col", "sample_timestamp",
                "--split-col", "split_outer_day", "--group-cols", "run_id",
                "--target-cols", LOCAL_TARGET,
                "--sequence-length", str(args.sequence_length),
            ], environment)
            run([
                python, str(ROOT / "pipelines/image_embeddings/train_rnn.py"),
                "--sequence-manifest", str(sequence_csv), "--embeddings", args.embeddings,
                "--split-col", "split_outer_day", "--target-cols", LOCAL_TARGET,
                "--cell", args.cell, "--seed", str(args.seed), "--device", args.device,
                "--target-transform", "none", "--output-dir", str(model_dir),
            ], environment)
            shutil.copy2(model_dir / "metrics.json", model_dir / "metrics_local_increment.json")
            report_total: dict[str, dict[str, dict[str, float]]] = {}
            report_local: dict[str, dict[str, dict[str, float]]] = {}
            for split_name in ("val", "test"):
                total_metric, local_metric, table = convert_predictions(
                    model_dir, sequence_csv, background, split=split_name,
                    target=args.target, background_col=args.background_col,
                )
                report_total[split_name] = {args.target: total_metric}
                report_local[split_name] = {LOCAL_TARGET: local_metric}
                if split_name == "test":
                    table = table.copy(); table["fold"] = fold
                    table["test_date"] = fold_info["test_date"]
                    predictions.append(table)
            (model_dir / "metrics.json").write_text(json.dumps(report_total, indent=2) + "\n")
            total_metrics_path.write_text(json.dumps({
                "total_pm25": report_total,
                "local_increment": report_local,
                "decomposition": f"{args.target} = {args.background_col} + {LOCAL_TARGET}",
            }, indent=2) + "\n")
        else:
            report_total = json.loads((model_dir / "metrics.json").read_text())
            # Re-materialize conventional prediction files from the preserved
            # local-increment outputs.  This also repairs outputs created by an
            # earlier version that included downstream-incompatible columns.
            for split_name in ("val", "test"):
                _, _, repaired = convert_predictions(
                    model_dir, sequence_csv, background, split=split_name,
                    target=args.target, background_col=args.background_col,
                )
                if split_name == "test":
                    table = repaired
            table["fold"] = fold; table["test_date"] = fold_info["test_date"]
            predictions.append(table)

        row = {
            "fold": fold, "test_date": fold_info["test_date"], "view": args.view,
            "sequence_length": args.sequence_length, "cell": args.cell,
        }
        for split_name in ("val", "test"):
            for metric, value in report_total[split_name][args.target].items():
                row[f"{split_name}_{metric.lower()}"] = value
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "model_ladder_results.csv", index=False)

    results = pd.DataFrame(rows)
    aggregate = (
        results.groupby(["view", "sequence_length", "cell"], as_index=False)
        .agg(
            folds=("fold", "nunique"), test_mae_mean=("test_mae", "mean"),
            test_rmse_mean=("test_rmse", "mean"), test_r2_mean=("test_r2", "mean"),
            test_rmse_worst=("test_rmse", "max"),
        )
    )
    aggregate.to_csv(output / "model_ladder_aggregate.csv", index=False)
    combined = pd.concat(predictions, ignore_index=True)
    combined.to_csv(output / "predictions_test_all_folds.csv", index=False)
    pooled = metric_values(combined[f"actual_{args.target}"], combined[f"predicted_{args.target}"])
    run_summary = {
        "model": {"view": args.view, "sequence_length": args.sequence_length, "cell": args.cell},
        "decomposition": (
            f"{args.target} = external {args.background_col} "
            "+ temporal image local increment"
        ),
        "background_target_fitted": False,
        "background_column": args.background_col,
        "primary_protocol": "leave_one_day_out",
        "outer_test_rows": len(combined),
        "pooled_test_metrics": pooled,
        "source_splits": str(source / "splits"),
        "background_csv": args.background_csv,
        "embedding_index": args.embedding_index,
        "embeddings": args.embeddings,
    }
    (output / "run.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(aggregate.to_string(index=False))
    print(json.dumps({"pooled_test_metrics": pooled}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
