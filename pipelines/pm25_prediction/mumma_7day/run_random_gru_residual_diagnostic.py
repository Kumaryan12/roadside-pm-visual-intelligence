"""Run an explicitly leakage-contaminated random sequence residual diagnostic.

Sequences are constructed first and then randomly assigned to train/validation/
test. Overlapping windows can therefore share raw frames across splits. This is
useful only for comparison with historical random-split scores and must never be
reported as unseen-date generalization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipelines.pm25_prediction.mumma_7day.model_current_data import estimators
from pipelines.pm25_prediction.mumma_7day.run_residual_fusion_current_data import metric_row
from roadside_pm.features.images.sequences import build_grouped_sequences


ROOT = Path(__file__).resolve().parents[3]
LOCAL_TARGET_SUFFIX = "_local_increment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-table", required=True)
    parser.add_argument("--embedding-index", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--tabular-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--residual-feature-set", default="visual_yolo_road")
    parser.add_argument(
        "--background-csv",
        help="Optional external background table; trains on target minus background.",
    )
    parser.add_argument("--background-col", default="background_cams_pm25_ug_m3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_random_sequences(
    args: argparse.Namespace,
    output: Path,
    *,
    model_target: str,
    background: pd.DataFrame | None,
) -> pd.DataFrame:
    sequence_path = output / "sequences_random.csv"
    if args.resume and sequence_path.is_file():
        return pd.read_csv(sequence_path)

    samples = pd.read_csv(args.sample_table)
    samples["sample_id"] = samples["sample_id"].astype(str)
    if background is not None:
        samples = samples.merge(
            background[["sample_id", args.background_col]],
            on="sample_id", how="left", validate="one_to_one",
        )
        if samples[args.background_col].isna().any():
            raise ValueError("Sample table has rows without external background")
        samples[model_target] = samples[args.target] - samples[args.background_col]
    index = pd.read_csv(args.embedding_index)
    index["sample_id"] = index["sample_id"].astype(str)
    index = index[index["embedding_status"] == "success"].copy()
    samples["sequence_pool"] = "all"
    merged = samples.merge(
        index[["sample_id", "embedding_row"]],
        on="sample_id", how="inner", validate="one_to_one",
    )
    sequences = build_grouped_sequences(
        merged,
        sequence_length=args.sequence_length,
        id_column="sample_id",
        embedding_row_column="embedding_row",
        timestamp_column="sample_timestamp",
        group_columns=["run_id"],
        split_column="sequence_pool",
        target_columns=[model_target],
    )

    train_val_ids, test_ids = train_test_split(
        sequences.index.to_numpy(), test_size=0.20, random_state=args.seed, shuffle=True,
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.25, random_state=args.seed, shuffle=True,
    )
    sequences["random_split"] = ""
    sequences.loc[train_ids, "random_split"] = "train"
    sequences.loc[val_ids, "random_split"] = "val"
    sequences.loc[test_ids, "random_split"] = "test"
    sequences.to_csv(sequence_path, index=False)
    return sequences


def overlap_audit(sequences: pd.DataFrame) -> dict[str, object]:
    exploded = sequences[["random_split", "sample_ids"]].copy()
    exploded["sample_id"] = exploded["sample_ids"].str.split("|")
    exploded = exploded.explode("sample_id")
    memberships = exploded.groupby("sample_id")["random_split"].nunique()
    split_sets = {
        split: set(group["sample_id"].astype(str))
        for split, group in exploded.groupby("random_split")
    }
    train_context = split_sets["train"]
    test_targets = set(
        sequences.loc[sequences["random_split"].eq("test"), "target_sample_id"].astype(str)
    )
    leaked_test_targets = test_targets & train_context
    return {
        "raw_samples_in_multiple_splits": int((memberships > 1).sum()),
        "unique_raw_samples": int(len(memberships)),
        "train_val_raw_overlap": int(len(split_sets["train"] & split_sets["val"])),
        "train_test_raw_overlap": int(len(split_sets["train"] & split_sets["test"])),
        "val_test_raw_overlap": int(len(split_sets["val"] & split_sets["test"])),
        "test_target_frames_seen_as_train_context": int(len(leaked_test_targets)),
        "test_target_frame_context_leakage_fraction": float(
            len(leaked_test_targets) / max(len(test_targets), 1)
        ),
        "leakage_contaminated": True,
    }


def attach_predictions(
    prediction_path: Path,
    sequences: pd.DataFrame,
    table: pd.DataFrame,
) -> pd.DataFrame:
    prediction = pd.read_csv(prediction_path)
    result = prediction.merge(
        sequences[["sequence_id", "target_sample_id"]],
        on="sequence_id", validate="one_to_one",
    )
    return result.merge(
        table, left_on="target_sample_id", right_on="sample_id",
        how="left", validate="one_to_one",
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    model_dir = output / f"{args.cell}_T{args.sequence_length}"
    output.mkdir(parents=True, exist_ok=True)
    background = None
    model_target = args.target
    if args.background_csv:
        background = pd.read_csv(args.background_csv)
        background["sample_id"] = background["sample_id"].astype(str)
        required = {"sample_id", args.background_col, "background_status"}
        missing = sorted(required - set(background.columns))
        if missing:
            raise ValueError(f"Background table is missing columns: {missing}")
        if background["sample_id"].duplicated().any():
            raise ValueError("Background sample_id must be unique")
        if not background["background_status"].eq("success").all():
            raise ValueError("Background table contains unsuccessful rows")
        model_target = f"{args.target}{LOCAL_TARGET_SUFFIX}"
    sequences = build_random_sequences(
        args, output, model_target=model_target, background=background,
    )
    audit = overlap_audit(sequences)
    (output / "overlap_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    metrics_path = model_dir / "metrics.json"
    if not (args.resume and metrics_path.is_file()):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        command = [
            sys.executable,
            str(ROOT / "pipelines/image_embeddings/train_rnn.py"),
            "--sequence-manifest", str(output / "sequences_random.csv"),
            "--embeddings", args.embeddings,
            "--split-col", "random_split",
            "--target-cols", model_target,
            "--cell", args.cell,
            "--seed", str(args.seed),
            "--device", args.device,
            "--output-dir", str(model_dir),
        ]
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)

    tabular_dir = Path(args.tabular_run_dir)
    table = pd.read_csv(tabular_dir / "modeling_table.csv")
    table["sample_id"] = table["sample_id"].astype(str)
    if background is not None:
        table = table.merge(
            background[["sample_id", args.background_col]],
            on="sample_id", how="left", validate="one_to_one",
        )
        if table[args.background_col].isna().any():
            raise ValueError("Tabular table has rows without external background")
    groups = json.loads((tabular_dir / "feature_groups.json").read_text())
    group_name = args.residual_feature_set
    if group_name not in groups:
        raise ValueError(f"Tabular run does not contain {group_name!r}")
    columns = list(groups[group_name]["columns"])

    validation = attach_predictions(model_dir / "predictions_val.csv", sequences, table)
    test = attach_predictions(model_dir / "predictions_test.csv", sequences, table)
    actual = f"actual_{model_target}"
    predicted = f"predicted_{model_target}"
    validation_residual = validation[actual].to_numpy() - validation[predicted].to_numpy()
    test_background = test[args.background_col].to_numpy() if background is not None else 0.0
    test_truth = test[actual].to_numpy() + test_background
    base_prediction = test[predicted].to_numpy() + test_background

    rows = [{
        "feature_set": "none",
        "correction_model": "image_base",
        "n_correction_train": len(validation),
        "n_test": len(test),
        **metric_row(test_truth, base_prediction),
    }]
    prediction_tables = []
    for model_name, model in estimators(args.seed).items():
        model.fit(validation[columns], validation_residual)
        corrected = base_prediction + model.predict(test[columns])
        rows.append({
            "feature_set": group_name,
            "correction_model": model_name,
            "n_correction_train": len(validation),
            "n_test": len(test),
            **metric_row(test_truth, corrected),
        })
        prediction_tables.append(pd.DataFrame({
            "sequence_id": test["sequence_id"],
            "target_sample_id": test["target_sample_id"],
            "actual": test_truth,
            "base_prediction": base_prediction,
            "corrected_prediction": corrected,
            "feature_set": group_name,
            "correction_model": model_name,
        }))

    metrics = pd.DataFrame(rows).sort_values("rmse")
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    run = {
        "protocol": "random_sequence_split_after_window_construction",
        "reportable_as_generalization": False,
        "warning": "Overlapping temporal windows share raw frames across splits.",
        "base": {"cell": args.cell, "sequence_length": args.sequence_length, "view": "provided_embeddings"},
        "target_decomposition": (
            f"{args.target} = external {args.background_col} + {model_target}"
            if background is not None else f"direct prediction of {args.target}"
        ),
        "background_target_fitted": False if background is not None else None,
        "background_csv": args.background_csv,
        "residual_feature_set": group_name,
        "seed": args.seed,
        "overlap_audit": audit,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
