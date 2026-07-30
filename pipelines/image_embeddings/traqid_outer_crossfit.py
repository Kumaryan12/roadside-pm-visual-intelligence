"""TRAQID date-grouped outer cross-fit for ResNet50-RNN plus residual correction."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from roadside_pm.validation.outer_folds import make_grouped_outer_folds


ROOT = Path(__file__).resolve().parents[2]


def canonicalize_sequences(source: pd.DataFrame) -> pd.DataFrame:
    required = ["sequence_id", "date", "seq_row_ids", "target_row_id", "target_PM2.5"]
    missing = [column for column in required if column not in source]
    if missing: raise ValueError(f"TRAQID sequence manifest missing: {missing}")
    result = source.copy()
    result["embedding_rows"] = result["seq_row_ids"]
    result["target_sample_id"] = result["target_row_id"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-manifest", default="experiments/traqid_pretraining_v1/data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest_time_balanced_purged.csv")
    parser.add_argument("--embeddings", default="experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/traqid_paper_resnet50_front_rear_concat_gap_features.npy")
    parser.add_argument("--engineered-table", default="experiments/traqid_pretraining_v1/data/processed/traqid_engineered_T7_sequence_table.csv")
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru"); parser.add_argument("--outer-folds", type=int, default=5); parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--patience", type=int, default=15); parser.add_argument("--device", default="auto"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residual-model", choices=["ridge", "extra_trees", "hist_gradient_boosting"], default="extra_trees"); parser.add_argument("--output-dir", default="artifacts/runs/traqid_resnet50_rnn_residual_outercv_v1")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--skip-residual", action="store_true")
    parser.add_argument("--exploratory-residual", action="store_true", help="Run non-nested residual CV for diagnostics only; its metric is not valid for model selection.")
    args = parser.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    canonical = canonicalize_sequences(pd.read_csv(args.sequence_manifest))
    folds = make_grouped_outer_folds(canonical, group_column="date", n_splits=args.outer_folds, seed=args.seed)
    if args.max_folds: folds = folds[:args.max_folds]
    prediction_files = []; summaries = []
    for fold_number, fold_frame in enumerate(folds, 1):
        fold_dir = output / f"fold_{fold_number}"; fold_dir.mkdir(parents=True, exist_ok=True); manifest = fold_dir / "sequence_manifest.csv"; fold_frame.to_csv(manifest, index=False)
        model_dir = fold_dir / args.cell
        command = [sys.executable, str(ROOT / "pipelines/image_embeddings/train_rnn.py"), "--sequence-manifest", str(manifest), "--embeddings", args.embeddings, "--split-col", "outer_split", "--target-cols", "target_PM2.5", "--cell", args.cell, "--epochs", str(args.epochs), "--patience", str(args.patience), "--seed", str(args.seed + fold_number), "--device", args.device, "--output-dir", str(model_dir)]
        print(" ".join(command))
        if not args.dry_run:
            result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if result.returncode: return result.returncode
            predictions = pd.read_csv(model_dir / "predictions_test.csv"); predictions["prediction_origin"] = "oof"; predictions["outer_fold"] = fold_number
            path = fold_dir / "oof_predictions.csv"; predictions.to_csv(path, index=False); prediction_files.append(path)
        summaries.append({"fold": fold_number, "train": int((fold_frame.outer_split == "train").sum()), "val": int((fold_frame.outer_split == "val").sum()), "test": int((fold_frame.outer_split == "test").sum()), "train_dates": int(fold_frame.loc[fold_frame.outer_split == "train", "date"].nunique()), "val_dates": int(fold_frame.loc[fold_frame.outer_split == "val", "date"].nunique()), "test_dates": int(fold_frame.loc[fold_frame.outer_split == "test", "date"].nunique())})
    pd.DataFrame(summaries).to_csv(output / "outer_fold_summary.csv", index=False)
    if args.dry_run: return 0
    combined = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    if combined.sequence_id.duplicated().any(): raise ValueError("Outer folds produced duplicate OOF sequences")
    combined_path = output / "image_base_oof_predictions.csv"; combined.to_csv(combined_path, index=False)
    canonical_path = output / "canonical_sequence_manifest.csv"; canonical.to_csv(canonical_path, index=False)
    run_residual = args.exploratory_residual and not args.skip_residual
    if run_residual:
        residual_command = [sys.executable, str(ROOT / "pipelines/image_embeddings/residual_correct.py"), "--base-predictions", str(combined_path), "--sequence-manifest", str(canonical_path), "--feature-table", args.engineered_table, "--target", "target_PM2.5", "--sample-key", "target_row_id", "--group-col", "date", "--model", args.residual_model, "--folds", str(args.outer_folds), "--seed", str(args.seed), "--exploratory-non-nested", "--output-dir", str(output / "residual_correction")]
        print(" ".join(residual_command)); result = subprocess.run(residual_command, cwd=ROOT, env=environment, check=False)
        if result.returncode: return result.returncode
    (output / "run_summary.json").write_text(json.dumps({"dataset": "TRAQID", "protocol": "date_grouped_outer_crossfit", "cell": args.cell, "outer_folds": len(folds), "base_predictions": str(combined_path), "residual_model": args.residual_model if run_residual else None, "residual_status": "exploratory_non_nested" if run_residual else "not_run_requires_nested_implementation"}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
