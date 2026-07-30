"""Run the MUMMA leave-one-day-out ResNet50 view/sequence/GRU-LSTM ladder.

The historical filename is retained for command compatibility; the runner now
supports any delivery with at least two complete collection dates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str], *, environment: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 2, 6])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[1, 3, 7, 13, 31])
    parser.add_argument("--cells", nargs="+", choices=["gru", "lstm"], default=["gru", "lstm"])
    parser.add_argument(
        "--views",
        nargs="+",
        default=None,
        help="Optional subset such as concat_1_2_6 or lens6; defaults to every extracted/fused view.",
    )
    parser.add_argument("--target", default="sPM2")
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "mobilenet_v2", "efficientnet_b0", "convnext_tiny"],
        default="resnet50",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="none")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--balance-by-date", action="store_true")
    parser.add_argument("--extreme-iqr-multiplier", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rebuild-sequences",
        action="store_true",
        help="Regenerate sequence manifests while reusing completed embeddings and models.",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    embeddings_dir = output / "embeddings"
    splits_dir = output / "splits"
    runs_dir = output / "models"
    for directory in (embeddings_dir, splits_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    python = sys.executable

    arrays: dict[str, Path] = {}
    indices: dict[str, Path] = {}
    for lens in args.lenses:
        label = f"lens{lens}"
        suffix = "resnet50" if args.backbone == "resnet50" else args.backbone
        arrays[label] = embeddings_dir / f"{label}_{suffix}.npy"
        indices[label] = embeddings_dir / (
            f"{label}_index.csv" if args.backbone == "resnet50" else f"{label}_{suffix}_index.csv"
        )
        if not (args.resume and arrays[label].is_file() and indices[label].is_file()):
            run([
                python, str(ROOT / "pipelines/image_embeddings/extract_torchvision.py"),
                "--manifest", args.preprocessed_manifest,
                "--image-col", "processed_frame_path", "--id-col", "sample_id",
                "--output-array", str(arrays[label]), "--output-index", str(indices[label]),
                "--backbone", args.backbone,
                "--batch-size", str(args.embedding_batch_size),
                "--filter-col", "lens_id", "--filter-value", str(lens),
                "--device", args.device,
            ], environment=environment)

    fusion_name = "lenses_1_2_6" if args.backbone == "resnet50" else f"lenses_1_2_6_{args.backbone}"
    fusion_prefix = embeddings_dir / fusion_name
    fusion_index = fusion_prefix.with_name(fusion_prefix.name + "_index.csv")
    fusion_mean = fusion_prefix.with_name(fusion_prefix.name + "_mean.npy")
    fusion_concat = fusion_prefix.with_name(fusion_prefix.name + "_concat.npy")
    if not (args.resume and fusion_index.is_file() and fusion_mean.is_file() and fusion_concat.is_file()):
        run([
            python, str(ROOT / "pipelines/image_embeddings/combine_lens_embeddings.py"),
            "--arrays", *[str(arrays[f"lens{lens}"]) for lens in args.lenses],
            "--indices", *[str(indices[f"lens{lens}"]) for lens in args.lenses],
            "--labels", *[f"lens{lens}" for lens in args.lenses],
            "--id-col", "sample_id", "--output-prefix", str(fusion_prefix),
        ], environment=environment)
    arrays["mean_1_2_6"] = fusion_mean
    arrays["concat_1_2_6"] = fusion_concat
    indices["mean_1_2_6"] = fusion_index
    indices["concat_1_2_6"] = fusion_index
    if args.views is not None:
        unknown = sorted(set(args.views) - set(arrays))
        if unknown:
            raise ValueError(f"Unknown views {unknown}; available views are {sorted(arrays)}")
        arrays = {view: arrays[view] for view in args.views}
        indices = {view: indices[view] for view in args.views}

    split_summary = splits_dir / "summary.json"
    if not (args.resume and split_summary.is_file()):
        run([
            python,
            str(ROOT / "pipelines/pm25_prediction/mumma_7day/make_two_day_image_outer_splits.py"),
            "--preprocessed-manifest", args.preprocessed_manifest,
            "--output-dir", str(splits_dir), "--lens-id", str(args.lenses[-1]),
        ], environment=environment)
    split_information = json.loads(split_summary.read_text())

    result_rows = []
    for fold_info in split_information["folds"]:
        fold = int(fold_info["fold"])
        split_csv = Path(fold_info["output_csv"])
        for view in arrays:
            for sequence_length in args.sequence_lengths:
                sequence_csv = runs_dir / f"fold_{fold}" / view / f"T{sequence_length}" / "sequences.csv"
                if args.rebuild_sequences or not (args.resume and sequence_csv.is_file()):
                    sequence_csv.parent.mkdir(parents=True, exist_ok=True)
                    run([
                        python, str(ROOT / "pipelines/image_embeddings/build_sequences.py"),
                        "--manifest", str(split_csv), "--embedding-index", str(indices[view]),
                        "--output-csv", str(sequence_csv), "--id-col", "sample_id",
                        "--timestamp-col", "sample_timestamp", "--split-col", "split_outer_day",
                        "--group-cols", "run_id", "--target-cols", args.target,
                        "--sequence-length", str(sequence_length),
                    ], environment=environment)
                for cell in args.cells:
                    model_dir = sequence_csv.parent / cell
                    metrics_path = model_dir / "metrics.json"
                    if not (args.resume and metrics_path.is_file()):
                        run([
                            python, str(ROOT / "pipelines/image_embeddings/train_rnn.py"),
                            "--sequence-manifest", str(sequence_csv), "--embeddings", str(arrays[view]),
                            "--split-col", "split_outer_day", "--target-cols", args.target,
                            "--cell", cell, "--device", args.device,
                            "--target-transform", args.target_transform,
                            "--huber-beta", str(args.huber_beta),
                            "--extreme-iqr-multiplier", str(args.extreme_iqr_multiplier),
                            *(["--balance-by-date"] if args.balance_by_date else []),
                            "--output-dir", str(model_dir),
                        ], environment=environment)
                    report = json.loads(metrics_path.read_text())
                    row = {
                        "fold": fold, "test_date": fold_info["test_date"], "view": view,
                        "sequence_length": sequence_length, "cell": cell,
                    }
                    for split in ("val", "test"):
                        for metric, value in report[split][args.target].items():
                            row[f"{split}_{metric.lower()}"] = value
                    result_rows.append(row)
                    pd.DataFrame(result_rows).to_csv(output / "model_ladder_results.csv", index=False)

    results = pd.DataFrame(result_rows)
    aggregate = (
        results.groupby(["view", "sequence_length", "cell"], as_index=False)
        .agg(
            folds=("fold", "nunique"), test_mae_mean=("test_mae", "mean"),
            test_rmse_mean=("test_rmse", "mean"), test_r2_mean=("test_r2", "mean"),
            test_rmse_worst=("test_rmse", "max"),
        )
        .sort_values("test_rmse_mean")
    )
    aggregate.to_csv(output / "model_ladder_aggregate.csv", index=False)
    (output / "backbone.json").write_text(json.dumps({
        "backbone": args.backbone,
        "pretrained_weights": "ImageNet",
        "views": list(arrays),
    }, indent=2) + "\n")
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
