"""Run ResNet50 extraction, leakage-safe sequence building, and GRU/LSTM training."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", required=True); parser.add_argument("--sample-table", required=True); parser.add_argument("--image-col", required=True)
    parser.add_argument("--id-col", default="sample_index"); parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--split-col", required=True); parser.add_argument("--group-cols", nargs="+", required=True); parser.add_argument("--target-cols", nargs="+", required=True)
    parser.add_argument("--sequence-length", type=int, default=7); parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--run-dir", required=True); parser.add_argument("--device", default="auto"); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--filter-col", default=None); parser.add_argument("--filter-value", default=None)
    args = parser.parse_args(); run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    array = run_dir / "resnet50_embeddings.npy"; index = run_dir / "resnet50_embedding_index.csv"; sequences = run_dir / f"sequences_T{args.sequence_length}.csv"; model_dir = run_dir / f"{args.cell}_T{args.sequence_length}"
    commands = [
        ([sys.executable, str(ROOT / "pipelines/image_embeddings/extract_resnet50.py"), "--manifest", args.image_manifest, "--image-col", args.image_col, "--id-col", args.id_col, "--output-array", str(array), "--output-index", str(index), "--device", args.device, *(["--filter-col", args.filter_col, "--filter-value", str(args.filter_value)] if args.filter_col else [])], [array, index]),
        ([sys.executable, str(ROOT / "pipelines/image_embeddings/build_sequences.py"), "--manifest", args.sample_table, "--embedding-index", str(index), "--output-csv", str(sequences), "--id-col", args.id_col, "--timestamp-col", args.timestamp_col, "--split-col", args.split_col, "--group-cols", *args.group_cols, "--target-cols", *args.target_cols, "--sequence-length", str(args.sequence_length)], [sequences]),
        ([sys.executable, str(ROOT / "pipelines/image_embeddings/train_rnn.py"), "--sequence-manifest", str(sequences), "--embeddings", str(array), "--split-col", args.split_col, "--target-cols", *args.target_cols, "--cell", args.cell, "--device", args.device, "--output-dir", str(model_dir)], [model_dir / "metrics.json"]),
    ]
    for command, outputs in commands:
        print(" ".join(command))
        if args.dry_run: continue
        if args.resume and all(path.exists() for path in outputs): print("Skipping existing outputs"); continue
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if result.returncode: return result.returncode
    return 0


if __name__ == "__main__": raise SystemExit(main())
