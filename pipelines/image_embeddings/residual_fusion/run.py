"""Combine safe base predictions and run grouped residual correction."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prediction-files", nargs="+", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--sample-key", default="sample_index")
    parser.add_argument("--group-col", required=True)
    parser.add_argument("--feature-cols", nargs="*", default=None)
    parser.add_argument("--model", choices=["ridge", "extra_trees", "hist_gradient_boosting"], default="extra_trees")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    safe_predictions = output / "safe_base_predictions.csv"
    commands = [
        [sys.executable, str(ROOT / "pipelines/image_embeddings/combine_base_predictions.py"), "--inputs", *args.base_prediction_files, "--output-csv", str(safe_predictions)],
        [sys.executable, str(ROOT / "pipelines/image_embeddings/residual_correct.py"), "--base-predictions", str(safe_predictions), "--sequence-manifest", args.sequence_manifest, "--feature-table", args.feature_table, "--target", args.target, "--sample-key", args.sample_key, "--group-col", args.group_col, "--model", args.model, "--folds", str(args.folds), "--output-dir", str(output / args.model), *(["--feature-cols", *args.feature_cols] if args.feature_cols else [])],
    ]
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
