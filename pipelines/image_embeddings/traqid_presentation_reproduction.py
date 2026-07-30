"""Reproduce the historical TRAQID architecture reported in the Keynote deck.

This is intentionally a compatibility wrapper around the original experiment.
Its random T=7 sliding-window split is overlap-contaminated, so the result is a
reproduction benchmark and must not be presented as deployment generalization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = ROOT / "experiments/traqid_pretraining_v1/scripts/10_pm25_oof_residual_correction.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                        help="Compatibility option; the historical script auto-selects MPS, CUDA, then CPU.")
    parser.add_argument("--output-dir", default="artifacts/runs/traqid_presentation_reproduction_seed42")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    inputs = {
        "sequence_table": ROOT / "experiments/traqid_pretraining_v1/data/processed/traqid_engineered_T7_sequence_table.csv",
        "embedding_npy": ROOT / "experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/traqid_paper_resnet50_front_rear_concat_gap_features.npy",
        "manifest": ROOT / "experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
        "yolo_features": ROOT / "experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_advanced_features_front.csv",
        "road_features": ROOT / "experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_road_features_front.csv",
    }
    missing = [f"{name}: {path}" for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing presentation-reproduction inputs:\n" + "\n".join(missing))

    output = ROOT / args.output_dir
    command = [
        sys.executable, str(LEGACY_SCRIPT),
        "--sequence-table", str(inputs["sequence_table"]),
        "--embedding-npy", str(inputs["embedding_npy"]),
        "--manifest", str(inputs["manifest"]),
        "--yolo-features", str(inputs["yolo_features"]),
        "--road-features", str(inputs["road_features"]),
        "--out-dir", str(output),
        "--split-mode", "random", "--train-frac", "0.8", "--seed", "42",
        "--folds", "3", "--batch-size", "64", "--epochs", "60", "--patience", "10",
        "--lr", "0.0001", "--weight-decay", "0.0001", "--projection-dim", "512",
        "--hidden-dim", "256", "--num-layers", "2", "--dropout", "0.25",
    ]
    print(" ".join(command))
    if args.dry_run:
        return 0

    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "historical_reproduction_overlap_contaminated",
        "source_presentation": "FINAL_TRAQID_Temporal_PM25_Residual_Fusion_Presentation.key",
        "expected_metrics": {"R2": 0.965556, "RMSE": 8.716917, "MAE": 4.211589},
        "architecture": {
            "T": 7, "embedding_dim": 4096, "projection_dim": 512,
            "cell": "gru", "hidden_dim": 256, "num_layers": 2, "dropout": 0.25,
            "oof_folds": 3, "residual_model": "extra_trees", "residual_clip": 12.5,
        },
        "command": command,
        "warning": "Random overlapping sliding windows; reproduce only, do not use for generalization claims.",
    }
    (output / "reproduction_manifest.json").write_text(json.dumps(provenance, indent=2))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
