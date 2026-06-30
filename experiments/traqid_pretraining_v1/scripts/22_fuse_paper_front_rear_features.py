from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--front", required=True)
    parser.add_argument("--rear", required=True)
    parser.add_argument("--out-mean", required=True)
    parser.add_argument("--out-concat", required=True)
    parser.add_argument("--report", required=True)

    args = parser.parse_args()

    front_path = Path(args.front)
    rear_path = Path(args.rear)
    out_mean_path = Path(args.out_mean)
    out_concat_path = Path(args.out_concat)
    report_path = Path(args.report)

    out_mean_path.parent.mkdir(parents=True, exist_ok=True)
    out_concat_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    front = np.load(front_path, mmap_mode="r")
    rear = np.load(rear_path, mmap_mode="r")

    if front.shape != rear.shape:
        raise ValueError(f"Shape mismatch: front={front.shape}, rear={rear.shape}")

    mean = ((front[:] + rear[:]) / 2.0).astype("float32")
    concat = np.concatenate([front[:], rear[:]], axis=1).astype("float32")

    np.save(out_mean_path, mean)
    np.save(out_concat_path, concat)

    summary = {
        "front": str(front_path),
        "rear": str(rear_path),
        "front_shape": list(front.shape),
        "rear_shape": list(rear.shape),
        "mean_shape": list(mean.shape),
        "concat_shape": list(concat.shape),
        "out_mean": str(out_mean_path),
        "out_concat": str(out_concat_path),
    }

    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()