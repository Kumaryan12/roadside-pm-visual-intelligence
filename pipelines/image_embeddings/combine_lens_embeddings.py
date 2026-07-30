"""Align successful per-lens embeddings and write mean/concatenated features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", nargs="+", required=True)
    parser.add_argument("--indices", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--id-col", default="sample_id")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if not (len(args.arrays) == len(args.indices) == len(args.labels)):
        raise ValueError("arrays, indices and labels must have equal lengths")
    if len(set(args.labels)) != len(args.labels):
        raise ValueError("labels must be unique")

    arrays = [np.load(path, mmap_mode="r") for path in args.arrays]
    indices = []
    for label, path, array in zip(args.labels, args.indices, arrays):
        index = pd.read_csv(path)
        index = index[index["embedding_status"] == "success"].copy()
        if index[args.id_col].duplicated().any():
            raise ValueError(f"Duplicate IDs in {label} index")
        if index["embedding_row"].max() >= len(array):
            raise ValueError(f"Embedding row outside {label} array")
        index = index[[args.id_col, "embedding_row"]].rename(
            columns={"embedding_row": f"row_{label}"}
        )
        indices.append(index)

    aligned = indices[0]
    for index in indices[1:]:
        aligned = aligned.merge(index, on=args.id_col, how="inner", validate="one_to_one")
    aligned = aligned.sort_values(args.id_col).reset_index(drop=True)
    if aligned.empty:
        raise ValueError("No IDs are shared by every embedding input")

    matrices = []
    for label, array in zip(args.labels, arrays):
        rows = aligned[f"row_{label}"].to_numpy(dtype=int)
        matrices.append(np.asarray(array[rows], dtype=np.float32))
    dimensions = [matrix.shape[1] for matrix in matrices]
    if len(set(dimensions)) != 1:
        raise ValueError(f"Mean fusion requires equal dimensions; found {dimensions}")

    mean = np.mean(np.stack(matrices, axis=0), axis=0).astype(np.float32)
    concatenated = np.concatenate(matrices, axis=1).astype(np.float32)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    mean_path = prefix.with_name(prefix.name + "_mean.npy")
    concat_path = prefix.with_name(prefix.name + "_concat.npy")
    index_path = prefix.with_name(prefix.name + "_index.csv")
    np.save(mean_path, mean)
    np.save(concat_path, concatenated)
    pd.DataFrame({
        args.id_col: aligned[args.id_col],
        "embedding_row": np.arange(len(aligned)),
        "embedding_status": "success",
    }).to_csv(index_path, index=False)
    summary = {
        "labels": args.labels,
        "rows": len(aligned),
        "input_dimensions": dimensions,
        "mean_shape": list(mean.shape),
        "concat_shape": list(concatenated.shape),
        "mean_path": str(mean_path),
        "concat_path": str(concat_path),
        "index_path": str(index_path),
    }
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
