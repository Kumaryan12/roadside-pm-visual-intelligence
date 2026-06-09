from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_memmap(path: Path, shape: tuple, overwrite: bool):
    if path.exists() and overwrite:
        path.unlink()

    if path.exists() and not overwrite:
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape:
            raise ValueError(
                f"Existing file has wrong shape: {path}, "
                f"found={arr.shape}, expected={shape}. Use --overwrite."
            )
        return arr

    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype="float32",
        shape=shape,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--front-emb",
        default="experiments/traqid_pretraining_v1/embeddings/traqid_supervised_pm25_random_front_embeddings.npy",
    )

    parser.add_argument(
        "--rear-emb",
        default="experiments/traqid_pretraining_v1/embeddings/traqid_supervised_pm25_random_rear_embeddings.npy",
    )

    parser.add_argument(
        "--front-index",
        default="experiments/traqid_pretraining_v1/embeddings/traqid_supervised_pm25_random_front_embedding_index.csv",
    )

    parser.add_argument(
        "--rear-index",
        default="experiments/traqid_pretraining_v1/embeddings/traqid_supervised_pm25_random_rear_embedding_index.csv",
    )

    parser.add_argument(
        "--out-prefix",
        default="experiments/traqid_pretraining_v1/embeddings/traqid_supervised_pm25_random_front_rear",
    )

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    front_emb_path = Path(args.front_emb)
    rear_emb_path = Path(args.rear_emb)
    front_index_path = Path(args.front_index)
    rear_index_path = Path(args.rear_index)
    out_prefix = Path(args.out_prefix)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    front = np.load(front_emb_path, mmap_mode="r")
    rear = np.load(rear_emb_path, mmap_mode="r")

    front_idx = pd.read_csv(front_index_path)
    rear_idx = pd.read_csv(rear_index_path)

    if front.shape != rear.shape:
        raise ValueError(f"Shape mismatch: front={front.shape}, rear={rear.shape}")

    if len(front_idx) != len(rear_idx):
        raise ValueError(f"Index row mismatch: front={len(front_idx)}, rear={len(rear_idx)}")

    check_cols = [c for c in ["row_id", "image_id", "created_at", "date"] if c in front_idx.columns and c in rear_idx.columns]

    for col in check_cols:
        if not front_idx[col].astype(str).equals(rear_idx[col].astype(str)):
            raise ValueError(f"Index mismatch in column: {col}")

    n, d = front.shape

    mean_path = Path(str(out_prefix) + "_mean_embeddings.npy")
    concat_path = Path(str(out_prefix) + "_concat_embeddings.npy")
    index_path = Path(str(out_prefix) + "_embedding_index.csv")
    summary_path = Path(str(out_prefix) + "_summary.json")

    mean_emb = make_memmap(mean_path, (n, d), overwrite=args.overwrite)
    concat_emb = make_memmap(concat_path, (n, d * 2), overwrite=args.overwrite)

    chunk = 2048

    for start in range(0, n, chunk):
        end = min(start + chunk, n)

        mean_emb[start:end, :] = (front[start:end, :] + rear[start:end, :]) / 2.0
        concat_emb[start:end, :d] = front[start:end, :]
        concat_emb[start:end, d:] = rear[start:end, :]

    mean_emb.flush()
    concat_emb.flush()

    fused_index = front_idx.copy()
    fused_index["front_embedding_source"] = str(front_emb_path)
    fused_index["rear_embedding_source"] = str(rear_emb_path)
    fused_index.to_csv(index_path, index=False)

    summary = {
        "front_emb": str(front_emb_path),
        "rear_emb": str(rear_emb_path),
        "front_index": str(front_index_path),
        "rear_index": str(rear_index_path),
        "rows": int(n),
        "feature_dim_each": int(d),
        "mean_embeddings": str(mean_path),
        "concat_embeddings": str(concat_path),
        "concat_dim": int(d * 2),
        "index_path": str(index_path),
        "checked_alignment_columns": check_cols,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 90)
    print("FRONT+REAR SUPERVISED EMBEDDINGS FUSED")
    print("=" * 90)
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(" -", mean_path)
    print(" -", concat_path)
    print(" -", index_path)
    print(" -", summary_path)


if __name__ == "__main__":
    main()