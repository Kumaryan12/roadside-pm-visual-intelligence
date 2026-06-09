from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index-csv",
        default="experiments/traqid_pretraining_v1/embeddings/traqid_mobilenetv2_torch_embedding_index.csv",
        help="Embedding index CSV. Its row order must match the embedding .npy file.",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=7,
        help="Number of consecutive rows per sequence.",
    )

    parser.add_argument(
        "--target",
        default="PM2.5",
        help="Target column, e.g. PM2.5, PM10, aqi.",
    )

    parser.add_argument(
        "--out",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_T7_front_sequence_manifest.csv",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/t7_sequence",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    index_csv = Path(args.index_csv)
    out_path = Path(args.out)
    report_dir = Path(args.report_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_csv)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "row_id",
        "created_at",
        "date",
        "split_date_chrono",
        args.target,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Index CSV missing required columns: {missing}")

    df = df.copy()
    df["embedding_row_index"] = range(len(df))
    df["created_at_parsed"] = pd.to_datetime(df["created_at"], errors="coerce")

    # Use Image if available for stable ordering among duplicate timestamps.
    sort_cols = ["date", "created_at_parsed"]
    if "image_id" in df.columns:
        sort_cols.append("image_id")
    elif "Image" in df.columns:
        sort_cols.append("Image")
    sort_cols.append("row_id")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    sequences = []
    seq_id = 0

    for date, group in df.groupby("date", sort=True):
        group = group.sort_values(sort_cols).reset_index(drop=True)

        if len(group) < args.seq_len:
            continue

        for end_pos in range(args.seq_len - 1, len(group)):
            window = group.iloc[end_pos - args.seq_len + 1 : end_pos + 1]
            target_row = window.iloc[-1]

            # For date-safe split, all rows in a date usually share same split.
            # Still, enforce same split inside sequence to avoid contamination.
            splits = set(window["split_date_chrono"].astype(str).tolist())
            if len(splits) != 1:
                continue

            seq_embedding_indices = window["embedding_row_index"].astype(int).tolist()
            seq_row_ids = window["row_id"].astype(int).tolist()

            row = {
                "sequence_id": seq_id,
                "date": str(date),
                "split_date_chrono": str(target_row["split_date_chrono"]),
                "target_row_id": int(target_row["row_id"]),
                "target_embedding_row_index": int(target_row["embedding_row_index"]),
                "target_created_at": str(target_row["created_at"]),
                "seq_len": args.seq_len,
                "seq_embedding_indices": "|".join(map(str, seq_embedding_indices)),
                "seq_row_ids": "|".join(map(str, seq_row_ids)),
                "target": args.target,
                "target_value": float(target_row[args.target]),
            }

            # Keep useful labels if available.
            for col in ["PM2.5", "PM10", "aqi", "Temperature", "Humidity", "Season", "Day_or_Night", "aqi_cat"]:
                if col in target_row.index:
                    row[col] = target_row[col]

            sequences.append(row)
            seq_id += 1

    seq_df = pd.DataFrame(sequences)

    if seq_df.empty:
        raise RuntimeError("No sequences were created. Check dates, seq_len, and split column.")

    seq_df.to_csv(out_path, index=False)

    summary = {
        "index_csv": str(index_csv),
        "out": str(out_path),
        "seq_len": args.seq_len,
        "target": args.target,
        "input_rows": int(len(df)),
        "sequence_rows": int(len(seq_df)),
        "unique_dates": int(seq_df["date"].nunique()),
        "split_counts": seq_df["split_date_chrono"].value_counts().to_dict(),
        "date_counts": seq_df["date"].value_counts().sort_index().to_dict(),
        "note": "Each sequence uses consecutive rows within the same TRAQID date. Target is the final row in the sequence.",
    }

    summary_path = report_dir / "traqid_T7_sequence_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 90)
    print("TRAQID T=7 SEQUENCE MANIFEST BUILT")
    print("=" * 90)
    print(json.dumps(summary, indent=2))

    print("\nPreview:")
    print(seq_df.head(10).to_string(index=False))

    print("\nSaved:")
    print(" -", out_path)
    print(" -", summary_path)


if __name__ == "__main__":
    main()