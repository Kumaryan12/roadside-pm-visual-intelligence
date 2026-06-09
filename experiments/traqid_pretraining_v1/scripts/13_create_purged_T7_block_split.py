from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_indices(s: str) -> set[int]:
    return set(int(x) for x in str(s).split("|") if str(x).strip())


def check_overlap_leakage(df: pd.DataFrame) -> dict:
    """
    Checks whether train/val/test sequences within the same date share any embedding row indices.
    If purge is working, train-val, train-test, val-test overlaps should be zero.
    """
    results = {}

    for date, g in df[df["split_date_chrono"].isin(["train", "val", "test"])].groupby("date"):
        split_sets = {}

        for split in ["train", "val", "test"]:
            sub = g[g["split_date_chrono"] == split]
            used = set()

            for s in sub["seq_embedding_indices"]:
                used |= parse_indices(s)

            split_sets[split] = used

        results[str(date)] = {
            "train_val_overlap": len(split_sets["train"] & split_sets["val"]),
            "train_test_overlap": len(split_sets["train"] & split_sets["test"]),
            "val_test_overlap": len(split_sets["val"] & split_sets["test"]),
        }

    total = {
        "total_train_val_overlap": sum(v["train_val_overlap"] for v in results.values()),
        "total_train_test_overlap": sum(v["train_test_overlap"] for v in results.values()),
        "total_val_test_overlap": sum(v["val_test_overlap"] for v in results.values()),
    }

    return {
        "by_date": results,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_T7_front_sequence_manifest.csv",
    )

    parser.add_argument(
        "--output",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_T7_front_sequence_manifest_purged_block_split.csv",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/t7_purged_block_split",
    )

    parser.add_argument("--seq-len", type=int, default=7)

    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)

    parser.add_argument(
        "--purge-size",
        type=int,
        default=None,
        help="Number of sequences to purge between blocks. Default = seq_len - 1.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_dir = Path(args.report_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.purge_size is None:
        purge_size = args.seq_len - 1
    else:
        purge_size = args.purge_size

    if abs((args.train_frac + args.val_frac + args.test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    df = pd.read_csv(input_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "sequence_id",
        "date",
        "target_created_at",
        "target_row_id",
        "seq_embedding_indices",
        "target_value",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input sequence manifest missing columns: {missing}")

    df = df.copy()
    df["target_created_at_parsed"] = pd.to_datetime(df["target_created_at"], errors="coerce")

    # Preserve original split, then overwrite split_date_chrono with purged block split.
    if "original_split_date_chrono" not in df.columns:
        df["original_split_date_chrono"] = df.get("split_date_chrono", "")

    all_parts = []
    per_date_summary = []

    for date, g in df.groupby("date", sort=True):
        g = g.sort_values(["target_created_at_parsed", "target_row_id", "sequence_id"]).reset_index(drop=True)
        n = len(g)

        train_end = int(n * args.train_frac)
        val_end = int(n * (args.train_frac + args.val_frac))

        g["local_sequence_pos"] = range(n)
        g["split_date_chrono"] = "purge"

        # Block layout:
        # train block
        # purge gap
        # val block
        # purge gap
        # test block

        train_start_idx = 0
        train_end_idx = max(train_start_idx, train_end - purge_size)

        val_start_idx = train_end + purge_size
        val_end_idx = max(val_start_idx, val_end - purge_size)

        test_start_idx = val_end + purge_size
        test_end_idx = n

        if train_end_idx > train_start_idx:
            g.loc[train_start_idx:train_end_idx - 1, "split_date_chrono"] = "train"

        if val_end_idx > val_start_idx:
            g.loc[val_start_idx:val_end_idx - 1, "split_date_chrono"] = "val"

        if test_end_idx > test_start_idx:
            g.loc[test_start_idx:test_end_idx - 1, "split_date_chrono"] = "test"

        counts = g["split_date_chrono"].value_counts().to_dict()

        per_date_summary.append(
            {
                "date": str(date),
                "total_sequences": int(n),
                "train": int(counts.get("train", 0)),
                "val": int(counts.get("val", 0)),
                "test": int(counts.get("test", 0)),
                "purge": int(counts.get("purge", 0)),
                "train_end_raw": int(train_end),
                "val_end_raw": int(val_end),
                "purge_size": int(purge_size),
            }
        )

        all_parts.append(g)

    out = pd.concat(all_parts, ignore_index=True)

    leakage = check_overlap_leakage(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    per_date_df = pd.DataFrame(per_date_summary)
    per_date_path = report_dir / "purged_T7_block_split_per_date_summary.csv"
    per_date_df.to_csv(per_date_path, index=False)

    split_counts = out["split_date_chrono"].value_counts().to_dict()

    split_by_date_path = report_dir / "purged_T7_block_split_date_crosstab.csv"
    pd.crosstab(out["date"], out["split_date_chrono"]).to_csv(split_by_date_path)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "seq_len": int(args.seq_len),
        "purge_size": int(purge_size),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "input_sequences": int(len(df)),
        "output_sequences": int(len(out)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "usable_sequences": int(out["split_date_chrono"].isin(["train", "val", "test"]).sum()),
        "purged_sequences": int((out["split_date_chrono"] == "purge").sum()),
        "overlap_leakage_check_total": leakage["total"],
        "note": (
            "Purged block split keeps all dates represented while preventing overlapping "
            "T=7 sequence frames from crossing train/val/test boundaries."
        ),
    }

    summary_path = report_dir / "purged_T7_block_split_summary.json"
    leakage_path = report_dir / "purged_T7_block_split_overlap_check.json"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    leakage_path.write_text(json.dumps(leakage, indent=2), encoding="utf-8")

    print("=" * 90)
    print("PURGED T=7 BLOCK SPLIT CREATED")
    print("=" * 90)
    print(json.dumps(summary, indent=2))

    print("\nPer-date summary:")
    print(per_date_df.to_string(index=False))

    print("\nDate × split table:")
    print(pd.crosstab(out["date"], out["split_date_chrono"]).to_string())

    print("\nSaved:")
    print(" -", output_path)
    print(" -", summary_path)
    print(" -", leakage_path)
    print(" -", per_date_path)
    print(" -", split_by_date_path)


if __name__ == "__main__":
    main()