import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGETS = ["PM2.5", "PM10", "aqi"]


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def find_time_column(df):
    for c in ["created_at", "target_created_at", "timestamp", "datetime"]:
        if c in df.columns:
            return c
    raise ValueError("No timestamp column found.")


def normalize_row_id(df):
    df = df.copy()

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
    df = df.dropna(subset=["row_id"]).copy()
    df["row_id"] = df["row_id"].astype(int)

    return df


def assign_random_row_split(df, train_frac, val_frac, seed):
    idx = np.arange(len(df))

    trainval_idx, test_idx = train_test_split(
        idx,
        train_size=train_frac + val_frac,
        random_state=seed,
        shuffle=True,
    )

    relative_val_frac = val_frac / (train_frac + val_frac)

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val_frac,
        random_state=seed,
        shuffle=True,
    )

    split = np.full(len(df), "unused", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"

    return split


def assign_chrono_split(df, train_frac, val_frac):
    n = len(df)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    split[val_end:] = "test"

    return split


def apply_purge(df, split_col, purge_rows):
    if purge_rows <= 0:
        return df

    df = df.copy()
    split = df[split_col].values.copy()

    boundaries = []
    for i in range(1, len(split)):
        if split[i] != split[i - 1]:
            boundaries.append(i)

    purge_mask = np.zeros(len(df), dtype=bool)

    for b in boundaries:
        start = max(0, b - purge_rows)
        end = min(len(df), b + purge_rows)
        purge_mask[start:end] = True

    df.loc[purge_mask, split_col] = "purged"

    return df


def build_sequences_within_split(df, split_name, T):
    part = df[df["split_first_split"] == split_name].copy()
    part = part.sort_values("row_id").reset_index(drop=True)

    rows = []

    if len(part) < T:
        return pd.DataFrame(rows)

    for i in range(T - 1, len(part)):
        window = part.iloc[i - T + 1:i + 1].copy()
        target = part.iloc[i]

        row_ids = window["row_id"].values

        # Require consecutive row_ids so no gaps / cross-split contamination.
        if not np.all(np.diff(row_ids) == 1):
            continue

        row = {
            "sequence_id": f"{split_name}_{int(target['row_id'])}",
            "target_row_id": int(target["row_id"]),
            "target_created_at": target["created_at"],
            "seq_start_row_id": int(row_ids[0]),
            "seq_end_row_id": int(row_ids[-1]),
            "T": T,
            "split_first_split": split_name,
        }

        for target_col in TARGETS:
            if target_col in target.index:
                row[target_col] = target[target_col]

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/data/processed/split_first_sequences",
    )
    parser.add_argument("--T", type=int, default=7)
    parser.add_argument("--split-mode", choices=["random_row", "chrono"], default="chrono")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--purge-rows", type=int, default=6)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = safe_read_csv(args.manifest)
    df = normalize_row_id(df)

    time_col = find_time_column(df)
    df["created_at"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["created_at"]).copy()

    missing_targets = [c for c in TARGETS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing targets in manifest: {missing_targets}")

    df = df.dropna(subset=TARGETS).copy()
    df = df.sort_values("row_id").reset_index(drop=True)

    if args.split_mode == "random_row":
        df["split_first_split"] = assign_random_row_split(
            df,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
    else:
        df["split_first_split"] = assign_chrono_split(
            df,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )

    df = apply_purge(df, "split_first_split", args.purge_rows)

    seq_parts = []
    for split_name in ["train", "val", "test"]:
        seq_part = build_sequences_within_split(df, split_name, args.T)
        seq_parts.append(seq_part)

    seq = pd.concat(seq_parts, ignore_index=True)

    out_path = out_dir / f"traqid_split_first_T{args.T}_{args.split_mode}_purge{args.purge_rows}.csv"
    summary_path = out_dir / f"traqid_split_first_T{args.T}_{args.split_mode}_purge{args.purge_rows}_summary.csv"

    seq.to_csv(out_path, index=False)

    summary = []

    for split_name in ["train", "val", "test", "purged"]:
        row_count = int((df["split_first_split"] == split_name).sum())
        seq_count = int((seq["split_first_split"] == split_name).sum()) if len(seq) else 0

        summary.append({
            "split": split_name,
            "raw_rows": row_count,
            "sequences": seq_count,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("SPLIT-FIRST T7 SEQUENCE BUILD COMPLETE")
    print("=" * 90)
    print("Manifest:", args.manifest)
    print("Split mode:", args.split_mode)
    print("T:", args.T)
    print("Purge rows:", args.purge_rows)
    print()
    print(summary_df.to_string(index=False))
    print()
    print("Saved:", out_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()