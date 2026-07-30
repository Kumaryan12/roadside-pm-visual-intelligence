from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("experiments/traqid_pretraining_v1")

SRC = ROOT / "data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest.csv"
OUT = ROOT / "data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest_time_balanced_purged.csv"

T = 7
PURGE = T - 1
BLOCK_SIZE = 96

BLOCK_PATTERN = [
    "train", "train", "test", "train", "val", "train", "test"
]


def parse_ids(s: str) -> set[int]:
    return set(int(x) for x in str(s).split("|") if str(x).strip())


def max_overlap(ids: set[int], train_sets: list[set[int]]) -> int:
    best = 0
    for tr in train_sets:
        ov = len(ids & tr)
        if ov > best:
            best = ov
    return best


def assign_date_blocks(g: pd.DataFrame) -> pd.DataFrame:
    # Stable sorting is important because many rows can share target_time.
    g = g.sort_values(["target_time", "sequence_id"]).reset_index(drop=True).copy()
    n = len(g)

    split = np.array(["purge"] * n, dtype=object)
    block_ids = np.full(n, -1, dtype=int)

    cursor = 0
    block_idx = 0

    while cursor < n:
        label = BLOCK_PATTERN[block_idx % len(BLOCK_PATTERN)]

        block_start = cursor
        block_end = min(cursor + BLOCK_SIZE, n)

        split[block_start:block_end] = label
        block_ids[block_start:block_end] = block_idx

        # Purge after every block.
        purge_start = block_end
        purge_end = min(block_end + PURGE, n)

        split[purge_start:purge_end] = "purge"
        block_ids[purge_start:purge_end] = block_idx

        cursor = purge_end
        block_idx += 1

    g["split_time_balanced_purged"] = split
    g["balanced_block_id"] = block_ids

    return g


def strict_remove_test_val_overlap_with_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    After block assignment, remove any val/test row that still shares frames
    with train. This guarantees train-test and train-val overlap are zero.
    """
    df = df.copy()

    train_sets = [
        parse_ids(x)
        for x in df.loc[df["split_time_balanced_purged"] == "train", "seq_row_ids"]
    ]

    changed = 0

    for idx, row in df.iterrows():
        split = row["split_time_balanced_purged"]

        if split not in ["val", "test"]:
            continue

        ids = parse_ids(row["seq_row_ids"])
        ov = max_overlap(ids, train_sets)

        if ov > 0:
            df.at[idx, "split_time_balanced_purged"] = "purge"
            changed += 1

    print(f"Strict cleanup moved {changed} val/test sequences to purge due to train overlap.")
    return df


def overlap_summary(df: pd.DataFrame) -> dict:
    used = df[df["split_time_balanced_purged"].isin(["train", "val", "test"])].copy()

    train = used[used["split_time_balanced_purged"] == "train"]
    val = used[used["split_time_balanced_purged"] == "val"]
    test = used[used["split_time_balanced_purged"] == "test"]

    train_sets = [parse_ids(x) for x in train["seq_row_ids"]]

    def collect(part):
        vals = []
        for x in part["seq_row_ids"]:
            vals.append(max_overlap(parse_ids(x), train_sets))
        return np.asarray(vals, dtype=int)

    test_ov = collect(test)
    val_ov = collect(val)

    return {
        "train_sequences": int(len(train)),
        "val_sequences": int(len(val)),
        "test_sequences": int(len(test)),
        "purge_sequences": int((df["split_time_balanced_purged"] == "purge").sum()),

        "test_max_overlap_frames": int(test_ov.max()) if len(test_ov) else None,
        "test_fraction_overlap_ge_1": float((test_ov >= 1).mean()) if len(test_ov) else None,
        "test_fraction_overlap_ge_6": float((test_ov >= 6).mean()) if len(test_ov) else None,

        "val_max_overlap_frames": int(val_ov.max()) if len(val_ov) else None,
        "val_fraction_overlap_ge_1": float((val_ov >= 1).mean()) if len(val_ov) else None,
        "val_fraction_overlap_ge_6": float((val_ov >= 6).mean()) if len(val_ov) else None,
    }


def main():
    df = pd.read_csv(SRC)

    required = [
        "date",
        "sequence_id",
        "seq_row_ids",
        "target_row_id",
        "target_time",
        "target_PM2.5",
        "target_PM10",
        "target_aqi",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["target_time"] = pd.to_datetime(df["target_time"], errors="coerce")

    parts = []
    for date, g in df.groupby("date", sort=True):
        part = assign_date_blocks(g)
        part["date"] = date
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)

    # Strict cleanup to guarantee no train-test / train-val frame overlap.
    out = strict_remove_test_val_overlap_with_train(out)

    summary = overlap_summary(out)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("=" * 90)
    print("TIME-BALANCED PURGED SPLIT CREATED")
    print("=" * 90)
    print("Saved:", OUT)
    print("Shape:", out.shape)

    print("\nSplit counts:")
    print(out["split_time_balanced_purged"].value_counts())

    print("\nOverlap check:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    used = out[out["split_time_balanced_purged"].isin(["train", "val", "test"])].copy()

    print("\nDates per split:")
    for split, g in used.groupby("split_time_balanced_purged"):
        print(split, g["date"].nunique())

    print("\nRows by date/split:")
    print(pd.crosstab(out["date"], out["split_time_balanced_purged"]).to_string())

    print("\nHour/day-night check:")
    used["hour"] = pd.to_datetime(used["target_time"]).dt.hour
    used["is_day"] = ((used["hour"] >= 6) & (used["hour"] < 18)).astype(int)

    for split, g in used.groupby("split_time_balanced_purged"):
        print(
            split,
            "n=", len(g),
            "hour_mean=", round(g["hour"].mean(), 3),
            "day_fraction=", round(g["is_day"].mean(), 3),
        )


if __name__ == "__main__":
    main()