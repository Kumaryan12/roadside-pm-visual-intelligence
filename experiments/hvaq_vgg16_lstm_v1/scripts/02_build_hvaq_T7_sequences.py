from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def make_random_split(n: int, seed: int, train_frac: float, val_frac: float):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    split = np.array(["test"] * n, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "val"

    return split.tolist()


def make_chrono_date_split(df: pd.DataFrame):
    dates = sorted(df["date"].unique().tolist())

    if len(dates) < 3:
        raise ValueError(f"Need at least 3 dates for chrono split. Found: {dates}")

    train_dates = dates[:-2]
    val_dates = [dates[-2]]
    test_dates = [dates[-1]]

    split = []
    for d in df["date"]:
        if d in train_dates:
            split.append("train")
        elif d in val_dates:
            split.append("val")
        elif d in test_dates:
            split.append("test")
        else:
            split.append("unused")

    return split, train_dates, val_dates, test_dates


def make_purged_block_split(df: pd.DataFrame, purge: int):
    split = pd.Series("unused", index=df.index, dtype=object)

    for _, g in df.groupby(["date", "location_id"], sort=False):
        g = g.sort_values("target_image_time")
        idx = g.index.to_list()
        n = len(idx)

        if n < 10:
            # Too small for block split. Keep as unused.
            continue

        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        train_cut = max(0, train_end - purge)
        val_start = min(n, train_end + purge)
        val_cut = max(val_start, val_end - purge)
        test_start = min(n, val_end + purge)

        for i in idx[:train_cut]:
            split.loc[i] = "train"

        for i in idx[val_start:val_cut]:
            split.loc[i] = "val"

        for i in idx[test_start:]:
            split.loc[i] = "test"

    return split.tolist()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/hvaq_vgg16_lstm_v1/data/processed/hvaq_image_label_manifest.csv",
    )
    parser.add_argument(
        "--out",
        default="experiments/hvaq_vgg16_lstm_v1/data/processed/hvaq_T7_sequence_manifest.csv",
    )
    parser.add_argument(
        "--report",
        default="experiments/hvaq_vgg16_lstm_v1/reports/hvaq_T7_sequence_summary.json",
    )

    parser.add_argument("--sequence-len", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--purge", type=int, default=3)

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    report_path = Path(args.report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "row_id",
        "image_path",
        "image_file",
        "image_time",
        "date",
        "location_id",
        "PM2.5",
        "PM10",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    df["image_time"] = pd.to_datetime(df["image_time"], errors="coerce")
    df = df.dropna(subset=["image_time"]).copy()

    sequences = []
    seq_id = 0

    for (date, loc), g in df.groupby(["date", "location_id"], sort=True):
        g = g.sort_values("image_time").reset_index(drop=True)

        if len(g) < args.sequence_len:
            continue

        for start in range(0, len(g) - args.sequence_len + 1):
            win = g.iloc[start:start + args.sequence_len]
            target = win.iloc[-1]

            sequences.append(
                {
                    "sequence_id": seq_id,
                    "date": date,
                    "location_id": int(loc),
                    "seq_row_ids": "|".join(win["row_id"].astype(int).astype(str).tolist()),
                    "seq_image_files": "|".join(win["image_file"].astype(str).tolist()),
                    "seq_image_paths": "|".join(win["image_path"].astype(str).tolist()),
                    "start_image_time": win.iloc[0]["image_time"],
                    "target_image_time": target["image_time"],
                    "target_row_id": int(target["row_id"]),
                    "target_PM2.5": float(target["PM2.5"]),
                    "target_PM10": float(target["PM10"]),
                    "target_temperature": float(target["Temperature"]) if "Temperature" in target and pd.notna(target["Temperature"]) else np.nan,
                    "target_humidity": float(target["Humidity"]) if "Humidity" in target and pd.notna(target["Humidity"]) else np.nan,
                }
            )
            seq_id += 1

    seq_df = pd.DataFrame(sequences)

    if seq_df.empty:
        raise RuntimeError("No T=7 sequences were created. Check manifest and sequence length.")

    # Random sequence split. This is paper-style but leakage-prone.
    seq_df["split_random"] = make_random_split(
        len(seq_df),
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    # Chronological date split.
    chrono_split, train_dates, val_dates, test_dates = make_chrono_date_split(seq_df)
    seq_df["split_chrono_date"] = chrono_split

    # Purged within-date/location block split.
    seq_df["split_purged_block"] = make_purged_block_split(seq_df, purge=args.purge)

    seq_df.to_csv(out_path, index=False)

    summary = {
        "input_manifest": str(manifest_path),
        "output_sequence_manifest": str(out_path),
        "sequence_len": args.sequence_len,
        "rows": int(len(seq_df)),
        "unique_dates": int(seq_df["date"].nunique()),
        "unique_locations": int(seq_df["location_id"].nunique()),
        "date_counts": seq_df["date"].value_counts().sort_index().to_dict(),
        "location_counts": seq_df["location_id"].value_counts().sort_index().to_dict(),
        "split_counts": {
            "random": seq_df["split_random"].value_counts().to_dict(),
            "chrono_date": seq_df["split_chrono_date"].value_counts().to_dict(),
            "purged_block": seq_df["split_purged_block"].value_counts().to_dict(),
        },
        "chrono_date_split_dates": {
            "train": train_dates,
            "val": val_dates,
            "test": test_dates,
        },
        "target_summary": seq_df[["target_PM2.5", "target_PM10"]].describe().to_dict(),
        "warning": "Random split is leakage-prone because sequences overlap and images repeat across locations. Use purged/date split for honest generalization.",
    }

    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("=" * 90)
    print("HVAQ T=7 SEQUENCE MANIFEST CREATED")
    print("=" * 90)
    print("Sequences:", len(seq_df))
    print("Dates:", sorted(seq_df["date"].unique().tolist()))
    print("Locations:", sorted(seq_df["location_id"].unique().tolist()))

    print("\nSplit counts:")
    print("Random:")
    print(seq_df["split_random"].value_counts())
    print("\nChrono date:")
    print(seq_df["split_chrono_date"].value_counts())
    print("\nPurged block:")
    print(seq_df["split_purged_block"].value_counts())

    print("\nTarget summary:")
    print(seq_df[["target_PM2.5", "target_PM10"]].describe())

    print("\nSaved:")
    print(" -", out_path)
    print(" -", report_path)


if __name__ == "__main__":
    main()