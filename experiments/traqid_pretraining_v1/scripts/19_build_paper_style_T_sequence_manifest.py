from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def make_twofold_split(n: int, seed: int):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    half = n // 2

    fold_a = np.array(["fold2_test"] * n, dtype=object)
    fold_a[idx[:half]] = "fold1_train"
    fold_a[idx[half:]] = "fold1_test"

    return fold_a.tolist()


def make_random_train_val_test(n: int, seed: int, train_frac: float, val_frac: float):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    split = np.array(["test"] * n, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "val"

    return split.tolist()


def make_chrono_date_split(seq_df: pd.DataFrame):
    dates = sorted(seq_df["date"].astype(str).unique().tolist())

    if len(dates) < 5:
        raise ValueError(f"Need enough dates for chrono split. Found: {dates}")

    train_dates = dates[: int(0.70 * len(dates))]
    val_dates = dates[int(0.70 * len(dates)) : int(0.85 * len(dates))]
    test_dates = dates[int(0.85 * len(dates)) :]

    split = []
    for d in seq_df["date"].astype(str):
        if d in train_dates:
            split.append("train")
        elif d in val_dates:
            split.append("val")
        elif d in test_dates:
            split.append("test")
        else:
            split.append("unused")

    return split, train_dates, val_dates, test_dates


def build_sequences(df: pd.DataFrame, sequence_len: int, view: str):
    image_col = f"{view}_path"
    exists_col = f"{view}_exists"

    if image_col not in df.columns:
        raise ValueError(f"Missing image column: {image_col}")

    if exists_col in df.columns:
        df = df[df[exists_col] == True].copy()

    required = [
        "row_id",
        "created_at",
        "date",
        image_col,
        "PM2.5",
        "PM10",
        "aqi",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    df["created_at_sort"] = pd.to_datetime(
        df.get("created_at_parsed", df["created_at"]),
        errors="coerce",
    )

    df = df.dropna(subset=["created_at_sort"]).copy()

    sequences = []
    seq_id = 0

    for date, g in df.groupby("date", sort=True):
        g = g.sort_values(["created_at_sort", "row_id"]).reset_index(drop=True)

        if len(g) < sequence_len:
            continue

        for start in range(0, len(g) - sequence_len + 1):
            win = g.iloc[start : start + sequence_len]
            target = win.iloc[-1]

            sequences.append(
                {
                    "sequence_id": seq_id,
                    "date": str(date),
                    "view": view,
                    "sequence_len": sequence_len,
                    "seq_row_ids": "|".join(win["row_id"].astype(int).astype(str).tolist()),
                    "seq_image_paths": "|".join(win[image_col].astype(str).tolist()),
                    "start_time": str(win.iloc[0]["created_at_sort"]),
                    "target_time": str(target["created_at_sort"]),
                    "target_row_id": int(target["row_id"]),
                    "target_PM2.5": float(target["PM2.5"]),
                    "target_PM10": float(target["PM10"]),
                    "target_aqi": float(target["aqi"]),
                }
            )
            seq_id += 1

    return pd.DataFrame(sequences)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )
    parser.add_argument("--view", default="front", choices=["front", "rear"])
    parser.add_argument("--min-t", type=int, default=2)
    parser.add_argument("--max-t", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/data/processed/paper_style_sequences",
    )
    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/paper_style_sequences",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["created_at"], errors="coerce").dt.date.astype(str)

    all_summaries = {}

    print("=" * 90)
    print("PAPER-STYLE TRAQID SEQUENCE MANIFEST BUILDER")
    print("=" * 90)
    print("Manifest:", manifest_path)
    print("Rows:", len(df))
    print("View:", args.view)
    print("T range:", args.min_t, "to", args.max_t)

    for t in range(args.min_t, args.max_t + 1):
        seq_df = build_sequences(df, sequence_len=t, view=args.view)

        if seq_df.empty:
            print(f"\nT={t}: no sequences created")
            continue

        seq_df["split_random"] = make_random_train_val_test(
            len(seq_df),
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )

        seq_df["split_twofold"] = make_twofold_split(len(seq_df), seed=args.seed)

        chrono_split, train_dates, val_dates, test_dates = make_chrono_date_split(seq_df)
        seq_df["split_chrono_date"] = chrono_split

        out_path = out_dir / f"traqid_paper_style_T{t}_{args.view}_sequence_manifest.csv"
        seq_df.to_csv(out_path, index=False)

        summary = {
            "manifest": str(manifest_path),
            "view": args.view,
            "sequence_len": t,
            "rows": int(len(seq_df)),
            "unique_dates": int(seq_df["date"].nunique()),
            "date_counts": seq_df["date"].value_counts().sort_index().to_dict(),
            "split_counts": {
                "random": seq_df["split_random"].value_counts().to_dict(),
                "twofold": seq_df["split_twofold"].value_counts().to_dict(),
                "chrono_date": seq_df["split_chrono_date"].value_counts().to_dict(),
            },
            "chrono_date_split_dates": {
                "train": train_dates,
                "val": val_dates,
                "test": test_dates,
            },
            "target_summary": seq_df[
                ["target_PM2.5", "target_PM10", "target_aqi"]
            ].describe().to_dict(),
            "output": str(out_path),
        }

        summary_path = report_dir / f"traqid_paper_style_T{t}_{args.view}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        all_summaries[f"T{t}"] = summary

        print("\n" + "-" * 90)
        print(f"T={t}")
        print("Sequences:", len(seq_df))
        print("Output:", out_path)
        print("Random split:")
        print(seq_df["split_random"].value_counts())
        print("Twofold split:")
        print(seq_df["split_twofold"].value_counts())
        print("Chrono-date split:")
        print(seq_df["split_chrono_date"].value_counts())

    all_summary_path = report_dir / f"traqid_paper_style_{args.view}_all_T_summary.json"
    all_summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    print("\nDONE")
    print("Saved sequence manifests to:", out_dir)
    print("Saved reports to:", report_dir)


if __name__ == "__main__":
    main()