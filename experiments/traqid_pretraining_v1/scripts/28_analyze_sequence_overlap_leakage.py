from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_ids(s):
    return set(int(x) for x in str(s).split("|") if str(x).strip() != "")


def max_overlap_against_train(test_ids_set, train_sets):
    max_ov = 0
    for tr in train_sets:
        ov = len(test_ids_set & tr)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def analyze_overlap(df, split_col, train_name, test_name, seq_len_col=None):
    train_df = df[df[split_col] == train_name].copy()
    test_df = df[df[split_col] == test_name].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Empty split for {split_col}: train={len(train_df)}, test={len(test_df)}")

    train_sets = [parse_ids(x) for x in train_df["seq_row_ids"]]

    rows = []
    for _, row in test_df.iterrows():
        ids = parse_ids(row["seq_row_ids"])
        max_ov = max_overlap_against_train(ids, train_sets)
        seq_len = len(ids)

        rows.append(
            {
                "sequence_id": int(row["sequence_id"]),
                "date": str(row.get("date", "")),
                "seq_len": seq_len,
                "max_train_overlap_frames": max_ov,
                "max_train_overlap_fraction": max_ov / seq_len if seq_len else 0.0,
            }
        )

    out = pd.DataFrame(rows)

    summary = {
        "split_col": split_col,
        "train_name": train_name,
        "test_name": test_name,
        "train_sequences": int(len(train_df)),
        "test_sequences": int(len(test_df)),
        "mean_max_overlap_frames": float(out["max_train_overlap_frames"].mean()),
        "median_max_overlap_frames": float(out["max_train_overlap_frames"].median()),
        "max_overlap_frames": int(out["max_train_overlap_frames"].max()),
        "fraction_test_overlap_ge_1": float((out["max_train_overlap_frames"] >= 1).mean()),
        "fraction_test_overlap_ge_3": float((out["max_train_overlap_frames"] >= 3).mean()),
        "fraction_test_overlap_ge_5": float((out["max_train_overlap_frames"] >= 5).mean()),
        "fraction_test_overlap_ge_6": float((out["max_train_overlap_frames"] >= 6).mean()),
        "fraction_test_overlap_eq_0": float((out["max_train_overlap_frames"] == 0).mean()),
    }

    return out, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--random-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest.csv",
    )
    parser.add_argument(
        "--purged-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest_purged_block_compatible_clean.csv",
    )
    parser.add_argument(
        "--chrono-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/paper_diagnostics/sequence_overlap",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyses = []

    random_df = pd.read_csv(args.random_manifest)
    purged_df = pd.read_csv(args.purged_manifest)
    chrono_df = pd.read_csv(args.chrono_manifest)

    jobs = [
        {
            "name": "random_70_15_15",
            "df": random_df,
            "split_col": "split_random",
            "train_name": "train",
            "test_name": "test",
        },
        {
            "name": "twofold_random",
            "df": random_df,
            "split_col": "split_twofold",
            "train_name": "fold1_train",
            "test_name": "fold1_test",
        },
        {
            "name": "purged_block",
            "df": purged_df,
            "split_col": "split_date_chrono",
            "train_name": "train",
            "test_name": "test",
        },
        {
            "name": "chrono_date",
            "df": chrono_df,
            "split_col": "split_chrono_date",
            "train_name": "train",
            "test_name": "test",
        },
    ]

    for job in jobs:
        print("\n" + "=" * 90)
        print("Analyzing:", job["name"])

        overlap_df, summary = analyze_overlap(
            job["df"],
            split_col=job["split_col"],
            train_name=job["train_name"],
            test_name=job["test_name"],
        )

        overlap_path = out_dir / f"{job['name']}_test_sequence_overlap.csv"
        summary_path = out_dir / f"{job['name']}_sequence_overlap_summary.json"

        overlap_df.to_csv(overlap_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        summary["protocol"] = job["name"]
        analyses.append(summary)

        print(json.dumps(summary, indent=2))
        print("Saved:", overlap_path)
        print("Saved:", summary_path)

    summary_df = pd.DataFrame(analyses)
    summary_csv = out_dir / "sequence_overlap_summary_all_protocols.csv"
    summary_md = out_dir / "sequence_overlap_summary_all_protocols.md"

    summary_df.to_csv(summary_csv, index=False)

    pretty = summary_df.copy()
    for c in pretty.columns:
        if pretty[c].dtype.kind in "fc":
            pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")
    summary_md.write_text(pretty.to_markdown(index=False), encoding="utf-8")

    print("\n" + "=" * 90)
    print("ALL PROTOCOLS SUMMARY")
    print(summary_df.to_string(index=False))
    print("\nSaved:", summary_csv)
    print("Saved:", summary_md)


if __name__ == "__main__":
    main()