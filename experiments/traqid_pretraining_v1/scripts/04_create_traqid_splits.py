from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest.csv",
    )

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/data/processed",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df["created_at_parsed"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["date"] = df["created_at_parsed"].dt.date.astype(str)
    df["hour_block"] = df["created_at_parsed"].dt.floor("h").astype(str)

    # Keep rows with valid image pairs.
    df = df[(df["front_exists"] == True) & (df["rear_exists"] == True)].copy()

    # Clean environment flag.
    df["env_plausible"] = (
        df["Temperature"].between(-10, 60)
        & df["Humidity"].between(0, 100)
    )

    date_stats = (
        df.groupby("date")
        .agg(
            rows=("row_id", "count"),
            pm25_mean=("PM2.5", "mean"),
            pm25_min=("PM2.5", "min"),
            pm25_max=("PM2.5", "max"),
            pm10_mean=("PM10", "mean"),
            aqi_mean=("aqi", "mean"),
            day_count=("Day_or_Night", lambda x: int((x == "Day").sum())),
            night_count=("Day_or_Night", lambda x: int((x == "Night").sum())),
            env_bad=("env_plausible", lambda x: int((~x).sum())),
        )
        .reset_index()
        .sort_values("date")
    )

    dates = date_stats["date"].tolist()

    # Date-based split:
    # Use middle majority for train, later dates for val/test.
    # This is not perfect, but much safer than random row/image split.
    n_dates = len(dates)

    if n_dates < 5:
        raise RuntimeError("Too few dates for date-based split.")

    train_cut = int(n_dates * 0.70)
    val_cut = int(n_dates * 0.85)

    train_dates = dates[:train_cut]
    val_dates = dates[train_cut:val_cut]
    test_dates = dates[val_cut:]

    def assign_split(date: str) -> str:
        if date in train_dates:
            return "train"
        if date in val_dates:
            return "val"
        return "test"

    df["split_date_chrono"] = df["date"].apply(assign_split)

    # Also create a grouped fold ID by date for cross-validation.
    date_to_fold = {date: i for i, date in enumerate(dates)}
    df["date_fold_id"] = df["date"].map(date_to_fold)

    split_counts = df["split_date_chrono"].value_counts().to_dict()

    split_target_summary = (
        df.groupby("split_date_chrono")[["PM2.5", "PM10", "aqi", "Temperature", "Humidity"]]
        .describe()
    )

    out_path = out_dir / "traqid_paired_manifest_with_splits.csv"
    date_stats_path = report_dir / "traqid_date_stats.csv"
    split_summary_path = report_dir / "traqid_split_summary.json"
    split_target_summary_path = report_dir / "traqid_split_target_summary.csv"

    df.to_csv(out_path, index=False)
    date_stats.to_csv(date_stats_path, index=False)
    split_target_summary.to_csv(split_target_summary_path)

    summary = {
        "input_manifest": str(manifest_path),
        "output_manifest": str(out_path),
        "rows": int(len(df)),
        "unique_dates": int(df["date"].nunique()),
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "env_implausible_rows": int((~df["env_plausible"]).sum()),
        "warning": (
            "Splits are date-based to reduce leakage from repeated timestamps, "
            "front/rear paired images, and near-duplicate sequential samples."
        ),
    }

    split_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 90)
    print("TRAQID LEAKAGE-SAFE SPLITS CREATED")
    print("=" * 90)
    print(json.dumps(summary, indent=2))

    print("\nDate stats:")
    print(date_stats.to_string(index=False))

    print("\nSplit target summary saved:")
    print(" -", split_target_summary_path)

    print("\nSaved:")
    print(" -", out_path)
    print(" -", date_stats_path)
    print(" -", split_summary_path)


if __name__ == "__main__":
    main()