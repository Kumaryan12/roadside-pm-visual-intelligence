import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["PM2.5", "PM10", "aqi"]


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def find_time_col(df):
    for c in ["created_at", "target_created_at", "timestamp", "datetime"]:
        if c in df.columns:
            return c
    raise ValueError("No timestamp column found.")


def summarize_group(df, group_col):
    rows = []

    for group, g in df.groupby(group_col):
        row = {
            group_col: group,
            "rows": len(g),
            "start_time": g["created_at"].min(),
            "end_time": g["created_at"].max(),
            "unique_timestamps": g["created_at"].nunique(),
        }

        for target in TARGETS:
            if target in g.columns:
                s = g[target].dropna()
                row[f"{target}_mean"] = s.mean()
                row[f"{target}_std"] = s.std()
                row[f"{target}_min"] = s.min()
                row[f"{target}_p25"] = s.quantile(0.25)
                row[f"{target}_median"] = s.median()
                row[f"{target}_p75"] = s.quantile(0.75)
                row[f"{target}_max"] = s.max()

        if "Day_or_Night" in g.columns:
            counts = g["Day_or_Night"].astype(str).value_counts()
            for k, v in counts.items():
                row[f"Day_or_Night_{k}"] = v

        if "Season" in g.columns:
            counts = g["Season"].astype(str).value_counts()
            for k, v in counts.items():
                row[f"Season_{k}"] = v

        if "Temperature" in g.columns:
            row["Temperature_mean"] = g["Temperature"].mean()
            row["Temperature_min"] = g["Temperature"].min()
            row["Temperature_max"] = g["Temperature"].max()

        if "Humidity" in g.columns:
            row["Humidity_mean"] = g["Humidity"].mean()
            row["Humidity_min"] = g["Humidity"].min()
            row["Humidity_max"] = g["Humidity"].max()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)


def print_basic(df):
    print("\n" + "=" * 100)
    print("BASIC TRAQID DATA CHECK")
    print("=" * 100)
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print("Time range:", df["created_at"].min(), "to", df["created_at"].max())
    print("Unique timestamps:", df["created_at"].nunique())
    print("Unique dates:", df["date"].nunique())
    print("Unique months:", df["month"].nunique())

    print("\nTargets:")
    for target in TARGETS:
        if target in df.columns:
            print("\n" + target)
            print(df[target].describe().to_string())

    for col in ["Day_or_Night", "Season"]:
        if col in df.columns:
            print("\n" + col)
            print(df[col].value_counts(dropna=False).to_string())


def make_candidate_fold_report(date_summary):
    rows = []

    total_rows = date_summary["rows"].sum()
    global_mean = np.average(date_summary["PM2.5_mean"], weights=date_summary["rows"])

    for _, r in date_summary.iterrows():
        test_rows = r["rows"]
        train_rows = total_rows - test_rows

        rows.append({
            "heldout_date": r["date"],
            "test_rows": test_rows,
            "train_rows": train_rows,
            "test_PM2.5_mean": r["PM2.5_mean"],
            "test_PM2.5_median": r["PM2.5_median"],
            "test_PM2.5_min": r["PM2.5_min"],
            "test_PM2.5_max": r["PM2.5_max"],
            "global_PM2.5_mean": global_mean,
            "mean_shift_vs_global": r["PM2.5_mean"] - global_mean,
            "abs_mean_shift_vs_global": abs(r["PM2.5_mean"] - global_mean),
        })

    out = pd.DataFrame(rows)
    return out.sort_values("abs_mean_shift_vs_global", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/traqid_group_structure_analysis",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = safe_read_csv(args.manifest)

    time_col = find_time_col(df)
    df["created_at"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["created_at"]).copy()

    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        print("Warning: missing targets:", missing_targets)

    df = df.dropna(subset=[t for t in TARGETS if t in df.columns]).copy()

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    df["date"] = df["created_at"].dt.date.astype(str)
    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    df["hour"] = df["created_at"].dt.hour
    df["dayofweek"] = df["created_at"].dt.dayofweek

    print_basic(df)

    date_summary = summarize_group(df, "date")
    month_summary = summarize_group(df, "month")
    hour_summary = summarize_group(df, "hour")
    fold_candidate = make_candidate_fold_report(date_summary)

    date_path = out_dir / "traqid_date_group_summary.csv"
    month_path = out_dir / "traqid_month_group_summary.csv"
    hour_path = out_dir / "traqid_hour_summary.csv"
    fold_path = out_dir / "traqid_leave_one_date_candidate_report.csv"

    date_summary.to_csv(date_path, index=False)
    month_summary.to_csv(month_path, index=False)
    hour_summary.to_csv(hour_path, index=False)
    fold_candidate.to_csv(fold_path, index=False)

    print("\n" + "=" * 100)
    print("DATE SUMMARY")
    print("=" * 100)
    cols = [
        "date", "rows", "start_time", "end_time",
        "PM2.5_mean", "PM2.5_median", "PM2.5_min", "PM2.5_max",
    ]
    cols = [c for c in cols if c in date_summary.columns]
    print(date_summary[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("MONTH SUMMARY")
    print("=" * 100)
    cols = [
        "month", "rows", "start_time", "end_time",
        "PM2.5_mean", "PM2.5_median", "PM2.5_min", "PM2.5_max",
    ]
    cols = [c for c in cols if c in month_summary.columns]
    print(month_summary[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("LEAVE-ONE-DATE DIFFICULTY CANDIDATES")
    print("=" * 100)
    print(fold_candidate.head(20).to_string(index=False))

    print("\nSaved:")
    print(date_path)
    print(month_path)
    print(hour_path)
    print(fold_path)


if __name__ == "__main__":
    main()