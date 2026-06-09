from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_hist(df: pd.DataFrame, col: str, out_path: Path, bins: int = 60) -> None:
    plt.figure(figsize=(8, 5))
    df[col].dropna().hist(bins=bins)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_box_by_category(df: pd.DataFrame, target: str, category: str, out_path: Path) -> None:
    groups = []
    labels = []

    for name, sub in df.groupby(category):
        values = sub[target].dropna()
        if len(values) > 0:
            groups.append(values)
            labels.append(str(name))

    if not groups:
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(f"{target} by {category}")
    plt.xlabel(category)
    plt.ylabel(target)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest.csv",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports",
    )

    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    report_dir = Path(args.report_dir)
    fig_dir = Path(args.fig_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df["created_at_parsed"] = pd.to_datetime(df["created_at"], errors="coerce")

    target_cols = [c for c in ["PM2.5", "PM10", "aqi", "Temperature", "Humidity"] if c in df.columns]

    # Full summary.
    numeric_summary = df[target_cols].describe().T
    numeric_summary.to_csv(report_dir / "traqid_paired_numeric_summary.csv")

    # Clean environmental subset.
    clean_env = df[
        df["Temperature"].between(-10, 60)
        & df["Humidity"].between(0, 100)
    ].copy()

    clean_env_summary = clean_env[target_cols].describe().T
    clean_env_summary.to_csv(report_dir / "traqid_paired_numeric_summary_clean_env.csv")

    # Timestamp duplicates / cadence.
    time_counts = (
        df.groupby("created_at")
        .size()
        .reset_index(name="rows_per_timestamp")
        .sort_values("rows_per_timestamp", ascending=False)
    )
    time_counts.to_csv(report_dir / "traqid_rows_per_timestamp.csv", index=False)

    # Hour/date counts.
    df["date"] = df["created_at_parsed"].dt.date.astype(str)
    df["hour_of_day"] = df["created_at_parsed"].dt.hour

    date_counts = df["date"].value_counts().sort_index()
    date_counts.to_csv(report_dir / "traqid_rows_per_date.csv")

    hour_counts = df["hour_of_day"].value_counts().sort_index()
    hour_counts.to_csv(report_dir / "traqid_rows_per_hour_of_day.csv")

    # Category summaries.
    for col in ["Season", "Day_or_Night", "aqi_cat"]:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            counts.to_csv(report_dir / f"traqid_{col}_counts.csv")

    # Correlation table.
    corr_cols = [c for c in ["PM2.5", "PM10", "aqi", "Temperature", "Humidity"] if c in df.columns]
    pearson = df[corr_cols].corr(method="pearson")
    spearman = df[corr_cols].corr(method="spearman")

    pearson.to_csv(report_dir / "traqid_pearson_correlations.csv")
    spearman.to_csv(report_dir / "traqid_spearman_correlations.csv")

    # Figures.
    for col in ["PM2.5", "PM10", "aqi", "Temperature", "Humidity"]:
        if col in df.columns:
            save_hist(df, col, fig_dir / f"hist_{col.replace('.', '_')}.png")

    for target in ["PM2.5", "PM10", "aqi"]:
        for cat in ["Season", "Day_or_Night", "aqi_cat"]:
            if target in df.columns and cat in df.columns:
                save_box_by_category(
                    df,
                    target,
                    cat,
                    fig_dir / f"box_{target.replace('.', '_')}_by_{cat}.png",
                )

    # Time-series plot sampled/aggregated hourly.
    hourly = (
        df.set_index("created_at_parsed")
        .sort_index()
        [["PM2.5", "PM10", "aqi"]]
        .resample("h")
        .mean()
    )

    plt.figure(figsize=(12, 5))
    for col in ["PM2.5", "PM10", "aqi"]:
        if col in hourly.columns:
            plt.plot(hourly.index, hourly[col], label=col)
    plt.title("Hourly mean PM/AQI over TRAQID")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "hourly_mean_pm_aqi_timeseries.png", dpi=160)
    plt.close()

    print("=" * 90)
    print("TRAQID PAIRED MANIFEST EDA COMPLETE")
    print("=" * 90)
    print("Rows:", len(df))
    print("Rows with plausible temperature/humidity:", len(clean_env), "/", len(df))

    print("\nNumeric summary:")
    print(numeric_summary.to_string())

    print("\nClean environmental numeric summary:")
    print(clean_env_summary.to_string())

    print("\nTop duplicated timestamps:")
    print(time_counts.head(20).to_string(index=False))

    print("\nPearson correlations:")
    print(pearson.to_string())

    print("\nSpearman correlations:")
    print(spearman.to_string())

    print("\nSaved reports to:", report_dir)
    print("Saved figures to:", fig_dir)


if __name__ == "__main__":
    main()