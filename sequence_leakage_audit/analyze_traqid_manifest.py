import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POLLUTANT_COLS = ["PM2.5", "PM10", "aqi"]
NUMERIC_CONTEXT_COLS = ["Temperature", "Humidity"]
CATEGORICAL_COLS = ["Season", "Day_or_Night", "aqi_cat"]


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_md(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append({
            "column": col,
            "missing_count": missing,
            "missing_frac": missing / len(df) if len(df) else np.nan,
            "dtype": str(df[col].dtype),
        })
    return pd.DataFrame(rows).sort_values(["missing_count", "column"], ascending=[False, True])


def plot_bar(series, title, xlabel, ylabel, out_path, top_n=None, rotate=45):
    s = series.copy()
    if top_n is not None:
        s = s.head(top_n)

    plt.figure(figsize=(10, 5))
    s.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_hist(df, col, out_path, bins=50):
    plt.figure(figsize=(8, 5))
    df[col].dropna().plot(kind="hist", bins=bins)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_line(series, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(11, 5))
    series.plot()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_scatter(df, x, y, out_path):
    plt.figure(figsize=(7, 5))
    plt.scatter(df[x], df[y], s=8, alpha=0.35)
    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Detailed EDA/audit for TRAQID paired manifest."
    )
    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
        help="Path to TRAQID paired manifest CSV.",
    )
    parser.add_argument(
        "--timestamp-col",
        default="created_at_parsed",
        help="Timestamp column to parse.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/traqid_eda",
        help="Output directory for EDA report.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"

    safe_mkdir(out_dir)
    safe_mkdir(fig_dir)
    safe_mkdir(table_dir)

    df = pd.read_csv(manifest_path)

    if args.timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column {args.timestamp_col} not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], errors="coerce")

    if "date" not in df.columns:
        df["date"] = df[args.timestamp_col].dt.date.astype(str)
    else:
        df["date"] = df["date"].astype(str)

    # Robust hour extraction.
# In TRAQID, "hour" may be stored as a timestamp-like string, a datetime string,
# or a numeric hour. Avoid np.issubdtype because pandas StringDtype can fail there.
    if "hour" not in df.columns:
        df["hour_numeric"] = df[args.timestamp_col].dt.hour
    else:
        hour_numeric_direct = pd.to_numeric(df["hour"], errors="coerce")

        # If the column is genuinely numeric hour values such as 0, 1, ..., 23.
        if hour_numeric_direct.notna().sum() > 0 and hour_numeric_direct.dropna().between(0, 23).all():
            df["hour_numeric"] = hour_numeric_direct.astype("Int64")
        else:
            # If the column is timestamp-like, parse it.
            hour_as_datetime = pd.to_datetime(df["hour"], errors="coerce")
            df["hour_numeric"] = hour_as_datetime.dt.hour

            # Fallback to created_at_parsed.
            if df["hour_numeric"].isna().all():
                df["hour_numeric"] = df[args.timestamp_col].dt.hour

    # ------------------------------------------------------------------
    # Basic tables
    # ------------------------------------------------------------------
    basic = {
        "manifest_path": str(manifest_path),
        "rows": len(df),
        "columns": len(df.columns),
        "timestamp_col": args.timestamp_col,
        "valid_timestamps": int(df[args.timestamp_col].notna().sum()),
        "unique_timestamps": int(df[args.timestamp_col].nunique()),
        "unique_dates": int(df["date"].nunique()),
        "start_time": str(df[args.timestamp_col].min()),
        "end_time": str(df[args.timestamp_col].max()),
    }

    pd.DataFrame([basic]).to_csv(table_dir / "basic_summary.csv", index=False)

    missing = describe_missing(df)
    missing.to_csv(table_dir / "missingness.csv", index=False)

    # ------------------------------------------------------------------
    # Timestamp-level aggregation
    # ------------------------------------------------------------------
    timestamp_counts = (
        df.groupby(args.timestamp_col)
        .size()
        .rename("num_rows_at_timestamp")
        .reset_index()
        .sort_values(args.timestamp_col)
    )

    timestamp_counts.to_csv(table_dir / "timestamp_counts.csv", index=False)

    images_per_timestamp_summary = timestamp_counts["num_rows_at_timestamp"].describe()
    images_per_timestamp_summary.to_csv(table_dir / "images_per_timestamp_summary.csv")

    # Time gaps between unique timestamps
    unique_times = timestamp_counts[args.timestamp_col].dropna().sort_values()
    gaps = unique_times.diff().dropna()
    gaps_minutes = gaps.dt.total_seconds() / 60.0

    gap_summary = gaps_minutes.describe().to_frame("gap_minutes")
    gap_summary.to_csv(table_dir / "timestamp_gap_summary_minutes.csv")

    gap_table = pd.DataFrame({
        "timestamp": unique_times.iloc[1:].values,
        "gap_minutes_from_previous": gaps_minutes.values,
    })
    gap_table.to_csv(table_dir / "timestamp_gaps.csv", index=False)

    # ------------------------------------------------------------------
    # Date/hour/category distributions
    # ------------------------------------------------------------------
    rows_per_date = df["date"].value_counts().sort_index()
    rows_per_date.to_csv(table_dir / "rows_per_date.csv")

    timestamps_per_date = (
        timestamp_counts.assign(date=timestamp_counts[args.timestamp_col].dt.date.astype(str))
        .groupby("date")
        .size()
        .rename("unique_timestamps")
    )
    timestamps_per_date.to_csv(table_dir / "unique_timestamps_per_date.csv")

    hour_dist = df["hour_numeric"].value_counts().sort_index()
    hour_dist.to_csv(table_dir / "rows_by_hour.csv")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col].value_counts(dropna=False).to_csv(table_dir / f"{col}_counts.csv")

    # ------------------------------------------------------------------
    # Numeric summaries
    # ------------------------------------------------------------------
    numeric_cols = [c for c in POLLUTANT_COLS + NUMERIC_CONTEXT_COLS if c in df.columns]
    numeric_summary = df[numeric_cols].describe().T
    numeric_summary.to_csv(table_dir / "numeric_summary.csv")

    if numeric_cols:
        corr = df[numeric_cols].corr()
        corr.to_csv(table_dir / "numeric_correlations.csv")

    # ------------------------------------------------------------------
    # Split summaries
    # ------------------------------------------------------------------
    split_cols = [c for c in df.columns if c.startswith("split") or "split" in c.lower()]
    split_summary_rows = []

    for split_col in split_cols:
        counts = df[split_col].value_counts(dropna=False)
        for name, count in counts.items():
            split_summary_rows.append({
                "split_col": split_col,
                "split_value": name,
                "rows": int(count),
                "fraction": count / len(df),
            })

        if numeric_cols:
            grouped = df.groupby(split_col)[numeric_cols].agg(["count", "mean", "std", "min", "max"])
            grouped.to_csv(table_dir / f"{split_col}_numeric_by_split.csv")

    split_summary = pd.DataFrame(split_summary_rows)
    if not split_summary.empty:
        split_summary.to_csv(table_dir / "split_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Image availability
    # ------------------------------------------------------------------
    image_availability = []

    for col in ["front_exists", "rear_exists"]:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            for val, count in vc.items():
                image_availability.append({
                    "column": col,
                    "value": val,
                    "count": int(count),
                    "fraction": count / len(df),
                })

    image_availability_df = pd.DataFrame(image_availability)
    if not image_availability_df.empty:
        image_availability_df.to_csv(table_dir / "image_availability.csv", index=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_bar(
        rows_per_date,
        title="Rows per date",
        xlabel="Date",
        ylabel="Number of image rows",
        out_path=fig_dir / "rows_per_date.png",
        rotate=60,
    )

    plot_bar(
        timestamps_per_date,
        title="Unique timestamps per date",
        xlabel="Date",
        ylabel="Unique timestamps",
        out_path=fig_dir / "unique_timestamps_per_date.png",
        rotate=60,
    )

    plot_bar(
        hour_dist,
        title="Rows by hour of day",
        xlabel="Hour",
        ylabel="Number of image rows",
        out_path=fig_dir / "rows_by_hour.png",
        rotate=0,
    )

    plot_hist(
        timestamp_counts,
        "num_rows_at_timestamp",
        fig_dir / "images_per_timestamp_hist.png",
        bins=30,
    )

    if len(gaps_minutes) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(gaps_minutes.clip(upper=np.nanpercentile(gaps_minutes, 99)), bins=50)
        plt.title("Timestamp gap distribution, clipped at 99th percentile")
        plt.xlabel("Gap from previous timestamp, minutes")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(fig_dir / "timestamp_gap_distribution_clipped.png", dpi=220)
        plt.close()

    for col in numeric_cols:
        plot_hist(df, col, fig_dir / f"distribution_{col.replace('.', '_')}.png", bins=60)

    if "PM2.5" in df.columns and "PM10" in df.columns:
        plot_scatter(df, "PM2.5", "PM10", fig_dir / "scatter_PM10_vs_PM25.png")

    if "PM2.5" in df.columns and "aqi" in df.columns:
        plot_scatter(df, "PM2.5", "aqi", fig_dir / "scatter_aqi_vs_PM25.png")

    if "Humidity" in df.columns and "PM2.5" in df.columns:
        plot_scatter(df, "Humidity", "PM2.5", fig_dir / "scatter_PM25_vs_Humidity.png")

    if "Temperature" in df.columns and "PM2.5" in df.columns:
        plot_scatter(df, "Temperature", "PM2.5", fig_dir / "scatter_PM25_vs_Temperature.png")

    # Time series over timestamp-level average
    if numeric_cols:
        ts_avg = (
            df.groupby(args.timestamp_col)[numeric_cols]
            .mean()
            .sort_index()
        )
        ts_avg.to_csv(table_dir / "timestamp_level_numeric_means.csv")

        for col in numeric_cols:
            plot_line(
                ts_avg[col],
                title=f"Timestamp-level mean {col}",
                xlabel="Timestamp",
                ylabel=col,
                out_path=fig_dir / f"timeseries_{col.replace('.', '_')}.png",
            )

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    report = []
    report.append("# TRAQID Manifest EDA Report\n")
    report.append("## Basic summary\n")
    for k, v in basic.items():
        report.append(f"- **{k}**: {v}")
    report.append("\n")

    report.append("## Key timestamp findings\n")
    report.append(f"- Total image rows: **{len(df)}**")
    report.append(f"- Unique timestamps: **{df[args.timestamp_col].nunique()}**")
    report.append(f"- Unique dates: **{df['date'].nunique()}**")
    report.append(f"- Start time: **{df[args.timestamp_col].min()}**")
    report.append(f"- End time: **{df[args.timestamp_col].max()}**")
    report.append("\n")

    report.append("## Images per timestamp\n")
    report.append(images_per_timestamp_summary.to_markdown())
    report.append("\n")

    report.append("## Timestamp gaps, minutes\n")
    report.append(gap_summary.to_markdown())
    report.append("\n")

    report.append("## Numeric summary\n")
    if numeric_cols:
        report.append(numeric_summary.to_markdown())
    else:
        report.append("No numeric pollutant/context columns found.")
    report.append("\n")

    report.append("## Correlations\n")
    if numeric_cols:
        report.append(corr.to_markdown())
    else:
        report.append("No numeric pollutant/context columns found.")
    report.append("\n")

    report.append("## Split columns detected\n")
    if split_cols:
        report.append(", ".join(split_cols))
    else:
        report.append("No split columns detected.")
    report.append("\n")

    report.append("## Generated figures\n")
    for fig in sorted(fig_dir.glob("*.png")):
        report.append(f"- `{fig}`")
    report.append("\n")

    save_md(out_dir / "traqid_eda_report.md", "\n".join(report))

    print("=" * 90)
    print("TRAQID EDA COMPLETE")
    print("=" * 90)
    print("Manifest:", manifest_path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Unique timestamps:", df[args.timestamp_col].nunique())
    print("Unique dates:", df["date"].nunique())
    print("Start:", df[args.timestamp_col].min())
    print("End:", df[args.timestamp_col].max())
    print()
    print("Images per timestamp:")
    print(images_per_timestamp_summary.to_string())
    print()
    print("Timestamp gap summary, minutes:")
    print(gap_summary.to_string())
    print()
    print("Numeric summary:")
    print(numeric_summary.to_string() if numeric_cols else "No numeric columns found.")
    print()
    print("Saved report:", out_dir / "traqid_eda_report.md")
    print("Saved tables:", table_dir)
    print("Saved figures:", fig_dir)


if __name__ == "__main__":
    main()