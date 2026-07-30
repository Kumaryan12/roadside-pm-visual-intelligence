from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


ROOT = Path("experiments/traqid_pretraining_v1")

MANIFEST = ROOT / "data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest_purged_block_compatible_clean.csv"

REPORT_OUT = ROOT / "reports/paper_diagnostics/purged_time_confound"
FIG_OUT = ROOT / "figures/paper_diagnostics/purged_time_confound"

REPORT_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)


def ks_summary(train_values, test_values):
    train_values = pd.Series(train_values).dropna()
    test_values = pd.Series(test_values).dropna()

    if len(train_values) == 0 or len(test_values) == 0:
        return {
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "train_mean": np.nan,
            "test_mean": np.nan,
            "train_std": np.nan,
            "test_std": np.nan,
        }

    ks = ks_2samp(train_values, test_values)

    return {
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "train_mean": float(train_values.mean()),
        "test_mean": float(test_values.mean()),
        "train_std": float(train_values.std()),
        "test_std": float(test_values.std()),
    }


def plot_hist(df, col, title, out_path, bins=24):
    plt.figure(figsize=(9, 5))

    for split in ["train", "val", "test"]:
        part = df[df["split_date_chrono"] == split]
        plt.hist(part[col].dropna(), bins=bins, alpha=0.45, density=True, label=split)

    plt.xlabel(col)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def main():
    df = pd.read_csv(MANIFEST)

    time_col = "target_time" if "target_time" in df.columns else "target_created_at"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df["hour"] = df[time_col].dt.hour
    df["minute"] = df[time_col].dt.minute
    df["time_of_day_float"] = df["hour"] + df["minute"] / 60.0
    df["is_day"] = ((df["hour"] >= 6) & (df["hour"] < 18)).astype(int)

    print("=" * 90)
    print("PURGED-BLOCK TIME / LOCAL-POSITION CONFOUND CHECK")
    print("=" * 90)
    print("Manifest:", MANIFEST)
    print("Shape:", df.shape)

    print("\nSplit counts:")
    print(df["split_date_chrono"].value_counts())

    # Basic split summaries
    summary_rows = []

    for split, g in df.groupby("split_date_chrono"):
        row = {
            "split": split,
            "n": int(len(g)),
            "num_dates": int(g["date"].nunique()),
            "time_min": str(g[time_col].min()),
            "time_max": str(g[time_col].max()),
            "hour_mean": float(g["time_of_day_float"].mean()),
            "hour_std": float(g["time_of_day_float"].std()),
            "day_fraction": float(g["is_day"].mean()),
        }

        if "local_sequence_pos" in g.columns:
            row.update({
                "local_pos_min": int(g["local_sequence_pos"].min()),
                "local_pos_max": int(g["local_sequence_pos"].max()),
                "local_pos_mean": float(g["local_sequence_pos"].mean()),
                "local_pos_std": float(g["local_sequence_pos"].std()),
            })

        summary_rows.append(row)

    split_summary = pd.DataFrame(summary_rows).sort_values("split")
    split_summary.to_csv(REPORT_OUT / "purged_split_time_summary.csv", index=False)

    print("\nSplit time summary:")
    print(split_summary.to_string(index=False))

    # Train vs test KS checks
    train = df[df["split_date_chrono"] == "train"]
    test = df[df["split_date_chrono"] == "test"]

    ks_rows = []

    for col in ["time_of_day_float", "hour", "is_day"]:
        s = ks_summary(train[col], test[col])
        s["feature"] = col
        ks_rows.append(s)

    if "local_sequence_pos" in df.columns:
        s = ks_summary(train["local_sequence_pos"], test["local_sequence_pos"])
        s["feature"] = "local_sequence_pos"
        ks_rows.append(s)

    ks_df = pd.DataFrame(ks_rows)
    ks_df = ks_df[["feature", "ks_stat", "ks_pvalue", "train_mean", "test_mean", "train_std", "test_std"]]
    ks_df.to_csv(REPORT_OUT / "purged_train_vs_test_time_ks.csv", index=False)

    print("\nTrain vs test KS/time-position shift:")
    print(ks_df.to_string(index=False))

    # Hour distribution by split
    hour_tab = pd.crosstab(df["hour"], df["split_date_chrono"], normalize="columns")
    hour_tab.to_csv(REPORT_OUT / "purged_hour_distribution_by_split.csv")

    daynight_tab = pd.crosstab(df["is_day"], df["split_date_chrono"], normalize="columns")
    daynight_tab.to_csv(REPORT_OUT / "purged_daynight_distribution_by_split.csv")

    print("\nHour distribution by split:")
    print(hour_tab.to_string())

    print("\nDay/night distribution by split:")
    print(daynight_tab.to_string())

    # Plots
    plot_hist(
        df,
        "time_of_day_float",
        "Purged-block split: time-of-day distribution",
        FIG_OUT / "purged_time_of_day_distribution.png",
        bins=24,
    )

    if "local_sequence_pos" in df.columns:
        plot_hist(
            df,
            "local_sequence_pos",
            "Purged-block split: local sequence position distribution",
            FIG_OUT / "purged_local_sequence_pos_distribution.png",
            bins=40,
        )

    # Target distributions by split too, quick sanity check
    target_cols = ["target_PM2.5", "target_PM10", "target_aqi"]
    target_summary_rows = []

    for target in target_cols:
        for split, g in df.groupby("split_date_chrono"):
            target_summary_rows.append({
                "target": target,
                "split": split,
                "n": int(len(g)),
                "mean": float(g[target].mean()),
                "std": float(g[target].std()),
                "min": float(g[target].min()),
                "max": float(g[target].max()),
            })

        plot_hist(
            df,
            target,
            f"Purged-block split: {target} distribution",
            FIG_OUT / f"purged_{target.replace('.', '').replace('_', '')}_distribution.png",
            bins=40,
        )

    target_summary = pd.DataFrame(target_summary_rows)
    target_summary.to_csv(REPORT_OUT / "purged_target_summary_by_split.csv", index=False)

    target_ks_rows = []
    for target in target_cols:
        s = ks_summary(train[target], test[target])
        s["target"] = target

        train_min = train[target].min()
        train_max = train[target].max()
        outside = ((test[target] < train_min) | (test[target] > train_max)).mean()

        s["test_outside_train_range_fraction"] = float(outside)
        s["train_min"] = float(train_min)
        s["train_max"] = float(train_max)
        s["test_min"] = float(test[target].min())
        s["test_max"] = float(test[target].max())

        target_ks_rows.append(s)

    target_ks = pd.DataFrame(target_ks_rows)
    target_ks = target_ks[
        [
            "target",
            "ks_stat",
            "ks_pvalue",
            "train_mean",
            "test_mean",
            "train_std",
            "test_std",
            "train_min",
            "train_max",
            "test_min",
            "test_max",
            "test_outside_train_range_fraction",
        ]
    ]
    target_ks.to_csv(REPORT_OUT / "purged_target_train_vs_test_ks.csv", index=False)

    print("\nTarget train vs test KS:")
    print(target_ks.to_string(index=False))

    # JSON summary for easy paper reading
    payload = {
        "manifest": str(MANIFEST),
        "split_summary": split_summary.to_dict(orient="records"),
        "time_ks": ks_df.to_dict(orient="records"),
        "target_ks": target_ks.to_dict(orient="records"),
    }

    (REPORT_OUT / "purged_time_confound_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSaved reports to:", REPORT_OUT)
    print("Saved figures to:", FIG_OUT)


if __name__ == "__main__":
    main()