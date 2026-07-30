from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("experiments/traqid_pretraining_v1")

MANIFEST = ROOT / "data/processed/traqid_paired_manifest_with_splits.csv"
SEQ_MANIFEST_DATE_SAFE = ROOT / "data/processed/traqid_T7_front_sequence_manifest.csv"
SEQ_MANIFEST_PURGED = ROOT / "data/processed/traqid_T7_front_sequence_manifest_purged_block_split.csv"

BEST_FUSION_HISTORY = (
    ROOT
    / "reports/t7_supervised_front_rear_mean_gru_tabular_fusion_purged_block"
    / "training_history_t7_gru_tabular_fusion.csv"
)

BEST_FUSION_PRED = (
    ROOT
    / "reports/t7_supervised_front_rear_mean_gru_tabular_fusion_purged_block"
    / "predictions_t7_gru_tabular_fusion.csv"
)

DATE_SAFE_PRED = (
    ROOT
    / "reports/t7_gru_tabular_fusion_date_safe_clipped"
    / "predictions_t7_gru_tabular_fusion.csv"
)

OUT_DIR = ROOT / "figures/ppt_final_traqid"


def setup():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def savefig(name: str):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("saved:", path)


def require(path: Path, label: str):
    if not path.exists():
        print(f"SKIP: missing {label}: {path}")
        return False
    return True


def plot_pm25_histogram(df: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    plt.hist(df["PM2.5"].astype(float), bins=50, edgecolor="black", alpha=0.8)
    plt.title("TRAQID PM2.5 distribution")
    plt.xlabel("PM2.5")
    plt.ylabel("Number of samples")
    plt.axvline(df["PM2.5"].mean(), linestyle="--", linewidth=2, label=f"Mean = {df['PM2.5'].mean():.1f}")
    plt.axvline(df["PM2.5"].median(), linestyle=":", linewidth=2, label=f"Median = {df['PM2.5'].median():.1f}")
    plt.legend()
    savefig("01_pm25_histogram.png")


def plot_aqi_category_counts(df: pd.DataFrame):
    if "aqi_cat" not in df.columns:
        return

    counts = df["aqi_cat"].value_counts()
    order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    counts = counts.reindex([x for x in order if x in counts.index])

    plt.figure(figsize=(9, 5))
    plt.bar(counts.index, counts.values)
    plt.title("TRAQID AQI category distribution")
    plt.xlabel("AQI category")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=20, ha="right")

    for i, v in enumerate(counts.values):
        plt.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)

    savefig("02_aqi_category_counts.png")


def plot_datewise_pm25(df: pd.DataFrame):
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["created_at"]).dt.date.astype(str)

    g = (
        df.groupby("date")
        .agg(
            pm25_mean=("PM2.5", "mean"),
            pm25_median=("PM2.5", "median"),
            rows=("PM2.5", "size"),
            split=("split_date_chrono", lambda x: x.mode().iloc[0] if len(x.mode()) else "unknown"),
        )
        .reset_index()
        .sort_values("date")
    )

    x = np.arange(len(g))

    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, g["pm25_mean"].values)

    # Use hatch patterns so this remains readable in black-and-white PPT.
    hatch_map = {"train": "", "val": "//", "test": "xx"}
    for bar, split in zip(bars, g["split"]):
        bar.set_hatch(hatch_map.get(split, ""))

    plt.title("Date-wise PM2.5 mean across TRAQID splits")
    plt.xlabel("Date")
    plt.ylabel("Mean PM2.5")
    plt.xticks(x, g["date"].astype(str), rotation=60, ha="right")

    # Legend using dummy bars.
    for split, hatch in hatch_map.items():
        plt.bar([], [], label=split, hatch=hatch)
    plt.legend(title="Split")

    savefig("03_datewise_pm25_mean_by_split.png")


def plot_split_boxplot(df: pd.DataFrame):
    if "split_date_chrono" not in df.columns:
        print("SKIP: no split_date_chrono column")
        return

    order = ["train", "val", "test"]
    data = [df.loc[df["split_date_chrono"] == s, "PM2.5"].astype(float).values for s in order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=order, showfliers=True)
    plt.title("PM2.5 distribution by date-safe split")
    plt.xlabel("Split")
    plt.ylabel("PM2.5")
    savefig("04_pm25_boxplot_by_split.png")


def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis("off")

    boxes = [
        ("Front image\nsequence", 0.05, 0.65),
        ("Rear image\nsequence", 0.05, 0.25),
        ("MobileNetV2\nPM-aware encoder", 0.28, 0.65),
        ("MobileNetV2\nPM-aware encoder", 0.28, 0.25),
        ("Front/rear\nmean fusion", 0.50, 0.45),
        ("T=7 GRU\ntemporal model", 0.67, 0.45),
        ("Tabular context\nTemp, RH, hour,\nseason, day/night", 0.67, 0.12),
        ("Fusion head", 0.83, 0.45),
        ("PM2.5\nprediction", 0.94, 0.45),
    ]

    for text, x, y in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=1.4),
            fontsize=11,
        )

    arrows = [
        ((0.14, 0.65), (0.22, 0.65)),
        ((0.14, 0.25), (0.22, 0.25)),
        ((0.38, 0.65), (0.46, 0.50)),
        ((0.38, 0.25), (0.46, 0.40)),
        ((0.57, 0.45), (0.62, 0.45)),
        ((0.74, 0.45), (0.79, 0.45)),
        ((0.74, 0.18), (0.82, 0.38)),
        ((0.87, 0.45), (0.91, 0.45)),
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.6),
        )

    ax.set_title("TRAQID MobileNetV2-GRU tabular fusion architecture", fontsize=16)
    savefig("05_model_architecture_diagram.png")


def plot_training_curve():
    if not require(BEST_FUSION_HISTORY, "best fusion training history"):
        return

    hist = pd.read_csv(BEST_FUSION_HISTORY)

    plt.figure(figsize=(10, 5))

    if "val_RMSE" in hist.columns:
        plt.plot(hist.index + 1, hist["val_RMSE"], marker="o", label="Validation RMSE")

    if "test_RMSE" in hist.columns:
        plt.plot(hist.index + 1, hist["test_RMSE"], marker="o", label="Test RMSE")

    plt.title("Training curve: best TRAQID fusion model")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    savefig("06_training_curve_best_fusion.png")


def plot_actual_vs_pred_purged():
    if not require(BEST_FUSION_PRED, "best fusion predictions"):
        return

    pred = pd.read_csv(BEST_FUSION_PRED)

    actual_col = "actual_PM25"
    pred_col = "predicted_PM25"

    if actual_col not in pred.columns or pred_col not in pred.columns:
        print("SKIP: prediction file missing actual_PM25/predicted_PM25 columns")
        return

    test = pred[pred["split"] == "test"].copy() if "split" in pred.columns else pred.copy()

    y = test[actual_col].astype(float).values
    p = test[pred_col].astype(float).values

    lo = min(y.min(), p.min())
    hi = max(y.max(), p.max())

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y, p, s=14, alpha=0.45)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    plt.title("Purged-block test: actual vs predicted PM2.5")
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    savefig("07_actual_vs_predicted_purged_test.png")


def plot_model_comparison_rmse():
    # Fallback values from our completed TRAQID runs.
    rows = [
        {
            "model": "Mean baseline\npurged",
            "rmse": 44.78,
            "r2": -0.060,
            "spearman": np.nan,
        },
        {
            "model": "Tabular-only\npurged",
            "rmse": 41.98,
            "r2": 0.068,
            "spearman": 0.314,
        },
        {
            "model": "Visual-only\nfront/rear mean",
            "rmse": 39.80,
            "r2": 0.163,
            "spearman": 0.363,
        },
        {
            "model": "Visual + tabular\nfusion",
            "rmse": 35.98,
            "r2": 0.316,
            "spearman": 0.470,
        },
        {
            "model": "Date-safe\nfusion",
            "rmse": 82.56,
            "r2": -2.977,
            "spearman": -0.346,
        },
    ]

    comp = pd.DataFrame(rows)

    plt.figure(figsize=(11, 5.8))
    bars = plt.bar(comp["model"], comp["rmse"])
    plt.title("TRAQID model comparison by test RMSE")
    plt.xlabel("Model / evaluation")
    plt.ylabel("Test RMSE")
    plt.xticks(rotation=15, ha="right")

    for bar, val in zip(bars, comp["rmse"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    savefig("08_model_comparison_test_rmse.png")

    out_csv = OUT_DIR / "model_comparison_values.csv"
    comp.to_csv(out_csv, index=False)
    print("saved:", out_csv)


def plot_model_comparison_r2_spearman():
    rows = [
        {"model": "Mean baseline\npurged", "R2": -0.060, "Spearman": np.nan},
        {"model": "Tabular-only\npurged", "R2": 0.068, "Spearman": 0.314},
        {"model": "Visual-only\nfront/rear mean", "R2": 0.163, "Spearman": 0.363},
        {"model": "Visual + tabular\nfusion", "R2": 0.316, "Spearman": 0.470},
        {"model": "Date-safe\nfusion", "R2": -2.977, "Spearman": -0.346},
    ]

    comp = pd.DataFrame(rows)

    x = np.arange(len(comp))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, comp["R2"], width, label="R²")
    plt.bar(x + width / 2, comp["Spearman"].fillna(0), width, label="Spearman")

    plt.axhline(0, linewidth=1)
    plt.title("TRAQID model comparison: R² and Spearman")
    plt.xlabel("Model / evaluation")
    plt.ylabel("Metric value")
    plt.xticks(x, comp["model"], rotation=15, ha="right")
    plt.legend()

    savefig("09_model_comparison_r2_spearman.png")


def plot_date_safe_actual_vs_predicted():
    if not require(DATE_SAFE_PRED, "date-safe predictions"):
        return

    if not require(SEQ_MANIFEST_DATE_SAFE, "date-safe sequence manifest"):
        return

    pred = pd.read_csv(DATE_SAFE_PRED)
    seq = pd.read_csv(SEQ_MANIFEST_DATE_SAFE)

    if "sequence_id" not in pred.columns or "sequence_id" not in seq.columns:
        print("SKIP: missing sequence_id column")
        return

    merged = pred.merge(seq[["sequence_id", "date"]], on="sequence_id", how="left")

    if "actual_PM25" not in merged.columns or "predicted_PM25" not in merged.columns:
        print("SKIP: date-safe predictions missing actual_PM25/predicted_PM25")
        return

    test = merged[merged["split"] == "test"].copy() if "split" in merged.columns else merged.copy()

    g = (
        test.groupby("date")
        .agg(
            actual_mean=("actual_PM25", "mean"),
            predicted_mean=("predicted_PM25", "mean"),
            n=("actual_PM25", "size"),
        )
        .reset_index()
        .sort_values("date")
    )

    x = np.arange(len(g))
    width = 0.36

    plt.figure(figsize=(9, 5.5))
    b1 = plt.bar(x - width / 2, g["actual_mean"], width, label="Actual mean")
    b2 = plt.bar(x + width / 2, g["predicted_mean"], width, label="Predicted mean")

    plt.title("Date-safe test: actual vs predicted PM2.5 by date")
    plt.xlabel("Test date")
    plt.ylabel("Mean PM2.5")
    plt.xticks(x, g["date"].astype(str), rotation=20, ha="right")
    plt.legend()

    for bars in [b1, b2]:
        for bar in bars:
            val = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    out_csv = OUT_DIR / "date_safe_per_date_actual_predicted.csv"
    g.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    savefig("10_date_safe_actual_vs_predicted_by_date.png")


def plot_front_rear_extraction_speed():
    rows = [
        {"view": "Front", "images_per_sec": 330.69, "time_sec": 80.67},
        {"view": "Rear", "images_per_sec": 333.25, "time_sec": 80.05},
    ]

    df = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(df["view"], df["images_per_sec"])
    plt.title("MobileNetV2 embedding extraction speed")
    plt.xlabel("View")
    plt.ylabel("Images per second")

    for bar, val in zip(bars, df["images_per_sec"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom")

    savefig("11_mobilenetv2_embedding_extraction_speed.png")


def main():
    setup()

    if not require(MANIFEST, "TRAQID paired manifest with splits"):
        return

    df = pd.read_csv(MANIFEST)

    if "created_at" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["created_at"], errors="coerce").dt.date.astype(str)

    print("TRAQID rows:", len(df))
    print("columns:", list(df.columns))

    plot_pm25_histogram(df)
    plot_aqi_category_counts(df)
    plot_datewise_pm25(df)
    plot_split_boxplot(df)
    plot_architecture_diagram()
    plot_training_curve()
    plot_actual_vs_pred_purged()
    plot_model_comparison_rmse()
    plot_model_comparison_r2_spearman()
    plot_date_safe_actual_vs_predicted()
    plot_front_rear_extraction_speed()

    print("\nDONE. PPT graphs saved to:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()