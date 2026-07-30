from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/traqid_pretraining_v1")

REPORT_OUT = ROOT / "reports/paper_final"
FIG_OUT = ROOT / "figures/paper_final"

REPORT_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text())


def metric_row(label: str, metrics_path: Path, split_note: str):
    d = load_json(metrics_path)
    test = d["test"]

    return {
        "Experiment": label,
        "Split": split_note,
        "Best epoch": d["best_checkpoint"]["epoch"],
        "PM2.5 R2": test["PM2.5"]["R2"],
        "PM2.5 RMSE": test["PM2.5"]["RMSE"],
        "PM10 R2": test["PM10"]["R2"],
        "PM10 RMSE": test["PM10"]["RMSE"],
        "AQI R2": test["AQI"]["R2"],
        "AQI RMSE": test["AQI"]["RMSE"],
        "Average R2": test["average"]["R2"],
        "Average RMSE": test["average"]["RMSE"],
        "metrics_path": str(metrics_path),
    }


def find_one(pattern: str, must_contain: list[str] | None = None, root: Path | None = None):
    root = root or ROOT / "reports"
    paths = sorted(root.rglob(pattern))

    if must_contain:
        paths = [p for p in paths if all(s in str(p) for s in must_contain)]

    if not paths:
        raise FileNotFoundError(f"No match for {pattern}, contains={must_contain}")

    if len(paths) > 1:
        print("\nMultiple matches found. Using first:")
        for p in paths:
            print(" -", p)

    return paths[0]


def save_table(df: pd.DataFrame, name: str):
    csv_path = REPORT_OUT / f"{name}.csv"
    md_path = REPORT_OUT / f"{name}.md"

    df.to_csv(csv_path, index=False)

    pretty = df.copy()
    for c in pretty.columns:
        if pretty[c].dtype.kind in "fc":
            pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")

    md_path.write_text(pretty.to_markdown(index=False))

    print("Saved:", csv_path)
    print("Saved:", md_path)


def make_ablation_table():
    rows = []

    # These folder names are based on your completed runs.
    candidates = [
        (
            "VGG16-LSTM front",
            ["paper_cnn_lstm", "traqid_paper_vgg16_front_gap", "split_random"],
            "Random 70/15/15",
        ),
        (
            "ResNet50-LSTM front",
            ["paper_cnn_lstm", "traqid_paper_resnet50_front_gap", "split_random"],
            "Random 70/15/15",
        ),
        (
            "ResNet50 front/rear mean LSTM",
            ["paper_cnn_lstm", "traqid_paper_resnet50_front_rear_mean_gap", "split_random"],
            "Random 70/15/15",
        ),
        (
            "ResNet50 front/rear concat LSTM",
            ["paper_cnn_lstm", "traqid_paper_resnet50_front_rear_concat_gap", "split_random"],
            "Random 70/15/15",
        ),
        (
            "ResNet50 front/rear concat LSTM + tabular fusion",
            ["paper_cnn_lstm_tabular_fusion", "traqid_paper_resnet50_front_rear_concat_gap", "split_random_tabular_fusion"],
            "Random 70/15/15",
        ),
    ]

    for label, contains, split_note in candidates:
        try:
            p = find_one("metrics_*.json", contains)
            rows.append(metric_row(label, p, split_note))
        except FileNotFoundError as e:
            print("Warning:", e)

    df = pd.DataFrame(rows)
    save_table(df, "final_ablation_table")
    return df


def make_split_comparison_table():
    rows = []

    # Best random split.
    random_p = find_one(
        "metrics_paper_cnn_lstm_tabular_fusion.json",
        ["paper_cnn_lstm_tabular_fusion", "traqid_paper_resnet50_front_rear_concat_gap", "split_random_tabular_fusion"],
    )
    rows.append(metric_row("Best enhanced model", random_p, "Random 70/15/15"))

    # Two-fold average from saved CSV.
    twofold_csv = ROOT / "reports/paper_cnn_lstm_tabular_fusion/twofold_average_summary.csv"
    if twofold_csv.exists():
        df2 = pd.read_csv(twofold_csv)
        avg = df2[
            ["PM25_R2", "PM25_RMSE", "PM10_R2", "PM10_RMSE", "AQI_R2", "AQI_RMSE", "Avg_R2", "Avg_RMSE"]
        ].mean()

        rows.append(
            {
                "Experiment": "Best enhanced model",
                "Split": "Two-fold CV average",
                "Best epoch": "-",
                "PM2.5 R2": avg["PM25_R2"],
                "PM2.5 RMSE": avg["PM25_RMSE"],
                "PM10 R2": avg["PM10_R2"],
                "PM10 RMSE": avg["PM10_RMSE"],
                "AQI R2": avg["AQI_R2"],
                "AQI RMSE": avg["AQI_RMSE"],
                "Average R2": avg["Avg_R2"],
                "Average RMSE": avg["Avg_RMSE"],
                "metrics_path": str(twofold_csv),
            }
        )
    else:
        print("Warning: twofold average CSV not found:", twofold_csv)

    # Chrono/date-safe.
    chrono_p = find_one(
        "metrics_paper_cnn_lstm_tabular_fusion.json",
        ["paper_cnn_lstm_tabular_fusion", "traqid_paper_resnet50_front_rear_concat_gap", "split_chrono_date_tabular_fusion"],
    )
    rows.append(metric_row("Best enhanced model", chrono_p, "Chronological date-wise"))

    df = pd.DataFrame(rows)
    save_table(df, "final_split_comparison_table")
    return df


def make_t_sweep_table_and_plots():
    ts_path = ROOT / "reports/paper_cnn_lstm_tabular_fusion/t_sweep_summary/t_sweep_random_split_summary.csv"

    if not ts_path.exists():
        raise FileNotFoundError(f"T-sweep summary not found: {ts_path}")

    df = pd.read_csv(ts_path)
    df = df.sort_values("T")

    keep = [
        "T",
        "best_epoch",
        "test_PM25_R2",
        "test_PM25_RMSE",
        "test_PM10_R2",
        "test_PM10_RMSE",
        "test_AQI_R2",
        "test_AQI_RMSE",
        "test_Avg_R2",
        "test_Avg_RMSE",
    ]

    out = df[keep].copy()
    save_table(out, "final_t_sweep_table")

    plt.figure(figsize=(9, 5))
    plt.plot(df["T"], df["test_PM25_R2"], marker="o", label="PM2.5 R²")
    plt.plot(df["T"], df["test_PM10_R2"], marker="o", label="PM10 R²")
    plt.plot(df["T"], df["test_AQI_R2"], marker="o", label="AQI R²")
    plt.plot(df["T"], df["test_Avg_R2"], marker="o", linestyle="--", label="Average R²")
    plt.xlabel("Sequence length T")
    plt.ylabel("Test R²")
    plt.title("Sequence-length sweep: test R² vs T")
    plt.xticks(df["T"])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_fig = FIG_OUT / "final_t_sweep_test_r2.png"
    plt.savefig(out_fig, dpi=240)
    plt.close()
    print("Saved:", out_fig)

    plt.figure(figsize=(9, 5))
    plt.plot(df["T"], df["test_PM25_RMSE"], marker="o", label="PM2.5 RMSE")
    plt.plot(df["T"], df["test_PM10_RMSE"], marker="o", label="PM10 RMSE")
    plt.plot(df["T"], df["test_AQI_RMSE"], marker="o", label="AQI RMSE")
    plt.plot(df["T"], df["test_Avg_RMSE"], marker="o", linestyle="--", label="Average RMSE")
    plt.xlabel("Sequence length T")
    plt.ylabel("Test RMSE")
    plt.title("Sequence-length sweep: test RMSE vs T")
    plt.xticks(df["T"])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_fig = FIG_OUT / "final_t_sweep_test_rmse.png"
    plt.savefig(out_fig, dpi=240)
    plt.close()
    print("Saved:", out_fig)

    return df


def make_split_comparison_plot(split_df: pd.DataFrame):
    labels = split_df["Split"].tolist()

    x = range(len(split_df))

    plt.figure(figsize=(9, 5))
    plt.bar(x, split_df["PM2.5 R2"])
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("PM2.5 R²")
    plt.title("Validation protocol sensitivity: PM2.5 R²")
    plt.axhline(0, linewidth=1)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_fig = FIG_OUT / "final_split_sensitivity_pm25_r2.png"
    plt.savefig(out_fig, dpi=240)
    plt.close()
    print("Saved:", out_fig)

    plt.figure(figsize=(9, 5))
    plt.bar(x, split_df["Average R2"])
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Average R² across PM2.5, PM10, AQI")
    plt.title("Validation protocol sensitivity: average R²")
    plt.axhline(0, linewidth=1)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_fig = FIG_OUT / "final_split_sensitivity_average_r2.png"
    plt.savefig(out_fig, dpi=240)
    plt.close()
    print("Saved:", out_fig)


def make_architecture_text_diagram():
    text = """
Final enhanced architecture:

Front image sequence (T=7) ─┐
                            ├─ ResNet50 GAP feature extraction ─┐
Rear image sequence (T=7)  ─┘                                    │
                                                                 ↓
                                                    Front/rear concatenated features
                                                                 ↓
                                                       LSTM temporal encoder
                                                                 ↓
Temperature, Humidity, hour_sin/hour_cos, Season, Day/Night ─→ Tabular MLP
                                                                 ↓
                                                    Fusion MLP regression head
                                                                 ↓
                                                 PM2.5, PM10, AQI predictions
""".strip()

    out = REPORT_OUT / "final_architecture_diagram_text.txt"
    out.write_text(text)
    print("Saved:", out)


def make_final_summary(ablation_df, split_df, ts_df):
    best_random = split_df[split_df["Split"] == "Random 70/15/15"].iloc[0]
    twofold = split_df[split_df["Split"] == "Two-fold CV average"].iloc[0]
    chrono = split_df[split_df["Split"] == "Chronological date-wise"].iloc[0]
    best_t = ts_df.loc[ts_df["test_Avg_R2"].idxmax()]

    summary = f"""
# Final Paper Experiment Summary

## Best random-split model

ResNet50 front/rear concat LSTM + tabular fusion achieved:

- PM2.5 R² = {best_random["PM2.5 R2"]:.4f}, RMSE = {best_random["PM2.5 RMSE"]:.4f}
- PM10 R² = {best_random["PM10 R2"]:.4f}, RMSE = {best_random["PM10 RMSE"]:.4f}
- AQI R² = {best_random["AQI R2"]:.4f}, RMSE = {best_random["AQI RMSE"]:.4f}
- Average R² = {best_random["Average R2"]:.4f}

## Two-fold CV average

- PM2.5 R² = {twofold["PM2.5 R2"]:.4f}
- PM10 R² = {twofold["PM10 R2"]:.4f}
- AQI R² = {twofold["AQI R2"]:.4f}
- Average R² = {twofold["Average R2"]:.4f}

## Chronological date-wise evaluation

- PM2.5 R² = {chrono["PM2.5 R2"]:.4f}
- PM10 R² = {chrono["PM10 R2"]:.4f}
- AQI R² = {chrono["AQI R2"]:.4f}
- Average R² = {chrono["Average R2"]:.4f}

## Sequence length

The T-sweep from T=2 to T=9 selected T={int(best_t["T"])} as the best overall sequence length by average R².

## Core conclusion

The enhanced CNN-LSTM model reaches paper-level random-split performance on TRAQID, but chronological date-wise testing fails, showing that high random-split R² does not guarantee unseen-date generalization.
""".strip()

    out = REPORT_OUT / "final_paper_experiment_summary.md"
    out.write_text(summary)
    print("Saved:", out)


def main():
    print("=" * 90)
    print("MAKING FINAL PAPER ASSETS")
    print("=" * 90)

    ablation_df = make_ablation_table()
    split_df = make_split_comparison_table()
    ts_df = make_t_sweep_table_and_plots()
    make_split_comparison_plot(split_df)
    make_architecture_text_diagram()
    make_final_summary(ablation_df, split_df, ts_df)

    print("\nDONE")
    print("Reports:", REPORT_OUT)
    print("Figures:", FIG_OUT)


if __name__ == "__main__":
    main()