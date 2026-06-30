from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/traqid_pretraining_v1")
REPORT_OUT = ROOT / "reports/paper_final"
FIG_OUT = ROOT / "figures/paper_final"

REPORT_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)


def load_metrics(label: str, split: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    d = json.loads(path.read_text())
    test = d["test"]

    return {
        "Experiment": label,
        "Split": split,
        "Best epoch": d["best_checkpoint"]["epoch"],
        "PM2.5 R2": test["PM2.5"]["R2"],
        "PM2.5 RMSE": test["PM2.5"]["RMSE"],
        "PM10 R2": test["PM10"]["R2"],
        "PM10 RMSE": test["PM10"]["RMSE"],
        "AQI R2": test["AQI"]["R2"],
        "AQI RMSE": test["AQI"]["RMSE"],
        "Average R2": test["average"]["R2"],
        "Average RMSE": test["average"]["RMSE"],
        "metrics_path": str(path),
    }


def save_table(df: pd.DataFrame, name: str) -> None:
    csv_path = REPORT_OUT / f"{name}.csv"
    md_path = REPORT_OUT / f"{name}.md"

    df.to_csv(csv_path, index=False)

    pretty = df.copy()
    for c in pretty.columns:
        if pretty[c].dtype.kind in "fc":
            pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")

    md_path.write_text(pretty.to_markdown(index=False), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


def main() -> None:
    # ---------------------------------------------------------------------
    # Exact metrics paths.
    # These avoid accidental selection of T2/T3/... sweep runs.
    # ---------------------------------------------------------------------

    vgg16_front = (
        ROOT
        / "reports/paper_cnn_lstm"
        / "traqid_paper_vgg16_front_gap_traqid_paper_style_T7_front_sequence_manifest_split_random"
        / "metrics_paper_cnn_lstm_multitarget.json"
    )

    resnet50_front = (
        ROOT
        / "reports/paper_cnn_lstm"
        / "traqid_paper_resnet50_front_gap_traqid_paper_style_T7_front_sequence_manifest_split_random"
        / "metrics_paper_cnn_lstm_multitarget.json"
    )

    resnet50_front_rear_mean = (
        ROOT
        / "reports/paper_cnn_lstm"
        / "traqid_paper_resnet50_front_rear_mean_gap_traqid_paper_style_T7_front_sequence_manifest_split_random"
        / "metrics_paper_cnn_lstm_multitarget.json"
    )

    resnet50_front_rear_concat = (
        ROOT
        / "reports/paper_cnn_lstm"
        / "traqid_paper_resnet50_front_rear_concat_gap_traqid_paper_style_T7_front_sequence_manifest_split_random"
        / "metrics_paper_cnn_lstm_multitarget.json"
    )

    # IMPORTANT:
    # This is the restored 80-epoch final run, not the overwritten 50-epoch T-sweep run.
    resnet50_front_rear_concat_tabular_final80 = (
        ROOT
        / "reports/paper_cnn_lstm_tabular_fusion_final80"
        / "traqid_paper_resnet50_front_rear_concat_gap_traqid_paper_style_T7_front_sequence_manifest_split_random_tabular_fusion"
        / "metrics_paper_cnn_lstm_tabular_fusion.json"
    )

    chrono_date = (
        ROOT
        / "reports/paper_cnn_lstm_tabular_fusion"
        / "traqid_paper_resnet50_front_rear_concat_gap_traqid_paper_style_T7_front_sequence_manifest_split_chrono_date_tabular_fusion"
        / "metrics_paper_cnn_lstm_tabular_fusion.json"
    )

    twofold_csv = (
        ROOT
        / "reports/paper_cnn_lstm_tabular_fusion"
        / "twofold_average_summary.csv"
    )

    t_sweep_csv = (
        ROOT
        / "reports/paper_cnn_lstm_tabular_fusion"
        / "t_sweep_summary"
        / "t_sweep_random_split_summary.csv"
    )

    # ---------------------------------------------------------------------
    # 1. Ablation table
    # ---------------------------------------------------------------------

    ablation_rows = [
        load_metrics("VGG16-LSTM front", "Random 70/15/15", vgg16_front),
        load_metrics("ResNet50-LSTM front", "Random 70/15/15", resnet50_front),
        load_metrics("ResNet50 front/rear mean LSTM", "Random 70/15/15", resnet50_front_rear_mean),
        load_metrics("ResNet50 front/rear concat LSTM", "Random 70/15/15", resnet50_front_rear_concat),
        load_metrics(
            "ResNet50 front/rear concat LSTM + tabular fusion",
            "Random 70/15/15",
            resnet50_front_rear_concat_tabular_final80,
        ),
    ]

    ablation_df = pd.DataFrame(ablation_rows)
    save_table(ablation_df, "final_ablation_table")

    # ---------------------------------------------------------------------
    # 2. Split comparison table
    # ---------------------------------------------------------------------

    split_rows = [
        load_metrics(
            "Best enhanced model",
            "Random 70/15/15",
            resnet50_front_rear_concat_tabular_final80,
        )
    ]

    if not twofold_csv.exists():
        raise FileNotFoundError(f"Missing two-fold summary: {twofold_csv}")

    twofold_df = pd.read_csv(twofold_csv)
    avg = twofold_df[
        [
            "PM25_R2",
            "PM25_RMSE",
            "PM10_R2",
            "PM10_RMSE",
            "AQI_R2",
            "AQI_RMSE",
            "Avg_R2",
            "Avg_RMSE",
        ]
    ].mean()

    split_rows.append(
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

    split_rows.append(
        load_metrics(
            "Best enhanced model",
            "Chronological date-wise",
            chrono_date,
        )
    )

    split_df = pd.DataFrame(split_rows)
    save_table(split_df, "final_split_comparison_table")

    # ---------------------------------------------------------------------
    # 3. T-sweep table
    # ---------------------------------------------------------------------

    if not t_sweep_csv.exists():
        raise FileNotFoundError(f"Missing T-sweep summary: {t_sweep_csv}")

    t_df = pd.read_csv(t_sweep_csv).sort_values("T")

    t_keep = [
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

    save_table(t_df[t_keep], "final_t_sweep_table")

    # ---------------------------------------------------------------------
    # 4. Plots
    # ---------------------------------------------------------------------

    plt.figure(figsize=(9, 5))
    plt.plot(t_df["T"], t_df["test_PM25_R2"], marker="o", label="PM2.5 R²")
    plt.plot(t_df["T"], t_df["test_PM10_R2"], marker="o", label="PM10 R²")
    plt.plot(t_df["T"], t_df["test_AQI_R2"], marker="o", label="AQI R²")
    plt.plot(t_df["T"], t_df["test_Avg_R2"], marker="o", linestyle="--", label="Average R²")
    plt.xlabel("Sequence length T")
    plt.ylabel("Test R²")
    plt.title("Sequence-length sweep: test R² vs T")
    plt.xticks(t_df["T"])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = FIG_OUT / "final_t_sweep_test_r2.png"
    plt.savefig(out, dpi=240)
    plt.close()
    print(f"Saved: {out}")

    plt.figure(figsize=(9, 5))
    plt.plot(t_df["T"], t_df["test_PM25_RMSE"], marker="o", label="PM2.5 RMSE")
    plt.plot(t_df["T"], t_df["test_PM10_RMSE"], marker="o", label="PM10 RMSE")
    plt.plot(t_df["T"], t_df["test_AQI_RMSE"], marker="o", label="AQI RMSE")
    plt.plot(t_df["T"], t_df["test_Avg_RMSE"], marker="o", linestyle="--", label="Average RMSE")
    plt.xlabel("Sequence length T")
    plt.ylabel("Test RMSE")
    plt.title("Sequence-length sweep: test RMSE vs T")
    plt.xticks(t_df["T"])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = FIG_OUT / "final_t_sweep_test_rmse.png"
    plt.savefig(out, dpi=240)
    plt.close()
    print(f"Saved: {out}")

    plt.figure(figsize=(9, 5))
    plt.bar(split_df["Split"], split_df["PM2.5 R2"])
    plt.axhline(0, linewidth=1)
    plt.ylabel("PM2.5 R²")
    plt.title("Validation protocol sensitivity: PM2.5 R²")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = FIG_OUT / "final_split_sensitivity_pm25_r2.png"
    plt.savefig(out, dpi=240)
    plt.close()
    print(f"Saved: {out}")

    plt.figure(figsize=(9, 5))
    plt.bar(split_df["Split"], split_df["Average R2"])
    plt.axhline(0, linewidth=1)
    plt.ylabel("Average R² across PM2.5, PM10, AQI")
    plt.title("Validation protocol sensitivity: average R²")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = FIG_OUT / "final_split_sensitivity_average_r2.png"
    plt.savefig(out, dpi=240)
    plt.close()
    print(f"Saved: {out}")

    # ---------------------------------------------------------------------
    # 5. Architecture text diagram
    # ---------------------------------------------------------------------

    architecture = """
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

    arch_path = REPORT_OUT / "final_architecture_diagram_text.txt"
    arch_path.write_text(architecture, encoding="utf-8")
    print(f"Saved: {arch_path}")

    # ---------------------------------------------------------------------
    # 6. Final summary markdown
    # ---------------------------------------------------------------------

    best = split_df[split_df["Split"] == "Random 70/15/15"].iloc[0]
    two = split_df[split_df["Split"] == "Two-fold CV average"].iloc[0]
    chrono = split_df[split_df["Split"] == "Chronological date-wise"].iloc[0]
    best_t = t_df.loc[t_df["test_Avg_R2"].idxmax()]

    summary = f"""# Final Paper Experiment Summary

## Best random-split model

ResNet50 front/rear concat LSTM + tabular fusion achieved:

- PM2.5 R² = {best["PM2.5 R2"]:.4f}, RMSE = {best["PM2.5 RMSE"]:.4f}
- PM10 R² = {best["PM10 R2"]:.4f}, RMSE = {best["PM10 RMSE"]:.4f}
- AQI R² = {best["AQI R2"]:.4f}, RMSE = {best["AQI RMSE"]:.4f}
- Average R² = {best["Average R2"]:.4f}, RMSE = {best["Average RMSE"]:.4f}

## Two-fold CV average

- PM2.5 R² = {two["PM2.5 R2"]:.4f}, RMSE = {two["PM2.5 RMSE"]:.4f}
- PM10 R² = {two["PM10 R2"]:.4f}, RMSE = {two["PM10 RMSE"]:.4f}
- AQI R² = {two["AQI R2"]:.4f}, RMSE = {two["AQI RMSE"]:.4f}
- Average R² = {two["Average R2"]:.4f}, RMSE = {two["Average RMSE"]:.4f}

## Chronological date-wise evaluation

- PM2.5 R² = {chrono["PM2.5 R2"]:.4f}, RMSE = {chrono["PM2.5 RMSE"]:.4f}
- PM10 R² = {chrono["PM10 R2"]:.4f}, RMSE = {chrono["PM10 RMSE"]:.4f}
- AQI R² = {chrono["AQI R2"]:.4f}, RMSE = {chrono["AQI RMSE"]:.4f}
- Average R² = {chrono["Average R2"]:.4f}, RMSE = {chrono["Average RMSE"]:.4f}

## Sequence length

The T-sweep from T=2 to T=9 selected T={int(best_t["T"])} as the best overall sequence length by average R².

## Core conclusion

The enhanced CNN-LSTM model reaches paper-level random-split performance on TRAQID, but chronological date-wise testing fails, showing that high random-split R² does not guarantee unseen-date generalization.
"""

    summary_path = REPORT_OUT / "final_paper_experiment_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Saved: {summary_path}")

    print("\nCorrected final summary:")
    print(summary)


if __name__ == "__main__":
    main()