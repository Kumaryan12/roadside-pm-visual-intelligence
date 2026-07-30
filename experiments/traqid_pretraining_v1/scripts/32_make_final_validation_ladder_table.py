from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("experiments/traqid_pretraining_v1")
OUT_DIR = ROOT / "reports/paper_final"
FIG_DIR = ROOT / "figures/paper_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_test_metrics(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    d = json.loads(path.read_text())
    return d["test"]


def row(protocol, metrics_path, overlap_ge6, overlap_ge1, note):
    test = load_test_metrics(metrics_path)
    return {
        "Protocol": protocol,
        "Test seq overlap >=1 frame": overlap_ge1,
        "Test seq overlap >=6/7 frames": overlap_ge6,
        "PM2.5 R2": test["PM2.5"]["R2"],
        "PM2.5 RMSE": test["PM2.5"]["RMSE"],
        "PM10 R2": test["PM10"]["R2"],
        "PM10 RMSE": test["PM10"]["RMSE"],
        "AQI R2": test["AQI"]["R2"],
        "AQI RMSE": test["AQI"]["RMSE"],
        "Average R2": test["average"]["R2"],
        "Average RMSE": test["average"]["RMSE"],
        "Interpretation": note,
        "metrics_path": str(metrics_path),
    }


def main():
    paths = {
        "random": ROOT / "reports/paper_cnn_lstm_tabular_fusion_final80/traqid_paper_resnet50_front_rear_concat_gap_traqid_paper_style_T7_front_sequence_manifest_split_random_tabular_fusion/metrics_paper_cnn_lstm_tabular_fusion.json",
        "twofold": ROOT / "reports/paper_cnn_lstm_tabular_fusion/twofold_average_summary.csv",
        "old_purged": ROOT / "reports/paper_cnn_lstm_tabular_fusion_purged_block/traqid_paper_resnet50_front_rear_concat_gap_traqid_paper_style_T7_front_sequence_manifest_purged_block_compatible_clean_split_date_chrono_tabular_fusion/metrics_paper_cnn_lstm_tabular_fusion.json",
        "time_balanced_purged": ROOT / "reports/paper_cnn_lstm_tabular_fusion_time_balanced_purged/traqid_paper_resnet50_front_rear_concat_gap_traqid_paper_style_T7_front_sequence_manifest_time_balanced_purged_split_time_balanced_purged_tabular_fusion/metrics_paper_cnn_lstm_tabular_fusion.json",
        "chrono": ROOT / "reports/paper_cnn_lstm_tabular_fusion/traqid_paper_resnet50_front_rear_concat_gap_traqid_paper_style_T7_front_sequence_manifest_split_chrono_date_tabular_fusion/metrics_paper_cnn_lstm_tabular_fusion.json",
    }

    rows = []

    rows.append(row(
        "Random 70/15/15",
        paths["random"],
        overlap_ge6=0.9091,
        overlap_ge1=1.0000,
        note="Paper-style random sequence split; heavy sliding-window overlap leakage.",
    ))

    # Two-fold is stored as an averaged CSV, not JSON.
    twofold = pd.read_csv(paths["twofold"])
    avg = twofold[[
        "PM25_R2", "PM25_RMSE",
        "PM10_R2", "PM10_RMSE",
        "AQI_R2", "AQI_RMSE",
        "Avg_R2", "Avg_RMSE"
    ]].mean()

    rows.append({
        "Protocol": "Two-fold random CV",
        "Test seq overlap >=1 frame": 0.9999,
        "Test seq overlap >=6/7 frames": 0.7481,
        "PM2.5 R2": avg["PM25_R2"],
        "PM2.5 RMSE": avg["PM25_RMSE"],
        "PM10 R2": avg["PM10_R2"],
        "PM10 RMSE": avg["PM10_RMSE"],
        "AQI R2": avg["AQI_R2"],
        "AQI RMSE": avg["AQI_RMSE"],
        "Average R2": avg["Avg_R2"],
        "Average RMSE": avg["Avg_RMSE"],
        "Interpretation": "Paper-style two-fold CV; still highly overlap-contaminated.",
        "metrics_path": str(paths["twofold"]),
    })

    rows.append(row(
        "Old purged-block",
        paths["old_purged"],
        overlap_ge6=0.0000,
        overlap_ge1=0.0000,
        note="No frame overlap, but test was later/night-skewed within each date.",
    ))

    rows.append(row(
        "Time-balanced purged",
        paths["time_balanced_purged"],
        overlap_ge6=0.0000,
        overlap_ge1=0.0000,
        note="No frame overlap; same dates; hour/day-night and target distributions broadly balanced.",
    ))

    rows.append(row(
        "Chronological date-wise",
        paths["chrono"],
        overlap_ge6=0.0000,
        overlap_ge1=0.0000,
        note="No frame overlap; future/unseen-date stress test.",
    ))

    df = pd.DataFrame(rows)

    csv_path = OUT_DIR / "final_validation_ladder_with_overlap.csv"
    md_path = OUT_DIR / "final_validation_ladder_with_overlap.md"

    df.to_csv(csv_path, index=False)

    pretty = df.copy()
    for c in pretty.columns:
        if pretty[c].dtype.kind in "fc":
            pretty[c] = pretty[c].map(lambda x: f"{x:.4f}")

    md_path.write_text(pretty.to_markdown(index=False), encoding="utf-8")

    print("Saved:", csv_path)
    print("Saved:", md_path)
    print("\nValidation ladder:")
    print(pretty[[
        "Protocol",
        "Test seq overlap >=6/7 frames",
        "PM2.5 R2",
        "PM10 R2",
        "AQI R2",
        "Average R2",
        "Interpretation",
    ]].to_string(index=False))

    # Plot average R2 ladder
    plt.figure(figsize=(10, 5))
    plt.bar(df["Protocol"], df["Average R2"])
    plt.axhline(0, linewidth=1)
    plt.ylabel("Average R²")
    plt.title("Validation protocol ladder: performance vs leakage/shift control")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_fig = FIG_DIR / "final_validation_ladder_average_r2.png"
    plt.savefig(out_fig, dpi=240)
    plt.close()
    print("Saved:", out_fig)

    # Plot overlap vs average R2
    plt.figure(figsize=(7, 5))
    plt.scatter(df["Test seq overlap >=6/7 frames"], df["Average R2"], s=90)
    for _, r in df.iterrows():
        plt.annotate(r["Protocol"], (r["Test seq overlap >=6/7 frames"], r["Average R2"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Fraction of test sequences sharing ≥6/7 frames with train")
    plt.ylabel("Average R²")
    plt.title("Random-split performance tracks sequence overlap leakage")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_fig = FIG_DIR / "final_overlap_vs_average_r2.png"
    plt.savefig(out_fig, dpi=240)
    plt.close()
    print("Saved:", out_fig)


if __name__ == "__main__":
    main()