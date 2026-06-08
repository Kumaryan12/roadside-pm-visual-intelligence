from pathlib import Path
import argparse
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CORR_CSV = (
    "outputs/pipeline_1s/reports/lag_window_analysis/"
    "lag_window_spearman_and_partial_correlations.csv"
)

DEFAULT_MODEL_CSV = (
    "outputs/pipeline_1s/features/"
    "sensor_level_lag_window_features_best3min.csv"
)

DEFAULT_OUTDIR = (
    "outputs/pipeline_1s/reports/lag_window_analysis/report_artifacts"
)


# ============================================================
# LABEL HELPERS
# ============================================================

def base_feature_name(feature: str) -> str:
    """
    Remove window prefix:
    w60_road_lens1_xxx -> road_lens1_xxx
    """
    parts = str(feature).split("_", 1)
    if len(parts) == 2 and parts[0].startswith("w") and parts[0][1:].isdigit():
        return parts[1]
    return str(feature)


def clean_feature_label(feature: str) -> str:
    base = base_feature_name(feature)

    mapping = {
        "road_lens1_road_area_vehicle_occlusion_fraction_v3_mean":
            "Road occlusion fraction mean",
        "road_lens1_vehicle_occluded_road_area_m2_conservative_v3_mean":
            "Vehicle-hidden road area mean",
        "road_lens1_visible_road_area_m2_depth_est_mean":
            "Visible road area mean",
        "road_lens1_road_area_m2_occlusion_adjusted_conservative_v3_mean":
            "Occlusion-adjusted road area mean",

        "veh_all_lenses_idd_total_vehicle_count_mean":
            "Total vehicle count mean",
        "veh_all_lenses_idd_bus_count_mean":
            "Bus count mean",
        "veh_all_lenses_idd_truck_count_mean":
            "Truck count mean",
        "veh_all_lenses_idd_motorcycle_count_mean":
            "Motorcycle count mean",
        "veh_all_lenses_idd_exhaust_proxy_initial_mean":
            "Exhaust proxy mean",
        "veh_all_lenses_idd_resuspension_vehicle_proxy_initial_mean":
            "Resuspension proxy mean",
    }

    return mapping.get(base, base.replace("_", " "))


def feature_family(feature: str) -> str:
    base = base_feature_name(feature)

    if "occlusion_fraction" in base or "vehicle_occluded_road_area" in base:
        return "Near-field occlusion"

    if "visible_road_area" in base or "occlusion_adjusted_conservative" in base:
        return "Road openness / exposure"

    if "bus_count" in base or "truck_count" in base or "motorcycle_count" in base:
        return "Vehicle type exposure"

    if "total_vehicle_count" in base:
        return "Overall traffic count"

    if "exhaust_proxy" in base or "resuspension" in base:
        return "Emission proxy"

    return "Other"


def target_label(target: str) -> str:
    return {
        "value.sPM1": "sPM1",
        "value.sPM2": "sPM2",
        "value.sPM10": "sPM10",
        "nPM2": "nPM2",
    }.get(target, target)


# ============================================================
# TABLE GENERATION
# ============================================================

def prepare_corr(corr: pd.DataFrame) -> pd.DataFrame:
    required = [
        "target",
        "window_sec",
        "feature",
        "raw_spearman_r",
        "partial_time_control_spearman_r",
        "partial_time_control_p_value",
    ]

    missing = [c for c in required if c not in corr.columns]
    if missing:
        raise ValueError(f"Correlation CSV missing columns: {missing}")

    out = corr.copy()
    out["feature_base"] = out["feature"].apply(base_feature_name)
    out["feature_label"] = out["feature"].apply(clean_feature_label)
    out["feature_family"] = out["feature"].apply(feature_family)
    out["target_label"] = out["target"].apply(target_label)
    out["abs_partial_r"] = out["partial_time_control_spearman_r"].abs()

    out = out.dropna(subset=["partial_time_control_spearman_r"]).copy()

    return out


def make_best_by_feature(corr: pd.DataFrame) -> pd.DataFrame:
    best = (
        corr.sort_values("abs_partial_r", ascending=False)
        .groupby(["target", "feature_base"], as_index=False)
        .head(1)
        .sort_values(["target", "abs_partial_r"], ascending=[True, False])
        .reset_index(drop=True)
    )

    cols = [
        "target",
        "target_label",
        "feature_family",
        "feature_label",
        "feature",
        "window_sec",
        "partial_time_control_spearman_r",
        "partial_time_control_p_value",
        "raw_spearman_r",
        "abs_partial_r",
    ]

    return best[cols]


def make_best_by_family(corr: pd.DataFrame) -> pd.DataFrame:
    best = (
        corr.sort_values("abs_partial_r", ascending=False)
        .groupby(["target", "feature_family"], as_index=False)
        .head(1)
        .sort_values(["target", "abs_partial_r"], ascending=[True, False])
        .reset_index(drop=True)
    )

    cols = [
        "target",
        "target_label",
        "feature_family",
        "feature_label",
        "window_sec",
        "partial_time_control_spearman_r",
        "partial_time_control_p_value",
        "raw_spearman_r",
        "abs_partial_r",
    ]

    return best[cols]


def make_pm2_compact_table(best_by_feature: pd.DataFrame) -> pd.DataFrame:
    pm2 = best_by_feature[best_by_feature["target"] == "value.sPM2"].copy()

    pm2 = pm2.sort_values("abs_partial_r", ascending=False).reset_index(drop=True)

    pm2["interpretation"] = pm2.apply(
        lambda r: "Positive association with PM2"
        if r["partial_time_control_spearman_r"] > 0
        else "Negative association with PM2",
        axis=1,
    )

    cols = [
        "feature_family",
        "feature_label",
        "window_sec",
        "partial_time_control_spearman_r",
        "partial_time_control_p_value",
        "raw_spearman_r",
        "interpretation",
    ]

    return pm2[cols]


# ============================================================
# PLOTS
# ============================================================

def save_pm2_best_bar(pm2_table: pd.DataFrame, outdir: Path):
    if len(pm2_table) == 0:
        return

    plot_df = pm2_table.head(12).copy()
    plot_df = plot_df.sort_values("partial_time_control_spearman_r")

    labels = [
        f"{row.feature_label}\n({int(row.window_sec)}s)"
        for _, row in plot_df.iterrows()
    ]

    plt.figure(figsize=(9, 7))
    plt.barh(labels, plot_df["partial_time_control_spearman_r"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Time-controlled Spearman correlation with PM2")
    plt.title("Best lag-window visual features for PM2")
    plt.tight_layout()
    plt.savefig(outdir / "pm2_best_lag_window_features_bar.png", dpi=220)
    plt.close()


def save_feature_window_lines(corr: pd.DataFrame, outdir: Path, target="value.sPM2"):
    selected_bases = [
        "road_lens1_road_area_vehicle_occlusion_fraction_v3_mean",
        "road_lens1_vehicle_occluded_road_area_m2_conservative_v3_mean",
        "road_lens1_visible_road_area_m2_depth_est_mean",
        "road_lens1_road_area_m2_occlusion_adjusted_conservative_v3_mean",
        "veh_all_lenses_idd_bus_count_mean",
        "veh_all_lenses_idd_total_vehicle_count_mean",
    ]

    sub = corr[
        (corr["target"] == target)
        & (corr["feature_base"].isin(selected_bases))
    ].copy()

    if len(sub) == 0:
        return

    plt.figure(figsize=(9, 6))

    for base in selected_bases:
        s = sub[sub["feature_base"] == base].sort_values("window_sec")
        if len(s) == 0:
            continue
        label = clean_feature_label(base)
        plt.plot(
            s["window_sec"],
            s["partial_time_control_spearman_r"],
            marker="o",
            label=label,
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Lag window length before sensor reading (seconds)")
    plt.ylabel("Time-controlled Spearman correlation with PM2")
    plt.title("How PM2 correlation changes with visual aggregation window")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "pm2_correlation_vs_lag_window_lines.png", dpi=220)
    plt.close()


def save_family_best_bar(best_family: pd.DataFrame, outdir: Path, target="value.sPM2"):
    sub = best_family[best_family["target"] == target].copy()

    if len(sub) == 0:
        return

    sub = sub.sort_values("partial_time_control_spearman_r")

    labels = [
        f"{row.feature_family}\n{row.feature_label} ({int(row.window_sec)}s)"
        for _, row in sub.iterrows()
    ]

    plt.figure(figsize=(9, 6))
    plt.barh(labels, sub["partial_time_control_spearman_r"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Best time-controlled Spearman correlation with PM2")
    plt.title("Best feature per visual-information family")
    plt.tight_layout()
    plt.savefig(outdir / "pm2_best_feature_by_family_bar.png", dpi=220)
    plt.close()


def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    return (s - s.mean()) / s.std(ddof=0)


def save_normalized_time_overlay(model_df: pd.DataFrame, outdir: Path):
    cols = {
        "value.sPM2": "PM2",
        "w60_road_lens1_road_area_vehicle_occlusion_fraction_v3_mean": "60s occlusion fraction",
        "w30_road_lens1_visible_road_area_m2_depth_est_mean": "30s visible road area",
        "w30_road_lens1_road_area_m2_occlusion_adjusted_conservative_v3_mean": "30s adjusted road area",
    }

    available = {c: label for c, label in cols.items() if c in model_df.columns}

    if "sensor_row_id" not in model_df.columns or len(available) < 2:
        return

    plt.figure(figsize=(10, 5))

    for c, label in available.items():
        plt.plot(model_df["sensor_row_id"], zscore(model_df[c]), marker="o", label=label)

    plt.axhline(0, linewidth=1)
    plt.xlabel("Sensor row ID")
    plt.ylabel("Z-score")
    plt.title("Normalized PM2 and key visual context features over time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "pm2_key_visual_features_time_overlay_zscore.png", dpi=220)
    plt.close()


# ============================================================
# MARKDOWN REPORT
# ============================================================

def write_markdown_report(pm2_table: pd.DataFrame, best_family: pd.DataFrame, outdir: Path):
    pm2_top = pm2_table.head(8).copy()

    lines = []
    lines.append("# Lag-window visual-feature analysis summary\n")
    lines.append("## Core finding\n")
    lines.append(
        "The 1-second sampling experiment shows that PM is more strongly associated "
        "with sustained road-context features than with a single timestamp frame or raw vehicle counts.\n"
    )

    lines.append("## Best PM2 features\n")
    for _, r in pm2_top.iterrows():
        lines.append(
            f"- **{r['feature_label']}** ({int(r['window_sec'])}s window): "
            f"time-controlled Spearman r = **{r['partial_time_control_spearman_r']:.3f}**, "
            f"p = {r['partial_time_control_p_value']:.4g}."
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        textwrap.dedent(
            """
            - Near-field road occlusion is the strongest positive PM signal.
            - Road openness / visible road area is strongly negatively associated with PM.
            - Raw vehicle count is not the strongest signal and can be misleading.
            - The 30–60 second windows outperform the 10 second window for the major road-context features.
            - This suggests that PM is better represented by sustained visual exposure history rather than one instant image.
            """
        ).strip()
    )

    lines.append("\n## Caution\n")
    lines.append(
        "These results are exploratory because the 23 sensor rows are sequential and "
        "60-second windows overlap. The correlations should be reported as evidence of "
        "association, not final causal proof."
    )

    report_path = outdir / "lag_window_analysis_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corr-csv", default=DEFAULT_CORR_CSV)
    parser.add_argument("--model-csv", default=DEFAULT_MODEL_CSV)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    args = parser.parse_args()

    corr_path = Path(args.corr_csv)
    model_path = Path(args.model_csv)
    outdir = Path(args.outdir)

    if not corr_path.exists():
        raise FileNotFoundError(f"Correlation CSV not found: {corr_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model CSV not found: {model_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    corr_raw = pd.read_csv(corr_path)
    model_df = pd.read_csv(model_path)

    corr = prepare_corr(corr_raw)

    best_by_feature = make_best_by_feature(corr)
    best_by_family = make_best_by_family(corr)
    pm2_table = make_pm2_compact_table(best_by_feature)

    best_by_feature_path = outdir / "best_lag_window_by_feature_all_targets.csv"
    best_by_family_path = outdir / "best_lag_window_by_family_all_targets.csv"
    pm2_table_path = outdir / "pm2_compact_lag_window_summary.csv"
    cleaned_corr_path = outdir / "cleaned_lag_window_correlations.csv"

    corr.to_csv(cleaned_corr_path, index=False)
    best_by_feature.to_csv(best_by_feature_path, index=False)
    best_by_family.to_csv(best_by_family_path, index=False)
    pm2_table.to_csv(pm2_table_path, index=False)

    save_pm2_best_bar(pm2_table, outdir)
    save_feature_window_lines(corr, outdir, target="value.sPM2")
    save_family_best_bar(best_by_family, outdir, target="value.sPM2")
    save_normalized_time_overlay(model_df, outdir)
    write_markdown_report(pm2_table, best_by_family, outdir)

    print("\nSaved report artifacts to:")
    print(outdir)

    print("\nGenerated files:")
    for p in sorted(outdir.glob("*")):
        print("-", p)

    print("\nPM2 compact summary:")
    print(pm2_table.to_string(index=False))


if __name__ == "__main__":
    main()