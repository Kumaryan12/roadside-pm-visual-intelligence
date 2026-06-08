from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


DEFAULT_INPUT = "outputs/pipeline_1s/modeling/modeling_table_1s_features_best3min.csv"
DEFAULT_OUTDIR = "outputs/pipeline_1s/reports/extra_1s_analysis"


TARGETS = ["value.sPM1", "value.sPM2", "value.sPM10", "nPM2"]


FEATURE_GROUPS = {
    "traffic_persistence": [
        "veh1s_all_lenses_idd_total_vehicle_count_mean",
        "veh1s_all_lenses_idd_exhaust_proxy_initial_mean",
        "veh1s_all_lenses_idd_resuspension_vehicle_proxy_initial_mean",
    ],
    "traffic_peak_events": [
        "veh1s_all_lenses_idd_total_vehicle_count_max",
        "veh1s_all_lenses_idd_bus_count_max",
        "veh1s_all_lenses_idd_truck_count_max",
    ],
    "vehicle_type_exposure": [
        "veh1s_all_lenses_idd_car_count_sum",
        "veh1s_all_lenses_idd_motorcycle_count_sum",
        "veh1s_all_lenses_idd_bus_count_sum",
        "veh1s_all_lenses_idd_truck_count_sum",
        "veh1s_all_lenses_idd_auto_rickshaw_count_sum",
    ],
    "near_field_occlusion": [
        "road1s_lens1_road_area_vehicle_occlusion_fraction_v3_mean",
        "road1s_lens1_road_area_vehicle_occlusion_fraction_v3_max",
        "road1s_lens1_vehicle_occluded_road_area_m2_conservative_v3_mean",
    ],
    "road_openness": [
        "road1s_lens1_visible_road_area_m2_depth_est_mean",
        "road1s_lens1_road_area_m2_occlusion_adjusted_conservative_v3_mean",
    ],
    "quality_reliability": [
        "road1s_final_road_area_feature_status_v3_use_primary_fraction",
        "veh1s_frame_completeness",
        "road1s_frame_completeness",
    ],
}


def spearman_table(df, targets, feature_groups):
    rows = []

    for group, features in feature_groups.items():
        for feature in features:
            if feature not in df.columns:
                continue

            for target in targets:
                if target not in df.columns:
                    continue

                sub = df[[feature, target]].dropna()
                if len(sub) < 5:
                    continue

                r, p = spearmanr(sub[feature], sub[target])

                rows.append({
                    "group": group,
                    "feature": feature,
                    "target": target,
                    "n": len(sub),
                    "spearman_r": r,
                    "p_value": p,
                    "abs_r": abs(r),
                })

    return pd.DataFrame(rows).sort_values(["target", "abs_r"], ascending=[True, False])


def plot_time_series(df, outdir):
    cols = [
        "value.sPM2",
        "road1s_lens1_road_area_vehicle_occlusion_fraction_v3_mean",
        "road1s_lens1_visible_road_area_m2_depth_est_mean",
        "veh1s_all_lenses_idd_total_vehicle_count_mean",
        "veh1s_all_lenses_idd_exhaust_proxy_initial_mean",
    ]

    cols = [c for c in cols if c in df.columns]

    for col in cols:
        plt.figure(figsize=(8, 4))
        plt.plot(df["sensor_row_id"], df[col], marker="o")
        plt.xlabel("sensor_row_id")
        plt.ylabel(col)
        plt.title(f"{col} over sensor time")
        plt.tight_layout()
        plt.savefig(outdir / f"time_{col}.png", dpi=200)
        plt.close()


def plot_key_scatter(df, outdir):
    pairs = [
        ("road1s_lens1_road_area_vehicle_occlusion_fraction_v3_mean", "value.sPM2"),
        ("road1s_lens1_visible_road_area_m2_depth_est_mean", "value.sPM2"),
        ("veh1s_all_lenses_idd_total_vehicle_count_mean", "value.sPM2"),
        ("veh1s_all_lenses_idd_bus_count_sum", "value.sPM2"),
        ("veh1s_all_lenses_idd_motorcycle_count_sum", "value.sPM2"),
    ]

    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            continue

        plt.figure(figsize=(6, 5))
        plt.scatter(df[x], df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{y} vs {x}")
        plt.tight_layout()
        plt.savefig(outdir / f"scatter_{y}_vs_{x}.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    targets = [t for t in TARGETS if t in df.columns]

    corr = spearman_table(df, targets, FEATURE_GROUPS)
    corr_path = outdir / "extra_1s_feature_group_spearman.csv"
    corr.to_csv(corr_path, index=False)

    top_path = outdir / "top_extra_1s_correlations_by_target.csv"
    top = corr.groupby("target", group_keys=False).head(15)
    top.to_csv(top_path, index=False)

    plot_time_series(df, outdir)
    plot_key_scatter(df, outdir)

    print("\nSaved:")
    print(corr_path)
    print(top_path)
    print("\nTop correlations:")
    print(top[["target", "group", "feature", "spearman_r", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()