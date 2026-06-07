from pathlib import Path
import argparse
import numpy as np
import pandas as pd


DEFAULT_OLD_CORR = Path("outputs/features/particle_density_feature_correlations_v2.csv")
DEFAULT_NEW_CORR = Path("outputs/features/particle_density_feature_correlations_idd_finetuned_v2.csv")

DEFAULT_OLD_TABLE = Path("outputs/features/particle_density_modeling_table_v2.csv")
DEFAULT_NEW_TABLE = Path("outputs/features/particle_density_modeling_table_idd_finetuned_v2.csv")

DEFAULT_OUTPUT_ALL = Path("outputs/features/pretrained_vs_idd_finetuned_all_correlation_comparison_v2.csv")
DEFAULT_OUTPUT_KEY = Path("outputs/features/pretrained_vs_idd_finetuned_key_correlation_comparison_v2.csv")
DEFAULT_OUTPUT_ROAD_STATS = Path("outputs/features/pretrained_vs_idd_finetuned_road_feature_stats_v2.csv")


KEY_TARGETS = [
    "value.sPM1",
    "value.sPM2",
    "value.sPM4",
    "value.sPM10",
    "value.sNPM1",
    "nPM2",
    "value.sNPM4",
    "value.sNPM10",
]


# canonical_name, possible feature names, group
KEY_FEATURES = [
    ("auto_rickshaw_count_sum", ["idd_auto_rickshaw_count_sum"], "vehicle"),
    ("auto_rickshaw_count_mean", ["idd_auto_rickshaw_count_mean"], "vehicle"),
    ("auto_rickshaw_count_max", ["idd_auto_rickshaw_count_max"], "vehicle"),

    ("car_count_sum", ["idd_car_count_sum"], "vehicle"),
    ("truck_count_sum", ["idd_truck_count_sum"], "vehicle"),
    ("heavy_vehicle_count_sum", ["idd_heavy_vehicle_count_sum"], "vehicle"),
    ("motorcycle_count_sum", ["idd_motorcycle_count_sum"], "vehicle"),

    ("exhaust_proxy_sum", ["idd_exhaust_proxy_initial_sum"], "vehicle_proxy"),
    ("resuspension_proxy_sum", ["idd_resuspension_vehicle_proxy_initial_sum"], "vehicle_proxy"),

    ("road_area_ratio_mean", ["road_area_ratio_mean"], "road"),
    ("road_brown_pixel_ratio_mean", ["road_brown_pixel_ratio_mean"], "road"),
    ("road_gray_dry_pixel_ratio_mean", ["road_gray_dry_pixel_ratio_mean"], "road"),
    ("road_edge_density_mean", ["road_edge_density_mean"], "road"),
    ("road_laplacian_std_mean", ["road_laplacian_std_mean"], "road"),
    ("road_contrast_std_mean", ["road_contrast_std_mean"], "road"),
    ("road_haze_flatness_proxy_mean", ["road_haze_flatness_proxy_mean"], "road"),

    (
        "vehicle_resuspension_x_road_brown",
        [
            "vehicle_resuspension_x_road_brown_mean",
            "vehicle_resuspension_x_brown_pixel_ratio",
        ],
        "interaction",
    ),
    (
        "vehicle_resuspension_x_road_gray_dry",
        [
            "vehicle_resuspension_x_road_gray_dry_mean",
            "vehicle_resuspension_x_gray_dry_pixel_ratio",
        ],
        "interaction",
    ),
    ("vehicle_resuspension_x_dry_air", ["vehicle_resuspension_x_dry_air"], "interaction"),
    ("vehicle_exhaust_x_dry_air", ["vehicle_exhaust_x_dry_air"], "interaction"),
    ("road_brown_x_edge_density", ["road_brown_x_edge_density"], "road_composite"),
    ("road_gray_dry_x_texture", ["road_gray_dry_x_texture"], "road_composite"),
    ("road_brown_x_texture", ["road_brown_x_texture"], "road_composite"),
]


ROAD_FEATURE_COLS = [
    "road_area_ratio_mean",
    "road_brown_pixel_ratio_mean",
    "road_gray_dry_pixel_ratio_mean",
    "road_edge_density_mean",
    "road_laplacian_std_mean",
    "road_contrast_std_mean",
    "road_mean_brightness_mean",
    "road_mean_saturation_mean",
    "road_haze_flatness_proxy_mean",
]


def to_bool_series(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def load_corr(path: Path, model_name: str):
    if not path.exists():
        raise FileNotFoundError(f"{model_name} correlation file not found: {path}")

    df = pd.read_csv(path)

    required = ["feature", "target", "spearman_corr"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{model_name} correlation CSV missing columns: {missing}")

    df["feature"] = df["feature"].astype(str)
    df["target"] = df["target"].astype(str)
    df["spearman_corr"] = pd.to_numeric(df["spearman_corr"], errors="coerce")

    if "abs_spearman_corr" not in df.columns:
        df["abs_spearman_corr"] = df["spearman_corr"].abs()
    else:
        df["abs_spearman_corr"] = pd.to_numeric(df["abs_spearman_corr"], errors="coerce")

    if "n" not in df.columns:
        df["n"] = np.nan
    else:
        df["n"] = pd.to_numeric(df["n"], errors="coerce")

    return df


def lookup_corr(corr_df, candidate_features, target):
    """
    Returns the first matching feature-target correlation from candidate_features.
    This helps compare old/new files even when an interaction feature has two aliases.
    """
    for feature in candidate_features:
        row = corr_df[
            (corr_df["feature"] == feature)
            & (corr_df["target"] == target)
        ]

        if len(row) > 0:
            row = row.iloc[0]
            return {
                "feature_used": feature,
                "spearman_corr": row["spearman_corr"],
                "abs_spearman_corr": row["abs_spearman_corr"],
                "n": row["n"],
            }

    return {
        "feature_used": None,
        "spearman_corr": np.nan,
        "abs_spearman_corr": np.nan,
        "n": np.nan,
    }


def compare_all_common(old_corr, new_corr):
    old_small = old_corr[[
        "feature",
        "target",
        "spearman_corr",
        "abs_spearman_corr",
        "n",
    ]].rename(columns={
        "spearman_corr": "old_corr",
        "abs_spearman_corr": "old_abs_corr",
        "n": "old_n",
    })

    new_small = new_corr[[
        "feature",
        "target",
        "spearman_corr",
        "abs_spearman_corr",
        "n",
    ]].rename(columns={
        "spearman_corr": "new_corr",
        "abs_spearman_corr": "new_abs_corr",
        "n": "new_n",
    })

    merged = old_small.merge(
        new_small,
        on=["feature", "target"],
        how="outer",
    )

    merged["delta_corr"] = merged["new_corr"] - merged["old_corr"]
    merged["delta_abs_corr"] = merged["new_abs_corr"] - merged["old_abs_corr"]
    merged["abs_delta_corr"] = merged["delta_corr"].abs()

    merged = merged.sort_values(
        ["abs_delta_corr", "new_abs_corr"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return merged


def compare_key_features(old_corr, new_corr):
    rows = []

    for canonical_name, candidates, group in KEY_FEATURES:
        for target in KEY_TARGETS:
            old_info = lookup_corr(old_corr, candidates, target)
            new_info = lookup_corr(new_corr, candidates, target)

            rows.append({
                "group": group,
                "canonical_feature": canonical_name,
                "target": target,

                "old_feature_used": old_info["feature_used"],
                "new_feature_used": new_info["feature_used"],

                "old_corr": old_info["spearman_corr"],
                "new_corr": new_info["spearman_corr"],
                "delta_corr": new_info["spearman_corr"] - old_info["spearman_corr"],

                "old_abs_corr": old_info["abs_spearman_corr"],
                "new_abs_corr": new_info["abs_spearman_corr"],
                "delta_abs_corr": new_info["abs_spearman_corr"] - old_info["abs_spearman_corr"],

                "old_n": old_info["n"],
                "new_n": new_info["n"],
                "delta_n": new_info["n"] - old_info["n"],
            })

    out = pd.DataFrame(rows)
    out["abs_delta_corr"] = out["delta_corr"].abs()

    out = out.sort_values(
        ["target", "new_abs_corr", "abs_delta_corr"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return out


def availability_summary(old_table_path, new_table_path):
    rows = []

    for name, path in [
        ("cityscapes_pretrained", old_table_path),
        ("idd_finetuned", new_table_path),
    ]:
        if not path.exists():
            print(f"Warning: table not found for availability summary: {path}")
            continue

        df = pd.read_csv(path)

        row = {
            "model": name,
            "rows": len(df),
        }

        for col in [
            "image_features_available",
            "vehicle_all_3_lenses_available",
            "vehicle_features_available",
            "road_all_2_lenses_available",
            "road_features_available",
        ]:
            if col in df.columns:
                bool_col = to_bool_series(df[col])
                row[f"{col}_true"] = int(bool_col.sum())
                row[f"{col}_false"] = int((~bool_col).sum())

        rows.append(row)

    return pd.DataFrame(rows)


def road_feature_stats(old_table_path, new_table_path):
    if not old_table_path.exists():
        raise FileNotFoundError(f"Old modeling table not found: {old_table_path}")

    if not new_table_path.exists():
        raise FileNotFoundError(f"New modeling table not found: {new_table_path}")

    old = pd.read_csv(old_table_path)
    new = pd.read_csv(new_table_path)

    rows = []

    for col in ROAD_FEATURE_COLS:
        if col not in old.columns or col not in new.columns:
            continue

        old_values = pd.to_numeric(old[col], errors="coerce")
        new_values = pd.to_numeric(new[col], errors="coerce")

        rows.append({
            "feature": col,
            "old_n": int(old_values.notna().sum()),
            "new_n": int(new_values.notna().sum()),

            "old_mean": old_values.mean(),
            "new_mean": new_values.mean(),
            "delta_mean": new_values.mean() - old_values.mean(),

            "old_std": old_values.std(),
            "new_std": new_values.std(),
            "delta_std": new_values.std() - old_values.std(),

            "old_median": old_values.median(),
            "new_median": new_values.median(),
            "delta_median": new_values.median() - old_values.median(),
        })

    return pd.DataFrame(rows)


def print_target_block(key_df, target):
    block = key_df[
        (key_df["target"] == target)
        & (key_df["old_corr"].notna() | key_df["new_corr"].notna())
    ].copy()

    block = block.sort_values("new_abs_corr", ascending=False)

    cols = [
        "group",
        "canonical_feature",
        "old_corr",
        "new_corr",
        "delta_corr",
        "old_n",
        "new_n",
    ]

    print(f"\nTop comparison for target: {target}")
    print(block[cols].head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--old-corr", default=str(DEFAULT_OLD_CORR))
    parser.add_argument("--new-corr", default=str(DEFAULT_NEW_CORR))
    parser.add_argument("--old-table", default=str(DEFAULT_OLD_TABLE))
    parser.add_argument("--new-table", default=str(DEFAULT_NEW_TABLE))

    parser.add_argument("--output-all", default=str(DEFAULT_OUTPUT_ALL))
    parser.add_argument("--output-key", default=str(DEFAULT_OUTPUT_KEY))
    parser.add_argument("--output-road-stats", default=str(DEFAULT_OUTPUT_ROAD_STATS))

    args = parser.parse_args()

    old_corr_path = Path(args.old_corr)
    new_corr_path = Path(args.new_corr)

    old_table_path = Path(args.old_table)
    new_table_path = Path(args.new_table)

    output_all = Path(args.output_all)
    output_key = Path(args.output_key)
    output_road_stats = Path(args.output_road_stats)

    old_corr = load_corr(old_corr_path, "old/pretrained")
    new_corr = load_corr(new_corr_path, "new/fine-tuned")

    all_comparison = compare_all_common(old_corr, new_corr)
    key_comparison = compare_key_features(old_corr, new_corr)
    road_stats = road_feature_stats(old_table_path, new_table_path)
    availability = availability_summary(old_table_path, new_table_path)

    output_all.parent.mkdir(parents=True, exist_ok=True)

    all_comparison.to_csv(output_all, index=False)
    key_comparison.to_csv(output_key, index=False)
    road_stats.to_csv(output_road_stats, index=False)

    print("\nSaved all correlation comparison:", output_all)
    print("Saved key correlation comparison:", output_key)
    print("Saved road feature stats comparison:", output_road_stats)

    print("\nAvailability summary:")
    print(availability.to_string(index=False))

    print("\nRoad feature stats comparison:")
    print(road_stats.to_string(index=False))

    print_target_block(key_comparison, "value.sPM2")
    print_target_block(key_comparison, "value.sPM10")
    print_target_block(key_comparison, "value.sNPM1")
    print_target_block(key_comparison, "nPM2")

    print("\nLargest absolute correlation changes overall:")
    show_cols = [
        "feature",
        "target",
        "old_corr",
        "new_corr",
        "delta_corr",
        "old_n",
        "new_n",
    ]
    show_cols = [c for c in show_cols if c in all_comparison.columns]

    print(all_comparison[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()