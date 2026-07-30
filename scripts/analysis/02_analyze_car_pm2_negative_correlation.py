from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


DEFAULT_INPUT_CSV = Path("outputs/features/particle_density_modeling_table_idd_finetuned_osm_v2.csv")
DEFAULT_VALIDATION_CSV = Path("outputs/validation/road_feature_validation_pack_v1/manual_label_sheet.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/car_pm2_negative_correlation")


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_corr(x, y, method="spearman"):
    temp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(temp) < 5:
        return np.nan, np.nan, len(temp)

    if temp["x"].nunique() < 2 or temp["y"].nunique() < 2:
        return np.nan, np.nan, len(temp)

    if method == "spearman":
        r, p = spearmanr(temp["x"], temp["y"])
    else:
        r, p = pearsonr(temp["x"], temp["y"])

    return float(r), float(p), len(temp)


def residualize_ranked(df, x_col, y_col, control_cols):
    """
    Partial Spearman approximation:
    rank-transform y, x, controls, regress y_rank and x_rank on controls_rank,
    correlate residuals.
    """
    cols = [x_col, y_col] + control_cols
    temp = df[cols].copy()

    for c in cols:
        temp[c] = pd.to_numeric(temp[c], errors="coerce")

    temp = temp.dropna()

    if len(temp) < 20:
        return np.nan, np.nan, len(temp)

    if temp[x_col].nunique() < 2 or temp[y_col].nunique() < 2:
        return np.nan, np.nan, len(temp)

    ranked = temp.copy()
    for c in cols:
        ranked[c] = rankdata(ranked[c].values)

    X_ctrl = ranked[control_cols].values
    x = ranked[x_col].values
    y = ranked[y_col].values

    model_x = LinearRegression().fit(X_ctrl, x)
    model_y = LinearRegression().fit(X_ctrl, y)

    x_res = x - model_x.predict(X_ctrl)
    y_res = y - model_y.predict(X_ctrl)

    r, p = spearmanr(x_res, y_res)
    return float(r), float(p), len(temp)


def add_validation_paths(df, validation_csv):
    if not validation_csv.exists():
        return df

    val = pd.read_csv(validation_csv)

    if "sample_index" not in val.columns or "contact_sheet_path" not in val.columns:
        return df

    grouped = (
        val.groupby("sample_index")["contact_sheet_path"]
        .apply(lambda x: " | ".join(x.dropna().astype(str).tolist()))
        .reset_index()
        .rename(columns={"contact_sheet_path": "validation_contact_sheet_paths"})
    )

    return df.merge(grouped, on="sample_index", how="left")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--validation-csv", default=str(DEFAULT_VALIDATION_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", default="value.sPM2")
    parser.add_argument("--car-col", default=None)
    parser.add_argument("--require-image", action="store_true")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    validation_csv = Path(args.validation_csv)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    car_col = args.car_col
    if car_col is None:
        car_col = first_existing(
            df,
            [
                "idd_car_count_sum",
                "idd_car_count_mean",
                "idd_car_count_max",
            ],
        )

    if car_col is None:
        raise ValueError("Could not find car count column.")

    if args.require_image and "image_features_available" in df.columns:
        before = len(df)
        df = df[df["image_features_available"] == True].copy()
        print(f"Filtered image_features_available: {before} -> {len(df)}")

    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df[car_col] = pd.to_numeric(df[car_col], errors="coerce")

    if "sample_index" in df.columns:
        df = df.sort_values("sample_index").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        df["sample_index"] = df.index

    df["time_order_index"] = np.arange(len(df))

    print("\nInput:", input_csv)
    print("Rows:", len(df))
    print("Target:", args.target)
    print("Car column:", car_col)

    # ------------------------------------------------------------
    # 1. Global correlations
    # ------------------------------------------------------------
    rows = []

    for method in ["spearman", "pearson"]:
        r, p, n = safe_corr(df[car_col], df[args.target], method=method)
        rows.append({
            "analysis": "global",
            "method": method,
            "feature": car_col,
            "target": args.target,
            "correlation": r,
            "p_value": p,
            "n": n,
        })

    global_corr = pd.DataFrame(rows)
    global_corr.to_csv(output_dir / "01_global_car_pm2_correlation.csv", index=False)

    print("\nGlobal correlation:")
    print(global_corr.to_string(index=False))

    # ------------------------------------------------------------
    # 2. Car-count bins
    # ------------------------------------------------------------
    temp = df[[car_col, args.target]].dropna().copy()

    try:
        df["car_count_bin"] = pd.qcut(
            df[car_col],
            q=5,
            duplicates="drop",
        )
    except Exception:
        df["car_count_bin"] = pd.cut(df[car_col], bins=5)

    bin_summary = (
        df.groupby("car_count_bin", observed=True)
        .agg(
            n=("sample_index", "count"),
            car_mean=(car_col, "mean"),
            car_median=(car_col, "median"),
            pm2_mean=(args.target, "mean"),
            pm2_median=(args.target, "median"),
            pm2_std=(args.target, "std"),
        )
        .reset_index()
    )

    bin_summary["car_count_bin"] = bin_summary["car_count_bin"].astype(str)
    bin_summary.to_csv(output_dir / "02_pm2_by_car_count_bins.csv", index=False)

    print("\nPM2 by car-count bins:")
    print(bin_summary.to_string(index=False))

    # ------------------------------------------------------------
    # 3. High-car subsets
    # ------------------------------------------------------------
    subset_rows = []

    for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]:
        threshold = df[car_col].quantile(q)
        sub = df[df[car_col] >= threshold].copy()

        r_s, p_s, n_s = safe_corr(sub[car_col], sub[args.target], "spearman")
        r_p, p_p, n_p = safe_corr(sub[car_col], sub[args.target], "pearson")

        subset_rows.append({
            "subset": f"car_count_top_{int((1-q)*100)}pct_or_above_q{q}",
            "car_threshold": threshold,
            "n": len(sub),
            "spearman_corr": r_s,
            "spearman_p": p_s,
            "pearson_corr": r_p,
            "pearson_p": p_p,
            "pm2_mean": sub[args.target].mean(),
            "pm2_median": sub[args.target].median(),
        })

    subset_corr = pd.DataFrame(subset_rows)
    subset_corr.to_csv(output_dir / "03_high_car_subset_correlations.csv", index=False)

    print("\nHigh-car subset correlations:")
    print(subset_corr.to_string(index=False))

    # ------------------------------------------------------------
    # 4. Confounder checks and partial correlation
    # ------------------------------------------------------------
    candidate_controls = [
        "time_order_index",
        "rh",
        "temp",
        "dry_air_fraction",

        "idd_auto_rickshaw_count_sum",
        "idd_motorcycle_count_sum",
        "idd_truck_count_sum",
        "idd_bus_count_sum",
        "idd_heavy_vehicle_count_sum",

        "idd_exhaust_proxy_initial_sum",
        "idd_resuspension_vehicle_proxy_initial_sum",

        "road_brown_pixel_ratio_mean",
        "road_gray_dry_pixel_ratio_mean",
        "road_area_ratio_mean",
        "road_edge_density_mean",
        "road_laplacian_std_mean",
    ]

    control_cols = [c for c in candidate_controls if c in df.columns and c != car_col]

    partial_rows = []

    control_sets = {
        "time_only": [c for c in ["time_order_index"] if c in control_cols],
        "weather_time": [c for c in ["time_order_index", "rh", "temp", "dry_air_fraction"] if c in control_cols],
        "traffic_weather_time": [
            c for c in [
                "time_order_index",
                "rh",
                "temp",
                "dry_air_fraction",
                "idd_auto_rickshaw_count_sum",
                "idd_motorcycle_count_sum",
                "idd_truck_count_sum",
                "idd_bus_count_sum",
                "idd_heavy_vehicle_count_sum",
            ]
            if c in control_cols
        ],
        "traffic_weather_road_time": control_cols,
    }

    for name, controls in control_sets.items():
        if len(controls) == 0:
            continue

        r, p, n = residualize_ranked(
            df=df,
            x_col=car_col,
            y_col=args.target,
            control_cols=controls,
        )

        partial_rows.append({
            "control_set": name,
            "num_controls": len(controls),
            "controls": " | ".join(controls),
            "partial_spearman_corr": r,
            "p_value": p,
            "n": n,
        })

    partial_df = pd.DataFrame(partial_rows)
    partial_df.to_csv(output_dir / "04_partial_spearman_car_pm2.csv", index=False)

    print("\nPartial Spearman:")
    print(partial_df.to_string(index=False))

    # ------------------------------------------------------------
    # 5. Lag analysis
    # lag k means car count k previous samples aligned to PM2 now
    # ------------------------------------------------------------
    lag_rows = []

    for lag in range(-5, 6):
        shifted_car = df[car_col].shift(lag)

        if lag > 0:
            meaning = f"car_count_from_{lag}_rows_before_vs_pm2_now"
        elif lag < 0:
            meaning = f"car_count_{abs(lag)}_rows_after_vs_pm2_now"
        else:
            meaning = "same_timestamp"

        r, p, n = safe_corr(shifted_car, df[args.target], "spearman")

        lag_rows.append({
            "lag_rows": lag,
            "meaning": meaning,
            "spearman_corr": r,
            "p_value": p,
            "n": n,
        })

    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(output_dir / "05_lag_correlation_car_pm2.csv", index=False)

    print("\nLag correlations:")
    print(lag_df.to_string(index=False))

    # ------------------------------------------------------------
    # 6. Visual audit candidates
    # ------------------------------------------------------------
    df_aug = add_validation_paths(df, validation_csv)

    car_hi = df_aug[car_col].quantile(0.75)
    car_lo = df_aug[car_col].quantile(0.25)
    pm_hi = df_aug[args.target].quantile(0.75)
    pm_lo = df_aug[args.target].quantile(0.25)

    useful_cols = [
        "sample_index",
        "timestamp",
        car_col,
        args.target,
        "idd_auto_rickshaw_count_sum",
        "idd_motorcycle_count_sum",
        "idd_truck_count_sum",
        "idd_heavy_vehicle_count_sum",
        "rh",
        "temp",
        "dry_air_fraction",
        "road_brown_pixel_ratio_mean",
        "road_gray_dry_pixel_ratio_mean",
        "road_area_ratio_mean",
        "image_features_available",
        "validation_contact_sheet_paths",
    ]

    useful_cols = [c for c in useful_cols if c in df_aug.columns]

    quadrants = {
        "06_high_car_low_pm2_visual_audit.csv": df_aug[
            (df_aug[car_col] >= car_hi) & (df_aug[args.target] <= pm_lo)
        ],
        "07_high_car_high_pm2_visual_audit.csv": df_aug[
            (df_aug[car_col] >= car_hi) & (df_aug[args.target] >= pm_hi)
        ],
        "08_low_car_high_pm2_visual_audit.csv": df_aug[
            (df_aug[car_col] <= car_lo) & (df_aug[args.target] >= pm_hi)
        ],
        "09_low_car_low_pm2_visual_audit.csv": df_aug[
            (df_aug[car_col] <= car_lo) & (df_aug[args.target] <= pm_lo)
        ],
    }

    for filename, qdf in quadrants.items():
        qdf = qdf[useful_cols].sort_values(
            [car_col, args.target],
            ascending=[False, False],
        )
        qdf.to_csv(output_dir / filename, index=False)

    print("\nSaved visual audit CSVs to:", output_dir)

    # ------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------
    plot_df = df[[car_col, args.target]].dropna().copy()

    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df[car_col], plot_df[args.target], alpha=0.65)
    plt.xlabel(car_col)
    plt.ylabel(args.target)
    plt.title("Car count vs PM2.5")
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_car_count_vs_pm2.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(bin_summary["car_count_bin"], bin_summary["pm2_median"], marker="o")
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Car count bin")
    plt.ylabel("Median PM2.5")
    plt.title("Median PM2.5 across car-count bins")
    plt.tight_layout()
    plt.savefig(output_dir / "pm2_median_by_car_count_bin.png", dpi=180)
    plt.close()

    if "sample_index" in df.columns:
        plt.figure(figsize=(10, 5))
        ax1 = plt.gca()
        ax1.plot(df["sample_index"], df[args.target], label=args.target)
        ax1.set_xlabel("sample_index")
        ax1.set_ylabel(args.target)

        ax2 = ax1.twinx()
        ax2.plot(df["sample_index"], df[car_col], label=car_col, alpha=0.65)
        ax2.set_ylabel(car_col)

        plt.title("PM2.5 and car count over time")
        plt.tight_layout()
        plt.savefig(output_dir / "timeseries_pm2_and_car_count.png", dpi=180)
        plt.close()

    print("\nSaved plots:")
    print(output_dir / "scatter_car_count_vs_pm2.png")
    print(output_dir / "pm2_median_by_car_count_bin.png")
    print(output_dir / "timeseries_pm2_and_car_count.png")


if __name__ == "__main__":
    main()