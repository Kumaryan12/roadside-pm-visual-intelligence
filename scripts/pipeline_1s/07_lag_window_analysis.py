from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


DEFAULT_SENSOR_CSV = "data/sensor/MC1S_best_3min_window.csv"

DEFAULT_VEHICLE_FRAME_CSV = (
    "outputs/pipeline_1s/vehicle_detections/"
    "idd_vehicle_detections_frame_level_1s_v1_best3min.csv"
)

DEFAULT_ROAD_AREA_V3_CSV = (
    "outputs/pipeline_1s/road_area_lens1_best3min_v1/"
    "lens1_depth_gated_occlusion_adjusted_road_area_v3_1s_best3min.csv"
)

DEFAULT_OUTPUT_FEATURES_CSV = (
    "outputs/pipeline_1s/features/"
    "sensor_level_lag_window_features_best3min.csv"
)

DEFAULT_OUTPUT_CORR_CSV = (
    "outputs/pipeline_1s/reports/lag_window_analysis/"
    "lag_window_spearman_and_partial_correlations.csv"
)


KEY_COLS = [
    "sensor_row_id",
    "sample_index",
    "sensor_timestamp",
    "sensor_unix",
]


TARGETS = [
    "value.sPM1",
    "value.sPM2",
    "value.sPM10",
    "nPM2",
]


WINDOWS_SEC = [10, 20, 30, 60]


VEHICLE_BASE_FEATURES = [
    "idd_total_vehicle_count",
    "idd_total_pm_relevant_objects",
    "idd_heavy_vehicle_count",
    "idd_motor_vehicle_count",
    "idd_car_count",
    "idd_motorcycle_count",
    "idd_auto_rickshaw_count",
    "idd_bus_count",
    "idd_truck_count",
    "idd_bicycle_count",
    "idd_exhaust_proxy_initial",
    "idd_resuspension_vehicle_proxy_initial",
    "idd_vehicle_box_area_ratio",
]


ROAD_BASE_FEATURES = [
    "visible_road_area_m2_depth_est",
    "vehicle_occluded_road_area_m2_conservative_v3",
    "road_area_m2_occlusion_adjusted_conservative_v3",
    "road_area_vehicle_occlusion_fraction_v3",
    "vehicle_count_detected_for_occlusion_v3",
    "vehicle_count_used_for_occlusion_v3",
    "vehicle_count_rejected_by_depth_v3",
    "road_depth_p50_m",
    "road_area_percent_px",
]


KEY_ANALYSIS_FEATURE_PATTERNS = [
    "road_lens1_road_area_vehicle_occlusion_fraction_v3_mean",
    "road_lens1_vehicle_occluded_road_area_m2_conservative_v3_mean",
    "road_lens1_visible_road_area_m2_depth_est_mean",
    "road_lens1_road_area_m2_occlusion_adjusted_conservative_v3_mean",
    "veh_all_lenses_idd_total_vehicle_count_mean",
    "veh_all_lenses_idd_bus_count_mean",
    "veh_all_lenses_idd_truck_count_mean",
    "veh_all_lenses_idd_motorcycle_count_mean",
    "veh_all_lenses_idd_exhaust_proxy_initial_mean",
    "veh_all_lenses_idd_resuspension_vehicle_proxy_initial_mean",
]


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def residualize(y, X):
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)

    model = LinearRegression()
    model.fit(X, y)

    return (y - model.predict(X)).ravel()


def safe_spearman(x, y):
    sub = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(sub) < 5:
        return np.nan, np.nan, len(sub)

    if sub["x"].nunique() <= 1 or sub["y"].nunique() <= 1:
        return np.nan, np.nan, len(sub)

    r, p = spearmanr(sub["x"], sub["y"])
    return float(r), float(p), len(sub)


def aggregate_window_vehicle(vehicle_df, window_sec):
    """
    Aggregate all-lens vehicle features for last N seconds.

    Since relative_time_sec runs from -60 to 0,
    window 10 means frames where -10 <= relative_time_sec <= 0.
    """
    start = -float(window_sec)

    df = vehicle_df[
        (vehicle_df["idd_detection_status"] == "success")
        & (vehicle_df["relative_time_sec"] >= start)
        & (vehicle_df["relative_time_sec"] <= 0)
    ].copy()

    vehicle_features = [c for c in VEHICLE_BASE_FEATURES if c in df.columns]

    for c in vehicle_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    expected_frames = (window_sec + 1) * 3

    avail = (
        df.groupby(KEY_COLS, dropna=False)
        .agg(
            **{
                f"w{window_sec}_veh_frame_count": ("processed_frame_key", "nunique"),
                f"w{window_sec}_veh_lens_count": ("lens_id", "nunique"),
                f"w{window_sec}_veh_relative_second_count": ("relative_time_sec", "nunique"),
                f"w{window_sec}_veh_min_relative_time_sec": ("relative_time_sec", "min"),
                f"w{window_sec}_veh_max_relative_time_sec": ("relative_time_sec", "max"),
            }
        )
        .reset_index()
    )

    avail[f"w{window_sec}_veh_expected_frame_count"] = expected_frames
    avail[f"w{window_sec}_veh_frame_completeness"] = (
        avail[f"w{window_sec}_veh_frame_count"] / expected_frames
    )

    if not vehicle_features:
        return avail

    agg = (
        df.groupby(KEY_COLS, dropna=False)[vehicle_features]
        .agg(["sum", "mean", "max", "std"])
        .reset_index()
    )

    agg.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        if isinstance(col, tuple)
        else str(col)
        for col in agg.columns
    ]

    rename = {}
    for c in agg.columns:
        if c in KEY_COLS:
            continue
        rename[c] = f"w{window_sec}_veh_all_lenses_{c}"

    agg = agg.rename(columns=rename)

    out = avail.merge(agg, on=KEY_COLS, how="left", validate="one_to_one")
    return out


def aggregate_window_road(road_df, window_sec):
    """
    Aggregate lens-1 road/depth/occlusion features for last N seconds.
    """
    start = -float(window_sec)

    df = road_df[
        (road_df["area_status"] == "success")
        & (road_df["relative_time_sec"] >= start)
        & (road_df["relative_time_sec"] <= 0)
    ].copy()

    if "lens_id" in df.columns:
        df = df[df["lens_id"].astype(int) == 1].copy()

    road_features = [c for c in ROAD_BASE_FEATURES if c in df.columns]

    for c in road_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    expected_frames = window_sec + 1

    avail = (
        df.groupby(KEY_COLS, dropna=False)
        .agg(
            **{
                f"w{window_sec}_road_frame_count": ("processed_frame_key", "nunique"),
                f"w{window_sec}_road_relative_second_count": ("relative_time_sec", "nunique"),
                f"w{window_sec}_road_min_relative_time_sec": ("relative_time_sec", "min"),
                f"w{window_sec}_road_max_relative_time_sec": ("relative_time_sec", "max"),
            }
        )
        .reset_index()
    )

    avail[f"w{window_sec}_road_expected_frame_count"] = expected_frames
    avail[f"w{window_sec}_road_frame_completeness"] = (
        avail[f"w{window_sec}_road_frame_count"] / expected_frames
    )

    if not road_features:
        return avail

    agg = (
        df.groupby(KEY_COLS, dropna=False)[road_features]
        .agg(["mean", "median", "max", "min", "std"])
        .reset_index()
    )

    agg.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        if isinstance(col, tuple)
        else str(col)
        for col in agg.columns
    ]

    rename = {}
    for c in agg.columns:
        if c in KEY_COLS:
            continue
        rename[c] = f"w{window_sec}_road_lens1_{c}"

    agg = agg.rename(columns=rename)

    out = avail.merge(agg, on=KEY_COLS, how="left", validate="one_to_one")
    return out


def make_lag_window_features(sensor_df, vehicle_df, road_df, windows):
    if "sensor_row_id" not in sensor_df.columns:
        sensor_df = sensor_df.copy()
        sensor_df["sensor_row_id"] = range(len(sensor_df))

    base_cols = [c for c in ["sensor_row_id", "timestamp"] if c in sensor_df.columns]
    base = sensor_df[base_cols].copy()

    # Bring canonical merge keys from vehicle data because it contains sensor_unix etc.
    key_base = vehicle_df[KEY_COLS].drop_duplicates().copy()
    out = base.merge(key_base, on="sensor_row_id", how="left")

    for window_sec in windows:
        print(f"\nAggregating window: last {window_sec}s")

        veh_w = aggregate_window_vehicle(vehicle_df, window_sec)
        road_w = aggregate_window_road(road_df, window_sec)

        out = out.merge(veh_w, on=KEY_COLS, how="left", validate="one_to_one")
        out = out.merge(road_w, on=KEY_COLS, how="left", validate="one_to_one")

    return out


def merge_with_sensor(sensor_df, features_df):
    if "sensor_row_id" not in sensor_df.columns:
        sensor_df = sensor_df.copy()
        sensor_df["sensor_row_id"] = range(len(sensor_df))

    merged = sensor_df.merge(
        features_df,
        on="sensor_row_id",
        how="left",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )

    return merged


def build_correlation_report(model_df, windows):
    rows = []

    targets = [t for t in TARGETS if t in model_df.columns]

    for window_sec in windows:
        feature_cols = []

        for pattern in KEY_ANALYSIS_FEATURE_PATTERNS:
            candidate = f"w{window_sec}_{pattern}"
            if candidate in model_df.columns:
                feature_cols.append(candidate)

        for target in targets:
            for feat in feature_cols:
                sub = model_df[[target, feat, "sensor_row_id"]].dropna()

                raw_r, raw_p, n = safe_spearman(sub[feat], sub[target])

                partial_r = np.nan
                partial_p = np.nan

                if len(sub) >= 5 and sub[feat].nunique() > 1:
                    y_res = residualize(sub[target], sub[["sensor_row_id"]])
                    x_res = residualize(sub[feat], sub[["sensor_row_id"]])
                    partial_r, partial_p, _ = safe_spearman(x_res, y_res)

                rows.append({
                    "target": target,
                    "window_sec": window_sec,
                    "feature": feat,
                    "n": n,
                    "raw_spearman_r": raw_r,
                    "raw_p_value": raw_p,
                    "partial_time_control_spearman_r": partial_r,
                    "partial_time_control_p_value": partial_p,
                    "abs_partial_r": abs(partial_r) if np.isfinite(partial_r) else np.nan,
                })

    report = pd.DataFrame(rows)

    if len(report) > 0:
        report = report.sort_values(
            ["target", "feature", "window_sec"]
        ).reset_index(drop=True)

    return report


def summarize_best_windows(corr_df):
    if len(corr_df) == 0:
        return pd.DataFrame()

    valid = corr_df.dropna(subset=["partial_time_control_spearman_r"]).copy()

    if len(valid) == 0:
        return pd.DataFrame()

    best = (
        valid.sort_values("abs_partial_r", ascending=False)
        .groupby(["target", "feature"], as_index=False)
        .head(1)
        .sort_values(["target", "abs_partial_r"], ascending=[True, False])
    )

    return best


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sensor-csv", default=DEFAULT_SENSOR_CSV)
    parser.add_argument("--vehicle-frame-csv", default=DEFAULT_VEHICLE_FRAME_CSV)
    parser.add_argument("--road-area-v3-csv", default=DEFAULT_ROAD_AREA_V3_CSV)
    parser.add_argument("--output-features-csv", default=DEFAULT_OUTPUT_FEATURES_CSV)
    parser.add_argument("--output-corr-csv", default=DEFAULT_OUTPUT_CORR_CSV)
    parser.add_argument("--windows", nargs="+", type=int, default=WINDOWS_SEC)

    args = parser.parse_args()

    sensor_path = Path(args.sensor_csv)
    vehicle_path = Path(args.vehicle_frame_csv)
    road_path = Path(args.road_area_v3_csv)
    output_features_path = Path(args.output_features_csv)
    output_corr_path = Path(args.output_corr_csv)

    sensor_df = pd.read_csv(sensor_path)
    vehicle_df = pd.read_csv(vehicle_path)
    road_df = pd.read_csv(road_path)

    require_columns(vehicle_df, KEY_COLS + ["relative_time_sec", "lens_id", "processed_frame_key"], "vehicle CSV")
    require_columns(road_df, KEY_COLS + ["relative_time_sec", "lens_id", "processed_frame_key"], "road CSV")

    print("\nSensor shape:", sensor_df.shape)
    print("Vehicle shape:", vehicle_df.shape)
    print("Road shape:", road_df.shape)
    print("Windows:", args.windows)

    features = make_lag_window_features(
        sensor_df=sensor_df,
        vehicle_df=vehicle_df,
        road_df=road_df,
        windows=args.windows,
    )

    model_df = merge_with_sensor(sensor_df, features)

    output_features_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(output_features_path, index=False)

    print("\nSaved lag-window modeling table:")
    print(output_features_path)
    print("Shape:", model_df.shape)

    corr = build_correlation_report(model_df, args.windows)

    output_corr_path.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(output_corr_path, index=False)

    best = summarize_best_windows(corr)
    best_path = output_corr_path.parent / "best_lag_windows_by_feature_target.csv"
    best.to_csv(best_path, index=False)

    print("\nSaved correlation report:")
    print(output_corr_path)

    print("\nSaved best-window report:")
    print(best_path)

    print("\nCompleteness check:")
    comp_cols = []
    for w in args.windows:
        comp_cols.extend([
            f"w{w}_veh_frame_completeness",
            f"w{w}_road_frame_completeness",
        ])
    comp_cols = [c for c in comp_cols if c in model_df.columns]
    print(model_df[comp_cols].describe().T)

    print("\nBest lag windows:")
    if len(best) > 0:
        show_cols = [
            "target",
            "feature",
            "window_sec",
            "partial_time_control_spearman_r",
            "partial_time_control_p_value",
            "raw_spearman_r",
        ]
        print(best[show_cols].to_string(index=False))
    else:
        print("No valid best-window rows.")


if __name__ == "__main__":
    main()