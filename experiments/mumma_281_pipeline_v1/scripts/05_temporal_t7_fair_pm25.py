from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("experiments/mumma_281_pipeline_v1")
DATA = ROOT / "data/processed/final_feature_table.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TARGET = "value.sPM2"
RANDOM_STATE = 42
N_SPLITS = 5
T = 7
EPS = 1e-6

OUT_RESULTS = REPORTS / "pm25_temporal_t7_fair_results.csv"
OUT_PREDS = REPORTS / "pm25_temporal_t7_fair_predictions.csv"
OUT_FEATURES = REPORTS / "pm25_temporal_t7_feature_groups.json"


def metrics(y_true, y_pred):
    sp = np.nan if np.std(y_pred) == 0 else spearmanr(y_true, y_pred).correlation
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(sp) if sp == sp else np.nan,
        "Bias": float(np.mean(y_pred - y_true)),
    }


def add_time_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df["timestamp_parsed"] = dt

    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df


def find_cols(df, patterns):
    return [c for c in df.columns if any(p in c.lower() for p in patterns)]


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_col(df, candidates, default=0.0):
    c = first_existing(df, candidates)
    if c is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce").fillna(0.0)


def add_engineered_features(df):
    df = df.copy()

    auto = safe_col(df, ["idd_auto_rickshaw_count_sum", "auto_count_sum"])
    car = safe_col(df, ["idd_car_count_sum", "car_count_sum"])
    truck = safe_col(df, ["idd_truck_count_sum", "truck_count_sum"])
    bus = safe_col(df, ["idd_bus_count_sum", "bus_count_sum"])
    motorcycle = safe_col(df, ["idd_motorcycle_count_sum", "motorcycle_count_sum"])
    bicycle = safe_col(df, ["idd_bicycle_count_sum", "bicycle_count_sum"])
    person = safe_col(df, ["idd_person_count_sum", "person_count_sum"])

    vehicle_count_cols = find_cols(df, ["count_sum"])
    vehicle_count_cols = [
        c for c in vehicle_count_cols
        if any(k in c.lower() for k in ["idd_", "car", "bus", "truck", "auto", "motorcycle", "bicycle", "person"])
    ]

    if vehicle_count_cols:
        total_vehicle = df[vehicle_count_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    else:
        total_vehicle = auto + car + truck + bus + motorcycle + bicycle

    heavy_vehicle = truck + bus
    two_wheeler = motorcycle + bicycle

    df["eng_total_vehicle_count"] = total_vehicle
    df["eng_heavy_vehicle_count"] = heavy_vehicle
    df["eng_two_wheeler_count"] = two_wheeler
    df["eng_auto_count"] = auto
    df["eng_car_count"] = car
    df["eng_person_count"] = person

    total = total_vehicle + EPS

    df["eng_heavy_vehicle_ratio"] = heavy_vehicle / total
    df["eng_two_wheeler_ratio"] = two_wheeler / total
    df["eng_auto_ratio"] = auto / total
    df["eng_car_ratio"] = car / total
    df["eng_person_per_vehicle"] = person / total

    df["eng_traffic_mix_index"] = (
        2.5 * heavy_vehicle
        + 1.5 * auto
        + 1.0 * car
        + 0.8 * two_wheeler
    )

    df["eng_heavy_traffic_score"] = heavy_vehicle * total_vehicle
    df["eng_two_wheeler_traffic_score"] = two_wheeler * total_vehicle

    road_area = safe_col(df, ["road_area_ratio_mean", "road_area_ratio"])
    road_brown = safe_col(df, ["road_brown_pixel_ratio_mean", "road_brown_pixel_ratio"])
    road_gray = safe_col(df, ["road_gray_dry_pixel_ratio_mean", "road_gray_dry_pixel_ratio"])
    road_shadow = safe_col(df, ["road_shadow_ratio_mean", "road_shadow_ratio"])
    road_glare = safe_col(df, ["road_glare_ratio_mean", "road_glare_ratio"])
    road_edge = safe_col(df, ["road_edge_density_mean", "road_edge_density"])
    road_lap = safe_col(df, ["road_laplacian_std_mean", "road_laplacian_std"])
    brightness = safe_col(df, ["road_mean_brightness_mean", "road_mean_brightness"], default=0.5)
    saturation = safe_col(df, ["road_mean_saturation_mean", "road_mean_saturation"], default=0.5)

    df["eng_road_area_ratio"] = road_area
    df["eng_road_brown_ratio"] = road_brown
    df["eng_road_gray_dry_ratio"] = road_gray
    df["eng_road_shadow_ratio"] = road_shadow
    df["eng_road_glare_ratio"] = road_glare
    df["eng_road_texture_score"] = road_edge + road_lap

    df["eng_visible_road_quality"] = road_area * (1 - road_shadow.clip(0, 1)) * (1 - road_glare.clip(0, 1))
    df["eng_road_brown_exposure"] = road_brown * road_area
    df["eng_road_gray_dry_exposure"] = road_gray * road_area
    df["eng_brightness_corrected_brown"] = road_brown / (brightness + EPS)
    df["eng_saturation_corrected_brown"] = road_brown / (saturation + EPS)

    df["eng_road_dust_score"] = (
        0.40 * df["eng_road_brown_exposure"]
        + 0.25 * df["eng_road_gray_dry_exposure"]
        + 0.20 * df["eng_brightness_corrected_brown"]
        + 0.15 * df["eng_road_texture_score"]
    )

    df["eng_road_dust_score"] = df["eng_road_dust_score"] * (1 - road_shadow.clip(0, 1)) * (1 - road_glare.clip(0, 1))

    df["eng_vehicle_per_road_area"] = total_vehicle / (road_area + EPS)
    df["eng_heavy_vehicle_per_road_area"] = heavy_vehicle / (road_area + EPS)
    df["eng_traffic_x_road_dust"] = total_vehicle * df["eng_road_dust_score"]
    df["eng_heavy_x_road_dust"] = heavy_vehicle * df["eng_road_dust_score"]
    df["eng_two_wheeler_x_road_dust"] = two_wheeler * df["eng_road_dust_score"]

    temp = pd.to_numeric(df["temp"], errors="coerce").fillna(df["temp"].median())
    rh = pd.to_numeric(df["rh"], errors="coerce").fillna(df["rh"].median())

    df["eng_temp_x_vehicle"] = temp * total_vehicle
    df["eng_rh_x_vehicle"] = rh * total_vehicle
    df["eng_rh_x_heavy"] = rh * heavy_vehicle
    df["eng_temp_x_road_dust"] = temp * df["eng_road_dust_score"]
    df["eng_rh_x_road_dust"] = rh * df["eng_road_dust_score"]

    return df


def get_fair_groups(df):
    cols = list(df.columns)

    id_cols = []
    for c in cols:
        lc = c.lower()
        if lc in ["timestamp", "timestamp_parsed", "sample_id", "sample_index"] or "path" in lc or "file" in lc:
            id_cols.append(c)

    target_like = []
    for c in cols:
        lc = c.lower()
        if c == TARGET:
            continue
        if (
            "spm" in lc
            or "npm" in lc
            or "pm1" in lc
            or "pm2" in lc
            or "pm4" in lc
            or "pm10" in lc
            or "density" in lc
        ):
            target_like.append(c)

    banned = set(id_cols + target_like + [TARGET])

    tabular = [
        c for c in [
            "temp", "rh",
            "value.co_ppb", "value.no2_ppb", "value.so2_ppb", "value.o3_ppb_compensated",
            "hour", "minute", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
        ]
        if c in df.columns and c not in banned
    ]

    raw_source = [
        c for c in cols
        if any(k in c.lower() for k in [
            "vehicle", "car", "bus", "truck", "auto", "motorcycle",
            "bicycle", "person", "traffic", "detections",
            "road", "brown", "gray", "grey", "dust", "shadow", "glare",
            "haze", "laplacian", "edge", "brightness", "saturation", "contrast",
            "osm", "highway", "primary", "secondary", "tertiary",
            "residential", "service", "motorway", "trunk", "distance",
            "nearest", "intersection", "junction", "roadarea", "bus_stop",
            "commercial", "retail", "human_activity"
        ])
        and c not in banned
        and not c.startswith("eng_")
    ]

    engineered = [
        c for c in cols
        if c.startswith("eng_") and c not in banned
    ]

    top30_source = [
        "road_mean_saturation_std",
        "road_gray_dry_pixel_ratio_std",
        "idd_auto_rickshaw_count_sum",
        "idd_auto_rickshaw_count_mean",
        "eng_auto_count",
        "osm_longitude_rounded",
        "idd_auto_rickshaw_count_max",
        "road_mean_saturation_max",
        "eng_auto_ratio",
        "road_gray_dry_pixel_ratio_min",
        "road_gray_dry_x_texture",
        "idd_bus_count_sum",
        "idd_motor_vehicle_count_sum",
        "vehicle_min_offset_sec",
        "vehicle_max_offset_sec",
        "idd_bus_count_mean",
        "bus_stop_count_250m",
        "idd_motor_vehicle_count_mean",
        "eng_traffic_mix_index",
        "road_laplacian_std_std",
        "eng_rh_x_vehicle",
        "idd_bus_count_max",
        "idd_total_vehicle_count_sum",
        "vehicle_exhaust_x_dry_air",
        "primary_road_count_250m",
        "idd_car_count_mean",
        "exhaust_x_osm_human_activity_intensity",
        "idd_car_count_sum",
        "eng_total_vehicle_count",
        "eng_temp_x_vehicle",
    ]

    top30_source = [c for c in top30_source if c in df.columns and c not in banned]

    groups = {
        "tabular": tabular,
        "raw_source": sorted(set(raw_source)),
        "top30_source": top30_source,
        "tabular_plus_top30_source": sorted(set(tabular + top30_source)),
        "tabular_plus_raw_source": sorted(set(tabular + raw_source)),
    }

    diagnostics = {
        "id_cols": id_cols,
        "target_like_excluded": target_like,
        "tabular": tabular,
        "raw_source": raw_source,
        "engineered": engineered,
        "top30_source": top30_source,
    }

    return groups, diagnostics


def make_temporal_table(df, feature_cols, target_col, t=7):
    rows = []
    y = []
    source_indices = []

    for i in range(t - 1, len(df)):
        item = {}
        window = df.iloc[i - t + 1:i + 1]

        for lag in range(t):
            lag_from_current = t - 1 - lag
            prefix = f"lag{lag_from_current}"
            row = window.iloc[lag]

            for c in feature_cols:
                item[f"{prefix}__{c}"] = row[c]

        rows.append(item)
        y.append(df.iloc[i][target_col])
        source_indices.append(i)

    X = pd.DataFrame(rows)
    y = np.array(y, dtype=float)
    source_indices = np.array(source_indices, dtype=int)

    return X, y, source_indices


def convert_bool(X):
    X = X.copy()
    for c in X.select_dtypes(include=["bool"]).columns.tolist():
        X[c] = X[c].astype("int64")
    return X


def make_preprocessor(X):
    X = convert_bool(X)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot),
            ]), categorical_cols),
        ],
        remainder="drop",
    )


def model_factory(model_name):
    if model_name == "et_leaf1":
        return ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "et_leaf2":
        return ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "et_leaf3":
        return ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=3,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "rf_leaf2":
        return RandomForestRegressor(
            n_estimators=700,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "hgb_slow":
        return HistGradientBoostingRegressor(
            max_iter=700,
            learning_rate=0.02,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )

    if model_name == "ridge":
        return Ridge(alpha=10.0)

    raise ValueError(model_name)


def predict_oof(X, y, model_name, log_target=False):
    X = convert_bool(X)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    target = np.log1p(y) if log_target else y

    pipe = Pipeline([
        ("preprocess", make_preprocessor(X)),
        ("model", model_factory(model_name)),
    ])

    pred = cross_val_predict(pipe, X, target, cv=cv, n_jobs=None)

    if log_target:
        pred = np.expm1(pred)

    return np.clip(pred, 0, None)


def residual_fusion_temporal_oof(X_base, X_res, y, base_model_name, residual_model_name, log_target=False):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    X_base = convert_bool(X_base)
    X_res = convert_bool(X_res)

    final_pred = np.zeros(len(y), dtype=float)
    base_oof = np.zeros(len(y), dtype=float)
    residual_oof = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in cv.split(X_base):
        Xb_train = X_base.iloc[train_idx].copy()
        Xb_test = X_base.iloc[test_idx].copy()
        Xr_train = X_res.iloc[train_idx].copy()
        Xr_test = X_res.iloc[test_idx].copy()

        y_train = y[train_idx]

        y_train_model = np.log1p(y_train) if log_target else y_train

        base_pipe = Pipeline([
            ("preprocess", make_preprocessor(Xb_train)),
            ("model", model_factory(base_model_name)),
        ])

        base_pipe.fit(Xb_train, y_train_model)

        base_train_pred = base_pipe.predict(Xb_train)
        base_test_pred = base_pipe.predict(Xb_test)

        if log_target:
            base_train_real = np.expm1(base_train_pred)
            base_test_real = np.expm1(base_test_pred)
        else:
            base_train_real = base_train_pred
            base_test_real = base_test_pred

        train_residual = y_train - base_train_real

        res_pipe = Pipeline([
            ("preprocess", make_preprocessor(Xr_train)),
            ("model", model_factory(residual_model_name)),
        ])

        res_pipe.fit(Xr_train, train_residual)
        res_pred = res_pipe.predict(Xr_test)

        pred = np.clip(base_test_real + res_pred, 0, None)

        final_pred[test_idx] = pred
        base_oof[test_idx] = np.clip(base_test_real, 0, None)
        residual_oof[test_idx] = res_pred

    return final_pred, base_oof, residual_oof


def add_record(records, pred_frames, experiment, feature_set, model, y, pred, source_indices, n_features, extra=None):
    row = {
        "experiment": experiment,
        "feature_set": feature_set,
        "model": model,
        "T": T,
        "n_features": n_features,
        "n_rows": len(y),
        "n_splits": N_SPLITS,
    }
    row.update(metrics(y, pred))
    if extra:
        row.update(extra)

    records.append(row)

    pred_frames.append(pd.DataFrame({
        "source_row_index": source_indices,
        "experiment": experiment,
        "feature_set": feature_set,
        "model": model,
        "y_true_pm25": y,
        "y_pred_pm25": pred,
        "residual_pred_minus_true": pred - y,
    }))


def main():
    df = pd.read_csv(DATA)
    df = add_time_features(df)
    df = add_engineered_features(df)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].copy()

    # Sort temporally. sample_index is used to stabilize duplicate timestamps.
    sort_cols = ["timestamp_parsed"]
    if "sample_index" in df.columns:
        sort_cols.append("sample_index")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    groups, diagnostics = get_fair_groups(df)

    print("Sorted shape:", df.shape)
    print("Timestamp range:", df["timestamp_parsed"].min(), "to", df["timestamp_parsed"].max())
    print("Target summary:")
    print(df[TARGET].describe().to_string())

    print("\nGroup sizes:")
    for k, v in groups.items():
        print(k, len(v))

    with open(OUT_FEATURES, "w") as f:
        json.dump({
            "target": TARGET,
            "T": T,
            "groups": groups,
            "diagnostics": diagnostics,
        }, f, indent=2)

    temporal_tables = {}
    for fs_name in [
        "tabular",
        "top30_source",
        "raw_source",
        "tabular_plus_top30_source",
        "tabular_plus_raw_source",
    ]:
        X, y, source_indices = make_temporal_table(df, groups[fs_name], TARGET, t=T)
        temporal_tables[fs_name] = (X, y, source_indices)
        print(f"{fs_name}: X={X.shape}, y={y.shape}")

    records = []
    pred_frames = []

    model_names = ["et_leaf1", "et_leaf2", "et_leaf3", "rf_leaf2", "hgb_slow", "ridge"]

    # Direct temporal models
    for fs_name, (X, y, source_indices) in temporal_tables.items():
        for model_name in model_names:
            for log_target in [False, True]:
                exp = f"direct_T{T}_{fs_name}_{model_name}_{'log1p' if log_target else 'raw'}"
                print("\n" + "-" * 100)
                print("START", exp, X.shape)

                pred = predict_oof(X, y, model_name, log_target=log_target)

                add_record(
                    records,
                    pred_frames,
                    exp,
                    fs_name,
                    model_name,
                    y,
                    pred,
                    source_indices,
                    X.shape[1],
                    extra={"mode": "direct_temporal", "log_target": log_target},
                )

                pd.DataFrame(records).sort_values("RMSE").to_csv(OUT_RESULTS, index=False)
                pd.concat(pred_frames, ignore_index=True).to_csv(OUT_PREDS, index=False)

                print(metrics(y, pred))

    # Temporal residual fusion
    X_base, y, source_indices = temporal_tables["tabular"]
    residual_sets = {
        "top30_source": temporal_tables["top30_source"][0],
        "raw_source": temporal_tables["raw_source"][0],
    }

    base_models = ["et_leaf1", "et_leaf2", "et_leaf3"]
    residual_models = ["et_leaf1", "et_leaf2", "et_leaf3", "rf_leaf2", "hgb_slow"]

    for res_name, X_res in residual_sets.items():
        for base_model in base_models:
            for residual_model in residual_models:
                for log_target in [False, True]:
                    exp = f"residual_T{T}_{res_name}_base-{base_model}_res-{residual_model}_{'log1p' if log_target else 'raw'}"
                    print("\n" + "-" * 100)
                    print("START", exp, "base", X_base.shape, "res", X_res.shape)

                    pred, base_oof, residual_oof = residual_fusion_temporal_oof(
                        X_base,
                        X_res,
                        y,
                        base_model,
                        residual_model,
                        log_target=log_target,
                    )

                    add_record(
                        records,
                        pred_frames,
                        exp,
                        res_name,
                        f"base_{base_model}_res_{residual_model}",
                        y,
                        pred,
                        source_indices,
                        X_base.shape[1] + X_res.shape[1],
                        extra={
                            "mode": "residual_temporal",
                            "log_target": log_target,
                            "base_model": base_model,
                            "residual_model": residual_model,
                            "n_base_features": X_base.shape[1],
                            "n_residual_features": X_res.shape[1],
                        },
                    )

                    pd.DataFrame(records).sort_values("RMSE").to_csv(OUT_RESULTS, index=False)
                    pd.concat(pred_frames, ignore_index=True).to_csv(OUT_PREDS, index=False)

                    print(metrics(y, pred))

    results = pd.DataFrame(records).sort_values("RMSE")
    results.to_csv(OUT_RESULTS, index=False)

    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(OUT_PREDS, index=False)

    print("\nFinal temporal T7 fair results:")
    print(results.head(50).to_string(index=False))

    print("\nSaved:")
    print(OUT_RESULTS)
    print(OUT_PREDS)
    print(OUT_FEATURES)


if __name__ == "__main__":
    main()