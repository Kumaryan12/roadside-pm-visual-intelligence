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
FIGURES = ROOT / "figures"

REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

OUT_RESULTS = REPORTS / "pm25_engineered_fair_cv_results.csv"
OUT_PREDS = REPORTS / "pm25_engineered_fair_cv_predictions.csv"
OUT_IMPORTANCE = REPORTS / "pm25_engineered_feature_importance.csv"
OUT_GROUPS = REPORTS / "pm25_engineered_feature_groups.json"

TARGET = "value.sPM2"
RANDOM_STATE = 42
N_SPLITS = 5
EPS = 1e-6


def metric_dict(y_true, y_pred):
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

    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df


def find_cols(df, patterns):
    cols = []
    for c in df.columns:
        lc = c.lower()
        if any(p in lc for p in patterns):
            cols.append(c)
    return cols


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

    # -----------------------------
    # Vehicle columns
    # -----------------------------
    auto = safe_col(df, [
        "idd_auto_rickshaw_count_sum", "auto_count_sum", "auto_rickshaw_count_sum"
    ])
    car = safe_col(df, [
        "idd_car_count_sum", "car_count_sum"
    ])
    truck = safe_col(df, [
        "idd_truck_count_sum", "truck_count_sum"
    ])
    bus = safe_col(df, [
        "idd_bus_count_sum", "bus_count_sum"
    ])
    motorcycle = safe_col(df, [
        "idd_motorcycle_count_sum", "motorcycle_count_sum", "two_wheeler_count_sum"
    ])
    bicycle = safe_col(df, [
        "idd_bicycle_count_sum", "bicycle_count_sum"
    ])
    person = safe_col(df, [
        "idd_person_count_sum", "person_count_sum"
    ])

    vehicle_count_cols = find_cols(df, [
        "count_sum"
    ])
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

    # -----------------------------
    # Road features
    # -----------------------------
    road_area = safe_col(df, [
        "road_area_ratio_mean",
        "road_area_ratio",
        "road_mask_area_ratio_mean",
    ])

    road_brown = safe_col(df, [
        "road_brown_pixel_ratio_mean",
        "road_brown_pixel_ratio",
    ])

    road_gray = safe_col(df, [
        "road_gray_dry_pixel_ratio_mean",
        "road_gray_dry_pixel_ratio",
    ])

    road_shadow = safe_col(df, [
        "road_shadow_ratio_mean",
        "road_shadow_ratio",
    ])

    road_glare = safe_col(df, [
        "road_glare_ratio_mean",
        "road_glare_ratio",
    ])

    road_edge = safe_col(df, [
        "road_edge_density_mean",
        "road_edge_density",
    ])

    road_lap = safe_col(df, [
        "road_laplacian_std_mean",
        "road_laplacian_std",
    ])

    brightness = safe_col(df, [
        "road_mean_brightness_mean",
        "road_mean_brightness",
    ], default=0.5)

    saturation = safe_col(df, [
        "road_mean_saturation_mean",
        "road_mean_saturation",
    ], default=0.5)

    df["eng_road_area_ratio"] = road_area
    df["eng_road_brown_ratio"] = road_brown
    df["eng_road_gray_dry_ratio"] = road_gray
    df["eng_road_shadow_ratio"] = road_shadow
    df["eng_road_glare_ratio"] = road_glare
    df["eng_road_texture_score"] = road_edge + road_lap

    df["eng_visible_road_quality"] = (
        road_area
        * (1 - road_shadow.clip(0, 1))
        * (1 - road_glare.clip(0, 1))
    )

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

    df["eng_road_dust_score"] = df["eng_road_dust_score"] * (
        1 - road_shadow.clip(0, 1)
    ) * (
        1 - road_glare.clip(0, 1)
    )

    # -----------------------------
    # Traffic × road interactions
    # -----------------------------
    df["eng_vehicle_per_road_area"] = total_vehicle / (road_area + EPS)
    df["eng_heavy_vehicle_per_road_area"] = heavy_vehicle / (road_area + EPS)
    df["eng_traffic_x_road_dust"] = total_vehicle * df["eng_road_dust_score"]
    df["eng_heavy_x_road_dust"] = heavy_vehicle * df["eng_road_dust_score"]
    df["eng_two_wheeler_x_road_dust"] = two_wheeler * df["eng_road_dust_score"]

    # -----------------------------
    # Met/gas interactions
    # -----------------------------
    temp = pd.to_numeric(df["temp"], errors="coerce").fillna(df["temp"].median())
    rh = pd.to_numeric(df["rh"], errors="coerce").fillna(df["rh"].median())

    df["eng_temp_x_vehicle"] = temp * total_vehicle
    df["eng_rh_x_vehicle"] = rh * total_vehicle
    df["eng_rh_x_heavy"] = rh * heavy_vehicle
    df["eng_temp_x_road_dust"] = temp * df["eng_road_dust_score"]
    df["eng_rh_x_road_dust"] = rh * df["eng_road_dust_score"]

    # -----------------------------
    # OSM engineered summaries
    # -----------------------------
    osm_numeric = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["osm", "highway", "primary", "secondary", "tertiary", "roadarea", "distance", "road_density"]):
            if c != TARGET:
                osm_numeric.append(c)

    # Keep numeric OSM only for simple summaries
    osm_numeric = [c for c in osm_numeric if pd.api.types.is_numeric_dtype(df[c])]

    if osm_numeric:
        osm_df = df[osm_numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["eng_osm_sum"] = osm_df.sum(axis=1)
        df["eng_osm_max"] = osm_df.max(axis=1)
        df["eng_osm_mean"] = osm_df.mean(axis=1)
    else:
        df["eng_osm_sum"] = 0.0
        df["eng_osm_max"] = 0.0
        df["eng_osm_mean"] = 0.0

    return df


def classify_fair_groups(df):
    cols = list(df.columns)

    id_cols = []
    for c in cols:
        lc = c.lower()
        if (
            lc in ["timestamp", "timestamp_parsed", "sample_id", "sample_index"]
            or "path" in lc
            or "file" in lc
        ):
            id_cols.append(c)

    # Target-adjacent particle/density columns: excluded from fair model.
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

    met_gas = [
        c for c in [
            "temp", "rh",
            "value.co_ppb", "value.no2_ppb", "value.so2_ppb",
            "value.o3_ppb_compensated",
            "hour", "minute", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
        ]
        if c in df.columns and c not in banned
    ]

    gps = [
        c for c in cols
        if any(k in c.lower() for k in ["lat", "long", "lon"])
        and c not in banned
    ]

    raw_vehicle = [
        c for c in cols
        if any(k in c.lower() for k in [
            "vehicle", "car", "bus", "truck", "auto", "motorcycle",
            "bicycle", "person", "traffic", "detections"
        ])
        and c not in banned
        and not c.startswith("eng_")
    ]

    raw_road = [
        c for c in cols
        if any(k in c.lower() for k in [
            "road", "brown", "gray", "grey", "dust", "shadow", "glare",
            "haze", "laplacian", "edge", "brightness", "saturation",
            "contrast", "segformer"
        ])
        and c not in banned
        and not c.startswith("eng_")
    ]

    raw_osm = [
        c for c in cols
        if any(k in c.lower() for k in [
            "osm", "highway", "primary", "secondary", "tertiary",
            "residential", "service", "motorway", "trunk", "distance",
            "nearest", "intersection", "junction", "roadarea"
        ])
        and c not in banned
        and not c.startswith("eng_")
    ]

    engineered = [c for c in cols if c.startswith("eng_") and c not in banned]

    engineered_vehicle = [
        c for c in engineered
        if any(k in c.lower() for k in ["vehicle", "traffic", "auto", "car", "heavy", "wheeler", "person"])
    ]

    engineered_road = [
        c for c in engineered
        if any(k in c.lower() for k in ["road", "dust", "brown", "gray", "shadow", "glare", "texture"])
    ]

    engineered_osm = [
        c for c in engineered
        if "osm" in c.lower()
    ]

    engineered_interactions = [
        c for c in engineered
        if "_x_" in c or "per_road_area" in c
    ]

    groups = {
        "mean_baseline": [],
        "tabular_met_gas": met_gas,
        "raw_vehicle_only": raw_vehicle,
        "raw_road_only": raw_road,
        "raw_osm_only": raw_osm,
        "engineered_vehicle_only": engineered_vehicle,
        "engineered_road_only": engineered_road,
        "engineered_source_proxy": sorted(set(engineered_vehicle + engineered_road + engineered_osm + engineered_interactions)),
        "raw_source_proxy": sorted(set(raw_vehicle + raw_road + raw_osm)),
        "tabular_plus_engineered_source": sorted(set(met_gas + engineered_vehicle + engineered_road + engineered_osm + engineered_interactions)),
        "tabular_plus_raw_source": sorted(set(met_gas + raw_vehicle + raw_road + raw_osm)),
        "tabular_plus_raw_and_engineered_source": sorted(set(met_gas + raw_vehicle + raw_road + raw_osm + engineered)),
    }

    diagnostics = {
        "id_cols": id_cols,
        "target_like_excluded_from_fair": target_like,
        "met_gas": met_gas,
        "gps": gps,
        "raw_vehicle": raw_vehicle,
        "raw_road": raw_road,
        "raw_osm": raw_osm,
        "engineered": engineered,
    }

    return groups, diagnostics


def convert_bool(X):
    X = X.copy()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
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

    preprocessor = ColumnTransformer(
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

    return preprocessor


def make_models():
    return {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
    }


def cv_predict_for_feature_set(df, y, feature_set, feature_cols):
    records = []
    pred_frames = []

    if len(feature_cols) == 0:
        pred = np.full_like(y, y.mean(), dtype=float)
        row = {
            "feature_set": feature_set,
            "model": "mean_baseline",
            "n_features": 0,
            "n_rows": len(y),
            "n_splits": N_SPLITS,
        }
        row.update(metric_dict(y, pred))
        records.append(row)

        pred_frames.append(pd.DataFrame({
            "row_index": df.index.values,
            "feature_set": feature_set,
            "model": "mean_baseline",
            "y_true_pm25": y,
            "y_pred_pm25": pred,
            "residual_pred_minus_true": pred - y,
        }))
        return records, pred_frames

    X = convert_bool(df[feature_cols].copy())

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model in make_models().items():
        print("\n" + "-" * 100)
        print(f"START {feature_set} | {model_name} | n_features={len(feature_cols)}")

        pipe = Pipeline([
            ("preprocess", make_preprocessor(X)),
            ("model", model),
        ])

        pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=None)
        pred = np.clip(pred, 0, None)

        row = {
            "feature_set": feature_set,
            "model": model_name,
            "n_features": len(feature_cols),
            "n_rows": len(y),
            "n_splits": N_SPLITS,
        }
        row.update(metric_dict(y, pred))
        records.append(row)

        pred_frames.append(pd.DataFrame({
            "row_index": df.index.values,
            "feature_set": feature_set,
            "model": model_name,
            "y_true_pm25": y,
            "y_pred_pm25": pred,
            "residual_pred_minus_true": pred - y,
        }))

        print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman", "Bias"]})

    return records, pred_frames


def fit_importance_model(df, y, feature_set, feature_cols):
    if len(feature_cols) == 0:
        return pd.DataFrame()

    X = convert_bool(df[feature_cols].copy())

    # For feature importance, use numeric-only processed data for simplicity.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols].copy()
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    X_num = X_num.fillna(X_num.median(numeric_only=True))

    model = ExtraTreesRegressor(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_num, y)

    fi = pd.DataFrame({
        "feature_set": feature_set,
        "feature": numeric_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return fi


def main():
    df = pd.read_csv(DATA)
    df = add_time_features(df)
    df = add_engineered_features(df)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target {TARGET}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].reset_index(drop=True)
    y = df[TARGET].values.astype(float)

    groups, diagnostics = classify_fair_groups(df)

    with open(OUT_GROUPS, "w") as f:
        json.dump({
            "target": TARGET,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "groups": groups,
            "diagnostics": diagnostics,
        }, f, indent=2)

    print("Data:", DATA)
    print("Shape:", df.shape)
    print("Target summary:")
    print(df[TARGET].describe().to_string())

    print("\nFeature groups:")
    for k, v in groups.items():
        print(f"{k}: {len(v)}")

    all_records = []
    all_preds = []
    all_fi = []

    run_order = [
        "mean_baseline",
        "tabular_met_gas",
        "raw_vehicle_only",
        "raw_road_only",
        "raw_osm_only",
        "engineered_vehicle_only",
        "engineered_road_only",
        "engineered_source_proxy",
        "raw_source_proxy",
        "tabular_plus_engineered_source",
        "tabular_plus_raw_source",
        "tabular_plus_raw_and_engineered_source",
    ]

    for fs in run_order:
        cols = groups[fs]
        recs, preds = cv_predict_for_feature_set(df, y, fs, cols)
        all_records.extend(recs)
        all_preds.extend(preds)

        fi = fit_importance_model(df, y, fs, cols)
        if len(fi):
            all_fi.append(fi)

        results = pd.DataFrame(all_records).sort_values("RMSE")
        results.to_csv(OUT_RESULTS, index=False)

        pred_df = pd.concat(all_preds, ignore_index=True)
        pred_df.to_csv(OUT_PREDS, index=False)

        if all_fi:
            fi_df = pd.concat(all_fi, ignore_index=True)
            fi_df.to_csv(OUT_IMPORTANCE, index=False)

    results = pd.DataFrame(all_records).sort_values("RMSE")
    results.to_csv(OUT_RESULTS, index=False)

    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(OUT_PREDS, index=False)

    if all_fi:
        fi_df = pd.concat(all_fi, ignore_index=True)
        fi_df.to_csv(OUT_IMPORTANCE, index=False)

    print("\nFinal engineered fair results:")
    print(results.to_string(index=False))

    print("\nSaved:")
    print(OUT_RESULTS)
    print(OUT_PREDS)
    print(OUT_IMPORTANCE)
    print(OUT_GROUPS)


if __name__ == "__main__":
    main()