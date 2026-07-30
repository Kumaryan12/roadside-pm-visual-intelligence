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
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("experiments/mumma_281_pipeline_v1")
DATA = ROOT / "data/processed/final_feature_table.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

OUT_RESULTS = REPORTS / "pm25_residual_fusion_results.csv"
OUT_PREDS = REPORTS / "pm25_residual_fusion_predictions.csv"
OUT_GROUPS = REPORTS / "pm25_residual_fusion_feature_groups.json"

TARGET = "value.sPM2"
RANDOM_STATE = 42
N_SPLITS = 5
EPS = 1e-6


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

    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df


def find_cols(df, patterns):
    return [
        c for c in df.columns
        if any(p in c.lower() for p in patterns)
    ]


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

    auto = safe_col(df, [
        "idd_auto_rickshaw_count_sum",
        "auto_count_sum",
        "auto_rickshaw_count_sum",
    ])

    car = safe_col(df, [
        "idd_car_count_sum",
        "car_count_sum",
    ])

    truck = safe_col(df, [
        "idd_truck_count_sum",
        "truck_count_sum",
    ])

    bus = safe_col(df, [
        "idd_bus_count_sum",
        "bus_count_sum",
    ])

    motorcycle = safe_col(df, [
        "idd_motorcycle_count_sum",
        "motorcycle_count_sum",
        "two_wheeler_count_sum",
    ])

    bicycle = safe_col(df, [
        "idd_bicycle_count_sum",
        "bicycle_count_sum",
    ])

    person = safe_col(df, [
        "idd_person_count_sum",
        "person_count_sum",
    ])

    vehicle_count_cols = find_cols(df, ["count_sum"])
    vehicle_count_cols = [
        c for c in vehicle_count_cols
        if any(k in c.lower() for k in [
            "idd_", "car", "bus", "truck", "auto",
            "motorcycle", "bicycle", "person"
        ])
    ]

    if vehicle_count_cols:
        total_vehicle = (
            df[vehicle_count_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
        )
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

    osm_numeric = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in [
            "osm", "highway", "primary", "secondary", "tertiary",
            "roadarea", "distance", "road_density", "bus_stop",
            "commercial", "retail", "human_activity"
        ]):
            if pd.api.types.is_numeric_dtype(df[c]):
                osm_numeric.append(c)

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


def classify_groups(df):
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
            "value.co_ppb",
            "value.no2_ppb",
            "value.so2_ppb",
            "value.o3_ppb_compensated",
            "hour", "minute", "hour_sin", "hour_cos",
            "minute_sin", "minute_cos",
        ]
        if c in df.columns and c not in banned
    ]

    raw_vehicle = [
        c for c in cols
        if any(k in c.lower() for k in [
            "vehicle", "car", "bus", "truck", "auto",
            "motorcycle", "bicycle", "person", "traffic",
            "detections"
        ])
        and c not in banned
        and not c.startswith("eng_")
    ]

    raw_road = [
        c for c in cols
        if any(k in c.lower() for k in [
            "road", "brown", "gray", "grey", "dust", "shadow",
            "glare", "haze", "laplacian", "edge",
            "brightness", "saturation", "contrast", "segformer"
        ])
        and c not in banned
        and not c.startswith("eng_")
    ]

    raw_osm = [
        c for c in cols
        if any(k in c.lower() for k in [
            "osm", "highway", "primary", "secondary", "tertiary",
            "residential", "service", "motorway", "trunk",
            "distance", "nearest", "intersection", "junction",
            "roadarea", "bus_stop", "commercial", "retail",
            "human_activity"
        ])
        and c not in banned
        and not c.startswith("eng_")
    ]

    engineered = [
        c for c in cols
        if c.startswith("eng_") and c not in banned
    ]

    engineered_vehicle = [
        c for c in engineered
        if any(k in c.lower() for k in [
            "vehicle", "traffic", "auto", "car", "heavy",
            "wheeler", "person"
        ])
    ]

    engineered_road = [
        c for c in engineered
        if any(k in c.lower() for k in [
            "road", "dust", "brown", "gray", "shadow",
            "glare", "texture"
        ])
    ]

    engineered_osm = [
        c for c in engineered
        if "osm" in c.lower()
    ]

    engineered_interactions = [
        c for c in engineered
        if "_x_" in c or "per_road_area" in c
    ]

    raw_source = sorted(set(raw_vehicle + raw_road + raw_osm))
    engineered_source = sorted(set(
        engineered_vehicle
        + engineered_road
        + engineered_osm
        + engineered_interactions
    ))

    groups = {
        "tabular": tabular,
        "raw_source": raw_source,
        "engineered_source": engineered_source,
        "raw_and_engineered_source": sorted(set(raw_source + engineered_source)),
    }

    diagnostics = {
        "id_cols": id_cols,
        "target_like_excluded": target_like,
        "tabular": tabular,
        "raw_vehicle": raw_vehicle,
        "raw_road": raw_road,
        "raw_osm": raw_osm,
        "engineered_source": engineered_source,
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


def make_pipeline(model):
    return Pipeline([
        ("preprocess", make_preprocessor(pd.DataFrame())),
        ("model", model),
    ])


def make_model(model_name):
    if model_name == "ridge":
        return Ridge(alpha=10.0)

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        )

    raise ValueError(model_name)


def fit_predict_fold(X_train, y_train, X_test, model_name):
    X_train = convert_bool(X_train)
    X_test = convert_bool(X_test)

    model = make_model(model_name)

    pipe = Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return pred


def main():
    df = pd.read_csv(DATA)
    df = add_time_features(df)
    df = add_engineered_features(df)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].reset_index(drop=True)

    y = df[TARGET].values.astype(float)

    groups, diagnostics = classify_groups(df)

    with open(OUT_GROUPS, "w") as f:
        json.dump({
            "target": TARGET,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "groups": groups,
            "diagnostics": diagnostics,
        }, f, indent=2)

    print("Shape:", df.shape)
    print("Target summary:")
    print(df[TARGET].describe().to_string())

    print("\nFeature groups:")
    for k, v in groups.items():
        print(k, len(v))

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    residual_sets = {
        "raw_source_residual": groups["raw_source"],
        "engineered_source_residual": groups["engineered_source"],
        "raw_and_engineered_source_residual": groups["raw_and_engineered_source"],
    }

    residual_models = ["ridge", "random_forest", "extra_trees", "hgb"]

    all_records = []
    all_pred_frames = []

    # Baseline tabular out-of-fold predictions
    tabular_pred = np.zeros(len(df), dtype=float)

    print("\nTraining tabular baseline OOF...")
    for fold, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
        X_train = df.iloc[train_idx][groups["tabular"]].copy()
        X_test = df.iloc[test_idx][groups["tabular"]].copy()
        y_train = y[train_idx]

        pred = fit_predict_fold(X_train, y_train, X_test, "extra_trees")
        tabular_pred[test_idx] = np.clip(pred, 0, None)

    row = {
        "experiment": "tabular_baseline",
        "base_model": "extra_trees",
        "residual_model": "none",
        "residual_feature_set": "none",
        "n_base_features": len(groups["tabular"]),
        "n_residual_features": 0,
        "n_rows": len(df),
        "n_splits": N_SPLITS,
    }
    row.update(metrics(y, tabular_pred))
    all_records.append(row)

    all_pred_frames.append(pd.DataFrame({
        "row_index": df.index.values,
        "experiment": "tabular_baseline",
        "y_true_pm25": y,
        "y_pred_pm25": tabular_pred,
        "residual_pred_minus_true": tabular_pred - y,
    }))

    print("Tabular baseline:", {k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman", "Bias"]})

    # Residual fusion
    for residual_set_name, residual_cols in residual_sets.items():
        for residual_model_name in residual_models:
            print("\n" + "-" * 100)
            print(f"Residual fusion: {residual_set_name} | {residual_model_name}")

            final_pred = np.zeros(len(df), dtype=float)
            residual_oof_pred = np.zeros(len(df), dtype=float)

            for fold, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
                # Base tabular model inside fold
                X_tab_train = df.iloc[train_idx][groups["tabular"]].copy()
                X_tab_test = df.iloc[test_idx][groups["tabular"]].copy()

                y_train = y[train_idx]
                y_test = y[test_idx]

                base_train_pred = fit_predict_fold(
                    X_tab_train, y_train, X_tab_train, "extra_trees"
                )
                base_test_pred = fit_predict_fold(
                    X_tab_train, y_train, X_tab_test, "extra_trees"
                )

                train_residual = y_train - base_train_pred

                X_res_train = df.iloc[train_idx][residual_cols].copy()
                X_res_test = df.iloc[test_idx][residual_cols].copy()

                residual_pred = fit_predict_fold(
                    X_res_train,
                    train_residual,
                    X_res_test,
                    residual_model_name,
                )

                pred = base_test_pred + residual_pred
                pred = np.clip(pred, 0, None)

                final_pred[test_idx] = pred
                residual_oof_pred[test_idx] = residual_pred

            row = {
                "experiment": f"{residual_set_name}_{residual_model_name}",
                "base_model": "extra_trees",
                "residual_model": residual_model_name,
                "residual_feature_set": residual_set_name,
                "n_base_features": len(groups["tabular"]),
                "n_residual_features": len(residual_cols),
                "n_rows": len(df),
                "n_splits": N_SPLITS,
            }
            row.update(metrics(y, final_pred))
            all_records.append(row)

            all_pred_frames.append(pd.DataFrame({
                "row_index": df.index.values,
                "experiment": row["experiment"],
                "y_true_pm25": y,
                "y_pred_pm25": final_pred,
                "residual_pred_minus_true": final_pred - y,
                "residual_correction_pred": residual_oof_pred,
            }))

            results = pd.DataFrame(all_records).sort_values("RMSE")
            results.to_csv(OUT_RESULTS, index=False)

            pred_df = pd.concat(all_pred_frames, ignore_index=True)
            pred_df.to_csv(OUT_PREDS, index=False)

            print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman", "Bias"]})

    results = pd.DataFrame(all_records).sort_values("RMSE")
    results.to_csv(OUT_RESULTS, index=False)

    pred_df = pd.concat(all_pred_frames, ignore_index=True)
    pred_df.to_csv(OUT_PREDS, index=False)

    print("\nFinal residual fusion results:")
    print(results.to_string(index=False))

    print("\nSaved:")
    print(OUT_RESULTS)
    print(OUT_PREDS)
    print(OUT_GROUPS)


if __name__ == "__main__":
    main()