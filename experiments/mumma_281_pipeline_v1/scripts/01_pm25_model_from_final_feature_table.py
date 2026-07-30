from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
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

OUT_RESULTS = REPORTS / "pm25_fair_ablation_cv_results.csv"
OUT_PREDS = REPORTS / "pm25_fair_ablation_cv_predictions.csv"
OUT_FEATURE_GROUPS = REPORTS / "pm25_fair_feature_group_columns.json"

TARGET = "value.sPM2"
RANDOM_STATE = 42
N_SPLITS = 5


def safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_spearman(y_true, y_pred):
    if np.std(y_pred) == 0:
        return np.nan
    return float(spearmanr(y_true, y_pred).correlation)


def get_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": safe_rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": safe_spearman(y_true, y_pred),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def add_time_features(df):
    df = df.copy()

    if "timestamp" in df.columns:
        # Your timestamp format is like 23-02-2026 11:54
        dt = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        df["timestamp_parsed"] = dt
        df["hour"] = dt.dt.hour
        df["minute"] = dt.dt.minute
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df


def classify_columns(df):
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}

    target_like = []
    for c in cols:
        lc = lower[c]
        if c == TARGET:
            continue
        # These are particle sensor features. Keep them only for upper-bound sensor-fusion.
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

    id_cols = [
        c for c in cols
        if c.lower() in {
            "timestamp", "timestamp_parsed", "sample_id", "sample_index",
            "sensor_timestamp", "original_sensor_row_id", "frame_path",
            "processed_frame_path", "image_path", "video_path"
        }
        or "path" in c.lower()
        or "file" in c.lower()
    ]

    gps_cols = [
        c for c in cols
        if any(k in lower[c] for k in ["lat", "long", "lon", "gps"])
        and c not in id_cols
    ]

    met_cols = [
        c for c in cols
        if c in ["temp", "rh", "Temperature", "Humidity", "hour", "minute", "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    ]

    gas_cols = [
        c for c in cols
        if any(k in lower[c] for k in ["co_ppb", "no2", "so2", "o3"])
    ]

    vehicle_cols = [
        c for c in cols
        if any(k in lower[c] for k in [
            "vehicle", "car", "bus", "truck", "auto", "motorcycle",
            "bicycle", "person", "traffic", "detections"
        ])
    ]

    road_cols = [
        c for c in cols
        if any(k in lower[c] for k in [
            "road", "segformer", "brown", "gray", "grey", "dust",
            "shadow", "glare", "haze", "laplacian", "edge", "brightness",
            "saturation", "contrast"
        ])
    ]

    osm_cols = [
        c for c in cols
        if any(k in lower[c] for k in [
            "osm", "highway", "primary", "secondary", "tertiary",
            "residential", "service", "motorway", "trunk", "distance",
            "nearest", "intersection", "junction", "road_density"
        ])
    ]

    # Remove target-like particle/density columns from fair groups.
    def clean(group):
        banned = set([TARGET]) | set(id_cols) | set(target_like)
        return [c for c in group if c not in banned and c in df.columns]

    groups = {
        "tabular_met_gas": clean(met_cols + gas_cols),
        "gps_only": clean(gps_cols),
        "vehicle_only": clean(vehicle_cols),
        "road_only": clean(road_cols),
        "osm_only": clean(osm_cols),
    }

    groups["vehicle_road"] = sorted(set(groups["vehicle_only"] + groups["road_only"]))
    groups["vehicle_road_osm"] = sorted(set(groups["vehicle_only"] + groups["road_only"] + groups["osm_only"]))
    groups["tabular_vehicle_road_osm"] = sorted(set(groups["tabular_met_gas"] + groups["vehicle_only"] + groups["road_only"] + groups["osm_only"]))

    # Upper-bound: includes particle sensor/density features also.
    upper = []
    for c in cols:
        if c == TARGET:
            continue
        if c in id_cols:
            continue
        if c == "timestamp_parsed":
            continue
        upper.append(c)
    #groups["upper_bound_all_sensor_visual_osm"] = upper

    return groups, {
        "id_cols": id_cols,
        "target_like_excluded_from_fair": target_like,
        "gps_cols": gps_cols,
        "met_cols": met_cols,
        "gas_cols": gas_cols,
        "vehicle_cols": vehicle_cols,
        "road_cols": road_cols,
        "osm_cols": osm_cols,
    }


def make_preprocessor(X):
    X = X.copy()

    # Convert bool columns to integers because SimpleImputer can fail on bool dtype
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        X[c] = X[c].astype("int64")

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
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=2,
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


def run_cv(df, y, feature_set_name, feature_cols):
    records = []
    pred_records = []

    if len(feature_cols) == 0:
        pred = np.full_like(y, fill_value=np.mean(y), dtype=float)
        row = {
            "feature_set": feature_set_name,
            "model": "mean_baseline",
            "n_features": 0,
            "n_rows": len(y),
            "n_splits": N_SPLITS,
        }
        row.update(get_metrics(y, pred))
        records.append(row)

        pred_records.append(pd.DataFrame({
            "feature_set": feature_set_name,
            "model": "mean_baseline",
            "row_index": df.index.values,
            "y_true_pm25": y,
            "y_pred_pm25": pred,
            "residual_pred_minus_true": pred - y,
        }))
        return records, pred_records

    X = df[feature_cols].copy()
    # Convert boolean columns to integers for sklearn preprocessing
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        X[c] = X[c].astype("int64")

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = make_models()

    for model_name, model in models.items():
        print("\n" + "-" * 100)
        print(f"START {feature_set_name} | {model_name} | n_features={len(feature_cols)}")

        pipe = Pipeline([
            ("preprocess", make_preprocessor(X)),
            ("model", model),
        ])

        pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=None)
        pred = np.clip(pred, 0, None)

        row = {
            "feature_set": feature_set_name,
            "model": model_name,
            "n_features": len(feature_cols),
            "n_rows": len(y),
            "n_splits": N_SPLITS,
        }
        row.update(get_metrics(y, pred))
        records.append(row)

        pred_records.append(pd.DataFrame({
            "feature_set": feature_set_name,
            "model": model_name,
            "row_index": df.index.values,
            "y_true_pm25": y,
            "y_pred_pm25": pred,
            "residual_pred_minus_true": pred - y,
        }))

        print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman", "Bias"]})

    return records, pred_records


def main():
    df = pd.read_csv(DATA)
    df = add_time_features(df)

    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}. Available PM-like columns: {[c for c in df.columns if 'pm' in c.lower()]}")

    df = df.copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].reset_index(drop=True)

    y = df[TARGET].values.astype(float)

    groups, diagnostics = classify_columns(df)

    with open(OUT_FEATURE_GROUPS, "w") as f:
        json.dump(
            {
                "groups": groups,
                "diagnostics": diagnostics,
                "target": TARGET,
                "n_rows": len(df),
                "n_cols": df.shape[1],
            },
            f,
            indent=2,
        )

    print("Data:", DATA)
    print("Shape:", df.shape)
    print("Target:", TARGET)
    print("\nTarget summary:")
    print(df[TARGET].describe().to_string())

    print("\nFeature groups:")
    for k, v in groups.items():
        print(f"{k}: {len(v)}")

    all_records = []
    all_preds = []

    # Mean baseline first
    rec, pred = run_cv(df, y, "mean_baseline", [])
    all_records += rec
    all_preds += pred

    run_order = [
    "tabular_met_gas",
    "gps_only",
    "vehicle_only",
    "road_only",
    "osm_only",
    "vehicle_road",
    "vehicle_road_osm",
    "tabular_vehicle_road_osm",
]

    for fs in run_order:
        cols = groups.get(fs, [])
        if len(cols) == 0:
            print(f"Skipping {fs}: no columns")
            continue
        rec, pred = run_cv(df, y, fs, cols)
        all_records += rec
        all_preds += pred

        results = pd.DataFrame(all_records).sort_values("RMSE")
        results.to_csv(OUT_RESULTS, index=False)

        preds = pd.concat(all_preds, ignore_index=True)
        preds.to_csv(OUT_PREDS, index=False)

    results = pd.DataFrame(all_records).sort_values("RMSE")
    results.to_csv(OUT_RESULTS, index=False)

    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_csv(OUT_PREDS, index=False)

    print("\nFinal results:")
    print(results.to_string(index=False))

    print("\nSaved:")
    print(OUT_RESULTS)
    print(OUT_PREDS)
    print(OUT_FEATURE_GROUPS)


if __name__ == "__main__":
    main()