from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

YOLO_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_vehicle_features_front.csv")
OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "vehicle_engineered_model_results.csv"
TARGET = "PM2.5"
EPS = 1e-6

vehicle_raw = [
    "car_count",
    "bus_count",
    "truck_count",
    "auto_count",
    "motorcycle_count",
    "bicycle_count",
    "person_count",
    "total_vehicle_count",
    "heavy_vehicle_count",
    "two_wheeler_count",
    "detections_total_raw",
]

categorical_features = ["Season", "Day_or_Night"]

def get_metrics(y_true, y_pred):
    sp = np.nan if np.std(y_pred) == 0 else spearmanr(y_true, y_pred).correlation
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Spearman": sp,
    }

def make_random_split(df):
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["aqi_cat"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.20,
        random_state=42,
        stratify=train_val["aqi_cat"],
    )
    return train.index.values, val.index.values, test.index.values

def make_chrono_split(df):
    train_idx = df.index[df["split_date_chrono"] == "train"].values
    val_idx = df.index[df["split_date_chrono"] == "val"].values
    test_idx = df.index[df["split_date_chrono"] == "test"].values
    return train_idx, val_idx, test_idx

def add_features(df):
    df = df.copy()

    df["created_at_parsed"] = pd.to_datetime(df["created_at"])
    df["hour_num"] = df["created_at_parsed"].dt.hour
    df["month_num"] = df["created_at_parsed"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_num"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_num"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    total = df["total_vehicle_count"] + EPS

    df["heavy_vehicle_ratio"] = df["heavy_vehicle_count"] / total
    df["two_wheeler_ratio"] = df["two_wheeler_count"] / total
    df["auto_ratio"] = df["auto_count"] / total
    df["car_ratio"] = df["car_count"] / total
    df["truck_ratio"] = df["truck_count"] / total
    df["bus_ratio"] = df["bus_count"] / total
    df["person_per_vehicle"] = df["person_count"] / total

    df["traffic_mix_index"] = (
        2.0 * df["heavy_vehicle_count"]
        + 1.5 * df["auto_count"]
        + 1.0 * df["car_count"]
        + 0.8 * df["two_wheeler_count"]
    )

    df["heavy_traffic_score"] = df["heavy_vehicle_count"] * df["total_vehicle_count"]
    df["two_wheeler_traffic_score"] = df["two_wheeler_count"] * df["total_vehicle_count"]

    df["humidity_x_total_vehicle"] = df["Humidity"] * df["total_vehicle_count"]
    df["humidity_x_heavy_vehicle"] = df["Humidity"] * df["heavy_vehicle_count"]
    df["temperature_x_total_vehicle"] = df["Temperature"] * df["total_vehicle_count"]

    return df

def build_matrix(df, train_idx, test_idx, feature_cols, use_cat=True):
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    X_train_num = train[feature_cols].values.astype(float)
    X_test_num = test[feature_cols].values.astype(float)

    if use_cat:
        combined = pd.concat([train[categorical_features], test[categorical_features]], axis=0)
        combined = pd.get_dummies(combined, columns=categorical_features, drop_first=False)

        X_train_cat = combined.iloc[:len(train)].values.astype(float)
        X_test_cat = combined.iloc[len(train):].values.astype(float)

        X_train = np.hstack([X_train_num, X_train_cat])
        X_test = np.hstack([X_test_num, X_test_cat])
    else:
        X_train = X_train_num
        X_test = X_test_num

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def make_model(name):
    if name == "ridge":
        return Ridge(alpha=10.0)

    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            random_state=42,
        )

    if name == "rf":
        return RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(name)

df = pd.read_csv(YOLO_CSV)
df = df[df["yolo_status"] == "success"].copy().reset_index(drop=True)

for c in vehicle_raw + ["Temperature", "Humidity"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df = add_features(df)

engineered_vehicle = [
    "heavy_vehicle_ratio",
    "two_wheeler_ratio",
    "auto_ratio",
    "car_ratio",
    "truck_ratio",
    "bus_ratio",
    "person_per_vehicle",
    "traffic_mix_index",
    "heavy_traffic_score",
    "two_wheeler_traffic_score",
    "humidity_x_total_vehicle",
    "humidity_x_heavy_vehicle",
    "temperature_x_total_vehicle",
]

tabular_features = [
    "Temperature",
    "Humidity",
    "hour_num",
    "month_num",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

feature_sets = {
    "tabular_only": tabular_features,
    "vehicle_raw_only": vehicle_raw,
    "vehicle_engineered_only": vehicle_raw + engineered_vehicle,
    "tabular_plus_vehicle_raw": tabular_features + vehicle_raw,
    "tabular_plus_vehicle_engineered": tabular_features + vehicle_raw + engineered_vehicle,
}

models = ["ridge", "hgb", "rf", "extra_trees"]

y = df[TARGET].values.astype(float)

splits = {
    "random_stratified": make_random_split(df),
    "chronological": make_chrono_split(df),
}

records = []

for split_name, (train_idx, val_idx, test_idx) in splits.items():
    print("\n" + "=" * 100)
    print("Split:", split_name)
    print("Rows:", {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)})

    pred = np.full(len(test_idx), np.mean(y[train_idx]), dtype=float)
    row = {
        "split": split_name,
        "feature_set": "none",
        "model": "mean_baseline",
        "n_features": 0,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "elapsed_sec": 0,
    }
    row.update(get_metrics(y[test_idx], pred))
    records.append(row)

    for fs_name, cols in feature_sets.items():
        for model_name in models:
            print("\n" + "-" * 100)
            print("START:", split_name, fs_name, model_name)
            start = time.time()

            X_train, X_test = build_matrix(df, train_idx, test_idx, cols, use_cat=True)
            print("Feature matrix:", X_train.shape, X_test.shape)

            model = make_model(model_name)
            model.fit(X_train, y[train_idx])
            pred = model.predict(X_test)

            elapsed = time.time() - start

            row = {
                "split": split_name,
                "feature_set": fs_name,
                "model": model_name,
                "n_features": X_train.shape[1],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "elapsed_sec": round(elapsed, 2),
            }
            row.update(get_metrics(y[test_idx], pred))
            records.append(row)

            pd.DataFrame(records).sort_values(["split", "RMSE"]).to_csv(OUT, index=False)

            print("DONE:", split_name, fs_name, model_name, round(elapsed, 2), "sec")
            print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman"]})
            print("Checkpoint saved:", OUT)

res = pd.DataFrame(records).sort_values(["split", "RMSE"])
res.to_csv(OUT, index=False)

print("\nFinal results:")
print(res.to_string(index=False))
print("\nSaved:", OUT)

corr_cols = vehicle_raw + engineered_vehicle + tabular_features + [TARGET, "PM10"]
corr = df[corr_cols].corr(numeric_only=True)
corr_out = OUTDIR / "vehicle_engineered_feature_correlations.csv"
corr.to_csv(corr_out)

print("\nTop absolute correlations with PM2.5:")
print(
    corr[TARGET]
    .drop(TARGET)
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .head(30)
    .to_string()
)

print("\nSaved:", corr_out)