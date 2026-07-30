from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

YOLO_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_vehicle_features_front.csv")
EMB_FILE = Path("experiments/traqid_pretraining_v1/embeddings/traqid_mobilenetv2_torch_front_rear_concat_embeddings.npy")

OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "random_only_best_model_results.csv"

TARGET = "PM2.5"
EPS = 1e-6
PCA_DIM = 64

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


def add_engineered_features(df):
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


def build_matrix(df, X_emb_raw, train_idx, test_idx, feature_cols, use_embedding):
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    X_train_num = train[feature_cols].values.astype(float)
    X_test_num = test[feature_cols].values.astype(float)

    combined_cat = pd.concat([train[categorical_features], test[categorical_features]], axis=0)
    combined_cat = pd.get_dummies(combined_cat, columns=categorical_features, drop_first=False)

    X_train_cat = combined_cat.iloc[:len(train)].values.astype(float)
    X_test_cat = combined_cat.iloc[len(train):].values.astype(float)

    parts_train = [X_train_num, X_train_cat]
    parts_test = [X_test_num, X_test_cat]

    if use_embedding:
        scaler_emb = StandardScaler()
        pca = PCA(n_components=PCA_DIM, random_state=42)

        emb_train = scaler_emb.fit_transform(X_emb_raw[train_idx])
        emb_train = pca.fit_transform(emb_train)

        emb_test = scaler_emb.transform(X_emb_raw[test_idx])
        emb_test = pca.transform(emb_test)

        parts_train.append(emb_train)
        parts_test.append(emb_test)

    X_train = np.hstack(parts_train)
    X_test = np.hstack(parts_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def make_models():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=700,
            max_depth=28,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=800,
            learning_rate=0.025,
            max_leaf_nodes=63,
            l2_regularization=0.001,
            random_state=42,
        ),
        "ridge": Ridge(alpha=5.0),
    }


df = pd.read_csv(YOLO_CSV)
df = df[df["yolo_status"] == "success"].copy().reset_index(drop=True)

for c in vehicle_raw + ["Temperature", "Humidity", "PM10"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df = add_engineered_features(df)

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

base_features = [
    "Temperature",
    "Humidity",
    "hour_num",
    "month_num",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

traffic_features = vehicle_raw + engineered_vehicle

fair_features = base_features + traffic_features

upper_bound_features = base_features + traffic_features + ["PM10"]

feature_sets = {
    "fair_tabular_traffic": {
        "features": fair_features,
        "use_embedding": False,
        "description": "No PM10, no AQI, no target leakage",
    },
    "fair_tabular_traffic_embeddings": {
        "features": fair_features,
        "use_embedding": True,
        "description": "No PM10/AQI, includes MobileNetV2 embeddings",
    },
    "upper_bound_with_pm10": {
        "features": upper_bound_features,
        "use_embedding": False,
        "description": "Includes PM10 sensor-fusion feature",
    },
    "upper_bound_with_pm10_embeddings": {
        "features": upper_bound_features,
        "use_embedding": True,
        "description": "Includes PM10 + embeddings",
    },
}

X_emb_all = np.load(EMB_FILE)
X_emb_raw = X_emb_all[df["row_id"].values]

y_raw = df[TARGET].values.astype(float)
y_log = np.log1p(y_raw)

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

train_idx = train.index.values
test_idx = test.index.values

records = []

print("Random-only split")
print({"train": len(train_idx), "test": len(test_idx)})

mean_pred = np.full(len(test_idx), np.mean(y_raw[train_idx]), dtype=float)
row = {
    "feature_set": "none",
    "description": "mean baseline",
    "model": "mean_baseline",
    "target_transform": "none",
    "use_embedding": False,
    "n_features": 0,
    "n_train": len(train_idx),
    "n_test": len(test_idx),
    "elapsed_sec": 0,
}
row.update(get_metrics(y_raw[test_idx], mean_pred))
records.append(row)

for fs_name, cfg in feature_sets.items():
    X_train, X_test = build_matrix(
        df=df,
        X_emb_raw=X_emb_raw,
        train_idx=train_idx,
        test_idx=test_idx,
        feature_cols=cfg["features"],
        use_embedding=cfg["use_embedding"],
    )

    for target_transform in ["none", "log1p"]:
        y_train = y_log[train_idx] if target_transform == "log1p" else y_raw[train_idx]

        for model_name, model in make_models().items():
            print("\n" + "-" * 100)
            print("START:", fs_name, model_name, "target:", target_transform)
            print("Feature matrix:", X_train.shape, X_test.shape)

            start = time.time()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            if target_transform == "log1p":
                pred = np.expm1(pred)

            pred = np.clip(pred, 0, None)
            elapsed = time.time() - start

            row = {
                "feature_set": fs_name,
                "description": cfg["description"],
                "model": model_name,
                "target_transform": target_transform,
                "use_embedding": cfg["use_embedding"],
                "n_features": X_train.shape[1],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "elapsed_sec": round(elapsed, 2),
            }
            row.update(get_metrics(y_raw[test_idx], pred))
            records.append(row)

            res = pd.DataFrame(records).sort_values("RMSE")
            res.to_csv(OUT, index=False)

            print("DONE:", round(elapsed, 2), "sec")
            print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman"]})
            print("Checkpoint saved:", OUT)

res = pd.DataFrame(records).sort_values("RMSE")
res.to_csv(OUT, index=False)

print("\nFinal results:")
print(res.to_string(index=False))
print("\nSaved:", OUT)