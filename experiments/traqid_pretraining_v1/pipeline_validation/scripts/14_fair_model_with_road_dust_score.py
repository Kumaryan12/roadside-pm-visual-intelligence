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
ROAD_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_road_features_front.csv")
EMB_FILE = Path("experiments/traqid_pretraining_v1/embeddings/traqid_mobilenetv2_torch_front_rear_concat_embeddings.npy")

OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "fair_model_with_road_dust_score_results.csv"

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

road_raw = [
    "road_area_ratio",
    "road_mean_brightness",
    "road_mean_saturation",
    "road_contrast_std",
    "road_shadow_ratio",
    "road_glare_ratio",
    "road_brown_pixel_ratio",
    "road_gray_dry_pixel_ratio",
    "road_edge_density",
    "road_laplacian_std",
    "road_haze_flatness_proxy",
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

    # ---------------------------------------------------------------------
    # Road / visual dust features
    # ---------------------------------------------------------------------
    # Important: these are visual proxies, not chemical source-apportionment.
    road_area = df["road_area_ratio"] + EPS
    brightness = df["road_mean_brightness"] + EPS
    saturation = df["road_mean_saturation"] + EPS

    df["visible_road_quality"] = (
        df["road_area_ratio"]
        * (1.0 - df["road_shadow_ratio"].clip(0, 1))
        * (1.0 - df["road_glare_ratio"].clip(0, 1))
    )

    df["road_brown_exposure"] = df["road_brown_pixel_ratio"] * df["road_area_ratio"]
    df["road_gray_dry_exposure"] = df["road_gray_dry_pixel_ratio"] * df["road_area_ratio"]

    df["brightness_corrected_brown"] = df["road_brown_pixel_ratio"] / brightness
    df["saturation_corrected_brown"] = df["road_brown_pixel_ratio"] / saturation

    df["road_texture_score"] = df["road_edge_density"] + df["road_laplacian_std"]

    # Main road-dust visual score.
    # This combines brown/dry pixels, visible road exposure, and texture,
    # while penalizing shadow/glare and extremely low road visibility.
    df["road_dust_score_raw"] = (
        0.45 * df["road_brown_exposure"]
        + 0.25 * df["road_gray_dry_exposure"]
        + 0.20 * df["brightness_corrected_brown"]
        + 0.10 * df["road_texture_score"]
    )

    df["road_dust_score"] = df["road_dust_score_raw"] * (
        1.0 - df["road_shadow_ratio"].clip(0, 1)
    ) * (
        1.0 - df["road_glare_ratio"].clip(0, 1)
    )

    # Road-normalized traffic exposure
    df["vehicle_per_road_area"] = df["total_vehicle_count"] / road_area
    df["heavy_vehicle_per_road_area"] = df["heavy_vehicle_count"] / road_area
    df["two_wheeler_per_road_area"] = df["two_wheeler_count"] / road_area

    # Road dust × traffic interactions
    df["traffic_x_road_dust"] = df["total_vehicle_count"] * df["road_dust_score"]
    df["heavy_vehicle_x_road_dust"] = df["heavy_vehicle_count"] * df["road_dust_score"]
    df["two_wheeler_x_road_dust"] = df["two_wheeler_count"] * df["road_dust_score"]

    # Meteorology × road dust interactions
    df["humidity_x_road_dust"] = df["Humidity"] * df["road_dust_score"]
    df["temperature_x_road_dust"] = df["Temperature"] * df["road_dust_score"]

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
        emb_scaler = StandardScaler()
        pca = PCA(n_components=PCA_DIM, random_state=42)

        emb_train = emb_scaler.fit_transform(X_emb_raw[train_idx])
        emb_train = pca.fit_transform(emb_train)

        emb_test = emb_scaler.transform(X_emb_raw[test_idx])
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
        "ridge": Ridge(alpha=5.0),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=24,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=700,
            learning_rate=0.03,
            max_leaf_nodes=63,
            l2_regularization=0.001,
            random_state=42,
        ),
    }


print("Loading YOLO features...")
yolo = pd.read_csv(YOLO_CSV)
yolo = yolo[yolo["yolo_status"] == "success"].copy()

print("Loading road features...")
road = pd.read_csv(ROAD_CSV)
road = road[road["road_condition_status"] == "success"].copy()
road = road.rename(columns={"sample_index": "row_id"})

road = road[["row_id"] + road_raw].copy()

print("Merging...")
df = yolo.merge(road, on="row_id", how="inner")
df = df.reset_index(drop=True)

print("Merged rows:", len(df))

for c in vehicle_raw + road_raw + ["Temperature", "Humidity"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[vehicle_raw] = df[vehicle_raw].fillna(0)
df[road_raw] = df[road_raw].fillna(df[road_raw].median(numeric_only=True))
df[["Temperature", "Humidity"]] = df[["Temperature", "Humidity"]].fillna(
    df[["Temperature", "Humidity"]].median(numeric_only=True)
)

df = add_engineered_features(df)

print("\nRoad dust score summary:")
print(df[["road_dust_score", "road_brown_pixel_ratio", "road_gray_dry_pixel_ratio", "road_area_ratio"]].describe().T)

X_emb_all = np.load(EMB_FILE)
X_emb_raw = X_emb_all[df["row_id"].values]

y = df[TARGET].values.astype(float)

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

print("\nRandom split:")
print({"train": len(train_idx), "test": len(test_idx)})

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

traffic_features = vehicle_raw + [
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

road_dust_features = road_raw + [
    "visible_road_quality",
    "road_brown_exposure",
    "road_gray_dry_exposure",
    "brightness_corrected_brown",
    "saturation_corrected_brown",
    "road_texture_score",
    "road_dust_score",
    "vehicle_per_road_area",
    "heavy_vehicle_per_road_area",
    "two_wheeler_per_road_area",
    "traffic_x_road_dust",
    "heavy_vehicle_x_road_dust",
    "two_wheeler_x_road_dust",
    "humidity_x_road_dust",
    "temperature_x_road_dust",
]

feature_sets = {
    "mean_baseline": {
        "features": [],
        "use_embedding": False,
    },
    "tabular_only": {
        "features": base_features,
        "use_embedding": False,
    },
    "traffic_only": {
        "features": traffic_features,
        "use_embedding": False,
    },
    "road_dust_only": {
        "features": road_dust_features,
        "use_embedding": False,
    },
    "tabular_plus_traffic": {
        "features": base_features + traffic_features,
        "use_embedding": False,
    },
    "tabular_plus_road_dust": {
        "features": base_features + road_dust_features,
        "use_embedding": False,
    },
    "traffic_plus_road_dust": {
        "features": traffic_features + road_dust_features,
        "use_embedding": False,
    },
    "tabular_plus_traffic_plus_road_dust": {
        "features": base_features + traffic_features + road_dust_features,
        "use_embedding": False,
    },
    "full_fair_with_embeddings": {
        "features": base_features + traffic_features + road_dust_features,
        "use_embedding": True,
    },
}

records = []

for fs_name, cfg in feature_sets.items():
    if fs_name == "mean_baseline":
        pred = np.full(len(test_idx), np.mean(y[train_idx]), dtype=float)
        row = {
            "feature_set": fs_name,
            "model": "mean_baseline",
            "use_embedding": False,
            "n_features": 0,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "elapsed_sec": 0,
        }
        row.update(get_metrics(y[test_idx], pred))
        records.append(row)
        continue

    X_train, X_test = build_matrix(
        df=df,
        X_emb_raw=X_emb_raw,
        train_idx=train_idx,
        test_idx=test_idx,
        feature_cols=cfg["features"],
        use_embedding=cfg["use_embedding"],
    )

    for model_name, model in make_models().items():
        print("\n" + "-" * 100)
        print("START:", fs_name, model_name)
        print("Feature matrix:", X_train.shape, X_test.shape)

        start = time.time()
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        pred = np.clip(pred, 0, None)
        elapsed = time.time() - start

        row = {
            "feature_set": fs_name,
            "model": model_name,
            "use_embedding": cfg["use_embedding"],
            "n_features": X_train.shape[1],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "elapsed_sec": round(elapsed, 2),
        }
        row.update(get_metrics(y[test_idx], pred))
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

corr_cols = base_features + traffic_features + road_dust_features + [TARGET]
corr = df[corr_cols].corr(numeric_only=True)
corr_out = OUTDIR / "fair_road_dust_feature_correlations.csv"
corr.to_csv(corr_out)

print("\nTop absolute correlations with PM2.5:")
print(
    corr[TARGET]
    .drop(TARGET)
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .head(40)
    .to_string()
)
print("\nSaved:", corr_out)