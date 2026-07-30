from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from scipy.stats import spearmanr

YOLO_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_vehicle_features_front.csv")
ROAD_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_road_features_front.csv")
EMB_FILE = Path("experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/traqid_paper_resnet50_front_rear_concat_gap_features.npy")

OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "residual_fusion_analysis_results.csv"
PRED_OUT = OUTDIR / "residual_fusion_predictions.csv"

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


def metrics(y_true, y_pred):
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

    brightness = df["road_mean_brightness"] + EPS
    saturation = df["road_mean_saturation"] + EPS
    road_area = df["road_area_ratio"] + EPS

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

    df["vehicle_per_road_area"] = df["total_vehicle_count"] / road_area
    df["heavy_vehicle_per_road_area"] = df["heavy_vehicle_count"] / road_area
    df["two_wheeler_per_road_area"] = df["two_wheeler_count"] / road_area

    df["traffic_x_road_dust"] = df["total_vehicle_count"] * df["road_dust_score"]
    df["heavy_vehicle_x_road_dust"] = df["heavy_vehicle_count"] * df["road_dust_score"]
    df["two_wheeler_x_road_dust"] = df["two_wheeler_count"] * df["road_dust_score"]

    df["humidity_x_road_dust"] = df["Humidity"] * df["road_dust_score"]
    df["temperature_x_road_dust"] = df["Temperature"] * df["road_dust_score"]

    return df


def load_dataset():
    yolo = pd.read_csv(YOLO_CSV)
    yolo = yolo[yolo["yolo_status"] == "success"].copy()

    road = pd.read_csv(ROAD_CSV)
    road = road[road["road_condition_status"] == "success"].copy()
    road = road.rename(columns={"sample_index": "row_id"})
    road = road[["row_id"] + road_raw].copy()

    df = yolo.merge(road, on="row_id", how="inner").reset_index(drop=True)

    for c in vehicle_raw + road_raw + ["Temperature", "Humidity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[vehicle_raw] = df[vehicle_raw].fillna(0)
    df[road_raw] = df[road_raw].fillna(df[road_raw].median(numeric_only=True))
    df[["Temperature", "Humidity"]] = df[["Temperature", "Humidity"]].fillna(
        df[["Temperature", "Humidity"]].median(numeric_only=True)
    )

    return add_engineered_features(df)


def build_matrix(df, X_emb_raw, train_idx, test_idx, feature_cols, use_embedding=False):
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


df = load_dataset()
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

road_features = road_raw + [
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

visual_features = traffic_features + road_features

# ---------------------------------------------------------------------
# Stage 1: tabular model
# ---------------------------------------------------------------------
X_tab_train, X_tab_test = build_matrix(
    df=df,
    X_emb_raw=X_emb_raw,
    train_idx=train_idx,
    test_idx=test_idx,
    feature_cols=base_features,
    use_embedding=False,
)

tab_model = ExtraTreesRegressor(
    n_estimators=700,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

tab_model.fit(X_tab_train, y[train_idx])
tab_pred_train = tab_model.predict(X_tab_train)
tab_pred_test = tab_model.predict(X_tab_test)

train_residual = y[train_idx] - tab_pred_train

# ---------------------------------------------------------------------
# Stage 2: residual models using visual/source-proxy features
# ---------------------------------------------------------------------
experiments = {
    "tabular_only": {
        "pred": tab_pred_test,
    },
}

residual_feature_sets = {
    "traffic_residual": {
        "features": traffic_features,
        "use_embedding": False,
    },
    "road_residual": {
        "features": road_features,
        "use_embedding": False,
    },
    "traffic_road_residual": {
        "features": traffic_features + road_features,
        "use_embedding": False,
    },
    "traffic_road_resnet50_residual": {
        "features": traffic_features + road_features,
        "use_embedding": True,
    },
}

residual_models = {
    "residual_random_forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=16,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    ),
    "residual_extra_trees": ExtraTreesRegressor(
        n_estimators=700,
        max_depth=18,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    ),
    "residual_hgb": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    ),
}

records = []
pred_records = []

row = {
    "experiment": "tabular_only",
    "model": "extra_trees",
    "residual_feature_set": "none",
    "n_train": len(train_idx),
    "n_test": len(test_idx),
}
row.update(metrics(y[test_idx], tab_pred_test))
records.append(row)

pred_records.append(pd.DataFrame({
    "row_id": df.iloc[test_idx]["row_id"].values,
    "created_at": df.iloc[test_idx]["created_at"].values,
    "aqi_cat": df.iloc[test_idx]["aqi_cat"].values,
    "experiment": "tabular_only",
    "y_true_pm25": y[test_idx],
    "y_pred_pm25": tab_pred_test,
    "residual_pred_minus_true": tab_pred_test - y[test_idx],
}))

for fs_name, cfg in residual_feature_sets.items():
    X_vis_train, X_vis_test = build_matrix(
        df=df,
        X_emb_raw=X_emb_raw,
        train_idx=train_idx,
        test_idx=test_idx,
        feature_cols=cfg["features"],
        use_embedding=cfg["use_embedding"],
    )

    for model_name, model in residual_models.items():
        print("\n" + "-" * 100)
        print("Residual fusion:", fs_name, model_name)

        model.fit(X_vis_train, train_residual)
        residual_pred_test = model.predict(X_vis_test)

        final_pred = tab_pred_test + residual_pred_test
        final_pred = np.clip(final_pred, 0, None)

        row = {
            "experiment": f"{fs_name}_{model_name}",
            "model": model_name,
            "residual_feature_set": fs_name,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        row.update(metrics(y[test_idx], final_pred))
        records.append(row)

        pred_records.append(pd.DataFrame({
            "row_id": df.iloc[test_idx]["row_id"].values,
            "created_at": df.iloc[test_idx]["created_at"].values,
            "aqi_cat": df.iloc[test_idx]["aqi_cat"].values,
            "experiment": row["experiment"],
            "y_true_pm25": y[test_idx],
            "y_pred_pm25": final_pred,
            "residual_pred_minus_true": final_pred - y[test_idx],
        }))

        res = pd.DataFrame(records).sort_values("RMSE")
        res.to_csv(OUT, index=False)

        preds = pd.concat(pred_records, ignore_index=True)
        preds.to_csv(PRED_OUT, index=False)

        print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman"]})

res = pd.DataFrame(records).sort_values("RMSE")
res.to_csv(OUT, index=False)

pd.concat(pred_records, ignore_index=True).to_csv(PRED_OUT, index=False)

print("\nFinal residual fusion results:")
print(res.to_string(index=False))
print("\nSaved:")
print(OUT)
print(PRED_OUT)