from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

YOLO_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_vehicle_features_front.csv")
EMB_FILE = Path("experiments/traqid_pretraining_v1/embeddings/traqid_mobilenetv2_torch_front_rear_concat_embeddings.npy")

OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "best_fair_traqid_model_results.csv"
PRED_OUT = OUTDIR / "best_fair_traqid_predictions.csv"

TARGET = "PM2.5"
EPS = 1e-6
PCA_DIM = 128

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

    return df


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


def build_feature_matrix(df, X_emb_raw, train_idx, test_idx, feature_cols, use_embedding=True):
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
        n_components = min(PCA_DIM, X_emb_raw.shape[1], len(train_idx) - 1)

        emb_scaler = StandardScaler()
        emb_pca = PCA(n_components=n_components, random_state=42)

        emb_train = emb_scaler.fit_transform(X_emb_raw[train_idx])
        emb_train = emb_pca.fit_transform(emb_train)

        emb_test = emb_scaler.transform(X_emb_raw[test_idx])
        emb_test = emb_pca.transform(emb_test)

        parts_train.append(emb_train)
        parts_test.append(emb_test)

    X_train = np.hstack(parts_train)
    X_test = np.hstack(parts_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def get_models():
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=24,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    et = ExtraTreesRegressor(
        n_estimators=600,
        max_depth=28,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    hgb = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.035,
        max_leaf_nodes=45,
        l2_regularization=0.01,
        random_state=42,
    )

    ridge = Ridge(alpha=5.0)

    stack = StackingRegressor(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("hgb", hgb),
            ("ridge", ridge),
        ],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1,
        passthrough=False,
    )

    return {
        "ridge": ridge,
        "random_forest": rf,
        "extra_trees": et,
        "hist_gradient_boosting": hgb,
        "stacking": stack,
    }


df = pd.read_csv(YOLO_CSV)
df = df[df["yolo_status"] == "success"].copy().reset_index(drop=True)

for c in vehicle_raw + ["Temperature", "Humidity"]:
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

feature_sets = {
    "tabular_only": base_features,
    "traffic_engineered_only": vehicle_raw + engineered_vehicle,
    "tabular_plus_traffic_engineered": base_features + vehicle_raw + engineered_vehicle,
    "tabular_plus_traffic_engineered_plus_embeddings": base_features + vehicle_raw + engineered_vehicle,
}

X_emb_all = np.load(EMB_FILE)
X_emb_raw = X_emb_all[df["row_id"].values]

y_raw = df[TARGET].values.astype(float)
y_log = np.log1p(y_raw)

splits = {
    "random_stratified": make_random_split(df),
    "chronological": make_chrono_split(df),
}

records = []
prediction_records = []

for split_name, (train_idx, val_idx, test_idx) in splits.items():
    print("\n" + "=" * 100)
    print("Split:", split_name)
    print("Rows:", {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)})

    mean_pred = np.full(len(test_idx), np.mean(y_raw[train_idx]), dtype=float)

    row = {
        "split": split_name,
        "feature_set": "none",
        "model": "mean_baseline",
        "target_transform": "none",
        "use_embedding": False,
        "n_features": 0,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "elapsed_sec": 0,
    }
    row.update(metrics(y_raw[test_idx], mean_pred))
    records.append(row)

    for fs_name, cols in feature_sets.items():
        use_embedding = "embeddings" in fs_name

        X_train, X_test = build_feature_matrix(
            df=df,
            X_emb_raw=X_emb_raw,
            train_idx=train_idx,
            test_idx=test_idx,
            feature_cols=cols,
            use_embedding=use_embedding,
        )

        for target_transform in ["none", "log1p"]:
            y_train = y_log[train_idx] if target_transform == "log1p" else y_raw[train_idx]

            for model_name, model in get_models().items():
                print("\n" + "-" * 100)
                print("START:", split_name, fs_name, model_name, "target:", target_transform)
                print("Feature matrix:", X_train.shape, X_test.shape)

                start = time.time()

                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                if target_transform == "log1p":
                    pred = np.expm1(pred)

                pred = np.clip(pred, 0, None)

                elapsed = time.time() - start

                row = {
                    "split": split_name,
                    "feature_set": fs_name,
                    "model": model_name,
                    "target_transform": target_transform,
                    "use_embedding": use_embedding,
                    "n_features": X_train.shape[1],
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "elapsed_sec": round(elapsed, 2),
                }
                row.update(metrics(y_raw[test_idx], pred))
                records.append(row)

                pred_df = pd.DataFrame({
                    "split": split_name,
                    "feature_set": fs_name,
                    "model": model_name,
                    "target_transform": target_transform,
                    "row_id": df.iloc[test_idx]["row_id"].values,
                    "created_at": df.iloc[test_idx]["created_at"].values,
                    "y_true_pm25": y_raw[test_idx],
                    "y_pred_pm25": pred,
                })
                prediction_records.append(pred_df)

                res = pd.DataFrame(records).sort_values(["split", "RMSE"])
                res.to_csv(OUT, index=False)

                preds_all = pd.concat(prediction_records, ignore_index=True)
                preds_all.to_csv(PRED_OUT, index=False)

                print("DONE:", round(elapsed, 2), "sec")
                print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman"]})
                print("Checkpoint saved:", OUT)

res = pd.DataFrame(records).sort_values(["split", "RMSE"])
res.to_csv(OUT, index=False)

pd.concat(prediction_records, ignore_index=True).to_csv(PRED_OUT, index=False)

print("\nFinal results:")
print(res.to_string(index=False))
print("\nSaved:")
print(OUT)
print(PRED_OUT)