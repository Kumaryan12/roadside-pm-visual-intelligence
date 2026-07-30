from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ADV_YOLO_CSV = Path(
    "experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_advanced_features_front.csv"
)

OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "advanced_yolo_feature_model_results.csv"
PRED_OUT = OUTDIR / "advanced_yolo_feature_model_predictions.csv"

TARGET = "PM2.5"
EPS = 1e-6


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def get_metrics(y_true, y_pred):
    sp = np.nan if np.std(y_pred) == 0 else spearmanr(y_true, y_pred).correlation
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Spearman": sp,
    }


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def add_time_features(df):
    df = df.copy()

    df["created_at_parsed"] = pd.to_datetime(df["created_at"])
    df["hour_num"] = df["created_at_parsed"].dt.hour
    df["month_num"] = df["created_at_parsed"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_num"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_num"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    return df


def add_advanced_yolo_interactions(df):
    df = df.copy()

    # Safe denominators
    total_count = df["total_vehicle_count"] + EPS
    total_area = df["total_vehicle_area_ratio"] + EPS

    # Count-density style features
    df["near_vehicle_ratio"] = df["near_vehicle_count"] / total_count
    df["near_heavy_ratio"] = df["near_heavy_vehicle_count"] / (df["heavy_vehicle_count"] + EPS)
    df["bottom_vehicle_ratio"] = df["bottom_half_vehicle_count"] / total_count
    df["center_lane_vehicle_ratio"] = df["center_lane_vehicle_count"] / total_count

    df["left_right_vehicle_imbalance"] = (
        df["left_half_vehicle_count"] - df["right_half_vehicle_count"]
    ) / total_count

    # Area dominance features
    df["near_vehicle_area_share"] = df["near_vehicle_area_ratio"] / total_area
    df["bottom_vehicle_area_share"] = df["bottom_half_vehicle_area_ratio"] / total_area
    df["center_lane_vehicle_area_share"] = df["center_lane_vehicle_area_ratio"] / total_area

    # Emission-relevance indices
    df["near_heavy_emission_proxy"] = (
        2.0 * df["near_heavy_vehicle_area_ratio"]
        + 1.5 * df["truck_near_area_ratio"]
        + 1.2 * df["bus_near_area_ratio"]
    )

    df["near_auto_tw_proxy"] = (
        1.5 * df["auto_near_area_ratio"]
        + 0.8 * df["near_two_wheeler_area_ratio"]
    )

    df["area_weighted_congestion_proxy"] = (
        df["total_vehicle_count"] * df["total_vehicle_area_ratio"]
    )

    df["near_area_weighted_congestion_proxy"] = (
        df["near_vehicle_count"] * df["near_vehicle_area_ratio"]
    )

    df["heavy_area_weighted_congestion_proxy"] = (
        df["heavy_vehicle_count"] * df["heavy_vehicle_area_ratio"]
    )

    # Meteorology × advanced traffic interactions
    df["humidity_x_vehicle_area"] = df["Humidity"] * df["total_vehicle_area_ratio"]
    df["humidity_x_near_vehicle_area"] = df["Humidity"] * df["near_vehicle_area_ratio"]
    df["humidity_x_heavy_area"] = df["Humidity"] * df["heavy_vehicle_area_ratio"]

    df["temperature_x_vehicle_area"] = df["Temperature"] * df["total_vehicle_area_ratio"]
    df["temperature_x_near_vehicle_area"] = df["Temperature"] * df["near_vehicle_area_ratio"]
    df["temperature_x_heavy_area"] = df["Temperature"] * df["heavy_vehicle_area_ratio"]

    # High traffic flags
    df["is_high_vehicle_count"] = (
        df["total_vehicle_count"] >= df["total_vehicle_count"].quantile(0.75)
    ).astype(int)

    df["is_high_vehicle_area"] = (
        df["total_vehicle_area_ratio"] >= df["total_vehicle_area_ratio"].quantile(0.75)
    ).astype(int)

    df["is_high_near_vehicle_area"] = (
        df["near_vehicle_area_ratio"] >= df["near_vehicle_area_ratio"].quantile(0.75)
    ).astype(int)

    df["is_heavy_vehicle_present"] = (df["heavy_vehicle_count"] > 0).astype(int)
    df["is_near_heavy_present"] = (df["near_heavy_vehicle_count"] > 0).astype(int)

    df["high_traffic_and_heavy"] = (
        df["is_high_vehicle_count"] * df["is_heavy_vehicle_present"]
    )

    df["high_near_area_and_heavy"] = (
        df["is_high_near_vehicle_area"] * df["is_heavy_vehicle_present"]
    )

    return df


def prepare_matrix(train_df, test_df, numeric_cols, categorical_cols):
    train_num = train_df[numeric_cols].values.astype(float)
    test_num = test_df[numeric_cols].values.astype(float)

    cat_all = pd.concat(
        [train_df[categorical_cols], test_df[categorical_cols]],
        axis=0,
    )

    cat_all = pd.get_dummies(cat_all, columns=categorical_cols, drop_first=False)

    train_cat = cat_all.iloc[: len(train_df)].values.astype(float)
    test_cat = cat_all.iloc[len(train_df) :].values.astype(float)

    X_train = np.hstack([train_num, train_cat])
    X_test = np.hstack([test_num, test_cat])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    feature_names = list(numeric_cols) + list(cat_all.columns)

    return X_train, X_test, feature_names


def main():
    print("Loading:", ADV_YOLO_CSV)

    df = pd.read_csv(ADV_YOLO_CSV)

    df = df[df["yolo_adv_status"] == "success"].copy()
    df = df.dropna(subset=[TARGET, "Temperature", "Humidity", "aqi_cat", "created_at"])

    df = add_time_features(df)
    df = add_advanced_yolo_interactions(df)

    # -----------------------------------------------------------------
    # Define feature groups
    # -----------------------------------------------------------------
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

    old_yolo_like_features = [
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
        "traffic_mix_count_index",
    ]

    advanced_yolo_raw_features = [
        # areas
        "total_vehicle_area_ratio",
        "car_area_ratio",
        "bus_area_ratio",
        "truck_area_ratio",
        "auto_area_ratio",
        "motorcycle_area_ratio",
        "bicycle_area_ratio",
        "heavy_vehicle_area_ratio",
        "two_wheeler_area_ratio",

        # confidence
        "total_vehicle_conf_sum",
        "total_vehicle_conf_area_ratio",
        "heavy_vehicle_conf_area_ratio",
        "two_wheeler_conf_area_ratio",
        "mean_vehicle_confidence",
        "max_vehicle_confidence",

        # near-field
        "near_vehicle_count",
        "near_vehicle_area_ratio",
        "near_heavy_vehicle_count",
        "near_heavy_vehicle_area_ratio",
        "near_two_wheeler_count",
        "near_two_wheeler_area_ratio",
        "car_near_count",
        "car_near_area_ratio",
        "bus_near_count",
        "bus_near_area_ratio",
        "truck_near_count",
        "truck_near_area_ratio",
        "auto_near_count",
        "auto_near_area_ratio",
        "motorcycle_near_count",
        "motorcycle_near_area_ratio",

        # spatial
        "bottom_half_vehicle_count",
        "bottom_half_vehicle_area_ratio",
        "center_lane_vehicle_count",
        "center_lane_vehicle_area_ratio",
        "left_half_vehicle_count",
        "right_half_vehicle_count",

        # composition
        "vehicle_area_share_heavy",
        "vehicle_area_share_two_wheeler",
        "vehicle_area_share_auto",
        "vehicle_count_share_heavy",
        "vehicle_count_share_two_wheeler",
        "vehicle_count_share_auto",

        # traffic mix
        "traffic_mix_area_index",
        "near_traffic_mix_count_index",
        "near_traffic_mix_area_index",

        # box stats
        "mean_vehicle_box_area_ratio",
        "max_vehicle_box_area_ratio",
    ]

    advanced_yolo_engineered_features = [
        "near_vehicle_ratio",
        "near_heavy_ratio",
        "bottom_vehicle_ratio",
        "center_lane_vehicle_ratio",
        "left_right_vehicle_imbalance",
        "near_vehicle_area_share",
        "bottom_vehicle_area_share",
        "center_lane_vehicle_area_share",
        "near_heavy_emission_proxy",
        "near_auto_tw_proxy",
        "area_weighted_congestion_proxy",
        "near_area_weighted_congestion_proxy",
        "heavy_area_weighted_congestion_proxy",
        "humidity_x_vehicle_area",
        "humidity_x_near_vehicle_area",
        "humidity_x_heavy_area",
        "temperature_x_vehicle_area",
        "temperature_x_near_vehicle_area",
        "temperature_x_heavy_area",
        "is_high_vehicle_count",
        "is_high_vehicle_area",
        "is_high_near_vehicle_area",
        "is_heavy_vehicle_present",
        "is_near_heavy_present",
        "high_traffic_and_heavy",
        "high_near_area_and_heavy",
    ]

    categorical_features = ["Season", "Day_or_Night"]

    feature_sets = {
        "tabular_only": tabular_features,
        "old_yolo_counts_only": old_yolo_like_features,
        "advanced_yolo_only": old_yolo_like_features
        + advanced_yolo_raw_features
        + advanced_yolo_engineered_features,
        "tabular_plus_old_yolo": tabular_features + old_yolo_like_features,
        "tabular_plus_advanced_yolo": tabular_features
        + old_yolo_like_features
        + advanced_yolo_raw_features
        + advanced_yolo_engineered_features,
    }

    # Clean numeric handling
    all_numeric = sorted(set(sum(feature_sets.values(), [])))

    for c in all_numeric:
        if c not in df.columns:
            print("Missing feature column, creating zero:", c)
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[all_numeric] = df[all_numeric].replace([np.inf, -np.inf], np.nan)
    df[all_numeric] = df[all_numeric].fillna(df[all_numeric].median(numeric_only=True))
    df[all_numeric] = df[all_numeric].fillna(0)

    print("Rows after cleaning:", len(df))
    print("AQI distribution:")
    print(df["aqi_cat"].value_counts())

    # -----------------------------------------------------------------
    # Split
    # -----------------------------------------------------------------
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

    print("\nSplit:")
    print({"train": len(train), "val": len(val), "test": len(test)})

    y_train = train[TARGET].values.astype(float)
    y_test = test[TARGET].values.astype(float)

    models = {
        "ridge": Ridge(alpha=10.0),
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

    records = []
    pred_records = []

    for fs_name, cols in feature_sets.items():
        X_train, X_test, feature_names = prepare_matrix(
            train,
            test,
            numeric_cols=cols,
            categorical_cols=categorical_features,
        )

        for model_name, model in models.items():
            print("\n" + "-" * 100)
            print("START:", fs_name, model_name)
            print("Feature matrix:", X_train.shape, X_test.shape)

            t0 = time.time()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred = np.clip(pred, 0, None)
            elapsed = time.time() - t0

            row = {
                "feature_set": fs_name,
                "model": model_name,
                "n_features": X_train.shape[1],
                "n_train": len(train),
                "n_test": len(test),
                "elapsed_sec": elapsed,
            }

            row.update(get_metrics(y_test, pred))
            records.append(row)

            pred_records.append(
                pd.DataFrame(
                    {
                        "row_id": test["row_id"].values,
                        "created_at": test["created_at"].values,
                        "aqi_cat": test["aqi_cat"].values,
                        "feature_set": fs_name,
                        "model": model_name,
                        "y_true_pm25": y_test,
                        "y_pred_pm25": pred,
                        "residual_pred_minus_true": pred - y_test,
                    }
                )
            )

            res = pd.DataFrame(records).sort_values("RMSE")
            res.to_csv(OUT, index=False)

            preds = pd.concat(pred_records, ignore_index=True)
            preds.to_csv(PRED_OUT, index=False)

            print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman"]})
            print("Checkpoint saved:", OUT)

    res = pd.DataFrame(records).sort_values("RMSE")
    res.to_csv(OUT, index=False)

    preds = pd.concat(pred_records, ignore_index=True)
    preds.to_csv(PRED_OUT, index=False)

    print("\nFinal results:")
    print(res.to_string(index=False))

    print("\nSaved:")
    print(OUT)
    print(PRED_OUT)


if __name__ == "__main__":
    main() 