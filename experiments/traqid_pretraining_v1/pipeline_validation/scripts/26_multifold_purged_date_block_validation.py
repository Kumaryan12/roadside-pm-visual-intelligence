from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================

ADV_YOLO_CSV = Path(
    "experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_advanced_features_front.csv"
)

ROAD_CSV = Path(
    "experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_road_features_front.csv"
)

RESNET_EMB = Path(
    "experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/traqid_paper_resnet50_front_rear_concat_gap_features.npy"
)

RESNET_INDEX = Path(
    "experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/traqid_paper_resnet50_front_gap_index.csv"
)

OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = OUTDIR / "multifold_purged_date_block_results.csv"
PRED_OUT = OUTDIR / "multifold_purged_date_block_predictions.csv"
SUMMARY_OUT = OUTDIR / "multifold_purged_date_block_summary.csv"

TARGET = "PM2.5"
RANDOM_STATE = 42
EPS = 1e-6


# ============================================================
# Metrics
# ============================================================

def get_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    sp = np.nan
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        sp = spearmanr(y_true, y_pred).correlation

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Spearman": sp,
    }


def clean_numeric(df, cols):
    df = df.copy()

    for c in cols:
        if c not in df.columns:
            print(f"Missing column created as zero: {c}")
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df[cols] = df[cols].fillna(df[cols].median(numeric_only=True))
    df[cols] = df[cols].fillna(0)

    return df


# ============================================================
# Feature engineering
# ============================================================

def add_advanced_yolo_interactions(df):
    df = df.copy()

    total_count = df["total_vehicle_count"] + EPS
    total_area = df["total_vehicle_area_ratio"] + EPS

    df["near_vehicle_ratio"] = df["near_vehicle_count"] / total_count
    df["near_heavy_ratio"] = df["near_heavy_vehicle_count"] / (df["heavy_vehicle_count"] + EPS)

    df["bottom_vehicle_ratio"] = df["bottom_half_vehicle_count"] / total_count
    df["center_lane_vehicle_ratio"] = df["center_lane_vehicle_count"] / total_count

    df["left_right_vehicle_imbalance"] = (
        df["left_half_vehicle_count"] - df["right_half_vehicle_count"]
    ) / total_count

    df["near_vehicle_area_share"] = df["near_vehicle_area_ratio"] / total_area
    df["bottom_vehicle_area_share"] = df["bottom_half_vehicle_area_ratio"] / total_area
    df["center_lane_vehicle_area_share"] = df["center_lane_vehicle_area_ratio"] / total_area

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

    return df


def add_road_dust_features(df):
    df = df.copy()

    for c in [
        "road_area_ratio",
        "road_brown_pixel_ratio",
        "road_gray_dry_pixel_ratio",
        "road_mean_brightness",
        "road_mean_saturation",
        "road_contrast_std",
        "road_shadow_ratio",
        "road_glare_ratio",
        "road_edge_density",
        "road_laplacian_std",
        "road_haze_flatness_proxy",
    ]:
        if c not in df.columns:
            df[c] = 0.0

    df["road_brown_exposure"] = (
        df["road_area_ratio"] * df["road_brown_pixel_ratio"]
    )

    df["road_gray_dry_exposure"] = (
        df["road_area_ratio"] * df["road_gray_dry_pixel_ratio"]
    )

    df["road_texture_score"] = (
        0.5 * df["road_edge_density"]
        + 0.5 * df["road_laplacian_std"]
    )

    df["visible_road_quality"] = (
        df["road_area_ratio"]
        * (1.0 - df["road_shadow_ratio"].clip(0, 1))
        * (1.0 - df["road_glare_ratio"].clip(0, 1))
    )

    df["brightness_corrected_brown"] = (
        df["road_brown_pixel_ratio"] * (1.0 - df["road_mean_brightness"].clip(0, 1))
    )

    df["saturation_corrected_brown"] = (
        df["road_brown_pixel_ratio"] * df["road_mean_saturation"].clip(0, 1)
    )

    df["road_dust_score"] = (
        0.35 * df["road_brown_exposure"]
        + 0.25 * df["road_gray_dry_exposure"]
        + 0.20 * df["road_texture_score"]
        + 0.20 * df["visible_road_quality"]
    )

    return df


def add_yolo_road_interactions(df):
    df = df.copy()

    road_area = df["road_area_ratio"] + EPS

    df["vehicle_area_per_road_area"] = df["total_vehicle_area_ratio"] / road_area
    df["heavy_vehicle_area_per_road_area"] = df["heavy_vehicle_area_ratio"] / road_area
    df["near_vehicle_area_per_road_area"] = df["near_vehicle_area_ratio"] / road_area
    df["traffic_mix_area_per_road_area"] = df["traffic_mix_area_index"] / road_area

    df["truck_area_x_gray_dry_road"] = df["truck_area_ratio"] * df["road_gray_dry_pixel_ratio"]
    df["bus_area_x_gray_dry_road"] = df["bus_area_ratio"] * df["road_gray_dry_pixel_ratio"]
    df["heavy_area_x_gray_dry_road"] = df["heavy_vehicle_area_ratio"] * df["road_gray_dry_pixel_ratio"]

    df["near_vehicle_x_road_dust"] = df["near_vehicle_area_ratio"] * df["road_dust_score"]
    df["heavy_vehicle_x_road_dust"] = df["heavy_vehicle_area_ratio"] * df["road_dust_score"]

    df["traffic_mix_x_road_texture"] = df["traffic_mix_area_index"] * df["road_texture_score"]
    df["traffic_mix_x_visible_road"] = df["traffic_mix_area_index"] * df["visible_road_quality"]

    return df


# ============================================================
# Dataset building
# ============================================================

def build_dataset():
    print("Loading advanced YOLO:", ADV_YOLO_CSV)
    yolo = pd.read_csv(ADV_YOLO_CSV)
    yolo = yolo[yolo["yolo_adv_status"] == "success"].copy()

    print("Loading road features:", ROAD_CSV)
    road = pd.read_csv(ROAD_CSV)

    if "processed_frame_key" in road.columns and "row_id" not in road.columns:
        road["image_id"] = pd.to_numeric(road["processed_frame_key"], errors="coerce")
        road["row_id"] = road["image_id"] - 1

    road = road[road["road_condition_status"] == "success"].copy()

    keep_road_cols = [
        "row_id",
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

    road = road[[c for c in keep_road_cols if c in road.columns]].copy()

    print("Merging YOLO + road...")
    df = yolo.merge(road, on="row_id", how="inner")
    df = df.dropna(subset=[TARGET, "created_at", "aqi_cat"]).copy()

    df["created_at_parsed"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at_parsed"].dt.date.astype(str)

    df = add_advanced_yolo_interactions(df)
    df = add_road_dust_features(df)
    df = add_yolo_road_interactions(df)

    print("Merged rows:", len(df))
    print("Unique dates:", df["date"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())

    return df


# ============================================================
# Feature definitions
# ============================================================

def get_feature_groups():
    yolo_count_features = [
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

    yolo_advanced_features = [
        "total_vehicle_area_ratio",
        "car_area_ratio",
        "bus_area_ratio",
        "truck_area_ratio",
        "auto_area_ratio",
        "motorcycle_area_ratio",
        "bicycle_area_ratio",
        "heavy_vehicle_area_ratio",
        "two_wheeler_area_ratio",
        "total_vehicle_conf_sum",
        "total_vehicle_conf_area_ratio",
        "heavy_vehicle_conf_area_ratio",
        "two_wheeler_conf_area_ratio",
        "mean_vehicle_confidence",
        "max_vehicle_confidence",
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
        "bottom_half_vehicle_count",
        "bottom_half_vehicle_area_ratio",
        "center_lane_vehicle_count",
        "center_lane_vehicle_area_ratio",
        "left_half_vehicle_count",
        "right_half_vehicle_count",
        "vehicle_area_share_heavy",
        "vehicle_area_share_two_wheeler",
        "vehicle_area_share_auto",
        "vehicle_count_share_heavy",
        "vehicle_count_share_two_wheeler",
        "vehicle_count_share_auto",
        "traffic_mix_area_index",
        "near_traffic_mix_count_index",
        "near_traffic_mix_area_index",
        "mean_vehicle_box_area_ratio",
        "max_vehicle_box_area_ratio",
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
    ]

    road_features = [
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
        "road_brown_exposure",
        "road_gray_dry_exposure",
        "road_texture_score",
        "visible_road_quality",
        "brightness_corrected_brown",
        "saturation_corrected_brown",
        "road_dust_score",
    ]

    yolo_road_interactions = [
        "vehicle_area_per_road_area",
        "heavy_vehicle_area_per_road_area",
        "near_vehicle_area_per_road_area",
        "traffic_mix_area_per_road_area",
        "truck_area_x_gray_dry_road",
        "bus_area_x_gray_dry_road",
        "heavy_area_x_gray_dry_road",
        "near_vehicle_x_road_dust",
        "heavy_vehicle_x_road_dust",
        "traffic_mix_x_road_texture",
        "traffic_mix_x_visible_road",
    ]

    image_all = (
        yolo_count_features
        + yolo_advanced_features
        + road_features
        + yolo_road_interactions
    )

    weak_env = ["Temperature", "Humidity"]
    daynight_cat = ["Day_or_Night"]

    return image_all, weak_env, daynight_cat


# ============================================================
# ResNet + matrix
# ============================================================

def load_resnet_embeddings(df):
    print("\nLoading ResNet embeddings...")
    emb = np.load(RESNET_EMB)
    idx = pd.read_csv(RESNET_INDEX)

    idx = idx.copy()
    idx["embedding_pos"] = np.arange(len(idx))

    merged = df[["row_id"]].merge(
        idx[["row_id", "embedding_pos"]],
        on="row_id",
        how="left",
    )

    pos = merged["embedding_pos"].fillna(0).astype(int).values
    X_emb = emb[pos].astype(np.float32)

    missing_mask = merged["embedding_pos"].isna().values
    if missing_mask.any():
        print("Missing ResNet embeddings:", missing_mask.sum())
        X_emb[missing_mask] = 0

    print("Aligned embedding matrix:", X_emb.shape)
    return X_emb


def prepare_matrix(
    train_df,
    test_df,
    numeric_cols,
    categorical_cols=None,
    train_emb=None,
    test_emb=None,
    pca_components=None,
):
    categorical_cols = categorical_cols or []

    X_train_parts = []
    X_test_parts = []

    if numeric_cols:
        X_train_parts.append(train_df[numeric_cols].values.astype(float))
        X_test_parts.append(test_df[numeric_cols].values.astype(float))

    if categorical_cols:
        cat_all = pd.concat(
            [train_df[categorical_cols], test_df[categorical_cols]],
            axis=0,
        )
        cat_all = pd.get_dummies(cat_all, columns=categorical_cols, drop_first=False)

        X_train_parts.append(cat_all.iloc[: len(train_df)].values.astype(float))
        X_test_parts.append(cat_all.iloc[len(train_df):].values.astype(float))

    if train_emb is not None and test_emb is not None:
        if pca_components is not None:
            pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
            train_emb = pca.fit_transform(train_emb)
            test_emb = pca.transform(test_emb)

        X_train_parts.append(train_emb)
        X_test_parts.append(test_emb)

    X_train = np.hstack(X_train_parts)
    X_test = np.hstack(X_test_parts)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


# ============================================================
# Multi-fold purged date-block split
# ============================================================

def make_date_blocks(df, n_folds=5, purge_days=2):
    unique_dates = pd.to_datetime(pd.Series(df["date"].unique())).sort_values().reset_index(drop=True)
    n_dates = len(unique_dates)

    blocks = np.array_split(unique_dates.values, n_folds)

    folds = []

    for fold_id, block in enumerate(blocks, start=1):
        test_dates = pd.to_datetime(pd.Series(block)).sort_values()
        test_start = test_dates.min()
        test_end = test_dates.max()

        purge_start = test_start - pd.Timedelta(days=purge_days)
        purge_end = test_end + pd.Timedelta(days=purge_days)

        date_series = pd.to_datetime(df["date"])

        test_mask = (date_series >= test_start) & (date_series <= test_end)
        purge_mask = (
            (date_series >= purge_start)
            & (date_series <= purge_end)
            & (~test_mask)
        )
        train_mask = ~(test_mask | purge_mask)

        train = df[train_mask].copy()
        purge = df[purge_mask].copy()
        test = df[test_mask].copy()

        folds.append(
            {
                "fold": fold_id,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train": train,
                "purge": purge,
                "test": test,
            }
        )

    return folds


# ============================================================
# Main
# ============================================================

def main():
    df = build_dataset()
    image_all, weak_env, daynight_cat = get_feature_groups()

    all_numeric = sorted(set(image_all + weak_env))
    df = clean_numeric(df, all_numeric)

    all_emb = load_resnet_embeddings(df)

    df_reset = df.reset_index(drop=True)
    row_id_to_pos = {rid: i for i, rid in enumerate(df_reset["row_id"].values)}

    feature_sets = {
        "mean_baseline": {
            "numeric": [],
            "categorical": [],
            "use_resnet": False,
            "pca": None,
            "description": "Mean baseline",
        },
        "strict_image_yolo_road_resnet_pca64": {
            "numeric": image_all,
            "categorical": [],
            "use_resnet": True,
            "pca": 64,
            "description": "Strict image-only: YOLO + road + ResNet50 PCA64",
        },
        "image_temp_humidity_daynight": {
            "numeric": image_all + weak_env,
            "categorical": daynight_cat,
            "use_resnet": True,
            "pca": 32,
            "description": "Image + Temp/Humidity + Day/Night",
        },
        "weak_env_only": {
            "numeric": weak_env,
            "categorical": daynight_cat,
            "use_resnet": False,
            "pca": None,
            "description": "Temperature/Humidity + Day/Night only",
        },
    }

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=24,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=700,
            learning_rate=0.03,
            max_leaf_nodes=63,
            l2_regularization=0.001,
            random_state=RANDOM_STATE,
        ),
    }

    folds = make_date_blocks(df, n_folds=5, purge_days=2)

    records = []
    pred_records = []

    for fold in folds:
        fold_id = fold["fold"]
        train = fold["train"]
        purge = fold["purge"]
        test = fold["test"]

        print("\n" + "=" * 100)
        print(f"FOLD {fold_id}: {fold['test_start']} to {fold['test_end']}")
        print("Rows:", {"train": len(train), "purge": len(purge), "test": len(test)})
        print("Test AQI distribution:")
        print(test["aqi_cat"].value_counts().to_string())

        if len(train) == 0 or len(test) == 0:
            print("Skipping empty fold.")
            continue

        y_train = train[TARGET].values.astype(float)
        y_test = test[TARGET].values.astype(float)

        train_pos = [row_id_to_pos[rid] for rid in train["row_id"].values]
        test_pos = [row_id_to_pos[rid] for rid in test["row_id"].values]

        emb_train = all_emb[train_pos]
        emb_test = all_emb[test_pos]

        for fs_name, cfg in feature_sets.items():
            if fs_name == "mean_baseline":
                pred = np.full(len(test), np.mean(y_train), dtype=float)

                row = {
                    "fold": fold_id,
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "feature_set": fs_name,
                    "description": cfg["description"],
                    "model": "mean_baseline",
                    "n_features": 0,
                    "n_train": len(train),
                    "n_purge": len(purge),
                    "n_test": len(test),
                    "test_pm25_mean": float(np.mean(y_test)),
                    "test_pm25_std": float(np.std(y_test)),
                    "elapsed_sec": 0.0,
                }
                row.update(get_metrics(y_test, pred))
                records.append(row)
                continue

            train_emb = emb_train if cfg["use_resnet"] else None
            test_emb = emb_test if cfg["use_resnet"] else None

            X_train, X_test = prepare_matrix(
                train,
                test,
                numeric_cols=cfg["numeric"],
                categorical_cols=cfg["categorical"],
                train_emb=train_emb,
                test_emb=test_emb,
                pca_components=cfg["pca"],
            )

            for model_name, model in models.items():
                print("-" * 80)
                print("Fold:", fold_id, "|", fs_name, "|", model_name)
                print("Feature matrix:", X_train.shape, X_test.shape)

                t0 = time.time()
                model.fit(X_train, y_train)
                pred = np.clip(model.predict(X_test), 0, None)
                elapsed = time.time() - t0

                row = {
                    "fold": fold_id,
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "feature_set": fs_name,
                    "description": cfg["description"],
                    "model": model_name,
                    "n_features": X_train.shape[1],
                    "n_train": len(train),
                    "n_purge": len(purge),
                    "n_test": len(test),
                    "test_pm25_mean": float(np.mean(y_test)),
                    "test_pm25_std": float(np.std(y_test)),
                    "elapsed_sec": elapsed,
                }
                row.update(get_metrics(y_test, pred))
                records.append(row)

                pred_records.append(
                    pd.DataFrame(
                        {
                            "fold": fold_id,
                            "date": test["date"].values,
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

                pd.DataFrame(records).to_csv(RESULTS_OUT, index=False)
                if pred_records:
                    pd.concat(pred_records, ignore_index=True).to_csv(PRED_OUT, index=False)

                print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman"]})

    results = pd.DataFrame(records)
    results.to_csv(RESULTS_OUT, index=False)

    if pred_records:
        pd.concat(pred_records, ignore_index=True).to_csv(PRED_OUT, index=False)

    # Summary by feature set/model
    summary = (
        results
        .groupby(["feature_set", "model"])
        .agg(
            folds=("fold", "nunique"),
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_R2=("R2", "mean"),
            std_R2=("R2", "std"),
            mean_Spearman=("Spearman", "mean"),
            std_Spearman=("Spearman", "std"),
            mean_test_pm25=("test_pm25_mean", "mean"),
        )
        .reset_index()
        .sort_values("mean_RMSE")
    )

    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n" + "=" * 100)
    print("MULTI-FOLD PURGED DATE-BLOCK SUMMARY:")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(RESULTS_OUT)
    print(PRED_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()