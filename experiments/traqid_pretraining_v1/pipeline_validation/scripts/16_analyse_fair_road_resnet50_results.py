from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from scipy.stats import spearmanr

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
YOLO_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_vehicle_features_front.csv")
ROAD_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_road_features_front.csv")
EMB_FILE = Path("experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/traqid_paper_resnet50_front_rear_concat_gap_features.npy")

REPORTS = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
FIGDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/figures/fair_road_resnet50_analysis")
FIGDIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = REPORTS / "fair_model_with_road_dust_score_resnet50_results.csv"

SUMMARY_OUT = REPORTS / "fair_road_resnet50_ablation_summary_clean.csv"
PRED_OUT = REPORTS / "fair_road_resnet50_best_model_predictions.csv"
ERR_AQI_OUT = REPORTS / "fair_road_resnet50_error_by_aqi_cat.csv"
ERR_ROAD_OUT = REPORTS / "fair_road_resnet50_error_by_road_visibility.csv"
FI_OUT = REPORTS / "fair_road_resnet50_feature_importance.csv"

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


def build_matrix(df, X_emb_raw, train_idx, test_idx, feature_cols, use_embedding):
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    X_train_num = train[feature_cols].values.astype(float)
    X_test_num = test[feature_cols].values.astype(float)

    numeric_names = list(feature_cols)

    combined_cat = pd.concat([train[categorical_features], test[categorical_features]], axis=0)
    combined_cat = pd.get_dummies(combined_cat, columns=categorical_features, drop_first=False)

    cat_names = combined_cat.columns.tolist()

    X_train_cat = combined_cat.iloc[:len(train)].values.astype(float)
    X_test_cat = combined_cat.iloc[len(train):].values.astype(float)

    parts_train = [X_train_num, X_train_cat]
    parts_test = [X_test_num, X_test_cat]
    feature_names = numeric_names + cat_names

    if use_embedding:
        emb_scaler = StandardScaler()
        pca = PCA(n_components=PCA_DIM, random_state=42)

        emb_train = emb_scaler.fit_transform(X_emb_raw[train_idx])
        emb_train = pca.fit_transform(emb_train)

        emb_test = emb_scaler.transform(X_emb_raw[test_idx])
        emb_test = pca.transform(emb_test)

        emb_names = [f"resnet50_pca_{i}" for i in range(emb_train.shape[1])]

        parts_train.append(emb_train)
        parts_test.append(emb_test)
        feature_names += emb_names

    X_train = np.hstack(parts_train)
    X_test = np.hstack(parts_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, feature_names


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

    df = add_engineered_features(df)

    return df


def plot_bar(df, metric, outpath, title):
    plt.figure(figsize=(11, 5))
    x = np.arange(len(df))
    plt.bar(x, df[metric].values)
    plt.xticks(x, df["label"].values, rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def main():
    print("Loading results...")
    results = pd.read_csv(RESULTS_CSV)

    # Clean ablation summary: one best row per major feature set.
    selected_sets = [
        "mean_baseline",
        "traffic_only",
        "road_dust_only",
        "tabular_only",
        "tabular_plus_traffic",
        "tabular_plus_road_dust",
        "tabular_plus_traffic_plus_road_dust",
        "full_fair_with_embeddings",
    ]

    summary_rows = []
    for fs in selected_sets:
        sub = results[results["feature_set"] == fs].copy()
        if len(sub) == 0:
            continue
        best = sub.sort_values("RMSE").iloc[0].copy()
        summary_rows.append(best)

    summary = pd.DataFrame(summary_rows)
    label_map = {
        "mean_baseline": "Mean",
        "traffic_only": "Traffic only",
        "road_dust_only": "Road only",
        "tabular_only": "Tabular only",
        "tabular_plus_traffic": "Tabular + traffic",
        "tabular_plus_road_dust": "Tabular + road",
        "tabular_plus_traffic_plus_road_dust": "Tabular + traffic + road",
        "full_fair_with_embeddings": "Full + ResNet50",
    }
    summary["label"] = summary["feature_set"].map(label_map)
    summary = summary[[
        "label", "feature_set", "model", "n_features", "MAE", "RMSE", "R2", "Spearman"
    ]]
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\nClean ablation summary:")
    print(summary.to_string(index=False))
    print("\nSaved:", SUMMARY_OUT)

    plot_bar(
        summary,
        "R2",
        FIGDIR / "ablation_r2_bar.png",
        "Fair TRAQID Ablation: R² by Feature Group",
    )

    plot_bar(
        summary,
        "RMSE",
        FIGDIR / "ablation_rmse_bar.png",
        "Fair TRAQID Ablation: RMSE by Feature Group",
    )

    print("Saved ablation plots.")

    # -----------------------------------------------------------------
    # Rebuild dataset and retrain selected models for predictions/importance.
    # -----------------------------------------------------------------
    print("\nRebuilding dataset...")
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

    print("Split used:")
    print({"train": len(train_idx), "val": len(val), "test": len(test_idx)})

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

    # Models to analyse deeply
    analysis_models = {
        "tabular_only_extra_trees": {
            "features": base_features,
            "use_embedding": False,
            "model": ExtraTreesRegressor(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        },
        "traffic_only_random_forest": {
            "features": traffic_features,
            "use_embedding": False,
            "model": RandomForestRegressor(
                n_estimators=500,
                max_depth=24,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        },
        "road_only_hgb": {
            "features": road_dust_features,
            "use_embedding": False,
            "model": HistGradientBoostingRegressor(
                max_iter=700,
                learning_rate=0.03,
                max_leaf_nodes=63,
                l2_regularization=0.001,
                random_state=42,
            ),
        },
        "full_fair_resnet50_extra_trees": {
            "features": base_features + traffic_features + road_dust_features,
            "use_embedding": True,
            "model": ExtraTreesRegressor(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        },
    }

    all_preds = []
    all_importances = []

    for name, cfg in analysis_models.items():
        print("\nTraining analysis model:", name)

        X_train, X_test, feat_names = build_matrix(
            df=df,
            X_emb_raw=X_emb_raw,
            train_idx=train_idx,
            test_idx=test_idx,
            feature_cols=cfg["features"],
            use_embedding=cfg["use_embedding"],
        )

        model = cfg["model"]
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        pred = np.clip(pred, 0, None)

        met = metrics(y[test_idx], pred)
        print(name, met)

        pred_df = df.iloc[test_idx][[
            "row_id",
            "created_at",
            "aqi_cat",
            "Season",
            "Day_or_Night",
            "road_area_ratio",
            "road_dust_score",
            "total_vehicle_count",
            "heavy_vehicle_count",
            "traffic_mix_index",
            TARGET,
        ]].copy()

        pred_df = pred_df.rename(columns={TARGET: "y_true_pm25"})
        pred_df["model"] = name
        pred_df["y_pred_pm25"] = pred
        pred_df["residual_pred_minus_true"] = pred_df["y_pred_pm25"] - pred_df["y_true_pm25"]
        pred_df["abs_error"] = pred_df["residual_pred_minus_true"].abs()
        all_preds.append(pred_df)

        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "model": name,
                "feature": feat_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            all_importances.append(fi)

            # Top-20 feature importance plot
            top = fi.head(20).iloc[::-1]
            plt.figure(figsize=(8, 6))
            plt.barh(top["feature"], top["importance"])
            plt.xlabel("Feature importance")
            plt.title(f"Top feature importances: {name}")
            plt.tight_layout()
            plt.savefig(FIGDIR / f"feature_importance_{name}.png", dpi=220)
            plt.close()

    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_csv(PRED_OUT, index=False)

    if all_importances:
        importances = pd.concat(all_importances, ignore_index=True)
        importances.to_csv(FI_OUT, index=False)
        print("Saved feature importance:", FI_OUT)

    print("Saved predictions:", PRED_OUT)

    # -----------------------------------------------------------------
    # Plots for best overall and full model
    # -----------------------------------------------------------------
    for model_name in ["tabular_only_extra_trees", "full_fair_resnet50_extra_trees"]:
        sub = preds[preds["model"] == model_name].copy()

        plt.figure(figsize=(6, 6))
        plt.scatter(sub["y_true_pm25"], sub["y_pred_pm25"], s=8, alpha=0.35)
        lim_max = max(sub["y_true_pm25"].max(), sub["y_pred_pm25"].max())
        plt.plot([0, lim_max], [0, lim_max], linestyle="--")
        plt.xlabel("Actual PM2.5")
        plt.ylabel("Predicted PM2.5")
        plt.title(f"Predicted vs Actual: {model_name}")
        plt.tight_layout()
        plt.savefig(FIGDIR / f"scatter_pred_actual_{model_name}.png", dpi=220)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.hist(sub["residual_pred_minus_true"], bins=60)
        plt.xlabel("Residual: predicted - actual PM2.5")
        plt.ylabel("Count")
        plt.title(f"Residual distribution: {model_name}")
        plt.tight_layout()
        plt.savefig(FIGDIR / f"residual_hist_{model_name}.png", dpi=220)
        plt.close()

    # -----------------------------------------------------------------
    # Error by AQI category
    # -----------------------------------------------------------------
    err_aqi = (
        preds
        .groupby(["model", "aqi_cat"])
        .agg(
            n=("row_id", "count"),
            MAE=("abs_error", "mean"),
            Bias=("residual_pred_minus_true", "mean"),
            RMSE=("residual_pred_minus_true", lambda x: np.sqrt(np.mean(x ** 2))),
        )
        .reset_index()
    )
    err_aqi.to_csv(ERR_AQI_OUT, index=False)

    # Plot AQI error for best overall and full model
    for model_name in ["tabular_only_extra_trees", "full_fair_resnet50_extra_trees"]:
        sub = err_aqi[err_aqi["model"] == model_name].copy()
        sub = sub.sort_values("MAE")

        plt.figure(figsize=(8, 4))
        plt.bar(sub["aqi_cat"], sub["MAE"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("MAE")
        plt.title(f"MAE by AQI category: {model_name}")
        plt.tight_layout()
        plt.savefig(FIGDIR / f"mae_by_aqi_{model_name}.png", dpi=220)
        plt.close()

    # -----------------------------------------------------------------
    # Error by road visibility bins
    # -----------------------------------------------------------------
    preds["road_visibility_bin"] = pd.qcut(
        preds["road_area_ratio"],
        q=3,
        labels=["low road visibility", "medium road visibility", "high road visibility"],
        duplicates="drop",
    )

    err_road = (
        preds
        .groupby(["model", "road_visibility_bin"], observed=False)
        .agg(
            n=("row_id", "count"),
            MAE=("abs_error", "mean"),
            Bias=("residual_pred_minus_true", "mean"),
            RMSE=("residual_pred_minus_true", lambda x: np.sqrt(np.mean(x ** 2))),
            mean_road_area=("road_area_ratio", "mean"),
        )
        .reset_index()
    )
    err_road.to_csv(ERR_ROAD_OUT, index=False)

    for model_name in ["road_only_hgb", "full_fair_resnet50_extra_trees"]:
        sub = err_road[err_road["model"] == model_name].copy()

        plt.figure(figsize=(8, 4))
        plt.bar(sub["road_visibility_bin"].astype(str), sub["MAE"])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("MAE")
        plt.title(f"MAE by road visibility: {model_name}")
        plt.tight_layout()
        plt.savefig(FIGDIR / f"mae_by_road_visibility_{model_name}.png", dpi=220)
        plt.close()

    print("\nSaved:")
    print(FIGDIR)
    print(ERR_AQI_OUT)
    print(ERR_ROAD_OUT)
    print(PRED_OUT)


if __name__ == "__main__":
    main()