from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ADV_YOLO_CSV = Path(
    "experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_advanced_features_front.csv"
)

OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = OUTDIR / "selected_advanced_yolo_fusion_results.csv"
RANKING_OUT = OUTDIR / "selected_advanced_yolo_feature_ranking.csv"
PRED_OUT = OUTDIR / "selected_advanced_yolo_predictions.csv"

TARGET = "PM2.5"
EPS = 1e-6
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def get_metrics(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    sp = np.nan
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        sp = spearmanr(y_true, y_pred).correlation

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

    # meteorology × traffic-area interactions
    df["humidity_x_vehicle_area"] = df["Humidity"] * df["total_vehicle_area_ratio"]
    df["humidity_x_near_vehicle_area"] = df["Humidity"] * df["near_vehicle_area_ratio"]
    df["humidity_x_heavy_area"] = df["Humidity"] * df["heavy_vehicle_area_ratio"]

    df["temperature_x_vehicle_area"] = df["Temperature"] * df["total_vehicle_area_ratio"]
    df["temperature_x_near_vehicle_area"] = df["Temperature"] * df["near_vehicle_area_ratio"]
    df["temperature_x_heavy_area"] = df["Temperature"] * df["heavy_vehicle_area_ratio"]

    # threshold-style features
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


# ---------------------------------------------------------------------
# Matrix helper
# ---------------------------------------------------------------------
def prepare_matrix(train_df, test_df, numeric_cols, categorical_cols=None):
    categorical_cols = categorical_cols or []

    X_train_num = train_df[numeric_cols].values.astype(float)
    X_test_num = test_df[numeric_cols].values.astype(float)

    parts_train = [X_train_num]
    parts_test = [X_test_num]
    feature_names = list(numeric_cols)

    if categorical_cols:
        cat_all = pd.concat(
            [train_df[categorical_cols], test_df[categorical_cols]],
            axis=0,
        )

        cat_all = pd.get_dummies(cat_all, columns=categorical_cols, drop_first=False)

        X_train_cat = cat_all.iloc[: len(train_df)].values.astype(float)
        X_test_cat = cat_all.iloc[len(train_df) :].values.astype(float)

        parts_train.append(X_train_cat)
        parts_test.append(X_test_cat)
        feature_names += list(cat_all.columns)

    X_train = np.hstack(parts_train)
    X_test = np.hstack(parts_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, feature_names


# ---------------------------------------------------------------------
# OOF prediction helper for residual fusion
# ---------------------------------------------------------------------
def get_oof_tabular_predictions(train_val_df, tabular_features, categorical_features, model):
    y = train_val_df[TARGET].values.astype(float)
    oof_pred = np.zeros(len(train_val_df), dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    split_labels = train_val_df["aqi_cat"].values

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_val_df, split_labels), start=1):
        fold_train = train_val_df.iloc[tr_idx].copy()
        fold_val = train_val_df.iloc[va_idx].copy()

        X_tr, X_va, _ = prepare_matrix(
            fold_train,
            fold_val,
            numeric_cols=tabular_features,
            categorical_cols=categorical_features,
        )

        m = clone(model)
        m.fit(X_tr, fold_train[TARGET].values.astype(float))
        oof_pred[va_idx] = m.predict(X_va)

        print(f"OOF fold {fold} done.")

    return oof_pred


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading:", ADV_YOLO_CSV)

    df = pd.read_csv(ADV_YOLO_CSV)

    df = df[df["yolo_adv_status"] == "success"].copy()
    df = df.dropna(subset=[TARGET, "Temperature", "Humidity", "aqi_cat", "created_at"])

    df = add_time_features(df)
    df = add_advanced_yolo_interactions(df)

    # -----------------------------------------------------------------
    # Feature definitions
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

    all_yolo_features = (
        old_yolo_like_features
        + advanced_yolo_raw_features
        + advanced_yolo_engineered_features
    )

    all_needed_numeric = sorted(set(tabular_features + all_yolo_features))

    # Create missing columns safely
    for c in all_needed_numeric:
        if c not in df.columns:
            print("Missing feature column, creating zero:", c)
            df[c] = 0.0

    # Clean numeric values
    for c in all_needed_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[all_needed_numeric] = df[all_needed_numeric].replace([np.inf, -np.inf], np.nan)
    df[all_needed_numeric] = df[all_needed_numeric].fillna(
        df[all_needed_numeric].median(numeric_only=True)
    )
    df[all_needed_numeric] = df[all_needed_numeric].fillna(0)

    print("\nRows after cleaning:", len(df))
    print("\nAQI distribution:")
    print(df["aqi_cat"].value_counts().to_string())

    # -----------------------------------------------------------------
    # Split: same pattern as previous script
    # -----------------------------------------------------------------
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["aqi_cat"],
    )

    train, val = train_test_split(
        train_val,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train_val["aqi_cat"],
    )

    print("\nSplit:")
    print({"train": len(train), "val": len(val), "test": len(test)})

    # -----------------------------------------------------------------
    # Step 1: train tabular model on train and use validation residuals
    # for YOLO feature ranking. No test leakage.
    # -----------------------------------------------------------------
    tabular_selector_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=24,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    X_train_tab, X_val_tab, _ = prepare_matrix(
        train,
        val,
        numeric_cols=tabular_features,
        categorical_cols=categorical_features,
    )

    tabular_selector_model.fit(X_train_tab, train[TARGET].values.astype(float))
    val_tab_pred = tabular_selector_model.predict(X_val_tab)

    val_residual = val[TARGET].values.astype(float) - val_tab_pred

    print("\nValidation tabular residual summary:")
    print(pd.Series(val_residual).describe().to_string())

    # -----------------------------------------------------------------
    # Step 2: rank YOLO features by relation to validation residual
    # -----------------------------------------------------------------
    ranking_rows = []

    for feat in all_yolo_features:
        x = val[feat].values.astype(float)

        if np.std(x) == 0 or np.std(val_residual) == 0:
            sp = 0.0
            pr = 0.0
        else:
            sp = spearmanr(x, val_residual).correlation
            pr = pearsonr(x, val_residual)[0]

            if np.isnan(sp):
                sp = 0.0
            if np.isnan(pr):
                pr = 0.0

        # also track direct relation with PM2.5 for interpretation only
        y_val = val[TARGET].values.astype(float)
        if np.std(x) == 0 or np.std(y_val) == 0:
            sp_pm = 0.0
        else:
            sp_pm = spearmanr(x, y_val).correlation
            if np.isnan(sp_pm):
                sp_pm = 0.0

        ranking_rows.append(
            {
                "feature": feat,
                "spearman_with_tabular_residual": sp,
                "abs_spearman_with_tabular_residual": abs(sp),
                "pearson_with_tabular_residual": pr,
                "abs_pearson_with_tabular_residual": abs(pr),
                "spearman_with_pm25_validation": sp_pm,
                "abs_spearman_with_pm25_validation": abs(sp_pm),
            }
        )

    ranking = pd.DataFrame(ranking_rows)

    ranking["selection_score"] = (
        0.70 * ranking["abs_spearman_with_tabular_residual"]
        + 0.30 * ranking["abs_pearson_with_tabular_residual"]
    )

    ranking = ranking.sort_values("selection_score", ascending=False)
    ranking.to_csv(RANKING_OUT, index=False)

    print("\nTop 30 YOLO features by residual relevance:")
    print(ranking.head(30).to_string(index=False))

    print("\nSaved feature ranking:", RANKING_OUT)

    # -----------------------------------------------------------------
    # Step 3: final train_val/test evaluation
    # -----------------------------------------------------------------
    train_val = train_val.copy()
    test = test.copy()

    y_train_val = train_val[TARGET].values.astype(float)
    y_test = test[TARGET].values.astype(float)

    final_models = {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=24,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=1,
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

    residual_models = {
        "residual_rf": RandomForestRegressor(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "residual_extra_trees": ExtraTreesRegressor(
            n_estimators=700,
            max_depth=18,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "residual_hgb": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
    }

    records = []
    pred_records = []

    # Baseline: tabular-only final model on train_val
    for model_name, model in final_models.items():
        X_tv, X_te, _ = prepare_matrix(
            train_val,
            test,
            numeric_cols=tabular_features,
            categorical_cols=categorical_features,
        )

        print("\n" + "-" * 100)
        print("FINAL BASELINE:", "tabular_only", model_name)

        t0 = time.time()
        m = clone(model)
        m.fit(X_tv, y_train_val)
        pred = np.clip(m.predict(X_te), 0, None)
        elapsed = time.time() - t0

        row = {
            "experiment": "tabular_only",
            "feature_set": "tabular_only",
            "model": model_name,
            "top_k_yolo": 0,
            "n_features": X_tv.shape[1],
            "n_train": len(train_val),
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
                    "experiment": row["experiment"],
                    "model": model_name,
                    "y_true_pm25": y_test,
                    "y_pred_pm25": pred,
                    "residual_pred_minus_true": pred - y_test,
                }
            )
        )

        print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman"]})

    # Compute OOF residuals for residual-fusion final models
    tabular_oof_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=24,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nBuilding OOF tabular residuals for residual fusion...")
    oof_tab_pred = get_oof_tabular_predictions(
        train_val,
        tabular_features=tabular_features,
        categorical_features=categorical_features,
        model=tabular_oof_model,
    )

    oof_residual = y_train_val - oof_tab_pred

    # Final tabular model for residual fusion test prediction
    X_tv_tab, X_te_tab, _ = prepare_matrix(
        train_val,
        test,
        numeric_cols=tabular_features,
        categorical_cols=categorical_features,
    )

    tab_final_model = clone(tabular_oof_model)
    tab_final_model.fit(X_tv_tab, y_train_val)
    tab_test_pred = tab_final_model.predict(X_te_tab)

    # Try different K values
    top_k_values = [5, 10, 15, 20, 30, 40]

    for k in top_k_values:
        selected = ranking.head(k)["feature"].tolist()

        print("\n" + "=" * 100)
        print(f"TOP-K SELECTED YOLO FEATURES: k={k}")
        print(selected)

        # -------------------------------------------------------------
        # Early fusion: tabular + selected YOLO
        # -------------------------------------------------------------
        fusion_cols = tabular_features + selected

        X_tv_fusion, X_te_fusion, _ = prepare_matrix(
            train_val,
            test,
            numeric_cols=fusion_cols,
            categorical_cols=categorical_features,
        )

        for model_name, model in final_models.items():
            print("\n" + "-" * 100)
            print("EARLY FUSION:", f"tabular_plus_selected_yolo_k{k}", model_name)
            print("Feature matrix:", X_tv_fusion.shape, X_te_fusion.shape)

            t0 = time.time()
            m = clone(model)
            m.fit(X_tv_fusion, y_train_val)
            pred = np.clip(m.predict(X_te_fusion), 0, None)
            elapsed = time.time() - t0

            row = {
                "experiment": f"early_fusion_top{k}",
                "feature_set": "tabular_plus_selected_yolo",
                "model": model_name,
                "top_k_yolo": k,
                "selected_yolo_features": "|".join(selected),
                "n_features": X_tv_fusion.shape[1],
                "n_train": len(train_val),
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
                        "experiment": row["experiment"],
                        "model": model_name,
                        "y_true_pm25": y_test,
                        "y_pred_pm25": pred,
                        "residual_pred_minus_true": pred - y_test,
                    }
                )
            )

            res = pd.DataFrame(records).sort_values("RMSE")
            res.to_csv(RESULTS_OUT, index=False)
            pd.concat(pred_records, ignore_index=True).to_csv(PRED_OUT, index=False)

            print({key: row[key] for key in ["MAE", "RMSE", "R2", "Spearman"]})

        # -------------------------------------------------------------
        # Residual fusion: selected YOLO predicts tabular residual
        # -------------------------------------------------------------
        X_tv_yolo, X_te_yolo, _ = prepare_matrix(
            train_val,
            test,
            numeric_cols=selected,
            categorical_cols=[],
        )

        for model_name, model in residual_models.items():
            print("\n" + "-" * 100)
            print("RESIDUAL FUSION:", f"selected_yolo_residual_k{k}", model_name)
            print("Feature matrix:", X_tv_yolo.shape, X_te_yolo.shape)

            t0 = time.time()
            m = clone(model)
            m.fit(X_tv_yolo, oof_residual)
            residual_pred = m.predict(X_te_yolo)

            final_pred = np.clip(tab_test_pred + residual_pred, 0, None)
            elapsed = time.time() - t0

            row = {
                "experiment": f"residual_fusion_top{k}",
                "feature_set": "selected_yolo_residual",
                "model": model_name,
                "top_k_yolo": k,
                "selected_yolo_features": "|".join(selected),
                "n_features": X_tv_yolo.shape[1],
                "n_train": len(train_val),
                "n_test": len(test),
                "elapsed_sec": elapsed,
            }

            row.update(get_metrics(y_test, final_pred))
            records.append(row)

            pred_records.append(
                pd.DataFrame(
                    {
                        "row_id": test["row_id"].values,
                        "created_at": test["created_at"].values,
                        "aqi_cat": test["aqi_cat"].values,
                        "experiment": row["experiment"],
                        "model": model_name,
                        "y_true_pm25": y_test,
                        "y_pred_pm25": final_pred,
                        "residual_pred_minus_true": final_pred - y_test,
                    }
                )
            )

            res = pd.DataFrame(records).sort_values("RMSE")
            res.to_csv(RESULTS_OUT, index=False)
            pd.concat(pred_records, ignore_index=True).to_csv(PRED_OUT, index=False)

            print({key: row[key] for key in ["MAE", "RMSE", "R2", "Spearman"]})

    # -----------------------------------------------------------------
    # Save final
    # -----------------------------------------------------------------
    res = pd.DataFrame(records).sort_values("RMSE")
    res.to_csv(RESULTS_OUT, index=False)

    preds = pd.concat(pred_records, ignore_index=True)
    preds.to_csv(PRED_OUT, index=False)

    print("\n" + "=" * 100)
    print("FINAL RESULTS:")
    print(res.to_string(index=False))

    print("\nSaved:")
    print(RESULTS_OUT)
    print(RANKING_OUT)
    print(PRED_OUT)


if __name__ == "__main__":
    main()