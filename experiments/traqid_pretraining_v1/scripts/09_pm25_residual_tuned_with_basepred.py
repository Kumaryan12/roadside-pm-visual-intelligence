import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


TARGET = "PM2.5"


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_row_id(df, fallback_name=None):
    df = df.copy()

    if "row_id" in df.columns:
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
        df = df.dropna(subset=["row_id"]).copy()
        df["row_id"] = df["row_id"].astype(int)
        return df

    if fallback_name and fallback_name in df.columns:
        df = df.rename(columns={fallback_name: "row_id"})
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
        df = df.dropna(subset=["row_id"]).copy()
        df["row_id"] = df["row_id"].astype(int)
        return df

    raise ValueError("Could not find row_id column.")


def add_time_features(df, time_col="created_at"):
    df = df.copy()

    if time_col not in df.columns:
        return df

    dt = pd.to_datetime(df[time_col], errors="coerce")

    hour = dt.dt.hour.fillna(0).astype(float)
    month = dt.dt.month.fillna(1).astype(float)
    dayofweek = dt.dt.dayofweek.fillna(0).astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    return df


def is_forbidden_input_column(col):
    c = col.lower()

    forbidden_exact = {
        "pm2.5",
        "pm10",
        "aqi",
        "log1p_pm2.5",
        "log1p_pm10",
        "log1p_aqi",
        "target_pm2.5",
        "target_pm10",
        "target_aqi",
        "y_true_pm25",
        "y_pred_pm25",
    }

    forbidden_contains = [
        "pm2.5",
        "pm25",
        "pm_2_5",
        "pm10",
        "aqi",
        "y_true",
        "y_pred",
        "target",
        "split",
        "date_chrono",
        "front_path",
        "rear_path",
        "image",
        "path",
        "file",
        "error",
        "status",
    ]

    if c in forbidden_exact:
        return True

    return any(x in c for x in forbidden_contains)


def build_row_feature_table(manifest_path, yolo_path, road_path):
    manifest = safe_read_csv(manifest_path)
    yolo = safe_read_csv(yolo_path)
    road = safe_read_csv(road_path)

    manifest = normalize_row_id(manifest)
    yolo = normalize_row_id(yolo)
    road = normalize_row_id(road, fallback_name="sample_index")

    manifest = add_time_features(manifest, "created_at")

    base_cols = [
        "row_id",
        "created_at",
        "Temperature",
        "Humidity",
        "Season",
        "Day_or_Night",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dayofweek_sin",
        "dayofweek_cos",
    ]

    base_cols = [c for c in base_cols if c in manifest.columns]
    base = manifest[base_cols].copy()

    yolo_keep = [
        c for c in yolo.columns
        if c == "row_id"
        or (
            pd.api.types.is_numeric_dtype(yolo[c])
            and not is_forbidden_input_column(c)
        )
    ]

    road_keep = [
        c for c in road.columns
        if c == "row_id"
        or (
            pd.api.types.is_numeric_dtype(road[c])
            and not is_forbidden_input_column(c)
        )
    ]

    yolo = yolo[yolo_keep].copy()
    road = road[road_keep].copy()

    merged = base.merge(yolo, on="row_id", how="left", suffixes=("", "_yolo"))
    merged = merged.merge(road, on="row_id", how="left", suffixes=("", "_road"))

    cat_cols = [c for c in ["Season", "Day_or_Night"] if c in merged.columns]
    if cat_cols:
        merged = pd.get_dummies(merged, columns=cat_cols, dummy_na=True)

    merged = merged.sort_values("row_id").reset_index(drop=True)

    feature_cols = []
    for c in merged.columns:
        if c == "row_id":
            continue
        if is_forbidden_input_column(c):
            continue
        if pd.api.types.is_numeric_dtype(merged[c]) or merged[c].dtype == bool:
            feature_cols.append(c)

    return merged, sorted(set(feature_cols))


def make_feature_matrix(row_features, feature_cols):
    row_features = row_features.dropna(subset=["row_id"]).copy()
    row_features["row_id"] = row_features["row_id"].astype(int)

    max_row_id = int(row_features["row_id"].max())

    mat = np.full(
        (max_row_id + 1, len(feature_cols)),
        np.nan,
        dtype=np.float32,
    )

    row_ids = row_features["row_id"].values
    mat[row_ids] = row_features[feature_cols].values.astype(np.float32)

    return mat


def aggregate_one_sequence(seq):
    seq = np.asarray(seq, dtype=np.float32)

    with np.errstate(all="ignore"):
        last = seq[-1]
        mean = np.nanmean(seq, axis=0)
        std = np.nanstd(seq, axis=0)
        minv = np.nanmin(seq, axis=0)
        maxv = np.nanmax(seq, axis=0)

    out = np.concatenate([last, mean, std, minv, maxv], axis=0)
    out[~np.isfinite(out)] = np.nan

    return out


def build_sequence_features(pred_df, feature_matrix, T=7):
    X = []
    keep = []

    for i, row in pred_df.iterrows():
        start = int(row["seq_start_row_id"])
        end = int(row["seq_end_row_id"])
        ids = np.arange(start, end + 1)

        if len(ids) != T:
            continue

        if ids.min() < 0:
            continue

        if ids.max() >= feature_matrix.shape[0]:
            continue

        seq = feature_matrix[ids]
        agg = aggregate_one_sequence(seq)

        X.append(agg)
        keep.append(i)

    X = np.asarray(X, dtype=np.float32)
    kept_df = pred_df.iloc[keep].reset_index(drop=True)

    return X, kept_df


def clean_feature_columns(X_train, X_test, names):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    finite_train = np.isfinite(X_train)
    observed_count = finite_train.sum(axis=0)

    keep_mask = observed_count > 0

    X_train = X_train[:, keep_mask]
    X_test = X_test[:, keep_mask]
    names = [n for n, keep in zip(names, keep_mask) if keep]

    return X_train, X_test, names


def evaluate(actual, pred):
    return {
        "R2": float(r2_score(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAE": float(mean_absolute_error(actual, pred)),
    }


def make_models(seed):
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),

        "extra_trees_default": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=700,
                random_state=seed,
                n_jobs=-1,
                max_features=0.5,
                min_samples_leaf=2,
                bootstrap=False,
            )),
        ]),

        "extra_trees_conservative": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=900,
                random_state=seed,
                n_jobs=-1,
                max_features=0.35,
                min_samples_leaf=4,
                bootstrap=False,
            )),
        ]),

        "extra_trees_wide": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=900,
                random_state=seed,
                n_jobs=-1,
                max_features=0.8,
                min_samples_leaf=2,
                bootstrap=False,
            )),
        ]),

        "extra_trees_smooth": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=1000,
                random_state=seed,
                n_jobs=-1,
                max_features=0.5,
                min_samples_leaf=8,
                bootstrap=False,
            )),
        ]),

        "random_forest_default": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=700,
                random_state=seed,
                n_jobs=-1,
                max_features=0.5,
                min_samples_leaf=2,
                bootstrap=True,
            )),
        ]),

        "random_forest_smooth": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=900,
                random_state=seed,
                n_jobs=-1,
                max_features=0.5,
                min_samples_leaf=6,
                bootstrap=True,
            )),
        ]),

        "hgb": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=400,
                learning_rate=0.035,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=seed,
            )),
        ]),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-val-predictions", required=True)
    parser.add_argument("--base-test-predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--yolo-features", required=True)
    parser.add_argument("--road-features", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--T", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_pred = safe_read_csv(args.base_val_predictions)
    test_pred = safe_read_csv(args.base_test_predictions)

    actual_col = f"actual_{TARGET}"
    pred_col = f"predicted_{TARGET}"

    required_cols = [
        "sequence_id",
        "target_row_id",
        "target_created_at",
        "seq_start_row_id",
        "seq_end_row_id",
        actual_col,
        pred_col,
    ]

    for name, df in [("val", val_pred), ("test", test_pred)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} predictions missing columns: {missing}")

    row_features, raw_feature_cols = build_row_feature_table(
        manifest_path=args.manifest,
        yolo_path=args.yolo_features,
        road_path=args.road_features,
    )

    feature_matrix = make_feature_matrix(row_features, raw_feature_cols)

    X_val, val_kept = build_sequence_features(val_pred, feature_matrix, T=args.T)
    X_test, test_kept = build_sequence_features(test_pred, feature_matrix, T=args.T)

    agg_names = []
    for prefix in ["last", "mean", "std", "min", "max"]:
        for col in raw_feature_cols:
            agg_names.append(f"{prefix}__{col}")

    X_val, X_test, agg_names = clean_feature_columns(X_val, X_test, agg_names)

    val_base = val_kept[pred_col].values.astype(float)
    val_actual = val_kept[actual_col].values.astype(float)
    residual_val = val_actual - val_base

    test_base = test_kept[pred_col].values.astype(float)
    test_actual = test_kept[actual_col].values.astype(float)

    X_val_base = np.column_stack([
        X_val,
        val_base,
        np.abs(val_base),
        val_base ** 2,
    ])

    X_test_base = np.column_stack([
        X_test,
        test_base,
        np.abs(test_base),
        test_base ** 2,
    ])

    feature_names = agg_names + [
        "base_pred_pm25",
        "abs_base_pred_pm25",
        "base_pred_pm25_squared",
    ]

    base_metrics = evaluate(test_actual, test_base)

    rows = [{
        "model": "image_base_only",
        "uses_base_pred_feature": False,
        "R2": base_metrics["R2"],
        "RMSE": base_metrics["RMSE"],
        "MAE": base_metrics["MAE"],
    }]

    prediction_outputs = []

    model_dict = make_models(args.seed)

    for model_name, model in model_dict.items():
        print("Training:", model_name)

        model.fit(X_val, residual_val)
        pred_residual = model.predict(X_test)
        final_pred = test_base + pred_residual

        m = evaluate(test_actual, final_pred)

        rows.append({
            "model": model_name,
            "uses_base_pred_feature": False,
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
        })

        out = test_kept.copy()
        out["residual_model"] = model_name
        out["uses_base_pred_feature"] = False
        out["base_pred_PM2.5"] = test_base
        out["predicted_residual_PM2.5"] = pred_residual
        out["final_predicted_PM2.5"] = final_pred
        out["final_residual_PM2.5"] = final_pred - test_actual
        prediction_outputs.append(out)

    for model_name, model in model_dict.items():
        model_name2 = f"{model_name}_plus_basepred"
        print("Training:", model_name2)

        model.fit(X_val_base, residual_val)
        pred_residual = model.predict(X_test_base)
        final_pred = test_base + pred_residual

        m = evaluate(test_actual, final_pred)

        rows.append({
            "model": model_name2,
            "uses_base_pred_feature": True,
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
        })

        out = test_kept.copy()
        out["residual_model"] = model_name2
        out["uses_base_pred_feature"] = True
        out["base_pred_PM2.5"] = test_base
        out["predicted_residual_PM2.5"] = pred_residual
        out["final_predicted_PM2.5"] = final_pred
        out["final_residual_PM2.5"] = final_pred - test_actual
        prediction_outputs.append(out)

    results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)

    all_predictions = pd.concat(prediction_outputs, ignore_index=True)

    results_path = out_dir / "pm25_tuned_residual_results.csv"
    results_md_path = out_dir / "pm25_tuned_residual_results.md"
    predictions_path = out_dir / "pm25_tuned_residual_predictions.csv"
    feature_path = out_dir / "pm25_tuned_residual_feature_columns.txt"

    results.to_csv(results_path, index=False)
    results_md_path.write_text(results.to_markdown(index=False), encoding="utf-8")
    all_predictions.to_csv(predictions_path, index=False)
    feature_path.write_text("\n".join(feature_names), encoding="utf-8")

    print("\n" + "=" * 90)
    print("PM2.5 TUNED RESIDUAL CORRECTION COMPLETE")
    print("=" * 90)
    print("Val rows used:", len(val_kept))
    print("Test rows used:", len(test_kept))
    print("Raw engineered features:", len(raw_feature_cols))
    print("Aggregated features after cleaning:", len(agg_names))
    print("Aggregated + base-pred features:", len(feature_names))
    print()
    print(results.to_string(index=False))
    print()
    print("Saved:", results_path)
    print("Saved:", results_md_path)
    print("Saved:", predictions_path)
    print("Saved:", feature_path)


if __name__ == "__main__":
    main()