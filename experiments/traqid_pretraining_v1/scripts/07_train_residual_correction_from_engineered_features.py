import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["PM2.5", "PM10", "aqi"]


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def add_time_features(df, time_col="created_at"):
    df = df.copy()
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
        "actual_pm2.5",
        "actual_pm10",
        "actual_aqi",
        "predicted_pm2.5",
        "predicted_pm10",
        "predicted_aqi",
        "y_true_pm25",
        "y_pred_pm25",
    }

    forbidden_contains = [
        "pm2.5",
        "pm25",
        "pm_2_5",
        "pm10",
        "aqi",
        "actual_",
        "predicted_",
        "residual_",
        "target",
        "y_true",
        "y_pred",
        "split",
        "date_chrono",
        "created_at",
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


def build_row_level_feature_table(manifest_path, yolo_path, road_path):
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

    return merged


def select_feature_cols(df):
    cols = []

    for c in df.columns:
        if c == "row_id":
            continue

        if is_forbidden_input_column(c):
            continue

        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool:
            cols.append(c)

    return sorted(set(cols))


def build_row_feature_matrix(row_features, feature_cols):
    max_row_id = int(row_features["row_id"].max())

    mat = np.full(
        (max_row_id + 1, len(feature_cols)),
        np.nan,
        dtype=np.float32,
    )

    row_ids = row_features["row_id"].astype(int).values
    mat[row_ids] = row_features[feature_cols].values.astype(np.float32)

    return mat


def sequence_features_from_predictions(pred_df, feature_matrix):
    rows = []
    keep_indices = []

    for idx, r in pred_df.iterrows():
        start = int(r["seq_start_row_id"])
        end = int(r["seq_end_row_id"])
        ids = np.arange(start, end + 1)

        if ids.min() < 0:
            continue

        if ids.max() >= feature_matrix.shape[0]:
            continue

        seq = feature_matrix[ids]

        seq_mean = np.nanmean(seq, axis=0)
        seq_std = np.nanstd(seq, axis=0)
        seq_min = np.nanmin(seq, axis=0)
        seq_max = np.nanmax(seq, axis=0)
        seq_last = seq[-1]

        feat = np.concatenate([
            seq_last,
            seq_mean,
            seq_std,
            seq_min,
            seq_max,
        ])

        rows.append(feat)
        keep_indices.append(idx)

    X = np.asarray(rows, dtype=np.float32)
    kept = pred_df.loc[keep_indices].reset_index(drop=True)

    X[~np.isfinite(X)] = np.nan

    return X, kept


def evaluate_arrays(y_true, y_pred, prefix):
    rows = []

    for i, target in enumerate(TARGETS):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        rows.append({
            "model": prefix,
            "target": target,
            "R2": float(r2_score(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)),
        })

    rows.append({
        "model": prefix,
        "target": "Average",
        "R2": float(np.mean([r["R2"] for r in rows])),
        "RMSE": float(np.mean([r["RMSE"] for r in rows])),
        "MAE": float(np.mean([r["MAE"] for r in rows])),
    })

    return rows


def get_y_actual_pred(df):
    y_true = []
    y_base = []

    for target in TARGETS:
        actual_col = f"actual_{target}"
        pred_col = f"predicted_{target}"

        if actual_col not in df.columns:
            raise ValueError(f"Missing column: {actual_col}")

        if pred_col not in df.columns:
            raise ValueError(f"Missing column: {pred_col}")

        y_true.append(df[actual_col].values.astype(float))
        y_base.append(df[pred_col].values.astype(float))

    y_true = np.vstack(y_true).T
    y_base = np.vstack(y_base).T

    return y_true, y_base


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-val-predictions", required=True)
    parser.add_argument("--base-test-predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--yolo-features", required=True)
    parser.add_argument("--road-features", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--correction-strength", type=float, default=1.0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_pred = safe_read_csv(args.base_val_predictions)
    test_pred = safe_read_csv(args.base_test_predictions)

    key_cols = [
        "sequence_id",
        "target_row_id",
        "target_created_at",
        "seq_start_row_id",
        "seq_end_row_id",
    ]

    for c in key_cols:
        if c not in val_pred.columns:
            raise ValueError(f"Validation predictions missing column: {c}")
        if c not in test_pred.columns:
            raise ValueError(f"Test predictions missing column: {c}")

    row_features = build_row_level_feature_table(
        manifest_path=args.manifest,
        yolo_path=args.yolo_features,
        road_path=args.road_features,
    )

    feature_cols = select_feature_cols(row_features)
    feature_matrix = build_row_feature_matrix(row_features, feature_cols)

    X_val, val_kept = sequence_features_from_predictions(val_pred, feature_matrix)
    X_test, test_kept = sequence_features_from_predictions(test_pred, feature_matrix)

    y_val_true, y_val_base = get_y_actual_pred(val_kept)
    y_test_true, y_test_base = get_y_actual_pred(test_kept)

    residual_val = y_val_true - y_val_base

    models = {
    "ridge_residual": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MultiOutputRegressor(Ridge(alpha=10.0)),
    ),
    "extra_trees_residual": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
        ),
    ),
    "random_forest_residual": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
        ),
    ),
}
    all_metric_rows = []
    all_predictions = {}

    all_metric_rows.extend(evaluate_arrays(
        y_test_true,
        y_test_base,
        "base_image_model",
    ))

    for model_name, model in models.items():
        print(f"Training residual model: {model_name}")

        try:
            model.fit(X_val, residual_val)
            residual_test_pred = model.predict(X_test)
        except Exception as e:
            print(f"Skipping {model_name} because it failed: {e}")
            continue
        corrected = y_test_base + args.correction_strength * residual_test_pred

        all_metric_rows.extend(evaluate_arrays(
            y_test_true,
            corrected,
            model_name,
        ))

        pred_df = test_kept[key_cols].copy()

        for i, target in enumerate(TARGETS):
            pred_df[f"actual_{target}"] = y_test_true[:, i]
            pred_df[f"base_predicted_{target}"] = y_test_base[:, i]
            pred_df[f"residual_correction_{target}"] = residual_test_pred[:, i]
            pred_df[f"corrected_predicted_{target}"] = corrected[:, i]
            pred_df[f"corrected_residual_{target}"] = corrected[:, i] - y_test_true[:, i]

        pred_path = out_dir / f"{model_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        all_predictions[model_name] = pred_path

    metrics = pd.DataFrame(all_metric_rows)
    metrics = metrics.sort_values(["target", "R2"], ascending=[True, False])

    avg = metrics[metrics["target"] == "Average"].sort_values("R2", ascending=False)

    metrics_path = out_dir / "residual_correction_metrics.csv"
    avg_path = out_dir / "residual_correction_average_ranking.csv"
    feature_path = out_dir / "residual_feature_columns.txt"

    metrics.to_csv(metrics_path, index=False)
    avg.to_csv(avg_path, index=False)
    feature_path.write_text("\n".join(feature_cols), encoding="utf-8")

    print("=" * 90)
    print("RESIDUAL CORRECTION COMPLETE")
    print("=" * 90)
    print("Validation rows used for residual training:", len(val_kept))
    print("Test rows:", len(test_kept))
    print("Base val predictions:", args.base_val_predictions)
    print("Base test predictions:", args.base_test_predictions)
    print("Engineered row features:", len(feature_cols))
    print("Residual feature dim after T aggregation:", X_val.shape[1])
    print()
    print("Average ranking:")
    print(avg.to_string(index=False))
    print()
    print("Saved:", metrics_path)
    print("Saved:", avg_path)
    print("Saved:", feature_path)


if __name__ == "__main__":
    main()