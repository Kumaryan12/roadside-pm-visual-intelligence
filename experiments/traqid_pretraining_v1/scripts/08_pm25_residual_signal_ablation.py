import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
    dt = pd.to_datetime(df[time_col], errors="coerce")

    hour = dt.dt.hour.fillna(0).astype(float)
    month = dt.dt.month.fillna(1).astype(float)
    dow = dt.dt.dayofweek.fillna(0).astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


def forbidden(col):
    c = col.lower()

    bad_exact = {
        "pm2.5",
        "pm10",
        "aqi",
        "log1p_pm2.5",
        "log1p_pm10",
        "log1p_aqi",
        "actual_pm2.5",
        "predicted_pm2.5",
        "residual_pm2.5",
        "target_pm2.5",
    }

    bad_contains = [
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

    if c in bad_exact:
        return True

    return any(x in c for x in bad_contains)


def build_feature_groups(manifest_path, yolo_path, road_path):
    manifest = normalize_row_id(safe_read_csv(manifest_path))
    yolo = normalize_row_id(safe_read_csv(yolo_path))
    road = normalize_row_id(safe_read_csv(road_path), fallback_name="sample_index")

    manifest = add_time_features(manifest, "created_at")

    tabular_cols = [
        "row_id",
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
    tabular_cols = [c for c in tabular_cols if c in manifest.columns]
    tabular = manifest[tabular_cols].copy()

    cat_cols = [c for c in ["Season", "Day_or_Night"] if c in tabular.columns]
    if cat_cols:
        tabular = pd.get_dummies(tabular, columns=cat_cols, dummy_na=True)

    yolo_keep = [
        c for c in yolo.columns
        if c == "row_id" or (
            pd.api.types.is_numeric_dtype(yolo[c])
            and not forbidden(c)
        )
    ]
    yolo = yolo[yolo_keep].copy()

    road_keep = [
        c for c in road.columns
        if c == "row_id" or (
            pd.api.types.is_numeric_dtype(road[c])
            and not forbidden(c)
        )
    ]
    road = road[road_keep].copy()

    row_ids = manifest[["row_id"]].drop_duplicates().copy()

    groups = {}

    groups["tabular_only"] = row_ids.merge(tabular, on="row_id", how="left")
    groups["yolo_only"] = row_ids.merge(yolo, on="row_id", how="left")
    groups["road_only"] = row_ids.merge(road, on="row_id", how="left")

    groups["yolo_plus_road"] = (
        row_ids
        .merge(yolo, on="row_id", how="left")
        .merge(road, on="row_id", how="left", suffixes=("", "_road"))
    )

    groups["yolo_plus_tabular"] = (
        row_ids
        .merge(yolo, on="row_id", how="left")
        .merge(tabular, on="row_id", how="left", suffixes=("", "_tabular"))
    )

    groups["road_plus_tabular"] = (
        row_ids
        .merge(road, on="row_id", how="left")
        .merge(tabular, on="row_id", how="left", suffixes=("", "_tabular"))
    )

    groups["full"] = (
        row_ids
        .merge(yolo, on="row_id", how="left")
        .merge(road, on="row_id", how="left", suffixes=("", "_road"))
        .merge(tabular, on="row_id", how="left", suffixes=("", "_tabular"))
    )

    return groups


def select_numeric_features(df):
    cols = []

    for c in df.columns:
        if c == "row_id":
            continue

        if forbidden(c):
            continue

        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool:
            cols.append(c)

    return sorted(set(cols))


def build_feature_matrix(row_features, feature_cols):
    max_row_id = int(row_features["row_id"].max())

    mat = np.full((max_row_id + 1, len(feature_cols)), np.nan, dtype=np.float32)

    row_ids = row_features["row_id"].astype(int).values
    mat[row_ids] = row_features[feature_cols].values.astype(np.float32)

    return mat


def sequence_aggregate_features(pred_df, feature_matrix):
    rows = []
    keep = []

    for idx, r in pred_df.iterrows():
        start = int(r["seq_start_row_id"])
        end = int(r["seq_end_row_id"])
        ids = np.arange(start, end + 1)

        if ids.min() < 0:
            continue
        if ids.max() >= feature_matrix.shape[0]:
            continue

        seq = feature_matrix[ids]

        with np.errstate(all="ignore"):
            seq_last = seq[-1]
            seq_mean = np.nanmean(seq, axis=0)
            seq_std = np.nanstd(seq, axis=0)
            seq_min = np.nanmin(seq, axis=0)
            seq_max = np.nanmax(seq, axis=0)

        feat = np.concatenate([seq_last, seq_mean, seq_std, seq_min, seq_max])
        rows.append(feat)
        keep.append(idx)

    X = np.asarray(rows, dtype=np.float32)
    kept = pred_df.loc[keep].reset_index(drop=True)

    X[~np.isfinite(X)] = np.nan

    return X, kept


def clean_features(X_val, X_test):
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    X_val[~np.isfinite(X_val)] = np.nan
    X_test[~np.isfinite(X_test)] = np.nan

    observed = np.isfinite(X_val).any(axis=0)

    X_val = X_val[:, observed]
    X_test = X_test[:, observed]

    med = np.nanmedian(X_val, axis=0)
    med[~np.isfinite(med)] = 0.0

    inds = np.where(np.isnan(X_val))
    X_val[inds] = np.take(med, inds[1])

    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(med, inds[1])

    std = np.std(X_val, axis=0)
    non_constant = std > 1e-12

    X_val = X_val[:, non_constant]
    X_test = X_test[:, non_constant]

    return X_val, X_test


def get_pm25_actual_base(df):
    actual_col = f"actual_{TARGET}"
    pred_col = f"predicted_{TARGET}"

    if actual_col not in df.columns:
        raise ValueError(f"Missing {actual_col}")

    if pred_col not in df.columns:
        raise ValueError(f"Missing {pred_col}")

    y_true = df[actual_col].values.astype(float)
    y_base = df[pred_col].values.astype(float)

    return y_true, y_base


def metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


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

    y_val_true, y_val_base = get_pm25_actual_base(val_pred)
    y_test_true, y_test_base = get_pm25_actual_base(test_pred)

    residual_val = y_val_true - y_val_base

    feature_groups = build_feature_groups(
        manifest_path=args.manifest,
        yolo_path=args.yolo_features,
        road_path=args.road_features,
    )

    models = {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=500,
                random_state=42,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
            ),
        ),
        "random_forest": make_pipeline(
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

    result_rows = []
    prediction_paths = []

    base_m = metrics(y_test_true, y_test_base)
    result_rows.append({
        "feature_group": "image_base_only",
        "residual_model": "none",
        "n_raw_features": 0,
        "n_aggregated_features": 0,
        **base_m,
    })

    for group_name, row_features in feature_groups.items():
        feature_cols = select_numeric_features(row_features)

        if not feature_cols:
            print(f"Skipping {group_name}: no features")
            continue

        feature_matrix = build_feature_matrix(row_features, feature_cols)

        X_val, val_kept = sequence_aggregate_features(val_pred, feature_matrix)
        X_test, test_kept = sequence_aggregate_features(test_pred, feature_matrix)

        X_val, X_test = clean_features(X_val, X_test)

        y_val_true_g, y_val_base_g = get_pm25_actual_base(val_kept)
        y_test_true_g, y_test_base_g = get_pm25_actual_base(test_kept)

        residual_val_g = y_val_true_g - y_val_base_g

        print("=" * 90)
        print("Feature group:", group_name)
        print("Raw features:", len(feature_cols))
        print("Aggregated features after cleaning:", X_val.shape[1])
        print("Val rows:", len(val_kept))
        print("Test rows:", len(test_kept))

        for model_name, model in models.items():
            print(f"Training {group_name} / {model_name}")

            try:
                model.fit(X_val, residual_val_g)
                residual_test_pred = model.predict(X_test)
            except Exception as e:
                print(f"Skipping {group_name} / {model_name}: {e}")
                continue

            corrected = y_test_base_g + args.correction_strength * residual_test_pred

            m = metrics(y_test_true_g, corrected)

            result_rows.append({
                "feature_group": group_name,
                "residual_model": model_name,
                "n_raw_features": len(feature_cols),
                "n_aggregated_features": X_val.shape[1],
                **m,
            })

            pred_df = test_kept[key_cols].copy()
            pred_df[f"actual_{TARGET}"] = y_test_true_g
            pred_df[f"base_predicted_{TARGET}"] = y_test_base_g
            pred_df[f"residual_correction_{TARGET}"] = residual_test_pred
            pred_df[f"corrected_predicted_{TARGET}"] = corrected
            pred_df[f"corrected_residual_{TARGET}"] = corrected - y_test_true_g
            pred_df["feature_group"] = group_name
            pred_df["residual_model"] = model_name

            pred_path = out_dir / f"predictions_{group_name}_{model_name}.csv"
            pred_df.to_csv(pred_path, index=False)
            prediction_paths.append(str(pred_path))

    results = pd.DataFrame(result_rows)
    results = results.sort_values("R2", ascending=False)

    results_path = out_dir / "pm25_residual_signal_ablation_results.csv"
    results_md_path = out_dir / "pm25_residual_signal_ablation_results.md"
    pred_index_path = out_dir / "prediction_files.txt"

    results.to_csv(results_path, index=False)
    results_md_path.write_text(results.to_markdown(index=False), encoding="utf-8")
    pred_index_path.write_text("\n".join(prediction_paths), encoding="utf-8")

    print("\n" + "=" * 90)
    print("PM2.5 RESIDUAL SIGNAL ABLATION COMPLETE")
    print("=" * 90)
    print("Base val predictions:", args.base_val_predictions)
    print("Base test predictions:", args.base_test_predictions)
    print()
    print(results.to_string(index=False))
    print()
    print("Saved:", results_path)
    print("Saved:", results_md_path)
    print("Saved:", pred_index_path)


if __name__ == "__main__":
    main()