import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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

    forbidden_contains = [
        "pm2.5",
        "pm25",
        "pm_2_5",
        "pm10",
        "aqi",
        "target",
        "actual",
        "predicted",
        "residual",
        "error",
        "split",
        "date_chrono",
        "front_path",
        "rear_path",
        "image",
        "path",
        "file",
        "status",
    ]

    return any(x in c for x in forbidden_contains)


def build_row_feature_table(manifest_path, yolo_path, road_path):
    manifest = safe_read_csv(manifest_path)
    yolo = safe_read_csv(yolo_path)
    road = safe_read_csv(road_path)

    manifest = normalize_row_id(manifest)
    yolo = normalize_row_id(yolo)
    road = normalize_row_id(road, fallback_name="sample_index")

    if "created_at" not in manifest.columns:
        raise ValueError("created_at missing from manifest")

    manifest["created_at"] = pd.to_datetime(manifest["created_at"], errors="coerce")
    manifest = manifest.dropna(subset=["created_at", TARGET]).copy()

    manifest = add_time_features(manifest, "created_at")

    base_cols = [
        "row_id",
        "created_at",
        TARGET,
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

    feature_cols = []
    for c in merged.columns:
        if c in ["row_id", "created_at", TARGET]:
            continue
        if is_forbidden_input_column(c):
            continue
        if pd.api.types.is_numeric_dtype(merged[c]) or merged[c].dtype == bool:
            feature_cols.append(c)

    merged = merged.sort_values("row_id").reset_index(drop=True)

    return merged, sorted(set(feature_cols))


def make_split(df, split_mode, train_frac, val_frac, seed):
    df = df.copy()

    if split_mode == "random":
        idx = np.arange(len(df))

        trainval_idx, test_idx = train_test_split(
            idx,
            train_size=train_frac + val_frac,
            random_state=seed,
            shuffle=True,
        )

        relative_val_frac = val_frac / (train_frac + val_frac)

        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=relative_val_frac,
            random_state=seed,
            shuffle=True,
        )

        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )

    if split_mode == "chrono":
        df = df.sort_values("created_at").reset_index(drop=True)

        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        return (
            df.iloc[:train_end].reset_index(drop=True),
            df.iloc[train_end:val_end].reset_index(drop=True),
            df.iloc[val_end:].reset_index(drop=True),
        )

    raise ValueError(f"Unknown split_mode: {split_mode}")


def evaluate(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def make_models(seed):
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),

        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=800,
                random_state=seed,
                n_jobs=-1,
                max_features=0.5,
                min_samples_leaf=2,
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
            )),
        ]),

        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=700,
                random_state=seed,
                n_jobs=-1,
                max_features=0.5,
                min_samples_leaf=3,
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

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--yolo-features", required=True)
    parser.add_argument("--road-features", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--split-mode", choices=["random", "chrono"], default="random")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = build_row_feature_table(
        manifest_path=args.manifest,
        yolo_path=args.yolo_features,
        road_path=args.road_features,
    )

    train_df, val_df, test_df = make_split(
        df,
        split_mode=args.split_mode,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET].values.astype(float)

    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET].values.astype(float)

    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET].values.astype(float)

    rows = []
    prediction_outputs = []

    # Baselines
    train_mean = float(np.mean(y_train))

    for split_name, y, n in [
        ("val", y_val, len(y_val)),
        ("test", y_test, len(y_test)),
    ]:
        pred = np.full(n, train_mean)
        m = evaluate(y, pred)
        rows.append({
            "model": "train_mean_baseline",
            "split": split_name,
            "target": TARGET,
            "split_mode": args.split_mode,
            **m,
            "n_features": len(feature_cols),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        })

    models = make_models(args.seed)

    for model_name, model in models.items():
        print("Training:", model_name)

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        for split_name, split_df, y, pred in [
            ("val", val_df, y_val, val_pred),
            ("test", test_df, y_test, test_pred),
        ]:
            m = evaluate(y, pred)

            rows.append({
                "model": model_name,
                "split": split_name,
                "target": TARGET,
                "split_mode": args.split_mode,
                **m,
                "n_features": len(feature_cols),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
            })

            pred_df = split_df[["row_id", "created_at", TARGET]].copy()
            pred_df = pred_df.rename(columns={TARGET: "actual_PM2.5"})
            pred_df["predicted_PM2.5"] = pred
            pred_df["model"] = model_name
            pred_df["split"] = split_name
            pred_df["split_mode"] = args.split_mode

            prediction_outputs.append(pred_df)

    results = pd.DataFrame(rows).sort_values(["split", "R2"], ascending=[True, False])
    predictions = pd.concat(prediction_outputs, ignore_index=True)

    metrics_path = out_dir / "metrics.csv"
    metrics_md_path = out_dir / "metrics.md"
    predictions_path = out_dir / "predictions.csv"
    feature_cols_path = out_dir / "feature_columns.txt"

    results.to_csv(metrics_path, index=False)
    metrics_md_path.write_text(results.to_markdown(index=False), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    feature_cols_path.write_text("\n".join(feature_cols), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SINGLE-IMAGE YOLO + ROAD + TABULAR PM2.5 COMPLETE")
    print("=" * 90)
    print("Split mode:", args.split_mode)
    print("Rows:", len(df))
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    print("Features:", len(feature_cols))
    print()
    print(results.to_string(index=False))
    print()
    print("Saved:", metrics_path)
    print("Saved:", metrics_md_path)
    print("Saved:", predictions_path)
    print("Saved:", feature_cols_path)


if __name__ == "__main__":
    main()