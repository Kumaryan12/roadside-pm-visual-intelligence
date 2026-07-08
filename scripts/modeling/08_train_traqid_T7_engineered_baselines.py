import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["PM2.5", "PM10", "aqi"]


def metric_rows(y_true, y_pred, model_name, feature_set, split_mode, train_rows, test_rows):
    rows = []
    target_rows = []

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    for i, target in enumerate(TARGETS):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae = float(np.mean(np.abs(yt - yp)))
        r2 = float(r2_score(yt, yp))

        row = {
            "split_mode": split_mode,
            "feature_set": feature_set,
            "model": model_name,
            "target": target,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "train_rows": train_rows,
            "test_rows": test_rows,
        }

        rows.append(row)
        target_rows.append(row)

    rows.append({
        "split_mode": split_mode,
        "feature_set": feature_set,
        "model": model_name,
        "target": "Average",
        "R2": float(np.mean([r["R2"] for r in target_rows])),
        "RMSE": float(np.mean([r["RMSE"] for r in target_rows])),
        "MAE": float(np.mean([r["MAE"] for r in target_rows])),
        "train_rows": train_rows,
        "test_rows": test_rows,
    })

    return rows


def select_lagged_cols(df, groups):
    cols = []

    for c in df.columns:
        if "__" not in c:
            continue

        base = c.split("__", 1)[1]

        if "tabular" in groups:
            tabular_terms = [
                "Temperature",
                "Humidity",
                "Season_",
                "Day_or_Night_",
                "aqi_cat_",
            ]
            if any(base.startswith(t) for t in tabular_terms):
                cols.append(c)
                continue

        if "yolo" in groups:
            yolo_terms = [
                "car_",
                "bus_",
                "truck_",
                "auto_",
                "motorcycle_",
                "bicycle_",
                "person_",
                "total_vehicle_",
                "heavy_vehicle_",
                "two_wheeler_",
                "near_",
                "mean_vehicle_",
                "max_vehicle_",
                "vehicle_area_share_",
                "vehicle_count_share_",
                "traffic_mix_",
                "bottom_half_",
                "center_lane_",
                "left_half_",
                "right_half_",
                "detections_total_raw",
                "image_width",
                "image_height",
                "image_area",
            ]
            if any(base.startswith(t) for t in yolo_terms):
                cols.append(c)
                continue

        if "road" in groups:
            road_terms = [
                "road_area_pixels",
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
            if any(base.startswith(t) for t in road_terms):
                cols.append(c)
                continue

    return sorted(set(cols))


def build_splits(df, split_mode, train_frac, seed):
    if split_mode == "random":
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_idx = int(len(shuffled) * train_frac)
        train = shuffled.iloc[:split_idx].copy()
        test = shuffled.iloc[split_idx:].copy()

    elif split_mode == "chrono":
        train = df[df["target_split_date_chrono"] == "train"].copy()
        test = df[df["target_split_date_chrono"] == "test"].copy()

    elif split_mode == "chrono_valtest":
        train = df[df["target_split_date_chrono"] == "train"].copy()
        test = df[df["target_split_date_chrono"].isin(["val", "test"])].copy()

    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    return train, test


def previous_sequence_persistence(df, test):
    full = df.sort_values("target_row_id").reset_index(drop=True)

    row_to_pos = {
        int(row_id): pos
        for pos, row_id in enumerate(full["target_row_id"].values)
    }

    full_y = full[TARGETS].to_numpy(dtype=float)

    true_rows = []
    pred_rows = []

    for row_id in test["target_row_id"].values:
        pos = row_to_pos.get(int(row_id), None)
        if pos is None or pos == 0:
            continue

        true_rows.append(full_y[pos])
        pred_rows.append(full_y[pos - 1])

    return np.asarray(true_rows), np.asarray(pred_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_engineered_T7_sequence_table.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/traqid_T7_engineered_baselines",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "chrono", "chrono_valtest"],
        default="random",
    )
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    table_path = Path(args.table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table_path)
    df = df.dropna(subset=TARGETS).copy()
    df = df.sort_values("target_row_id").reset_index(drop=True)

    train, test = build_splits(df, args.split_mode, args.train_frac, args.seed)

    y_train = train[TARGETS].to_numpy(dtype=float)
    y_test = test[TARGETS].to_numpy(dtype=float)

    feature_sets = {
        "T7_tabular_only": select_lagged_cols(df, ["tabular"]),
        "T7_yolo_only": select_lagged_cols(df, ["yolo"]),
        "T7_road_only": select_lagged_cols(df, ["road"]),
        "T7_tabular_plus_yolo": select_lagged_cols(df, ["tabular", "yolo"]),
        "T7_tabular_plus_road": select_lagged_cols(df, ["tabular", "road"]),
        "T7_tabular_plus_yolo_plus_road": select_lagged_cols(df, ["tabular", "yolo", "road"]),
    }

    model_specs = {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MultiOutputRegressor(Ridge(alpha=10.0)),
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            random_state=args.seed,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            random_state=args.seed,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
        ),
        "hist_gradient_boosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=args.seed,
            )
        ),
    }

    all_rows = []

    # Mean baseline
    mean_pred = np.tile(y_train.mean(axis=0), (len(test), 1))
    all_rows.extend(metric_rows(
        y_test,
        mean_pred,
        "train_mean",
        "baseline",
        args.split_mode,
        len(train),
        len(test),
    ))

    # Previous sequence persistence
    pers_true, pers_pred = previous_sequence_persistence(df, test)
    all_rows.extend(metric_rows(
        pers_true,
        pers_pred,
        "previous_sequence_persistence",
        "baseline",
        args.split_mode,
        len(train),
        len(pers_true),
    ))

    feature_count_rows = []

    for feature_set, cols in feature_sets.items():
        feature_count_rows.append({
            "feature_set": feature_set,
            "n_features": len(cols),
        })

        if not cols:
            print(f"Skipping {feature_set}: no columns found")
            continue

        X_train = train[cols].copy()
        X_test = test[cols].copy()

        print("\n" + "-" * 90)
        print("Feature set:", feature_set)
        print("Columns:", len(cols))

        for model_name, model in model_specs.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                all_rows.extend(metric_rows(
                    y_test,
                    pred,
                    model_name,
                    feature_set,
                    args.split_mode,
                    len(train),
                    len(test),
                ))

            except Exception as e:
                print(f"FAILED {feature_set} / {model_name}: {e}")

    results = pd.DataFrame(all_rows)
    results["seed"] = args.seed
    results["train_frac"] = args.train_frac

    results = results.sort_values(["target", "R2"], ascending=[True, False])

    out_csv = out_dir / f"traqid_T7_engineered_baselines_{args.split_mode}.csv"
    out_md = out_dir / f"traqid_T7_engineered_baselines_{args.split_mode}.md"

    results.to_csv(out_csv, index=False)
    out_md.write_text(results.to_markdown(index=False), encoding="utf-8")

    feature_counts = pd.DataFrame(feature_count_rows)
    feature_counts.to_csv(out_dir / f"traqid_T7_feature_counts_{args.split_mode}.csv", index=False)

    avg = results[results["target"] == "Average"].sort_values("R2", ascending=False)

    print("\n" + "=" * 90)
    print("TRAQID T7 ENGINEERED BASELINES COMPLETE")
    print("=" * 90)
    print("Table:", table_path)
    print("Rows:", len(df))
    print("Split mode:", args.split_mode)
    print("Train rows:", len(train))
    print("Test rows:", len(test))
    print("Output:", out_dir)
    print()
    print("Feature counts:")
    print(feature_counts.to_string(index=False))
    print()
    print("Average target ranking:")
    print(avg.to_string(index=False))
    print()
    print("Saved:", out_csv)
    print("Saved:", out_md)


if __name__ == "__main__":
    main()