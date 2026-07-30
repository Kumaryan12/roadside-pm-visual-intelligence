import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["PM2.5", "PM10"]


def metric_rows(y_true, y_pred, model_name, feature_set, train_rows, test_rows):
    rows = []
    target_metrics = []

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    for i, target in enumerate(TARGETS):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae = float(np.mean(np.abs(yt - yp)))
        r2 = float(r2_score(yt, yp))

        row = {
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
        target_metrics.append(row)

    rows.append({
        "feature_set": feature_set,
        "model": model_name,
        "target": "Average",
        "R2": float(np.mean([r["R2"] for r in target_metrics])),
        "RMSE": float(np.mean([r["RMSE"] for r in target_metrics])),
        "MAE": float(np.mean([r["MAE"] for r in target_metrics])),
        "train_rows": train_rows,
        "test_rows": test_rows,
    })

    return rows


def select_existing(df, prefixes=None, exact=None, contains=None, exclude=None):
    prefixes = prefixes or []
    exact = exact or []
    contains = contains or []
    exclude = exclude or []

    cols = []

    for c in df.columns:
        ok = False

        if c in exact:
            ok = True

        if any(c.startswith(p) for p in prefixes):
            ok = True

        if any(k in c for k in contains):
            ok = True

        if any(k in c for k in exclude):
            ok = False

        if ok:
            cols.append(c)

    numeric_cols = []
    for c in cols:
        if c in TARGETS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool:
            numeric_cols.append(c)

    return sorted(set(numeric_cols))


def build_persistence_predictions(full_df, test_df):
    """
    Predict PM_t using PM_(t-1), based on chronological sample_index order.
    This remains valid even when model split is random.
    """

    full_df = full_df.sort_values("sample_index").reset_index(drop=True)
    full_y = full_df[TARGETS].values

    sample_to_pos = {
        int(sample_idx): pos
        for pos, sample_idx in enumerate(full_df["sample_index"].values)
    }

    pers_true = []
    pers_pred = []

    for sample_idx in test_df["sample_index"].values:
        pos = sample_to_pos.get(int(sample_idx), None)

        if pos is None or pos == 0:
            continue

        pers_true.append(full_y[pos])
        pers_pred.append(full_y[pos - 1])

    return np.asarray(pers_true), np.asarray(pers_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument(
        "--split-mode",
        choices=["chronological", "random"],
        default="chronological",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    table_path = Path(args.table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table_path)
    df = df.sort_values("sample_index").reset_index(drop=True)
    df = df.dropna(subset=TARGETS).reset_index(drop=True)

    n = len(df)
    split_idx = int(n * args.train_frac)

    if args.split_mode == "chronological":
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()

    elif args.split_mode == "random":
        shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        train = shuffled.iloc[:split_idx].copy()
        test = shuffled.iloc[split_idx:].copy()

    else:
        raise ValueError(f"Unsupported split mode: {args.split_mode}")

    y_train = train[TARGETS].values
    y_test = test[TARGETS].values

    sensor_cols = select_existing(
        df,
        exact=[
            "temperature",
            "humidity",
            "CO_ppb",
            "NO2_ppb",
            "SO2_ppb",
            "O3_ppb",
            "lat",
            "long",
            "temp",
            "rh",
            "value.co_ppb",
            "value.no2_ppb",
            "value.so2_ppb",
            "value.o3_ppb_compensated",
            "value.lat",
            "value.long",
        ],
        contains=["rh_fraction", "dry_air_fraction"],
    )

    vehicle_cols = select_existing(
        df,
        prefixes=["idd_", "vehicle_"],
        exclude=["resnet50_"],
    )

    road_cols = select_existing(
        df,
        prefixes=["road_"],
        contains=[
            "brown",
            "gray_dry",
            "texture",
            "haze",
            "laplacian",
            "glare",
            "shadow",
        ],
        exclude=["resnet50_"],
    )

    osm_cols = select_existing(
        df,
        prefixes=["osm_"],
        contains=[
            "_count_250m",
            "road_length",
            "intensity_250m",
            "road_fraction",
        ],
        exclude=["resnet50_"],
    )

    road_area_cols = select_existing(
        df,
        contains=[
            "road_area_m2",
            "visible_road_area",
            "occlusion_adjusted",
            "vehicle_occlusion_fraction",
            "occlusion_added",
        ],
        exclude=["resnet50_"],
    )

    resnet_cols = select_existing(df, prefixes=["resnet50_"])

    engineered_cols = sorted(
        set(sensor_cols + vehicle_cols + road_cols + osm_cols + road_area_cols)
    )

    feature_sets = {
        "sensor_context_only": sensor_cols,
        "vehicle_IDD_only": vehicle_cols,
        "road_condition_only": sorted(set(road_cols + road_area_cols)),
        "OSM_only": osm_cols,
        "engineered_all_without_resnet": engineered_cols,
        "resnet50_only": resnet_cols,
        "engineered_all_plus_resnet50": sorted(set(engineered_cols + resnet_cols)),
    }

    all_rows = []

    # ------------------------------------------------------------------
    # Baseline: train mean
    # ------------------------------------------------------------------
    pred_mean = np.tile(y_train.mean(axis=0), (len(test), 1))

    all_rows.extend(
        metric_rows(
            y_test,
            pred_mean,
            "train_mean",
            "baseline",
            len(train),
            len(test),
        )
    )

    # ------------------------------------------------------------------
    # Baseline: previous-step persistence
    # ------------------------------------------------------------------
    pers_true, pers_pred = build_persistence_predictions(df, test)

    all_rows.extend(
        metric_rows(
            pers_true,
            pers_pred,
            "previous_step_persistence",
            "baseline",
            len(train),
            len(pers_true),
        )
    )

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
    }

    for feature_set, cols in feature_sets.items():
        if not cols:
            print(f"Skipping {feature_set}: no columns found")
            continue

        X_train = train[cols].copy()
        X_test = test[cols].copy()

        print(f"\nFeature set: {feature_set}")
        print(f"Columns: {len(cols)}")

        for model_name, model in model_specs.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                all_rows.extend(
                    metric_rows(
                        y_test,
                        pred,
                        model_name,
                        feature_set,
                        len(train),
                        len(test),
                    )
                )

            except Exception as e:
                print(f"Failed {feature_set} / {model_name}: {e}")

    results = pd.DataFrame(all_rows)

    results["split_mode"] = args.split_mode
    results["seed"] = args.seed
    results["train_frac"] = args.train_frac

    results = results.sort_values(["target", "R2"], ascending=[True, False])

    out_csv = out_dir / "mumma_pm25_ablation_metrics.csv"
    out_md = out_dir / "mumma_pm25_ablation_metrics.md"

    results.to_csv(out_csv, index=False)
    out_md.write_text(results.to_markdown(index=False), encoding="utf-8")

    avg = results[results["target"] == "Average"].sort_values("R2", ascending=False)

    feature_counts = pd.DataFrame(
        [{"feature_set": k, "num_features": len(v)} for k, v in feature_sets.items()]
    )

    feature_counts.to_csv(out_dir / "mumma_pm25_feature_set_counts.csv", index=False)

    print("\n" + "=" * 90)
    print("MUMMA PM2.5/PM10 ABLATION COMPLETE")
    print("=" * 90)
    print("Table:", table_path)
    print("Rows:", n)
    print("Split mode:", args.split_mode)
    print("Seed:", args.seed)
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