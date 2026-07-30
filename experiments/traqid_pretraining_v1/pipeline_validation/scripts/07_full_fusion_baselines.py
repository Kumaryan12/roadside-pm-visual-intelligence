from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

YOLO_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_vehicle_features_front.csv")
EMB_DIR = Path("experiments/traqid_pretraining_v1/embeddings")
OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "PM2.5"
EMB_FILE = EMB_DIR / "traqid_mobilenetv2_torch_front_rear_concat_embeddings.npy"

RESULTS_CKPT = OUTDIR / "full_fusion_baseline_results_checkpoint.csv"
FINAL_OUT = OUTDIR / "full_fusion_baseline_results.csv"

vehicle_features = [
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

tabular_numeric = ["Temperature", "Humidity", "hour_num"]
tabular_categorical = ["Season", "Day_or_Night"]


def get_metrics(y_true, y_pred):
    sp = np.nan if np.std(y_pred) == 0 else spearmanr(y_true, y_pred).correlation
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Spearman": sp,
    }


def make_random_split(df):
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
    return train.index.values, val.index.values, test.index.values


def make_chrono_split(df):
    train_idx = df.index[df["split_date_chrono"] == "train"].values
    val_idx = df.index[df["split_date_chrono"] == "val"].values
    test_idx = df.index[df["split_date_chrono"] == "test"].values
    return train_idx, val_idx, test_idx


def build_tabular_dummies(df_train, df_test):
    combined = pd.concat(
        [
            df_train[tabular_numeric + tabular_categorical],
            df_test[tabular_numeric + tabular_categorical],
        ],
        axis=0,
    )
    combined = pd.get_dummies(
        combined,
        columns=tabular_categorical,
        drop_first=False,
    )

    train_tab = combined.iloc[: len(df_train)].reset_index(drop=True)
    test_tab = combined.iloc[len(df_train):].reset_index(drop=True)

    return train_tab.values.astype(float), test_tab.values.astype(float)


def make_features(df, X_emb_raw, train_idx, test_idx, feature_groups, pca_dim):
    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    parts_train = []
    parts_test = []

    if "vehicle" in feature_groups:
        parts_train.append(df_train[vehicle_features].values.astype(float))
        parts_test.append(df_test[vehicle_features].values.astype(float))

    if "tabular" in feature_groups:
        train_tab, test_tab = build_tabular_dummies(df_train, df_test)
        parts_train.append(train_tab)
        parts_test.append(test_tab)

    if "embedding" in feature_groups:
        n_components = min(pca_dim, X_emb_raw.shape[1], len(train_idx) - 1)

        emb_scaler = StandardScaler()
        emb_pca = PCA(n_components=n_components, random_state=42)

        emb_train_scaled = emb_scaler.fit_transform(X_emb_raw[train_idx])
        emb_train_pca = emb_pca.fit_transform(emb_train_scaled)

        emb_test_scaled = emb_scaler.transform(X_emb_raw[test_idx])
        emb_test_pca = emb_pca.transform(emb_test_scaled)

        parts_train.append(emb_train_pca.astype(float))
        parts_test.append(emb_test_pca.astype(float))

    if not parts_train:
        return None, None

    X_train = np.hstack(parts_train)
    X_test = np.hstack(parts_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def make_model(model_name, fast=False):
    if model_name == "ridge":
        return Ridge(alpha=10.0)

    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=80 if fast else 200,
            max_depth=14 if fast else 18,
            min_samples_leaf=8 if fast else 5,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=120 if fast else 300,
            learning_rate=0.05 if fast else 0.04,
            max_leaf_nodes=31,
            random_state=42,
        )

    raise ValueError(model_name)


def load_done_keys():
    if not RESULTS_CKPT.exists():
        return set(), []

    old = pd.read_csv(RESULTS_CKPT)
    keys = set(zip(old["split"], old["model"]))
    return keys, old.to_dict("records")


def save_results(records):
    res = pd.DataFrame(records)
    res = res.sort_values(["split", "RMSE"])
    res.to_csv(RESULTS_CKPT, index=False)
    res.to_csv(FINAL_OUT, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use faster, lighter models.")
    parser.add_argument("--pca-dim", type=int, default=64, help="PCA dimensions for embeddings.")
    parser.add_argument("--split", type=str, default="both", choices=["both", "random", "chrono"])
    parser.add_argument("--resume", action="store_true", help="Skip completed models from checkpoint.")
    args = parser.parse_args()

    print("Loading YOLO features...")
    df = pd.read_csv(YOLO_CSV)
    df = df[df["yolo_status"] == "success"].copy().reset_index(drop=True)
    df["created_at_parsed"] = pd.to_datetime(df["created_at"])
    df["hour_num"] = df["created_at_parsed"].dt.hour

    for c in vehicle_features:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    print("Rows:", len(df))

    print("Loading embeddings...")
    X_emb_all = np.load(EMB_FILE)
    X_emb_raw = X_emb_all[df["row_id"].values]
    y = df[TARGET].values.astype(float)

    splits = {}
    if args.split in ["both", "random"]:
        splits["random_stratified"] = make_random_split(df)
    if args.split in ["both", "chrono"]:
        splits["chronological"] = make_chrono_split(df)

    experiments = [
        ("mean_baseline", [], None),
        ("vehicle_only_ridge", ["vehicle"], "ridge"),
        ("vehicle_only_hgb", ["vehicle"], "hgb"),
        ("tabular_only_hgb", ["tabular"], "hgb"),
        ("vehicle_plus_tabular_hgb", ["vehicle", "tabular"], "hgb"),
        ("embedding_only_ridge", ["embedding"], "ridge"),
        ("embedding_plus_vehicle_ridge", ["embedding", "vehicle"], "ridge"),
        ("embedding_plus_tabular_ridge", ["embedding", "tabular"], "ridge"),
        ("full_fusion_ridge", ["vehicle", "tabular", "embedding"], "ridge"),
    ]

    # RF models are useful but slow, so keep them only in non-fast mode.
    if not args.fast:
        experiments += [
            ("vehicle_only_rf", ["vehicle"], "rf"),
            ("vehicle_plus_tabular_rf", ["vehicle", "tabular"], "rf"),
            ("embedding_plus_vehicle_rf", ["embedding", "vehicle"], "rf"),
            ("full_fusion_rf", ["vehicle", "tabular", "embedding"], "rf"),
        ]

    done_keys, records = load_done_keys() if args.resume else (set(), [])

    print("\nCheckpoint:", RESULTS_CKPT)
    print("Resume:", args.resume)
    print("Already done:", len(done_keys))
    print("Fast mode:", args.fast)
    print("PCA dim:", args.pca_dim)

    for split_name, (train_idx, val_idx, test_idx) in splits.items():
        print("\n" + "=" * 100)
        print("Split:", split_name)
        print("Rows:", {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)})

        for exp_name, feature_groups, model_name in experiments:
            key = (split_name, exp_name)

            if args.resume and key in done_keys:
                print(f"SKIP completed: {split_name} | {exp_name}")
                continue

            print("\n" + "-" * 100)
            print(f"START: {split_name} | {exp_name} | features={feature_groups}")
            start = time.time()

            if exp_name == "mean_baseline":
                pred = np.full(len(test_idx), np.mean(y[train_idx]), dtype=float)
            else:
                X_train, X_test = make_features(
                    df=df,
                    X_emb_raw=X_emb_raw,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    feature_groups=feature_groups,
                    pca_dim=args.pca_dim,
                )

                print("Feature matrix:", X_train.shape, X_test.shape)

                model = make_model(model_name, fast=args.fast)
                model.fit(X_train, y[train_idx])
                pred = model.predict(X_test)

            elapsed = time.time() - start

            row = {
                "split": split_name,
                "model": exp_name,
                "features": "+".join(feature_groups) if feature_groups else "none",
                "target": TARGET,
                "embedding": f"mobilenetv2_front_rear_concat_pca{args.pca_dim}" if "embedding" in feature_groups else "none",
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "elapsed_sec": round(elapsed, 2),
                "fast_mode": args.fast,
            }
            row.update(get_metrics(y[test_idx], pred))

            records.append(row)
            save_results(records)

            print(f"DONE: {split_name} | {exp_name} in {elapsed:.1f}s")
            print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman"]})
            print("Checkpoint saved.")

    final = pd.DataFrame(records).sort_values(["split", "RMSE"])
    print("\nFinal results:")
    print(final.to_string(index=False))
    print("\nSaved:", FINAL_OUT)


if __name__ == "__main__":
    main()