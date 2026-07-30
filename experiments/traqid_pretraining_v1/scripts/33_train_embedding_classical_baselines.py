from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLS = ["target_PM2.5", "target_PM10", "target_aqi"]
TARGET_NAMES = ["PM2.5", "PM10", "AQI"]

NUM_COLS = ["Temperature", "Humidity", "hour_sin", "hour_cos"]
CAT_COLS = ["Season", "Day_or_Night"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_seq_row_ids(s: str) -> list[int]:
    return [int(x) for x in str(s).split("|") if str(x).strip()]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["target_time"], errors="coerce")
    hour = dt.dt.hour.fillna(0).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return df


def attach_context(seq_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["row_id", "Temperature", "Humidity", "Season", "Day_or_Night"]
    missing = [c for c in cols if c not in base_df.columns]
    if missing:
        raise ValueError(f"Base manifest missing columns: {missing}")

    meta = base_df[cols].rename(columns={"row_id": "target_row_id"})
    out = seq_df.merge(meta, on="target_row_id", how="left")
    out = add_time_features(out)
    return out


def build_sequence_embeddings(seq_df: pd.DataFrame, frame_features: np.ndarray, pooling: str) -> np.ndarray:
    rows = []
    n = len(seq_df)

    log(f"Building sequence embeddings with pooling={pooling} for {n} sequences...")

    for i, s in enumerate(seq_df["seq_row_ids"]):
        ids = parse_seq_row_ids(s)
        x = frame_features[ids]

        if pooling == "mean":
            pooled = x.mean(axis=0)
        elif pooling == "last":
            pooled = x[-1]
        elif pooling == "mean_std":
            pooled = np.concatenate([x.mean(axis=0), x.std(axis=0)], axis=0)
        else:
            raise ValueError("pooling must be one of: mean, last, mean_std")

        rows.append(pooled)

        if (i + 1) % 3000 == 0 or (i + 1) == n:
            log(f"  pooled {i + 1}/{n} sequences")

    out = np.asarray(rows, dtype=np.float32)
    log(f"Built pooled embedding matrix: {out.shape}")
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {}

    for i, name in enumerate(TARGET_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        finite = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[finite]
        yp = yp[finite]

        mae = mean_absolute_error(yt, yp)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)

        if np.std(yt) == 0 or np.std(yp) == 0:
            pear = float("nan")
            spear = float("nan")
        else:
            pear = pearsonr(yt, yp).statistic
            spear = spearmanr(yt, yp).statistic

        metrics[name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "Pearson": float(pear),
            "Spearman": float(spear),
            "finite_fraction": float(finite.mean()),
        }

    metrics["average"] = {
        "R2": float(np.nanmean([metrics[t]["R2"] for t in TARGET_NAMES])),
        "RMSE": float(np.nanmean([metrics[t]["RMSE"] for t in TARGET_NAMES])),
    }

    return metrics


def make_model(args):
    if args.model == "ridge":
        return Ridge(alpha=args.ridge_alpha)

    if args.model == "rf":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )

    if args.model == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )

    if args.model == "hgb":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=args.hgb_iter,
                learning_rate=0.05,
                l2_regularization=1e-3,
                random_state=args.seed,
                verbose=args.verbose,
            )
        )

    if args.model == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=args.seed,
            verbose=bool(args.verbose),
        )

    raise ValueError(args.model)


def maybe_pca(X_train, X_val, X_test, n_components: int, seed: int):
    if n_components <= 0:
        return X_train, X_val, X_test, None

    log(f"Applying PCA: {X_train.shape[1]} -> {n_components}")
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    X_train_p = pca.fit_transform(X_train)
    X_val_p = pca.transform(X_val)
    X_test_p = pca.transform(X_test)

    evr = float(pca.explained_variance_ratio_.sum())
    log(f"PCA explained variance ratio: {evr:.4f}")

    return X_train_p, X_val_p, X_test_p, {
        "n_components": int(n_components),
        "explained_variance_ratio_sum": evr,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/paper_style_sequences/traqid_paper_style_T7_front_sequence_manifest_time_balanced_purged.csv",
    )
    parser.add_argument(
        "--base-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )
    parser.add_argument(
        "--features",
        default="experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/traqid_paper_resnet50_front_rear_concat_gap_features.npy",
    )
    parser.add_argument("--split-col", default="split_time_balanced_purged")
    parser.add_argument("--train-name", default="train")
    parser.add_argument("--val-name", default="val")
    parser.add_argument("--test-name", default="test")

    parser.add_argument("--model", default="rf", choices=["ridge", "rf", "extratrees", "hgb", "mlp"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "last", "mean_std"])
    parser.add_argument("--include-tabular", action="store_true")
    parser.add_argument("--pca-components", type=int, default=256)

    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--hgb-iter", type=int, default=200)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/embedding_classical_baselines",
    )

    args = parser.parse_args()

    t0 = time.time()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 90)
    log("EMBEDDING CLASSICAL BASELINE")
    log("=" * 90)
    log(f"Model: {args.model}")
    log(f"Pooling: {args.pooling}")
    log(f"Include tabular: {args.include_tabular}")
    log(f"PCA components: {args.pca_components}")
    log(f"n_estimators: {args.n_estimators}")
    log(f"max_features: {args.max_features}")
    log(f"min_samples_leaf: {args.min_samples_leaf}")

    log("Loading manifests...")
    seq_df = pd.read_csv(args.sequence_manifest)
    base_df = pd.read_csv(args.base_manifest)

    log("Loading frame features...")
    frame_features = np.load(args.features, mmap_mode="r")
    log(f"Feature shape: {frame_features.shape}")

    log("Attaching context...")
    df = attach_context(seq_df, base_df)
    df = df[df[args.split_col].isin([args.train_name, args.val_name, args.test_name])].copy()

    train_df = df[df[args.split_col] == args.train_name].copy()
    val_df = df[df[args.split_col] == args.val_name].copy()
    test_df = df[df[args.split_col] == args.test_name].copy()

    log(f"Train/val/test: {len(train_df)} / {len(val_df)} / {len(test_df)}")

    X_train_img = build_sequence_embeddings(train_df, frame_features, args.pooling)
    X_val_img = build_sequence_embeddings(val_df, frame_features, args.pooling)
    X_test_img = build_sequence_embeddings(test_df, frame_features, args.pooling)

    y_train = train_df[TARGET_COLS].to_numpy(dtype=float)
    y_val = val_df[TARGET_COLS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLS].to_numpy(dtype=float)

    log("Scaling image embeddings...")
    scaler = StandardScaler()
    X_train_img = scaler.fit_transform(X_train_img)
    X_val_img = scaler.transform(X_val_img)
    X_test_img = scaler.transform(X_test_img)

    pca_info = None
    X_train_img, X_val_img, X_test_img, pca_info = maybe_pca(
        X_train_img,
        X_val_img,
        X_test_img,
        n_components=args.pca_components,
        seed=args.seed,
    )

    if args.include_tabular:
        log("Preparing tabular features...")
        num_scaler = StandardScaler()
        X_train_num = num_scaler.fit_transform(train_df[NUM_COLS])
        X_val_num = num_scaler.transform(val_df[NUM_COLS])
        X_test_num = num_scaler.transform(test_df[NUM_COLS])

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_train_cat = enc.fit_transform(train_df[CAT_COLS].astype(str))
        X_val_cat = enc.transform(val_df[CAT_COLS].astype(str))
        X_test_cat = enc.transform(test_df[CAT_COLS].astype(str))

        X_train = np.concatenate([X_train_img, X_train_num, X_train_cat], axis=1)
        X_val = np.concatenate([X_val_img, X_val_num, X_val_cat], axis=1)
        X_test = np.concatenate([X_test_img, X_test_num, X_test_cat], axis=1)
    else:
        X_train, X_val, X_test = X_train_img, X_val_img, X_test_img

    log(f"Final input dim: {X_train.shape[1]}")
    log(f"Training model {args.model}...")
    model = make_model(args)
    model.fit(X_train, y_train)

    log("Predicting val...")
    val_pred = model.predict(X_val)
    log("Predicting test...")
    test_pred = model.predict(X_test)

    log("Computing metrics...")
    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    variant = "embedding_plus_tabular" if args.include_tabular else "embedding_only"
    pca_tag = f"pca{args.pca_components}" if args.pca_components > 0 else "nopca"
    run_name = f"{args.model}_{variant}_{args.pooling}_{pca_tag}_{Path(args.sequence_manifest).stem}_{args.split_col}"

    out_dir = report_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": args.model,
        "variant": variant,
        "pooling": args.pooling,
        "include_tabular": args.include_tabular,
        "pca": pca_info,
        "sequence_manifest": args.sequence_manifest,
        "base_manifest": args.base_manifest,
        "features": args.features,
        "split": {
            "split_col": args.split_col,
            "train_name": args.train_name,
            "val_name": args.val_name,
            "test_name": args.test_name,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
        },
        "input_dim": int(X_train.shape[1]),
        "model_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
        },
        "val": val_metrics,
        "test": test_metrics,
        "runtime_sec": float(time.time() - t0),
    }

    (out_dir / "metrics_embedding_classical.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_rows = []
    for split_name, part_df, y_true, y_pred in [
        ("val", val_df, y_val, val_pred),
        ("test", test_df, y_test, test_pred),
    ]:
        for sid, yt, yp in zip(part_df["sequence_id"], y_true, y_pred):
            pred_rows.append({
                "split": split_name,
                "sequence_id": int(sid),
                "actual_PM2.5": float(yt[0]),
                "predicted_PM2.5": float(yp[0]),
                "actual_PM10": float(yt[1]),
                "predicted_PM10": float(yp[1]),
                "actual_aqi": float(yt[2]),
                "predicted_aqi": float(yp[2]),
            })

    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions_embedding_classical.csv", index=False)

    log("VAL:")
    print(json.dumps(val_metrics, indent=2), flush=True)
    log("TEST:")
    print(json.dumps(test_metrics, indent=2), flush=True)

    log(f"Saved: {out_dir}")
    log(f"Runtime sec: {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()