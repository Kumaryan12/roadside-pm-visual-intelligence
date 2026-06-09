from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["PM2.5", "PM10", "aqi"]


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def safe_corr(y_true, y_pred, method: str) -> float:
    s1 = pd.Series(y_true)
    s2 = pd.Series(y_pred)

    if s1.nunique() < 2 or s2.nunique() < 2:
        return np.nan

    return float(s1.corr(s2, method=method))


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 0)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": safe_corr(y_true, y_pred, "pearson"),
        "Spearman": safe_corr(y_true, y_pred, "spearman"),
    }


def save_scatter(y_true, y_pred, title: str, out_path: Path) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)

    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def load_embeddings(embeddings_dir: Path, prefix: str, kind: str):
    paths = {
        "front": embeddings_dir / f"{prefix}_front_embeddings.npy",
        "rear": embeddings_dir / f"{prefix}_rear_embeddings.npy",
        "mean": embeddings_dir / f"{prefix}_front_rear_mean_embeddings.npy",
        "concat": embeddings_dir / f"{prefix}_front_rear_concat_embeddings.npy",
    }

    if kind not in paths:
        raise ValueError(f"Unknown embedding kind: {kind}")

    path = paths[kind]

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    arr = np.load(path, mmap_mode="r")
    return arr, path


def prepare_target(y_raw: np.ndarray, target_mode: str):
    if target_mode == "raw":
        return y_raw.astype(float)

    if target_mode == "log1p":
        return np.log1p(y_raw.astype(float))

    raise ValueError(f"Unknown target_mode: {target_mode}")


def invert_target(y_pred: np.ndarray, target_mode: str):
    if target_mode == "raw":
        return y_pred

    if target_mode == "log1p":
        return np.expm1(y_pred)

    raise ValueError(f"Unknown target_mode: {target_mode}")


def fit_predict_ridge(
    X_train,
    y_train,
    X_val,
    X_test,
    *,
    alpha: float,
    pca_components: int | None,
    random_state: int,
):
    steps = [
        ("scaler", StandardScaler()),
    ]

    if pca_components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    svd_solver="randomized",
                    random_state=random_state,
                ),
            )
        )

    steps.append(
        (
            "ridge",
            Ridge(
                alpha=alpha,
                solver="lsqr",
                random_state=random_state,
            ),
        )
    )

    model = Pipeline(steps)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    return model, val_pred, test_pred


def fit_predict_hgb(
    X_train,
    y_train,
    X_val,
    X_test,
    *,
    pca_components: int,
    random_state: int,
):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=pca_components,
                    svd_solver="randomized",
                    random_state=random_state,
                ),
            ),
            (
                "hgb",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    return model, val_pred, test_pred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index-csv",
        default="experiments/traqid_pretraining_v1/embeddings/traqid_mobilenetv2_torch_embedding_index.csv",
    )

    parser.add_argument(
        "--embeddings-dir",
        default="experiments/traqid_pretraining_v1/embeddings",
    )

    parser.add_argument(
        "--prefix",
        default="traqid_mobilenetv2_torch",
    )

    parser.add_argument(
        "--embedding-kinds",
        default="mean",
        help="Comma-separated: front,rear,mean,concat",
    )

    parser.add_argument(
        "--target-modes",
        default="raw,log1p",
        help="Comma-separated: raw,log1p",
    )

    parser.add_argument(
        "--pca-components",
        default="64,128,256",
        help="Comma-separated PCA components. Use none to skip PCA only.",
    )

    parser.add_argument(
        "--alphas",
        default="1,10,100,1000",
        help="Comma-separated Ridge alpha values.",
    )

    parser.add_argument(
        "--include-no-pca",
        action="store_true",
        help="Also train Ridge directly on full embeddings without PCA.",
    )

    parser.add_argument(
        "--include-hgb",
        action="store_true",
        help="Also train HistGradientBoosting on PCA embeddings. Slower.",
    )

    parser.add_argument(
        "--require-env-plausible",
        action="store_true",
        help="Filter to plausible Temperature/Humidity rows for fair comparison with tabular baseline.",
    )

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/embedding_baselines",
    )

    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures/embedding_baselines",
    )

    parser.add_argument(
        "--model-dir",
        default="experiments/traqid_pretraining_v1/models/embedding_baselines",
    )

    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    index_csv = Path(args.index_csv)
    embeddings_dir = Path(args.embeddings_dir)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    model_dir = Path(args.model_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_csv)

    required_cols = [
        "row_id",
        "image_id",
        "created_at",
        "date",
        "split_date_chrono",
        "PM2.5",
        "PM10",
        "aqi",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Index CSV missing columns: {missing}")

    if args.require_env_plausible:
        df = df[
            df["Temperature"].between(-10, 60)
            & df["Humidity"].between(0, 100)
        ].copy()

    df = df.reset_index(drop=False).rename(columns={"index": "embedding_row_index"})

    train_mask = df["split_date_chrono"] == "train"
    val_mask = df["split_date_chrono"] == "val"
    test_mask = df["split_date_chrono"] == "test"

    train_idx = df.loc[train_mask, "embedding_row_index"].to_numpy()
    val_idx = df.loc[val_mask, "embedding_row_index"].to_numpy()
    test_idx = df.loc[test_mask, "embedding_row_index"].to_numpy()

    embedding_kinds = [x.strip() for x in args.embedding_kinds.split(",") if x.strip()]
    target_modes = [x.strip() for x in args.target_modes.split(",") if x.strip()]

    pca_components = []
    for x in args.pca_components.split(","):
        x = x.strip()
        if not x:
            continue
        if x.lower() == "none":
            continue
        pca_components.append(int(x))

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    print("=" * 90)
    print("TRAQID EMBEDDING BASELINES")
    print("=" * 90)
    print("Index CSV:", index_csv)
    print("Rows:", len(df))
    print("Embedding kinds:", embedding_kinds)
    print("Target modes:", target_modes)
    print("PCA components:", pca_components)
    print("Ridge alphas:", alphas)
    print("Include no PCA:", args.include_no_pca)
    print("Include HGB:", args.include_hgb)
    print("Require env plausible:", args.require_env_plausible)

    print("\nSplit counts:")
    print(df["split_date_chrono"].value_counts().to_string())

    results = []
    prediction_frames = []

    # Mean baselines first.
    for target in TARGETS:
        y_train_raw = df.loc[train_mask, target].astype(float).to_numpy()
        y_val_raw = df.loc[val_mask, target].astype(float).to_numpy()
        y_test_raw = df.loc[test_mask, target].astype(float).to_numpy()

        for target_mode in target_modes:
            y_train_model = prepare_target(y_train_raw, target_mode)

            dummy = DummyRegressor(strategy="mean")
            dummy.fit(np.zeros((len(y_train_model), 1)), y_train_model)

            val_pred_model = dummy.predict(np.zeros((len(y_val_raw), 1)))
            test_pred_model = dummy.predict(np.zeros((len(y_test_raw), 1)))

            val_pred = invert_target(val_pred_model, target_mode)
            test_pred = invert_target(test_pred_model, target_mode)

            val_metrics = evaluate(y_val_raw, val_pred)
            test_metrics = evaluate(y_test_raw, test_pred)

            row = {
                "embedding_kind": "none",
                "target": target,
                "target_mode": target_mode,
                "model": "mean_baseline",
                "pca_components": np.nan,
                "alpha": np.nan,
            }

            for k, v in val_metrics.items():
                row[f"val_{k}"] = v
            for k, v in test_metrics.items():
                row[f"test_{k}"] = v

            results.append(row)

    # Embedding models.
    for embedding_kind in embedding_kinds:
        X_all, emb_path = load_embeddings(
            embeddings_dir=embeddings_dir,
            prefix=args.prefix,
            kind=embedding_kind,
        )

        if X_all.shape[0] != len(pd.read_csv(index_csv)):
            raise ValueError(
                f"Embedding rows do not match original index rows. "
                f"Embedding={X_all.shape[0]}, index={len(pd.read_csv(index_csv))}"
            )

        X_train = np.asarray(X_all[train_idx], dtype=np.float32)
        X_val = np.asarray(X_all[val_idx], dtype=np.float32)
        X_test = np.asarray(X_all[test_idx], dtype=np.float32)

        print("\n" + "=" * 90)
        print(f"Embedding kind: {embedding_kind}")
        print("Embedding file:", emb_path)
        print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)
        print("=" * 90)

        for target in TARGETS:
            y_train_raw = df.loc[train_mask, target].astype(float).to_numpy()
            y_val_raw = df.loc[val_mask, target].astype(float).to_numpy()
            y_test_raw = df.loc[test_mask, target].astype(float).to_numpy()

            for target_mode in target_modes:
                y_train = prepare_target(y_train_raw, target_mode)

                model_specs = []

                if args.include_no_pca:
                    for alpha in alphas:
                        model_specs.append(
                            {
                                "model_type": "ridge",
                                "model_name": "ridge_no_pca",
                                "pca": None,
                                "alpha": alpha,
                            }
                        )

                for pca_n in pca_components:
                    for alpha in alphas:
                        model_specs.append(
                            {
                                "model_type": "ridge",
                                "model_name": f"ridge_pca{pca_n}",
                                "pca": pca_n,
                                "alpha": alpha,
                            }
                        )

                    if args.include_hgb:
                        model_specs.append(
                            {
                                "model_type": "hgb",
                                "model_name": f"hgb_pca{pca_n}",
                                "pca": pca_n,
                                "alpha": np.nan,
                            }
                        )

                for spec in model_specs:
                    model_type = spec["model_type"]
                    model_name = spec["model_name"]
                    pca_n = spec["pca"]
                    alpha = spec["alpha"]

                    if model_type == "ridge":
                        model, val_pred_model, test_pred_model = fit_predict_ridge(
                            X_train,
                            y_train,
                            X_val,
                            X_test,
                            alpha=alpha,
                            pca_components=pca_n,
                            random_state=args.random_state,
                        )
                    elif model_type == "hgb":
                        model, val_pred_model, test_pred_model = fit_predict_hgb(
                            X_train,
                            y_train,
                            X_val,
                            X_test,
                            pca_components=pca_n,
                            random_state=args.random_state,
                        )
                    else:
                        raise ValueError(model_type)

                    val_pred = invert_target(val_pred_model, target_mode)
                    test_pred = invert_target(test_pred_model, target_mode)

                    val_pred = np.maximum(val_pred, 0)
                    test_pred = np.maximum(test_pred, 0)

                    val_metrics = evaluate(y_val_raw, val_pred)
                    test_metrics = evaluate(y_test_raw, test_pred)

                    result_row = {
                        "embedding_kind": embedding_kind,
                        "target": target,
                        "target_mode": target_mode,
                        "model": model_name,
                        "pca_components": pca_n if pca_n is not None else np.nan,
                        "alpha": alpha,
                    }

                    for k, v in val_metrics.items():
                        result_row[f"val_{k}"] = v

                    for k, v in test_metrics.items():
                        result_row[f"test_{k}"] = v

                    results.append(result_row)

                    print(
                        f"{embedding_kind:6s} | {target:5s} | {target_mode:5s} | "
                        f"{model_name:14s} | alpha={alpha} | "
                        f"VAL RMSE={val_metrics['RMSE']:.3f}, R2={val_metrics['R2']:.3f}, Spearman={val_metrics['Spearman']:.3f} | "
                        f"TEST RMSE={test_metrics['RMSE']:.3f}, R2={test_metrics['R2']:.3f}, Spearman={test_metrics['Spearman']:.3f}"
                    )

                    # Save only selected prediction rows for every model; useful for later diagnostics.
                    for split_name, split_df, y_true, y_pred in [
                        ("val", df.loc[val_mask], y_val_raw, val_pred),
                        ("test", df.loc[test_mask], y_test_raw, test_pred),
                    ]:
                        temp = pd.DataFrame(
                            {
                                "row_id": split_df["row_id"].values,
                                "image_id": split_df["image_id"].values,
                                "created_at": split_df["created_at"].values,
                                "date": split_df["date"].values,
                                "split": split_name,
                                "embedding_kind": embedding_kind,
                                "target": target,
                                "target_mode": target_mode,
                                "model": model_name,
                                "pca_components": pca_n if pca_n is not None else np.nan,
                                "alpha": alpha,
                                "actual": y_true,
                                "predicted": y_pred,
                            }
                        )
                        prediction_frames.append(temp)

                    # Save scatter for reasonable number of models.
                    scatter_name = (
                        f"scatter_{embedding_kind}_{target.replace('.', '_')}_"
                        f"{target_mode}_{model_name}_"
                        f"pca{pca_n if pca_n is not None else 'none'}_"
                        f"alpha{alpha}.png"
                    )

                    save_scatter(
                        y_test_raw,
                        test_pred,
                        f"TRAQID test | {embedding_kind} | {target} | {target_mode} | {model_name}",
                        fig_dir / scatter_name,
                    )

                    # Save only the best-ish model objects later would require selection;
                    # to avoid huge clutter, save compact selected candidates.
                    if (
                        embedding_kind == "mean"
                        and target == "PM2.5"
                        and target_mode == "raw"
                        and model_name == "ridge_pca128"
                        and float(alpha) == 100.0
                    ):
                        model_path = model_dir / "example_mean_PM25_raw_ridge_pca128_alpha100.joblib"
                        joblib.dump(model, model_path)

    results_df = pd.DataFrame(results)
    pred_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    results_out = out_dir / "traqid_embedding_baseline_results.csv"
    pred_out = out_dir / "traqid_embedding_baseline_predictions.csv"
    best_out = out_dir / "traqid_embedding_best_by_val_rmse.csv"
    summary_out = out_dir / "traqid_embedding_baseline_summary.json"

    results_df.to_csv(results_out, index=False)
    pred_df.to_csv(pred_out, index=False)

    best = (
        results_df.sort_values(["target", "val_RMSE"])
        .groupby("target")
        .head(12)
        .reset_index(drop=True)
    )
    best.to_csv(best_out, index=False)

    summary = {
        "index_csv": str(index_csv),
        "embeddings_dir": str(embeddings_dir),
        "prefix": args.prefix,
        "rows_used": int(len(df)),
        "embedding_kinds": embedding_kinds,
        "targets": TARGETS,
        "target_modes": target_modes,
        "pca_components": pca_components,
        "alphas": alphas,
        "include_no_pca": bool(args.include_no_pca),
        "include_hgb": bool(args.include_hgb),
        "require_env_plausible": bool(args.require_env_plausible),
        "split_counts": df["split_date_chrono"].value_counts().to_dict(),
        "main_question": (
            "Do frozen visual embeddings beat the date-safe mean baseline "
            "under validation and test splits?"
        ),
    }

    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" -", results_out)
    print(" -", pred_out)
    print(" -", best_out)
    print(" -", summary_out)
    print(" - figures:", fig_dir)

    print("\nBest by validation RMSE:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()