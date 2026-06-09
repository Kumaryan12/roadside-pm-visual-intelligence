from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_TARGETS = ["PM2.5"]


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
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": safe_corr(y_true, y_pred, "pearson"),
        "Spearman": safe_corr(y_true, y_pred, "spearman"),
    }


def prepare_target(y, mode: str):
    y = np.asarray(y, dtype=float)

    if mode == "raw":
        return y

    if mode == "log1p":
        return np.log1p(y)

    raise ValueError(f"Unknown target mode: {mode}")


def invert_target(y_pred, mode: str):
    y_pred = np.asarray(y_pred, dtype=float)

    if mode == "raw":
        return y_pred

    if mode == "log1p":
        return np.expm1(y_pred)

    raise ValueError(f"Unknown target mode: {mode}")


def load_embedding(embeddings_dir: Path, prefix: str, kind: str):
    paths = {
        "front": embeddings_dir / f"{prefix}_front_embeddings.npy",
        "rear": embeddings_dir / f"{prefix}_rear_embeddings.npy",
        "mean": embeddings_dir / f"{prefix}_front_rear_mean_embeddings.npy",
        "concat": embeddings_dir / f"{prefix}_front_rear_concat_embeddings.npy",
    }

    if kind not in paths:
        raise ValueError(f"Unknown embedding kind: {kind}. Use front,rear,mean,concat.")

    path = paths[kind]

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    return np.load(path, mmap_mode="r"), path


def build_ridge_model(pca_components: int | None, alpha: float, random_state: int):
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

    return Pipeline(steps)


def plot_date_rmse(results_df: pd.DataFrame, out_path: Path, title: str):
    if results_df.empty:
        return

    plot_df = results_df.copy()
    plot_df = plot_df.sort_values("heldout_date")

    plt.figure(figsize=(12, 5))

    for model_label, sub in plot_df.groupby("model_label"):
        plt.plot(
            sub["heldout_date"],
            sub["RMSE"],
            marker="o",
            linewidth=1.5,
            label=model_label,
        )

    plt.title(title)
    plt.xlabel("Held-out date")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_date_spearman(results_df: pd.DataFrame, out_path: Path, title: str):
    if results_df.empty:
        return

    plot_df = results_df.copy()
    plot_df = plot_df.sort_values("heldout_date")

    plt.figure(figsize=(12, 5))

    for model_label, sub in plot_df.groupby("model_label"):
        plt.plot(
            sub["heldout_date"],
            sub["Spearman"],
            marker="o",
            linewidth=1.5,
            label=model_label,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Held-out date")
    plt.ylabel("Spearman correlation")
    plt.xticks(rotation=45, ha="right")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


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
        default="front,rear,mean",
        help="Comma-separated: front,rear,mean,concat",
    )

    parser.add_argument(
        "--targets",
        default="PM2.5",
        help="Comma-separated targets, e.g. PM2.5,PM10,aqi",
    )

    parser.add_argument(
        "--target-modes",
        default="log1p",
        help="Comma-separated: raw,log1p",
    )

    parser.add_argument(
        "--pca-components",
        default="64,128,256",
        help="Comma-separated PCA components. Example: 64,128,256",
    )

    parser.add_argument(
        "--alphas",
        default="100,1000",
        help="Comma-separated Ridge alpha values.",
    )

    parser.add_argument(
        "--include-mean-baseline",
        action="store_true",
        help="Include date-wise mean baseline.",
    )

    parser.add_argument(
        "--require-env-plausible",
        action="store_true",
        help="Filter rows to plausible Temperature/Humidity.",
    )

    parser.add_argument(
        "--max-dates",
        type=int,
        default=None,
        help="Optional quick test: evaluate only first N dates.",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save row-level held-out predictions. Can be large.",
    )

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports/lodo_embedding_baselines",
    )

    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures/lodo_embedding_baselines",
    )

    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    index_csv = Path(args.index_csv)
    embeddings_dir = Path(args.embeddings_dir)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(index_csv)

    required_cols = [
        "row_id",
        "image_id",
        "created_at",
        "date",
        "PM2.5",
        "PM10",
        "aqi",
        "Temperature",
        "Humidity",
    ]

    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Index CSV missing required columns: {missing}")

    # Preserve original embedding row index before filtering.
    df_all = df_all.reset_index(drop=False).rename(columns={"index": "embedding_row_index"})

    if args.require_env_plausible:
        df_all = df_all[
            df_all["Temperature"].between(-10, 60)
            & df_all["Humidity"].between(0, 100)
        ].copy()

    df_all = df_all.reset_index(drop=True)

    dates = sorted(df_all["date"].dropna().unique().tolist())

    if args.max_dates is not None:
        dates = dates[: args.max_dates]

    embedding_kinds = [x.strip() for x in args.embedding_kinds.split(",") if x.strip()]
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    target_modes = [x.strip() for x in args.target_modes.split(",") if x.strip()]
    pca_components_list = [int(x.strip()) for x in args.pca_components.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    print("=" * 90)
    print("TRAQID LEAVE-ONE-DATE-OUT EMBEDDING BASELINE")
    print("=" * 90)
    print("Index CSV:", index_csv)
    print("Rows:", len(df_all))
    print("Unique dates:", len(dates))
    print("Dates:", dates)
    print("Embedding kinds:", embedding_kinds)
    print("Targets:", targets)
    print("Target modes:", target_modes)
    print("PCA components:", pca_components_list)
    print("Alphas:", alphas)
    print("Include mean baseline:", args.include_mean_baseline)
    print("Require env plausible:", args.require_env_plausible)

    date_stats = (
        df_all.groupby("date")
        .agg(
            rows=("row_id", "count"),
            pm25_mean=("PM2.5", "mean"),
            pm25_min=("PM2.5", "min"),
            pm25_max=("PM2.5", "max"),
            pm10_mean=("PM10", "mean"),
            aqi_mean=("aqi", "mean"),
        )
        .reset_index()
    )

    date_stats.to_csv(out_dir / "lodo_date_stats.csv", index=False)

    results = []
    pred_frames = []

    # Mean baseline diagnostics.
    if args.include_mean_baseline:
        for heldout_date in dates:
            train_df = df_all[df_all["date"] != heldout_date].copy()
            test_df = df_all[df_all["date"] == heldout_date].copy()

            for target in targets:
                y_train_raw = train_df[target].astype(float).to_numpy()
                y_test_raw = test_df[target].astype(float).to_numpy()

                for target_mode in target_modes:
                    y_train = prepare_target(y_train_raw, target_mode)

                    dummy = DummyRegressor(strategy="mean")
                    dummy.fit(np.zeros((len(y_train), 1)), y_train)

                    test_pred_model = dummy.predict(np.zeros((len(y_test_raw), 1)))
                    test_pred = invert_target(test_pred_model, target_mode)
                    test_pred = np.maximum(test_pred, 0)

                    metrics = evaluate(y_test_raw, test_pred)

                    row = {
                        "heldout_date": heldout_date,
                        "train_rows": int(len(train_df)),
                        "test_rows": int(len(test_df)),
                        "embedding_kind": "none",
                        "target": target,
                        "target_mode": target_mode,
                        "model": "mean_baseline",
                        "pca_components": np.nan,
                        "alpha": np.nan,
                        "model_label": f"mean_baseline_{target_mode}",
                    }

                    row.update(metrics)
                    results.append(row)

    # Embedding models.
    for embedding_kind in embedding_kinds:
        X_all, emb_path = load_embedding(
            embeddings_dir=embeddings_dir,
            prefix=args.prefix,
            kind=embedding_kind,
        )

        print("\n" + "=" * 90)
        print("Embedding kind:", embedding_kind)
        print("Embedding file:", emb_path)
        print("Embedding shape:", X_all.shape)
        print("=" * 90)

        for heldout_idx, heldout_date in enumerate(dates, start=1):
            train_df = df_all[df_all["date"] != heldout_date].copy()
            test_df = df_all[df_all["date"] == heldout_date].copy()

            train_idx = train_df["embedding_row_index"].to_numpy()
            test_idx = test_df["embedding_row_index"].to_numpy()

            X_train = np.asarray(X_all[train_idx], dtype=np.float32)
            X_test = np.asarray(X_all[test_idx], dtype=np.float32)

            print(
                f"\n[{heldout_idx}/{len(dates)}] Held-out date: {heldout_date} | "
                f"train={len(train_df)} test={len(test_df)}"
            )

            for target in targets:
                y_train_raw = train_df[target].astype(float).to_numpy()
                y_test_raw = test_df[target].astype(float).to_numpy()

                for target_mode in target_modes:
                    y_train = prepare_target(y_train_raw, target_mode)

                    for pca_n in pca_components_list:
                        for alpha in alphas:
                            model = build_ridge_model(
                                pca_components=pca_n,
                                alpha=alpha,
                                random_state=args.random_state,
                            )

                            model.fit(X_train, y_train)

                            test_pred_model = model.predict(X_test)
                            test_pred = invert_target(test_pred_model, target_mode)
                            test_pred = np.maximum(test_pred, 0)

                            metrics = evaluate(y_test_raw, test_pred)

                            model_label = f"{embedding_kind}_{target_mode}_pca{pca_n}_a{alpha:g}"

                            row = {
                                "heldout_date": heldout_date,
                                "train_rows": int(len(train_df)),
                                "test_rows": int(len(test_df)),
                                "embedding_kind": embedding_kind,
                                "target": target,
                                "target_mode": target_mode,
                                "model": "ridge",
                                "pca_components": int(pca_n),
                                "alpha": float(alpha),
                                "model_label": model_label,
                            }

                            row.update(metrics)
                            results.append(row)

                            print(
                                f"{embedding_kind:5s} | {target:5s} | {target_mode:5s} | "
                                f"pca={pca_n:3d} alpha={alpha:g} | "
                                f"RMSE={metrics['RMSE']:.3f} "
                                f"R2={metrics['R2']:.3f} "
                                f"Spearman={metrics['Spearman']:.3f}"
                            )

                            if args.save_predictions:
                                temp = pd.DataFrame(
                                    {
                                        "heldout_date": heldout_date,
                                        "row_id": test_df["row_id"].values,
                                        "image_id": test_df["image_id"].values,
                                        "created_at": test_df["created_at"].values,
                                        "date": test_df["date"].values,
                                        "embedding_kind": embedding_kind,
                                        "target": target,
                                        "target_mode": target_mode,
                                        "pca_components": int(pca_n),
                                        "alpha": float(alpha),
                                        "actual": y_test_raw,
                                        "predicted": test_pred,
                                    }
                                )
                                pred_frames.append(temp)

    results_df = pd.DataFrame(results)

    results_path = out_dir / "lodo_embedding_results.csv"
    results_df.to_csv(results_path, index=False)

    if pred_frames:
        pred_df = pd.concat(pred_frames, ignore_index=True)
        pred_df.to_csv(out_dir / "lodo_embedding_predictions.csv", index=False)

    # Aggregate across held-out dates.
    group_cols = [
        "embedding_kind",
        "target",
        "target_mode",
        "model",
        "pca_components",
        "alpha",
        "model_label",
    ]

    agg = (
        results_df.groupby(group_cols, dropna=False)
        .agg(
            n_dates=("heldout_date", "nunique"),
            mean_RMSE=("RMSE", "mean"),
            median_RMSE=("RMSE", "median"),
            std_RMSE=("RMSE", "std"),
            mean_MAE=("MAE", "mean"),
            median_MAE=("MAE", "median"),
            mean_R2=("R2", "mean"),
            median_R2=("R2", "median"),
            mean_Spearman=("Spearman", "mean"),
            median_Spearman=("Spearman", "median"),
            positive_spearman_dates=("Spearman", lambda x: int((x > 0).sum())),
            negative_spearman_dates=("Spearman", lambda x: int((x < 0).sum())),
            min_Spearman=("Spearman", "min"),
            max_Spearman=("Spearman", "max"),
        )
        .reset_index()
        .sort_values(["target", "median_RMSE"])
    )

    agg_path = out_dir / "lodo_embedding_aggregate_results.csv"
    agg.to_csv(agg_path, index=False)

    best = (
        agg.sort_values(["target", "median_RMSE"])
        .groupby("target")
        .head(15)
        .reset_index(drop=True)
    )

    best_path = out_dir / "lodo_embedding_best_by_median_rmse.csv"
    best.to_csv(best_path, index=False)

    # Worst-date diagnostics for best model per target.
    worst_rows = []

    for target in targets:
        best_for_target = best[best["target"] == target].head(1)

        if len(best_for_target) == 0:
            continue

        label = best_for_target.iloc[0]["model_label"]

        sub = results_df[
            (results_df["target"] == target)
            & (results_df["model_label"] == label)
        ].copy()

        sub = sub.sort_values("RMSE", ascending=False).head(10)
        worst_rows.append(sub)

        plot_date_rmse(
            sub.sort_values("heldout_date"),
            fig_dir / f"worst_dates_rmse_{target.replace('.', '_')}_{label}.png",
            f"LODO RMSE worst dates | {target} | {label}",
        )

    if worst_rows:
        worst_df = pd.concat(worst_rows, ignore_index=True)
        worst_df.to_csv(out_dir / "lodo_worst_dates_for_best_models.csv", index=False)

    # Plots for top few PM2.5 models.
    for target in targets:
        top_labels = (
            best[best["target"] == target]["model_label"]
            .dropna()
            .head(5)
            .tolist()
        )

        plot_df = results_df[
            (results_df["target"] == target)
            & (results_df["model_label"].isin(top_labels))
        ].copy()

        plot_date_rmse(
            plot_df,
            fig_dir / f"lodo_rmse_by_date_top_{target.replace('.', '_')}.png",
            f"LODO RMSE by date | top models | {target}",
        )

        plot_date_spearman(
            plot_df,
            fig_dir / f"lodo_spearman_by_date_top_{target.replace('.', '_')}.png",
            f"LODO Spearman by date | top models | {target}",
        )

    summary = {
        "index_csv": str(index_csv),
        "embeddings_dir": str(embeddings_dir),
        "prefix": args.prefix,
        "rows_used": int(len(df_all)),
        "unique_dates_evaluated": int(len(dates)),
        "dates": dates,
        "embedding_kinds": embedding_kinds,
        "targets": targets,
        "target_modes": target_modes,
        "pca_components": pca_components_list,
        "alphas": alphas,
        "include_mean_baseline": bool(args.include_mean_baseline),
        "require_env_plausible": bool(args.require_env_plausible),
        "main_question": (
            "Does frozen visual embedding signal remain stable when each collection date "
            "is held out completely?"
        ),
    }

    summary_path = out_dir / "lodo_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("LODO COMPLETE")
    print("=" * 90)
    print("Saved:")
    print(" -", results_path)
    print(" -", agg_path)
    print(" -", best_path)
    print(" -", summary_path)
    print(" - figures:", fig_dir)

    print("\nBest by median RMSE:")
    print(best.to_string(index=False))

    print("\nInterpretation rule:")
    print("Strong visual signal: low median RMSE AND mostly positive Spearman dates.")
    print("Weak/unstable signal: acceptable RMSE but many negative Spearman dates.")
    print("No useful signal: mean baseline remains competitive or best.")


if __name__ == "__main__":
    main()