import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = ["PM2.5", "PM10", "aqi"]

KEY_COLS = [
    "sequence_id",
    "target_row_id",
    "target_created_at",
    "seq_start_row_id",
    "seq_end_row_id",
]


def score_predictions(df, pred_prefix):
    rows = []

    for target in TARGETS:
        yt = df[f"actual_{target}"].values
        yp = df[f"{pred_prefix}_{target}"].values

        rows.append({
            "target": target,
            "R2": float(r2_score(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)),
        })

    rows.append({
        "target": "Average",
        "R2": float(np.mean([r["R2"] for r in rows])),
        "RMSE": float(np.mean([r["RMSE"] for r in rows])),
        "MAE": float(np.mean([r["MAE"] for r in rows])),
    })

    return pd.DataFrame(rows)


def read_prediction_file(path, model_index):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = KEY_COLS.copy()
    for target in TARGETS:
        required += [f"actual_{target}", f"predicted_{target}"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    keep = KEY_COLS.copy()
    for target in TARGETS:
        keep.append(f"actual_{target}")
        keep.append(f"predicted_{target}")

    df = df[keep].copy()

    rename = {}
    for target in TARGETS:
        rename[f"predicted_{target}"] = f"predicted_{target}_model{model_index}"

    return df.rename(columns=rename)


def merge_predictions(prediction_files):
    dfs = []

    summaries = []

    for i, p in enumerate(prediction_files):
        df = read_prediction_file(p, i)
        dfs.append(df)

        summaries.append({
            "model_index": i,
            "prediction_file": p,
            "rows": len(df),
            "unique_sequence_id": df["sequence_id"].nunique(),
            "min_target_row_id": df["target_row_id"].min(),
            "max_target_row_id": df["target_row_id"].max(),
        })

    merged = dfs[0].copy()

    for i, df in enumerate(dfs[1:], start=1):
        pred_cols = [f"predicted_{target}_model{i}" for target in TARGETS]
        actual_cols = [f"actual_{target}" for target in TARGETS]

        merged = merged.merge(
            df[KEY_COLS + actual_cols + pred_cols],
            on=KEY_COLS,
            how="inner",
            suffixes=("", f"_check{i}"),
        )

        for target in TARGETS:
            a = merged[f"actual_{target}"].values
            b = merged[f"actual_{target}_check{i}"].values

            if not np.allclose(a, b, equal_nan=True):
                raise ValueError(f"Actual target mismatch for {target} in model {i}")

            merged = merged.drop(columns=[f"actual_{target}_check{i}"])

    return merged, pd.DataFrame(summaries)


def add_equal_ensemble(df, model_indices, name):
    out = df.copy()

    for target in TARGETS:
        pred_cols = [f"predicted_{target}_model{i}" for i in model_indices]
        out[f"{name}_{target}"] = out[pred_cols].mean(axis=1)

    return out


def add_weighted_ensemble(df, model_indices, weights, name):
    out = df.copy()
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    for target in TARGETS:
        pred_cols = [f"predicted_{target}_model{i}" for i in model_indices]
        pred_matrix = out[pred_cols].values
        out[f"{name}_{target}"] = pred_matrix @ weights

    return out


def evaluate_single_models(df, n_models):
    rows = []

    for i in range(n_models):
        temp = df.copy()

        for target in TARGETS:
            temp[f"single_model_{target}"] = temp[f"predicted_{target}_model{i}"]

        metrics = score_predictions(temp, "single_model")
        avg = metrics[metrics["target"] == "Average"].iloc[0]

        rows.append({
            "kind": "single",
            "model_indices": str([i]),
            "num_models": 1,
            "weights": str([1.0]),
            "R2": avg["R2"],
            "RMSE": avg["RMSE"],
            "MAE": avg["MAE"],
        })

    return pd.DataFrame(rows)


def exhaustive_equal_subset_search(df, n_models, max_subset_size=None):
    rows = []

    if max_subset_size is None:
        max_subset_size = n_models

    for k in range(2, max_subset_size + 1):
        for subset in combinations(range(n_models), k):
            name = "ensemble_tmp"
            temp = add_equal_ensemble(df, subset, name)
            metrics = score_predictions(temp, name)
            avg = metrics[metrics["target"] == "Average"].iloc[0]

            rows.append({
                "kind": "equal_subset",
                "model_indices": str(list(subset)),
                "num_models": k,
                "weights": str([round(1 / k, 6)] * k),
                "R2": avg["R2"],
                "RMSE": avg["RMSE"],
                "MAE": avg["MAE"],
            })

    return pd.DataFrame(rows)


def greedy_forward_search(df, n_models):
    selected = []
    remaining = list(range(n_models))
    rows = []

    best_global_r2 = -1e18

    while remaining:
        candidates = []

        for candidate in remaining:
            subset = selected + [candidate]
            name = "greedy_tmp"

            temp = add_equal_ensemble(df, subset, name)
            metrics = score_predictions(temp, name)
            avg = metrics[metrics["target"] == "Average"].iloc[0]

            candidates.append({
                "candidate": candidate,
                "subset": subset,
                "R2": avg["R2"],
                "RMSE": avg["RMSE"],
                "MAE": avg["MAE"],
            })

        cand_df = pd.DataFrame(candidates).sort_values("R2", ascending=False)
        best = cand_df.iloc[0]

        if best["R2"] <= best_global_r2:
            break

        selected = list(best["subset"])
        remaining = [x for x in remaining if x not in selected]
        best_global_r2 = best["R2"]

        rows.append({
            "kind": "greedy_equal",
            "model_indices": str(selected),
            "num_models": len(selected),
            "weights": str([round(1 / len(selected), 6)] * len(selected)),
            "R2": best["R2"],
            "RMSE": best["RMSE"],
            "MAE": best["MAE"],
        })

    return pd.DataFrame(rows)


def random_weight_search(df, model_indices, n_trials, seed):
    rng = np.random.default_rng(seed)

    rows = []

    model_indices = list(model_indices)
    n = len(model_indices)

    if n < 2:
        return pd.DataFrame(rows)

    for trial in range(n_trials):
        weights = rng.dirichlet(np.ones(n))
        name = "weighted_tmp"

        temp = add_weighted_ensemble(df, model_indices, weights, name)
        metrics = score_predictions(temp, name)
        avg = metrics[metrics["target"] == "Average"].iloc[0]

        rows.append({
            "kind": "random_weighted",
            "trial": trial,
            "model_indices": str(model_indices),
            "num_models": n,
            "weights": str([round(float(w), 6) for w in weights]),
            "R2": avg["R2"],
            "RMSE": avg["RMSE"],
            "MAE": avg["MAE"],
        })

    return pd.DataFrame(rows)


def parse_indices(s):
    s = s.strip()

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    if not s:
        return []

    return [int(x.strip()) for x in s.split(",")]


def parse_weights(s):
    s = s.strip()

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    if not s:
        return []

    return [float(x.strip()) for x in s.split(",")]


def make_final_prediction_file(df, best_row):
    model_indices = parse_indices(best_row["model_indices"])
    weights = parse_weights(best_row["weights"])

    if best_row["kind"] in ["single", "equal_subset", "greedy_equal"]:
        name = "best_ensemble_predicted"
        temp = add_equal_ensemble(df, model_indices, name)

    elif best_row["kind"] == "random_weighted":
        name = "best_ensemble_predicted"
        temp = add_weighted_ensemble(df, model_indices, weights, name)

    else:
        raise ValueError(f"Unknown kind: {best_row['kind']}")

    keep = KEY_COLS.copy()

    for target in TARGETS:
        keep.append(f"actual_{target}")

    for target in TARGETS:
        keep.append(f"{name}_{target}")
        temp[f"best_ensemble_residual_{target}"] = temp[f"{name}_{target}"] - temp[f"actual_{target}"]
        keep.append(f"best_ensemble_residual_{target}")

    return temp[keep].copy(), name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-files", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-subset-size", type=int, default=None)
    parser.add_argument("--random-weight-trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged, input_summary = merge_predictions(args.prediction_files)
    n_models = len(args.prediction_files)

    single_results = evaluate_single_models(merged, n_models)
    subset_results = exhaustive_equal_subset_search(
        merged,
        n_models,
        max_subset_size=args.max_subset_size,
    )
    greedy_results = greedy_forward_search(merged, n_models)

    search_base = pd.concat(
        [single_results, subset_results, greedy_results],
        ignore_index=True,
    ).sort_values("R2", ascending=False)

    top_equal = search_base.head(10).copy()

    weighted_results_all = []

    for _, row in top_equal.iterrows():
        model_indices = parse_indices(row["model_indices"])

        if len(model_indices) >= 2:
            wr = random_weight_search(
                merged,
                model_indices=model_indices,
                n_trials=args.random_weight_trials,
                seed=args.seed,
            )
            weighted_results_all.append(wr)

    if weighted_results_all:
        weighted_results = pd.concat(weighted_results_all, ignore_index=True)
        all_results = pd.concat(
            [search_base, weighted_results],
            ignore_index=True,
        ).sort_values("R2", ascending=False)
    else:
        weighted_results = pd.DataFrame()
        all_results = search_base.copy()

    best_row = all_results.iloc[0].copy()

    final_predictions, pred_prefix = make_final_prediction_file(merged, best_row)
    final_metrics = score_predictions(final_predictions, pred_prefix)

    input_summary_path = out_dir / "input_file_summary.csv"
    all_results_path = out_dir / "ensemble_search_results.csv"
    top_results_path = out_dir / "top_ensemble_search_results.csv"
    final_pred_path = out_dir / "best_ensemble_predictions.csv"
    final_metrics_path = out_dir / "best_ensemble_metrics.csv"
    final_metrics_md_path = out_dir / "best_ensemble_metrics.md"

    input_summary.to_csv(input_summary_path, index=False)
    all_results.to_csv(all_results_path, index=False)
    all_results.head(50).to_csv(top_results_path, index=False)
    final_predictions.to_csv(final_pred_path, index=False)
    final_metrics.to_csv(final_metrics_path, index=False)
    final_metrics_md_path.write_text(final_metrics.to_markdown(index=False), encoding="utf-8")

    print("=" * 90)
    print("T7 GREEDY / WEIGHTED ENSEMBLE SEARCH COMPLETE")
    print("=" * 90)
    print("Models:", n_models)
    print("Rows used:", len(merged))
    print()
    print("Input files:")
    print(input_summary.to_string(index=False))
    print()
    print("Best ensemble:")
    print(best_row.to_string())
    print()
    print("Final metrics:")
    print(final_metrics.to_string(index=False))
    print()
    print("Top 20 search results:")
    print(all_results.head(20).to_string(index=False))
    print()
    print("Saved:", input_summary_path)
    print("Saved:", all_results_path)
    print("Saved:", top_results_path)
    print("Saved:", final_pred_path)
    print("Saved:", final_metrics_path)
    print("Saved:", final_metrics_md_path)


if __name__ == "__main__":
    main()