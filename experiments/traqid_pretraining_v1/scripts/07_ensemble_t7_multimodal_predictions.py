import argparse
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


def evaluate(df):
    rows = []

    for target in TARGETS:
        yt = df[f"actual_{target}"].values
        yp = df[f"ensemble_predicted_{target}"].values

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


def read_prediction_file(path, model_idx):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = KEY_COLS.copy()
    for target in TARGETS:
        required.append(f"actual_{target}")
        required.append(f"predicted_{target}")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df[required].copy()

    dup_count = df.duplicated(KEY_COLS).sum()
    if dup_count > 0:
        raise ValueError(
            f"{path} has {dup_count} duplicate prediction keys. "
            "Cannot safely ensemble."
        )

    rename = {}
    for target in TARGETS:
        rename[f"predicted_{target}"] = f"predicted_{target}_model{model_idx}"

    df = df.rename(columns=rename)

    return df


def check_target_consistency(merged, model_idx):
    for target in TARGETS:
        main_col = f"actual_{target}"
        check_col = f"actual_{target}_check{model_idx}"

        if check_col not in merged.columns:
            continue

        a = merged[main_col].values
        b = merged[check_col].values

        if not np.allclose(a, b, equal_nan=True):
            max_abs_diff = np.nanmax(np.abs(a - b))
            raise ValueError(
                f"Actual target mismatch for {target} in model {model_idx}. "
                f"Max abs diff = {max_abs_diff}"
            )

        merged = merged.drop(columns=[check_col])

    return merged


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction-files", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-common-rows", type=int, default=5000)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(args.prediction_files) < 2:
        raise ValueError("Give at least two prediction files for ensembling.")

    dfs = []
    file_summaries = []

    for i, p in enumerate(args.prediction_files):
        df = read_prediction_file(p, i)
        dfs.append(df)

        file_summaries.append({
            "model_index": i,
            "prediction_file": str(p),
            "rows": len(df),
            "unique_sequence_id": df["sequence_id"].nunique(),
            "min_target_row_id": df["target_row_id"].min(),
            "max_target_row_id": df["target_row_id"].max(),
        })

    summary_df = pd.DataFrame(file_summaries)

    merged = dfs[0].copy()

    for i, df in enumerate(dfs[1:], start=1):
        pred_cols = [f"predicted_{target}_model{i}" for target in TARGETS]
        actual_cols = [f"actual_{target}" for target in TARGETS]

        df_merge = df[KEY_COLS + actual_cols + pred_cols].copy()

        before = len(merged)

        merged = merged.merge(
            df_merge,
            on=KEY_COLS,
            how="inner",
            suffixes=("", f"_check{i}"),
        )

        after = len(merged)

        print(f"After merging model {i}: {before} -> {after} common rows")

        merged = check_target_consistency(merged, i)

    if len(merged) < args.min_common_rows:
        raise RuntimeError(
            f"Only {len(merged)} common rows found across models. "
            f"Expected at least {args.min_common_rows}. "
            "This usually means the models were trained/evaluated on different test splits. "
            "Make sure all runs used the same --split-seed and only changed --seed."
        )

    for target in TARGETS:
        pred_cols = [
            c for c in merged.columns
            if c.startswith(f"predicted_{target}_model")
        ]

        if len(pred_cols) != len(args.prediction_files):
            raise RuntimeError(
                f"For {target}, expected {len(args.prediction_files)} prediction columns, "
                f"found {len(pred_cols)}: {pred_cols}"
            )

        merged[f"ensemble_predicted_{target}"] = merged[pred_cols].mean(axis=1)
        merged[f"ensemble_std_{target}"] = merged[pred_cols].std(axis=1)
        merged[f"ensemble_residual_{target}"] = (
            merged[f"ensemble_predicted_{target}"] - merged[f"actual_{target}"]
        )
        merged[f"ensemble_abs_error_{target}"] = merged[f"ensemble_residual_{target}"].abs()

    metrics = evaluate(merged)

    pred_path = out_dir / "ensemble_predictions.csv"
    metrics_path = out_dir / "ensemble_metrics.csv"
    metrics_md_path = out_dir / "ensemble_metrics.md"
    summary_path = out_dir / "ensemble_input_file_summary.csv"

    merged.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    metrics_md_path.write_text(metrics.to_markdown(index=False), encoding="utf-8")

    print("=" * 90)
    print("T7 MULTIMODAL GRU ENSEMBLE COMPLETE")
    print("=" * 90)
    print("Models:", len(args.prediction_files))
    print("Rows used:", len(merged))
    print()
    print("Input prediction files:")
    print(summary_df.to_string(index=False))
    print()
    print(metrics.to_string(index=False))
    print()
    print("Saved:", pred_path)
    print("Saved:", metrics_path)
    print("Saved:", metrics_md_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()