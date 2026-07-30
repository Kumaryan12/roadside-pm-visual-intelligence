import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_value(y_true, y_pred, metric):
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "rmse":
        return rmse(y_true, y_pred)
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_ci(y_true, y_pred, metric, n_boot, seed, ci):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_value(y_true[idx], y_pred[idx], metric))

    vals = np.asarray(vals)
    alpha = (100 - ci) / 2

    return {
        "mean_boot": float(np.mean(vals)),
        "std_boot": float(np.std(vals, ddof=1)),
        "ci_low": float(np.percentile(vals, alpha)),
        "ci_high": float(np.percentile(vals, 100 - alpha)),
    }


def get_target_columns(df, target):
    true_candidates = [
        f"actual_{target}",
        f"{target}_actual",
        f"true_{target}",
        f"{target}_true",
        f"y_true_{target}",
    ]

    pred_candidates = [
        f"predicted_{target}",
        f"{target}_predicted",
        f"pred_{target}",
        f"{target}_pred",
        f"y_pred_{target}",
    ]

    true_col = next((c for c in true_candidates if c in df.columns), None)
    pred_col = next((c for c in pred_candidates if c in df.columns), None)

    return true_col, pred_col


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for prediction metrics."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--targets", nargs="+", default=["PM2.5", "PM10", "aqi"])
    parser.add_argument("--metrics", nargs="+", default=["r2", "rmse", "mae"])
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)

    if args.split_col in df.columns:
        df = df[df[args.split_col].astype(str) == args.eval_split].copy()

    if df.empty:
        raise ValueError(
            f"No rows found after filtering {args.split_col} == {args.eval_split}. "
            "Check available split values."
        )

    rows = []

    for target in args.targets:
        true_col, pred_col = get_target_columns(df, target)

        if true_col is None or pred_col is None:
            print(f"Skipping {target}: could not find actual/predicted columns.")
            continue

        y_true = df[true_col].astype(float).to_numpy()
        y_pred = df[pred_col].astype(float).to_numpy()

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        for metric in args.metrics:
            point = metric_value(y_true, y_pred, metric)
            ci_stats = bootstrap_ci(
                y_true=y_true,
                y_pred=y_pred,
                metric=metric,
                n_boot=args.n_boot,
                seed=args.seed,
                ci=args.ci,
            )

            rows.append({
                "prediction_file": str(pred_path),
                "eval_split": args.eval_split,
                "target": target,
                "metric": metric,
                "n": len(y_true),
                "point": point,
                **ci_stats,
            })

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("No metrics computed. Check target column names.")

    # Add average rows across targets for each metric using point metrics only.
    for metric in args.metrics:
        metric_rows = out[out["metric"] == metric]
        if not metric_rows.empty:
            out = pd.concat([
                out,
                pd.DataFrame([{
                    "prediction_file": str(pred_path),
                    "eval_split": args.eval_split,
                    "target": "Average",
                    "metric": metric,
                    "n": int(metric_rows["n"].min()),
                    "point": float(metric_rows["point"].mean()),
                    "mean_boot": float(metric_rows["mean_boot"].mean()),
                    "std_boot": float(metric_rows["std_boot"].mean()),
                    "ci_low": float(metric_rows["ci_low"].mean()),
                    "ci_high": float(metric_rows["ci_high"].mean()),
                }])
            ], ignore_index=True)

    safe_name = pred_path.parent.name + "_" + pred_path.stem
    csv_path = out_dir / f"{safe_name}_{args.eval_split}_bootstrap_ci.csv"
    md_path = out_dir / f"{safe_name}_{args.eval_split}_bootstrap_ci.md"

    out.to_csv(csv_path, index=False)
    md_path.write_text(out.to_markdown(index=False))

    print("=" * 90)
    print("BOOTSTRAP METRIC CI")
    print("=" * 90)
    print("Predictions:", pred_path)
    print("Eval split:", args.eval_split)
    print("Rows:", len(df))
    print("Saved:", csv_path)
    print("Saved:", md_path)
    print()
    print(out[out["target"].eq("Average")].to_string(index=False))


if __name__ == "__main__":
    main()