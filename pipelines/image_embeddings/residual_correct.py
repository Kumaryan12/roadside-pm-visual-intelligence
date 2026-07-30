"""Cross-fit tabular residual correction over safe image-base predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from roadside_pm.modeling.residual_fusion import build_residual_dataset, make_residual_pipeline, select_numeric_residual_features


def metric_row(actual, predicted):
    return {"MAE": float(mean_absolute_error(actual, predicted)), "RMSE": float(mean_squared_error(actual, predicted) ** 0.5), "R2": float(r2_score(actual, predicted)), "Bias": float(np.mean(predicted - actual))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", required=True); parser.add_argument("--sequence-manifest", required=True); parser.add_argument("--feature-table", required=True)
    parser.add_argument("--target", required=True); parser.add_argument("--sample-key", default="sample_index"); parser.add_argument("--group-col", required=True)
    parser.add_argument("--feature-cols", nargs="*", default=None); parser.add_argument("--model", choices=["ridge", "extra_trees", "hist_gradient_boosting"], default="extra_trees")
    parser.add_argument("--folds", type=int, default=5); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exploratory-non-nested", action="store_true", help="Acknowledge that independent residual CV is exploratory and not a valid nested-stacking estimate.")
    args = parser.parse_args()
    if not args.exploratory_non_nested:
        raise ValueError(
            "Independent residual GroupKFold is not nested inside the image outer folds. "
            "It can leak outer-evaluation groups through meta-training base predictions. "
            "Use a nested residual pipeline, or pass --exploratory-non-nested only for diagnostics."
        )
    dataset = build_residual_dataset(pd.read_csv(args.base_predictions), pd.read_csv(args.sequence_manifest), pd.read_csv(args.feature_table), target=args.target, sample_key=args.sample_key, group_column=args.group_col)
    features = select_numeric_residual_features(dataset, key_columns=["sequence_id", "target_sample_id", args.sample_key, args.group_col], allow_columns=args.feature_cols)
    groups = dataset[args.group_col].astype(str); unique_groups = groups.nunique()
    if unique_groups < 2: raise ValueError("Residual correction requires at least two groups")
    folds = min(args.folds, unique_groups); splitter = GroupKFold(n_splits=folds)
    residual_prediction = np.full(len(dataset), np.nan); fold_ids = np.full(len(dataset), -1)
    for fold, (train_index, eval_index) in enumerate(splitter.split(dataset[features], dataset["base_residual"], groups), 1):
        pipeline = make_residual_pipeline(args.model, features, args.seed + fold)
        pipeline.fit(dataset.iloc[train_index][features], dataset.iloc[train_index]["base_residual"])
        residual_prediction[eval_index] = pipeline.predict(dataset.iloc[eval_index][features]); fold_ids[eval_index] = fold
    actual = dataset[f"actual_{args.target}"].to_numpy(float); base = dataset[f"predicted_{args.target}"].to_numpy(float); corrected = np.clip(base + residual_prediction, 0, None)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    predictions = dataset[["sequence_id", "target_sample_id", args.group_col, "prediction_origin"]].copy()
    predictions["residual_fold"] = fold_ids; predictions[f"actual_{args.target}"] = actual; predictions[f"base_{args.target}"] = base; predictions[f"predicted_residual_{args.target}"] = residual_prediction; predictions[f"corrected_{args.target}"] = corrected
    predictions.to_csv(output / "residual_corrected_predictions.csv", index=False)
    report = {"target": args.target, "model": args.model, "protocol_status": "exploratory_non_nested_do_not_use_for_model_selection", "folds": folds, "n_rows": len(dataset), "n_groups": unique_groups, "n_features": len(features), "features": features, "base": metric_row(actual, base), "corrected": metric_row(actual, corrected)}
    (output / "metrics.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
