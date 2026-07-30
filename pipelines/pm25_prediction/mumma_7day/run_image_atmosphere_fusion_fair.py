"""Fair MUMMA target-image plus atmospheric-feature fusion experiment.

This is the MUMMA counterpart of the TRAQID Architecture-2 Stage-1 ablation.
It compares a lens-6 ResNet50 target-frame embedding, atmospheric inputs, and
their early concatenation under complete-date outer holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.image_embeddings.traqid_image_atmosphere_fusion import (
    ATMOSPHERIC_FEATURES,
)
from pipelines.pm25_prediction.mumma_7day.run_nested_background_ensemble import (
    conformal_radius,
)
from roadside_pm.validation.outer_folds import make_grouped_outer_folds


VARIANTS = ("image_only", "atmosphere_only", "early_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", required=True)
    parser.add_argument("--embedding-index", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--background-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--interval-alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "bias": float(np.mean(prediction - truth)),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.interval_alpha < 1.0:
        raise ValueError("--interval-alpha must be strictly between zero and one")
    if args.pca_components < 1:
        raise ValueError("--pca-components must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.modeling_table)
    table["sample_id"] = table["sample_id"].astype(str)
    table["date"] = table["date"].astype(str)
    if not table["sample_id"].is_unique:
        raise ValueError("Modeling-table sample IDs must be unique")
    if args.target not in table:
        raise ValueError(f"Target column is missing: {args.target}")

    index = pd.read_csv(args.embedding_index)
    index["sample_id"] = index["sample_id"].astype(str)
    if not index["sample_id"].is_unique:
        raise ValueError("Embedding-index sample IDs must be unique")
    if "embedding_status" in index and not index["embedding_status"].eq("success").all():
        raise ValueError("Embedding index contains unsuccessful rows")
    table = table.merge(
        index[["sample_id", "embedding_row"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if table["embedding_row"].isna().any():
        raise ValueError("Some modeling rows lack lens-6 embeddings")
    table["embedding_row"] = table["embedding_row"].astype(int)

    background = pd.read_csv(args.background_csv)
    background["sample_id"] = background["sample_id"].astype(str)
    required = {"sample_id", "background_status", *ATMOSPHERIC_FEATURES}
    missing = sorted(required - set(background.columns))
    if missing:
        raise ValueError(f"Atmospheric table is missing: {missing}")
    if not background["sample_id"].is_unique:
        raise ValueError("Atmospheric sample IDs must be unique")
    if not background["background_status"].eq("success").all():
        raise ValueError("Atmospheric table contains unsuccessful rows")
    table = table.merge(
        background[["sample_id", *ATMOSPHERIC_FEATURES]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if table[ATMOSPHERIC_FEATURES].isna().all(axis=1).any():
        raise ValueError("Some rows lack every atmospheric feature")

    embeddings = np.load(args.embeddings, mmap_mode="r")
    embedding_rows = table["embedding_row"].to_numpy(dtype=int)
    if embedding_rows.min() < 0 or embedding_rows.max() >= len(embeddings):
        raise ValueError("Embedding rows exceed the embedding array")

    folds = make_grouped_outer_folds(
        table,
        group_column="date",
        n_splits=args.outer_folds,
        seed=args.seed,
    )
    if args.max_folds:
        folds = folds[: args.max_folds]

    fold_rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []
    for fold_number, fold in enumerate(folds, start=1):
        train = fold["outer_split"].eq("train").to_numpy()
        validation = fold["outer_split"].eq("val").to_numpy()
        test = fold["outer_split"].eq("test").to_numpy()
        if not train.any() or not validation.any() or not test.any():
            raise ValueError(f"Fold {fold_number} contains an empty partition")

        fold_embedding_rows = fold["embedding_row"].to_numpy(dtype=int)
        image_train = np.asarray(
            embeddings[fold_embedding_rows[train]], dtype=np.float32,
        )
        n_components = min(
            args.pca_components,
            image_train.shape[0] - 1,
            image_train.shape[1],
        )
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=args.seed + fold_number,
        )
        image_by_split = {
            "train": pca.fit_transform(image_train),
            "val": pca.transform(
                np.asarray(embeddings[fold_embedding_rows[validation]], dtype=np.float32)
            ),
            "test": pca.transform(
                np.asarray(embeddings[fold_embedding_rows[test]], dtype=np.float32)
            ),
        }
        imputer = SimpleImputer(strategy="median")
        atmosphere_by_split = {
            "train": imputer.fit_transform(fold.loc[train, ATMOSPHERIC_FEATURES]),
            "val": imputer.transform(fold.loc[validation, ATMOSPHERIC_FEATURES]),
            "test": imputer.transform(fold.loc[test, ATMOSPHERIC_FEATURES]),
        }
        targets = {
            "train": fold.loc[train, args.target].to_numpy(dtype=float),
            "val": fold.loc[validation, args.target].to_numpy(dtype=float),
            "test": fold.loc[test, args.target].to_numpy(dtype=float),
        }
        matrices = {
            "image_only": image_by_split,
            "atmosphere_only": atmosphere_by_split,
            "early_fusion": {
                split: np.concatenate(
                    [image_by_split[split], atmosphere_by_split[split]],
                    axis=1,
                )
                for split in ("train", "val", "test")
            },
        }
        for variant in args.variants:
            model = ExtraTreesRegressor(
                n_estimators=args.trees,
                min_samples_leaf=args.min_samples_leaf,
                max_features=0.8,
                n_jobs=-1,
                random_state=args.seed + 100 * fold_number,
            )
            model.fit(matrices[variant]["train"], targets["train"])
            validation_prediction = model.predict(matrices[variant]["val"])
            radius = conformal_radius(
                targets["val"],
                validation_prediction,
                alpha=args.interval_alpha,
            )
            prediction = model.predict(matrices[variant]["test"])
            lower = prediction - radius
            upper = prediction + radius
            covered = (
                (targets["test"] >= lower) & (targets["test"] <= upper)
            )
            fold_rows.append(
                {
                    "fold": fold_number,
                    "test_dates": "|".join(sorted(fold.loc[test, "date"].unique())),
                    "variant": variant,
                    "n_train": int(train.sum()),
                    "n_validation": int(validation.sum()),
                    "n_test": int(test.sum()),
                    **metrics(targets["test"], prediction),
                    "interval_alpha": args.interval_alpha,
                    "interval_radius": radius,
                    "interval_coverage": float(np.mean(covered)),
                }
            )
            prediction_tables.append(
                pd.DataFrame(
                    {
                        "sample_id": fold.loc[test, "sample_id"].to_numpy(),
                        "date": fold.loc[test, "date"].to_numpy(),
                        "fold": fold_number,
                        "variant": variant,
                        f"actual_{args.target}": targets["test"],
                        f"predicted_{args.target}": prediction,
                        "interval_alpha": args.interval_alpha,
                        "interval_radius": radius,
                        "interval_lower": lower,
                        "interval_upper": upper,
                        "interval_covered": covered,
                    }
                )
            )

    by_fold = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    aggregate_rows: list[dict[str, object]] = []
    for variant, part in predictions.groupby("variant", sort=False):
        matching = by_fold[by_fold["variant"].eq(variant)]
        aggregate_rows.append(
            {
                "variant": variant,
                "n_test": len(part),
                **metrics(
                    part[f"actual_{args.target}"].to_numpy(dtype=float),
                    part[f"predicted_{args.target}"].to_numpy(dtype=float),
                ),
                "mean_fold_r2": float(matching["r2"].mean()),
                "mean_fold_rmse": float(matching["rmse"].mean()),
                "worst_fold_rmse": float(matching["rmse"].max()),
                "empirical_interval_coverage": float(
                    part["interval_covered"].mean()
                ),
                "mean_interval_width": float(
                    2.0 * part["interval_radius"].mean()
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values("rmse")
    by_fold.to_csv(output / "metrics_by_fold.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    (output / "atmospheric_feature_columns.txt").write_text(
        "\n".join(ATMOSPHERIC_FEATURES) + "\n"
    )
    run = {
        "dataset": "MUMMA five-day",
        "experiment": "lens-6 image plus atmospheric-feature early fusion",
        "target": args.target,
        "pure_atmospheric_target_available": False,
        "source_specific_outputs": False,
        "protocol": "complete-date outer holdout",
        "input_image": "target-frame lens-6 ResNet50 embedding",
        "image_preprocessing": (
            f"train-only randomized PCA, requested components={args.pca_components}"
        ),
        "atmospheric_inputs": ATMOSPHERIC_FEATURES,
        "model": {
            "type": "ExtraTreesRegressor",
            "trees": args.trees,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": 0.8,
        },
        "uncertainty": {
            "method": "validation absolute-residual conformal interval",
            "alpha": args.interval_alpha,
            "outer_test_targets_used_for_calibration": False,
        },
        "warning": (
            "This tests predictive multimodal fusion. The target is roadside "
            "PM2.5, not pure atmospheric PM2.5 or source apportionment."
        ),
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    print("\nMUMMA IMAGE + ATMOSPHERIC FUSION\n")
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
