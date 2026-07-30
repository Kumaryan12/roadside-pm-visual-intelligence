"""Fair TRAQID image-plus-atmospheric-feature fusion experiment.

This runner tests the trainable first stage of the proposed architecture:

    f(ResNet50 image embedding, CAMS/MERRA-2/ERA5 features) -> roadside PM2.5

It does *not* identify a pure atmospheric target or source-specific road and
vehicle concentrations. Complete acquisition dates are held out. PCA,
imputation, model fitting, and conformal calibration are restricted to the
appropriate outer-train or outer-validation partitions.
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

from pipelines.image_embeddings.traqid_outer_crossfit import canonicalize_sequences
from pipelines.pm25_prediction.mumma_7day.run_nested_background_ensemble import (
    conformal_radius,
)
from roadside_pm.validation.outer_folds import make_grouped_outer_folds


ATMOSPHERIC_FEATURES = [
    "background_cams_forecast_hour",
    "background_cams_pm25_ug_m3",
    "background_cams_pm10_ug_m3",
    "background_cams_pm1_ug_m3",
    "background_cams_aod550",
    "background_cams_dust_aod550",
    "background_cams_black_carbon_aod550",
    "background_merra2_pm25_ug_m3",
    "background_merra2_dust25_ug_m3",
    "background_merra2_sea_salt25_ug_m3",
    "background_merra2_black_carbon_ug_m3",
    "background_merra2_organic_carbon_ug_m3",
    "background_merra2_sulphate_ug_m3",
    "background_era5_temperature_2m_c",
    "background_era5_dewpoint_2m_c",
    "background_era5_relative_humidity_pct",
    "background_era5_wind_u_10m_m_s",
    "background_era5_wind_v_10m_m_s",
    "background_era5_wind_speed_10m_m_s",
    "background_era5_surface_pressure_hpa",
    "background_era5_precipitation_hourly_mm",
    "background_era5_solar_radiation_hourly_w_m2",
]
VARIANTS = ("image_only", "atmosphere_only", "early_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-manifest",
        default=(
            "experiments/traqid_pretraining_v1/data/processed/"
            "paper_style_sequences/"
            "traqid_paper_style_T7_front_sequence_manifest_time_balanced_purged.csv"
        ),
    )
    parser.add_argument(
        "--embeddings",
        default=(
            "experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/"
            "traqid_paper_resnet50_front_rear_concat_gap_features.npy"
        ),
    )
    parser.add_argument("--background-csv", required=True)
    parser.add_argument("--output-dir", required=True)
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

    canonical = canonicalize_sequences(pd.read_csv(args.sequence_manifest))
    canonical["target_sample_id"] = canonical["target_row_id"].astype(str)
    canonical["date"] = canonical["date"].astype(str)
    if canonical["sequence_id"].duplicated().any():
        raise ValueError("Sequence IDs must be unique")

    embeddings = np.load(args.embeddings, mmap_mode="r")
    target_rows = canonical["target_row_id"].to_numpy(dtype=int)
    if target_rows.min() < 0 or target_rows.max() >= len(embeddings):
        raise ValueError("Target row IDs exceed the embedding array")

    background = pd.read_csv(args.background_csv)
    background["sample_id"] = background["sample_id"].astype(str)
    required = {"sample_id", "background_status", *ATMOSPHERIC_FEATURES}
    missing = sorted(required - set(background.columns))
    if missing:
        raise ValueError(f"Atmospheric table is missing: {missing}")
    if background["sample_id"].duplicated().any():
        raise ValueError("Atmospheric sample IDs must be unique")
    if not background["background_status"].eq("success").all():
        raise ValueError("Atmospheric table contains unsuccessful rows")
    canonical = canonical.merge(
        background[["sample_id", *ATMOSPHERIC_FEATURES]],
        left_on="target_sample_id",
        right_on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if canonical[ATMOSPHERIC_FEATURES].isna().all(axis=1).any():
        raise ValueError("Some sequence targets lack every atmospheric feature")

    folds = make_grouped_outer_folds(
        canonical,
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

        fold_target_rows = fold["target_row_id"].to_numpy(dtype=int)
        image_train = np.asarray(
            embeddings[fold_target_rows[train]], dtype=np.float32,
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
                np.asarray(embeddings[fold_target_rows[validation]], dtype=np.float32)
            ),
            "test": pca.transform(
                np.asarray(embeddings[fold_target_rows[test]], dtype=np.float32)
            ),
        }

        atmospheric_imputer = SimpleImputer(strategy="median")
        atmospheric_by_split = {
            "train": atmospheric_imputer.fit_transform(
                fold.loc[train, ATMOSPHERIC_FEATURES]
            ),
            "val": atmospheric_imputer.transform(
                fold.loc[validation, ATMOSPHERIC_FEATURES]
            ),
            "test": atmospheric_imputer.transform(
                fold.loc[test, ATMOSPHERIC_FEATURES]
            ),
        }
        targets = {
            "train": fold.loc[train, "target_PM2.5"].to_numpy(dtype=float),
            "val": fold.loc[validation, "target_PM2.5"].to_numpy(dtype=float),
            "test": fold.loc[test, "target_PM2.5"].to_numpy(dtype=float),
        }

        matrices = {
            "image_only": image_by_split,
            "atmosphere_only": atmospheric_by_split,
            "early_fusion": {
                split: np.concatenate(
                    [image_by_split[split], atmospheric_by_split[split]],
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
                    "test_dates": "|".join(
                        sorted(fold.loc[test, "date"].unique())
                    ),
                    "variant": variant,
                    "n_train": int(train.sum()),
                    "n_validation": int(validation.sum()),
                    "n_test": int(test.sum()),
                    "pca_components": n_components if variant != "atmosphere_only" else 0,
                    "atmospheric_features": (
                        len(ATMOSPHERIC_FEATURES) if variant != "image_only" else 0
                    ),
                    **metrics(targets["test"], prediction),
                    "interval_alpha": args.interval_alpha,
                    "interval_radius": radius,
                    "interval_coverage": float(np.mean(covered)),
                }
            )
            prediction_tables.append(
                pd.DataFrame(
                    {
                        "sequence_id": fold.loc[test, "sequence_id"].to_numpy(),
                        "date": fold.loc[test, "date"].to_numpy(),
                        "fold": fold_number,
                        "variant": variant,
                        "actual_PM2.5": targets["test"],
                        "predicted_PM2.5": prediction,
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
        corresponding = by_fold[by_fold["variant"].eq(variant)]
        aggregate_rows.append(
            {
                "variant": variant,
                "n_test": len(part),
                **metrics(
                    part["actual_PM2.5"].to_numpy(dtype=float),
                    part["predicted_PM2.5"].to_numpy(dtype=float),
                ),
                "mean_fold_r2": float(corresponding["r2"].mean()),
                "mean_fold_rmse": float(corresponding["rmse"].mean()),
                "worst_fold_rmse": float(corresponding["rmse"].max()),
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
        "dataset": "TRAQID",
        "experiment": "image plus atmospheric-feature early fusion",
        "target": "roadside PM2.5",
        "pure_atmospheric_target_available": False,
        "source_specific_outputs": False,
        "protocol": "complete-date outer holdout",
        "input_image": "target-frame front/rear ResNet50 concatenated embedding",
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
            "This tests predictive multimodal fusion. Because the target is a "
            "roadside sensor, the prediction must not be called pure "
            "atmospheric PM2.5 or chemical source apportionment."
        ),
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    print("\nTRAQID IMAGE + ATMOSPHERIC FUSION\n")
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
