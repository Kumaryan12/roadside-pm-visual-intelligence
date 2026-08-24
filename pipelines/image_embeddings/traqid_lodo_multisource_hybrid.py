"""Leakage-safe TRAQID multi-source background and modality study.

The runner evaluates the background-decomposition hypothesis under either
leave-one-date-out (LODO) or leave-one-season-out transfer.  Every fitted
transform and estimator is restricted to the outer-training dates.  One
complete training date is reserved for background/model/fusion selection; the
outer-test date or season is never used for selection.

The study includes:

* training-mean, raw MERRA-2, direct-model and background-only baselines;
* validation-selected CAMS/MERRA-2/local-reference background calibration;
* local-increment models using nonvisual, structured and image features;
* matched incremental-vision comparisons; and
* a validation-selected three-branch convex hybrid.

The local reference monitor is an inference-time PM2.5 input.  Results using it
must therefore be described as reference-assisted, not camera-only prediction.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipelines.image_embeddings.traqid_final_architecture_random_diagnostic import (
    ATMOSPHERIC_FEATURES,
    build_multiscale_features,
)


TARGET = "PM2.5"
MERRA = "background_merra2_pm25_ug_m3"
CAMS = "background_cams_pm25_ug_m3"
REFERENCE = "reference_pm25_ug_m3"
SEASON_COLUMNS = {
    "Monsoon": "lag0__Season_Monsoon",
    "Winter": "lag0__Season_Winter",
    "Summer": "lag0__Season_Summer",
}
NONVISUAL_PREFIXES = (
    "t7__Temperature__",
    "t7__Humidity__",
    "t7__Day_or_Night_",
    "t7__Season_",
    "background_",
    "atmospheric_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engineered-table",
        default=(
            "experiments/traqid_pretraining_v1/data/processed/"
            "traqid_engineered_T7_sequence_table.csv"
        ),
    )
    parser.add_argument(
        "--background-csv",
        default="artifacts/runs/traqid_atmospheric_background_v1/background_hourly.csv",
    )
    parser.add_argument(
        "--local-reference-csv",
        default=(
            "artifacts/runs/traqid_local_reference_openaq_v1/"
            "local_reference_pm25.csv"
        ),
    )
    parser.add_argument(
        "--embeddings",
        default=(
            "experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/"
            "traqid_paper_resnet50_front_rear_mean_gap_features.npy"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["lodo", "season_transfer"],
        default=["lodo", "season_transfer"],
    )
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--background-trees", type=int, default=300)
    parser.add_argument("--background-min-samples-leaf", type=int, default=20)
    parser.add_argument("--background-ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--allow-target-fitted-background",
        action="store_true",
        help=(
            "Diagnostic only: allow supervised ridge/ExtraTrees estimates to be "
            "selected as the decomposition background. By default, B is restricted "
            "to target-independent external PM products."
        ),
    )
    parser.add_argument(
        "--minimum-reference-coverage",
        type=float,
        default=0.95,
        help=(
            "Minimum local-reference coverage required in both validation and test "
            "partitions before the local-reference background is eligible."
        ),
    )
    parser.add_argument("--pca-components", type=int, default=48)
    parser.add_argument("--simplex-step", type=float, default=0.05)
    parser.add_argument("--inner-background-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--max-splits", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "bias": float(np.mean(prediction - truth)),
    }


def date_balanced_weights(dates: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(dates).astype(str))
    counts = series.value_counts()
    weights = series.map(1.0 / counts).to_numpy(dtype=float)
    return weights * len(weights) / weights.sum()


def background_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[ATMOSPHERIC_FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    reference = pd.to_numeric(frame[REFERENCE], errors="coerce")
    result[REFERENCE] = reference
    result["reference_available"] = reference.notna().astype(float)
    result["cams_merra_mean"] = 0.5 * (result[CAMS] + result[MERRA])
    result["cams_minus_merra"] = result[CAMS] - result[MERRA]
    result["reference_minus_merra"] = reference - result[MERRA]
    result["reference_minus_cams"] = reference - result[CAMS]
    return result


def background_estimator(kind: str, args: argparse.Namespace, seed: int) -> Pipeline:
    if kind == "ridge_calibrated":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=args.background_ridge_alpha)),
            ]
        )
    if kind == "extra_trees_calibrated":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=args.background_trees,
                        min_samples_leaf=args.background_min_samples_leaf,
                        max_features=0.8,
                        random_state=seed,
                        n_jobs=args.jobs,
                    ),
                ),
            ]
        )
    raise ValueError(kind)


def fit_with_weights(model: Pipeline, features: pd.DataFrame | np.ndarray,
                     target: np.ndarray, weights: np.ndarray) -> None:
    final_name = model.steps[-1][0]
    model.fit(features, target, **{f"{final_name}__sample_weight": weights})


def raw_background(kind: str, frame: pd.DataFrame) -> np.ndarray:
    merra = pd.to_numeric(frame[MERRA], errors="coerce").to_numpy(dtype=float)
    cams = pd.to_numeric(frame[CAMS], errors="coerce").to_numpy(dtype=float)
    reference = pd.to_numeric(frame[REFERENCE], errors="coerce").to_numpy(dtype=float)
    mean = 0.5 * (merra + cams)
    if kind == "raw_merra2":
        return merra
    if kind == "raw_cams":
        return cams
    if kind == "raw_cams_merra_mean":
        return mean
    if kind == "local_reference_fallback":
        return np.where(np.isfinite(reference), reference, mean)
    raise ValueError(kind)


def calibrated_background_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, pd.DataFrame]:
    """Select background on validation and return OOF train/val/test values."""
    external_candidates = [
        "raw_merra2",
        "raw_cams",
        "raw_cams_merra_mean",
        "local_reference_fallback",
    ]
    fitted = {"ridge_calibrated", "extra_trees_calibrated"}
    candidates = list(external_candidates)
    if args.allow_target_fitted_background:
        candidates.extend(sorted(fitted))
    x_train = background_matrix(train)
    x_validation = background_matrix(validation)
    x_test = background_matrix(test)
    y_train = train[TARGET].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    groups = train["date"].astype(str).to_numpy()
    weights = date_balanced_weights(groups)
    candidate_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    audit_rows: list[dict[str, object]] = []
    validation_reference_coverage = float(validation[REFERENCE].notna().mean())
    test_reference_coverage = float(test[REFERENCE].notna().mean())

    for candidate in candidates:
        if candidate not in fitted:
            train_prediction = raw_background(candidate, train)
            validation_prediction = raw_background(candidate, validation)
            test_prediction = raw_background(candidate, test)
        else:
            unique_groups = np.unique(groups)
            folds = min(args.inner_background_folds, len(unique_groups))
            if folds < 2:
                raise ValueError("Background cross-fitting requires two training dates")
            train_prediction = np.full(len(train), np.nan, dtype=float)
            for inner, (fit, held) in enumerate(
                GroupKFold(n_splits=folds).split(x_train, groups=groups), start=1
            ):
                model = background_estimator(
                    candidate, args, split_seed + 1000 + inner
                )
                fit_with_weights(
                    model,
                    x_train.iloc[fit],
                    y_train[fit],
                    weights[fit],
                )
                train_prediction[held] = model.predict(x_train.iloc[held])
            model = background_estimator(candidate, args, split_seed + 1099)
            fit_with_weights(model, x_train, y_train, weights)
            validation_prediction = model.predict(x_validation)
            test_prediction = model.predict(x_test)
        if not all(
            np.isfinite(values).all()
            for values in (train_prediction, validation_prediction, test_prediction)
        ):
            raise RuntimeError(f"Non-finite background prediction: {candidate}")
        candidate_predictions[candidate] = (
            train_prediction,
            validation_prediction,
            test_prediction,
        )
        eligible = True
        exclusion_reason = ""
        if candidate == "local_reference_fallback" and min(
            validation_reference_coverage, test_reference_coverage
        ) < args.minimum_reference_coverage:
            eligible = False
            exclusion_reason = "insufficient_validation_or_test_reference_coverage"
        audit_rows.append(
            {
                "background_candidate": candidate,
                "target_fitted": candidate in fitted,
                "eligible_for_selection": eligible,
                "exclusion_reason": exclusion_reason,
                "validation_reference_coverage": validation_reference_coverage,
                "test_reference_coverage": test_reference_coverage,
                **{
                    f"validation_{key}": value
                    for key, value in metrics(
                        y_validation, validation_prediction
                    ).items()
                },
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(
        ["validation_rmse", "validation_mae"]
    )
    eligible_audit = audit[audit["eligible_for_selection"]]
    if eligible_audit.empty:
        raise RuntimeError("No eligible external background candidate")
    selected = str(eligible_audit.iloc[0]["background_candidate"])
    train_prediction, validation_prediction, test_prediction = (
        candidate_predictions[selected]
    )
    return (
        train_prediction,
        validation_prediction,
        test_prediction,
        selected,
        audit,
    )


def image_sequence_features(
    embeddings: np.ndarray,
    train: pd.DataFrame,
    partitions: list[pd.DataFrame],
    components: int,
    seed: int,
) -> list[np.ndarray]:
    train_rows = np.unique(
        np.concatenate(
            [
                np.fromstring(value, sep="|", dtype=int)
                for value in train["seq_row_ids"].astype(str)
            ]
        )
    )
    n_components = min(components, len(train_rows) - 1, embeddings.shape[1])
    if n_components < 1:
        raise ValueError("Not enough training rows for image PCA")
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
    )
    pca.fit(np.asarray(embeddings[train_rows], dtype=np.float32))
    projected = pca.transform(np.asarray(embeddings, dtype=np.float32))
    outputs: list[np.ndarray] = []
    for frame in partitions:
        indices = np.vstack(
            [
                np.fromstring(value, sep="|", dtype=int)
                for value in frame["seq_row_ids"].astype(str)
            ]
        )
        values = projected[indices]
        outputs.append(
            np.concatenate(
                [
                    values[:, -1, :],
                    values.mean(axis=1),
                    values.std(axis=1),
                    values[:, -1, :] - values[:, 0, :],
                ],
                axis=1,
            ).astype(np.float32)
        )
    return outputs


def prepare_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    image: np.ndarray | None = None,
) -> np.ndarray:
    tabular = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    if image is None:
        return tabular
    return np.concatenate([tabular, image], axis=1)


def fit_local_model(
    x_train: np.ndarray,
    target: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    train_dates: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=args.trees,
                    min_samples_leaf=args.min_samples_leaf,
                    max_features=0.7,
                    random_state=seed,
                    n_jobs=args.jobs,
                ),
            ),
        ]
    )
    fit_with_weights(
        model,
        x_train,
        np.asarray(target, dtype=float),
        date_balanced_weights(train_dates),
    )
    return model.predict(x_validation), model.predict(x_test)


def simplex_weights(step: float) -> list[np.ndarray]:
    if not 0.0 < step <= 1.0:
        raise ValueError("--simplex-step must be in (0, 1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--simplex-step must divide one exactly")
    return [
        np.asarray([a, b, units - a - b], dtype=float) / units
        for a, b in product(range(units + 1), repeat=2)
        if a + b <= units
    ]


def select_hybrid(
    truth: np.ndarray,
    validation_predictions: list[np.ndarray],
    test_predictions: list[np.ndarray],
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validation_stack = np.column_stack(validation_predictions)
    test_stack = np.column_stack(test_predictions)
    best_weight: np.ndarray | None = None
    best_rmse = np.inf
    for weight in simplex_weights(step):
        prediction = validation_stack @ weight
        rmse = mean_squared_error(truth, prediction) ** 0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = weight
    assert best_weight is not None
    return validation_stack @ best_weight, test_stack @ best_weight, best_weight


def derive_season(engineered: pd.DataFrame) -> pd.Series:
    values = engineered[list(SEASON_COLUMNS.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    reverse = {column: season for season, column in SEASON_COLUMNS.items()}
    return values.idxmax(axis=1).map(reverse)


def build_splits(frame: pd.DataFrame, protocols: list[str]) -> list[dict[str, object]]:
    dates = sorted(frame["date"].astype(str).unique())
    splits: list[dict[str, object]] = []
    if "lodo" in protocols:
        for index, test_date in enumerate(dates):
            validation_date = dates[(index + 1) % len(dates)]
            splits.append(
                {
                    "protocol": "lodo",
                    "split_id": test_date,
                    "test_groups": [test_date],
                    "validation_groups": [validation_date],
                    "test_mask": frame["date"].eq(test_date).to_numpy(),
                    "validation_mask": frame["date"].eq(validation_date).to_numpy(),
                }
            )
    if "season_transfer" in protocols:
        for index, test_season in enumerate(sorted(frame["season"].unique())):
            remaining_dates = sorted(
                frame.loc[frame["season"].ne(test_season), "date"].unique()
            )
            validation_date = remaining_dates[index % len(remaining_dates)]
            splits.append(
                {
                    "protocol": "season_transfer",
                    "split_id": test_season,
                    "test_groups": [test_season],
                    "validation_groups": [validation_date],
                    "test_mask": frame["season"].eq(test_season).to_numpy(),
                    "validation_mask": frame["date"].eq(validation_date).to_numpy(),
                }
            )
    return splits


def run_split(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    split: dict[str, object],
    candidate_features: list[str],
    args: argparse.Namespace,
    split_number: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    test_mask = np.asarray(split["test_mask"], dtype=bool)
    validation_mask = np.asarray(split["validation_mask"], dtype=bool)
    train_mask = ~(test_mask | validation_mask)
    train = frame.loc[train_mask].reset_index(drop=True)
    validation = frame.loc[validation_mask].reset_index(drop=True)
    test = frame.loc[test_mask].reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(f"Empty partition in split {split['split_id']}")

    train_features = frame.loc[train_mask, candidate_features]
    usable = [
        column
        for column in candidate_features
        if train_features[column].nunique(dropna=True) > 1
    ]
    nonvisual = [
        column for column in usable if column.startswith(NONVISUAL_PREFIXES)
    ]
    structured = [column for column in usable if column not in nonvisual]
    if not nonvisual or not structured:
        raise ValueError("Feature family separation produced an empty group")

    seed = args.seed + 10000 * split_number
    b_train, b_validation, b_test, selected_background, background_audit = (
        calibrated_background_predictions(train, validation, test, args, seed)
    )
    image_train, image_validation, image_test = image_sequence_features(
        embeddings,
        train,
        [train, validation, test],
        args.pca_components,
        seed + 500,
    )
    y_train = train[TARGET].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)
    local_target = y_train - b_train

    matrices = {
        "nonvisual_local": (
            prepare_matrix(train, nonvisual),
            prepare_matrix(validation, nonvisual),
            prepare_matrix(test, nonvisual),
        ),
        "nonvisual_plus_images_local": (
            prepare_matrix(train, nonvisual, image_train),
            prepare_matrix(validation, nonvisual, image_validation),
            prepare_matrix(test, nonvisual, image_test),
        ),
        "nonvisual_plus_structured_local": (
            prepare_matrix(train, usable),
            prepare_matrix(validation, usable),
            prepare_matrix(test, usable),
        ),
        "full_local": (
            prepare_matrix(train, usable, image_train),
            prepare_matrix(validation, usable, image_validation),
            prepare_matrix(test, usable, image_test),
        ),
    }
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for model_index, (name, (x_train, x_validation, x_test)) in enumerate(
        matrices.items(), start=1
    ):
        validation_increment, test_increment = fit_local_model(
            x_train,
            local_target,
            x_validation,
            x_test,
            train["date"].to_numpy(),
            args,
            seed + model_index,
        )
        validation_predictions[name] = b_validation + validation_increment
        test_predictions[name] = b_test + test_increment

    # Matched direct tabular model; no external decomposition target.
    direct_validation, direct_test = fit_local_model(
        matrices["nonvisual_plus_structured_local"][0],
        y_train,
        matrices["nonvisual_plus_structured_local"][1],
        matrices["nonvisual_plus_structured_local"][2],
        train["date"].to_numpy(),
        args,
        seed + 100,
    )
    validation_predictions["direct_tabular"] = direct_validation
    test_predictions["direct_tabular"] = direct_test

    # Raw-MERRA decomposition uses the same structured predictors.
    merra_train = raw_background("raw_merra2", train)
    merra_validation = raw_background("raw_merra2", validation)
    merra_test = raw_background("raw_merra2", test)
    merra_validation_increment, merra_test_increment = fit_local_model(
        matrices["nonvisual_plus_structured_local"][0],
        y_train - merra_train,
        matrices["nonvisual_plus_structured_local"][1],
        matrices["nonvisual_plus_structured_local"][2],
        train["date"].to_numpy(),
        args,
        seed + 101,
    )
    validation_predictions["raw_merra2_plus_local"] = (
        merra_validation + merra_validation_increment
    )
    test_predictions["raw_merra2_plus_local"] = merra_test + merra_test_increment

    validation_predictions["training_mean"] = np.full(
        len(validation), np.average(y_train, weights=date_balanced_weights(train["date"]))
    )
    test_predictions["training_mean"] = np.full(
        len(test), validation_predictions["training_mean"][0]
    )
    validation_predictions["selected_background_only"] = b_validation
    test_predictions["selected_background_only"] = b_test
    validation_predictions["raw_merra2_only"] = merra_validation
    test_predictions["raw_merra2_only"] = merra_test

    hybrid_branches = [
        "nonvisual_local",
        "nonvisual_plus_images_local",
        "nonvisual_plus_structured_local",
    ]
    _, hybrid_test, hybrid_weight = select_hybrid(
        y_validation,
        [validation_predictions[name] for name in hybrid_branches],
        [test_predictions[name] for name in hybrid_branches],
        args.simplex_step,
    )
    test_predictions["validation_selected_hybrid"] = hybrid_test
    test_predictions["fixed_equal_hybrid_diagnostic"] = np.mean(
        np.column_stack([test_predictions[name] for name in hybrid_branches]),
        axis=1,
    )

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    selected_background_target_fitted = selected_background in {
        "ridge_calibrated",
        "extra_trees_calibrated",
    }
    for method, prediction in test_predictions.items():
        prediction_rows.append(
            pd.DataFrame(
                {
                    "protocol": split["protocol"],
                    "split_id": split["split_id"],
                    "sequence_id": test["sequence_id"].to_numpy(),
                    "date": test["date"].to_numpy(),
                    "season": test["season"].to_numpy(),
                    "method": method,
                    "actual_PM2.5": y_test,
                    "predicted_PM2.5": prediction,
                    "selected_background": selected_background,
                    "selected_background_target_fitted": (
                        selected_background_target_fitted
                    ),
                }
            )
        )
        metric_rows.append(
            {
                "protocol": split["protocol"],
                "split_id": split["split_id"],
                "test_groups": "|".join(split["test_groups"]),
                "validation_groups": "|".join(split["validation_groups"]),
                "method": method,
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
                "selected_background": selected_background,
                "selected_background_target_fitted": (
                    selected_background_target_fitted
                ),
                **metrics(y_test, prediction),
            }
        )
    metadata = {
        "protocol": split["protocol"],
        "split_id": split["split_id"],
        "test_groups": split["test_groups"],
        "validation_groups": split["validation_groups"],
        "train_dates": sorted(train["date"].unique()),
        "selected_background": selected_background,
        "selected_background_target_fitted": selected_background_target_fitted,
        "hybrid_branch_order": hybrid_branches,
        "hybrid_weights": hybrid_weight.tolist(),
        "usable_tabular_features": len(usable),
        "nonvisual_features": len(nonvisual),
        "structured_features": len(structured),
        "image_features_after_aggregation": int(image_train.shape[1]),
        "test_reference_coverage": float(test[REFERENCE].notna().mean()),
    }
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows), {
        "metadata": metadata,
        "background_audit": background_audit,
    }


def aggregate_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict[str, object]] = []
    date_rows: list[dict[str, object]] = []
    for (protocol, method), group in predictions.groupby(["protocol", "method"]):
        y = group["actual_PM2.5"].to_numpy()
        p = group["predicted_PM2.5"].to_numpy()
        per_date = []
        for date, date_group in group.groupby("date"):
            row = {
                "protocol": protocol,
                "method": method,
                "date": date,
                "season": date_group["season"].iloc[0],
                "n_test": len(date_group),
                **metrics(
                    date_group["actual_PM2.5"].to_numpy(),
                    date_group["predicted_PM2.5"].to_numpy(),
                ),
            }
            date_rows.append(row)
            per_date.append(row)
        aggregate_rows.append(
            {
                "protocol": protocol,
                "method": method,
                "n_test": len(group),
                **metrics(y, p),
                "mean_date_r2": float(np.mean([row["r2"] for row in per_date])),
                "median_date_r2": float(np.median([row["r2"] for row in per_date])),
                "positive_date_r2_count": int(sum(row["r2"] > 0 for row in per_date)),
                "dates_evaluated": len(per_date),
                "mean_date_mae": float(np.mean([row["mae"] for row in per_date])),
                "mean_date_rmse": float(np.mean([row["rmse"] for row in per_date])),
                "worst_date_rmse": float(np.max([row["rmse"] for row in per_date])),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["protocol", "rmse", "mae"]
    )
    by_date = pd.DataFrame(date_rows).sort_values(["protocol", "method", "date"])
    return aggregate, by_date


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    engineered = pd.read_csv(args.engineered_table, low_memory=False)
    metadata = pd.DataFrame(
        {
            "date": pd.to_datetime(
                engineered["target_created_at"], errors="raise"
            ).dt.date.astype(str),
            "season": derive_season(engineered),
        },
        index=engineered.index,
    )
    engineered = pd.concat([engineered, metadata], axis=1).copy()
    if engineered["sequence_id"].duplicated().any():
        raise ValueError("Engineered sequence IDs must be unique")

    background = pd.read_csv(args.background_csv)
    background["sample_id"] = background["sample_id"].astype(str)
    if background["sample_id"].duplicated().any():
        raise ValueError("Atmospheric sample IDs must be unique")
    background_indexed = background.set_index("sample_id")
    multiscale, candidate_features, feature_audit = build_multiscale_features(
        engineered, background_indexed
    )
    frame = engineered.merge(multiscale, on="sequence_id", validate="one_to_one")
    target_ids = frame["target_row_id"].astype(str)

    reference = pd.read_csv(args.local_reference_csv)
    reference["sample_id"] = reference["sample_id"].astype(str)
    reference = reference.drop_duplicates("sample_id").set_index("sample_id")
    reference_values = pd.to_numeric(reference[REFERENCE], errors="coerce")
    reference_status = reference["reference_status"].astype(str).eq("success")
    reference_values = reference_values.where(reference_status)
    frame = pd.concat(
        [
            frame,
            target_ids.map(reference_values).rename(REFERENCE),
        ],
        axis=1,
    ).copy()
    frame["seq_row_ids"] = frame["seq_row_ids"].astype(str)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")

    embeddings = np.load(args.embeddings, mmap_mode="r")
    if frame["target_row_id"].max() >= len(embeddings):
        raise ValueError("Embedding array does not cover every target row")
    splits = build_splits(frame, args.protocols)
    if args.max_splits:
        splits = splits[: args.max_splits]

    design = {
        "dataset": "TRAQID",
        "rows": len(frame),
        "dates": sorted(frame["date"].unique()),
        "seasons": {
            season: sorted(group["date"].unique())
            for season, group in frame.groupby("season")
        },
        "protocols": args.protocols,
        "planned_splits": len(splits),
        "background_candidates": [
            "raw_merra2",
            "raw_cams",
            "raw_cams_merra_mean",
            "local_reference_fallback",
        ]
        + (
            ["ridge_calibrated", "extra_trees_calibrated"]
            if args.allow_target_fitted_background
            else []
        ),
        "background_selection_policy": (
            "target_fitted_diagnostic"
            if args.allow_target_fitted_background
            else "external_target_independent_only"
        ),
        "minimum_reference_coverage": args.minimum_reference_coverage,
        "local_reference_is_inference_time_pm_input": True,
        "outer_test_targets_used_for_selection": False,
        "random_or_overlapping_split_used": False,
        "feature_audit": feature_audit,
        "seed": args.seed,
    }
    (output / "study_design.json").write_text(json.dumps(design, indent=2))
    if args.dry_run:
        print(json.dumps(design, indent=2))
        return 0

    prediction_tables: list[pd.DataFrame] = []
    split_metric_tables: list[pd.DataFrame] = []
    for split_number, split in enumerate(splits, start=1):
        safe_id = str(split["split_id"]).replace("/", "-")
        split_dir = output / "splits" / f"{split['protocol']}__{safe_id}"
        split_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = split_dir / "predictions.csv"
        metric_path = split_dir / "metrics.csv"
        if args.resume and prediction_path.exists() and metric_path.exists():
            print(f"Resume {split_number}/{len(splits)}: {split_dir.name}")
            prediction_tables.append(pd.read_csv(prediction_path))
            split_metric_tables.append(pd.read_csv(metric_path))
            continue
        print(
            f"TRAQID split {split_number}/{len(splits)} | "
            f"{split['protocol']} test={split['split_id']}"
        )
        predictions, split_metrics, audit = run_split(
            frame,
            embeddings,
            split,
            candidate_features,
            args,
            split_number,
        )
        predictions.to_csv(prediction_path, index=False)
        split_metrics.to_csv(metric_path, index=False)
        audit["background_audit"].to_csv(
            split_dir / "background_selection.csv", index=False
        )
        (split_dir / "metadata.json").write_text(
            json.dumps(audit["metadata"], indent=2)
        )
        prediction_tables.append(predictions)
        split_metric_tables.append(split_metrics)

    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    all_split_metrics = pd.concat(split_metric_tables, ignore_index=True)
    aggregate, by_date = aggregate_predictions(all_predictions)
    all_predictions.to_csv(output / "predictions.csv", index=False)
    all_split_metrics.to_csv(output / "metrics_by_split.csv", index=False)
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    by_date.to_csv(output / "metrics_by_date.csv", index=False)

    lodo = aggregate[aggregate["protocol"].eq("lodo")].set_index("method")
    incremental_rows = []
    for base, augmented, addition in [
        ("nonvisual_local", "nonvisual_plus_images_local", "images"),
        (
            "nonvisual_plus_structured_local",
            "full_local",
            "images_given_structured_visual",
        ),
        ("nonvisual_local", "nonvisual_plus_structured_local", "structured_visual"),
    ]:
        if base in lodo.index and augmented in lodo.index:
            incremental_rows.append(
                {
                    "base": base,
                    "augmented": augmented,
                    "addition": addition,
                    "delta_mae": float(lodo.loc[augmented, "mae"] - lodo.loc[base, "mae"]),
                    "delta_rmse": float(lodo.loc[augmented, "rmse"] - lodo.loc[base, "rmse"]),
                    "delta_pooled_r2": float(lodo.loc[augmented, "r2"] - lodo.loc[base, "r2"]),
                    "delta_mean_date_r2": float(
                        lodo.loc[augmented, "mean_date_r2"]
                        - lodo.loc[base, "mean_date_r2"]
                    ),
                    "delta_positive_dates": int(
                        lodo.loc[augmented, "positive_date_r2_count"]
                        - lodo.loc[base, "positive_date_r2_count"]
                    ),
                }
            )
    pd.DataFrame(incremental_rows).to_csv(
        output / "incremental_modality_effects.csv", index=False
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                **design,
                "completed_splits": len(splits),
                "outputs": [
                    "metrics_aggregate.csv",
                    "metrics_by_date.csv",
                    "metrics_by_split.csv",
                    "incremental_modality_effects.csv",
                    "predictions.csv",
                ],
            },
            indent=2,
        )
    )
    print("\nTRAQID MULTI-SOURCE LODO / SEASON-TRANSFER RESULTS")
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
