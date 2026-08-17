"""Run the final background-decomposed TRAQID architecture on random windows.

This is an intentionally leakage-contaminated diagnostic.  It reuses the
MERRA-2 ResNet50--GRU T=7 branch trained by
``traqid_random_background_comparison.py`` and adds the components used by the
final architecture:

* a cross-fitted Random Forest correction of the temporal branch;
* an enhanced T=7 multiscale ExtraTrees tabular local-increment branch;
* validation-selected global and conditional convex fusion; and
* validation-residual conformal diagnostics.

The train/validation/test assignment is made after overlapping windows have
been constructed.  Consequently, the resulting scores must never be reported
as unseen-date or deployment generalization.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from pipelines.image_embeddings.traqid_random_background_comparison import (
    overlap_audit,
    residual_columns,
)


BACKGROUND = "background_merra2_pm25_ug_m3"
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
INTERACTION_BASES = [
    "total_vehicle_count",
    "heavy_vehicle_count",
    "near_vehicle_count",
    "near_heavy_vehicle_count",
    "total_vehicle_area_ratio",
    "heavy_vehicle_area_ratio",
    "near_vehicle_area_ratio",
    "truck_count",
    "bus_count",
    "auto_count",
]
ROAD_DUST_BASES = [
    "road_brown_pixel_ratio",
    "road_gray_dry_pixel_ratio",
    "road_area_ratio",
]
FORBIDDEN_TOKENS = ("pm2.5", "pm10", "aqi", "date_fold", "row_id", "image_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-manifest",
        default=(
            "artifacts/runs/traqid_T7_background_random_comparison_v1/"
            "merra2/sequence_manifest_random.csv"
        ),
    )
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
        "--temporal-run-dir",
        default=(
            "artifacts/runs/traqid_T7_background_random_comparison_v1/"
            "merra2/gru_T7"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--temporal-min-leaf", type=int, default=5)
    parser.add_argument("--gate-min-leaf", type=int, default=20)
    parser.add_argument("--interval-alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def metric_row(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
        "bias": float(np.mean(prediction - truth)),
    }


def estimator(kind: str, args: argparse.Namespace, seed: int) -> Pipeline:
    if kind == "temporal_correction":
        model = RandomForestRegressor(
            n_estimators=args.trees,
            min_samples_leaf=args.temporal_min_leaf,
            max_features=0.7,
            random_state=seed,
            n_jobs=args.jobs,
        )
    elif kind == "tabular":
        model = ExtraTreesRegressor(
            n_estimators=args.trees,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=seed,
            n_jobs=args.jobs,
        )
    elif kind == "gate":
        model = ExtraTreesRegressor(
            n_estimators=args.trees,
            min_samples_leaf=args.gate_min_leaf,
            max_features=0.7,
            random_state=seed,
            n_jobs=args.jobs,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def build_multiscale_features(
    engineered: pd.DataFrame, background: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """Reproduce the final TRAQID T=7 enhanced feature construction."""
    if engineered["sequence_id"].duplicated().any():
        raise ValueError("Engineered sequence_id must be unique")
    lag_groups: dict[str, dict[int, str]] = {}
    for column in engineered.columns:
        match = re.fullmatch(r"lag(\d+)__(.+)", column)
        if not match:
            continue
        lag, base = int(match.group(1)), match.group(2)
        lower = base.lower()
        if any(token in lower for token in FORBIDDEN_TOKENS):
            continue
        lag_groups.setdefault(base, {})[lag] = column

    feature_data: dict[str, np.ndarray] = {
        "sequence_id": engineered["sequence_id"].to_numpy()
    }
    summarized: list[str] = []
    for base, lag_columns in sorted(lag_groups.items()):
        if sorted(lag_columns) != list(range(7)):
            continue
        ordered = [lag_columns[lag] for lag in range(7)]
        values = engineered[ordered].apply(pd.to_numeric, errors="coerce")
        prefix = f"t7__{base}"
        feature_data[f"{prefix}__latest"] = values.iloc[:, 0].to_numpy()
        feature_data[f"{prefix}__mean"] = values.mean(axis=1).to_numpy()
        feature_data[f"{prefix}__std"] = values.std(axis=1, ddof=0).to_numpy()
        feature_data[f"{prefix}__min"] = values.min(axis=1).to_numpy()
        feature_data[f"{prefix}__max"] = values.max(axis=1).to_numpy()
        feature_data[f"{prefix}__change"] = (
            values.iloc[:, 0] - values.iloc[:, -1]
        ).to_numpy()
        summarized.append(base)

    target_ids = engineered["target_row_id"].astype(str)
    atmospheric = background.reindex(target_ids)
    missing = atmospheric[ATMOSPHERIC_FEATURES].isna().all(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} sequences lack atmospheric context")
    for column in ATMOSPHERIC_FEATURES:
        feature_data[column] = pd.to_numeric(
            atmospheric[column], errors="coerce"
        ).to_numpy()

    result = pd.DataFrame(feature_data)
    hour = result["background_cams_forecast_hour"].to_numpy(dtype=float)
    derived: dict[str, np.ndarray] = {
        "atmospheric_hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
        "atmospheric_hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
    }
    wind = result["background_era5_wind_speed_10m_m_s"].clip(lower=0.0)
    inverse_dispersion = 1.0 / (wind + 0.5)
    rain = result["background_era5_precipitation_hourly_mm"].clip(lower=0.0)
    dry_factor = 1.0 / (1.0 + rain)
    for base in INTERACTION_BASES:
        column = f"t7__{base}__mean"
        if column in result:
            derived[f"physics__{base}__inverse_wind"] = (
                result[column] * inverse_dispersion
            ).to_numpy()
            derived[f"physics__{base}__wind_ventilation"] = (
                result[column] * wind
            ).to_numpy()
    for base in ROAD_DUST_BASES:
        column = f"t7__{base}__mean"
        if column in result:
            derived[f"physics__{base}__dry_windy"] = (
                result[column] * wind * dry_factor
            ).to_numpy()
    derived["atmospheric__cams_minus_merra2_pm25"] = (
        result["background_cams_pm25_ug_m3"]
        - result["background_merra2_pm25_ug_m3"]
    ).to_numpy()
    derived["atmospheric__aerosol_species_sum"] = result[
        [
            "background_merra2_dust25_ug_m3",
            "background_merra2_sea_salt25_ug_m3",
            "background_merra2_black_carbon_ug_m3",
            "background_merra2_organic_carbon_ug_m3",
            "background_merra2_sulphate_ug_m3",
        ]
    ].sum(axis=1, min_count=1).to_numpy()
    result = pd.concat(
        [result, pd.DataFrame(derived, index=result.index)], axis=1
    )
    columns = [column for column in result if column != "sequence_id"]
    audit = {
        "source_lag_bases": len(lag_groups),
        "summarized_lag_bases": len(summarized),
        "atmospheric_features": len(ATMOSPHERIC_FEATURES),
        "candidate_feature_count": len(columns),
        "true_wind_relative_road_angle_used": False,
    }
    return result, columns, audit


def attach_temporal_predictions(
    path: Path, sequences: pd.DataFrame, frame: pd.DataFrame
) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    predicted = [column for column in predictions if column.startswith("predicted_")]
    if len(predicted) != 1:
        raise ValueError(f"Expected one predicted target in {path}: {predicted}")
    return (
        predictions[["sequence_id", predicted[0]]]
        .rename(columns={predicted[0]: "temporal_base_local"})
        .merge(
            sequences[
                [
                    "sequence_id",
                    "date",
                    "balanced_block_id",
                    "target_PM2.5",
                    "split_random",
                ]
            ],
            on="sequence_id",
            validate="one_to_one",
        )
        .merge(frame, on="sequence_id", validate="one_to_one")
    )


def crossfit_regression(
    features: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
    kind: str,
) -> np.ndarray:
    groups = np.asarray(groups).astype(str)
    folds = min(5, len(np.unique(groups)))
    if folds < 2:
        raise ValueError(f"{kind} cross-fitting requires at least two groups")
    prediction = np.full(len(features), np.nan, dtype=float)
    for fold, (fit, held) in enumerate(
        GroupKFold(n_splits=folds).split(features, groups=groups), start=1
    ):
        model = estimator(kind, args, args.seed + fold)
        model.fit(features.iloc[fit], np.asarray(target)[fit])
        prediction[held] = model.predict(features.iloc[held])
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{kind} cross-fitting left missing predictions")
    return prediction


def gate_features(
    frame: pd.DataFrame,
    temporal_prediction: np.ndarray,
    tabular_prediction: np.ndarray,
) -> pd.DataFrame:
    selected = [
        BACKGROUND,
        "background_cams_pm25_ug_m3",
        "background_era5_temperature_2m_c",
        "background_era5_relative_humidity_pct",
        "background_era5_wind_speed_10m_m_s",
        "background_era5_precipitation_hourly_mm",
    ]
    compact_bases = [
        "total_vehicle_count",
        "heavy_vehicle_count",
        "near_vehicle_count",
        "near_heavy_vehicle_count",
        "truck_count",
        "bus_count",
        "auto_count",
        *ROAD_DUST_BASES,
    ]
    for base in compact_bases:
        for statistic in ("latest", "mean", "std", "max", "change"):
            selected.append(f"t7__{base}__{statistic}")
    selected = [column for column in selected if column in frame]
    temporal = np.asarray(temporal_prediction, dtype=float)
    tabular = np.asarray(tabular_prediction, dtype=float)
    branch = pd.DataFrame(
        {
            "gate__temporal_prediction": temporal,
            "gate__tabular_prediction": tabular,
            "gate__branch_mean": 0.5 * (temporal + tabular),
            "gate__signed_disagreement": temporal - tabular,
            "gate__absolute_disagreement": np.abs(temporal - tabular),
        },
        index=frame.index,
    )
    context = frame[selected].apply(pd.to_numeric, errors="coerce").copy()
    context.index = frame.index
    return pd.concat([branch, context], axis=1)


def oracle_weight(
    truth: np.ndarray, temporal: np.ndarray, tabular: np.ndarray
) -> np.ndarray:
    truth = np.asarray(truth, dtype=float)
    temporal = np.asarray(temporal, dtype=float)
    tabular = np.asarray(tabular, dtype=float)
    difference = temporal - tabular
    weight = np.full(len(truth), 0.5, dtype=float)
    informative = np.abs(difference) > 1e-6
    weight[informative] = (
        truth[informative] - tabular[informative]
    ) / difference[informative]
    return np.clip(weight, 0.0, 1.0)


def global_weight(
    truth: np.ndarray, temporal: np.ndarray, tabular: np.ndarray
) -> float:
    difference = np.asarray(temporal) - np.asarray(tabular)
    denominator = float(np.dot(difference, difference))
    if denominator <= 1e-12:
        return 0.5
    numerator = float(np.dot(np.asarray(truth) - np.asarray(tabular), difference))
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def conformal_radius(truth: np.ndarray, prediction: np.ndarray, alpha: float) -> float:
    scores = np.abs(np.asarray(truth) - np.asarray(prediction))
    level = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sequences = pd.read_csv(args.sequence_manifest)
    required = {
        "sequence_id",
        "date",
        "balanced_block_id",
        "target_row_id",
        "target_PM2.5",
        "split_random",
        "seq_row_ids",
    }
    missing = sorted(required - set(sequences))
    if missing:
        raise ValueError(f"Sequence manifest is missing: {missing}")
    if not {"train", "val", "test"}.issubset(set(sequences["split_random"])):
        raise ValueError("split_random must contain train, val and test")

    engineered = pd.read_csv(args.engineered_table)
    background = pd.read_csv(args.background_csv)
    background["sample_id"] = background["sample_id"].astype(str)
    if background["sample_id"].duplicated().any():
        raise ValueError("Background sample_id must be unique")
    background = background.set_index("sample_id")
    multiscale, candidate_features, feature_audit = build_multiscale_features(
        engineered, background
    )
    frame = engineered.merge(multiscale, on="sequence_id", validate="one_to_one")

    train_ids = set(
        sequences.loc[sequences["split_random"].eq("train"), "sequence_id"]
    )
    train_feature_frame = multiscale[multiscale["sequence_id"].isin(train_ids)]
    features = [
        column
        for column in candidate_features
        if train_feature_frame[column].nunique(dropna=True) > 1
    ]
    feature_audit.update(
        {
            "constant_features_removed_using_train_only": len(candidate_features)
            - len(features),
            "final_feature_count": len(features),
        }
    )
    residual_features = residual_columns(engineered)
    audit = overlap_audit(sequences)

    (output / "feature_columns.txt").write_text("\n".join(features) + "\n")
    (output / "temporal_residual_feature_columns.txt").write_text(
        "\n".join(residual_features) + "\n"
    )
    (output / "feature_audit.json").write_text(
        json.dumps(feature_audit, indent=2) + "\n"
    )
    (output / "overlap_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    print(
        f"Prepared {len(sequences):,} sequences; enhanced features={len(features):,}; "
        f"temporal correction features={len(residual_features):,}",
        flush=True,
    )
    print(json.dumps(audit, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; no estimators were fitted.", flush=True)
        return 0

    temporal_dir = Path(args.temporal_run_dir)
    validation = attach_temporal_predictions(
        temporal_dir / "predictions_val.csv", sequences, frame
    )
    test = attach_temporal_predictions(
        temporal_dir / "predictions_test.csv", sequences, frame
    )
    sequence_columns = [
        "sequence_id",
        "date",
        "balanced_block_id",
        "target_PM2.5",
        "split_random",
    ]
    train = sequences.loc[
        sequences["split_random"].eq("train"), sequence_columns
    ].merge(frame, on="sequence_id", validate="one_to_one")

    for part in (train, validation, test):
        part["target_row_id"] = part["target_row_id"].astype(str)
        part[BACKGROUND] = part["target_row_id"].map(background[BACKGROUND])
        if part[BACKGROUND].isna().any():
            raise ValueError("A sequence target lacks the MERRA-2 background")
        part["local_target"] = part["target_PM2.5"] - part[BACKGROUND]

    validation_residual = (
        validation["local_target"].to_numpy(dtype=float)
        - validation["temporal_base_local"].to_numpy(dtype=float)
    )
    validation_residual_oof = crossfit_regression(
        validation[residual_features],
        validation_residual,
        validation["date"].to_numpy(),
        args,
        "temporal_correction",
    )
    temporal_correction = estimator("temporal_correction", args, args.seed)
    temporal_correction.fit(validation[residual_features], validation_residual)
    test_residual = temporal_correction.predict(test[residual_features])
    validation_temporal = (
        validation[BACKGROUND].to_numpy(dtype=float)
        + validation["temporal_base_local"].to_numpy(dtype=float)
        + validation_residual_oof
    )
    test_temporal = (
        test[BACKGROUND].to_numpy(dtype=float)
        + test["temporal_base_local"].to_numpy(dtype=float)
        + test_residual
    )

    tabular = estimator("tabular", args, args.seed)
    tabular.fit(train[features], train["local_target"])
    validation_tabular = (
        validation[BACKGROUND].to_numpy(dtype=float)
        + tabular.predict(validation[features])
    )
    test_tabular = (
        test[BACKGROUND].to_numpy(dtype=float) + tabular.predict(test[features])
    )

    validation_truth = validation["target_PM2.5"].to_numpy(dtype=float)
    test_truth = test["target_PM2.5"].to_numpy(dtype=float)
    selected_weight = global_weight(
        validation_truth, validation_temporal, validation_tabular
    )
    validation_global = (
        selected_weight * validation_temporal
        + (1.0 - selected_weight) * validation_tabular
    )
    test_global = selected_weight * test_temporal + (1.0 - selected_weight) * test_tabular

    validation_gate_x = gate_features(
        validation, validation_temporal, validation_tabular
    )
    test_gate_x = gate_features(test, test_temporal, test_tabular)
    gate_target = oracle_weight(
        validation_truth, validation_temporal, validation_tabular
    )
    validation_gate_weight = crossfit_regression(
        validation_gate_x,
        gate_target,
        validation["date"].to_numpy(),
        args,
        "gate",
    )
    validation_gate_weight = np.clip(validation_gate_weight, 0.0, 1.0)
    validation_gate_prediction = (
        validation_gate_weight * validation_temporal
        + (1.0 - validation_gate_weight) * validation_tabular
    )
    gate = estimator("gate", args, args.seed)
    gate.fit(validation_gate_x, gate_target)
    test_gate_weight = np.clip(gate.predict(test_gate_x), 0.0, 1.0)
    test_gate_prediction = (
        test_gate_weight * test_temporal
        + (1.0 - test_gate_weight) * test_tabular
    )

    predictions = {
        "temporal_branch": (validation_temporal, test_temporal),
        "enhanced_tabular_branch": (validation_tabular, test_tabular),
        "validation_selected_global_weight": (validation_global, test_global),
        "fixed_equal_weight_diagnostic": (
            0.5 * (validation_temporal + validation_tabular),
            0.5 * (test_temporal + test_tabular),
        ),
        "conditional_validation_gate": (
            validation_gate_prediction,
            test_gate_prediction,
        ),
    }
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for method, (validation_prediction, test_prediction) in predictions.items():
        radius = conformal_radius(
            validation_truth, validation_prediction, args.interval_alpha
        )
        lower, upper = test_prediction - radius, test_prediction + radius
        row = {
            "method": method,
            "n_test": len(test),
            **metric_row(test_truth, test_prediction),
            "interval_coverage": float(np.mean((test_truth >= lower) & (test_truth <= upper))),
            "interval_width": float(2.0 * radius),
        }
        rows.append(row)
        weight = (
            test_gate_weight
            if method == "conditional_validation_gate"
            else np.full(len(test), selected_weight if method == "validation_selected_global_weight" else np.nan)
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "sequence_id": test["sequence_id"].to_numpy(),
                    "target_row_id": test["target_row_id"].to_numpy(),
                    "date": test["date"].to_numpy(),
                    "method": method,
                    "actual_PM2.5": test_truth,
                    "predicted_PM2.5": test_prediction,
                    "interval_lower": lower,
                    "interval_upper": upper,
                    "temporal_weight": weight,
                }
            )
        )

    metrics = pd.DataFrame(rows).sort_values("rmse")
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output / "predictions.csv", index=False
    )
    joblib.dump(tabular, output / "enhanced_tabular_model.joblib")
    joblib.dump(temporal_correction, output / "temporal_correction_model.joblib")
    joblib.dump(gate, output / "conditional_gate_model.joblib")

    run = {
        "dataset": "TRAQID",
        "experiment": "exact final background-decomposed conditional architecture",
        "protocol": "random_sequence_split_after_overlapping_T7_window_construction",
        "reportable_as_generalization": False,
        "warning": "Raw image context overlaps across train/validation/test partitions.",
        "background": "raw MERRA-2 PM2.5; not fitted to roadside targets",
        "temporal_branch": "ResNet50-GRU T7 plus cross-fitted Random Forest residual correction",
        "tabular_branch": "enhanced multiscale T7 ExtraTrees local-increment model",
        "fusion": "validation-selected global blend and cross-fitted conditional ExtraTrees gate",
        "selected_global_temporal_weight": selected_weight,
        "conditional_gate_test_weight_mean": float(np.mean(test_gate_weight)),
        "conditional_gate_test_weight_min": float(np.min(test_gate_weight)),
        "conditional_gate_test_weight_max": float(np.max(test_gate_weight)),
        "outer_test_targets_used_for_model_selection": False,
        "overlap_audit": audit,
        "feature_audit": feature_audit,
        "seed": args.seed,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print("\nTRAQID FINAL ARCHITECTURE — RANDOM OVERLAPPING DIAGNOSTIC\n", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print("\nWARNING: these metrics are leakage-contaminated and not generalization estimates.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
