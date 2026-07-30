"""Evaluate CAMS, MERRA-2, and ERA5 atmospheric baselines fairly.

The primary protocol is leave-one-date-out.  Three fixed external backgrounds
are compared without fitting the mobile PM2.5 target:

* raw CAMS PM2.5;
* raw MERRA-2 reconstructed PM2.5;
* the arithmetic mean of CAMS and MERRA-2.

A fourth, explicitly target-fitted atmospheric baseline is included as an
exploratory candidate.  Its regression algorithm is selected using inner
leave-one-date-out predictions from the outer-training dates only.  Those
inner out-of-fold predictions also define the local-increment targets used to
train the visual/geospatial correction, avoiding in-sample background
residuals.  The held-out outer date is never used for model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.pm25_prediction.mumma_7day.model_current_data import estimators
from pipelines.pm25_prediction.mumma_7day.run_residual_fusion_current_data import (
    metric_row,
)


CAMS = "background_cams_pm25_ug_m3"
MERRA = "background_merra2_pm25_ug_m3"

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

FORBIDDEN_PM_OPC = {
    "sPM1",
    "sPM2",
    "sPM4",
    "sPM10",
    "sNPMp5",
    "sNPM1",
    "sNPM2",
    "sNPM4",
    "sNPM10",
    "sTPS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--feature-groups", type=Path, required=True)
    parser.add_argument("--background-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--date-col", default="date")
    parser.add_argument(
        "--local-feature-set",
        default="visual_yolo_road_osm_alphaearth",
    )
    parser.add_argument("--local-model", default="extra_trees")
    parser.add_argument(
        "--atmospheric-models",
        nargs="+",
        default=["ridge", "random_forest", "extra_trees", "hist_gradient_boosting"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fixed_background(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "raw_cams":
        return frame[CAMS].to_numpy(dtype=float)
    if name == "raw_merra2":
        return frame[MERRA].to_numpy(dtype=float)
    if name == "mean_cams_merra2":
        return 0.5 * (
            frame[CAMS].to_numpy(dtype=float)
            + frame[MERRA].to_numpy(dtype=float)
        )
    raise ValueError(f"Unknown fixed background: {name}")


def inner_oof_atmospheric_predictions(
    outer_train: pd.DataFrame,
    *,
    target: str,
    date_col: str,
    model_names: list[str],
    seed: int,
) -> tuple[str, np.ndarray, pd.DataFrame]:
    dates = sorted(outer_train[date_col].astype(str).unique())
    if len(dates) < 2:
        raise ValueError("Atmospheric model selection needs at least two dates")
    candidates = estimators(seed)
    missing = sorted(set(model_names) - set(candidates))
    if missing:
        raise ValueError(f"Unknown atmospheric models: {missing}")
    predictions: dict[str, np.ndarray] = {
        name: np.full(len(outer_train), np.nan, dtype=float)
        for name in model_names
    }
    date_values = outer_train[date_col].astype(str).to_numpy()
    truth = outer_train[target].to_numpy(dtype=float)
    for inner_fold, validation_date in enumerate(dates, start=1):
        train_mask = date_values != validation_date
        validation_mask = ~train_mask
        fold_candidates = estimators(seed + inner_fold - 1)
        for name in model_names:
            model = fold_candidates[name]
            model.fit(
                outer_train.loc[train_mask, ATMOSPHERIC_FEATURES],
                truth[train_mask],
            )
            predictions[name][validation_mask] = model.predict(
                outer_train.loc[validation_mask, ATMOSPHERIC_FEATURES]
            )
    rows: list[dict[str, object]] = []
    for name, prediction in predictions.items():
        if np.isnan(prediction).any():
            raise RuntimeError(f"Atmospheric OOF predictions incomplete for {name}")
        rows.append(
            {
                "atmospheric_model": name,
                "n_inner_oof": len(prediction),
                **metric_row(truth, prediction),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    selected = str(ranking.iloc[0]["atmospheric_model"])
    return selected, predictions[selected], ranking


def fit_local_increment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    train_background: np.ndarray,
    test_background: np.ndarray,
    target: str,
    features: list[str],
    model_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = estimators(seed)
    if model_name not in candidates:
        raise ValueError(f"Unknown local model: {model_name}")
    local_target = train[target].to_numpy(dtype=float) - train_background
    model = candidates[model_name]
    model.fit(train[features], local_target)
    local_prediction = model.predict(test[features])
    return local_prediction, test_background + local_prediction


def append_predictions(
    output: list[pd.DataFrame],
    test: pd.DataFrame,
    *,
    fold: int,
    test_date: str,
    method: str,
    target: str,
    background_prediction: np.ndarray,
    local_prediction: np.ndarray,
    prediction: np.ndarray,
    atmospheric_model: str,
    target_fitted_background: bool,
) -> None:
    output.append(
        pd.DataFrame(
            {
                "sample_id": test["sample_id"].to_numpy(),
                "date": test["date"].to_numpy(),
                "fold": fold,
                "test_date": test_date,
                "method": method,
                "atmospheric_model": atmospheric_model,
                "target_fitted_background": target_fitted_background,
                "actual": test[target].to_numpy(dtype=float),
                "background_prediction": background_prediction,
                "local_increment_prediction": local_prediction,
                "prediction": prediction,
            }
        )
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = json.loads(args.feature_groups.read_text())
    if args.local_feature_set not in groups:
        raise ValueError(f"Unknown local feature set: {args.local_feature_set}")
    local_features = list(groups[args.local_feature_set]["columns"])
    leaked = sorted(set(local_features) & FORBIDDEN_PM_OPC)
    if leaked:
        raise ValueError(f"Local features contain PM/OPC target proxies: {leaked}")

    table = pd.read_csv(args.modeling_table)
    table["sample_id"] = table["sample_id"].astype(str)
    if not table["sample_id"].is_unique:
        raise ValueError("Modeling table sample_id must be unique")
    background = pd.read_csv(args.background_csv)
    background["sample_id"] = background["sample_id"].astype(str)
    if not background["sample_id"].is_unique:
        raise ValueError("Background sample_id must be unique")
    if "background_status" in background and not background[
        "background_status"
    ].eq("success").all():
        raise ValueError("Background table contains unsuccessful rows")
    available_background_columns = [
        column
        for column in background.columns
        if column.startswith("background_") and column not in table.columns
    ]
    frame = table.merge(
        background[["sample_id", *available_background_columns]],
        on="sample_id",
        validate="one_to_one",
    ).copy()
    required = {
        args.target,
        args.date_col,
        *local_features,
        *ATMOSPHERIC_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    if frame[[*local_features, *ATMOSPHERIC_FEATURES]].isna().all(axis=0).any():
        empty = frame[[*local_features, *ATMOSPHERIC_FEATURES]].columns[
            frame[[*local_features, *ATMOSPHERIC_FEATURES]].isna().all(axis=0)
        ].tolist()
        raise ValueError(f"Entirely missing predictors: {empty}")
    frame[args.date_col] = frame[args.date_col].astype(str)

    metric_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    dates = sorted(frame[args.date_col].unique())
    fixed_names = ["raw_cams", "raw_merra2", "mean_cams_merra2"]

    for fold, test_date in enumerate(dates, start=1):
        outer_train = frame[~frame[args.date_col].eq(test_date)].copy()
        outer_test = frame[frame[args.date_col].eq(test_date)].copy()
        truth = outer_test[args.target].to_numpy(dtype=float)

        for name in fixed_names:
            train_background = fixed_background(outer_train, name)
            test_background = fixed_background(outer_test, name)
            metric_rows.append(
                {
                    "fold": fold,
                    "test_date": test_date,
                    "method": f"{name}_only",
                    "atmospheric_model": name,
                    "target_fitted_background": False,
                    "n_test": len(outer_test),
                    **metric_row(truth, test_background),
                }
            )
            append_predictions(
                prediction_parts,
                outer_test,
                fold=fold,
                test_date=test_date,
                method=f"{name}_only",
                target=args.target,
                background_prediction=test_background,
                local_prediction=np.zeros(len(outer_test), dtype=float),
                prediction=test_background,
                atmospheric_model=name,
                target_fitted_background=False,
            )
            local_prediction, prediction = fit_local_increment(
                outer_train,
                outer_test,
                train_background=train_background,
                test_background=test_background,
                target=args.target,
                features=local_features,
                model_name=args.local_model,
                seed=args.seed + fold,
            )
            method = f"{name}_plus_local"
            metric_rows.append(
                {
                    "fold": fold,
                    "test_date": test_date,
                    "method": method,
                    "atmospheric_model": name,
                    "target_fitted_background": False,
                    "n_test": len(outer_test),
                    **metric_row(truth, prediction),
                }
            )
            append_predictions(
                prediction_parts,
                outer_test,
                fold=fold,
                test_date=test_date,
                method=method,
                target=args.target,
                background_prediction=test_background,
                local_prediction=local_prediction,
                prediction=prediction,
                atmospheric_model=name,
                target_fitted_background=False,
            )

        selected, oof_background, inner_ranking = (
            inner_oof_atmospheric_predictions(
                outer_train,
                target=args.target,
                date_col=args.date_col,
                model_names=args.atmospheric_models,
                seed=args.seed + 1000 * fold,
            )
        )
        inner_ranking.insert(0, "fold", fold)
        inner_ranking.insert(1, "outer_test_date", test_date)
        inner_ranking["selected"] = inner_ranking["atmospheric_model"].eq(selected)
        selection_parts.append(inner_ranking)

        atmospheric_model = estimators(args.seed + 10_000 + fold)[selected]
        atmospheric_model.fit(
            outer_train[ATMOSPHERIC_FEATURES],
            outer_train[args.target].to_numpy(dtype=float),
        )
        calibrated_test_background = atmospheric_model.predict(
            outer_test[ATMOSPHERIC_FEATURES]
        )
        metric_rows.append(
            {
                "fold": fold,
                "test_date": test_date,
                "method": "nested_calibrated_atmosphere_only",
                "atmospheric_model": selected,
                "target_fitted_background": True,
                "n_test": len(outer_test),
                **metric_row(truth, calibrated_test_background),
            }
        )
        append_predictions(
            prediction_parts,
            outer_test,
            fold=fold,
            test_date=test_date,
            method="nested_calibrated_atmosphere_only",
            target=args.target,
            background_prediction=calibrated_test_background,
            local_prediction=np.zeros(len(outer_test), dtype=float),
            prediction=calibrated_test_background,
            atmospheric_model=selected,
            target_fitted_background=True,
        )
        local_prediction, prediction = fit_local_increment(
            outer_train,
            outer_test,
            train_background=oof_background,
            test_background=calibrated_test_background,
            target=args.target,
            features=local_features,
            model_name=args.local_model,
            seed=args.seed + fold,
        )
        metric_rows.append(
            {
                "fold": fold,
                "test_date": test_date,
                "method": "nested_calibrated_atmosphere_plus_local",
                "atmospheric_model": selected,
                "target_fitted_background": True,
                "n_test": len(outer_test),
                **metric_row(truth, prediction),
            }
        )
        append_predictions(
            prediction_parts,
            outer_test,
            fold=fold,
            test_date=test_date,
            method="nested_calibrated_atmosphere_plus_local",
            target=args.target,
            background_prediction=calibrated_test_background,
            local_prediction=local_prediction,
            prediction=prediction,
            atmospheric_model=selected,
            target_fitted_background=True,
        )
        print(
            f"completed fold={fold} test_date={test_date} "
            f"selected_atmospheric_model={selected}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    aggregate_rows: list[dict[str, object]] = []
    for method, part in predictions.groupby("method", sort=False):
        fold_metrics = metrics[metrics["method"].eq(method)]
        aggregate_rows.append(
            {
                "method": method,
                "target_fitted_background": bool(
                    part["target_fitted_background"].iloc[0]
                ),
                "n_test": len(part),
                **metric_row(
                    part["actual"].to_numpy(dtype=float),
                    part["prediction"].to_numpy(dtype=float),
                ),
                "mean_fold_r2": float(fold_metrics["r2"].mean()),
                "mean_fold_rmse": float(fold_metrics["rmse"].mean()),
                "worst_fold_rmse": float(fold_metrics["rmse"].max()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values("rmse")
    selections = pd.concat(selection_parts, ignore_index=True)

    metrics.to_csv(args.output_dir / "metrics_by_fold.csv", index=False)
    aggregate.to_csv(args.output_dir / "metrics_aggregate.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    selections.to_csv(
        args.output_dir / "atmospheric_model_selection.csv",
        index=False,
    )
    run = {
        "protocol": (
            "leave-one-date-out; atmospheric model selected with inner "
            "leave-one-date-out on outer-training dates"
        ),
        "target": args.target,
        "fixed_external_backgrounds": fixed_names,
        "target_fitted_atmospheric_candidate": True,
        "target_fitted_candidate_is_physical_background": False,
        "local_feature_set": args.local_feature_set,
        "local_features": len(local_features),
        "local_model": args.local_model,
        "atmospheric_features": ATMOSPHERIC_FEATURES,
        "atmospheric_models": args.atmospheric_models,
        "pm_opc_predictors_used": False,
        "outer_test_used_for_selection": False,
        "confirmatory": False,
        "reason": (
            "The multi-source background design was introduced after inspecting "
            "the original five-day CAMS results."
        ),
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2) + "\n"
    )
    print("\nPooled multi-source background comparison:")
    print(aggregate.to_string(index=False))
    print("\nNested atmospheric selections:")
    print(
        selections[selections["selected"]][
            ["fold", "outer_test_date", "atmospheric_model", "rmse", "r2"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
