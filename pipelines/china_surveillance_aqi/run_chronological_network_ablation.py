"""Run leakage-audited Taiwan monitoring-network ablations.

The target site's PM2.5 is excluded at the prediction timestamp and at every
lag.  All tabular models are fitted on the chronological training partition,
candidate model/ensemble selection uses validation only, and the final metrics
are computed once on the chronological test partition.

Meteorology and station geometry are optional, explicit inputs.  They are never
silently synthesized from the target or replaced by constants.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


PARTITIONS = ("train", "val", "test")


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
        "pearson": float(pearsonr(actual, predicted)[0]),
        "spearman": float(spearmanr(actual, predicted)[0]),
    }


def exact_time_lookup(
    times: pd.Series,
    target_sites: pd.Series,
    wide: pd.DataFrame,
    sites: list[str],
    lags: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Return other-site current/lag values and robust summaries.

    The target site's column is set missing at every lag before any summary is
    calculated.  Lags use exact clock-hour offsets, never future observations.
    """
    blocks: list[pd.DataFrame] = []
    columns: list[str] = []
    target_array = target_sites.astype(str).to_numpy()
    for lag in lags:
        lookup_times = pd.DatetimeIndex(times - pd.to_timedelta(lag, unit="h"))
        values = wide.reindex(lookup_times).loc[:, sites].to_numpy(dtype=float, copy=True)
        for col_index, site in enumerate(sites):
            values[target_array == site, col_index] = np.nan
        raw_names = [f"other_site_{site}_lag{lag}h" for site in sites]
        # Missing exact-hour lags are retained for train-fitted imputation.
        # Suppress only the expected warnings for rows where every other site
        # is unavailable at a particular lag.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            summary = np.column_stack(
                [
                    np.nanmedian(values, axis=1),
                    np.nanmean(values, axis=1),
                    np.nanstd(values, axis=1),
                    np.nanmin(values, axis=1),
                    np.nanmax(values, axis=1),
                    np.nanpercentile(values, 25, axis=1),
                    np.nanpercentile(values, 75, axis=1),
                    np.isfinite(values).sum(axis=1),
                ]
            )
        summary_names = [
            f"network_median_lag{lag}h",
            f"network_mean_lag{lag}h",
            f"network_std_lag{lag}h",
            f"network_min_lag{lag}h",
            f"network_max_lag{lag}h",
            f"network_q25_lag{lag}h",
            f"network_q75_lag{lag}h",
            f"network_count_lag{lag}h",
        ]
        blocks.append(pd.DataFrame(np.column_stack([values, summary])))
        columns.extend(raw_names + summary_names)
    result = pd.concat(blocks, axis=1, ignore_index=True)
    result.columns = columns
    return result, columns


def time_and_site_features(frame: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame["target_time"], errors="raise")
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    day = timestamp.dt.dayofyear
    result = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_sin": np.sin(2 * np.pi * day / 365.25),
            "day_cos": np.cos(2 * np.pi * day / 365.25),
        }
    )
    target_site = frame["site"].astype(str)
    for site in sites:
        result[f"target_site_{site}"] = target_site.eq(site).astype(float)
    return result


def optional_meteorology(
    frame: pd.DataFrame, path: str | None
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    if not path:
        return None, {"used": False, "reason": "No --meteorology-csv supplied"}
    met = pd.read_csv(path, low_memory=False)
    if "timestamp" not in met:
        raise ValueError("Meteorology CSV must contain timestamp")
    met["timestamp"] = pd.to_datetime(met["timestamp"], errors="raise")
    keys = ["timestamp"]
    work = frame.assign(timestamp=pd.to_datetime(frame["target_time"], errors="raise"))
    if "site" in met:
        met["site"] = met["site"].astype(str)
        work["site"] = work["site"].astype(str)
        keys.append("site")
    numeric = [
        col for col in met.columns
        if col not in keys and pd.api.types.is_numeric_dtype(met[col])
    ]
    if not numeric:
        raise ValueError("Meteorology CSV has no numeric feature columns")
    merged = work[keys].merge(met[keys + numeric], on=keys, how="left", validate="many_to_one")
    usable = [col for col in numeric if merged[col].notna().mean() >= 0.80 and merged[col].nunique() > 1]
    if not usable:
        raise ValueError("No nonconstant meteorology column has >=80% coverage")
    return merged[usable], {"used": True, "columns": usable, "coverage": merged[usable].notna().mean().to_dict()}


def optional_geometry(
    frame: pd.DataFrame,
    path: str | None,
    wide: pd.DataFrame,
    sites: list[str],
    lags: list[int],
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    if not path:
        return None, {"used": False, "reason": "No --station-metadata-csv supplied"}
    metadata = pd.read_csv(path)
    required = {"site", "latitude", "longitude"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Station metadata needs {sorted(required)}")
    metadata["site"] = metadata["site"].astype(str)
    metadata = metadata.drop_duplicates("site").set_index("site")
    missing = sorted(set(sites) - set(metadata.index))
    if missing:
        raise ValueError(f"Station metadata missing sites: {missing}")
    target = frame["site"].astype(str)
    lat = target.map(metadata["latitude"]).to_numpy(dtype=float)
    lon = target.map(metadata["longitude"]).to_numpy(dtype=float)
    target_sites = target.to_numpy()
    station_lat = metadata.loc[sites, "latitude"].to_numpy(dtype=float)
    station_lon = metadata.loc[sites, "longitude"].to_numpy(dtype=float)
    # Haversine distance from each target row to every supporting station.
    radius = 6371.0088
    lat1 = np.radians(lat[:, None])
    lon1 = np.radians(lon[:, None])
    lat2 = np.radians(station_lat[None, :])
    lon2 = np.radians(station_lon[None, :])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    distance = radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))
    for index, site in enumerate(sites):
        distance[target_sites == site, index] = np.nan
    result = pd.DataFrame({"target_latitude": lat, "target_longitude": lon})
    for index, site in enumerate(sites):
        result[f"distance_to_support_site_{site}_km"] = distance[:, index]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result["nearest_support_station_km"] = np.nanmin(distance, axis=1)
        result["mean_support_station_distance_km"] = np.nanmean(distance, axis=1)
    for lag in lags:
        lookup = pd.DatetimeIndex(frame["target_time"] - pd.to_timedelta(lag, unit="h"))
        values = wide.reindex(lookup).loc[:, sites].to_numpy(dtype=float, copy=True)
        for index, site in enumerate(sites):
            values[target_sites == site, index] = np.nan
        weights = np.where(np.isfinite(values), 1.0 / np.maximum(distance, 0.1), 0.0)
        denominator = weights.sum(axis=1)
        numerator = np.nansum(values * weights, axis=1)
        result[f"inverse_distance_network_pm25_lag{lag}h"] = np.divide(
            numerator, denominator, out=np.full(len(frame), np.nan), where=denominator > 0
        )
    return result, {
        "used": True,
        "sites": sites,
        "source": str(path),
        "features": list(result.columns),
        "target_station_excluded_from_distance_weighted_pm25": True,
    }


def candidates(seed: int) -> dict[str, object]:
    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=600, min_samples_leaf=10, max_features=0.8,
                random_state=seed, n_jobs=-1,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=seed,
            ),
        ),
    }


def fit_stage(
    name: str,
    features: pd.DataFrame,
    frame: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    indices = {part: np.flatnonzero(frame["split"].eq(part).to_numpy()) for part in PARTITIONS}
    actual = frame["target_pm25"].to_numpy(dtype=float)
    trial = []
    fitted = {}
    for model_name, model in candidates(seed).items():
        model.fit(features.iloc[indices["train"]], actual[indices["train"]])
        val_prediction = model.predict(features.iloc[indices["val"]])
        val_rmse = mean_squared_error(actual[indices["val"]], val_prediction) ** 0.5
        trial.append({"model": model_name, "validation_rmse": float(val_rmse)})
        fitted[model_name] = model
    selected = min(trial, key=lambda row: row["validation_rmse"])["model"]
    model = fitted[selected]
    predictions = {part: model.predict(features.iloc[index]) for part, index in indices.items()}
    report = {
        "feature_set": name,
        "n_features": int(features.shape[1]),
        "candidate_validation_metrics": trial,
        "validation_selected_model": selected,
        "validation": score(actual[indices["val"]], predictions["val"]),
        "test": score(actual[indices["test"]], predictions["test"]),
    }
    return predictions, report


def simplex_weights(actual: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    count = matrix.shape[1]
    result = minimize(
        lambda weights: np.mean((actual - matrix @ weights) ** 2),
        np.full(count, 1.0 / count), method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Simplex optimization failed: {result.message}")
    return result.x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--image-val-predictions")
    parser.add_argument("--image-test-predictions")
    parser.add_argument("--meteorology-csv")
    parser.add_argument("--station-metadata-csv")
    parser.add_argument("--lags-hours", default="0,1,2,3,6,12,24")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--test-status",
        choices=("previously_inspected", "previously_unseen"),
        default="previously_inspected",
        help="Controls whether the result may be described as confirmatory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.sequence_manifest, low_memory=False)
    if set(frame["split"].unique()) != set(PARTITIONS):
        raise ValueError(f"Expected train/val/test, got {sorted(frame['split'].unique())}")
    frame["target_time"] = pd.to_datetime(frame["target_time"], errors="raise")
    dates = {part: set(frame.loc[frame["split"].eq(part), "target_time"].dt.date) for part in PARTITIONS}
    date_overlap = {
        "train_val": len(dates["train"] & dates["val"]),
        "train_test": len(dates["train"] & dates["test"]),
        "val_test": len(dates["val"] & dates["test"]),
    }
    if any(date_overlap.values()):
        raise RuntimeError(f"Complete-date overlap detected: {date_overlap}")

    raw = pd.read_csv(args.raw_manifest, low_memory=False)
    raw = raw.loc[raw["dataset"].eq("taiwan")].copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="raise")
    raw["site"] = raw["site"].astype(str)
    raw["pm25"] = pd.to_numeric(raw["pm25"], errors="coerce")
    sites = sorted(raw["site"].dropna().unique())
    wide = raw.pivot_table(index="timestamp", columns="site", values="pm25", aggfunc="median")
    lags = sorted({int(value) for value in args.lags_hours.split(",")})
    if not lags or lags[0] != 0 or min(lags) < 0:
        raise ValueError("Lags must be nonnegative and include 0")

    current, current_names = exact_time_lookup(frame["target_time"], frame["site"], wide, sites, [0])
    lagged, lagged_names = exact_time_lookup(frame["target_time"], frame["site"], wide, sites, lags)
    common = time_and_site_features(frame, sites)
    spatial_features = pd.concat([common, current], axis=1)
    spatiotemporal_features = pd.concat([common, lagged], axis=1)

    met, met_audit = optional_meteorology(frame, args.meteorology_csv)
    geometry, geometry_audit = optional_geometry(
        frame, args.station_metadata_csv, wide, sites, lags
    )
    augmented_parts = [spatiotemporal_features]
    if met is not None:
        augmented_parts.append(met)
    if geometry is not None:
        augmented_parts.append(geometry)
    augmented_features = pd.concat(augmented_parts, axis=1)

    stages = {
        "learned_spatial_background": spatial_features,
        "lagged_spatiotemporal_background": spatiotemporal_features,
    }
    if met is not None or geometry is not None:
        stages["spatiotemporal_plus_verified_met_geometry"] = augmented_features

    predictions: dict[str, dict[str, np.ndarray]] = {}
    reports = []
    for offset, (name, features) in enumerate(stages.items()):
        prediction, report = fit_stage(name, features, frame, args.seed + offset)
        predictions[name] = prediction
        reports.append(report)

    indices = {part: np.flatnonzero(frame["split"].eq(part).to_numpy()) for part in PARTITIONS}
    actual = {part: frame.iloc[index]["target_pm25"].to_numpy(dtype=float) for part, index in indices.items()}
    baseline = {
        part: frame.iloc[index]["background_reference_pm25"].to_numpy(dtype=float)
        for part, index in indices.items()
    }
    reports.insert(0, {
        "feature_set": "monitoring_network_median",
        "n_features": 1,
        "validation_selected_model": "none",
        "validation": score(actual["val"], baseline["val"]),
        "test": score(actual["test"], baseline["test"]),
    })
    predictions["monitoring_network_median"] = baseline

    if bool(args.image_val_predictions) != bool(args.image_test_predictions):
        raise ValueError("Supply both image validation and test predictions, or neither")
    if args.image_val_predictions:
        image_predictions = {}
        for part, path in (("val", args.image_val_predictions), ("test", args.image_test_predictions)):
            image = pd.read_csv(path)
            required = {"sequence_id", "predicted_reference_context_plus_conditioned_image"}
            if not required.issubset(image.columns):
                raise ValueError(f"Image predictions missing {sorted(required - set(image.columns))}")
            aligned = frame.iloc[indices[part]][["sequence_id"]].merge(
                image[list(required)], on="sequence_id", how="left", validate="one_to_one"
            )
            if aligned.isna().any().any():
                raise RuntimeError(f"Incomplete image alignment for {part}")
            image_predictions[part] = aligned["predicted_reference_context_plus_conditioned_image"].to_numpy(float)
        predictions["conditioned_image_model"] = image_predictions
        reports.append({
            "feature_set": "conditioned_image_model",
            "n_features": None,
            "validation_selected_model": "previously_validation_selected_frozen_model",
            "validation": score(actual["val"], image_predictions["val"]),
            "test": score(actual["test"], image_predictions["test"]),
        })

    ensemble_names = [name for name in predictions if "median" not in name and "conditioned_image" not in name]
    if "conditioned_image_model" in predictions:
        ensemble_names.append("conditioned_image_model")
    val_matrix = np.column_stack([predictions[name]["val"] for name in ensemble_names])
    test_matrix = np.column_stack([predictions[name]["test"] for name in ensemble_names])
    weights = simplex_weights(actual["val"], val_matrix)
    ensemble_val = val_matrix @ weights
    ensemble_test = test_matrix @ weights
    reports.append({
        "feature_set": "validation_selected_convex_ensemble",
        "n_features": None,
        "validation_selected_model": "nonnegative_simplex",
        "members": ensemble_names,
        "weights": {name: float(weight) for name, weight in zip(ensemble_names, weights, strict=True)},
        "validation": score(actual["val"], ensemble_val),
        "test": score(actual["test"], ensemble_test),
    })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "chronological_complete_date_60_20_20_validation_only_selection",
        "test_status": args.test_status,
        "reportable_as_confirmatory_future_period_generalization": args.test_status == "previously_unseen",
        "caveat": (
            "Split-level leakage controls are valid, but this is exploratory because the test period "
            "was previously inspected. Use a newly collected/future period for confirmation."
            if args.test_status == "previously_inspected"
            else "Test period declared untouched before this pre-specified run."
        ),
        "target_current_pm25_used_as_predictor": False,
        "target_lagged_pm25_used_as_predictor": False,
        "other_station_contemporaneous_pm25_required_at_inference": True,
        "test_target_used_for_fit_or_selection": False,
        "date_overlap": date_overlap,
        "lags_hours": lags,
        "sites": sites,
        "meteorology": met_audit,
        "station_geometry": geometry_audit,
        "stages": reports,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    flat = []
    for row in reports:
        flat.append({
            "stage": row["feature_set"],
            "selected_model": row["validation_selected_model"],
            **{f"val_{key}": value for key, value in row["validation"].items()},
            **{f"test_{key}": value for key, value in row["test"].items()},
        })
    pd.DataFrame(flat).sort_values("test_rmse").to_csv(output / "comparison.csv", index=False)
    test_table = pd.DataFrame({
        "sequence_id": frame.iloc[indices["test"]]["sequence_id"].to_numpy(),
        "actual_pm25": actual["test"],
        **{f"predicted_{name}": values["test"] for name, values in predictions.items()},
        "predicted_validation_selected_ensemble": ensemble_test,
    })
    test_table.to_csv(output / "predictions_test.csv", index=False)
    print(pd.DataFrame(flat).sort_values("test_rmse").to_string(index=False))
    print(json.dumps({"ensemble_weights": report["stages"][-1]["weights"], "audit": {
        "date_overlap": date_overlap,
        "target_current_pm25_used": False,
        "target_lagged_pm25_used": False,
        "test_target_used_for_selection": False,
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
