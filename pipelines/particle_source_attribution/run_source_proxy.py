"""Fit exploratory PM/OPC source proxies and a contextual prediction model.

This is not chemically validated source apportionment. NMF components are fit
only on the training partition, then a non-PM context model predicts their
fractions from meteorology, gases, mobility, YOLO, and road features. Component
names remain neutral until FTIR or reference-source evidence is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


PARTICLE_COLUMNS = [
    "sPM1", "sPM2", "sPM4", "sPM10",
    "sNPMp5", "sNPM1", "sNPM2", "sNPM4", "sNPM10", "sTPS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", required=True)
    parser.add_argument("--feature-groups", required=True)
    parser.add_argument("--context-feature-set", default="sensor_plus_visual")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=["random", "date"], default="random")
    parser.add_argument("--test-date", help="Required for the date protocol")
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_split(frame: pd.DataFrame, protocol: str, test_date: str | None,
               test_size: float, seed: int) -> pd.Series:
    split = pd.Series("train", index=frame.index, dtype="object")
    if protocol == "date":
        if not test_date:
            raise ValueError("--test-date is required with --protocol date")
        test = frame["date"].astype(str).eq(test_date)
        if not test.any() or test.all():
            raise ValueError(f"test date {test_date!r} must select some but not all rows")
        split.loc[test] = "test"
        return split
    if not 0 < test_size < 1:
        raise ValueError("test-size must lie between zero and one")
    rng = np.random.default_rng(seed)
    test_count = max(1, round(test_size * len(frame)))
    split.iloc[rng.choice(len(frame), size=test_count, replace=False)] = "test"
    return split


def positive_scale(values: np.ndarray) -> np.ndarray:
    scale = np.nanmedian(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return scale


def fractions(weights: np.ndarray, profiles: np.ndarray, pm25_index: int) -> np.ndarray:
    contribution = np.maximum(weights * profiles[:, pm25_index][None, :], 0.0)
    total = contribution.sum(axis=1, keepdims=True)
    fallback = np.full_like(contribution, 1.0 / contribution.shape[1])
    return np.divide(contribution, total, out=fallback, where=total > 0)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(values, 0.0)
    total = clipped.sum(axis=1, keepdims=True)
    fallback = np.full_like(clipped, 1.0 / clipped.shape[1])
    return np.divide(clipped, total, out=fallback, where=total > 0)


def component_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    rows = {}
    for component in range(actual.shape[1]):
        truth = actual[:, component]
        estimate = predicted[:, component]
        rows[f"component_{component + 1}"] = {
            "MAE_fraction": float(mean_absolute_error(truth, estimate)),
            "RMSE_fraction": float(mean_squared_error(truth, estimate) ** 0.5),
            "R2": float(r2_score(truth, estimate)),
        }
    rows["macro_MAE_fraction"] = float(mean_absolute_error(actual, predicted))
    return rows


def proxy_groups(columns: list[str]) -> dict[str, list[str]]:
    lowered = {column: column.lower() for column in columns}
    selectors = {
        "traffic_proxy": ("vehicle", "car", "truck", "bus", "motorcycle", "auto_rickshaw", "exhaust"),
        "road_dust_proxy": ("resuspension", "road_brown", "road_gray_dry", "road_area_ratio"),
        "combustion_gas_proxy": ("co_ppb", "no2_ppb", "so2_ppb", "svoci"),
        "meteorology_proxy": ("temp", "rh", "humidity"),
    }
    return {
        name: [column for column, low in lowered.items() if any(token in low for token in tokens)]
        for name, tokens in selectors.items()
    }


def standardized_proxy_scores(frame: pd.DataFrame, groups: dict[str, list[str]],
                              train_mask: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for name, columns in groups.items():
        if not columns:
            continue
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        median = numeric.loc[train_mask].median().fillna(0.0)
        filled = numeric.fillna(median)
        mean = filled.loc[train_mask].mean()
        std = filled.loc[train_mask].std(ddof=0).replace(0, 1.0).fillna(1.0)
        result[name] = ((filled - mean) / std).mean(axis=1)
    return result


def effective_density_proxy(frame: pd.DataFrame) -> pd.Series:
    """Spherical effective-density proxy using documented provisional units."""
    diameter_m = pd.to_numeric(frame["sTPS"], errors="coerce") * 1e-6
    number_m3 = pd.to_numeric(frame["sNPM2"], errors="coerce") * 1e6
    mass_kg_m3 = pd.to_numeric(frame["sPM2"], errors="coerce") * 1e-9
    volume = number_m3 * (np.pi / 6.0) * diameter_m.pow(3)
    return (mass_kg_m3 / volume).replace([np.inf, -np.inf], np.nan)


def main() -> int:
    args = parse_args()
    if args.components < 2:
        raise ValueError("At least two components are required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Consolidate the wide CSV into a fresh frame before adding derived fields.
    frame = pd.read_csv(args.modeling_table).copy()
    required = ["sample_id", "date", *PARTICLE_COLUMNS]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique")

    groups = json.loads(Path(args.feature_groups).read_text())
    if args.context_feature_set not in groups:
        raise ValueError(f"Unknown context feature set: {args.context_feature_set}")
    context_columns = list(groups[args.context_feature_set]["columns"])
    forbidden = sorted(set(context_columns) & set(PARTICLE_COLUMNS))
    if forbidden:
        raise ValueError(f"Particle/PM columns cannot enter the contextual predictor: {forbidden}")

    frame["split"] = make_split(
        frame, args.protocol, args.test_date, args.test_size, args.seed,
    )
    train_mask = frame.split.eq("train").to_numpy()
    test_mask = ~train_mask

    particle = frame[PARTICLE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    train_median = particle.loc[train_mask].median().fillna(0.0).to_numpy()
    particle_values = particle.to_numpy(dtype=float)
    particle_values = np.where(np.isfinite(particle_values), particle_values, train_median)
    particle_values = np.maximum(particle_values, 0.0)
    scale = positive_scale(particle_values[train_mask])
    normalized = particle_values / scale

    nmf = NMF(
        n_components=args.components, init="nndsvda", solver="cd",
        beta_loss="frobenius", max_iter=3000, tol=1e-5, random_state=args.seed,
    )
    train_weights = nmf.fit_transform(normalized[train_mask])
    weights = np.zeros((len(frame), args.components), dtype=float)
    weights[train_mask] = train_weights
    weights[test_mask] = nmf.transform(normalized[test_mask])
    pm25_index = PARTICLE_COLUMNS.index("sPM2")
    inferred_fraction = fractions(weights, nmf.components_, pm25_index)

    context = frame[context_columns].apply(pd.to_numeric, errors="coerce")
    context_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=args.trees, min_samples_leaf=3, max_features=0.7,
            n_jobs=-1, random_state=args.seed,
        )),
    ])
    context_model.fit(context.loc[train_mask], inferred_fraction[train_mask])
    predicted_fraction = normalize_rows(context_model.predict(context))

    estimates = frame[["sample_id", "date", "run_id", "sample_timestamp", "split", "sPM2"]].copy()
    estimates["effective_density_proxy_kg_m3"] = effective_density_proxy(frame)
    for component in range(args.components):
        name = f"component_{component + 1}"
        estimates[f"{name}_nmf_fraction"] = inferred_fraction[:, component]
        estimates[f"{name}_context_fraction"] = predicted_fraction[:, component]
        estimates[f"{name}_context_percent"] = 100.0 * predicted_fraction[:, component]
        estimates[f"{name}_pm25_ug_m3"] = frame.sPM2.to_numpy(dtype=float) * predicted_fraction[:, component]
    estimates["context_fraction_sum"] = predicted_fraction.sum(axis=1)
    estimates["attributed_pm25_sum_ug_m3"] = estimates[
        [f"component_{component + 1}_pm25_ug_m3" for component in range(args.components)]
    ].sum(axis=1)
    estimates["mass_closure_error_ug_m3"] = estimates.attributed_pm25_sum_ug_m3 - estimates.sPM2
    estimates.to_csv(output / "source_proxy_estimates.csv", index=False)

    profile = pd.DataFrame(
        nmf.components_ * scale[None, :],
        index=[f"component_{component + 1}" for component in range(args.components)],
        columns=PARTICLE_COLUMNS,
    )
    profile.index.name = "component"
    profile.to_csv(output / "nmf_particle_profiles.csv")

    proxy = standardized_proxy_scores(frame, proxy_groups(context_columns), train_mask)
    correlation_rows = []
    train_index = np.flatnonzero(train_mask)
    for component in range(args.components):
        for column in proxy:
            correlation_rows.append({
                "component": f"component_{component + 1}",
                "proxy": column,
                "train_spearman": float(pd.Series(inferred_fraction[train_index, component]).corr(
                    proxy.loc[train_mask, column].reset_index(drop=True), method="spearman"
                )),
            })
    pd.DataFrame(correlation_rows).to_csv(output / "component_proxy_correlations.csv", index=False)

    reconstruction = weights @ nmf.components_
    reconstruction_rmse = float(np.sqrt(np.mean((normalized - reconstruction) ** 2)))
    metrics = {
        "context_prediction_test": component_metrics(
            inferred_fraction[test_mask], predicted_fraction[test_mask],
        ),
        "nmf_normalized_reconstruction_RMSE_all_rows": reconstruction_rmse,
        "mass_closure_MAE_ug_m3": float(estimates.mass_closure_error_ug_m3.abs().mean()),
        "effective_density_proxy": {
            "median_kg_m3": float(estimates.effective_density_proxy_kg_m3.median()),
            "provisional": True,
            "warning": "Requires confirmation of OPC number and sTPS diameter units.",
        },
    }
    run = {
        "protocol": args.protocol,
        "reportable_as_chemical_source_apportionment": False,
        "terminology": "exploratory latent source-proxy components",
        "warning": "Component identities require FTIR/chemical reference evidence.",
        "rows": len(frame),
        "split_counts": {key: int(value) for key, value in frame.split.value_counts().items()},
        "particle_inputs": PARTICLE_COLUMNS,
        "context_feature_set": args.context_feature_set,
        "context_feature_count": len(context_columns),
        "components": args.components,
        "nmf_fit_partition": "train only",
        "context_model": "ExtraTreesRegressor",
        "context_pm_opc_inputs_used": False,
        "metrics": metrics,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    joblib.dump({
        "nmf": nmf, "particle_scale": scale, "particle_median": train_median,
        "context_model": context_model, "particle_columns": PARTICLE_COLUMNS,
        "context_columns": context_columns,
    }, output / "source_proxy_models.joblib", compress=3)
    print(json.dumps(run, indent=2), flush=True)
    print("\nNMF particle profiles:\n", profile.round(4).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
