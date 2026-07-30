"""Estimate source scenarios conditional on explicit, uncertain profiles.

This runner uses Monte Carlo perturbations of versioned assumed mass-size
profiles. For each draw and observation, nonnegative least squares estimates a
mixture. The median scenario fractions become targets for a non-PM contextual
model. Outputs are scenario analyses, never chemically validated attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from pipelines.particle_source_attribution.run_source_proxy import make_split, normalize_rows


MASS_COLUMNS = ["sPM1", "sPM2", "sPM4", "sPM10"]
BIN_COLUMNS = ["pm_0_1", "pm_1_2p5", "pm_2p5_4", "pm_4_10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", required=True)
    parser.add_argument("--feature-groups", required=True)
    parser.add_argument("--context-feature-set", default="sensor_plus_visual")
    parser.add_argument("--assumptions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=["random", "date"], default="random")
    parser.add_argument("--test-date")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def incremental_mass(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    cumulative = frame[MASS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if cumulative.isna().any().any():
        raise ValueError("PM mass channels contain missing or nonnumeric values")
    raw = np.column_stack([
        cumulative.sPM1,
        cumulative.sPM2 - cumulative.sPM1,
        cumulative.sPM4 - cumulative.sPM2,
        cumulative.sPM10 - cumulative.sPM4,
    ]).astype(float)
    negative = raw < 0
    audit = {
        "negative_increment_values": int(negative.sum()),
        "rows_with_negative_increment": int(negative.any(axis=1).sum()),
        "fraction_rows_with_negative_increment": float(negative.any(axis=1).mean()),
        "policy": "clip_to_zero_and_report",
    }
    return np.maximum(raw, 0.0), audit


def load_assumptions(path: str | Path) -> tuple[dict[str, object], list[str], np.ndarray, np.ndarray]:
    config = yaml.safe_load(Path(path).read_text())
    if config.get("status") != "illustrative_unvalidated":
        raise ValueError("Assumption config must explicitly declare illustrative_unvalidated status")
    sources = config.get("sources", {})
    if len(sources) < 2:
        raise ValueError("At least two assumed source scenarios are required")
    names, means, concentrations = [], [], []
    for name, values in sources.items():
        mean = np.asarray(values["mass_bin_mean"], dtype=float)
        if mean.shape != (len(BIN_COLUMNS),) or (mean < 0).any() or mean.sum() <= 0:
            raise ValueError(f"Invalid mass_bin_mean for {name}")
        names.append(name)
        means.append(mean / mean.sum())
        concentrations.append(float(values["dirichlet_concentration"]))
    return config, names, np.asarray(means), np.asarray(concentrations)


def infer_scenarios(mass: np.ndarray, profile_means: np.ndarray,
                    concentrations: np.ndarray, draws: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    n_rows, n_sources = len(mass), len(profile_means)
    samples = np.empty((draws, n_rows, n_sources), dtype="float32")
    reconstruction_errors = []
    raw_pm25_errors = []
    for draw in range(draws):
        profiles = np.vstack([
            rng.dirichlet(np.maximum(mean * concentration, 1e-3))
            for mean, concentration in zip(profile_means, concentrations)
        ])
        for row, observed in enumerate(mass):
            coefficients, _ = nnls(profiles.T, observed, maxiter=500)
            reconstructed = coefficients @ profiles
            reconstruction_errors.append(float(np.sqrt(np.mean((observed - reconstructed) ** 2))))
            pm25_contribution = coefficients * profiles[:, :2].sum(axis=1)
            raw_pm25_errors.append(float(pm25_contribution.sum() - observed[:2].sum()))
            total = pm25_contribution.sum()
            if total > 0:
                samples[draw, row] = pm25_contribution / total
            else:
                samples[draw, row] = 1.0 / n_sources
    median = np.median(samples, axis=0)
    median = normalize_rows(median)
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    diagnostics = {
        "incremental_mass_reconstruction_RMSE_median_ug_m3": float(np.median(reconstruction_errors)),
        "raw_pm25_closure_MAE_median_ug_m3": float(np.median(np.abs(raw_pm25_errors))),
        "median_90pct_interval_width_fraction": float(np.median(upper - lower)),
    }
    return median, lower, upper, diagnostics


def metrics(actual: np.ndarray, predicted: np.ndarray, names: list[str]) -> dict[str, object]:
    report = {}
    for index, name in enumerate(names):
        truth, estimate = actual[:, index], predicted[:, index]
        report[name] = {
            "MAE_fraction": float(mean_absolute_error(truth, estimate)),
            "RMSE_fraction": float(mean_squared_error(truth, estimate) ** 0.5),
            "R2": float(r2_score(truth, estimate)),
        }
    report["macro_MAE_fraction"] = float(mean_absolute_error(actual, predicted))
    return report


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.modeling_table).copy()
    required = ["sample_id", "date", "run_id", "sample_timestamp", *MASS_COLUMNS]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique")

    config, source_names, profile_means, concentrations = load_assumptions(args.assumptions)
    mass, increment_audit = incremental_mass(frame)
    inferred, lower, upper, inference_diagnostics = infer_scenarios(
        mass, profile_means, concentrations, args.draws, args.seed,
    )
    frame["split"] = make_split(frame, args.protocol, args.test_date, args.test_size, args.seed)
    train = frame.split.eq("train").to_numpy()
    test = ~train

    groups = json.loads(Path(args.feature_groups).read_text())
    if args.context_feature_set not in groups:
        raise ValueError(f"Unknown context feature set {args.context_feature_set!r}")
    context_columns = list(groups[args.context_feature_set]["columns"])
    forbidden = sorted(set(context_columns) & set(MASS_COLUMNS + ["sNPMp5", "sNPM1", "sNPM2", "sNPM4", "sNPM10", "sTPS"]))
    if forbidden:
        raise ValueError(f"PM/OPC columns cannot enter context prediction: {forbidden}")
    context = frame[context_columns].apply(pd.to_numeric, errors="coerce")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=args.trees, min_samples_leaf=3, max_features=0.7,
            n_jobs=-1, random_state=args.seed,
        )),
    ])
    model.fit(context.loc[train], inferred[train])
    predicted = normalize_rows(model.predict(context))

    estimates = frame[["sample_id", "date", "run_id", "sample_timestamp", "split", "sPM2"]].copy()
    for bin_index, bin_name in enumerate(BIN_COLUMNS):
        estimates[bin_name] = mass[:, bin_index]
    for index, name in enumerate(source_names):
        estimates[f"{name}_scenario_fraction_median"] = inferred[:, index]
        estimates[f"{name}_scenario_fraction_p05"] = lower[:, index]
        estimates[f"{name}_scenario_fraction_p95"] = upper[:, index]
        estimates[f"{name}_context_fraction"] = predicted[:, index]
        estimates[f"{name}_context_percent"] = 100.0 * predicted[:, index]
        estimates[f"{name}_pm25_ug_m3"] = frame.sPM2.to_numpy(float) * predicted[:, index]
    estimates["context_fraction_sum"] = predicted.sum(axis=1)
    estimates.to_csv(output / "assumption_conditioned_estimates.csv", index=False)

    profile_table = pd.DataFrame(profile_means, index=source_names, columns=BIN_COLUMNS)
    profile_table.index.name = "source_scenario"
    profile_table.to_csv(output / "assumed_profile_means.csv")
    summary_rows = []
    for index, name in enumerate(source_names):
        summary_rows.append({
            "source_scenario": name,
            "median_fraction_all_rows": float(np.median(inferred[:, index])),
            "mean_context_fraction_test": float(predicted[test, index].mean()),
            "median_interval_width_all_rows": float(np.median(upper[:, index] - lower[:, index])),
        })
    pd.DataFrame(summary_rows).to_csv(output / "scenario_summary.csv", index=False)

    test_metrics = metrics(inferred[test], predicted[test], source_names)
    profile_norm = profile_means / np.maximum(np.linalg.norm(profile_means, axis=1, keepdims=True), 1e-12)
    profile_cosine = profile_norm @ profile_norm.T
    off_diagonal = profile_cosine[~np.eye(len(source_names), dtype=bool)]
    global_medians = np.median(inferred, axis=0)
    dominant_index = int(np.argmax(global_medians))
    identifiability = {
        "assumed_profile_matrix_rank": int(np.linalg.matrix_rank(profile_means)),
        "assumed_profile_condition_number": float(np.linalg.cond(profile_means)),
        "maximum_pairwise_profile_cosine": float(off_diagonal.max()),
        "dominant_scenario": source_names[dominant_index],
        "dominant_scenario_median_fraction": float(global_medians[dominant_index]),
        "degenerate_dominant_solution": bool(global_medians[dominant_index] > 0.90),
    }
    run = {
        "protocol": args.protocol,
        "reportable_as_chemical_source_apportionment": False,
        "terminology": "assumption-conditioned source scenario analysis",
        "rows": len(frame),
        "split_counts": {key: int(value) for key, value in frame.split.value_counts().items()},
        "assumptions_file": str(args.assumptions),
        "source_scenarios": source_names,
        "monte_carlo_draws": args.draws,
        "context_feature_set": args.context_feature_set,
        "context_feature_count": len(context_columns),
        "context_pm_opc_inputs_used": False,
        "opc_number_constraint_used": False,
        "density_assumptions_used_in_solver": False,
        "increment_audit": increment_audit,
        "identifiability_audit": identifiability,
        "inference_diagnostics": inference_diagnostics,
        "context_prediction_test": test_metrics,
        "warning": (
            "Fractions are conditional on illustrative profiles and are not measured source contributions. "
            "A degenerate dominant solution must be treated as an assumption-sensitivity failure."
        ),
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (output / "assumptions_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    joblib.dump({
        "context_model": model, "context_columns": context_columns,
        "source_names": source_names, "profile_means": profile_means,
        "profile_concentrations": concentrations,
    }, output / "scenario_context_model.joblib", compress=3)
    print(json.dumps(run, indent=2), flush=True)
    print("\nScenario summary:\n", pd.DataFrame(summary_rows).round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
