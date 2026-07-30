"""Model directly observed particle-size regimes from non-PM context.

Unlike source attribution, these targets are deterministic transformations of
measured cumulative PM/OPC channels. Models never receive PM/OPC predictors.
Optional ResNet embeddings are reduced using PCA fitted on training rows only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from pipelines.particle_source_attribution.run_source_proxy import (
    effective_density_proxy,
    make_split,
    normalize_rows,
)


FRACTION_TARGETS = [
    "mass_fraction_0_1_of_pm10",
    "mass_fraction_1_2p5_of_pm10",
    "mass_fraction_2p5_4_of_pm10",
    "mass_fraction_4_10_of_pm10",
]
SCALAR_TARGETS = [
    "effective_diameter_um",
    "effective_density_proxy_kg_m3",
    "total_number_proxy_sNPM2",
]
FORBIDDEN_CONTEXT = {
    "sPM1", "sPM2", "sPM4", "sPM10", "sNPMp5", "sNPM1", "sNPM2",
    "sNPM4", "sNPM10", "sTPS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", required=True)
    parser.add_argument("--feature-groups", required=True)
    parser.add_argument("--context-feature-set", default="sensor_plus_visual")
    parser.add_argument("--embedding-index")
    parser.add_argument("--embeddings")
    parser.add_argument("--embedding-pca-components", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=["random", "date"], default="random")
    parser.add_argument("--test-date")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def derive_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = ["sPM1", "sPM2", "sPM4", "sPM10", "sNPM2", "sTPS"]
    values = frame[required].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("Particle target columns contain missing or nonnumeric values")
    increments = np.column_stack([
        values.sPM1,
        values.sPM2 - values.sPM1,
        values.sPM4 - values.sPM2,
        values.sPM10 - values.sPM4,
    ]).astype(float)
    negative = increments < 0
    increments = np.maximum(increments, 0.0)
    fractions = np.divide(
        increments, increments.sum(axis=1, keepdims=True),
        out=np.full_like(increments, 0.25),
        where=increments.sum(axis=1, keepdims=True) > 0,
    )
    result = pd.DataFrame(fractions, columns=FRACTION_TARGETS, index=frame.index)
    result["effective_diameter_um"] = values.sTPS
    result["effective_density_proxy_kg_m3"] = effective_density_proxy(frame)
    result["total_number_proxy_sNPM2"] = values.sNPM2
    audit = {
        "negative_increment_values": int(negative.sum()),
        "rows_with_negative_increment": int(negative.any(axis=1).sum()),
        "fraction_rows_with_negative_increment": float(negative.any(axis=1).mean()),
        "fraction_sum_max_error": float(np.abs(fractions.sum(axis=1) - 1).max()),
    }
    return result, audit


def metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "R2": float(r2_score(actual, predicted)),
    }


def load_context(frame: pd.DataFrame, context_columns: list[str], train: np.ndarray,
                 args: argparse.Namespace) -> tuple[np.ndarray, list[str], PCA | None, dict[str, object]]:
    context = frame[context_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    names = list(context_columns)
    pca = None
    report = {"resnet_embeddings_used": False, "pca_components": 0}
    if bool(args.embedding_index) != bool(args.embeddings):
        raise ValueError("Provide both --embedding-index and --embeddings, or neither")
    if args.embeddings:
        index = pd.read_csv(args.embedding_index)
        index = index[index.embedding_status.eq("success")][["sample_id", "embedding_row"]]
        if index.sample_id.duplicated().any():
            raise ValueError("Embedding index contains duplicate sample_id values")
        aligned = frame[["sample_id"]].merge(index, on="sample_id", how="left", validate="one_to_one")
        if aligned.embedding_row.isna().any():
            raise ValueError("Some modeling rows do not have successful embeddings")
        array = np.load(args.embeddings, mmap_mode="r")
        embedding = np.asarray(array[aligned.embedding_row.to_numpy(dtype=int)], dtype="float32")
        components = min(args.embedding_pca_components, int(train.sum()) - 1, embedding.shape[1])
        if components < 1:
            raise ValueError("embedding-pca-components must be positive")
        pca = PCA(n_components=components, svd_solver="randomized", random_state=args.seed)
        reduced = np.empty((len(frame), components), dtype="float32")
        reduced[train] = pca.fit_transform(embedding[train])
        reduced[~train] = pca.transform(embedding[~train])
        context = np.column_stack([context, reduced])
        names.extend([f"resnet_pca_{index + 1:03d}" for index in range(components)])
        report = {
            "resnet_embeddings_used": True,
            "pca_components": components,
            "pca_fit_partition": "train only",
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        }
    return context, names, pca, report


def make_model(trees: int, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=trees, min_samples_leaf=3, max_features=0.7,
            n_jobs=-1, random_state=seed,
        )),
    ])


def importances(model: Pipeline, raw_names: list[str], model_name: str) -> pd.DataFrame:
    imputer = model.named_steps["imputer"]
    names = imputer.get_feature_names_out(raw_names)
    importance = model.named_steps["model"].feature_importances_
    return pd.DataFrame({
        "model": model_name, "feature": names, "importance": importance,
    }).sort_values("importance", ascending=False)


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.modeling_table).copy()
    required = ["sample_id", "date", "run_id", "sample_timestamp"]
    missing = [column for column in required if column not in frame]
    if missing or frame.sample_id.duplicated().any():
        raise ValueError(f"Invalid modeling table; missing={missing}, duplicate_ids={frame.sample_id.duplicated().sum()}")
    target, target_audit = derive_targets(frame)
    valid = np.isfinite(target.to_numpy()).all(axis=1)
    if not valid.all():
        frame, target = frame.loc[valid].reset_index(drop=True), target.loc[valid].reset_index(drop=True)
    frame["split"] = make_split(frame, args.protocol, args.test_date, args.test_size, args.seed)
    train = frame.split.eq("train").to_numpy()
    test = ~train

    groups = json.loads(Path(args.feature_groups).read_text())
    if args.context_feature_set not in groups:
        raise ValueError(f"Unknown context feature set {args.context_feature_set!r}")
    context_columns = list(groups[args.context_feature_set]["columns"])
    forbidden = sorted(set(context_columns) & FORBIDDEN_CONTEXT)
    if forbidden:
        raise ValueError(f"PM/OPC predictors are forbidden: {forbidden}")
    context, context_names, pca, embedding_report = load_context(frame, context_columns, train, args)

    predictions = frame[["sample_id", "date", "run_id", "sample_timestamp", "split"]].copy()
    for column in target:
        predictions[f"actual_{column}"] = target[column]
    reports, baselines, models, importance_tables = {}, {}, {}, []

    composition_model = make_model(args.trees, args.seed)
    composition_model.fit(context[train], target.loc[train, FRACTION_TARGETS])
    composition_prediction = normalize_rows(composition_model.predict(context))
    models["size_composition"] = composition_model
    importance_tables.append(importances(composition_model, context_names, "size_composition"))
    train_mean = normalize_rows(target.loc[train, FRACTION_TARGETS].mean().to_numpy()[None, :])[0]
    for index, name in enumerate(FRACTION_TARGETS):
        predictions[f"predicted_{name}"] = composition_prediction[:, index]
        reports[name] = metric_row(target.loc[test, name], composition_prediction[test, index])
        baselines[name] = metric_row(target.loc[test, name], np.full(test.sum(), train_mean[index]))

    for offset, name in enumerate(SCALAR_TARGETS, start=1):
        # Log targets are more stable for density and number; diameter remains linear.
        use_log = name != "effective_diameter_um"
        y = target[name].to_numpy(dtype=float)
        fitted_y = np.log1p(np.maximum(y, 0.0)) if use_log else y
        model = make_model(args.trees, args.seed + offset)
        model.fit(context[train], fitted_y[train])
        fitted_prediction = model.predict(context)
        prediction = np.expm1(fitted_prediction) if use_log else fitted_prediction
        predictions[f"predicted_{name}"] = prediction
        reports[name] = metric_row(y[test], prediction[test])
        baseline = np.full(test.sum(), np.median(y[train]))
        baselines[name] = metric_row(y[test], baseline)
        models[name] = model
        importance_tables.append(importances(model, context_names, name))

    predictions.to_csv(output / "particle_regime_predictions.csv", index=False)
    pd.concat(importance_tables, ignore_index=True).to_csv(output / "feature_importance.csv", index=False)
    target_table = pd.concat([frame[["sample_id", "date", "run_id", "sample_timestamp", "split"]], target], axis=1)
    target_table.to_csv(output / "particle_regime_targets.csv", index=False)
    run = {
        "protocol": args.protocol,
        "reportable_as_unseen_date_generalization": args.protocol == "date",
        "terminology": "directly derived particle-size regime targets",
        "rows": len(frame),
        "split_counts": {key: int(value) for key, value in frame.split.value_counts().items()},
        "context_feature_set": args.context_feature_set,
        "tabular_context_feature_count": len(context_columns),
        "total_context_feature_count": len(context_names),
        "context_pm_opc_inputs_used": False,
        "embedding_preprocessing": embedding_report,
        "target_audit": target_audit,
        "density_warning": "Effective density is provisional until sNPM2 and sTPS units are confirmed.",
        "test_metrics": reports,
        "training_baseline_test_metrics": baselines,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps({"models": reports, "baselines": baselines}, indent=2) + "\n")
    joblib.dump({
        "models": models, "pca": pca, "context_columns": context_columns,
        "context_feature_names": context_names, "fraction_targets": FRACTION_TARGETS,
        "scalar_targets": SCALAR_TARGETS,
    }, output / "particle_regime_models.joblib", compress=3)
    print(json.dumps(run, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
