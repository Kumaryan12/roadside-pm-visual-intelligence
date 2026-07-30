"""Strict image-base plus tabular-residual fusion utilities."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SAFE_PREDICTION_ORIGINS = {"oof", "heldout", "outer_test", "outer_val"}
FORBIDDEN_PATTERNS = re.compile(
    r"pm1|pm2|pm4|pm10|pm25|npm|opc|density|aqi|air_quality_index|"
    r"target|actual|predicted|residual|split|fold|embedding|image|sequence|"
    r"row_id|image_id|timestamp|created_at|date",
    re.I,
)


def validate_base_predictions(predictions: pd.DataFrame, *, target: str) -> None:
    required = ["sequence_id", f"actual_{target}", f"predicted_{target}", "prediction_origin"]
    missing = [column for column in required if column not in predictions]
    if missing: raise ValueError(f"Missing base-prediction columns: {missing}")
    unsafe = set(predictions["prediction_origin"].dropna().astype(str)) - SAFE_PREDICTION_ORIGINS
    if unsafe: raise ValueError(f"Unsafe base prediction origins: {sorted(unsafe)}. Training residuals require out-of-fold/held-out predictions.")
    if predictions["sequence_id"].duplicated().any(): raise ValueError("Duplicate sequence_id in base predictions")


def select_numeric_residual_features(table: pd.DataFrame, *, key_columns: list[str], allow_columns: list[str] | None = None) -> list[str]:
    candidates = allow_columns if allow_columns is not None else list(table.columns)
    selected = []
    for column in candidates:
        if column in key_columns or column not in table or FORBIDDEN_PATTERNS.search(column): continue
        if not table[column].notna().any():
            continue
        if pd.api.types.is_numeric_dtype(table[column]) or table[column].dtype == bool:
            selected.append(column)
    if not selected: raise ValueError("No safe numeric residual features selected")
    return selected


def build_residual_dataset(predictions: pd.DataFrame, sequences: pd.DataFrame, table: pd.DataFrame, *, target: str, sample_key: str, group_column: str) -> pd.DataFrame:
    validate_base_predictions(predictions, target=target)
    required_sequence = ["sequence_id", "target_sample_id"]
    missing = [column for column in required_sequence if column not in sequences]
    if missing: raise ValueError(f"Sequence manifest missing: {missing}")
    if sample_key not in table: raise ValueError(f"Feature table missing {sample_key!r}")
    if table[sample_key].duplicated().any(): raise ValueError(f"Feature table has duplicate {sample_key!r}")
    joined = predictions.merge(sequences[required_sequence + ([group_column] if group_column in sequences else [])], on="sequence_id", validate="one_to_one")
    joined = joined.merge(table, left_on="target_sample_id", right_on=sample_key, how="left", validate="many_to_one", suffixes=("", "_tab"))
    if group_column not in joined: raise ValueError(f"Group column {group_column!r} unavailable after merge")
    joined = joined.copy()
    joined["base_residual"] = joined[f"actual_{target}"] - joined[f"predicted_{target}"]
    return joined


def make_residual_pipeline(model: str, feature_columns: list[str], seed: int) -> Pipeline:
    preprocess = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), feature_columns)], remainder="drop")
    if model == "ridge": estimator = Ridge(alpha=10.0)
    elif model == "extra_trees": estimator = ExtraTreesRegressor(n_estimators=700, min_samples_leaf=3, max_features=0.7, random_state=seed, n_jobs=-1)
    elif model == "hist_gradient_boosting": estimator = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, l2_regularization=0.1, random_state=seed)
    else: raise ValueError("model must be ridge, extra_trees, or hist_gradient_boosting")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])
