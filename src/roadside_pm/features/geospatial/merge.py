"""Strict sample-level geospatial feature merging."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def merge_sample_features(
    base: pd.DataFrame,
    features: pd.DataFrame,
    *,
    key: str | Sequence[str] = "sample_index",
    feature_prefix: str | None = None,
) -> pd.DataFrame:
    keys = [key] if isinstance(key, str) else list(key)
    missing = [column for column in keys if column not in base.columns or column not in features.columns]
    if missing:
        raise ValueError(f"Both tables must contain merge keys; missing {missing}")
    if base.duplicated(keys).any():
        raise ValueError(f"Base table has duplicate key values for {keys!r}")
    if features.duplicated(keys).any():
        raise ValueError(f"Feature table has duplicate key values for {keys!r}")

    selected = features.copy()
    if feature_prefix:
        rename = {
            column: f"{feature_prefix}{column}"
            for column in selected.columns
            if column not in keys and not column.startswith(feature_prefix)
        }
        selected = selected.rename(columns=rename)

    overlap = (set(base.columns) & set(selected.columns)) - set(keys)
    if overlap:
        raise ValueError(f"Feature columns already exist in base table: {sorted(overlap)}")
    return base.merge(selected, on=keys, how="left", validate="one_to_one")
