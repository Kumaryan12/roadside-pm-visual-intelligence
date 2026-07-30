"""AlphaEarth annual embedding helpers."""

from __future__ import annotations

import pandas as pd


DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]


def measurement_years(series: pd.Series, fallback_year: int | None = None) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    years = parsed.dt.year
    if fallback_year is not None:
        years = years.fillna(fallback_year)
    if years.isna().any():
        raise ValueError("Could not derive AlphaEarth year for every sample")
    return years.astype(int)

