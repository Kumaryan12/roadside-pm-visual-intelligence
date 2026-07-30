"""Leakage-safe split columns for multi-day mobile monitoring data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .splits import balanced_deterministic_group_split


def add_spatial_blocks(frame: pd.DataFrame, *, latitude_column: str, longitude_column: str,
                       block_size_m: float = 500.0) -> pd.DataFrame:
    if block_size_m <= 0: raise ValueError("block_size_m must be positive")
    result = frame.copy(); lat = pd.to_numeric(result[latitude_column], errors="coerce"); lon = pd.to_numeric(result[longitude_column], errors="coerce")
    if lat.isna().any() or lon.isna().any(): raise ValueError("Spatial split requires complete numeric latitude/longitude")
    northing = lat * 111_320.0; easting = lon * 111_320.0 * np.cos(np.deg2rad(lat))
    result["spatial_block_id"] = (np.floor(easting / block_size_m).astype(int).astype(str) + "_" + np.floor(northing / block_size_m).astype(int).astype(str))
    return result


def assign_evaluation_protocols(frame: pd.DataFrame, *, date_column: str, trip_column: str,
                                latitude_column: str, longitude_column: str,
                                chronological_test_days: int = 2, seed: int = 42,
                                spatial_block_size_m: float = 500.0) -> pd.DataFrame:
    required = [date_column, trip_column, latitude_column, longitude_column]
    missing = [column for column in required if column not in frame]
    if missing: raise ValueError(f"Missing split columns: {missing}")
    result = add_spatial_blocks(frame, latitude_column=latitude_column, longitude_column=longitude_column, block_size_m=spatial_block_size_m)
    dates = pd.to_datetime(result[date_column], errors="raise").dt.date.astype(str); unique_dates = sorted(dates.unique())
    if len(unique_dates) < 3: raise ValueError("Need at least three dates for train/validation/test evaluation")
    if chronological_test_days < 1 or chronological_test_days >= len(unique_dates) - 1: raise ValueError("chronological_test_days leaves insufficient training/validation dates")
    result["evaluation_date"] = dates
    day_to_fold = {day: index + 1 for index, day in enumerate(unique_dates)}
    result["logo_day_fold"] = dates.map(day_to_fold).astype(int)
    test_dates = set(unique_dates[-chronological_test_days:]); remaining = unique_dates[:-chronological_test_days]
    val_dates = {remaining[-1]}
    result["split_chronological_day"] = np.where(dates.isin(test_dates), "test", np.where(dates.isin(val_dates), "val", "train"))
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    trip_assignment = balanced_deterministic_group_split(result[trip_column].astype(str), fractions, seed=seed)
    spatial_assignment = balanced_deterministic_group_split(result["spatial_block_id"], fractions, seed=seed)
    result["split_grouped_trip"] = result[trip_column].astype(str).map(trip_assignment)
    result["split_grouped_spatial"] = result["spatial_block_id"].map(spatial_assignment)
    return result
