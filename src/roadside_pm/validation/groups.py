"""Canonical grouping helpers used before any train/evaluation split."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def require_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    """Raise a clear error when a table is missing identity columns."""
    available = set(columns)
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_group_key(
    row: Mapping[str, Any],
    columns: Iterable[str],
    *,
    separator: str = "::",
) -> str:
    """Build a stable group key without silently accepting missing values.

    Group keys should represent indivisible collection units such as dataset,
    trip, date, or spatial block. They must be created before model fitting.
    """
    columns = tuple(columns)
    require_columns(row.keys(), columns)

    values: list[str] = []
    for column in columns:
        value = row[column]
        if value is None or str(value).strip() == "":
            raise ValueError(f"Group column {column!r} has a missing value")
        values.append(str(value).strip())
    return separator.join(values)

