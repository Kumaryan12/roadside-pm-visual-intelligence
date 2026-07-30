"""Leakage-aware grouping, split, overlap, and validation contracts."""

from .groups import build_group_key, require_columns
from .overlap import assert_disjoint_values, find_split_overlap

__all__ = [
    "assert_disjoint_values",
    "build_group_key",
    "find_split_overlap",
    "require_columns",
]

