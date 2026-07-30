"""Dependency-free split overlap checks.

These checks are deliberately small so pipeline entry points can fail before
expensive feature extraction or model training begins.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping


def find_split_overlap(
    split_values: Mapping[str, Iterable[Hashable]],
) -> dict[tuple[str, str], set[Hashable]]:
    """Return values shared by every pair of named splits."""
    normalized = {name: set(values) for name, values in split_values.items()}
    names = sorted(normalized)
    overlaps: dict[tuple[str, str], set[Hashable]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = normalized[left] & normalized[right]
            if shared:
                overlaps[(left, right)] = shared
    return overlaps


def assert_disjoint_values(
    split_values: Mapping[str, Iterable[Hashable]],
    *,
    value_name: str = "values",
    preview_limit: int = 10,
) -> None:
    """Fail when raw frames, groups, timestamps, or windows cross splits."""
    overlaps = find_split_overlap(split_values)
    if not overlaps:
        return

    details = []
    for (left, right), shared in overlaps.items():
        preview = sorted(map(str, shared))[:preview_limit]
        details.append(f"{left}<->{right}: {len(shared)} shared ({preview})")
    raise ValueError(f"Split leakage detected for {value_name}: " + "; ".join(details))


def invert_memberships(
    split_values: Mapping[str, Iterable[Hashable]],
) -> dict[Hashable, set[str]]:
    """Return split memberships for audit/report generation."""
    memberships: dict[Hashable, set[str]] = defaultdict(set)
    for split_name, values in split_values.items():
        for value in values:
            memberships[value].add(split_name)
    return dict(memberships)

