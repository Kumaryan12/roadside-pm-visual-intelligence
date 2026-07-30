"""Deterministic group-level split assignment primitives."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Iterable, Mapping


def deterministic_group_split(
    groups: Iterable[Hashable],
    fractions: Mapping[str, float],
    *,
    seed: int = 42,
) -> dict[Hashable, str]:
    """Assign whole groups by stable hash.

    This is a foundation utility, not a replacement for chronological or
    spatially buffered evaluation. Fractions must be positive and sum to one.
    """
    if not fractions:
        raise ValueError("At least one split fraction is required")
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("All split fractions must be positive")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    cumulative: list[tuple[str, float]] = []
    cursor = 0.0
    for name, fraction in fractions.items():
        cursor += fraction
        cumulative.append((name, cursor))

    assignments: dict[Hashable, str] = {}
    for group in sorted(set(groups), key=str):
        digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
        unit_value = int.from_bytes(digest[:8], "big") / 2**64
        assignments[group] = cumulative[-1][0]
        for name, boundary in cumulative:
            if unit_value < boundary:
                assignments[group] = name
                break
    return assignments


def balanced_deterministic_group_split(
    groups: Iterable[Hashable], fractions: Mapping[str, float], *, seed: int = 42
) -> dict[Hashable, str]:
    """Assign complete groups with deterministic, near-proportional split counts.

    Unlike independent hash thresholds, this guarantees every requested split
    is non-empty when there are at least as many unique groups as splits.
    """
    unique = sorted(set(groups), key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    if len(unique) < len(fractions): raise ValueError("Need at least one unique group per requested split")
    if any(value <= 0 for value in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must be positive and sum to 1.0")
    names = list(fractions); counts = {name: 1 for name in names}; remaining = len(unique) - len(names)
    ideal_extra = {name: max(0.0, fractions[name] * len(unique) - 1) for name in names}
    for name in names:
        take = min(remaining, math.floor(ideal_extra[name])); counts[name] += take; remaining -= take
    ranking = sorted(names, key=lambda name: (ideal_extra[name] - math.floor(ideal_extra[name]), fractions[name]), reverse=True)
    for index in range(remaining): counts[ranking[index % len(ranking)]] += 1
    assignments = {}; cursor = 0
    for name in names:
        for group in unique[cursor:cursor + counts[name]]: assignments[group] = name
        cursor += counts[name]
    return assignments
