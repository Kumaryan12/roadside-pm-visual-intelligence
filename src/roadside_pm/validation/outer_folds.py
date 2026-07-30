"""Outer grouped-fold assignment with validation groups drawn from training only."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_grouped_outer_folds(frame: pd.DataFrame, *, group_column: str, n_splits: int = 5, seed: int = 42) -> list[pd.DataFrame]:
    if group_column not in frame: raise ValueError(f"Missing group column {group_column!r}")
    groups = frame[group_column].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 3: raise ValueError("Need at least three groups for train/val/test")
    n_splits = min(n_splits, unique_groups)
    outputs = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_pool_index, test_index) in enumerate(splitter.split(frame, groups=groups), 1):
        train_pool_groups = sorted(set(groups.iloc[train_pool_index]))
        ranked = sorted(train_pool_groups, key=lambda value: hashlib.sha256(f"{seed + fold}:{value}".encode()).hexdigest())
        n_val = max(1, round(len(ranked) * 0.2)); val_groups = set(ranked[:n_val]); test_groups = set(groups.iloc[test_index])
        assignment = frame.copy(); assignment["outer_fold"] = fold; assignment["outer_split"] = "train"
        assignment.loc[groups.isin(val_groups), "outer_split"] = "val"; assignment.loc[groups.isin(test_groups), "outer_split"] = "test"
        if set(assignment.loc[assignment.outer_split == "train", group_column].astype(str)) & (val_groups | test_groups): raise AssertionError("Group leakage")
        outputs.append(assignment)
    return outputs

