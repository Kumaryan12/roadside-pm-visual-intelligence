"""Leakage-safe temporal sequence construction."""

from __future__ import annotations

import pandas as pd


def build_grouped_sequences(
    rows: pd.DataFrame,
    *,
    sequence_length: int,
    id_column: str,
    embedding_row_column: str,
    timestamp_column: str,
    group_columns: list[str],
    split_column: str,
    target_columns: list[str],
) -> pd.DataFrame:
    required = [id_column, embedding_row_column, timestamp_column, split_column, *group_columns, *target_columns]
    missing = [column for column in required if column not in rows]
    if missing: raise ValueError(f"Missing sequence columns: {missing}")
    if sequence_length < 1: raise ValueError("sequence_length must be positive")
    if rows[id_column].duplicated().any(): raise ValueError(f"Duplicate {id_column!r}")
    work = rows.copy()
    raw_timestamp = work[timestamp_column].astype("string")
    iso_timestamp = raw_timestamp.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]|$)", na=False)
    parsed_timestamp = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
    parsed_timestamp.loc[iso_timestamp] = pd.to_datetime(
        raw_timestamp.loc[iso_timestamp], errors="coerce", format="mixed", dayfirst=False,
    )
    parsed_timestamp.loc[~iso_timestamp] = pd.to_datetime(
        raw_timestamp.loc[~iso_timestamp], errors="coerce", format="mixed", dayfirst=True,
    )
    work[timestamp_column] = parsed_timestamp
    if work[timestamp_column].isna().any(): raise ValueError("Unparseable timestamps")
    sequences = []
    grouping = [*group_columns, split_column]
    sequence_id = 0
    for keys, group in work.groupby(grouping, sort=True, dropna=False):
        group = group.sort_values([timestamp_column, id_column], kind="stable").reset_index(drop=True)
        for end in range(sequence_length - 1, len(group)):
            window = group.iloc[end - sequence_length + 1:end + 1]
            target = window.iloc[-1]
            row = {
                "sequence_id": sequence_id,
                "sequence_length": sequence_length,
                "sample_ids": "|".join(map(str, window[id_column])),
                "embedding_rows": "|".join(map(str, window[embedding_row_column].astype(int))),
                "target_sample_id": target[id_column],
                "target_time": target[timestamp_column],
                split_column: target[split_column],
            }
            for column, value in zip(grouping, keys if isinstance(keys, tuple) else (keys,)):
                row[column] = value
            for target_column in target_columns: row[target_column] = target[target_column]
            sequences.append(row); sequence_id += 1
    return pd.DataFrame(sequences)
