import unittest

import pandas as pd

from roadside_pm.features.images.sequences import build_grouped_sequences


class ImageSequenceTests(unittest.TestCase):
    def test_windows_never_cross_split_or_trip(self):
        rows = pd.DataFrame({
            "sample_index": range(8), "embedding_row": range(8),
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="min"),
            "trip_id": ["a"] * 4 + ["b"] * 4,
            "split": ["train"] * 3 + ["test"] + ["test"] * 4,
            "pm25": range(8),
        })
        result = build_grouped_sequences(rows, sequence_length=3, id_column="sample_index", embedding_row_column="embedding_row", timestamp_column="timestamp", group_columns=["trip_id"], split_column="split", target_columns=["pm25"])
        self.assertEqual(len(result), 3)
        for _, sequence in result.iterrows():
            ids = [int(value) for value in sequence["sample_ids"].split("|")]
            source = rows[rows.sample_index.isin(ids)]
            self.assertEqual(source["split"].nunique(), 1)
            self.assertEqual(source["trip_id"].nunique(), 1)

    def test_iso_timestamp_is_not_interpreted_as_day_first(self):
        rows = pd.DataFrame({
            "sample_index": [0],
            "embedding_row": [0],
            "timestamp": ["2026-02-01 08:00:00"],
            "trip_id": ["a"],
            "split": ["test"],
            "pm25": [10.0],
        })
        result = build_grouped_sequences(
            rows,
            sequence_length=1,
            id_column="sample_index",
            embedding_row_column="embedding_row",
            timestamp_column="timestamp",
            group_columns=["trip_id"],
            split_column="split",
            target_columns=["pm25"],
        )
        self.assertEqual(pd.Timestamp(result.loc[0, "target_time"]), pd.Timestamp("2026-02-01 08:00:00"))


if __name__ == "__main__": unittest.main()
