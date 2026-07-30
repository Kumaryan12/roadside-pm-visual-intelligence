import unittest

from roadside_pm.validation.groups import build_group_key, require_columns


class GroupTests(unittest.TestCase):
    def test_build_group_key(self):
        row = {"dataset_id": "mumma", "trip_id": "trip-1", "date": "2026-01-01"}
        self.assertEqual(
            build_group_key(row, ["dataset_id", "trip_id", "date"]),
            "mumma::trip-1::2026-01-01",
        )

    def test_missing_column_fails(self):
        with self.assertRaisesRegex(ValueError, "trip_id"):
            require_columns(["sample_id"], ["sample_id", "trip_id"])

    def test_empty_group_value_fails(self):
        with self.assertRaisesRegex(ValueError, "trip_id"):
            build_group_key({"trip_id": ""}, ["trip_id"])


if __name__ == "__main__":
    unittest.main()

