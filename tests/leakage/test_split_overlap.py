import unittest

from roadside_pm.validation.overlap import assert_disjoint_values, find_split_overlap


class SplitOverlapTests(unittest.TestCase):
    def test_disjoint_splits_pass(self):
        assert_disjoint_values({"train": [1, 2], "test": [3, 4]}, value_name="frames")

    def test_overlap_is_reported(self):
        overlaps = find_split_overlap({"train": [1, 2], "val": [2, 3], "test": [4]})
        self.assertEqual(overlaps[("train", "val")], {2})

    def test_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            assert_disjoint_values(
                {"train": ["frame-a"], "test": ["frame-a"]},
                value_name="raw frames",
            )


if __name__ == "__main__":
    unittest.main()

