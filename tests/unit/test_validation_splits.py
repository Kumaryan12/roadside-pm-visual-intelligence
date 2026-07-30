import unittest

from roadside_pm.validation.splits import balanced_deterministic_group_split, deterministic_group_split


class DeterministicSplitTests(unittest.TestCase):
    def test_assignment_is_stable_and_group_level(self):
        fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
        first = deterministic_group_split(["trip-b", "trip-a", "trip-a"], fractions)
        second = deterministic_group_split(["trip-a", "trip-b"], fractions)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"trip-a", "trip-b"})

    def test_invalid_fractions_fail(self):
        with self.assertRaisesRegex(ValueError, "sum"):
            deterministic_group_split(["trip-a"], {"train": 0.8, "test": 0.3})

    def test_balanced_assignment_keeps_all_splits_nonempty(self):
        assignment = balanced_deterministic_group_split(
            [f"trip-{index}" for index in range(7)], {"train": 0.7, "val": 0.15, "test": 0.15}
        )
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})
        self.assertEqual(len(assignment), 7)


if __name__ == "__main__":
    unittest.main()
