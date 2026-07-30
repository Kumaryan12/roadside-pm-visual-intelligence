import unittest

import pandas as pd

from roadside_pm.validation.outer_folds import make_grouped_outer_folds


class OuterFoldTests(unittest.TestCase):
    def test_dates_do_not_cross_train_val_test(self):
        frame = pd.DataFrame({"date": sum(([f"d{i}"] * 3 for i in range(6)), []), "x": range(18)})
        folds = make_grouped_outer_folds(frame, group_column="date", n_splits=3)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            groups = {name: set(fold.loc[fold.outer_split == name, "date"]) for name in ("train", "val", "test")}
            self.assertFalse(groups["train"] & groups["val"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["val"] & groups["test"])


if __name__ == "__main__": unittest.main()
