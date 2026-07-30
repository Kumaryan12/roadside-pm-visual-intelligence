import unittest

import pandas as pd

from roadside_pm.features.geospatial.merge import merge_sample_features


class GeospatialMergeTests(unittest.TestCase):
    def test_alphaearth_columns_are_prefixed(self):
        base = pd.DataFrame({"sample_index": [1, 2], "pm25": [10.0, 20.0]})
        features = pd.DataFrame({"sample_index": [1, 2], "A00": [0.1, 0.2]})
        result = merge_sample_features(base, features, feature_prefix="aef_")
        self.assertIn("aef_A00", result.columns)

    def test_duplicate_features_fail(self):
        base = pd.DataFrame({"sample_index": [1]})
        features = pd.DataFrame({"sample_index": [1, 1], "A00": [0.1, 0.2]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge_sample_features(base, features)


if __name__ == "__main__":
    unittest.main()

