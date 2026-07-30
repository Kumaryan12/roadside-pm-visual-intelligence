import unittest

import numpy as np
import pandas as pd

from pipelines.particle_source_attribution.run_source_proxy import (
    fractions,
    make_split,
    normalize_rows,
)


class SourceProxyTests(unittest.TestCase):
    def test_component_fractions_are_nonnegative_and_close(self):
        weights = np.array([[1.0, 2.0], [0.0, 0.0]])
        profiles = np.array([[1.0, 3.0], [2.0, 1.0]])
        result = fractions(weights, profiles, pm25_index=1)
        self.assertTrue((result >= 0).all())
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_context_predictions_are_constrained(self):
        result = normalize_rows(np.array([[-1.0, 2.0], [0.0, 0.0]]))
        self.assertTrue((result >= 0).all())
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_date_split_holds_out_complete_date(self):
        frame = pd.DataFrame({"date": ["a", "a", "b"]})
        split = make_split(frame, "date", "b", 0.2, 42)
        self.assertEqual(split.tolist(), ["train", "train", "test"])


if __name__ == "__main__":
    unittest.main()
