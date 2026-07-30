import unittest

import numpy as np
import pandas as pd

from pipelines.particle_source_attribution.run_assumption_scenarios import (
    incremental_mass,
    infer_scenarios,
)


class AssumptionScenarioTests(unittest.TestCase):
    def test_incremental_mass_and_clipping_audit(self):
        frame = pd.DataFrame({
            "sPM1": [5.0], "sPM2": [7.0], "sPM4": [6.0], "sPM10": [10.0],
        })
        mass, audit = incremental_mass(frame)
        np.testing.assert_allclose(mass, [[5.0, 2.0, 0.0, 4.0]])
        self.assertEqual(audit["negative_increment_values"], 1)

    def test_monte_carlo_fractions_close(self):
        mass = np.array([[8.0, 1.0, 0.5, 0.5], [1.0, 2.0, 3.0, 4.0]])
        profiles = np.array([[0.8, 0.1, 0.05, 0.05], [0.1, 0.2, 0.3, 0.4]])
        median, lower, upper, _ = infer_scenarios(
            mass, profiles, np.array([100.0, 100.0]), draws=4, seed=42,
        )
        self.assertTrue((median >= 0).all())
        np.testing.assert_allclose(median.sum(axis=1), 1.0)
        self.assertTrue((upper >= lower).all())


if __name__ == "__main__":
    unittest.main()
