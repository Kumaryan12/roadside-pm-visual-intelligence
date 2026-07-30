import unittest

import numpy as np
import pandas as pd

from pipelines.particle_source_attribution.run_particle_regimes import (
    FRACTION_TARGETS,
    derive_targets,
)


class ParticleRegimeTests(unittest.TestCase):
    def test_size_fractions_close_and_negative_increment_is_clipped(self):
        frame = pd.DataFrame({
            "sPM1": [8.0], "sPM2": [9.0], "sPM4": [8.5], "sPM10": [10.0],
            "sNPM2": [100.0], "sTPS": [0.5],
        })
        target, audit = derive_targets(frame)
        self.assertEqual(audit["negative_increment_values"], 1)
        self.assertAlmostEqual(float(target[FRACTION_TARGETS].sum(axis=1).iloc[0]), 1.0)
        self.assertTrue((target[FRACTION_TARGETS] >= 0).all().all())

    def test_effective_targets_are_emitted(self):
        frame = pd.DataFrame({
            "sPM1": [8.0], "sPM2": [9.0], "sPM4": [9.5], "sPM10": [10.0],
            "sNPM2": [100.0], "sTPS": [0.5],
        })
        target, _ = derive_targets(frame)
        self.assertTrue(np.isfinite(target.effective_density_proxy_kg_m3.iloc[0]))
        self.assertEqual(target.effective_diameter_um.iloc[0], 0.5)


if __name__ == "__main__":
    unittest.main()
