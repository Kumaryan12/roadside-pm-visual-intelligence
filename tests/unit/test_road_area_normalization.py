import unittest

import pandas as pd

from roadside_pm.features.road.area import normalize_depth_gated_v3


class RoadAreaNormalizationTests(unittest.TestCase):
    def test_v3_is_mapped_to_canonical_columns(self):
        source = pd.DataFrame(
            {
                "visible_road_area_m2_depth_est": [20.0],
                "road_area_m2_occlusion_adjusted_conservative_v3": [25.0],
                "road_area_vehicle_occlusion_fraction_v3": [0.2],
                "vehicle_occlusion_quality_v3": ["good"],
            }
        )
        result = normalize_depth_gated_v3(source)
        self.assertEqual(result.loc[0, "road_area_vehicle_occluded_m2"], 5.0)
        self.assertEqual(result.loc[0, "road_area_effective_visible_m2"], 25.0)
        self.assertTrue(result.loc[0, "road_area_features_available"])

    def test_missing_columns_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            normalize_depth_gated_v3(pd.DataFrame({"x": [1]}))


if __name__ == "__main__":
    unittest.main()

