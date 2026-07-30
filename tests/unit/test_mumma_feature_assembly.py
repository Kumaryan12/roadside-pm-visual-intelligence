import unittest

import pandas as pd

from pipelines.pm25_prediction.mumma_7day.assemble_feature_table import (
    aggregate_road_features,
    aggregate_vehicle_features,
)


class MummaFeatureAssemblyTests(unittest.TestCase):
    def test_vehicle_cross_view_uses_max_not_sum(self):
        frame = pd.DataFrame(
            {
                "sample_id": ["a", "a"],
                "lens_id": [1, 6],
                "idd_detection_status": ["success", "success"],
                "idd_car_count": [2, 2],
            }
        )
        result = aggregate_vehicle_features(frame)
        self.assertEqual(result.loc[0, "vehicle_cross_view_max_idd_car_count"], 2)
        self.assertEqual(result.loc[0, "vehicle_lens1_idd_car_count"], 2)
        self.assertEqual(result.loc[0, "vehicle_lens6_idd_car_count"], 2)

    def test_road_failure_remains_missing(self):
        frame = pd.DataFrame(
            {
                "processed_frame_key": ["f1", "f6"],
                "lens_id": [1, 6],
                "road_condition_status": ["failed", "success"],
                "road_area_ratio": [None, 0.2],
            }
        )
        manifest = pd.DataFrame(
            {"processed_frame_key": ["f1", "f6"], "sample_id": ["a", "a"]}
        )
        result = aggregate_road_features(frame, manifest)
        self.assertTrue(pd.isna(result.loc[0, "road_lens1_road_area_ratio"]))
        self.assertEqual(result.loc[0, "road_cross_view_mean_road_area_ratio"], 0.2)
        self.assertEqual(result.loc[0, "road_valid_lens_count"], 1)


if __name__ == "__main__":
    unittest.main()
