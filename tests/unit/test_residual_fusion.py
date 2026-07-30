import unittest

import numpy as np
import pandas as pd

from roadside_pm.modeling.residual_fusion import build_residual_dataset, select_numeric_residual_features, validate_base_predictions


class ResidualFusionTests(unittest.TestCase):
    def test_in_sample_predictions_are_rejected(self):
        frame = pd.DataFrame({"sequence_id": [1], "actual_pm25": [10.0], "predicted_pm25": [8.0], "prediction_origin": ["in_sample"]})
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            validate_base_predictions(frame, target="pm25")

    def test_sequence_predictions_join_target_sample_features(self):
        predictions = pd.DataFrame({"sequence_id": [1], "actual_pm25": [10.0], "predicted_pm25": [8.0], "prediction_origin": ["oof"]})
        sequences = pd.DataFrame({"sequence_id": [1], "target_sample_id": [7], "trip": ["a"]})
        features = pd.DataFrame({"sample_index": [7], "veh_count": [3.0], "target_pm25": [10.0]})
        result = build_residual_dataset(predictions, sequences, features, target="pm25", sample_key="sample_index", group_column="trip")
        self.assertEqual(result.loc[0, "base_residual"], 2.0)
        selected = select_numeric_residual_features(result, key_columns=["sequence_id", "target_sample_id", "sample_index", "trip"])
        self.assertEqual(selected, ["veh_count"])


if __name__ == "__main__": unittest.main()
