import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipelines.image_embeddings.run_background_increment_fair import convert_predictions
from pipelines.shared.extract_atmospheric_background import relative_humidity_percent


class BackgroundIncrementTests(unittest.TestCase):
    def test_adds_external_background_back_to_local_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                "sequence_id": [7, 8],
                "prediction_origin": ["outer_test", "outer_test"],
                "actual_sPM2_local_increment": [80.0, 30.0],
                "predicted_sPM2_local_increment": [75.0, 40.0],
            }).to_csv(root / "predictions_test.csv", index=False)
            pd.DataFrame({
                "sequence_id": [7, 8], "target_sample_id": ["sample-a", "sample-b"],
            }).to_csv(root / "sequences.csv", index=False)
            background = pd.DataFrame({
                "sample_id": ["sample-a", "sample-b"],
                "background_cams_pm25_ug_m3": [40.0, 40.0],
            })
            total_metrics, _, result = convert_predictions(
                root, root / "sequences.csv", background, split="test",
                target="sPM2", background_col="background_cams_pm25_ug_m3",
            )
            self.assertEqual(result.loc[0, "actual_sPM2"], 120.0)
            self.assertEqual(result.loc[0, "predicted_sPM2"], 115.0)
            self.assertEqual(total_metrics["MAE"], 7.5)

    def test_relative_humidity_is_bounded(self):
        self.assertAlmostEqual(relative_humidity_percent(293.15, 293.15), 100.0)
        self.assertGreater(relative_humidity_percent(303.15, 293.15), 0.0)
        self.assertLess(relative_humidity_percent(303.15, 293.15), 100.0)


if __name__ == "__main__":
    unittest.main()
