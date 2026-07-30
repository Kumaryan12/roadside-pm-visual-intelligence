import unittest

import pandas as pd
import torch

from pipelines.pm25_prediction.mumma_7day.run_multimodal_temporal import add_causal_target
from roadside_pm.modeling.multimodal_temporal import GatedMultiViewTemporalRegressor


class MultimodalTemporalTests(unittest.TestCase):
    def test_model_shape_and_attention_sum(self):
        model = GatedMultiViewTemporalRegressor(8, 5, image_hidden_dim=6,
                                                tabular_hidden_dim=4,
                                                temporal_hidden_dim=7,
                                                dropout=0.0)
        prediction, attention = model(
            torch.randn(2, 3, 3, 8), torch.randn(2, 3, 5),
            return_attention=True,
        )
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertEqual(tuple(attention.shape), (2, 3, 3))
        self.assertTrue(torch.allclose(attention.sum(dim=2), torch.ones(2, 3)))

    def test_trailing_target_never_crosses_run(self):
        frame = pd.DataFrame({
            "sample_id": range(6),
            "run_id": ["a"] * 3 + ["b"] * 3,
            "sample_timestamp": pd.date_range("2026-01-01", periods=6, freq="10s"),
            "sPM2": [1, 2, 3, 100, 200, 300],
        })
        result, target = add_causal_target(frame, "sPM2", 3)
        self.assertEqual(result[target].tolist(), [2.0, 200.0])


if __name__ == "__main__":
    unittest.main()
