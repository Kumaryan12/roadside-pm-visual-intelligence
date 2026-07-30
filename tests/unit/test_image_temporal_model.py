import unittest

import numpy as np
import torch

from pipelines.image_embeddings.train_rnn import inverse_targets, transform_targets
from roadside_pm.modeling.image_temporal import ImageTemporalRegressor


class ImageTemporalModelTests(unittest.TestCase):
    def test_gru_and_lstm_shapes(self):
        inputs = torch.randn(4, 7, 32)
        for cell in ("gru", "lstm"):
            model = ImageTemporalRegressor(32, 2, cell=cell, hidden_dim=16)
            self.assertEqual(tuple(model(inputs).shape), (4, 2))

    def test_tabular_fusion_shape(self):
        model = ImageTemporalRegressor(32, 1, tabular_dim=5, hidden_dim=16)
        output = model(torch.randn(4, 7, 32), torch.randn(4, 5))
        self.assertEqual(tuple(output.shape), (4, 1))

    def test_log_target_transform_round_trip_and_nonnegative_predictions(self):
        values = np.array([[0.0], [10.0], [500.0]], dtype="float32")
        recovered = inverse_targets(transform_targets(values, "log1p"), "log1p")
        np.testing.assert_allclose(recovered, values, rtol=1e-5)
        self.assertGreaterEqual(float(inverse_targets(np.array([[-1.0]]), "log1p")[0, 0]), 0.0)

    def test_log_target_rejects_negative_truth(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            transform_targets(np.array([[-0.1]], dtype="float32"), "log1p")


if __name__ == "__main__": unittest.main()
