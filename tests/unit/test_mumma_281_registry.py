import unittest

from pipelines.pm25_prediction.mumma_281.stages import STAGES, validate_stage_registry


class Mumma281RegistryTests(unittest.TestCase):
    def test_all_historical_stages_exist(self):
        validate_stage_registry()
        self.assertEqual(len(STAGES), 9)

    def test_high_risk_random_stages_are_labeled(self):
        for name in ("temporal_t7_random2fold", "resnet_t7_random2fold"):
            self.assertIn("not evidence of generalization", STAGES[name].validation_note)


if __name__ == "__main__":
    unittest.main()

