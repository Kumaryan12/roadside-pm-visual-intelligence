import unittest

from pipelines.shared.feature_table.stages import STAGES, STAGE_BY_NAME


class FeatureTableRegistryTests(unittest.TestCase):
    def test_stage_names_are_unique(self):
        names = [stage.name for stage in STAGES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(STAGE_BY_NAME), len(STAGES))

    def test_dependencies_precede_consumers(self):
        positions = {stage.name: index for index, stage in enumerate(STAGES)}
        for stage in STAGES:
            for dependency in stage.dependencies:
                self.assertLess(positions[dependency], positions[stage.name])

    def test_final_stage_is_alphaearth_merge(self):
        self.assertEqual(STAGES[-1].name, "merge_alphaearth")
        self.assertEqual(STAGES[-1].outputs, ("final_table",))


if __name__ == "__main__":
    unittest.main()

