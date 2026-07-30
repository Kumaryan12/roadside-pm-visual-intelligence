import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipelines.pm25_prediction.mumma_7day.model_multimodal_current_data import (
    validated_alphaearth,
    validated_osm,
)
from roadside_pm.features.geospatial.alphaearth import BANDS


class MummaOsmModelingTests(unittest.TestCase):
    def write_csv(self, frame: pd.DataFrame) -> Path:
        temporary = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        temporary.close()
        path = Path(temporary.name)
        frame.to_csv(path, index=False)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_requires_complete_successful_sample_coverage(self):
        path = self.write_csv(pd.DataFrame({
            "sample_id": ["a", "b"],
            "osm_status": ["success", "success"],
            "osm_error": ["", ""],
            "road_segment_count_250m": [3, 4],
        }))
        table, features = validated_osm(path, {"a", "b"})
        self.assertEqual(features, ["road_segment_count_250m"])
        self.assertEqual(len(table), 2)

    def test_rejects_failed_rows(self):
        path = self.write_csv(pd.DataFrame({
            "sample_id": ["a"],
            "osm_status": ["error"],
            "osm_error": ["timeout"],
            "road_segment_count_250m": [0],
        }))
        with self.assertRaisesRegex(ValueError, "failed rows"):
            validated_osm(path, {"a"})

    def test_alphaearth_requires_all_complete_embedding_bands(self):
        rows = []
        for sample_id in ("a", "b"):
            rows.append({
                "sample_id": sample_id,
                "alphaearth_status": "success",
                **{band: 0.1 for band in BANDS},
            })
        path = self.write_csv(pd.DataFrame(rows))
        table, features = validated_alphaearth(path, {"a", "b"})
        self.assertEqual(features, [f"alphaearth_{band}" for band in BANDS])
        self.assertEqual(table.shape, (2, 65))

    def test_alphaearth_rejects_incomplete_coverage(self):
        path = self.write_csv(pd.DataFrame([{
            "sample_id": "a",
            "alphaearth_status": "success",
            **{band: 0.1 for band in BANDS},
        }]))
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validated_alphaearth(path, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
