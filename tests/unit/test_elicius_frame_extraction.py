import unittest

import pandas as pd

from pipelines.pm25_prediction.mumma_7day.extract_frames import build_extraction_jobs


class EliciusFrameExtractionTests(unittest.TestCase):
    def test_build_jobs_joins_every_lens_and_computes_offset(self):
        sensor = pd.DataFrame(
            {
                "sample_id": ["run_a_000010"],
                "run_id": ["run_a"],
                "sample_timestamp": ["2026-02-01 10:00:10"],
            }
        )
        videos = pd.DataFrame(
            {
                "run_id": ["run_a", "run_a", "run_a"],
                "lens_id": [1, 4, 6],
                "video_path": ["one.mp4", "four.mp4", "six.mp4"],
                "start_time_ist": ["2026-02-01 10:00:00"] * 3,
                "end_time_ist": ["2026-02-01 10:01:00"] * 3,
                "timezone": ["Asia/Kolkata"] * 3,
            }
        )

        jobs = build_extraction_jobs(sensor, videos, (1, 4, 6), 5.0)

        self.assertEqual(jobs["lens_id"].tolist(), [1, 4, 6])
        self.assertEqual(jobs["video_offset_seconds"].tolist(), [15.0, 15.0, 15.0])
        self.assertEqual(jobs["video_offset_sec"].tolist(), [15.0, 15.0, 15.0])
        self.assertEqual(jobs["sample_index"].tolist(), [0, 0, 0])
        self.assertEqual(jobs["sample_unix"].tolist(), [1769920210.0] * 3)
        self.assertTrue(jobs["within_declared_run"].all())


if __name__ == "__main__":
    unittest.main()
