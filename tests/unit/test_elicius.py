import unittest
import pandas as pd
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from roadside_pm.data.elicius import _choose_video, resample_aqi_run

class EliciusTests(unittest.TestCase):
    @patch("roadside_pm.data.elicius.read_aqi_export")
    def test_resampling_clips_to_metadata_and_aggregates_ten_seconds(self, read):
        read.return_value=pd.DataFrame({"timestamp_local":pd.date_range("2026-02-01 10:00:00",periods=31,freq="1s"),"sPM2":range(31),"lat":[13.0]*31,"long":[80.2]*31})
        output,audit=resample_aqi_run(Path("unused.csv"),run_id="run1",date="2026-02-01",trip_id="trip1",start_time="2026-02-01 10:00:05",end_time="2026-02-01 10:00:24")
        self.assertEqual(len(output),2); self.assertEqual(audit["clipped_rows"],20); self.assertEqual(audit["dropped_outside_bounds"],11)
        self.assertEqual(output.raw_observation_count.tolist(),[10,10]); self.assertEqual(output.sample_id.tolist(),["run1_000000","run1_000010"])

    def test_video_discovery_ignores_macos_appledouble_sidecar(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            lens = run / "LENS1"
            lens.mkdir()
            (lens / "._video_lens1.mp4").write_bytes(b"sidecar")
            video = lens / "video_lens1.mp4"
            video.write_bytes(b"video")

            chosen, count, status = _choose_video(run, 1)

            self.assertEqual(chosen, str(video))
            self.assertEqual(count, 1)
            self.assertEqual(status, "present")
if __name__ == "__main__": unittest.main()
