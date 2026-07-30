import unittest
import pandas as pd
from roadside_pm.data.collection_audit import audit_sensor_table

class AuditTests(unittest.TestCase):
    def test_ten_second_sensor_audit(self):
        frame=pd.DataFrame({"time":pd.date_range("2026-01-01",periods=4,freq="10s"),"pm25":[10,11,12,13]})
        result=audit_sensor_table(frame,timestamp_column="time",target_column="pm25")
        self.assertEqual(result["median_interval_seconds"],10.0); self.assertEqual(result["gaps_over_2x_expected"],0)
    def test_missing_target_fails(self):
        with self.assertRaisesRegex(ValueError,"pm25"): audit_sensor_table(pd.DataFrame({"time":[]}),timestamp_column="time",target_column="pm25")
if __name__ == "__main__": unittest.main()
