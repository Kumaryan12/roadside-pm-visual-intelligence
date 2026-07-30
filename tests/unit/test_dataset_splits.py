import unittest
import pandas as pd
from roadside_pm.validation.dataset_splits import assign_evaluation_protocols

class DatasetSplitTests(unittest.TestCase):
    def test_whole_days_and_groups_are_not_split(self):
        rows=[]
        for day in range(1,8):
            for i in range(3): rows.append({"date":f"2026-01-{day:02d}","trip":f"trip-{day}","lat":28.60+day*.001,"lon":77.20+i*.0001})
        out=assign_evaluation_protocols(pd.DataFrame(rows),date_column="date",trip_column="trip",latitude_column="lat",longitude_column="lon",spatial_block_size_m=100)
        self.assertTrue((out.groupby("evaluation_date").split_chronological_day.nunique()==1).all())
        self.assertTrue((out.groupby("trip").split_grouped_trip.nunique()==1).all())
        self.assertTrue((out.groupby("spatial_block_id").split_grouped_spatial.nunique()==1).all())
        self.assertEqual(out.logo_day_fold.nunique(),7)
        self.assertEqual(set(out.split_grouped_trip), {"train", "val", "test"})
if __name__ == "__main__": unittest.main()
