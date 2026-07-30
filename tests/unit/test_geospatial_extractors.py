import unittest

import pandas as pd

from roadside_pm.features.geospatial.alphaearth import BANDS, measurement_years
from roadside_pm.features.geospatial.osm import (
    build_overpass_bbox_query,
    build_overpass_query,
    summarize_elements,
    summarize_nearby_elements,
)


class GeospatialExtractorTests(unittest.TestCase):
    def test_alphaearth_has_all_64_bands(self):
        self.assertEqual(len(BANDS), 64)
        self.assertEqual((BANDS[0], BANDS[-1]), ("A00", "A63"))

    def test_measurement_year_is_derived_from_timestamp(self):
        result = measurement_years(pd.Series(["23-02-2026 11:54"]))
        self.assertEqual(result.iloc[0], 2026)

    def test_overpass_query_contains_radius_and_roads(self):
        query = build_overpass_query(19.1, 72.9, 250)
        self.assertIn("around:250,19.1,72.9", query)
        self.assertIn('["highway"]', query)

    def test_overpass_bbox_query_contains_bounds_and_roads(self):
        query = build_overpass_bbox_query(12.9, 80.1, 13.0, 80.2)
        self.assertIn("(12.9,80.1,13.0,80.2)", query)
        self.assertIn('way["highway"]', query)

    def test_osm_summary_counts_and_lengths(self):
        elements = [
            {"tags": {"amenity": "fuel"}},
            {"tags": {"highway": "primary"}, "geometry": [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 0.001}]},
        ]
        result = summarize_elements(elements, 250)
        self.assertEqual(result["fuel_station_count_250m"], 1)
        self.assertEqual(result["primary_road_count_250m"], 1)
        self.assertGreater(result["total_road_length_250m"], 100)

    def test_local_radius_summary_excludes_far_tile_elements(self):
        elements = [
            {"id": 1, "type": "node", "lat": 13.0, "lon": 80.0, "tags": {"amenity": "fuel"}},
            {"id": 2, "type": "node", "lat": 13.1, "lon": 80.1, "tags": {"amenity": "fuel"}},
            {
                "id": 3,
                "type": "way",
                "tags": {"highway": "primary"},
                "geometry": [{"lat": 13.0, "lon": 80.0}, {"lat": 13.0, "lon": 80.001}],
            },
        ]
        result = summarize_nearby_elements(elements, (13.0, 80.0), 250)
        self.assertEqual(result["fuel_station_count_250m"], 1)
        self.assertEqual(result["primary_road_count_250m"], 1)


if __name__ == "__main__":
    unittest.main()
