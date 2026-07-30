"""Extract annual AlphaEarth point and buffer embeddings with Earth Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from roadside_pm.features.geospatial.alphaearth import BANDS, DATASET_ID, measurement_years


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--lat-col", default="value.lat")
    parser.add_argument("--lon-col", default="value.long")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--sample-col", default="sample_index")
    parser.add_argument("--project", default=None, help="Earth Engine/Google Cloud project.")
    parser.add_argument("--fallback-year", type=int, default=None)
    parser.add_argument("--max-available-year", type=int, default=2025)
    parser.add_argument("--buffers-m", nargs="*", type=int, default=[50, 100, 250])
    parser.add_argument("--scale-m", type=int, default=10)
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--cache-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Install the geospatial extra: pip install earthengine-api") from exc
    try:
        ee.Initialize(project=args.project)
    except Exception as exc:
        raise RuntimeError("Earth Engine initialization failed. Authenticate with `earthengine authenticate` and provide --project if required.") from exc
    source = pd.read_csv(args.sensor_csv)
    if args.sample_col not in source: source[args.sample_col] = source.index
    if args.limit is not None: source = source.head(args.limit).copy()
    source["_aef_year"] = measurement_years(source[args.timestamp_col], args.fallback_year)
    output = Path(args.output_csv)
    cache_path = Path(args.cache_json) if args.cache_json else output.with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    rows = []
    for _, record in source.iterrows():
        lat, lon = float(record[args.lat_col]), float(record[args.lon_col])
        measurement_year = int(record["_aef_year"])
        year = min(measurement_year, args.max_available_year)
        rounded_lat, rounded_lon = round(lat, args.round_decimals), round(lon, args.round_decimals)
        cache_key = f"{year}:{rounded_lat:.{args.round_decimals}f}:{rounded_lon:.{args.round_decimals}f}:{args.scale_m}:{','.join(map(str, args.buffers_m))}"
        identity = {
            "sample_id": record[args.sample_col],
            "sample_index": record.get("sample_index", record[args.sample_col]),
        }
        if cache_key in cache:
            row = {**identity, **cache[cache_key]}
        else:
            base = {"alphaearth_measurement_year": measurement_year, "alphaearth_year": year, "alphaearth_year_lag": measurement_year - year, "alphaearth_dataset": DATASET_ID}
            try:
                point = ee.Geometry.Point([rounded_lon, rounded_lat])
                image = ee.ImageCollection(DATASET_ID).filterDate(f"{year}-01-01", f"{year + 1}-01-01").filterBounds(point).mosaic().select(BANDS)
                reductions = [image.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=args.scale_m, maxPixels=10000)]
                reductions.extend(
                    image.reduceRegion(reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True), geometry=point.buffer(radius), scale=args.scale_m, maxPixels=1_000_000)
                    for radius in args.buffers_m
                )
                results = ee.List(reductions).getInfo()
                point_values = results[0]
                features = {band: point_values.get(band) for band in BANDS}
                for radius, stats in zip(args.buffers_m, results[1:], strict=True):
                    for band in BANDS:
                        features[f"{band}_mean_{radius}m"] = stats.get(f"{band}_mean")
                        features[f"{band}_std_{radius}m"] = stats.get(f"{band}_stdDev")
                cached_row = {**base, "alphaearth_status": "success", "alphaearth_error": "", **features}
                cache[cache_key] = cached_row
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=2) + "\n")
                row = {**identity, **cached_row}
            except Exception as exc:
                row = {**identity, **base, "alphaearth_status": "error", "alphaearth_error": str(exc)}
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved AlphaEarth features for {len(rows)} samples: {output}")


if __name__ == "__main__": main()
