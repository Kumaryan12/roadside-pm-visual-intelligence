"""Extract cached AlphaEarth point embeddings in resumable Earth Engine batches.

This entry point is intended for route-scale datasets.  It submits one
FeatureCollection per batch instead of making one synchronous request per
sample.  Only the 64 annual point-embedding bands are extracted; neighbourhood
summaries should be treated as a separate, explicitly justified experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from roadside_pm.features.geospatial.alphaearth import BANDS, DATASET_ID, measurement_years


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--cache-json", required=True)
    parser.add_argument("--lat-col", default="value.lat")
    parser.add_argument("--lon-col", default="value.long")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--sample-col", default="sample_index")
    parser.add_argument("--project", default=None)
    parser.add_argument("--fallback-year", type=int, default=None)
    parser.add_argument("--max-available-year", type=int, default=2025)
    parser.add_argument("--scale-m", type=int, default=10)
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--tile-scale", type=float, default=4.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cache_key(year: int, latitude: float, longitude: float, scale_m: int, decimals: int) -> str:
    return f"{year}:{latitude:.{decimals}f}:{longitude:.{decimals}f}:{scale_m}:point"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def chunks(frame: pd.DataFrame, size: int):
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


def fetch_batch(ee, batch: pd.DataFrame, *, scale_m: int, tile_scale: float) -> dict[str, dict[str, Any]]:
    """Return band properties keyed by the internal AlphaEarth cache key."""
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row.longitude), float(row.latitude)]),
            {"_alphaearth_cache_key": row.cache_key},
        )
        for row in batch.itertuples(index=False)
    ]
    collection = ee.FeatureCollection(features)
    responses: dict[str, dict[str, Any]] = {}
    for year, year_rows in batch.groupby("alphaearth_year", sort=True):
        keys = year_rows["cache_key"].tolist()
        subset = collection.filter(ee.Filter.inList("_alphaearth_cache_key", keys))
        image = (
            ee.ImageCollection(DATASET_ID)
            .filterDate(f"{int(year)}-01-01", f"{int(year) + 1}-01-01")
            .mosaic()
            .select(BANDS)
        )
        sampled = image.sampleRegions(
            collection=subset,
            properties=["_alphaearth_cache_key"],
            scale=scale_m,
            tileScale=tile_scale,
            geometries=False,
        )
        payload = sampled.getInfo()
        for feature in payload.get("features", []):
            properties = feature.get("properties", {})
            key = properties.get("_alphaearth_cache_key")
            if key is not None:
                responses[str(key)] = {band: properties.get(band) for band in BANDS}
    return responses


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.request_retries < 1:
        raise ValueError("--request-retries must be positive")

    source = pd.read_csv(args.sensor_csv)
    required = [args.lat_col, args.lon_col, args.timestamp_col]
    missing = sorted(set(required) - set(source.columns))
    if missing:
        raise ValueError(f"Sensor CSV is missing columns: {missing}")
    if args.sample_col not in source:
        source[args.sample_col] = source.index
    if source[args.sample_col].duplicated().any():
        raise ValueError(f"{args.sample_col} must be unique")
    if args.limit is not None:
        source = source.head(args.limit).copy()

    source["alphaearth_measurement_year"] = measurement_years(
        source[args.timestamp_col], args.fallback_year
    )
    source["alphaearth_year"] = source["alphaearth_measurement_year"].clip(
        upper=args.max_available_year
    )
    source["latitude"] = pd.to_numeric(source[args.lat_col], errors="raise").round(
        args.round_decimals
    )
    source["longitude"] = pd.to_numeric(source[args.lon_col], errors="raise").round(
        args.round_decimals
    )
    source["cache_key"] = [
        cache_key(int(year), float(lat), float(lon), args.scale_m, args.round_decimals)
        for year, lat, lon in zip(
            source["alphaearth_year"], source["latitude"], source["longitude"], strict=True
        )
    ]
    locations = source[
        ["cache_key", "alphaearth_year", "latitude", "longitude"]
    ].drop_duplicates("cache_key")

    cache_path = Path(args.cache_json)
    cache: dict[str, dict[str, Any]] = (
        json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    )
    complete_keys = {
        key for key, row in cache.items() if all(row.get(band) is not None for band in BANDS)
    }
    pending = locations[~locations["cache_key"].isin(complete_keys)].reset_index(drop=True)
    batch_count = (len(pending) + args.batch_size - 1) // args.batch_size
    print(
        "Starting batched AlphaEarth extraction: "
        f"samples={len(source)}, unique_locations={len(locations)}, "
        f"cached_complete={len(complete_keys & set(locations.cache_key))}, "
        f"pending={len(pending)}, batches={batch_count}",
        flush=True,
    )
    if args.dry_run:
        print("Dry run requested; no Earth Engine requests or files were written.", flush=True)
        return 0

    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Install the geospatial extra: pip install earthengine-api") from exc
    try:
        ee.Initialize(project=args.project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run `earthengine authenticate` and provide "
            "--project if required."
        ) from exc

    failures: list[dict[str, Any]] = []
    fetched = 0
    for batch_number, batch in enumerate(chunks(pending, args.batch_size), start=1):
        error: Exception | None = None
        values: dict[str, dict[str, Any]] = {}
        for attempt in range(1, args.request_retries + 1):
            try:
                values = fetch_batch(
                    ee, batch, scale_m=args.scale_m, tile_scale=args.tile_scale
                )
                error = None
                break
            except Exception as exc:  # Earth Engine exposes several transport exception types.
                error = exc
                print(
                    f"AlphaEarth batch {batch_number}/{batch_count} attempt "
                    f"{attempt}/{args.request_retries} failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt < args.request_retries:
                    time.sleep(args.delay_seconds * attempt)

        if error is not None:
            failures.append({
                "batch": batch_number,
                "keys": batch["cache_key"].tolist(),
                "error_type": type(error).__name__,
                "error": str(error),
            })
        else:
            for record in batch.itertuples(index=False):
                bands = values.get(record.cache_key)
                if bands is None or any(bands.get(band) is None for band in BANDS):
                    failures.append({
                        "batch": batch_number,
                        "keys": [record.cache_key],
                        "error_type": "MissingPixel",
                        "error": "Earth Engine returned no complete 64-band pixel.",
                    })
                    continue
                cache[record.cache_key] = {
                    "alphaearth_measurement_year": int(record.alphaearth_year),
                    "alphaearth_year": int(record.alphaearth_year),
                    "alphaearth_year_lag": 0,
                    "alphaearth_dataset": DATASET_ID,
                    "alphaearth_status": "success",
                    "alphaearth_error": "",
                    **bands,
                }
                fetched += 1
            write_json_atomic(cache_path, cache)
        print(
            f"AlphaEarth progress {batch_number}/{batch_count} | "
            f"fetched={fetched} cache_entries={len(cache)} failures={len(failures)}",
            flush=True,
        )
        if batch_number < batch_count:
            time.sleep(args.delay_seconds)

    output = Path(args.output_csv)
    failure_path = output.with_suffix(".failures.json")
    unresolved = [key for key in locations["cache_key"] if key not in cache]
    if failures or unresolved:
        write_json_atomic(failure_path, {
            "failures": failures,
            "unresolved_unique_locations": unresolved,
            "successful_cache_entries": len(cache),
        })
        print(
            f"AlphaEarth extraction incomplete: failures={len(failures)}, "
            f"unresolved={len(unresolved)}. Rerun the same command to resume. "
            f"Details: {failure_path}",
            flush=True,
        )
        return 1

    rows = []
    for record in source.itertuples(index=False):
        cached = cache[record.cache_key]
        measurement_year = int(record.alphaearth_measurement_year)
        selected_year = int(record.alphaearth_year)
        rows.append({
            "sample_id": getattr(record, args.sample_col),
            "sample_index": getattr(record, "sample_index", getattr(record, args.sample_col)),
            "alphaearth_location_key": record.cache_key,
            **cached,
            "alphaearth_measurement_year": measurement_year,
            "alphaearth_year": selected_year,
            "alphaearth_year_lag": measurement_year - selected_year,
        })
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    failure_path.unlink(missing_ok=True)
    summary = {
        "output_csv": str(output),
        "cache_json": str(cache_path),
        "samples": len(result),
        "unique_locations": len(locations),
        "bands": len(BANDS),
        "scale_m": args.scale_m,
        "round_decimals": args.round_decimals,
        "batch_size": args.batch_size,
        "max_available_year": args.max_available_year,
        "point_embeddings_only": True,
        "status_counts": result["alphaearth_status"].value_counts().to_dict(),
    }
    write_json_atomic(output.with_suffix(".summary.json"), summary)
    print(f"Saved AlphaEarth features for {len(result)} samples: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
