"""Extract OSM context with buffered route tiles and local radius filtering.

Unlike the legacy point extractor, this runner sends one Overpass request per
occupied route tile, caches every successful tile separately, and calculates
the exact per-location 250 m summaries locally. An interrupted run resumes from
the tile cache. A final CSV is written only when every required tile is present.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from roadside_pm.features.geospatial.osm import (
    build_overpass_bbox_query,
    fetch_overpass,
    summarize_nearby_elements,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--lat-col", default="lat")
    parser.add_argument("--lon-col", default="long")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--radius-m", type=int, default=250)
    parser.add_argument("--round-decimals", type=int, default=3)
    parser.add_argument(
        "--tile-degrees",
        type=float,
        default=0.02,
        help="Occupied route-grid tile width; 0.02 degrees gives about 2 km tiles.",
    )
    parser.add_argument(
        "--endpoint",
        default="https://overpass.private.coffee/api/interpreter",
    )
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print route/tile counts without making network requests.",
    )
    return parser.parse_args()


def tile_index(latitude: float, longitude: float, tile_degrees: float) -> tuple[int, int]:
    return math.floor(latitude / tile_degrees), math.floor(longitude / tile_degrees)


def tile_bounds(
    index: tuple[int, int],
    tile_degrees: float,
    radius_m: int,
) -> tuple[float, float, float, float]:
    lat_index, lon_index = index
    south = lat_index * tile_degrees
    north = south + tile_degrees
    west = lon_index * tile_degrees
    east = west + tile_degrees
    center_latitude = (south + north) / 2
    latitude_padding = radius_m / 111_320.0
    longitude_padding = radius_m / (111_320.0 * max(math.cos(math.radians(center_latitude)), 0.1))
    return (
        south - latitude_padding,
        west - longitude_padding,
        north + latitude_padding,
        east + longitude_padding,
    )


def tile_key(index: tuple[int, int], tile_degrees: float) -> str:
    precision = max(0, len(str(tile_degrees).split(".")[-1].rstrip("0")))
    return f"tile_{index[0]}_{index[1]}_d{precision}"


def deduplicate_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    anonymous = 0
    for element in elements:
        if "id" in element:
            key = (str(element.get("type", "unknown")), str(element["id"]))
        else:
            key = ("anonymous", str(anonymous))
            anonymous += 1
        unique[key] = element
    return list(unique.values())


def read_cached_tile(path: Path, expected: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    for key, value in expected.items():
        if payload.get("metadata", {}).get(key) != value:
            raise ValueError(f"Cached tile metadata mismatch for {path}: {key}")
    return payload["elements"]


def write_cached_tile(path: Path, metadata: dict[str, Any], elements: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"metadata": metadata, "elements": elements}) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.radius_m <= 0:
        raise ValueError("radius-m must be positive")
    if args.tile_degrees <= 0:
        raise ValueError("tile-degrees must be positive")
    source = pd.read_csv(args.sensor_csv)
    required = [args.lat_col, args.lon_col, args.sample_col]
    missing = sorted(set(required) - set(source.columns))
    if missing:
        raise ValueError(f"Sensor table is missing columns: {missing}")
    if source[args.sample_col].duplicated().any():
        raise ValueError(f"{args.sample_col} must be unique")

    source[args.lat_col] = pd.to_numeric(source[args.lat_col], errors="raise")
    source[args.lon_col] = pd.to_numeric(source[args.lon_col], errors="raise")
    source["osm_latitude_rounded_work"] = source[args.lat_col].round(args.round_decimals)
    source["osm_longitude_rounded_work"] = source[args.lon_col].round(args.round_decimals)
    source["osm_tile_work"] = [
        tile_index(latitude, longitude, args.tile_degrees)
        for latitude, longitude in zip(
            source["osm_latitude_rounded_work"], source["osm_longitude_rounded_work"]
        )
    ]
    occupied_tiles = sorted(source["osm_tile_work"].unique())
    unique_locations = source[
        ["osm_latitude_rounded_work", "osm_longitude_rounded_work", "osm_tile_work"]
    ].drop_duplicates()
    print(
        f"Starting corridor OSM extraction: samples={len(source)}, "
        f"unique_locations={len(unique_locations)}, occupied_tiles={len(occupied_tiles)}, "
        f"tile_degrees={args.tile_degrees}",
        flush=True,
    )
    if args.dry_run:
        print("Dry run requested; no OSM requests or output files were created.", flush=True)
        return 0

    cache_directory = Path(args.cache_dir)
    elements_by_tile: dict[tuple[int, int], list[dict[str, Any]]] = {}
    fetched = reused = 0
    failures: list[dict[str, Any]] = []
    for position, index in enumerate(occupied_tiles, start=1):
        bounds = tile_bounds(index, args.tile_degrees, args.radius_m)
        metadata = {
            "tile_index": list(index),
            "tile_degrees": args.tile_degrees,
            "radius_m": args.radius_m,
            "bounds": list(bounds),
        }
        path = cache_directory / f"{tile_key(index, args.tile_degrees)}.json"
        elements = read_cached_tile(path, metadata)
        status = "cache"
        if elements is None:
            query = build_overpass_bbox_query(*bounds, timeout_s=args.request_timeout_seconds)
            try:
                elements = fetch_overpass(
                    args.endpoint,
                    query,
                    timeout_s=args.request_timeout_seconds,
                    retries=args.request_retries,
                )
                elements = deduplicate_elements(elements)
                write_cached_tile(path, metadata, elements)
                fetched += 1
                status = "fetched"
            except Exception as exc:
                failures.append({
                    "tile": list(index),
                    "bounds": list(bounds),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                status = f"error:{type(exc).__name__}"
            if args.delay_seconds > 0:
                time.sleep(args.delay_seconds)
        else:
            reused += 1
        if elements is not None:
            elements_by_tile[index] = elements
        print(
            f"OSM tile {position}/{len(occupied_tiles)} {index} {status} "
            f"elements={len(elements) if elements is not None else 0} "
            f"cached={reused} fetched={fetched} failures={len(failures)}",
            flush=True,
        )

    failure_path = Path(args.output_csv).with_suffix(".tile_failures.json")
    if failures:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(
            f"{len(failures)} of {len(occupied_tiles)} tiles failed. Successful tiles are cached; "
            "rerun the same command to retry only failed tiles. No incomplete output CSV was written. "
            f"Details: {failure_path}"
        )
    failure_path.unlink(missing_ok=True)

    summaries: dict[tuple[float, float], dict[str, Any]] = {}
    for position, row in enumerate(unique_locations.itertuples(index=False), start=1):
        latitude = float(row[0])
        longitude = float(row[1])
        index = row[2]
        summaries[(latitude, longitude)] = summarize_nearby_elements(
            elements_by_tile[index], (latitude, longitude), args.radius_m
        )
        if position % 100 == 0 or position == len(unique_locations):
            print(f"Local radius summaries {position}/{len(unique_locations)}", flush=True)

    rows = []
    for record in source.itertuples(index=False):
        values = record._asdict()
        latitude = float(values["osm_latitude_rounded_work"])
        longitude = float(values["osm_longitude_rounded_work"])
        sample = values[args.sample_col]
        rows.append({
            "sample_id": sample,
            "sample_index": values.get("sample_index", sample),
            "osm_location_key": f"{latitude:.{args.round_decimals}f}_{longitude:.{args.round_decimals}f}",
            "osm_latitude_rounded": latitude,
            "osm_longitude_rounded": longitude,
            **summaries[(latitude, longitude)],
            "osm_status": "success",
            "osm_error": "",
        })
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    manifest = {
        "sensor_csv": args.sensor_csv,
        "output_csv": str(output),
        "cache_dir": str(cache_directory),
        "samples": len(rows),
        "unique_locations": len(unique_locations),
        "occupied_tiles": len(occupied_tiles),
        "radius_m": args.radius_m,
        "round_decimals": args.round_decimals,
        "tile_degrees": args.tile_degrees,
        "endpoint": args.endpoint,
        "network_requests_this_run": fetched,
        "cached_tiles_reused": reused,
        "local_radius_filtering": True,
    }
    output.with_suffix(".run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
