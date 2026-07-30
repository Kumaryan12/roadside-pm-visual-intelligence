"""Extract cached OSM context features for unique sensor coordinates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from roadside_pm.features.geospatial.osm import build_overpass_query, fetch_overpass, summarize_elements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--lat-col", default="value.lat")
    parser.add_argument("--lon-col", default="value.long")
    parser.add_argument("--sample-col", default="sample_index")
    parser.add_argument("--radius-m", type=int, default=250)
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--endpoint", default="https://overpass-api.de/api/interpreter")
    parser.add_argument("--cache-json", default=None)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a live progress line after every N input samples.",
    )
    args = parser.parse_args()
    source = pd.read_csv(args.sensor_csv)
    for column in (args.lat_col, args.lon_col):
        if column not in source:
            raise ValueError(f"Missing coordinate column {column!r}")
    if args.sample_col not in source:
        source[args.sample_col] = source.index
    if args.limit is not None:
        source = source.head(args.limit).copy()
    output = Path(args.output_csv)
    cache_path = Path(args.cache_json) if args.cache_json else output.with_suffix(".cache.json")
    persistent_cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    cache = {}
    rows = []
    fetched_locations = 0
    reused_locations = 0
    error_locations = 0
    print(
        f"Starting OSM extraction: samples={len(source)}, "
        f"rounded_unique_locations={source[[args.lat_col, args.lon_col]].round(args.round_decimals).drop_duplicates().shape[0]}, "
        f"persistent_cache_entries={len(persistent_cache)}",
        flush=True,
    )
    for position, (_, record) in enumerate(source.iterrows(), start=1):
        lat, lon = float(record[args.lat_col]), float(record[args.lon_col])
        key = (round(lat, args.round_decimals), round(lon, args.round_decimals))
        cache_key = f"{args.radius_m}:{key[0]:.{args.round_decimals}f}:{key[1]:.{args.round_decimals}f}"
        if key not in cache:
            if cache_key in persistent_cache:
                cache[key] = persistent_cache[cache_key]
                reused_locations += 1
            else:
                try:
                    elements = fetch_overpass(args.endpoint, build_overpass_query(*key, args.radius_m))
                    cache[key] = {**summarize_elements(elements, args.radius_m), "osm_status": "success", "osm_error": ""}
                    persistent_cache[cache_key] = cache[key]
                    fetched_locations += 1
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(persistent_cache, indent=2) + "\n")
                except Exception as exc:
                    cache[key] = {**summarize_elements([], args.radius_m), "osm_status": "error", "osm_error": str(exc)}
                    error_locations += 1
                    print(
                        f"OSM query error at {cache_key}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                if args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)
        rows.append({"sample_id": record[args.sample_col], "sample_index": record.get("sample_index", record[args.sample_col]), "osm_location_key": f"{key[0]:.{args.round_decimals}f}_{key[1]:.{args.round_decimals}f}", "osm_latitude_rounded": key[0], "osm_longitude_rounded": key[1], **cache[key]})
        if args.progress_every > 0 and (position % args.progress_every == 0 or position == len(source)):
            print(
                f"OSM progress {position}/{len(source)} | unique_seen={len(cache)} "
                f"fetched={fetched_locations} persistent_cache={reused_locations} "
                f"errors={error_locations}",
                flush=True,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved OSM features for {len(rows)} samples ({len(cache)} unique locations): {output}")


if __name__ == "__main__": main()
