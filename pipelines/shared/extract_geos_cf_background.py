"""Extract hourly NASA GEOS-CF v2 surface air-quality background.

The extractor reads only the required time/grid points from NASA's public
GrADS/OPeNDAP ASCII endpoint.  Mobile samples are matched to their nearest
0.25-degree GEOS-CF grid-cell and the hourly average containing the sample
timestamp.  This is a retrospective analysis match: the containing-hour
average is not necessarily available in real time at the sample timestamp.

All downloaded values are cached by variable, time index, latitude index and
longitude index, so interrupted runs can be resumed without repeating
successful requests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


GRID_DEGREES = 0.25

PRODUCTS = {
    "v1": {
        "dataset_url": (
            "https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/assim/"
            "aqc_tavg_1hr_g1440x721_v1"
        ),
        "start": pd.Timestamp("2018-01-01T00:30:00Z"),
        "end": pd.Timestamp("2026-01-02T11:30:00Z"),
        "dataset": "NASA GEOS-CF v1 assimilation aqc_tavg_1hr",
        "variables": {
            "pm25_rh35_gcc": "background_geos_cf_pm25_ug_m3",
            "co": "background_geos_cf_co_mol_mol",
            "no2": "background_geos_cf_no2_mol_mol",
            "o3": "background_geos_cf_o3_mol_mol",
            "so2": "background_geos_cf_so2_mol_mol",
        },
    },
    "v2": {
        "dataset_url": (
            "https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/v2/ana/"
            "aqc_tavg_1hr_glo_L1440x721_slv"
        ),
        "start": pd.Timestamp("2025-08-04T09:30:00Z"),
        "end": pd.Timestamp("2026-06-08T08:30:00Z"),
        "dataset": "NASA GEOS-CF v2 analysis aqc_tavg_1hr",
        "variables": {
            "pm25_rh35": "background_geos_cf_pm25_ug_m3",
            "pm10_rh35": "background_geos_cf_pm10_ug_m3",
            "co": "background_geos_cf_co_mol_mol",
            "no2": "background_geos_cf_no2_mol_mol",
            "o3": "background_geos_cf_o3_mol_mol",
            "so2": "background_geos_cf_so2_mol_mol",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--cache-json", type=Path, required=True)
    parser.add_argument("--timestamp-col", default="sample_timestamp")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--lat-col", default="lat")
    parser.add_argument("--lon-col", default="long")
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--product-profile", choices=sorted(PRODUCTS), default="v2")
    parser.add_argument(
        "--dataset-url",
        help="Optional endpoint override for the selected product profile",
    )
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def timestamp_utc(series: pd.Series, timezone_name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError(
            f"{int(parsed.isna().sum())} sample timestamps could not be parsed"
        )
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            timezone_name, ambiguous="raise", nonexistent="raise"
        )
    return parsed.dt.tz_convert("UTC")


def grid_indices(latitude: pd.Series, longitude: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lat = pd.to_numeric(latitude, errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(longitude, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Latitude/longitude contain missing or non-finite values")
    if ((lat < -90.0) | (lat > 90.0)).any():
        raise ValueError("Latitude lies outside [-90, 90]")
    lat_index = np.rint((lat + 90.0) / GRID_DEGREES).astype(int)
    normalized_lon = ((lon + 180.0) % 360.0) - 180.0
    lon_index = (
        np.rint((normalized_lon + 180.0) / GRID_DEGREES).astype(int) % 1440
    )
    return np.clip(lat_index, 0, 720), lon_index


def valid_times(sample_utc: pd.Series) -> pd.Series:
    # GEOS-CF hourly averages are timestamped at the middle of each hour.
    return sample_utc.dt.floor("h") + pd.Timedelta(minutes=30)


def time_indices(
    valid_time: pd.Series, dataset_start: pd.Timestamp, dataset_end: pd.Timestamp
) -> np.ndarray:
    if (valid_time < dataset_start).any() or (valid_time > dataset_end).any():
        earliest = valid_time.min().isoformat()
        latest = valid_time.max().isoformat()
        raise ValueError(
            "Requested GEOS-CF times fall outside the selected product "
            f"coverage {dataset_start.isoformat()} to {dataset_end.isoformat()}: "
            f"{earliest} to {latest}"
        )
    hours = (valid_time - dataset_start) / pd.Timedelta(hours=1)
    rounded = np.rint(hours.to_numpy(dtype=float)).astype(int)
    if not np.allclose(hours.to_numpy(dtype=float), rounded):
        raise ValueError("GEOS-CF valid times did not map to the hourly index")
    return rounded


def load_cache(path: Path, dataset_url: str) -> dict[str, object]:
    if not path.is_file():
        return {"dataset_url": dataset_url, "entries": {}}
    payload = json.loads(path.read_text())
    if payload.get("dataset_url") != dataset_url:
        raise ValueError(
            "Existing GEOS-CF cache belongs to a different dataset URL"
        )
    if not isinstance(payload.get("entries"), dict):
        raise ValueError("GEOS-CF cache is malformed")
    return payload


def save_cache(path: Path, cache: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def cache_key(variable: str, time_index: int, lat_index: int, lon_index: int) -> str:
    return f"{variable}|{time_index}|{lat_index}|{lon_index}"


def contiguous_ranges(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ordered = sorted(set(values))
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return ranges


def parse_ascii_values(text: str, expected: int) -> list[float]:
    values: list[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("[") or "]," not in stripped:
            continue
        left, right = stripped.split(",", 1)
        # Data rows have four indices. Coordinate rows are intentionally ignored.
        if left.count("][") != 2:
            continue
        try:
            values.append(float(right.strip()))
        except ValueError:
            continue
    if len(values) != expected:
        preview = "\n".join(text.splitlines()[:12])
        raise RuntimeError(
            f"GEOS-CF response contained {len(values)} values; expected "
            f"{expected}. Response begins:\n{preview}"
        )
    return values


def fetch_range(
    session: requests.Session,
    *,
    dataset_url: str,
    variable: str,
    start_index: int,
    end_index: int,
    lat_index: int,
    lon_index: int,
    retries: int,
    timeout_seconds: float,
    delay_seconds: float,
) -> list[float]:
    query = (
        f"{variable}[{start_index}:1:{end_index}]"
        f"[0:1:0][{lat_index}:1:{lat_index}][{lon_index}:1:{lon_index}]"
    )
    url = f"{dataset_url}.ascii?{quote(query, safe='[]:,')}"
    final_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            return parse_ascii_values(
                response.text, expected=end_index - start_index + 1
            )
        except Exception as error:  # network and malformed server responses
            final_error = error
            if attempt == retries:
                break
            time.sleep(delay_seconds * attempt)
    raise RuntimeError(
        f"GEOS-CF request failed after {retries} attempts: {url}"
    ) from final_error


def main() -> int:
    args = parse_args()
    product = PRODUCTS[args.product_profile]
    dataset_url = str(args.dataset_url or product["dataset_url"])
    dataset_start = pd.Timestamp(product["start"])
    dataset_end = pd.Timestamp(product["end"])
    variables = dict(product["variables"])
    frame = pd.read_csv(args.sensor_csv)
    required = {
        args.sample_col,
        args.timestamp_col,
        args.lat_col,
        args.lon_col,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Sensor table is missing columns: {missing}")
    if args.limit is not None:
        frame = frame.head(args.limit).copy()
    if frame[args.sample_col].duplicated().any():
        raise ValueError("Sensor sample IDs must be unique")

    sample_utc = timestamp_utc(frame[args.timestamp_col], args.timezone)
    matched_time = valid_times(sample_utc)
    time_index = time_indices(matched_time, dataset_start, dataset_end)
    lat_index, lon_index = grid_indices(frame[args.lat_col], frame[args.lon_col])

    cache = load_cache(args.cache_json, dataset_url)
    entries = cache["entries"]
    assert isinstance(entries, dict)
    grid_time = pd.DataFrame(
        {
            "time_index": time_index,
            "lat_index": lat_index,
            "lon_index": lon_index,
        }
    ).drop_duplicates()

    session = requests.Session()
    requests_made = 0
    for variable in variables:
        for (lat_i, lon_i), group in grid_time.groupby(
            ["lat_index", "lon_index"], sort=True
        ):
            missing_times = [
                int(value)
                for value in group["time_index"]
                if cache_key(
                    variable, int(value), int(lat_i), int(lon_i)
                )
                not in entries
            ]
            for start_index, end_index in contiguous_ranges(missing_times):
                values = fetch_range(
                    session,
                    dataset_url=dataset_url,
                    variable=variable,
                    start_index=start_index,
                    end_index=end_index,
                    lat_index=int(lat_i),
                    lon_index=int(lon_i),
                    retries=args.request_retries,
                    timeout_seconds=args.request_timeout_seconds,
                    delay_seconds=args.delay_seconds,
                )
                requests_made += 1
                for offset, value in enumerate(values):
                    entries[
                        cache_key(
                            variable,
                            start_index + offset,
                            int(lat_i),
                            int(lon_i),
                        )
                    ] = value
                save_cache(args.cache_json, cache)
                print(
                    "GEOS-CF "
                    f"{variable} grid=({int(lat_i)},{int(lon_i)}) "
                    f"time={start_index}:{end_index} cached={len(entries)}",
                    flush=True,
                )
                if args.delay_seconds:
                    time.sleep(args.delay_seconds)

    output = pd.DataFrame(
        {
            "sample_id": frame[args.sample_col].astype(str),
            "sample_timestamp": frame[args.timestamp_col],
            "background_geos_cf_sample_time_utc": sample_utc.map(
                lambda value: value.isoformat()
            ),
            "background_geos_cf_valid_time_utc": matched_time.map(
                lambda value: value.isoformat()
            ),
            "background_geos_cf_time_offset_minutes": (
                (matched_time - sample_utc) / pd.Timedelta(minutes=1)
            ).to_numpy(dtype=float),
            "background_geos_cf_grid_latitude": -90.0
            + GRID_DEGREES * lat_index,
            "background_geos_cf_grid_longitude": -180.0
            + GRID_DEGREES * lon_index,
        }
    )
    for variable, output_column in variables.items():
        output[output_column] = [
            float(
                entries[
                    cache_key(variable, int(time_i), int(lat_i), int(lon_i))
                ]
            )
            for time_i, lat_i, lon_i in zip(
                time_index, lat_index, lon_index, strict=True
            )
        ]
    finite = np.isfinite(
        output[list(variables.values())].to_numpy(dtype=float)
    ).all(axis=1)
    output["background_geos_cf_status"] = np.where(
        finite, "success", "failed"
    )
    # Compatibility with the existing external-background model runner.
    output["background_status"] = output["background_geos_cf_status"]
    output["background_error"] = ""

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    summary = {
        "sensor_csv": str(args.sensor_csv),
        "output_csv": str(args.output_csv),
        "dataset_url": dataset_url,
        "dataset": product["dataset"],
        "product_profile": args.product_profile,
        "spatial_resolution_degrees": GRID_DEGREES,
        "temporal_resolution": "hourly average",
        "time_matching": "retrospective containing-hour average",
        "rows": len(output),
        "unique_grid_cells": int(
            output[
                [
                    "background_geos_cf_grid_latitude",
                    "background_geos_cf_grid_longitude",
                ]
            ].drop_duplicates().shape[0]
        ),
        "unique_valid_times": int(
            output["background_geos_cf_valid_time_utc"].nunique()
        ),
        "network_requests_this_run": requests_made,
        "status_counts": output["background_geos_cf_status"]
        .value_counts()
        .to_dict(),
        "pm25_summary_ug_m3": output[
            "background_geos_cf_pm25_ug_m3"
        ].describe().to_dict(),
        "warning": (
            "GEOS-CF is a regional research product, not a roadside monitor. "
            "The containing-hour retrospective match may use an hourly average "
            "that was not yet available at the sample timestamp."
        ),
    }
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
