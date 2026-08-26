"""Fetch and align nearby OpenAQ PM2.5 observations to TRAQID rows.

This utility is intentionally independent of the prediction target: station
selection uses only distance, parameter and data availability.  It can merge a
previously downloaded hourly table so that a wider-radius retry does not
discard already verified measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


API = "https://api.openaq.org/v3"
OPENAQ_MAX_POINT_RADIUS_M = 25_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--radius-m", type=int, default=50_000)
    parser.add_argument("--max-time-gap-minutes", type=float, default=90.0)
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--api-key-env", default="OPENAQ_API_KEY")
    parser.add_argument("--existing-hourly-csv")
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Optional YYYY-MM-DD dates to fetch; alignment is still written for all rows.",
    )
    parser.add_argument("--request-pause-seconds", type=float, default=0.20)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def api_get(path: str, params: dict[str, object], api_key: str, retries: int) -> dict:
    url = f"{API}{path}?{urlencode(params)}"
    for attempt in range(retries):
        request = Request(url, headers={"X-API-Key": api_key})
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == retries:
                try:
                    response_body = error.read().decode("utf-8", errors="replace")
                except Exception:
                    response_body = ""
                detail = f"; response={response_body}" if response_body else ""
                raise RuntimeError(
                    f"OpenAQ request failed: {url}: {error}{detail}"
                ) from error
        except (TimeoutError, URLError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"OpenAQ request failed: {url}: {error}") from error
        delay = min(30.0, 2.0**attempt)
        print(f"OpenAQ retry {attempt + 1}/{retries} in {delay:g}s: {url}", flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    value = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def location_search(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    """Build a valid OpenAQ geospatial query for the requested radial extent.

    OpenAQ caps point-and-radius searches at 25 km.  For a wider requested
    extent, query the enclosing WGS84 bounding box and then apply the exact
    Haversine-radius filter locally in ``discover_sensors``.
    """
    common: dict[str, object] = {
        "parameters_id": 2,
        "monitor": "true",
        "mobile": "false",
        "limit": 1000,
        "page": 1,
    }
    if args.radius_m <= 0:
        raise ValueError("--radius-m must be positive")
    if args.radius_m <= OPENAQ_MAX_POINT_RADIUS_M:
        return (
            {
                "coordinates": f"{args.latitude:.4f},{args.longitude:.4f}",
                "radius": args.radius_m,
                **common,
            },
            "point_radius",
        )

    radius_km = args.radius_m / 1000.0
    latitude_delta = radius_km / 111.32
    cosine = math.cos(math.radians(args.latitude))
    if abs(cosine) < 1e-6:
        raise ValueError("Bounding-box search is undefined this close to a pole")
    longitude_delta = radius_km / (111.32 * cosine)
    bbox = (
        args.longitude - longitude_delta,
        args.latitude - latitude_delta,
        args.longitude + longitude_delta,
        args.latitude + latitude_delta,
    )
    if not (
        -180 <= bbox[0] <= 180
        and -90 <= bbox[1] <= 90
        and -180 <= bbox[2] <= 180
        and -90 <= bbox[3] <= 90
    ):
        raise ValueError(f"Requested search produces an invalid WGS84 bounding box: {bbox}")
    return (
        {
            "bbox": ",".join(f"{coordinate:.4f}" for coordinate in bbox),
            **common,
        },
        "bounding_box_then_haversine_filter",
    )


def discover_sensors(args: argparse.Namespace, api_key: str) -> pd.DataFrame:
    query, search_mode = location_search(args)
    print(
        f"OpenAQ station discovery mode={search_mode} requested_radius_m={args.radius_m}",
        flush=True,
    )
    payload = api_get(
        "/locations",
        query,
        api_key,
        args.retries,
    )
    rows: list[dict[str, object]] = []
    for location in payload.get("results", []):
        coordinates = location.get("coordinates") or {}
        lat = coordinates.get("latitude")
        lon = coordinates.get("longitude")
        if lat is None or lon is None:
            continue
        distance = haversine_km(args.latitude, args.longitude, float(lat), float(lon))
        # A bounding box encloses (but is larger than) the requested circle.
        if distance * 1000 > args.radius_m:
            continue
        for sensor in location.get("sensors") or []:
            parameter = sensor.get("parameter") or {}
            name = str(parameter.get("name", "")).lower().replace(".", "")
            if parameter.get("id") != 2 and name not in {"pm25", "pm2_5"}:
                continue
            rows.append(
                {
                    "location_id": location.get("id"),
                    "location_name": location.get("name"),
                    "locality": location.get("locality"),
                    "provider": (location.get("provider") or {}).get("name"),
                    "owner": (location.get("owner") or {}).get("name"),
                    "latitude": lat,
                    "longitude": lon,
                    "distance_km": distance,
                    "sensor_id": sensor.get("id"),
                    "sensor_name": sensor.get("name"),
                    "parameter": parameter.get("name"),
                    "units": parameter.get("units"),
                }
            )
    result = pd.DataFrame(rows).drop_duplicates("sensor_id")
    if result.empty:
        raise RuntimeError("OpenAQ returned no nearby fixed PM2.5 sensors")
    return result.sort_values(["distance_km", "sensor_id"]).reset_index(drop=True)


def result_timestamp(item: dict) -> object:
    period = item.get("period") or {}
    for container in (period.get("datetimeFrom"), item.get("datetimeFrom"), item.get("datetime")):
        if isinstance(container, dict):
            if container.get("utc"):
                return container["utc"]
        elif container:
            return container
    return None


def fetch_hours(
    sensors: pd.DataFrame,
    dates: list[str],
    args: argparse.Namespace,
    api_key: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    margin = pd.Timedelta(minutes=args.max_time_gap_minutes + 60)
    for sensor_number, sensor in sensors.iterrows():
        sensor_id = int(sensor["sensor_id"])
        for date_number, date in enumerate(dates, 1):
            print(
                f"OpenAQ sensor {sensor_number + 1}/{len(sensors)} "
                f"day {date_number}/{len(dates)}: {date}",
                flush=True,
            )
            local_start = pd.Timestamp(date).tz_localize(args.timezone) - margin
            local_end = pd.Timestamp(date).tz_localize(args.timezone) + pd.Timedelta(days=1) + margin
            payload = api_get(
                f"/sensors/{sensor_id}/hours",
                {
                    "datetime_from": local_start.tz_convert("UTC").isoformat(),
                    "datetime_to": local_end.tz_convert("UTC").isoformat(),
                    "limit": 1000,
                    "page": 1,
                },
                api_key,
                args.retries,
            )
            for item in payload.get("results", []):
                value = pd.to_numeric(item.get("value"), errors="coerce")
                timestamp = result_timestamp(item)
                if timestamp is None or not np.isfinite(value) or value < 0 or value > 2000:
                    continue
                rows.append(
                    {
                        "reference_timestamp_utc": timestamp,
                        "reference_pm25_ug_m3": float(value),
                        "location_id": sensor["location_id"],
                        "location_name": sensor["location_name"],
                        "sensor_id": sensor_id,
                        "distance_km": float(sensor["distance_km"]),
                        "parameter": sensor["parameter"],
                        "units": sensor["units"],
                    }
                )
            time.sleep(args.request_pause_seconds)
    columns = [
        "reference_timestamp_utc", "reference_pm25_ug_m3", "location_id",
        "location_name", "sensor_id", "distance_km", "parameter", "units",
    ]
    return pd.DataFrame(rows, columns=columns)


def align(manifest: pd.DataFrame, hourly: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    samples = manifest[["row_id", "created_at_parsed", "date"]].copy()
    samples.columns = ["sample_id", "sample_timestamp", "date"]
    samples["sample_timestamp"] = pd.to_datetime(samples["sample_timestamp"], errors="raise")
    hourly = hourly.copy()
    hourly["reference_timestamp_utc"] = pd.to_datetime(
        hourly["reference_timestamp_utc"], utc=True, errors="coerce"
    )
    hourly["reference_timestamp_local"] = (
        hourly["reference_timestamp_utc"].dt.tz_convert(args.timezone).dt.tz_localize(None)
    )
    hourly = hourly.dropna(subset=["reference_timestamp_local", "reference_pm25_ug_m3"])
    output_rows: list[dict[str, object]] = []
    tolerance = pd.Timedelta(minutes=args.max_time_gap_minutes)
    by_sensor = {key: frame.sort_values("reference_timestamp_local") for key, frame in hourly.groupby("sensor_id")}
    for _, sample in samples.iterrows():
        candidates: list[pd.Series] = []
        for frame in by_sensor.values():
            times = frame["reference_timestamp_local"]
            index = times.searchsorted(sample["sample_timestamp"])
            possible = [i for i in (index - 1, index) if 0 <= i < len(frame)]
            if not possible:
                continue
            chosen = min(possible, key=lambda i: abs(times.iloc[i] - sample["sample_timestamp"]))
            row = frame.iloc[chosen].copy()
            row["gap"] = abs(row["reference_timestamp_local"] - sample["sample_timestamp"])
            if row["gap"] <= tolerance:
                candidates.append(row)
        if candidates:
            matched = pd.DataFrame(candidates)
            nearest = matched.sort_values(["distance_km", "gap"]).iloc[0]
            value = float(matched["reference_pm25_ug_m3"].median())
            output_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "sample_timestamp": sample["sample_timestamp"],
                    "reference_pm25_ug_m3": value,
                    "reference_status": "success",
                    "reference_station_count": int(matched["location_id"].nunique()),
                    "reference_sensor_count": int(matched["sensor_id"].nunique()),
                    "reference_nearest_location_id": nearest["location_id"],
                    "reference_nearest_location_name": nearest["location_name"],
                    "reference_nearest_distance_km": float(nearest["distance_km"]),
                    "reference_max_distance_km": float(matched["distance_km"].max()),
                    "reference_max_abs_time_gap_minutes": float(matched["gap"].max().total_seconds() / 60),
                    "reference_sensor_ids": ";".join(map(str, sorted(matched["sensor_id"].astype(int).unique()))),
                }
            )
        else:
            output_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "sample_timestamp": sample["sample_timestamp"],
                    "reference_pm25_ug_m3": np.nan,
                    "reference_status": "missing",
                    "reference_station_count": 0,
                }
            )
    return pd.DataFrame(output_rows)


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before running this command")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    required = {"row_id", "created_at_parsed", "date"}
    if missing := sorted(required - set(manifest.columns)):
        raise ValueError(f"Manifest is missing columns: {missing}")
    all_dates = sorted(manifest["date"].astype(str).unique())
    fetch_dates = args.dates or all_dates
    invalid = sorted(set(fetch_dates) - set(all_dates))
    if invalid:
        raise ValueError(f"Requested dates are absent from manifest: {invalid}")
    sensors = discover_sensors(args, api_key)
    sensors.to_csv(output / "reference_stations.csv", index=False)
    hourly = fetch_hours(sensors, fetch_dates, args, api_key)
    if args.existing_hourly_csv:
        existing = pd.read_csv(args.existing_hourly_csv)
        hourly = pd.concat([existing, hourly], ignore_index=True)
    hourly = hourly.drop_duplicates(
        ["sensor_id", "reference_timestamp_utc"], keep="last"
    ).sort_values(["reference_timestamp_utc", "distance_km"])
    hourly.to_csv(output / "reference_hourly_long.csv", index=False)
    aligned = align(manifest, hourly, args)
    aligned.to_csv(output / "local_reference_pm25.csv", index=False)
    coverage = (
        aligned.assign(date=pd.to_datetime(aligned["sample_timestamp"]).dt.date.astype(str))
        .groupby("date", as_index=False)
        .agg(count=("sample_id", "size"), matched=("reference_pm25_ug_m3", "count"))
    )
    coverage["coverage_fraction"] = coverage["matched"] / coverage["count"]
    coverage.to_csv(output / "coverage_by_date.csv", index=False)
    audit = {
        "manifest": args.manifest,
        "radius_m": args.radius_m,
        "station_search_mode": location_search(args)[1],
        "max_time_gap_minutes": args.max_time_gap_minutes,
        "fetched_dates": fetch_dates,
        "sensors": int(len(sensors)),
        "hourly_rows": int(len(hourly)),
        "matched_rows": int(aligned["reference_pm25_ug_m3"].notna().sum()),
        "total_rows": int(len(aligned)),
        "fully_covered_dates": coverage.loc[coverage["coverage_fraction"].eq(1), "date"].tolist(),
        "warning": "Reference PM2.5 is an inference-time input; results are reference-assisted.",
    }
    (output / "alignment_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(coverage.to_string(index=False))
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
