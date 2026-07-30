"""Extract hourly regional aerosol and meteorological background features.

Sensor timestamps are interpreted in the supplied local timezone and matched
causally to the latest available analysis/forecast valid time.  A single route
centroid is sampled because CAMS, MERRA-2, and ERA5-Land are regional products
whose pixels are much coarser than the mobile route.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd


CAMS = "ECMWF/CAMS/NRT"
MERRA = "NASA/GSFC/MERRA/aer/2"
ERA5 = "ECMWF/ERA5_LAND/HOURLY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--cache-json", required=True)
    parser.add_argument("--timestamp-col", default="sample_timestamp")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--lat-col", default="lat")
    parser.add_argument("--lon-col", default="long")
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--project", default=None)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def relative_humidity_percent(temperature_k: float, dewpoint_k: float) -> float:
    temperature_c = temperature_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    ratio = math.exp(
        (17.625 * dewpoint_c) / (243.04 + dewpoint_c)
        - (17.625 * temperature_c) / (243.04 + temperature_c)
    )
    return max(0.0, min(100.0, 100.0 * ratio))


def fetch_hour(ee, utc_hour: pd.Timestamp, latitude: float, longitude: float) -> dict[str, Any]:
    start = utc_hour.isoformat().replace("+00:00", "Z")
    end = (utc_hour + pd.Timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    cams_hour = utc_hour.floor("3h")
    cams_start = cams_hour.isoformat().replace("+00:00", "Z")
    cams_end = (cams_hour + pd.Timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    point = ee.Geometry.Point([longitude, latitude])

    # Multiple CAMS forecasts can share a valid time.  The smallest forecast
    # horizon is the freshest causal forecast available for that valid time.
    cams = ee.Image(
        ee.ImageCollection(CAMS)
        .filterDate(cams_start, cams_end)
        .sort("model_forecast_hour")
        .first()
    )
    merra = ee.Image(ee.ImageCollection(MERRA).filterDate(start, end).first())
    era = ee.Image(ee.ImageCollection(ERA5).filterDate(start, end).first())

    cams_values = cams.select([
        "particulate_matter_d_less_than_25_um_surface",
        "particulate_matter_d_less_than_10_um_surface",
        "particulate_matter_d_less_than_1_um_surface",
        "total_aerosol_optical_depth_at_550nm_surface",
        "dust_aerosol_optical_depth_at_550nm_surface",
        "black_carbon_aerosol_optical_depth_at_550nm_surface",
    ]).reduceRegion(ee.Reducer.first(), point, 40_000, maxPixels=10_000)
    merra_values = merra.select([
        "DUSMASS25", "SSSMASS25", "BCSMASS", "OCSMASS", "SO4SMASS",
    ]).reduceRegion(ee.Reducer.first(), point, 60_000, maxPixels=10_000)
    era_values = era.select([
        "temperature_2m", "dewpoint_temperature_2m",
        "u_component_of_wind_10m", "v_component_of_wind_10m",
        "surface_pressure", "total_precipitation_hourly",
        "surface_solar_radiation_downwards_hourly",
    ]).reduceRegion(ee.Reducer.first(), point, 12_000, maxPixels=10_000)
    payload = ee.Dictionary({
        "cams": cams_values,
        "cams_forecast_hour": cams.get("model_forecast_hour"),
        "cams_initialization": cams.get("model_initialization_datetime"),
        "merra": merra_values,
        "era5": era_values,
    }).getInfo()

    c = payload["cams"]
    m = payload["merra"]
    e = payload["era5"]
    required = [*c.values(), *m.values(), *e.values()]
    if any(value is None for value in required):
        raise ValueError("One or more atmospheric fields are missing at the route centroid")

    dust = float(m["DUSMASS25"])
    sea_salt = float(m["SSSMASS25"])
    black_carbon = float(m["BCSMASS"])
    organic_carbon = float(m["OCSMASS"])
    sulphate = float(m["SO4SMASS"])
    merra_pm25 = dust + sea_salt + black_carbon + 1.4 * organic_carbon + 1.375 * sulphate
    u = float(e["u_component_of_wind_10m"])
    v = float(e["v_component_of_wind_10m"])
    temperature = float(e["temperature_2m"])
    dewpoint = float(e["dewpoint_temperature_2m"])
    return {
        "background_utc_hour": utc_hour.isoformat(),
        "background_cams_valid_time_utc": cams_hour.isoformat(),
        "background_cams_forecast_hour": int(payload["cams_forecast_hour"]),
        "background_cams_initialization_utc": payload["cams_initialization"],
        "background_cams_pm25_ug_m3": float(c["particulate_matter_d_less_than_25_um_surface"]) * 1e9,
        "background_cams_pm10_ug_m3": float(c["particulate_matter_d_less_than_10_um_surface"]) * 1e9,
        "background_cams_pm1_ug_m3": float(c["particulate_matter_d_less_than_1_um_surface"]) * 1e9,
        "background_cams_aod550": float(c["total_aerosol_optical_depth_at_550nm_surface"]),
        "background_cams_dust_aod550": float(c["dust_aerosol_optical_depth_at_550nm_surface"]),
        "background_cams_black_carbon_aod550": float(c["black_carbon_aerosol_optical_depth_at_550nm_surface"]),
        "background_merra2_pm25_ug_m3": merra_pm25 * 1e9,
        "background_merra2_dust25_ug_m3": dust * 1e9,
        "background_merra2_sea_salt25_ug_m3": sea_salt * 1e9,
        "background_merra2_black_carbon_ug_m3": black_carbon * 1e9,
        "background_merra2_organic_carbon_ug_m3": organic_carbon * 1e9,
        "background_merra2_sulphate_ug_m3": sulphate * 1e9,
        "background_era5_temperature_2m_c": temperature - 273.15,
        "background_era5_dewpoint_2m_c": dewpoint - 273.15,
        "background_era5_relative_humidity_pct": relative_humidity_percent(temperature, dewpoint),
        "background_era5_wind_u_10m_m_s": u,
        "background_era5_wind_v_10m_m_s": v,
        "background_era5_wind_speed_10m_m_s": math.hypot(u, v),
        "background_era5_surface_pressure_hpa": float(e["surface_pressure"]) / 100.0,
        "background_era5_precipitation_hourly_mm": float(e["total_precipitation_hourly"]) * 1000.0,
        "background_era5_solar_radiation_hourly_w_m2": float(e["surface_solar_radiation_downwards_hourly"]) / 3600.0,
        "background_status": "success",
        "background_error": "",
    }


def main() -> int:
    args = parse_args()
    source = pd.read_csv(args.sensor_csv)
    required = {args.sample_col, args.timestamp_col, args.lat_col, args.lon_col}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Sensor CSV is missing columns: {missing}")
    if source[args.sample_col].duplicated().any():
        raise ValueError(f"{args.sample_col} must be unique")
    if args.limit is not None:
        source = source.head(args.limit).copy()
    local = pd.to_datetime(source[args.timestamp_col], errors="raise").dt.tz_localize(
        args.timezone, ambiguous="raise", nonexistent="raise"
    )
    source["_utc_timestamp"] = local.dt.tz_convert("UTC")
    source["_utc_hour"] = source["_utc_timestamp"].dt.floor("h")
    latitude = float(pd.to_numeric(source[args.lat_col], errors="raise").median())
    longitude = float(pd.to_numeric(source[args.lon_col], errors="raise").median())
    hours = sorted(source["_utc_hour"].unique())

    cache_path = Path(args.cache_json)
    cache: dict[str, dict[str, Any]] = (
        json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    )
    print(
        f"Starting atmospheric extraction: samples={len(source)}, unique_utc_hours={len(hours)}, "
        f"route_centroid={latitude:.5f},{longitude:.5f}, cache_entries={len(cache)}",
        flush=True,
    )
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Install earthengine-api") from exc
    ee.Initialize(project=args.project)

    failures: list[dict[str, str]] = []
    for position, value in enumerate(hours, start=1):
        hour = pd.Timestamp(value)
        key = f"{hour.isoformat()}:{latitude:.5f}:{longitude:.5f}:v1"
        if key in cache and cache[key].get("background_status") == "success":
            status = "cached"
        else:
            error: Exception | None = None
            for attempt in range(1, args.request_retries + 1):
                try:
                    cache[key] = fetch_hour(ee, hour, latitude, longitude)
                    write_json_atomic(cache_path, cache)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    print(
                        f"Background hour {position}/{len(hours)} attempt {attempt}/"
                        f"{args.request_retries} failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if attempt < args.request_retries:
                        time.sleep(args.delay_seconds * attempt)
            status = "fetched" if error is None else "failed"
            if error is not None:
                failures.append({"key": key, "error": f"{type(error).__name__}: {error}"})
        print(
            f"Atmospheric progress {position}/{len(hours)} | {hour.isoformat()} {status} "
            f"cache_entries={len(cache)} failures={len(failures)}",
            flush=True,
        )
        time.sleep(args.delay_seconds)

    output = Path(args.output_csv)
    if failures:
        write_json_atomic(output.with_suffix(".failures.json"), failures)
        print("Extraction incomplete; rerun the same command to retry failed hours.", flush=True)
        return 1

    rows = []
    for record in source.to_dict("records"):
        hour = pd.Timestamp(record["_utc_hour"])
        key = f"{hour.isoformat()}:{latitude:.5f}:{longitude:.5f}:v1"
        values = cache[key]
        rows.append({
            "sample_id": record[args.sample_col],
            "sample_timestamp": record[args.timestamp_col],
            "background_route_latitude": latitude,
            "background_route_longitude": longitude,
            "background_match_age_minutes": (
                pd.Timestamp(record["_utc_timestamp"]) - hour
            ).total_seconds() / 60.0,
            **values,
        })
    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    output.with_suffix(".failures.json").unlink(missing_ok=True)
    write_json_atomic(output.with_suffix(".summary.json"), {
        "samples": len(result),
        "unique_utc_hours": len(hours),
        "timezone": args.timezone,
        "route_centroid": {"latitude": latitude, "longitude": longitude},
        "datasets": {"cams": CAMS, "merra2": MERRA, "era5_land": ERA5},
        "causal_time_matching": "local timestamp converted to UTC and floored to prior hour; CAMS floored to prior 3-hour valid time",
        "status_counts": result["background_status"].value_counts().to_dict(),
    })
    print(f"Saved atmospheric background for {len(result)} samples: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
