"""Build pre-specified local-reference PM2.5 aggregation variants for TRAQID.

The source table is the long-form OpenAQ hourly download.  For every TRAQID
timestamp, the nearest observation in time is selected independently for each
sensor within a fixed tolerance.  Duplicate sensors at the same monitoring
location are collapsed before three location-level summaries are computed:

* unweighted median across available monitoring locations;
* nearest monitoring location; and
* inverse-distance-squared mean across available monitoring locations.

No roadside PM2.5 target is read or used by this utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hourly-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--max-time-gap-minutes", type=float, default=90.0)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--idw-distance-floor-km", type=float, default=1.0)
    return parser.parse_args()


def nearest_sensor_observation(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> pd.Series | None:
    times = frame["reference_timestamp_local"]
    index = int(times.searchsorted(timestamp))
    possible = [position for position in (index - 1, index) if 0 <= position < len(frame)]
    if not possible:
        return None
    chosen = min(possible, key=lambda position: abs(times.iloc[position] - timestamp))
    row = frame.iloc[chosen].copy()
    row["gap"] = abs(row["reference_timestamp_local"] - timestamp)
    return row


def aggregate_locations(
    matched_sensors: pd.DataFrame,
    power: float,
    distance_floor_km: float,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    locations = (
        matched_sensors.assign(
            location_key=matched_sensors["location_id"].fillna(
                matched_sensors["location_name"]
            ).astype(str)
        )
        .groupby("location_key", as_index=False)
        .agg(
            reference_pm25_ug_m3=("reference_pm25_ug_m3", "median"),
            distance_km=("distance_km", "min"),
            gap=("gap", "min"),
            location_id=("location_id", "first"),
            location_name=("location_name", "first"),
            sensor_count=("sensor_id", "nunique"),
        )
    )
    nearest = locations.sort_values(["distance_km", "gap", "location_key"]).iloc[0]
    distances = np.maximum(
        locations["distance_km"].to_numpy(dtype=float),
        distance_floor_km,
    )
    weights = 1.0 / np.power(distances, power)
    values = locations["reference_pm25_ug_m3"].to_numpy(dtype=float)
    summaries: dict[str, float | str] = {
        "reference_median_pm25_ug_m3": float(np.median(values)),
        "reference_nearest_pm25_ug_m3": float(nearest["reference_pm25_ug_m3"]),
        "reference_idw_pm25_ug_m3": float(np.average(values, weights=weights)),
        "reference_nearest_location_id": str(nearest["location_id"]),
        "reference_nearest_location_name": str(nearest["location_name"]),
        "reference_nearest_distance_km": float(nearest["distance_km"]),
        "reference_max_distance_km": float(locations["distance_km"].max()),
        "reference_max_abs_time_gap_minutes": float(
            locations["gap"].max().total_seconds() / 60.0
        ),
    }
    return locations, summaries


def main() -> int:
    args = parse_args()
    if args.max_time_gap_minutes <= 0:
        raise ValueError("--max-time-gap-minutes must be positive")
    if args.idw_power <= 0 or args.idw_distance_floor_km <= 0:
        raise ValueError("IDW power and distance floor must be positive")

    manifest = pd.read_csv(args.manifest, low_memory=False)
    required_manifest = {"row_id", "created_at_parsed", "date"}
    if missing := sorted(required_manifest - set(manifest.columns)):
        raise ValueError(f"Manifest is missing columns: {missing}")
    samples = manifest[["row_id", "created_at_parsed", "date"]].copy()
    samples.columns = ["sample_id", "sample_timestamp", "date"]
    samples["sample_timestamp"] = pd.to_datetime(
        samples["sample_timestamp"], errors="raise", format="mixed"
    )

    hourly = pd.read_csv(args.hourly_csv, low_memory=False)
    required_hourly = {
        "reference_timestamp_utc",
        "reference_pm25_ug_m3",
        "location_id",
        "location_name",
        "sensor_id",
        "distance_km",
    }
    if missing := sorted(required_hourly - set(hourly.columns)):
        raise ValueError(f"Hourly table is missing columns: {missing}")
    hourly["reference_timestamp_utc"] = pd.to_datetime(
        hourly["reference_timestamp_utc"],
        errors="coerce",
        utc=True,
        format="mixed",
    )
    hourly["reference_timestamp_local"] = (
        hourly["reference_timestamp_utc"]
        .dt.tz_convert(args.timezone)
        .dt.tz_localize(None)
    )
    hourly["reference_pm25_ug_m3"] = pd.to_numeric(
        hourly["reference_pm25_ug_m3"], errors="coerce"
    )
    hourly = hourly.dropna(
        subset=[
            "reference_timestamp_local",
            "reference_pm25_ug_m3",
            "sensor_id",
            "distance_km",
        ]
    )
    by_sensor = {
        sensor_id: frame.sort_values("reference_timestamp_local").reset_index(drop=True)
        for sensor_id, frame in hourly.groupby("sensor_id")
    }
    tolerance = pd.Timedelta(minutes=args.max_time_gap_minutes)
    output_rows: list[dict[str, object]] = []

    for sample_number, sample in samples.iterrows():
        if sample_number and sample_number % 2000 == 0:
            print(f"Reference variants {sample_number}/{len(samples)}", flush=True)
        candidates: list[pd.Series] = []
        for frame in by_sensor.values():
            row = nearest_sensor_observation(frame, sample["sample_timestamp"])
            if row is not None and row["gap"] <= tolerance:
                candidates.append(row)
        base: dict[str, object] = {
            "sample_id": sample["sample_id"],
            "sample_timestamp": sample["sample_timestamp"],
            "date": sample["date"],
        }
        if not candidates:
            output_rows.append(
                {
                    **base,
                    "reference_status": "missing",
                    "reference_pm25_ug_m3": np.nan,
                    "reference_median_pm25_ug_m3": np.nan,
                    "reference_nearest_pm25_ug_m3": np.nan,
                    "reference_idw_pm25_ug_m3": np.nan,
                    "reference_location_count": 0,
                    "reference_sensor_count": 0,
                }
            )
            continue
        matched = pd.DataFrame(candidates)
        locations, summaries = aggregate_locations(
            matched,
            args.idw_power,
            args.idw_distance_floor_km,
        )
        output_rows.append(
            {
                **base,
                "reference_status": "success",
                # Backward-compatible alias is the location-level median.
                "reference_pm25_ug_m3": summaries[
                    "reference_median_pm25_ug_m3"
                ],
                **summaries,
                "reference_location_count": int(len(locations)),
                "reference_sensor_count": int(matched["sensor_id"].nunique()),
            }
        )

    result = pd.DataFrame(output_rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    value_columns = [
        "reference_median_pm25_ug_m3",
        "reference_nearest_pm25_ug_m3",
        "reference_idw_pm25_ug_m3",
    ]
    coverage_rows: list[dict[str, object]] = []
    for date, group in result.groupby("date"):
        for column in value_columns:
            coverage_rows.append(
                {
                    "date": date,
                    "reference_variant": column,
                    "rows": len(group),
                    "matched": int(group[column].notna().sum()),
                    "coverage_fraction": float(group[column].notna().mean()),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_path.with_name(f"{output_path.stem}_coverage.csv"), index=False)
    audit = {
        "manifest": args.manifest,
        "hourly_csv": args.hourly_csv,
        "rows": len(result),
        "sensors_with_downloaded_observations": len(by_sensor),
        "locations_with_downloaded_observations": int(hourly["location_id"].nunique()),
        "max_time_gap_minutes": args.max_time_gap_minutes,
        "idw_power": args.idw_power,
        "idw_distance_floor_km": args.idw_distance_floor_km,
        "target_used": False,
        "variants": value_columns,
        "variants_numerically_identical": bool(
            np.allclose(
                result.loc[result[value_columns].notna().all(axis=1), value_columns],
                result.loc[
                    result[value_columns].notna().all(axis=1),
                    value_columns[0],
                ].to_numpy()[:, None],
            )
        ),
    }
    output_path.with_name(f"{output_path.stem}_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(f"Reference variants {len(samples)}/{len(samples)}", flush=True)
    print(coverage.to_string(index=False))
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
