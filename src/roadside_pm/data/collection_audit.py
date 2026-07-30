"""Read-only validation for multi-day sensor/video collections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COLLECTION_COLUMNS = ("collection_id", "date", "trip_id", "sensor_csv", "video_path", "lens_id", "timezone")


def audit_sensor_table(table: pd.DataFrame, *, timestamp_column: str, target_column: str,
                       expected_interval_seconds: float = 10.0) -> dict[str, Any]:
    missing = [column for column in (timestamp_column, target_column) if column not in table]
    if missing: raise ValueError(f"Sensor table missing columns: {missing}")
    timestamp = pd.to_datetime(table[timestamp_column], errors="coerce")
    valid = timestamp.dropna().sort_values()
    intervals = valid.diff().dt.total_seconds().dropna()
    positive = intervals[intervals > 0]
    return {
        "rows": int(len(table)),
        "valid_timestamps": int(timestamp.notna().sum()),
        "invalid_timestamps": int(timestamp.isna().sum()),
        "duplicate_timestamps": int(timestamp.duplicated(keep=False).sum()),
        "timestamps_monotonic_in_source": bool(timestamp.dropna().is_monotonic_increasing),
        "start": valid.min().isoformat() if len(valid) else None,
        "end": valid.max().isoformat() if len(valid) else None,
        "median_interval_seconds": float(positive.median()) if len(positive) else None,
        "p95_interval_seconds": float(positive.quantile(0.95)) if len(positive) else None,
        "intervals_near_expected_fraction": float(np.mean(np.isclose(positive, expected_interval_seconds, atol=1.0))) if len(positive) else None,
        "gaps_over_2x_expected": int((positive > 2 * expected_interval_seconds).sum()),
        "target_missing": int(table[target_column].isna().sum()),
        "target_min": float(pd.to_numeric(table[target_column], errors="coerce").min()),
        "target_max": float(pd.to_numeric(table[target_column], errors="coerce").max()),
    }


def audit_collection_manifest(manifest: pd.DataFrame, *, project_root: Path, timestamp_column: str,
                              target_column: str, expected_interval_seconds: float = 10.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = [column for column in COLLECTION_COLUMNS if column not in manifest]
    if missing: raise ValueError(f"Collection manifest missing columns: {missing}")
    if manifest["collection_id"].duplicated().any(): raise ValueError("collection_id must be unique")
    rows = []
    for record in manifest.to_dict("records"):
        sensor = Path(str(record["sensor_csv"])).expanduser(); video = Path(str(record["video_path"])).expanduser()
        if not sensor.is_absolute(): sensor = project_root / sensor
        if not video.is_absolute(): video = project_root / video
        result: dict[str, Any] = {"collection_id": record["collection_id"], "date": record["date"],
            "trip_id": record["trip_id"], "sensor_csv": str(sensor), "video_path": str(video),
            "sensor_exists": sensor.is_file(), "video_exists": video.is_file()}
        if sensor.is_file():
            result.update(audit_sensor_table(pd.read_csv(sensor), timestamp_column=timestamp_column,
                target_column=target_column, expected_interval_seconds=expected_interval_seconds))
        rows.append(result)
    detail = pd.DataFrame(rows)
    summary = {"collections": int(len(detail)), "unique_dates": int(manifest["date"].astype(str).nunique()),
        "sensor_files_present": int(detail.sensor_exists.sum()), "video_files_present": int(detail.video_exists.sum()),
        "total_sensor_rows": int(detail.get("rows", pd.Series(dtype=float)).fillna(0).sum()),
        "expected_interval_seconds": expected_interval_seconds,
        "status": "pass" if detail.sensor_exists.all() and detail.video_exists.all() else "fail_missing_inputs"}
    return detail, summary
