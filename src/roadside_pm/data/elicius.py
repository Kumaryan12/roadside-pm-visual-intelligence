"""Read-only discovery and 10-second formatting for ELICIUS MUMMA exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _camera_root(day: Path) -> Path | None:
    for name in ("CAMERA", "Camera", "camera"):
        candidate = day / name
        if candidate.is_dir(): return candidate
    return None


def _choose_video(run_dir: Path | None, lens: int) -> tuple[str | None, int, str]:
    if run_dir is None: return None, 0, "missing_run_directory"
    folder = run_dir / f"LENS{lens}"
    # macOS writes AppleDouble sidecars such as ``._video_lens1.mp4`` on some
    # external filesystems. They match ``*.mp4`` but are metadata, not videos.
    candidates = sorted(
        path for path in folder.glob("*.mp4")
        if not path.name.startswith(".") and path.is_file()
    ) if folder.is_dir() else []
    if not candidates: return None, 0, "missing"
    exact_name = f"video_lens{lens}.mp4"
    exact = [path for path in candidates if path.name.lower() == exact_name]
    preferred = exact or [path for path in candidates if "(" not in path.name]
    chosen = preferred[0] if preferred else max(candidates, key=lambda path: path.stat().st_size)
    status = "duplicate_candidates" if len(candidates) > 1 else "present"
    return str(chosen), len(candidates), status


def discover_elicius_runs(root: Path, *, required_lenses: tuple[int, ...] = (1, 4, 6)) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit() and len(path.name) == 8):
        metadata_path = day / "metadata.json"
        if not metadata_path.is_file(): continue
        metadata = json.loads(metadata_path.read_text()); camera_root = _camera_root(day)
        for trip_id, records in metadata.get("trips", {}).items():
            for record in records:
                run_id = str(record["run_id"]); run_dir = camera_root / run_id if camera_root else None
                start = pd.Timestamp(record["start_time_ist"]); stamp = start.strftime("%Y-%m-%d_%H-%M-%S")
                aqi = day / "AQI" / f"aqi_{stamp}.csv"
                row: dict[str, Any] = {"collection_id": run_id, "run_id": run_id, "date": start.date().isoformat(),
                    "trip_id": f"{day.name}_trip{trip_id}", "start_time_ist": record["start_time_ist"],
                    "end_time_ist": record["end_time_ist"], "sensor_csv": str(aqi), "sensor_exists": aqi.is_file(),
                    "day_root": str(day), "metadata_path": str(metadata_path)}
                for lens in required_lenses:
                    path, count, status = _choose_video(run_dir, lens); row[f"video_lens{lens}"] = path
                    row[f"video_lens{lens}_candidates"] = count; row[f"video_lens{lens}_status"] = status
                missing_lenses = [lens for lens in required_lenses if not row[f"video_lens{lens}"]]
                row["required_lenses_complete"] = not missing_lenses
                row["eligible"] = bool(row["sensor_exists"] and not missing_lenses)
                reasons = []
                if not row["sensor_exists"]: reasons.append("missing_aqi")
                if missing_lenses: reasons.append("missing_lenses_" + "_".join(map(str, missing_lenses)))
                row["exclusion_reason"] = ";".join(reasons)
                rows.append(row)
    return pd.DataFrame(rows)


def read_aqi_export(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, comment="#")
    if "timestamp_local" not in table or "sPM2" not in table:
        raise ValueError(f"AQI export missing timestamp_local or sPM2: {path}")
    table["timestamp_local"] = pd.to_datetime(table["timestamp_local"], errors="coerce")
    return table.dropna(subset=["timestamp_local"]).sort_values("timestamp_local").reset_index(drop=True)


def resample_aqi_run(path: Path, *, run_id: str, date: str, trip_id: str,
                     start_time: str, end_time: str, interval_seconds: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = read_aqi_export(path); start = pd.Timestamp(start_time); end = pd.Timestamp(end_time)
    clipped = raw[(raw.timestamp_local >= start) & (raw.timestamp_local <= end)].copy()
    if clipped.empty: raise ValueError(f"No AQI rows within metadata bounds for {run_id}")
    elapsed = (clipped.timestamp_local - start).dt.total_seconds()
    clipped["sample_offset_seconds"] = (np.floor(elapsed / interval_seconds) * interval_seconds).astype(int)
    numeric = [column for column in clipped.select_dtypes(include=[np.number, "bool"]).columns
               if column not in {"sample_offset_seconds", "t_unix", "t_rel_s"}]
    grouped = clipped.groupby("sample_offset_seconds", sort=True)
    output = grouped[numeric].mean().reset_index()
    output["raw_observation_count"] = grouped.size().to_numpy()
    output["sample_timestamp"] = start + pd.to_timedelta(output.sample_offset_seconds, unit="s")
    output["sample_id"] = [f"{run_id}_{offset:06d}" for offset in output.sample_offset_seconds]
    output["collection_id"] = run_id; output["run_id"] = run_id; output["date"] = date; output["trip_id"] = trip_id
    front = ["sample_id", "collection_id", "run_id", "date", "trip_id", "sample_timestamp", "sample_offset_seconds", "raw_observation_count"]
    output = output[front + [column for column in output if column not in front]]
    audit = {"run_id": run_id, "raw_rows": len(raw), "clipped_rows": len(clipped), "dropped_outside_bounds": len(raw)-len(clipped),
        "samples_10s": len(output), "first_sample": output.sample_timestamp.min().isoformat(),
        "last_sample": output.sample_timestamp.max().isoformat(), "pm25_missing": int(output.sPM2.isna().sum()),
        "gps_missing": int((output.lat.isna() | output.long.isna()).sum()) if {"lat", "long"}.issubset(output) else len(output)}
    return output, audit
