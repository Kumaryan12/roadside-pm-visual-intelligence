"""Extract aligned frames from the ELICIUS video manifest.

The external SSD is treated as read-only. Images and the extraction manifest are
written beneath the requested local artifact directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_ROOT = Path("artifacts/runs/mumma_3day_ingestion_v1")
DEFAULT_OUTPUT_ROOT = Path("artifacts/runs/mumma_3day_frame_pilot_v1")


def build_extraction_jobs(
    sensor: pd.DataFrame,
    videos: pd.DataFrame,
    lenses: tuple[int, ...],
    frame_offset_seconds: float = 0.0,
) -> pd.DataFrame:
    """Join canonical samples to videos and calculate auditable seek offsets."""
    required_sensor = {"sample_id", "run_id", "sample_timestamp"}
    required_video = {
        "run_id",
        "lens_id",
        "video_path",
        "start_time_ist",
        "end_time_ist",
        "timezone",
    }
    missing_sensor = required_sensor.difference(sensor.columns)
    missing_video = required_video.difference(videos.columns)
    if missing_sensor:
        raise ValueError(f"Sensor table missing columns: {sorted(missing_sensor)}")
    if missing_video:
        raise ValueError(f"Video manifest missing columns: {sorted(missing_video)}")

    selected_videos = videos[videos["lens_id"].astype(int).isin(lenses)].copy()
    duplicated = selected_videos.duplicated(["run_id", "lens_id"], keep=False)
    if duplicated.any():
        keys = selected_videos.loc[duplicated, ["run_id", "lens_id"]]
        raise ValueError(f"Duplicate run/lens video rows: {keys.to_dict('records')}")

    requested = sensor.copy()
    if "sample_index" not in requested.columns:
        requested["sample_index"] = range(len(requested))
    requested["_join_key"] = 1
    lens_frame = pd.DataFrame({"lens_id": list(lenses), "_join_key": 1})
    requested = requested.merge(lens_frame, on="_join_key", how="inner").drop(
        columns="_join_key"
    )
    jobs = requested.merge(
        selected_videos,
        on=["run_id", "lens_id"],
        how="left",
        suffixes=("", "_video"),
        validate="many_to_one",
    )

    sample_time = pd.to_datetime(jobs["sample_timestamp"], errors="raise")
    start_time = pd.to_datetime(jobs["start_time_ist"], errors="coerce")
    end_time = pd.to_datetime(jobs["end_time_ist"], errors="coerce")
    jobs["video_offset_seconds"] = (
        sample_time - start_time
    ).dt.total_seconds() + frame_offset_seconds
    # Legacy aliases remain at the manifest boundary while new code uses the
    # stable sample_id and the explicitly named video_offset_seconds.
    jobs["sample_unix"] = [
        pd.Timestamp(timestamp).tz_localize(timezone).timestamp()
        for timestamp, timezone in zip(sample_time, jobs["timezone"], strict=True)
    ]
    jobs["video_offset_sec"] = jobs["video_offset_seconds"]
    jobs["declared_video_duration_seconds"] = (
        end_time - start_time
    ).dt.total_seconds()
    jobs["frame_offset_adjustment_seconds"] = frame_offset_seconds
    jobs["within_declared_run"] = (
        jobs["video_offset_seconds"].ge(0)
        & jobs["video_offset_seconds"].le(jobs["declared_video_duration_seconds"])
    )
    return jobs


def extract_frame(video_path: Path, offset_seconds: float, output_path: Path) -> tuple[bool, str]:
    """Extract one JPEG with ffmpeg, returning success and a compact error."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{offset_seconds:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip()[-1000:]
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False, "ffmpeg_completed_without_a_nonempty_output"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensor-csv", default=str(DEFAULT_INPUT_ROOT / "sensor_10s.csv")
    )
    parser.add_argument(
        "--video-manifest",
        default=str(DEFAULT_INPUT_ROOT / "video_manifest_lenses_1_2_6.csv"),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 2, 6])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frame-offset-seconds", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent ffmpeg extractions. Use a small value for external SSDs.",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    sensor_path = Path(args.sensor_csv)
    video_manifest_path = Path(args.video_manifest)
    for path in (sensor_path, video_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sensor = pd.read_csv(sensor_path)
    if "sample_index" not in sensor.columns:
        sensor["sample_index"] = range(len(sensor))
    videos = pd.read_csv(video_manifest_path)
    if args.run_id:
        sensor = sensor[sensor["run_id"].eq(args.run_id)].copy()
        if sensor.empty:
            raise ValueError(f"No sensor samples found for run {args.run_id!r}")
    sensor = sensor.sort_values(["sample_timestamp", "sample_id"])
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.stride > 1:
        sensor = sensor.iloc[:: args.stride].copy()
    if args.limit is not None:
        sensor = sensor.head(args.limit).copy()

    output_root = Path(args.output_root)
    frame_root = output_root / "frames" / "extracted"
    manifest_path = output_root / "manifests" / "extracted_frames.csv"
    jobs = build_extraction_jobs(
        sensor,
        videos,
        lenses=tuple(args.lenses),
        frame_offset_seconds=args.frame_offset_seconds,
    )

    total = len(jobs)
    def process_job(item: tuple[int, dict[str, object]]) -> tuple[int, dict[str, object]]:
        position, row = item
        sample_id = str(row["sample_id"])
        run_id = str(row["run_id"])
        lens_id = int(row["lens_id"])
        frame_key = f"{sample_id}_lens{lens_id}"
        output_path = frame_root / run_id / f"lens{lens_id}" / f"{frame_key}.jpg"
        video_value = row.get("video_path")
        video_path = Path(str(video_value)) if pd.notna(video_value) else None

        error = ""
        status = "failed"
        reused_existing = False
        if video_path is None:
            error = "no_video_manifest_match"
        elif not bool(row["within_declared_run"]):
            error = "timestamp_outside_declared_run"
        elif not video_path.is_file():
            error = f"source_video_not_found: {video_path}"
        elif args.skip_existing and output_path.is_file() and output_path.stat().st_size > 0:
            status = "success"
            reused_existing = True
        else:
            ok, error = extract_frame(
                video_path, float(row["video_offset_seconds"]), output_path
            )
            status = "success" if ok else "failed"

        result_row = dict(row)
        result_row.update(
            {
                "sensor_timestamp": row["sample_timestamp"],
                "matched_run_id": run_id,
                "source_video_path": str(video_path) if video_path else "",
                "processed_frame_key": frame_key,
                "processed_frame_path": str(output_path) if status == "success" else "",
                "preprocess_status": status,
                "preprocess_error": error,
                "reused_existing": reused_existing,
            }
        )
        return position, result_row

    indexed_jobs = list(enumerate(jobs.to_dict("records"), start=1))
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for position, result_row in executor.map(process_job, indexed_jobs):
            rows.append(result_row)
            print(
                f"[{position}/{total}] {result_row['sample_id']} "
                f"lens {result_row['lens_id']}: {result_row['preprocess_status']}",
                flush=True,
            )

    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    status_counts = manifest["preprocess_status"].value_counts().to_dict()
    summary = {
        "sensor_csv": str(sensor_path),
        "video_manifest": str(video_manifest_path),
        "output_manifest": str(manifest_path),
        "samples": int(len(sensor)),
        "requested_lenses": args.lenses,
        "expected_frames": int(total),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "frame_offset_seconds": args.frame_offset_seconds,
        "workers": args.workers,
        "external_source_modified": False,
    }
    (output_root / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if status_counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
