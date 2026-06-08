"""
Extract 1-second sampled video frames around each sensor timestamp.

Purpose:
- For each sensor row timestamp T, extract frames from a time window:
    [T - window_before_sec, T + window_after_sec]
- Sample every sampling_interval_sec seconds.
- Extract for selected lens IDs.
- Save a manifest that maps each extracted frame back to the sensor row.

Important:
- Original videos are treated as read-only.
- This script is for the systematic 1-second pipeline.
"""

from pathlib import Path
import argparse
import json
import subprocess
import pandas as pd
import numpy as np


DEFAULT_SENSOR_CSV = Path("data/sensor/MC1S_window_115430_124150.csv")
DEFAULT_VIDEO_ROOT = Path("/Volumes/New Volume/23_02_2026_navneet")

DEFAULT_OUTPUT_ROOT = Path("outputs/pipeline_1s/frames_raw")
DEFAULT_OUTPUT_MANIFEST = Path("outputs/pipeline_1s/manifests/frame_manifest_1s_v1.csv")

DEFAULT_LENSES = [1, 4, 6]


def read_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def build_run_index(video_root: Path) -> pd.DataFrame:
    rows = []

    for run_dir in sorted(video_root.glob("run_*")):
        run_t0 = read_json(run_dir / "run_t0.json")
        capture_stats = read_json(run_dir / "capture_stats.json")

        run_id = run_t0.get("run_id", run_dir.name)
        t0_unix = run_t0.get("t0_unix", None)
        received_utc = run_t0.get("received_utc", "")

        per_lens = capture_stats.get("per_lens", {})

        for lens_id in range(1, 7):
            video_path = run_dir / f"LENS{lens_id}" / f"video_lens{lens_id}.mp4"
            lens_stats = per_lens.get(str(lens_id), {})

            duration_sec = lens_stats.get("duration_sec", None)
            avg_fps = lens_stats.get("avg_fps", None)

            run_end_unix = None
            if t0_unix is not None and duration_sec is not None:
                run_end_unix = float(t0_unix) + float(duration_sec)

            rows.append({
                "run_id": run_id,
                "run_dir": str(run_dir),
                "lens_id": lens_id,
                "video_path": str(video_path),
                "video_exists": video_path.exists(),
                "t0_unix": t0_unix,
                "received_utc": received_utc,
                "duration_sec": duration_sec,
                "avg_fps": avg_fps,
                "run_end_unix": run_end_unix,
            })

    return pd.DataFrame(rows)


def parse_sensor_timestamps(sensor_df: pd.DataFrame, timestamp_col: str) -> pd.Series:
    """
    Parse sensor timestamps as Asia/Kolkata local time and convert to Unix seconds.

    The CSV timestamp is minute-level. If multiple rows share the same timestamp,
    distribute them evenly across that minute to avoid identical base timestamps.
    """
    ts = pd.to_datetime(
        sensor_df[timestamp_col],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    if ts.isna().any():
        bad = sensor_df.loc[ts.isna(), timestamp_col].head(10).tolist()
        raise ValueError(f"Could not parse some timestamps. Examples: {bad}")

    ts = ts.dt.tz_localize("Asia/Kolkata")

    adjusted = ts.copy()

    grouped_rank = sensor_df.groupby(timestamp_col).cumcount()
    group_sizes = sensor_df.groupby(timestamp_col)[timestamp_col].transform("size")

    offsets = grouped_rank * (60.0 / group_sizes)

    adjusted = adjusted + pd.to_timedelta(offsets, unit="s")

    unix_sec = adjusted.astype("int64") / 1e9

    return unix_sec


def find_matching_run(run_index: pd.DataFrame, sample_unix: float, lens_id: int):
    candidates = run_index[
        (run_index["lens_id"] == lens_id)
        & (run_index["video_exists"] == True)
        & (run_index["t0_unix"].astype(float) <= sample_unix)
        & (run_index["run_end_unix"].astype(float) >= sample_unix)
    ].copy()

    if candidates.empty:
        return None

    candidates["time_from_start"] = sample_unix - candidates["t0_unix"].astype(float)
    candidates = candidates.sort_values("time_from_start")

    return candidates.iloc[-1].to_dict()


def extract_frame_ffmpeg(video_path: Path, offset_sec: float, output_path: Path) -> tuple:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{offset_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        return False, result.stderr[-1000:]

    if not output_path.exists():
        return False, "ffmpeg_completed_but_output_missing"

    return True, ""


def build_relative_offsets(window_before_sec, window_after_sec, sampling_interval_sec):
    """
    Example:
    window_before=60, window_after=0, interval=1
    returns [-60, -59, ..., 0]
    """
    start = -float(window_before_sec)
    end = float(window_after_sec)
    step = float(sampling_interval_sec)

    if step <= 0:
        raise ValueError("sampling_interval_sec must be > 0")

    values = []
    current = start

    while current <= end + 1e-9:
        values.append(round(current, 6))
        current += step

    return values


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sensor-csv", default=str(DEFAULT_SENSOR_CSV))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST))

    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--lenses", nargs="+", type=int, default=DEFAULT_LENSES)

    parser.add_argument("--window-before-sec", type=float, default=60.0)
    parser.add_argument("--window-after-sec", type=float, default=0.0)
    parser.add_argument("--sampling-interval-sec", type=float, default=1.0)

    parser.add_argument("--limit-sensor-rows", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    sensor_csv = Path(args.sensor_csv)
    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    output_manifest = Path(args.output_manifest)

    if not sensor_csv.exists():
        raise FileNotFoundError(f"Sensor CSV not found: {sensor_csv}")

    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    sensor_df = pd.read_csv(sensor_csv)

    if args.limit_sensor_rows is not None:
        sensor_df = sensor_df.head(args.limit_sensor_rows).copy()

    if args.timestamp_col not in sensor_df.columns:
        raise ValueError(f"Timestamp column not found: {args.timestamp_col}")

    print("\nBuilding run index...")
    run_index = build_run_index(video_root)

    print("Run index shape:", run_index.shape)
    print("Video exists counts:")
    print(run_index["video_exists"].value_counts(dropna=False))

    print("\nParsing sensor timestamps...")
    sensor_df["sensor_row_id"] = range(len(sensor_df))
    sensor_df["sample_index"] = range(len(sensor_df))
    sensor_df["sample_unix"] = parse_sensor_timestamps(sensor_df, args.timestamp_col)

    relative_offsets = build_relative_offsets(
        window_before_sec=args.window_before_sec,
        window_after_sec=args.window_after_sec,
        sampling_interval_sec=args.sampling_interval_sec,
    )

    print("\n1-second extraction config")
    print("=" * 60)
    print("Sensor CSV:", sensor_csv)
    print("Video root:", video_root)
    print("Output root:", output_root)
    print("Output manifest:", output_manifest)
    print("Lenses:", args.lenses)
    print("Window before sec:", args.window_before_sec)
    print("Window after sec:", args.window_after_sec)
    print("Sampling interval sec:", args.sampling_interval_sec)
    print("Relative offsets per sensor row:", len(relative_offsets))
    print("Sensor rows:", len(sensor_df))

    total_expected = len(sensor_df) * len(args.lenses) * len(relative_offsets)
    print("Total expected frame attempts:", total_expected)

    rows = []
    done = 0

    for _, sample in sensor_df.iterrows():
        sensor_row_id = int(sample["sensor_row_id"])
        sample_index = int(sample["sample_index"])
        sensor_timestamp = str(sample[args.timestamp_col])
        sensor_unix = float(sample["sample_unix"])

        for relative_time_sec in relative_offsets:
            frame_sample_unix = sensor_unix + float(relative_time_sec)

            for lens_id in args.lenses:
                done += 1

                match = find_matching_run(run_index, frame_sample_unix, lens_id)

                base_row = {
                    "sensor_row_id": sensor_row_id,
                    "sample_index": sample_index,
                    "sensor_timestamp": sensor_timestamp,
                    "sensor_unix": sensor_unix,
                    "frame_sample_unix": frame_sample_unix,
                    "relative_time_sec": relative_time_sec,
                    "lens_id": lens_id,
                }

                if match is None:
                    processed_frame_key = (
                        f"sensor_{sensor_row_id:05d}"
                        f"_rel_{int(relative_time_sec * 1000):+07d}ms"
                        f"_lens{lens_id}"
                    )

                    rows.append({
                        **base_row,
                        "matched_run_id": "",
                        "video_offset_sec": np.nan,
                        "source_video_path": "",
                        "processed_frame_key": processed_frame_key,
                        "processed_frame_path": "",
                        "preprocess_status": "failed",
                        "preprocess_error": "no_matching_video_run",
                        "frame_extraction_status": "failed",
                        "frame_extraction_error": "no_matching_video_run",
                    })

                    print(
                        f"[{done}/{total_expected}] "
                        f"sensor {sensor_row_id} rel {relative_time_sec:+.1f}s "
                        f"lens {lens_id}: no matching run"
                    )
                    continue

                video_path = Path(match["video_path"])
                video_offset_sec = frame_sample_unix - float(match["t0_unix"])
                run_id = match["run_id"]

                rel_ms = int(round(relative_time_sec * 1000))
                offset_ms = int(round(video_offset_sec * 1000))

                processed_frame_key = (
                    f"sensor_{sensor_row_id:05d}"
                    f"_sample_{sample_index:05d}"
                    f"_{run_id}"
                    f"_rel_{rel_ms:+07d}ms"
                    f"_offset_{offset_ms:09d}"
                    f"_lens{lens_id}"
                )

                output_path = (
                    output_root
                    / run_id
                    / f"sensor_{sensor_row_id:05d}"
                    / f"lens{lens_id}"
                    / f"{processed_frame_key}.jpg"
                )

                if args.skip_existing and output_path.exists():
                    ok = True
                    error = ""
                else:
                    ok, error = extract_frame_ffmpeg(
                        video_path=video_path,
                        offset_sec=video_offset_sec,
                        output_path=output_path,
                    )

                status = "success" if ok else "failed"

                rows.append({
                    **base_row,
                    "matched_run_id": run_id,
                    "video_offset_sec": video_offset_sec,
                    "source_video_path": str(video_path),
                    "processed_frame_key": processed_frame_key,
                    "processed_frame_path": str(output_path) if ok else "",
                    "preprocess_status": status,
                    "preprocess_error": error,
                    "frame_extraction_status": status,
                    "frame_extraction_error": error,
                })

                print(
                    f"[{done}/{total_expected}] "
                    f"sensor {sensor_row_id} rel {relative_time_sec:+.1f}s "
                    f"lens {lens_id}: {status}"
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_manifest, index=False)

    print("\nDone.")
    print("Saved manifest:", output_manifest)
    print("Shape:", out_df.shape)

    print("\nFrame extraction status:")
    print(out_df["frame_extraction_status"].value_counts(dropna=False))

    print("\nLens counts:")
    print(out_df["lens_id"].value_counts(dropna=False).sort_index())

    print("\nSensor rows represented:")
    print(out_df["sensor_row_id"].nunique())


if __name__ == "__main__":
    main()