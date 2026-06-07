"""
Extract video frames from external hard-drive camera runs at sensor timestamps.

Input:
- Sensor CSV with timestamp column.
- External hard-drive run folders containing LENS*/video_lens*.mp4 and capture_stats/run_t0 metadata.

Output:
- Extracted frame images.
- Frame manifest CSV.

Important:
- Original videos are treated as read-only.
- Frames are written into this repo's outputs/extracted_frames_v2/.
"""

from pathlib import Path
import argparse
import json
import subprocess
import pandas as pd


DEFAULT_SENSOR_CSV = Path("data/sensor/MC1S_window_115430_124150.csv")
DEFAULT_VIDEO_ROOT = Path("/Volumes/New Volume/23_02_2026_navneet")
DEFAULT_OUTPUT_ROOT = Path("outputs/extracted_frames_v2")
DEFAULT_OUTPUT_MANIFEST = Path("outputs/features/processed_frame_manifest_v2.csv")

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
    distribute them evenly across that minute to avoid extracting duplicate frames.
    """
    ts = pd.to_datetime(
        sensor_df[timestamp_col],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    if ts.isna().any():
        bad = sensor_df.loc[ts.isna(), timestamp_col].head(10).tolist()
        raise ValueError(f"Could not parse some timestamps. Examples: {bad}")

    # Localize to IST
    ts = ts.dt.tz_localize("Asia/Kolkata")

    # Distribute repeated same-minute rows inside the minute
    adjusted = ts.copy()

    grouped = sensor_df.groupby(timestamp_col).cumcount()
    group_sizes = sensor_df.groupby(timestamp_col)[timestamp_col].transform("size")

    # offset seconds = rank * 60 / group_size
    offsets = grouped * (60.0 / group_sizes)

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

    # If multiple runs overlap, choose nearest start before sample
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sensor-csv", default=str(DEFAULT_SENSOR_CSV))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST))
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--lenses", nargs="+", type=int, default=DEFAULT_LENSES)
    parser.add_argument("--limit", type=int, default=None)
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

    sensor_df = pd.read_csv(sensor_csv)

    if args.limit is not None:
        sensor_df = sensor_df.head(args.limit).copy()

    if args.timestamp_col not in sensor_df.columns:
        raise ValueError(f"Timestamp column not found: {args.timestamp_col}")

    print("\nBuilding run index...")
    run_index = build_run_index(video_root)

    print("Run index shape:", run_index.shape)
    print("Video exists counts:")
    print(run_index["video_exists"].value_counts(dropna=False))

    print("\nParsing sensor timestamps...")
    sensor_df["sample_index"] = range(len(sensor_df))
    sensor_df["sample_unix"] = parse_sensor_timestamps(sensor_df, args.timestamp_col)

    rows = []

    total_expected = len(sensor_df) * len(args.lenses)
    done = 0

    for _, sample in sensor_df.iterrows():
        sample_index = int(sample["sample_index"])
        sample_timestamp = str(sample[args.timestamp_col])
        sample_unix = float(sample["sample_unix"])

        for lens_id in args.lenses:
            done += 1

            match = find_matching_run(run_index, sample_unix, lens_id)

            if match is None:
                rows.append({
                    "sample_index": sample_index,
                    "sensor_timestamp": sample_timestamp,
                    "sample_unix": sample_unix,
                    "lens_id": lens_id,
                    "matched_run_id": "",
                    "video_offset_sec": None,
                    "source_video_path": "",
                    "processed_frame_key": f"sample_{sample_index:04d}_lens{lens_id}",
                    "processed_frame_path": "",
                    "preprocess_status": "failed",
                    "preprocess_error": "no_matching_video_run",
                })
                print(f"[{done}/{total_expected}] sample {sample_index} lens {lens_id}: no matching run")
                continue

            video_path = Path(match["video_path"])
            video_offset_sec = sample_unix - float(match["t0_unix"])
            run_id = match["run_id"]

            processed_frame_key = f"sample_{sample_index:04d}_{run_id}_offset_{int(video_offset_sec*1000):09d}_lens{lens_id}"

            output_path = (
                output_root
                / run_id
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

            rows.append({
                "sample_index": sample_index,
                "sensor_timestamp": sample_timestamp,
                "sample_unix": sample_unix,
                "lens_id": lens_id,
                "matched_run_id": run_id,
                "video_offset_sec": video_offset_sec,
                "source_video_path": str(video_path),
                "processed_frame_key": processed_frame_key,
                "processed_frame_path": str(output_path),
                "preprocess_status": "success" if ok else "failed",
                "preprocess_error": error,
            })

            status = "success" if ok else "failed"
            print(f"[{done}/{total_expected}] sample {sample_index} lens {lens_id}: {status}")

    manifest = pd.DataFrame(rows)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)

    print("\nDone.")
    print("Saved manifest:", output_manifest)
    print("Shape:", manifest.shape)

    print("\nStatus counts:")
    print(manifest["preprocess_status"].value_counts(dropna=False))

    print("\nRows by lens:")
    print(manifest.groupby("lens_id")["preprocess_status"].value_counts())


if __name__ == "__main__":
    main()
