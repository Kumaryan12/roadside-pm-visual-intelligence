"""Map sudden PM events to videos and build compact raw-sensor request windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--context-seconds", type=int, default=120)
    parser.add_argument("--merge-gap-seconds", type=int, default=60)
    return parser.parse_args()


def offset_text(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> int:
    args = parse_args()
    events = pd.read_csv(args.events_csv)
    events["sample_id"] = events["sample_id"].astype(str)
    events["sample_timestamp"] = pd.to_datetime(events["sample_timestamp"], errors="raise")
    frames = pd.read_csv(args.frame_manifest, low_memory=False)
    frames["sample_id"] = frames["sample_id"].astype(str)
    columns = [
        "sample_id", "lens_id", "source_video_path", "video_offset_sec",
        "start_time_ist", "end_time_ist", "within_declared_run",
    ]
    mapped = events.merge(frames[columns], on="sample_id", how="left", validate="one_to_many")
    if mapped["source_video_path"].isna().any() or mapped["video_offset_sec"].isna().any():
        raise ValueError("One or more events lack video coverage")
    mapped["video_offset_hhmmss"] = mapped["video_offset_sec"].map(offset_text)
    mapped = mapped.sort_values(["date", "sample_timestamp", "event_rank", "lens_id"])
    event_video_columns = [
        "event_rank", "date", "run_id", "sample_id", "sample_timestamp",
        "previous_timestamp", "previous_pm25", "sPM2", "delta_pm25",
        "absolute_delta_pm25", "direction", "immediate_reversal", "lens_id",
        "source_video_path", "video_offset_sec", "video_offset_hhmmss",
        "start_time_ist", "end_time_ist", "within_declared_run",
    ]

    context = pd.Timedelta(seconds=args.context_seconds)
    merge_gap = pd.Timedelta(seconds=args.merge_gap_seconds)
    clusters: list[dict[str, object]] = []
    for run_id, group in events.sort_values("sample_timestamp").groupby("run_id", sort=False):
        run_frames = mapped[mapped["run_id"].eq(run_id)]
        run_start = pd.to_datetime(run_frames["start_time_ist"].iloc[0])
        run_end = pd.to_datetime(run_frames["end_time_ist"].iloc[0])
        current: dict[str, object] | None = None
        for row in group.itertuples(index=False):
            start = max(row.sample_timestamp - context, run_start)
            end = min(row.sample_timestamp + context, run_end)
            if current is None or start > current["end_ist"] + merge_gap:
                if current is not None:
                    clusters.append(current)
                current = {
                    "date": str(row.date), "run_id": run_id,
                    "start_ist": start, "end_ist": end,
                    "event_ranks": [int(row.event_rank)],
                    "event_timestamps_ist": [row.sample_timestamp],
                    "maximum_absolute_change": float(row.absolute_delta_pm25),
                    "maximum_pm25": float(row.sPM2),
                }
            else:
                current["end_ist"] = max(current["end_ist"], end)
                current["event_ranks"].append(int(row.event_rank))
                current["event_timestamps_ist"].append(row.sample_timestamp)
                current["maximum_absolute_change"] = max(
                    current["maximum_absolute_change"], float(row.absolute_delta_pm25)
                )
                current["maximum_pm25"] = max(current["maximum_pm25"], float(row.sPM2))
        if current is not None:
            clusters.append(current)

    video_paths = (
        mapped[["run_id", "lens_id", "source_video_path"]]
        .drop_duplicates().pivot(index="run_id", columns="lens_id", values="source_video_path")
    )
    rows = []
    for index, cluster in enumerate(sorted(clusters, key=lambda item: item["start_ist"]), start=1):
        run_id = cluster["run_id"]
        run_rows = mapped[mapped["run_id"].eq(run_id)]
        run_start = pd.to_datetime(run_rows["start_time_ist"].iloc[0])
        start_offset = (cluster["start_ist"] - run_start).total_seconds()
        end_offset = (cluster["end_ist"] - run_start).total_seconds()
        start_local = cluster["start_ist"].tz_localize(args.timezone)
        end_local = cluster["end_ist"].tz_localize(args.timezone)
        paths = video_paths.loc[run_id]
        rows.append({
            "request_window_id": index,
            "date": cluster["date"],
            "timezone": args.timezone,
            "run_id": run_id,
            "start_ist": cluster["start_ist"],
            "end_ist": cluster["end_ist"],
            "start_utc": start_local.tz_convert("UTC").tz_localize(None),
            "end_utc": end_local.tz_convert("UTC").tz_localize(None),
            "duration_seconds": int((cluster["end_ist"] - cluster["start_ist"]).total_seconds()),
            "video_start_offset_seconds": start_offset,
            "video_end_offset_seconds": end_offset,
            "video_start_offset_hhmmss": offset_text(start_offset),
            "video_end_offset_hhmmss": offset_text(end_offset),
            "event_ranks": "|".join(map(str, cluster["event_ranks"])),
            "event_timestamps_ist": "|".join(map(str, cluster["event_timestamps_ist"])),
            "maximum_absolute_change_ug_m3": cluster["maximum_absolute_change"],
            "maximum_pm25_ug_m3": cluster["maximum_pm25"],
            "lens1_video": paths.get(1, ""),
            "lens2_video": paths.get(2, ""),
            "lens6_video": paths.get(6, ""),
        })
    windows = pd.DataFrame(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapped[event_video_columns].to_csv(output / "all_5day_event_video_timestamps.csv", index=False)
    windows.to_csv(output / "all_5day_raw_1s_request_windows.csv", index=False)
    summary = {
        "events": int(events["event_rank"].nunique()),
        "event_video_rows": len(mapped),
        "lenses": sorted(int(value) for value in mapped["lens_id"].unique()),
        "request_windows": len(windows),
        "dates": sorted(events["date"].astype(str).unique()),
        "total_requested_seconds": int(windows["duration_seconds"].sum()),
        "context_seconds_each_side": args.context_seconds,
        "merge_gap_seconds": args.merge_gap_seconds,
        "timezone": args.timezone,
    }
    (output / "all_5day_request_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(windows[[
        "request_window_id", "date", "run_id", "start_ist", "end_ist",
        "video_start_offset_hhmmss", "video_end_offset_hhmmss", "event_ranks",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
