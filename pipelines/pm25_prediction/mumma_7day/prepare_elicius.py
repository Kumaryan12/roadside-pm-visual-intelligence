"""Discover ELICIUS runs and build a metadata-clipped 10-second sensor table."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd
from roadside_pm.data.elicius import discover_elicius_runs, resample_aqi_run


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--ssd-root",default="/Volumes/ELICIUS")
    parser.add_argument("--output-dir",default="artifacts/runs/mumma_3day_ingestion_v1"); parser.add_argument("--interval-seconds",type=int,default=10)
    parser.add_argument("--lenses",nargs="+",type=int,default=[1,2,6])
    parser.add_argument(
        "--dates",
        nargs="+",
        default=None,
        help="Optional ISO dates to include (for example 2026-02-02 2026-02-04).",
    )
    args=parser.parse_args()
    root=Path(args.ssd_root); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    runs=discover_elicius_runs(root,required_lenses=tuple(args.lenses))
    if args.dates:
        requested_dates = set(args.dates)
        available_dates = set(runs["date"].astype(str))
        missing_dates = sorted(requested_dates - available_dates)
        if missing_dates:
            raise ValueError(f"Requested dates were not discovered: {missing_dates}")
        runs = runs[runs["date"].astype(str).isin(requested_dates)].copy()
    runs.to_csv(output/"all_discovered_runs.csv",index=False)
    eligible=runs[runs.eligible].copy(); tables=[]; audits=[]
    for row in eligible.to_dict("records"):
        table,audit=resample_aqi_run(Path(row["sensor_csv"]),run_id=row["run_id"],date=row["date"],trip_id=row["trip_id"],start_time=row["start_time_ist"],end_time=row["end_time_ist"],interval_seconds=args.interval_seconds)
        tables.append(table); audits.append(audit)
    sensor=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame(); sensor.to_csv(output/"sensor_10s.csv",index=False)
    pd.DataFrame(audits).to_csv(output/"run_sensor_audit.csv",index=False)
    video_rows=[]
    for row in eligible.to_dict("records"):
        for lens in args.lenses: video_rows.append({"collection_id":row["collection_id"],"run_id":row["run_id"],"date":row["date"],"trip_id":row["trip_id"],"lens_id":lens,"video_path":row[f"video_lens{lens}"],"sensor_csv":row["sensor_csv"],"start_time_ist":row["start_time_ist"],"end_time_ist":row["end_time_ist"],"timezone":"Asia/Kolkata"})
    lens_suffix = "_".join(str(lens) for lens in args.lenses)
    pd.DataFrame(video_rows).to_csv(output/f"video_manifest_lenses_{lens_suffix}.csv",index=False)
    summary={"ssd_root":str(root),"requested_dates":args.dates,"discovered_runs":len(runs),"eligible_runs":len(eligible),"excluded_runs":int((~runs.eligible).sum()),"sensor_10s_rows":len(sensor),"dates":sorted(sensor.date.unique().tolist()) if len(sensor) else [],"lenses":args.lenses,"raw_ssd_modified":False}
    (output/"ingestion_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
