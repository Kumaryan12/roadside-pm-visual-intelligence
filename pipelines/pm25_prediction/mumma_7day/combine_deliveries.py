"""Combine independently processed MUMMA deliveries by stable sample identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_and_concat(paths: list[str], *, label: str) -> pd.DataFrame:
    frames = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "sample_id" not in frame:
            raise ValueError(f"{label} input has no sample_id: {path}")
        frame["source_delivery_csv"] = str(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csvs", nargs="+", required=True)
    parser.add_argument("--frame-manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 2, 6])
    args = parser.parse_args()

    sensor = _read_and_concat(args.sensor_csvs, label="sensor")
    if sensor["sample_id"].duplicated().any():
        duplicates = sensor.loc[sensor["sample_id"].duplicated(False), "sample_id"].unique()
        raise ValueError(f"Duplicate sensor sample_id values across deliveries: {duplicates[:10].tolist()}")

    manifest = _read_and_concat(args.frame_manifests, label="frame manifest")
    required = {"sample_id", "lens_id", "preprocess_status", "processed_frame_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Frame manifests missing columns: {sorted(missing)}")
    manifest = manifest[manifest["lens_id"].astype(int).isin(args.lenses)].copy()
    if manifest.duplicated(["sample_id", "lens_id"]).any():
        raise ValueError("Duplicate sample_id/lens_id rows across frame manifests")
    expected = len(sensor) * len(args.lenses)
    if len(manifest) != expected:
        raise ValueError(f"Expected {expected} frame rows but found {len(manifest)}")
    if set(manifest["sample_id"].astype(str)) != set(sensor["sample_id"].astype(str)):
        raise ValueError("Sensor and frame-manifest sample_id coverage differs")
    failed = ~manifest["preprocess_status"].astype(str).eq("success")
    if failed.any():
        raise ValueError(f"Frame manifests contain {int(failed.sum())} unsuccessful rows")

    sensor = sensor.sort_values(["sample_timestamp", "sample_id"]).reset_index(drop=True)
    sensor["sample_index"] = range(len(sensor))
    index_map = sensor[["sample_id", "sample_index"]]
    manifest = manifest.drop(columns=["sample_index"], errors="ignore").merge(
        index_map, on="sample_id", how="left", validate="many_to_one"
    )
    manifest = manifest.sort_values(["sample_index", "lens_id"]).reset_index(drop=True)

    output = Path(args.output_dir)
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    sensor_path = output / "sensor_10s.csv"
    frame_path = manifests / "preprocessed_frames.csv"
    sensor.to_csv(sensor_path, index=False)
    manifest.to_csv(frame_path, index=False)
    summary = {
        "sensor_csv": str(sensor_path),
        "preprocessed_frame_manifest": str(frame_path),
        "sensor_inputs": args.sensor_csvs,
        "frame_manifest_inputs": args.frame_manifests,
        "rows": int(len(sensor)),
        "frame_rows": int(len(manifest)),
        "dates": sorted(sensor["date"].astype(str).unique().tolist()),
        "runs": int(sensor["run_id"].nunique()),
        "lenses": args.lenses,
        "identity_policy": "sample_id_unique_and_exactly_one_frame_per_required_lens",
    }
    (output / "combine_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
