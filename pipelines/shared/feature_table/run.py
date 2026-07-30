"""Run the canonical sensor/video-to-feature-table pipeline reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .stages import STAGES


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def build_context(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    inputs = config["inputs"]
    models = config["models"]
    processing = config["processing"]
    camera = config["camera"]
    feature_dir = run_dir / "features"
    manifest_dir = run_dir / "manifests"
    frame_dir = run_dir / "frames"

    return {
        "sensor_csv": resolve_input(inputs["sensor_csv"]),
        "video_root": resolve_input(inputs["video_root"]),
        "yolo_model": resolve_input(models["yolo_idd"]),
        "road_model": resolve_input(models["road_segformer"]),
        "depth_model": models["metric_depth"],
        "lenses": [str(x) for x in processing["lenses"]],
        "road_lenses": [str(x) for x in processing["road_feature_lenses"]],
        "area_lens": str(camera["road_area_lens_id"]),
        "fx": "" if camera["fx_px"] is None else str(camera["fx_px"]),
        "fy": str(camera["fy_px"] if camera["fy_px"] is not None else camera["fx_px"] or ""),
        "yolo_conf": str(processing["yolo_confidence"]),
        "yolo_imgsz": str(processing["yolo_image_size"]),
        "area_width": str(processing["road_area_width_px"]),
        "min_depth": str(processing["min_depth_m"]),
        "max_depth": str(processing["max_depth_m"]),
        "save_masks": "--save-masks" if processing.get("save_road_masks") else "",
        "save_depth": "--save-depth-npy" if processing.get("save_depth_arrays") else "",
        "lat_col": config["geospatial"]["latitude_column"],
        "lon_col": config["geospatial"]["longitude_column"],
        "timestamp_col": config["geospatial"]["timestamp_column"],
        "sample_col": config["geospatial"]["sample_id_column"],
        "osm_radius": str(config["geospatial"]["osm_radius_m"]),
        "osm_endpoint": config["geospatial"]["osm_endpoint"],
        "ee_project_flag": "--project" if config["geospatial"].get("earth_engine_project") else "",
        "ee_project": config["geospatial"].get("earth_engine_project") or "",
        "aef_buffers": [str(x) for x in config["geospatial"]["alphaearth_buffers_m"]],
        "aef_max_year": str(config["geospatial"]["alphaearth_max_available_year"]),
        "extracted_frames": str(frame_dir / "extracted"),
        "preprocessed_frames": str(frame_dir / "preprocessed"),
        "extracted_manifest": str(manifest_dir / "extracted_frames.csv"),
        "preprocessed_manifest": str(manifest_dir / "preprocessed_frames.csv"),
        "vehicle_frame": str(feature_dir / "vehicle_frame.csv"),
        "vehicle_object": str(feature_dir / "vehicle_objects.csv"),
        "vehicle_object_nms": str(feature_dir / "vehicle_objects_lens1_nms.csv"),
        "road_frame": str(feature_dir / "road_frame.csv"),
        "road_area_dir": str(run_dir / "road_area"),
        "road_area_base": str(feature_dir / "road_area_depth_base.csv"),
        "road_area_v3": str(feature_dir / "road_area_depth_gated_v3.csv"),
        "road_area_vehicle_detail": str(feature_dir / "road_area_vehicle_detail_v3.csv"),
        "road_area_canonical": str(feature_dir / "road_area_canonical.csv"),
        "osm_csv": str(feature_dir / "osm_features.csv"),
        "alphaearth_csv": str(feature_dir / "alphaearth_features.csv"),
        "vehicle_sensor": str(feature_dir / "vehicle_sensor.csv"),
        "road_sensor": str(feature_dir / "road_sensor.csv"),
        "visual_table": str(run_dir / "tables/01_visual_sensor_table.csv"),
        "osm_table": str(run_dir / "tables/02_visual_osm_table.csv"),
        "road_area_table": str(run_dir / "tables/03_visual_osm_road_area_table.csv"),
        "final_table": str(run_dir / "tables/final_feature_table.csv"),
    }


def expand_arguments(arguments: tuple[str, ...], context: dict[str, Any]) -> list[str]:
    expanded: list[str] = []
    for token in arguments:
        if token == "{lenses}":
            expanded.extend(context["lenses"])
        elif token == "{road_lenses}":
            expanded.extend(context["road_lenses"])
        elif token == "{aef_buffers}":
            expanded.extend(context["aef_buffers"])
        else:
            value = token.format(**context)
            if value:
                expanded.append(value)
    return expanded


def select_stages(start: str | None, stop: str | None) -> tuple[Any, ...]:
    names = [stage.name for stage in STAGES]
    start_index = names.index(start) if start else 0
    stop_index = names.index(stop) + 1 if stop else len(STAGES)
    if start_index >= stop_index:
        raise ValueError("--from-stage must not come after --to-stage")
    return STAGES[start_index:stop_index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipelines/feature_table_v1.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--from-stage", choices=[stage.name for stage in STAGES])
    parser.add_argument("--to-stage", choices=[stage.name for stage in STAGES])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = yaml.safe_load(config_path.read_text())
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_feature_table_v1")
    run_root = Path(config["outputs"]["root"])
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    run_dir = run_root / run_id
    context = build_context(config, run_dir)
    selected = select_stages(args.from_stage, args.to_stage)

    external_inputs = {
        "sensor_csv": Path(context["sensor_csv"]),
        "video_root": Path(context["video_root"]),
        "yolo_model": Path(context["yolo_model"]),
        "road_model": Path(context["road_model"]),
    }
    required_names = {name for stage in selected for name in stage.inputs}
    selected_external_inputs = {
        name: path for name, path in external_inputs.items() if name in required_names
    }
    missing = [name for name, path in selected_external_inputs.items() if not path.exists()]
    if args.preflight:
        report = {name: {"path": str(path), "exists": path.exists()} for name, path in selected_external_inputs.items()}
        if any(stage.name == "estimate_road_area" for stage in selected):
            report["camera_fx"] = {"value": config["camera"]["fx_px"], "configured": config["camera"]["fx_px"] is not None}
        print(json.dumps(report, indent=2))
        needs_fx = any(stage.name == "estimate_road_area" for stage in selected)
        return 1 if missing or (needs_fx and config["camera"]["fx_px"] is None) else 0
    if config["camera"]["fx_px"] is None and any(s.name == "estimate_road_area" for s in selected):
        raise ValueError("Set camera.fx_px in the pipeline config before running road area.")
    if missing and not args.dry_run:
        raise FileNotFoundError(f"Missing external pipeline inputs: {missing}. Run --preflight for paths.")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifests").mkdir(exist_ok=True)
    (run_dir / "features").mkdir(exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    manifest_path = run_dir / "run.json"

    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")

    for stage in selected:
        script = PROJECT_ROOT / stage.script
        outputs = [Path(context[name]) for name in stage.outputs]
        command = [sys.executable, str(script), *expand_arguments(stage.arguments, context)]
        record = {"stage": stage.name, "description": stage.description, "command": command, "outputs": list(map(str, outputs))}
        if args.resume and outputs and all(path.exists() for path in outputs):
            record["status"] = "skipped_existing"
            records.append(record)
            continue
        print(f"\n[{stage.name}] {stage.description}")
        print(" ".join(command))
        if args.dry_run:
            record["status"] = "dry_run"
        else:
            started = datetime.now(timezone.utc).isoformat()
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
            record.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode, started_at=started, finished_at=datetime.now(timezone.utc).isoformat())
            if result.returncode != 0:
                records.append(record)
                break
        records.append(record)
        manifest_path.write_text(json.dumps({"pipeline_id": config["pipeline_id"], "run_id": run_id, "config_path": str(config_path), "config_sha256": sha256(config_path), "created_at": datetime.now(timezone.utc).isoformat(), "stages": records}, indent=2))

    manifest_path.write_text(json.dumps({"pipeline_id": config["pipeline_id"], "run_id": run_id, "config_path": str(config_path), "config_sha256": sha256(config_path), "created_at": datetime.now(timezone.utc).isoformat(), "external_inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in selected_external_inputs.items() if path.is_file()}, "stages": records, "final_table": context["final_table"]}, indent=2))
    print(f"\nRun manifest: {manifest_path}")
    return 1 if any(record.get("status") == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
