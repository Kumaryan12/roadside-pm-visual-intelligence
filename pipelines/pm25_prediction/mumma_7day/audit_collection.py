"""Audit MUMMA sensor/video inputs without modifying raw data."""

import argparse, json
from pathlib import Path
import pandas as pd, yaml
from roadside_pm.data.collection_audit import audit_collection_manifest

ROOT = Path(__file__).resolve().parents[3]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config", default="configs/datasets/mumma_7day.yaml"); p.add_argument("--output-dir", required=True); a=p.parse_args()
    config_path=Path(a.config); config_path=config_path if config_path.is_absolute() else ROOT/config_path; c=yaml.safe_load(config_path.read_text())
    manifest_path=Path(c["collection_manifest"]); manifest_path=manifest_path if manifest_path.is_absolute() else ROOT/manifest_path
    detail, summary=audit_collection_manifest(pd.read_csv(manifest_path), project_root=ROOT, timestamp_column=c["columns"]["timestamp"], target_column=c["columns"]["pm25"], expected_interval_seconds=float(c["sampling_interval_seconds"]))
    out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); detail.to_csv(out/"collection_audit.csv",index=False); (out/"collection_audit.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); return 0 if summary["status"]=="pass" else 1
if __name__ == "__main__": raise SystemExit(main())
