"""Create day-, trip-, chronological-, and spatial-group split columns."""

import argparse
from pathlib import Path
import pandas as pd, yaml
from roadside_pm.validation.dataset_splits import assign_evaluation_protocols

ROOT=Path(__file__).resolve().parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="configs/datasets/mumma_7day.yaml"); p.add_argument("--input-csv",required=True); p.add_argument("--output-csv",required=True); a=p.parse_args()
    cp=Path(a.config); cp=cp if cp.is_absolute() else ROOT/cp; c=yaml.safe_load(cp.read_text()); cols=c["columns"]; v=c["validation"]
    result=assign_evaluation_protocols(pd.read_csv(a.input_csv),date_column=cols["date"],trip_column=cols["trip_id"],latitude_column=cols["latitude"],longitude_column=cols["longitude"],chronological_test_days=int(v["chronological_test_days"]),seed=int(v["seed"]),spatial_block_size_m=float(v["spatial_block_size_m"]))
    out=Path(a.output_csv); out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False); print(result[["evaluation_date","split_chronological_day","split_grouped_trip","split_grouped_spatial"]].value_counts().sort_index()); return 0
if __name__ == "__main__": raise SystemExit(main())
