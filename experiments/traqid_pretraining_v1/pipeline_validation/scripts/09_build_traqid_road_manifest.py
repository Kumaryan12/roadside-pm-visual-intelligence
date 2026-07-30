from pathlib import Path
import pandas as pd

MANIFEST = Path("experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv")
OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "traqid_front_road_manifest.csv"

df = pd.read_csv(MANIFEST)

df = df[
    (df["front_exists"] == True)
    & (df["rear_exists"] == True)
    & (df["env_plausible"] == True)
].copy().reset_index(drop=True)

road_manifest = pd.DataFrame({
    "sample_index": df["row_id"].astype(int),
    "sensor_timestamp": df["created_at"].astype(str),
    "sample_unix": "",
    "lens_id": 1,
    "matched_run_id": "traqid_front",
    "video_offset_sec": df.index.astype(int),
    "processed_frame_key": df["image_id"].astype(str),
    "processed_frame_path": df["front_path"].astype(str),

    # Required by scripts/road_segmentation/03_extract_segformer_road_condition_features.py
    "preprocess_status": "success",
    "preprocess_error": "",
})

road_manifest.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Shape:", road_manifest.shape)
print(road_manifest.head().to_string(index=False))