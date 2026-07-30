import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-frame-features", required=True)
    parser.add_argument("--out-sample-features", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_frame_path = Path(args.out_frame_features)
    out_sample_path = Path(args.out_sample_features)

    out_frame_path.parent.mkdir(parents=True, exist_ok=True)
    out_sample_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    required = ["sample_index", "lens_id", "image_path", "PM2.5", "PM10"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    transform = weights.transforms()

    rows = []
    batch_imgs = []
    batch_meta = []

    def flush_batch():
        nonlocal batch_imgs, batch_meta, rows

        if not batch_imgs:
            return

        x = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            feats = model(x).detach().cpu().numpy()

        for meta, feat in zip(batch_meta, feats):
            row = dict(meta)
            for i, v in enumerate(feat):
                row[f"resnet50_{i}"] = float(v)
            rows.append(row)

        batch_imgs = []
        batch_meta = []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(str(r["image_path"]))

        if not img_path.exists():
            print("Missing image:", img_path)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            x = transform(img)
        except Exception as e:
            print("Failed:", img_path, e)
            continue

        batch_imgs.append(x)
        batch_meta.append({
            "mumma_frame_row_id": r.get("mumma_frame_row_id", np.nan),
            "sample_index": int(r["sample_index"]),
            "sensor_timestamp": r.get("sensor_timestamp", r.get("timestamp", "")),
            "lens_id": int(r["lens_id"]),
            "image_path": str(img_path),
            "PM2.5": float(r["PM2.5"]),
            "PM10": float(r["PM10"]),
            "temperature": float(r["temperature"]) if "temperature" in df.columns and pd.notna(r["temperature"]) else np.nan,
            "humidity": float(r["humidity"]) if "humidity" in df.columns and pd.notna(r["humidity"]) else np.nan,
        })

        if len(batch_imgs) >= args.batch_size:
            flush_batch()

    flush_batch()

    frame_feats = pd.DataFrame(rows)
    frame_feats.to_csv(out_frame_path, index=False)

    feat_cols = [c for c in frame_feats.columns if c.startswith("resnet50_")]

    # Mean-fuse 3 lenses per sample_index.
    sample_meta_cols = ["sample_index", "sensor_timestamp", "PM2.5", "PM10", "temperature", "humidity"]
    sample_meta_cols = [c for c in sample_meta_cols if c in frame_feats.columns]

    sample_meta = (
        frame_feats
        .sort_values(["sample_index", "lens_id"])
        .groupby("sample_index")[sample_meta_cols]
        .first()
        .reset_index(drop=True)
    )

    sample_feats = (
        frame_feats
        .groupby("sample_index")[feat_cols]
        .mean()
        .reset_index()
    )

    lens_counts = (
        frame_feats
        .groupby("sample_index")["lens_id"]
        .nunique()
        .rename("lens_count")
        .reset_index()
    )

    sample_out = sample_meta.merge(sample_feats, on="sample_index", how="left")
    sample_out = sample_out.merge(lens_counts, on="sample_index", how="left")
    sample_out = sample_out.sort_values("sample_index").reset_index(drop=True)

    sample_out.to_csv(out_sample_path, index=False)

    print("=" * 90)
    print("MUMMA RESNET50 FEATURE EXTRACTION COMPLETE")
    print("=" * 90)
    print("Manifest:", manifest_path)
    print("Frame feature rows:", len(frame_feats))
    print("Sample feature rows:", len(sample_out))
    print("Feature columns:", len(feat_cols))
    print("Saved frame features:", out_frame_path)
    print("Saved sample mean-fused features:", out_sample_path)
    print()
    print("Lens count summary:")
    print(sample_out["lens_count"].describe().to_string())
    print()
    print("Target summary:")
    print(sample_out[["PM2.5", "PM10"]].describe().T.to_string())


if __name__ == "__main__":
    main()