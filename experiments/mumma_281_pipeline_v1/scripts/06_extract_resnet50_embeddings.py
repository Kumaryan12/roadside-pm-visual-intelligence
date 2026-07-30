from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models


ROOT = Path("experiments/mumma_281_pipeline_v1")
DATA = ROOT / "data/processed/final_feature_table.csv"

OUT_DIR = ROOT / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPY = OUT_DIR / "mumma281_resnet50_gap_embeddings.npy"
OUT_INDEX = OUT_DIR / "mumma281_resnet50_gap_index.csv"

PROJECT_ROOT = Path(".")


def resolve_image_path(row):
    key = str(row["processed_frame_key"])

    direct_candidates = [
        PROJECT_ROOT / "outputs/preprocessed_frames_v2/run_20260223_114432_8289/lens1" / f"{key}.jpg",
        PROJECT_ROOT / "outputs/preprocessed_frames_v2/run_20260223_114432_8289/lens1" / f"{key}.png",
        PROJECT_ROOT / "outputs/figures/idd_detector_annotations" / f"{key}_lens1_idd_pred.jpg",
    ]

    for p in direct_candidates:
        if p.exists():
            return p

    # Fallback search. Slower, but useful if folder name differs.
    matches = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        matches.extend(PROJECT_ROOT.rglob(f"*{key}*{ext[-4:]}"))

    # Prefer original preprocessed frame over masks/contact sheets/annotations.
    preferred = [
        p for p in matches
        if "preprocessed_frames" in str(p)
        and "mask" not in str(p).lower()
        and "sheet" not in str(p).lower()
    ]

    if preferred:
        return preferred[0]

    if matches:
        return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(DATA)

    if "processed_frame_key" not in df.columns:
        raise ValueError("processed_frame_key column not found in final_feature_table.csv")

    if args.limit:
        df = df.head(args.limit).copy()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("Device:", device)
    print("Rows:", len(df))

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    preprocess = weights.transforms()

    embeddings = []
    index_rows = []

    for i, row in df.iterrows():
        img_path = resolve_image_path(row)

        try:
            if img_path is None:
                raise FileNotFoundError(f"No image found for key: {row['processed_frame_key']}")

            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(x).detach().cpu().numpy().reshape(-1)

            status = "success"
            error = ""

        except Exception as e:
            emb = np.full(2048, np.nan, dtype=np.float32)
            status = "failed"
            error = str(e)

        embeddings.append(emb.astype(np.float32))

        index_rows.append({
            "row_id": i,
            "timestamp": row.get("timestamp", None),
            "value.sPM2": row.get("value.sPM2", None),
            "processed_frame_key": row.get("processed_frame_key", None),
            "image_path": str(img_path) if img_path is not None else "",
            "resnet_status": status,
            "resnet_error": error,
        })

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    E = np.vstack(embeddings)
    idx = pd.DataFrame(index_rows)

    np.save(OUT_NPY, E)
    idx.to_csv(OUT_INDEX, index=False)

    print("\nSaved:")
    print(OUT_NPY)
    print(OUT_INDEX)
    print("Embedding shape:", E.shape)
    print("Status:")
    print(idx["resnet_status"].value_counts())

    print("\nPreview:")
    print(idx[["processed_frame_key", "image_path", "resnet_status", "resnet_error"]].head(10).to_string(index=False))

    print("\nFinite rows:", np.isfinite(E).all(axis=1).sum(), "/", len(E))


if __name__ == "__main__":
    main()