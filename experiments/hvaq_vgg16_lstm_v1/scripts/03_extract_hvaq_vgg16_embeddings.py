from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_device(prefer: str) -> torch.device:
    prefer = prefer.lower()

    if prefer == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError("device must be auto, mps, or cpu")


class UniqueImageDataset(Dataset):
    def __init__(self, image_paths: list[str], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        try:
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
            ok = True
            err = ""
        except Exception as exc:
            x = torch.zeros(3, 224, 224)
            ok = False
            err = str(exc)

        return x, path, ok, err


class VGG16Embedder(nn.Module):
    def __init__(self):
        super().__init__()

        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)

        self.transform = weights.transforms()
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512

    def forward(self, x):
        z = self.features(x)
        z = self.pool(z)
        z = torch.flatten(z, 1)
        return z


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/hvaq_vgg16_lstm_v1/data/processed/hvaq_image_label_manifest.csv",
    )

    parser.add_argument(
        "--out-embeddings",
        default="experiments/hvaq_vgg16_lstm_v1/embeddings/hvaq_vgg16_avgpool512_row_embeddings.npy",
    )

    parser.add_argument(
        "--out-index",
        default="experiments/hvaq_vgg16_lstm_v1/embeddings/hvaq_vgg16_avgpool512_embedding_index.csv",
    )

    parser.add_argument(
        "--summary",
        default="experiments/hvaq_vgg16_lstm_v1/reports/hvaq_vgg16_embedding_summary.json",
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_emb = Path(args.out_embeddings)
    out_index = Path(args.out_index)
    summary_path = Path(args.summary)

    out_emb.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "row_id" not in df.columns or "image_path" not in df.columns:
        raise ValueError("Manifest must contain row_id and image_path.")

    df = df.sort_values("row_id").reset_index(drop=True)

    if not np.array_equal(df["row_id"].to_numpy(), np.arange(len(df))):
        raise ValueError("row_id must be continuous from 0 to N-1 for row-level embeddings.")

    device = get_device(args.device)

    print("=" * 90)
    print("HVAQ VGG16 EMBEDDING EXTRACTION")
    print("=" * 90)
    print("Manifest:", manifest_path)
    print("Rows:", len(df))
    print("Unique images:", df["image_path"].nunique())
    print("Device:", device)

    model = VGG16Embedder().to(device)
    model.eval()

    unique_paths = sorted(df["image_path"].unique().tolist())

    ds = UniqueImageDataset(unique_paths, transform=model.transform)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    unique_embeddings = {}
    status_rows = []

    with torch.inference_mode():
        for x, paths, ok_flags, errors in tqdm(loader, desc="extract unique images"):
            x = x.to(device)
            z = model(x).detach().cpu().numpy().astype("float32")

            for i, path in enumerate(paths):
                unique_embeddings[path] = z[i]
                status_rows.append(
                    {
                        "image_path": path,
                        "embed_status": "success" if bool(ok_flags[i]) else "image_load_failed",
                        "embed_error": "" if bool(ok_flags[i]) else str(errors[i]),
                    }
                )

    if out_emb.exists() and args.overwrite:
        out_emb.unlink()

    if out_emb.exists() and not args.overwrite:
        raise FileExistsError(f"{out_emb} exists. Use --overwrite.")

    emb = np.lib.format.open_memmap(
        out_emb,
        mode="w+",
        dtype="float32",
        shape=(len(df), model.feature_dim),
    )

    for i, row in tqdm(df.iterrows(), total=len(df), desc="writing row embeddings"):
        emb[int(row["row_id"]), :] = unique_embeddings[row["image_path"]]

    emb.flush()

    index_df = df.copy()
    index_df.to_csv(out_index, index=False)

    summary = {
        "manifest": str(manifest_path),
        "rows": int(len(df)),
        "unique_images": int(len(unique_paths)),
        "embedding_shape": [int(len(df)), int(model.feature_dim)],
        "out_embeddings": str(out_emb),
        "out_index": str(out_index),
        "status_counts": pd.DataFrame(status_rows)["embed_status"].value_counts().to_dict(),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" - embeddings:", out_emb)
    print(" - index:", out_index)
    print(" - summary:", summary_path)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()