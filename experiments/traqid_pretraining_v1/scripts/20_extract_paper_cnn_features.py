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

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    if prefer == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError(f"Unknown device: {prefer}")


class RowImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_col: str, transform):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row[self.image_col])
        row_id = int(row["row_id"])

        try:
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
            ok = True
            err = ""
        except Exception as exc:
            x = torch.zeros(3, 224, 224)
            ok = False
            err = str(exc)

        return x, row_id, path, ok, err


class VGG16GAP(nn.Module):
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


class ResNet50GAP(nn.Module):
    def __init__(self):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        self.transform = weights.transforms()

        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )

        self.feature_dim = 2048

    def forward(self, x):
        z = self.backbone(x)
        z = torch.flatten(z, 1)
        return z


def build_model(name: str):
    name = name.lower()

    if name == "vgg16":
        return VGG16GAP()

    if name == "resnet50":
        return ResNet50GAP()

    raise ValueError("model must be vgg16 or resnet50")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )
    parser.add_argument("--view", default="front", choices=["front", "rear"])
    parser.add_argument("--model", default="vgg16", choices=["vgg16", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/embeddings/paper_style_cnn",
    )
    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/paper_style_cnn_features",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df.columns = [str(c).strip() for c in df.columns]

    image_col = f"{args.view}_path"
    exists_col = f"{args.view}_exists"

    if image_col not in df.columns:
        raise ValueError(f"Missing image column: {image_col}")

    if exists_col in df.columns:
        df = df[df[exists_col] == True].copy()

    df = df.sort_values("row_id").reset_index(drop=True)

    expected = np.arange(len(df))
    actual = df["row_id"].to_numpy()

    if not np.array_equal(actual, expected):
        raise ValueError(
            "row_id must be continuous from 0 to N-1. "
            "Your manifest row_id order is not continuous after filtering."
        )

    device = get_device(args.device)

    model = build_model(args.model)
    transform = model.transform
    feature_dim = model.feature_dim

    model = model.to(device)
    model.eval()

    out_prefix = f"traqid_paper_{args.model}_{args.view}_gap"
    emb_path = out_dir / f"{out_prefix}_features.npy"
    index_path = out_dir / f"{out_prefix}_index.csv"
    summary_path = report_dir / f"{out_prefix}_summary.json"

    if emb_path.exists() and not args.overwrite:
        raise FileExistsError(f"{emb_path} exists. Use --overwrite to replace it.")

    print("=" * 90)
    print("PAPER-STYLE CNN FEATURE EXTRACTION")
    print("=" * 90)
    print("Manifest:", manifest_path)
    print("Rows:", len(df))
    print("View:", args.view)
    print("Image column:", image_col)
    print("Model:", args.model)
    print("Feature dim:", feature_dim)
    print("Device:", device)
    print("Output:", emb_path)

    ds = RowImageDataset(df, image_col=image_col, transform=transform)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    features = np.lib.format.open_memmap(
        emb_path,
        mode="w+",
        dtype="float32",
        shape=(len(df), feature_dim),
    )

    status_rows = []

    with torch.inference_mode():
        for x, row_ids, paths, ok_flags, errors in tqdm(loader, desc=f"extract {args.model}"):
            x = x.to(device)
            z = model(x).detach().cpu().numpy().astype("float32")

            row_ids_np = row_ids.numpy()

            for i, row_id in enumerate(row_ids_np):
                features[int(row_id), :] = z[i]

                status_rows.append(
                    {
                        "row_id": int(row_id),
                        "image_path": str(paths[i]),
                        "status": "success" if bool(ok_flags[i]) else "image_load_failed",
                        "error": "" if bool(ok_flags[i]) else str(errors[i]),
                    }
                )

    features.flush()

    index_df = df[
        [
            "row_id",
            "created_at",
            "date",
            image_col,
            "PM2.5",
            "PM10",
            "aqi",
            "Season",
            "Day_or_Night",
        ]
    ].copy()

    index_df.to_csv(index_path, index=False)

    status_df = pd.DataFrame(status_rows)

    summary = {
        "manifest": str(manifest_path),
        "view": args.view,
        "image_col": image_col,
        "model": args.model,
        "feature_type": "global_average_pooling",
        "rows": int(len(df)),
        "feature_dim": int(feature_dim),
        "embedding_shape": [int(len(df)), int(feature_dim)],
        "embeddings_path": str(emb_path),
        "index_path": str(index_path),
        "status_counts": status_df["status"].value_counts().to_dict(),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" - features:", emb_path)
    print(" - index:", index_path)
    print(" - summary:", summary_path)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()