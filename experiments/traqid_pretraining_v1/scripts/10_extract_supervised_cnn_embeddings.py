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


class CNNRegressor(nn.Module):
    def __init__(self, backbone_name: str = "mobilenetv2", dropout: float = 0.25):
        super().__init__()

        backbone_name = backbone_name.lower()

        if backbone_name == "mobilenetv2":
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
            self.transform = weights.transforms()
            self.features = model.features
            self.feature_dim = 1280

        elif backbone_name == "efficientnetb0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
            self.transform = weights.transforms()
            self.features = model.features
            self.feature_dim = 1280

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            self.transform = weights.transforms()
            self.features = nn.Sequential(*list(model.children())[:-2])
            self.feature_dim = 2048

        else:
            raise ValueError("backbone must be mobilenetv2, efficientnetb0, or resnet50")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        z = self.features(x)
        z = self.pool(z)
        return self.head(z)

    @torch.inference_mode()
    def extract_embedding(self, x):
        z = self.features(x)
        z = self.pool(z)
        z = torch.flatten(z, 1)
        return z


class ImageManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_col: str, transform):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row[self.image_col])

        try:
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
            ok = True
            err = ""
        except Exception as exc:
            x = torch.zeros(3, 224, 224)
            ok = False
            err = str(exc)

        return x, ok, err


def get_device(prefer: str) -> torch.device:
    prefer = prefer.lower()

    if prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("WARNING: MPS requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError("device must be auto, mps, or cpu")


def make_memmap(path: Path, shape: tuple, overwrite: bool):
    if path.exists() and overwrite:
        path.unlink()

    if path.exists() and not overwrite:
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape:
            raise ValueError(
                f"Existing array shape mismatch: {path}. "
                f"Found {arr.shape}, expected {shape}. Use --overwrite."
            )
        return arr

    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype="float32",
        shape=shape,
    )


def load_trained_model(checkpoint_path: Path, backbone: str, device: torch.device) -> CNNRegressor:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_args = checkpoint.get("args", {})
    dropout = float(ckpt_args.get("dropout", 0.25))

    model = CNNRegressor(backbone_name=backbone, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loaded checkpoint:", checkpoint_path)
    print("Checkpoint phase:", checkpoint.get("phase"))
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint manifest:", checkpoint.get("resolved_manifest") or ckpt_args.get("manifest"))
    print("Dropout used:", dropout)

    return model


@torch.inference_mode()
def extract_embeddings(
    *,
    df: pd.DataFrame,
    image_col: str,
    model: CNNRegressor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_path: Path,
    overwrite: bool,
):
    dataset = ImageManifestDataset(df, image_col=image_col, transform=model.transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    n = len(df)
    feature_dim = model.feature_dim

    emb = make_memmap(output_path, (n, feature_dim), overwrite=overwrite)

    status = []
    errors = []

    row_start = 0

    for batch in tqdm(loader, desc="extract"):
        x, ok_flags, err_batch = batch

        bsz = x.shape[0]
        row_end = row_start + bsz

        x = x.to(device)
        z = model.extract_embedding(x).detach().cpu().numpy().astype("float32")

        emb[row_start:row_end, :] = z

        ok_np = ok_flags.numpy().astype(bool)

        for i in range(bsz):
            if ok_np[i]:
                status.append("success")
                errors.append("")
            else:
                status.append("image_load_failed")
                errors.append(str(err_batch[i]))

        row_start = row_end

    emb.flush()

    status_df = df.copy()
    status_df["embed_status"] = status
    status_df["embed_error"] = errors

    return status_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-col", default="front_path")
    parser.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0", "resnet50"])
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    checkpoint_path = Path(args.checkpoint)
    out_prefix = Path(args.out_prefix)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df.columns = [str(c).strip() for c in df.columns]

    if args.image_col not in df.columns:
        raise ValueError(f"Image column not found: {args.image_col}")

    df = df[df[args.image_col].notna()].copy().reset_index(drop=True)

    device = get_device(args.device)

    print("=" * 90)
    print("TRAQID-SUPERVISED CNN EMBEDDING EXTRACTION")
    print("=" * 90)
    print("Manifest:", manifest_path)
    print("Rows:", len(df))
    print("Checkpoint:", checkpoint_path)
    print("Image column:", args.image_col)
    print("Backbone:", args.backbone)
    print("Device:", device)
    print("Output prefix:", out_prefix)

    model = load_trained_model(
        checkpoint_path=checkpoint_path,
        backbone=args.backbone,
        device=device,
    )

    embeddings_path = Path(str(out_prefix) + "_embeddings.npy")
    index_path = Path(str(out_prefix) + "_embedding_index.csv")
    summary_path = Path(str(out_prefix) + "_summary.json")

    status_df = extract_embeddings(
        df=df,
        image_col=args.image_col,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_path=embeddings_path,
        overwrite=args.overwrite,
    )

    keep_cols = [
        c for c in [
            "row_id",
            "image_id",
            "created_at",
            "date",
            "split_date_chrono",
            "PM2.5",
            "PM10",
            "aqi",
            "Temperature",
            "Humidity",
            "Season",
            "Day_or_Night",
            "aqi_cat",
            args.image_col,
            "embed_status",
            "embed_error",
        ]
        if c in status_df.columns
    ]

    index_df = status_df[keep_cols].copy()
    index_df.to_csv(index_path, index=False)

    summary = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "image_col": args.image_col,
        "rows": int(len(df)),
        "backbone": args.backbone,
        "feature_dim": int(model.feature_dim),
        "device": str(device),
        "embeddings_path": str(embeddings_path),
        "index_path": str(index_path),
        "status_counts": status_df["embed_status"].value_counts(dropna=False).to_dict(),
        "note": "These embeddings come from a TRAQID-supervised CNN encoder.",
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" - embeddings:", embeddings_path)
    print(" - index:", index_path)
    print(" - summary:", summary_path)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()