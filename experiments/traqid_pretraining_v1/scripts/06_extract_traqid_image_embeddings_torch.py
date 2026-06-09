from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        try:
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
            ok = True
            err = ""
        except Exception as exc:
            # Return black image if loading fails.
            x = torch.zeros(3, 224, 224)
            ok = False
            err = str(exc)

        return x, ok, err


def get_device(prefer: str) -> torch.device:
    prefer = prefer.lower()

    if prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("WARNING: MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError("device must be auto, mps, or cpu")


def build_model(model_name: str, device: torch.device):
    model_name = model_name.lower()

    if model_name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        feature_dim = 1280

        # Remove classifier. Output shape becomes [B, 1280, 7, 7],
        # then we global-average-pool manually.
        feature_extractor = model.features

        preprocess = weights.transforms()

    elif model_name == "efficientnetb0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        feature_dim = 1280
        feature_extractor = model.features
        preprocess = weights.transforms()

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        feature_dim = 2048

        # Everything before avgpool/fc.
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
        preprocess = weights.transforms()

    else:
        raise ValueError(
            f"Unknown model_name={model_name}. "
            "Use mobilenetv2, efficientnetb0, or resnet50."
        )

    feature_extractor.eval()
    feature_extractor.to(device)

    return feature_extractor, preprocess, feature_dim


def make_memmap(path: Path, shape: tuple, overwrite: bool):
    if path.exists() and overwrite:
        path.unlink()

    if path.exists() and not overwrite:
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape:
            raise ValueError(
                f"Existing embedding has wrong shape: {path}, "
                f"found={arr.shape}, expected={shape}. "
                "Use --overwrite."
            )
        return arr

    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype="float32",
        shape=shape,
    )


@torch.inference_mode()
def extract_view_embeddings(
    *,
    df: pd.DataFrame,
    view: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    feature_dim: int,
    output_path: Path,
    status_df: pd.DataFrame,
    overwrite: bool,
):
    path_col = f"{view}_path"
    status_col = f"{view}_embed_status"
    error_col = f"{view}_embed_error"

    if path_col not in df.columns:
        raise ValueError(f"Manifest missing {path_col}")

    paths = df[path_col].astype(str).tolist()
    n = len(paths)

    emb = make_memmap(output_path, (n, feature_dim), overwrite=overwrite)

    status_df[status_col] = "pending"
    status_df[error_col] = ""

    dataset = ImagePathDataset(paths, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"\nExtracting {view} embeddings")
    print("Rows:", n)
    print("Output:", output_path)
    print("Batch size:", batch_size)
    print("Num workers:", num_workers)

    start_time = time.time()
    row_start = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"{view}"), start=1):
        x, ok_flags, errors = batch

        bsz = x.shape[0]
        row_end = row_start + bsz

        x = x.to(device)

        feats = model(x)

        # Global average pooling.
        if feats.ndim == 4:
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, output_size=(1, 1))
            feats = feats.flatten(1)

        feats_np = feats.detach().cpu().numpy().astype("float32")

        emb[row_start:row_end, :] = feats_np

        ok_flags = ok_flags.numpy().astype(bool)

        for i in range(bsz):
            global_i = row_start + i

            if ok_flags[i]:
                status_df.at[global_i, status_col] = "success"
                status_df.at[global_i, error_col] = ""
            else:
                status_df.at[global_i, status_col] = "image_load_failed"
                status_df.at[global_i, error_col] = str(errors[i])

        row_start = row_end

    emb.flush()

    elapsed = time.time() - start_time
    success_count = int((status_df[status_col] == "success").sum())
    fail_count = int((status_df[status_col] != "success").sum())

    print(f"\n{view} complete")
    print("Success:", success_count)
    print("Failed :", fail_count)
    print("Time sec:", round(elapsed, 2))
    print("Images/sec:", round(n / max(elapsed, 1e-9), 2))

    return status_df


def create_fused_embeddings(
    *,
    front_path: Path,
    rear_path: Path,
    mean_path: Path,
    concat_path: Path,
    n: int,
    feature_dim: int,
    overwrite: bool,
):
    print("\nCreating front/rear fused embeddings...")

    front = np.load(front_path, mmap_mode="r")
    rear = np.load(rear_path, mmap_mode="r")

    mean_emb = make_memmap(mean_path, (n, feature_dim), overwrite=overwrite)
    concat_emb = make_memmap(concat_path, (n, feature_dim * 2), overwrite=overwrite)

    chunk = 2048

    for start in tqdm(range(0, n, chunk), desc="fusion"):
        end = min(start + chunk, n)

        mean_emb[start:end, :] = (front[start:end, :] + rear[start:end, :]) / 2.0
        concat_emb[start:end, :feature_dim] = front[start:end, :]
        concat_emb[start:end, feature_dim:] = rear[start:end, :]

    mean_emb.flush()
    concat_emb.flush()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/embeddings",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/embedding_extraction",
    )

    parser.add_argument(
        "--model",
        default="mobilenetv2",
        choices=["mobilenetv2", "efficientnetb0", "resnet50"],
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
    )

    parser.add_argument(
        "--views",
        default="front,rear",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    required_cols = [
        "row_id",
        "image_id",
        "created_at",
        "date",
        "split_date_chrono",
        "front_path",
        "rear_path",
        "PM2.5",
        "PM10",
        "aqi",
        "Temperature",
        "Humidity",
        "Season",
        "Day_or_Night",
        "aqi_cat",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    if "front_exists" in df.columns:
        df = df[df["front_exists"] == True].copy()
    if "rear_exists" in df.columns:
        df = df[df["rear_exists"] == True].copy()

    df = df.reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).copy().reset_index(drop=True)

    views = [v.strip().lower() for v in args.views.split(",") if v.strip()]
    for v in views:
        if v not in {"front", "rear"}:
            raise ValueError("views must contain only front and/or rear")

    device = get_device(args.device)

    print("=" * 90)
    print("TRAQID IMAGE EMBEDDING EXTRACTION — PYTORCH")
    print("=" * 90)
    print("Manifest:", manifest_path)
    print("Rows:", len(df))
    print("Model:", args.model)
    print("Device:", device)
    print("Batch size:", args.batch_size)
    print("Num workers:", args.num_workers)
    print("Views:", views)

    print("\nSplit counts:")
    print(df["split_date_chrono"].value_counts().to_string())

    model, transform, feature_dim = build_model(args.model, device)

    status_df = df[
        [
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
            "front_path",
            "rear_path",
        ]
    ].copy()

    prefix = f"traqid_{args.model}_torch"
    output_files = {}

    for view in views:
        output_path = out_dir / f"{prefix}_{view}_embeddings.npy"

        status_df = extract_view_embeddings(
            df=df,
            view=view,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_dim=feature_dim,
            output_path=output_path,
            status_df=status_df,
            overwrite=args.overwrite,
        )

        output_files[f"{view}_embeddings"] = str(output_path)

    if "front" in views and "rear" in views:
        front_path = out_dir / f"{prefix}_front_embeddings.npy"
        rear_path = out_dir / f"{prefix}_rear_embeddings.npy"

        mean_path = out_dir / f"{prefix}_front_rear_mean_embeddings.npy"
        concat_path = out_dir / f"{prefix}_front_rear_concat_embeddings.npy"

        create_fused_embeddings(
            front_path=front_path,
            rear_path=rear_path,
            mean_path=mean_path,
            concat_path=concat_path,
            n=len(df),
            feature_dim=feature_dim,
            overwrite=args.overwrite,
        )

        output_files["front_rear_mean_embeddings"] = str(mean_path)
        output_files["front_rear_concat_embeddings"] = str(concat_path)

    index_out = out_dir / f"{prefix}_embedding_index.csv"
    status_df.to_csv(index_out, index=False)

    summary = {
        "manifest": str(manifest_path),
        "rows": int(len(df)),
        "model": args.model,
        "backend": "pytorch",
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "feature_dim_per_view": int(feature_dim),
        "views": views,
        "output_files": output_files,
        "index_csv": str(index_out),
        "split_counts": df["split_date_chrono"].value_counts().to_dict(),
        "warning": (
            "Do not evaluate by random image split. Use date/split_date_chrono groups."
        ),
    }

    for view in views:
        status_col = f"{view}_embed_status"
        summary[f"{view}_status_counts"] = (
            status_df[status_col].value_counts(dropna=False).to_dict()
        )

    summary_out = report_dir / f"{prefix}_embedding_extraction_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    for k, v in output_files.items():
        print(f" - {k}: {v}")
    print(" - index:", index_out)
    print(" - summary:", summary_out)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()