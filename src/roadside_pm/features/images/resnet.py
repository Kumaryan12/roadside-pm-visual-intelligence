"""Canonical ImageNet CNN global-average-pool embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def select_device(preference: str = "auto"):
    import torch
    if preference == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "mps": return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if preference == "cpu": return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def extract_torchvision_embeddings(
    manifest: pd.DataFrame,
    *,
    image_column: str,
    id_column: str,
    output_array: Path,
    output_index: Path,
    batch_size: int = 32,
    device: str = "auto",
    backbone: str = "resnet50",
) -> tuple[np.ndarray, pd.DataFrame]:
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models

    for column in (image_column, id_column):
        if column not in manifest:
            raise ValueError(f"Manifest missing {column!r}")
    if manifest[id_column].duplicated().any():
        raise ValueError(f"Manifest {id_column!r} must be unique")

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        embedding_dimension = 2048
    elif backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Identity()
        embedding_dimension = 1280
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
        embedding_dimension = 1280
    elif backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Identity()
        embedding_dimension = 768
    else:
        raise ValueError(
            "backbone must be resnet50, mobilenet_v2, efficientnet_b0, or convnext_tiny"
        )
    model.eval().to(select_device(device))
    transform = weights.transforms()

    class Images(Dataset):
        def __len__(self): return len(manifest)
        def __getitem__(self, index):
            row = manifest.iloc[index]
            path = Path(str(row[image_column]))
            try:
                tensor = transform(Image.open(path).convert("RGB")); status, error = "success", ""
            except Exception as exc:
                tensor = torch.zeros(3, 224, 224); status, error = "failed", str(exc)
            return tensor, index, str(path), status, error

    output_array.parent.mkdir(parents=True, exist_ok=True)
    output_index.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(output_array, mode="w+", dtype="float32", shape=(len(manifest), embedding_dimension))
    rows = []
    model_device = next(model.parameters()).device
    with torch.inference_mode():
        for images, positions, paths, statuses, errors in DataLoader(Images(), batch_size=batch_size, shuffle=False, num_workers=0):
            values = model(images.to(model_device)).detach().cpu().numpy().astype("float32")
            for offset, position in enumerate(positions.numpy()):
                embeddings[position] = values[offset] if statuses[offset] == "success" else np.nan
                rows.append({"embedding_row": int(position), id_column: manifest.iloc[position][id_column], "image_path": paths[offset], "embedding_status": statuses[offset], "embedding_error": errors[offset]})
    embeddings.flush()
    index = pd.DataFrame(rows).sort_values("embedding_row")
    index.to_csv(output_index, index=False)
    return np.asarray(embeddings), index


def extract_resnet50_embeddings(
    manifest: pd.DataFrame,
    *,
    image_column: str,
    id_column: str,
    output_array: Path,
    output_index: Path,
    batch_size: int = 32,
    device: str = "auto",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Backward-compatible ResNet50 entry point."""
    return extract_torchvision_embeddings(
        manifest,
        image_column=image_column,
        id_column=id_column,
        output_array=output_array,
        output_index=output_index,
        batch_size=batch_size,
        device=device,
        backbone="resnet50",
    )
