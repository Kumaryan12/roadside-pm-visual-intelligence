"""Extract ImageNet CNN embeddings from a registered image manifest."""

import argparse
from pathlib import Path

import pandas as pd

from roadside_pm.features.images.resnet import extract_torchvision_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-col", required=True)
    parser.add_argument("--id-col", default="sample_index")
    parser.add_argument("--output-array", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "mobilenet_v2", "efficientnet_b0", "convnext_tiny"],
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional PM25Vision ResNet50 encoder checkpoint.",
    )
    parser.add_argument("--filter-col")
    parser.add_argument("--filter-value")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.filter_col:
        if args.filter_col not in manifest:
            raise ValueError(f"Missing filter column {args.filter_col!r}")
        manifest = manifest[manifest[args.filter_col].astype(str) == str(args.filter_value)].copy()
    extract_torchvision_embeddings(
        manifest.reset_index(drop=True),
        image_column=args.image_col,
        id_column=args.id_col,
        output_array=Path(args.output_array),
        output_index=Path(args.output_index),
        batch_size=args.batch_size,
        device=args.device,
        backbone=args.backbone,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
