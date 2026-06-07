from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model-path",
        default="models/road_segmentation/best_segformer_b0_idd_binary_road/best_segformer_b0_idd_binary_road",
    )
    parser.add_argument(
        "--output-mask",
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    output_mask = Path(args.output_mask)
    output_mask.parent.mkdir(parents=True, exist_ok=True)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    processor = SegformerImageProcessor.from_pretrained(args.model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    )

    if upsampled_logits.shape[1] == 1:
        probs = torch.sigmoid(upsampled_logits)[0, 0].cpu().numpy()
        mask = probs >= args.threshold
    else:
        pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        mask = pred == 1

    mask_uint8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_mask), mask_uint8)

    print("Saved road mask:", output_mask)


if __name__ == "__main__":
    main()