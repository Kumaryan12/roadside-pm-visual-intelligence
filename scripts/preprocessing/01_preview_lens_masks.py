from pathlib import Path
import cv2
import numpy as np
import pandas as pd


MANIFEST_PATH = Path("outputs/features/processed_frame_manifest_v2.csv")
OUTPUT_DIR = Path("outputs/figures/lens_mask_previews")


# Polygon coordinates are normalized: x_ratio, y_ratio.
# These are initial masks. We will tune them after preview.
LENS_MASKS = {
    1: [
        # lower-left own vehicle/platform
        [(0.00, 0.48), (0.42, 0.48), (0.42, 1.00), (0.00, 1.00)],
        # left fisheye/edge strip
        [(0.00, 0.00), (0.05, 0.00), (0.05, 1.00), (0.00, 1.00)],
    ],
    4: [
        # left platform/body
        [(0.00, 0.00), (0.38, 0.00), (0.38, 1.00), (0.00, 1.00)],
        # bottom-left platform extension
        [(0.00, 0.62), (0.55, 0.62), (0.55, 1.00), (0.00, 1.00)],
    ],
    6: [
        # left own vehicle/body
        [(0.00, 0.00), (0.34, 0.00), (0.34, 1.00), (0.00, 1.00)],
        # lower-left platform
        [(0.00, 0.55), (0.48, 0.55), (0.48, 1.00), (0.00, 1.00)],
    ],
}


def poly_norm_to_pixels(poly, w, h):
    return np.array(
        [[int(x * w), int(y * h)] for x, y in poly],
        dtype=np.int32
    )


def apply_masks(image, lens_id):
    h, w = image.shape[:2]
    masked = image.copy()
    overlay = image.copy()

    polygons = LENS_MASKS.get(lens_id, [])

    for poly in polygons:
        pts = poly_norm_to_pixels(poly, w, h)

        # Red overlay for preview
        cv2.fillPoly(overlay, [pts], (0, 0, 255))

        # Black mask for actual preprocessed image
        cv2.fillPoly(masked, [pts], (0, 0, 0))

    preview = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

    return masked, preview


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)
    df = df[df["preprocess_status"] == "success"].copy()

    # Take a few examples from each selected lens
    preview_rows = df[df["lens_id"].isin([1, 4, 6])].groupby("lens_id").head(5)

    for _, row in preview_rows.iterrows():
        lens_id = int(row["lens_id"])
        frame_key = str(row["processed_frame_key"])
        frame_path = Path(str(row["processed_frame_path"]))

        if not frame_path.is_absolute():
            frame_path = Path.cwd() / frame_path

        image = cv2.imread(str(frame_path))
        if image is None:
            print("Could not read:", frame_path)
            continue

        masked, preview = apply_masks(image, lens_id)

        preview_path = OUTPUT_DIR / f"{frame_key}_lens{lens_id}_mask_preview.jpg"
        masked_path = OUTPUT_DIR / f"{frame_key}_lens{lens_id}_masked.jpg"

        cv2.imwrite(str(preview_path), preview)
        cv2.imwrite(str(masked_path), masked)

        print("Saved:", preview_path)
        print("Saved:", masked_path)


if __name__ == "__main__":
    main()