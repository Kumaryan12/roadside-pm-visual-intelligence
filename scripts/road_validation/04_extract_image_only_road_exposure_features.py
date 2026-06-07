from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


def read_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    return mask > 0


def compute_road_exposure_features(mask):
    h, w = mask.shape
    total_pixels = h * w
    road_pixels = int(mask.sum())

    road_area_ratio = road_pixels / total_pixels if total_pixels > 0 else np.nan

    y = np.arange(h)
    y_norm = y / max(h - 1, 1)

    top_cut = h // 3
    mid_cut = 2 * h // 3

    far_mask = mask[:top_cut, :]
    mid_mask = mask[top_cut:mid_cut, :]
    near_mask = mask[mid_cut:, :]

    far_zone_pixels = far_mask.size
    mid_zone_pixels = mid_mask.size
    near_zone_pixels = near_mask.size

    far_road_ratio = far_mask.sum() / far_zone_pixels
    mid_road_ratio = mid_mask.sum() / mid_zone_pixels
    near_road_ratio = near_mask.sum() / near_zone_pixels

    if road_pixels > 0:
        far_road_share_of_total_road = far_mask.sum() / road_pixels
        mid_road_share_of_total_road = mid_mask.sum() / road_pixels
        near_road_share_of_total_road = near_mask.sum() / road_pixels
    else:
        far_road_share_of_total_road = np.nan
        mid_road_share_of_total_road = np.nan
        near_road_share_of_total_road = np.nan

    weights = (y_norm ** 2).reshape(h, 1)
    weighted_road_sum = (mask * weights).sum()
    total_weight_sum = np.ones_like(mask, dtype=float) * weights
    total_weight_sum = total_weight_sum.sum()

    bottom_weighted_road_exposure = (
        weighted_road_sum / total_weight_sum
        if total_weight_sum > 0
        else np.nan
    )

    # Concentration of road pixels near the bottom.
    if road_pixels > 0:
        road_y_positions = np.where(mask)[0]
        road_vertical_centroid_norm = road_y_positions.mean() / max(h - 1, 1)
        road_vertical_spread_norm = road_y_positions.std() / max(h - 1, 1)
    else:
        road_vertical_centroid_norm = np.nan
        road_vertical_spread_norm = np.nan

    return {
        "road_area_ratio_px": road_area_ratio,
        "road_area_percent_px": road_area_ratio * 100.0,
        "far_road_ratio": far_road_ratio,
        "mid_road_ratio": mid_road_ratio,
        "near_road_ratio": near_road_ratio,
        "far_road_share_of_total_road": far_road_share_of_total_road,
        "mid_road_share_of_total_road": mid_road_share_of_total_road,
        "near_road_share_of_total_road": near_road_share_of_total_road,
        "bottom_weighted_road_exposure": bottom_weighted_road_exposure,
        "road_vertical_centroid_norm": road_vertical_centroid_norm,
        "road_vertical_spread_norm": road_vertical_spread_norm,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manual-sheet",
        default="outputs/validation/road_feature_validation_pack_v1/manual_label_sheet.csv",
    )

    parser.add_argument(
        "--mask-dir",
        default="outputs/validation/road_feature_validation_pack_v1/masks",
    )

    parser.add_argument(
        "--output-csv",
        default="outputs/validation/road_feature_validation_pack_v1/image_only_road_exposure_features.csv",
    )

    args = parser.parse_args()

    manual_sheet = Path(args.manual_sheet)
    mask_dir = Path(args.mask_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manual_sheet)

    rows = []

    for idx, row in df.iterrows():
        candidate_mask_paths = []

        for col in ["mask_path", "road_mask_path", "mask_file", "file_name", "image_name"]:
            if col in df.columns and pd.notna(row[col]):
                candidate_mask_paths.append(mask_dir / Path(str(row[col])).name)

        candidate_mask_paths.append(mask_dir / f"{idx:04d}_road_mask.png")
        candidate_mask_paths.append(mask_dir / f"{idx}_road_mask.png")

        mask_path = None

        for p in candidate_mask_paths:
            if p.exists():
                mask_path = p
                break

        if mask_path is None:
            print(f"[WARN] No mask found for row {idx}")
            continue

        mask = read_mask(mask_path)
        features = compute_road_exposure_features(mask)

        out_row = row.to_dict()
        out_row["resolved_mask_path"] = str(mask_path)
        out_row.update(features)

        rows.append(out_row)

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Shape:", out.shape)

    feature_cols = [
        "road_area_percent_px",
        "far_road_ratio",
        "mid_road_ratio",
        "near_road_ratio",
        "bottom_weighted_road_exposure",
        "road_vertical_centroid_norm",
    ]

    existing = [c for c in feature_cols if c in out.columns]
    print(out[existing].describe().T)


if __name__ == "__main__":
    main()