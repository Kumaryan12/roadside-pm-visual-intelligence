from pathlib import Path
import argparse
import re
import subprocess

import cv2
import numpy as np
import pandas as pd


# Lens-1 preprocessing config used in your project
LENS1_CROP = (0.25, 0.10, 0.70, 0.88)
LENS1_ROTATION = "rot90_counterclockwise"
LENS1_MASK = (0.00, 0.82, 1.00, 1.00)


def resolve_path(value):
    if value is None or pd.isna(value):
        return None

    p = Path(str(value))

    if p.is_absolute() and p.exists():
        return p

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return None


def parse_offset_from_key(key):
    """
    Example:
    sample_0000_run_20260223_114432_8289_offset_000567169_lens1
    means offset = 567.169 seconds
    """
    if key is None or pd.isna(key):
        return None

    m = re.search(r"offset_(\d+)_lens", str(key))
    if not m:
        return None

    millis = int(m.group(1))
    return millis / 1000.0


def get_video_offset(row):
    for col in ["video_offset_sec", "offset_sec"]:
        if col in row.index:
            try:
                val = float(row[col])
                if np.isfinite(val):
                    return val
            except Exception:
                pass

    for col in ["source_frame_key", "processed_frame_key"]:
        if col in row.index:
            val = parse_offset_from_key(row[col])
            if val is not None:
                return val

    return None


def extract_raw_frame(video_path, offset_sec, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{offset_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:])

    if not out_path.exists():
        raise RuntimeError("ffmpeg completed but output image missing")

    return out_path


def crop_image(image, crop_ratios):
    h, w = image.shape[:2]
    x1r, y1r, x2r, y2r = crop_ratios

    x1 = int(x1r * w)
    y1 = int(y1r * h)
    x2 = int(x2r * w)
    y2 = int(y2r * h)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return image[y1:y2, x1:x2]


def rotate_image(image, rotation_name):
    if rotation_name == "rot90_counterclockwise":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation_name == "rot90_clockwise":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_name == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def apply_platform_mask(image, mask_ratios):
    h, w = image.shape[:2]
    x1r, y1r, x2r, y2r = mask_ratios

    x1 = int(x1r * w)
    y1 = int(y1r * h)
    x2 = int(x2r * w)
    y2 = int(y2r * h)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    out = image.copy()
    out[y1:y2, x1:x2] = (0, 0, 0)

    return out


def preprocess_lens1(raw_image):
    cropped = crop_image(raw_image, LENS1_CROP)
    rotated = rotate_image(cropped, LENS1_ROTATION)
    masked = apply_platform_mask(rotated, LENS1_MASK)
    return masked


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def select_two_sample_indices(v3_df):
    """
    Select same two samples for all versions:
    1. typical/median occlusion
    2. high-occlusion case
    """
    occ_col = find_col(
        v3_df,
        [
            "road_area_vehicle_occlusion_fraction_v3",
            "road_area_vehicle_occlusion_fraction",
        ],
    )

    if occ_col is None:
        return list(v3_df["sample_index"].head(2).astype(int))

    df = v3_df.copy()
    df[occ_col] = pd.to_numeric(df[occ_col], errors="coerce")
    df = df.dropna(subset=[occ_col])

    median_val = df[occ_col].median()

    typical = df.iloc[(df[occ_col] - median_val).abs().argsort().iloc[:1]]
    high = df.sort_values(occ_col, ascending=False).head(1)

    selected = pd.concat([typical, high]).drop_duplicates(subset=["sample_index"])
    return list(selected["sample_index"].astype(int).head(2))


def version_metric_lines(row, version):
    v = version.lower()

    visible_col = None
    for c in ["visible_road_area_m2_depth_est", "estimated_road_area_m2"]:
        if c in row.index:
            visible_col = c
            break

    adjusted_col = None
    for c in [
        f"road_area_m2_occlusion_adjusted_conservative_{v}",
        "road_area_m2_occlusion_adjusted_conservative_v3",
        "road_area_m2_occlusion_adjusted_conservative_v2",
        "road_area_m2_occlusion_adjusted_conservative",
    ]:
        if c in row.index:
            adjusted_col = c
            break

    occlusion_col = None
    for c in [
        f"road_area_vehicle_occlusion_fraction_{v}",
        "road_area_vehicle_occlusion_fraction_v3",
        "road_area_vehicle_occlusion_fraction_v2",
        "road_area_vehicle_occlusion_fraction",
    ]:
        if c in row.index:
            occlusion_col = c
            break

    status_col = None
    for c in [
        f"final_road_area_feature_status_{v}",
        "final_road_area_feature_status_v3",
        "final_road_area_feature_status_v2",
        "area_quality_flag",
    ]:
        if c in row.index:
            status_col = c
            break

    quality_col = None
    for c in [
        f"vehicle_occlusion_quality_{v}",
        "vehicle_occlusion_quality_v3",
        "vehicle_occlusion_quality_v2",
    ]:
        if c in row.index:
            quality_col = c
            break

    sample_index = row.get("sample_index", "NA")

    lines = [
        f"{version.upper()} ROAD AREA VERSION",
        f"sample_index = {sample_index}",
    ]

    if visible_col:
        try:
            lines.append(f"visible road area = {float(row[visible_col]):.2f} m2")
        except Exception:
            pass

    if adjusted_col:
        try:
            lines.append(f"occlusion-adjusted area = {float(row[adjusted_col]):.2f} m2")
        except Exception:
            pass

    if occlusion_col:
        try:
            lines.append(f"vehicle occlusion fraction = {float(row[occlusion_col]):.3f}")
        except Exception:
            pass

    if quality_col:
        lines.append(f"occlusion quality = {row[quality_col]}")

    if status_col:
        lines.append(f"status = {row[status_col]}")

    return lines


def draw_text_panel(image, lines):
    img = image.copy()
    h, w = img.shape[:2]

    panel_h = 190
    out = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    out[panel_h:, :, :] = img

    y = 36
    for line in lines:
        cv2.putText(
            out,
            str(line),
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 32

    return out


def get_or_regenerate_processed_frame(row, cache_dir):
    """
    Priority:
    1. existing area_input_image_path / processed_frame_path
    2. regenerate from source_video_path + offset
    """
    for col in ["area_input_image_path", "processed_frame_path", "source_frame_path"]:
        if col in row.index:
            p = resolve_path(row[col])
            if p is not None:
                img = cv2.imread(str(p))
                if img is not None:
                    return img

    video_path = resolve_path(row.get("source_video_path", None))
    offset_sec = get_video_offset(row)

    if video_path is None:
        raise FileNotFoundError("No existing image and source_video_path not found.")

    if offset_sec is None:
        raise ValueError("Could not infer video offset from row.")

    sample_index = int(row.get("sample_index", 0))
    raw_path = cache_dir / "raw_regenerated" / f"sample_{sample_index:05d}_raw.jpg"

    extract_raw_frame(video_path, offset_sec, raw_path)

    raw_img = cv2.imread(str(raw_path))
    if raw_img is None:
        raise RuntimeError(f"Could not read regenerated raw frame: {raw_path}")

    return preprocess_lens1(raw_img)


def save_version_images(version, csv_path, sample_indices, outdir):
    df = pd.read_csv(csv_path)

    if "sample_index" not in df.columns:
        raise ValueError(f"{csv_path} has no sample_index column")

    saved = []
    metadata = []

    for rank, sample_index in enumerate(sample_indices, start=1):
        sub = df[df["sample_index"].astype(int) == int(sample_index)]

        if sub.empty:
            print(f"[{version}] sample_index {sample_index} not found")
            continue

        row = sub.iloc[0]

        try:
            processed = get_or_regenerate_processed_frame(row, outdir)
        except Exception as exc:
            print(f"[{version}] failed sample {sample_index}: {exc}")
            continue

        lines = version_metric_lines(row, version)
        labelled = draw_text_panel(processed, lines)

        out_path = outdir / f"{version.lower()}_sample_{sample_index:05d}_image_{rank}.jpg"
        cv2.imwrite(str(out_path), labelled)

        saved.append(out_path)

        record = row.to_dict()
        record["_version"] = version
        record["_selected_rank"] = rank
        record["_output_image_path"] = str(out_path)
        metadata.append(record)

        print(f"[{version}] saved {out_path}")

    return saved, metadata


def make_contact_sheet(image_paths, out_path):
    imgs = []

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        target_w = 900
        h, w = img.shape[:2]
        scale = target_w / w
        img = cv2.resize(img, (target_w, int(h * scale)))
        imgs.append(img)

    if not imgs:
        return

    max_w = max(img.shape[1] for img in imgs)
    padded = []

    for img in imgs:
        h, w = img.shape[:2]
        if w < max_w:
            pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)
        padded.append(img)

    sheet = np.concatenate(padded, axis=0)
    cv2.imwrite(str(out_path), sheet)
    print("Saved contact sheet:", out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--v1-csv", required=True)
    parser.add_argument("--v2-csv", required=True)
    parser.add_argument("--v3-csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--sample-indices", nargs="*", type=int, default=None)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    v3_df = pd.read_csv(args.v3_csv)

    if args.sample_indices:
        sample_indices = args.sample_indices[:2]
    else:
        sample_indices = select_two_sample_indices(v3_df)

    print("Selected sample indices:", sample_indices)

    all_saved = []
    all_metadata = []

    for version, csv_path in [
        ("V1", args.v1_csv),
        ("V2", args.v2_csv),
        ("V3", args.v3_csv),
    ]:
        saved, metadata = save_version_images(
            version=version,
            csv_path=csv_path,
            sample_indices=sample_indices,
            outdir=outdir,
        )
        all_saved.extend(saved)
        all_metadata.extend(metadata)

    pd.DataFrame(all_metadata).to_csv(outdir / "selected_visual_pack_metadata.csv", index=False)
    make_contact_sheet(all_saved, outdir / "v1_v2_v3_regenerated_contact_sheet.jpg")

    print("\nDone.")
    print("Output folder:", outdir)


if __name__ == "__main__":
    main()
