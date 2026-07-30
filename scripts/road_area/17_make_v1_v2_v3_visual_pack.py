from pathlib import Path
import argparse
import cv2
import pandas as pd
import numpy as np


IMAGE_PATH_CANDIDATES = [
    "processed_frame_path",
    "preprocessed_frame_path",
    "frame_path",
    "image_path",
    "source_frame_path",
]


def resolve_path(value):
    if pd.isna(value):
        return None

    p = Path(str(value))

    if p.is_absolute() and p.exists():
        return p

    p2 = Path.cwd() / p
    if p2.exists():
        return p2

    return None


def find_image_path(row):
    for col in IMAGE_PATH_CANDIDATES:
        if col in row.index:
            p = resolve_path(row[col])
            if p is not None:
                return p
    return None


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_samples(df, version):
    """
    Choose 2 representative rows:
    1. Typical/median occlusion row
    2. High occlusion row
    """
    occ_candidates = [
        f"road_area_vehicle_occlusion_fraction_{version.lower()}",
        "road_area_vehicle_occlusion_fraction_v3",
        "road_area_vehicle_occlusion_fraction_v2",
        "road_area_vehicle_occlusion_fraction",
    ]

    occ_col = find_col(df, occ_candidates)

    if occ_col is None:
        # fallback: just take first 2 valid image rows
        return df.head(2).copy()

    usable = df.copy()
    usable[occ_col] = pd.to_numeric(usable[occ_col], errors="coerce")
    usable = usable.dropna(subset=[occ_col])

    if len(usable) == 0:
        return df.head(2).copy()

    median_val = usable[occ_col].median()
    typical = usable.iloc[(usable[occ_col] - median_val).abs().argsort().iloc[:1]]
    high = usable.sort_values(occ_col, ascending=False).head(1)

    out = pd.concat([typical, high], axis=0).drop_duplicates()

    if len(out) < 2:
        out = usable.head(2)

    return out.head(2).copy()


def metric_text(row, version):
    v = version.lower()

    visible_col = find_col(
        row.to_frame().T,
        [
            "visible_road_area_m2_depth_est",
            "estimated_road_area_m2",
        ],
    )

    adjusted_col = find_col(
        row.to_frame().T,
        [
            f"road_area_m2_occlusion_adjusted_conservative_{v}",
            "road_area_m2_occlusion_adjusted_conservative_v3",
            "road_area_m2_occlusion_adjusted_conservative_v2",
            "road_area_m2_occlusion_adjusted_conservative",
        ],
    )

    occ_col = find_col(
        row.to_frame().T,
        [
            f"road_area_vehicle_occlusion_fraction_{v}",
            "road_area_vehicle_occlusion_fraction_v3",
            "road_area_vehicle_occlusion_fraction_v2",
            "road_area_vehicle_occlusion_fraction",
        ],
    )

    status_col = find_col(
        row.to_frame().T,
        [
            f"final_road_area_feature_status_{v}",
            "final_road_area_feature_status_v3",
            "final_road_area_feature_status_v2",
            "final_road_area_feature_status",
        ],
    )

    quality_col = find_col(
        row.to_frame().T,
        [
            f"vehicle_occlusion_quality_{v}",
            "vehicle_occlusion_quality_v3",
            "vehicle_occlusion_quality_v2",
            "vehicle_occlusion_quality",
        ],
    )

    sample = row.get("sample_index", row.get("sensor_row_id", "NA"))

    parts = [
        f"{version.upper()} | sample={sample}",
    ]

    if visible_col:
        parts.append(f"visible={float(row[visible_col]):.2f} m2")

    if adjusted_col:
        parts.append(f"adjusted={float(row[adjusted_col]):.2f} m2")

    if occ_col:
        parts.append(f"occ_frac={float(row[occ_col]):.3f}")

    if status_col:
        parts.append(f"status={row[status_col]}")

    if quality_col:
        parts.append(f"quality={row[quality_col]}")

    return parts


def draw_label(image, lines):
    img = image.copy()

    h, w = img.shape[:2]
    banner_h = min(160, max(90, 35 + 28 * len(lines)))

    canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
    canvas[banner_h:, :, :] = img

    y = 35
    for line in lines:
        cv2.putText(
            canvas,
            str(line),
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    return canvas


def process_version(name, csv_path, outdir):
    df = pd.read_csv(csv_path)

    selected = choose_samples(df, name)

    rows_out = []
    saved = []

    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        img_path = find_image_path(row)

        record = row.to_dict()
        record["_version"] = name
        record["_selected_rank"] = i
        record["_resolved_image_path"] = str(img_path) if img_path else ""
        rows_out.append(record)

        if img_path is None:
            print(f"[{name}] sample {i}: image path missing/not found")
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[{name}] sample {i}: could not read image: {img_path}")
            continue

        lines = metric_text(row, name)
        labelled = draw_label(img, lines)

        out_path = outdir / f"{name.lower()}_image_{i}.jpg"
        cv2.imwrite(str(out_path), labelled)
        saved.append(out_path)

        print(f"[{name}] saved:", out_path)

    return rows_out, saved


def make_contact_sheet(saved_paths, out_path):
    imgs = []

    for p in saved_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        h, w = img.shape[:2]
        target_w = 900
        scale = target_w / w
        target_h = int(h * scale)
        img = cv2.resize(img, (target_w, target_h))
        imgs.append(img)

    if not imgs:
        return

    max_w = max(i.shape[1] for i in imgs)
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

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_saved = []

    configs = [
        ("V1", Path(args.v1_csv)),
        ("V2", Path(args.v2_csv)),
        ("V3", Path(args.v3_csv)),
    ]

    for name, path in configs:
        if not path.exists():
            print(f"[{name}] missing CSV:", path)
            continue

        rows, saved = process_version(name, path, outdir)
        all_rows.extend(rows)
        all_saved.extend(saved)

    selected_csv = outdir / "selected_rows_for_visual_pack.csv"
    pd.DataFrame(all_rows).to_csv(selected_csv, index=False)
    print("Saved selected-row metadata:", selected_csv)

    make_contact_sheet(all_saved, outdir / "v1_v2_v3_contact_sheet.jpg")


if __name__ == "__main__":
    main()
