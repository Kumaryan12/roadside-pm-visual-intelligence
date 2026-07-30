from pathlib import Path
import argparse
import html

import cv2
import numpy as np
import pandas as pd


DEFAULT_MODELING_CSV = Path(
    "outputs/features/particle_density_modeling_table_idd_finetuned_osm_v2.csv"
)

DEFAULT_MANIFEST_CSV = Path(
    "outputs/features/processed_frame_manifest_preprocessed_v2.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/analysis/high_traffic_visual_audit_pack"
)


def safe_float(row, col, default=np.nan):
    if col not in row:
        return default
    try:
        return float(row[col])
    except Exception:
        return default


def safe_value(row, col, default=""):
    if col not in row:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return val


def resolve_path(path_value):
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return Path.cwd() / p


def load_image_or_blank(path, width=480, height=320):
    if path is None or pd.isna(path) or str(path).strip() == "":
        img = np.ones((height, width, 3), dtype=np.uint8) * 245
        cv2.putText(
            img,
            "Missing frame",
            (40, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    path = resolve_path(path)

    if not path.exists():
        img = np.ones((height, width, 3), dtype=np.uint8) * 245
        cv2.putText(
            img,
            "File not found",
            (40, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    img = cv2.imread(str(path))

    if img is None:
        img = np.ones((height, width, 3), dtype=np.uint8) * 245
        cv2.putText(
            img,
            "Unreadable frame",
            (40, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    img = cv2.resize(img, (width, height))
    return img


def put_text_block(img, lines, x=16, y=28, line_h=24, font_scale=0.62):
    out = img.copy()

    # semi-transparent background box
    max_len = max([len(str(line)) for line in lines]) if lines else 0
    box_w = min(out.shape[1] - 20, max(300, int(max_len * 12)))
    box_h = line_h * len(lines) + 16

    overlay = out.copy()
    cv2.rectangle(
        overlay,
        (x - 8, y - 22),
        (x - 8 + box_w, y - 22 + box_h),
        (255, 255, 255),
        -1,
    )
    out = cv2.addWeighted(overlay, 0.78, out, 0.22, 0)

    for i, line in enumerate(lines):
        cv2.putText(
            out,
            str(line),
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def add_panel_title(img, title):
    h, w = img.shape[:2]
    header_h = 44

    canvas = np.ones((h + header_h, w, 3), dtype=np.uint8) * 255
    canvas[header_h:, :, :] = img

    cv2.putText(
        canvas,
        title,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return canvas


def make_sample_sheet(sample_row, manifest_sample, group_name, lenses, panel_w=480, panel_h=320):
    sample_index = int(sample_row["sample_index"])
    timestamp = safe_value(sample_row, "timestamp", "")

    pm2 = safe_float(sample_row, "value.sPM2")
    pm10 = safe_float(sample_row, "value.sPM10")

    car = safe_float(sample_row, "idd_car_count_sum")
    auto = safe_float(sample_row, "idd_auto_rickshaw_count_sum")
    moto = safe_float(sample_row, "idd_motorcycle_count_sum")
    truck = safe_float(sample_row, "idd_truck_count_sum")
    heavy = safe_float(sample_row, "idd_heavy_vehicle_count_sum")

    rh = safe_float(sample_row, "rh")
    temp = safe_float(sample_row, "temp")
    dry_air = safe_float(sample_row, "dry_air_fraction")

    road_brown = safe_float(sample_row, "road_brown_pixel_ratio_mean")
    road_gray = safe_float(sample_row, "road_gray_dry_pixel_ratio_mean")
    road_area = safe_float(sample_row, "road_area_ratio_mean")

    metric_lines = [
        f"Group: {group_name}",
        f"sample_index: {sample_index} | {timestamp}",
        f"PM2.5: {pm2:.2f} | PM10: {pm10:.2f}",
        f"cars: {car:.0f} | autos: {auto:.0f} | motorcycles: {moto:.0f}",
        f"trucks: {truck:.0f} | heavy vehicles: {heavy:.0f}",
        f"RH: {rh:.1f}% | Temp: {temp:.1f} C | dry_air: {dry_air:.3f}",
        f"road_brown: {road_brown:.3f} | road_gray: {road_gray:.3f}",
        f"road_area_ratio: {road_area:.3f}",
    ]

    panels = []

    for lens_id in lenses:
        lens_rows = manifest_sample[manifest_sample["lens_id"] == lens_id]

        if len(lens_rows) == 0:
            img = load_image_or_blank(None, width=panel_w, height=panel_h)
        else:
            frame_path = lens_rows.iloc[0]["processed_frame_path"]
            img = load_image_or_blank(frame_path, width=panel_w, height=panel_h)

        # Put metric block only on first lens panel
        if lens_id == lenses[0]:
            img = put_text_block(img, metric_lines)

        panel = add_panel_title(img, f"Lens {lens_id}")
        panels.append(panel)

    sheet = np.hstack(panels)

    footer_h = 50
    footer = np.ones((footer_h, sheet.shape[1], 3), dtype=np.uint8) * 255

    footer_text = (
        "Visual audit purpose: compare high-car/low-PM and high-car/high-PM scenes "
        "to check whether car count is confounded by vehicle mix, road condition, or scene context."
    )

    cv2.putText(
        footer,
        footer_text[:170],
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    sheet = np.vstack([sheet, footer])

    return sheet


def create_group_overview(group_dir, group_name, image_paths, output_path, thumb_w=360, thumb_h=250, cols=2):
    if not image_paths:
        return None

    thumbs = []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue

        img = cv2.resize(img, (thumb_w, thumb_h))
        title = path.stem[:55]

        canvas = np.ones((thumb_h + 36, thumb_w, 3), dtype=np.uint8) * 255
        canvas[36:, :, :] = img

        cv2.putText(
            canvas,
            title,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        thumbs.append(canvas)

    if not thumbs:
        return None

    rows = []
    for i in range(0, len(thumbs), cols):
        row_imgs = thumbs[i:i + cols]

        while len(row_imgs) < cols:
            row_imgs.append(np.ones_like(thumbs[0]) * 255)

        rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)

    header_h = 60
    header = np.ones((header_h, grid.shape[1], 3), dtype=np.uint8) * 255

    cv2.putText(
        header,
        f"Group overview: {group_name}",
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    out = np.vstack([header, grid])
    cv2.imwrite(str(output_path), out)

    return output_path


def build_groups(df, car_col, pm_col, top_n_per_group):
    car_q25 = df[car_col].quantile(0.25)
    car_q75 = df[car_col].quantile(0.75)
    car_q90 = df[car_col].quantile(0.90)

    pm_q25 = df[pm_col].quantile(0.25)
    pm_q75 = df[pm_col].quantile(0.75)

    groups = {}

    groups["high_car_low_pm2"] = (
        df[(df[car_col] >= car_q75) & (df[pm_col] <= pm_q25)]
        .sort_values([car_col, pm_col], ascending=[False, True])
        .head(top_n_per_group)
    )

    groups["high_car_high_pm2"] = (
        df[(df[car_col] >= car_q75) & (df[pm_col] >= pm_q75)]
        .sort_values([car_col, pm_col], ascending=[False, False])
        .head(top_n_per_group)
    )

    groups["low_car_high_pm2"] = (
        df[(df[car_col] <= car_q25) & (df[pm_col] >= pm_q75)]
        .sort_values([pm_col, car_col], ascending=[False, True])
        .head(top_n_per_group)
    )

    groups["top_car_overall"] = (
        df[df[car_col] >= car_q90]
        .sort_values(car_col, ascending=False)
        .head(top_n_per_group)
    )

    thresholds = {
        "car_q25": car_q25,
        "car_q75": car_q75,
        "car_q90": car_q90,
        "pm_q25": pm_q25,
        "pm_q75": pm_q75,
    }

    return groups, thresholds


def make_html_report(index_df, output_html, output_dir):
    groups = index_df["group"].dropna().unique().tolist()

    parts = []
    parts.append("<html><head><title>High Traffic Visual Audit</title>")
    parts.append("""
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif; margin: 28px; background: #fafafa; color: #111; }
    h1 { margin-bottom: 6px; }
    h2 { margin-top: 34px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
    .card { background: white; padding: 14px; margin: 18px 0; border-radius: 14px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
    img { max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }
    table { border-collapse: collapse; margin-top: 12px; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #eee; text-align: left; padding: 6px; }
    .muted { color: #666; }
    </style>
    """)
    parts.append("</head><body>")
    parts.append("<h1>High Traffic Visual Audit Report</h1>")
    parts.append("<p class='muted'>Generated to validate whether high car-count scenes correspond to low/high PM2.5 under different visual contexts.</p>")

    for group in groups:
        gdf = index_df[index_df["group"] == group].copy()
        parts.append(f"<h2>{html.escape(group)}</h2>")

        overview = gdf["group_overview_path"].dropna().astype(str).unique()
        if len(overview) > 0:
            rel = Path(overview[0]).relative_to(output_dir)
            parts.append(f"<div class='card'><img src='{html.escape(str(rel))}'></div>")

        for _, row in gdf.iterrows():
            rel = Path(row["sheet_path"]).relative_to(output_dir)

            parts.append("<div class='card'>")
            parts.append(
                f"<h3>sample {int(row['sample_index'])} | "
                f"PM2.5={row['pm2']:.2f} | cars={row['car_count']:.0f}</h3>"
            )
            parts.append(f"<img src='{html.escape(str(rel))}'>")
            parts.append("</div>")

    parts.append("</body></html>")

    output_html.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--modeling-csv", default=str(DEFAULT_MODELING_CSV))
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--pm-col", default="value.sPM2")
    parser.add_argument("--car-col", default="idd_car_count_sum")
    parser.add_argument("--require-image", action="store_true")

    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 2, 6])
    parser.add_argument("--top-n-per-group", type=int, default=20)

    args = parser.parse_args()

    modeling_csv = Path(args.modeling_csv)
    manifest_csv = Path(args.manifest_csv)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(modeling_csv)
    manifest = pd.read_csv(manifest_csv)

    if args.pm_col not in df.columns:
        raise ValueError(f"PM column not found: {args.pm_col}")

    if args.car_col not in df.columns:
        raise ValueError(f"Car column not found: {args.car_col}")

    if "sample_index" not in df.columns:
        raise ValueError("modeling CSV missing sample_index")

    if "sample_index" not in manifest.columns:
        raise ValueError("manifest CSV missing sample_index")

    if "lens_id" not in manifest.columns:
        raise ValueError("manifest CSV missing lens_id")

    if "processed_frame_path" not in manifest.columns:
        raise ValueError("manifest CSV missing processed_frame_path")

    if args.require_image and "image_features_available" in df.columns:
        before = len(df)
        df = df[df["image_features_available"] == True].copy()
        print(f"Filtered image_features_available: {before} -> {len(df)}")

    df[args.pm_col] = pd.to_numeric(df[args.pm_col], errors="coerce")
    df[args.car_col] = pd.to_numeric(df[args.car_col], errors="coerce")
    df = df[df[args.pm_col].notna() & df[args.car_col].notna()].copy()

    groups, thresholds = build_groups(
        df=df,
        car_col=args.car_col,
        pm_col=args.pm_col,
        top_n_per_group=args.top_n_per_group,
    )

    print("\nThresholds:")
    for k, v in thresholds.items():
        print(f"{k}: {v}")

    index_rows = []

    for group_name, group_df in groups.items():
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGroup: {group_name}")
        print("Rows:", len(group_df))

        image_paths = []

        for _, sample_row in group_df.iterrows():
            sample_index = int(sample_row["sample_index"])
            manifest_sample = manifest[manifest["sample_index"] == sample_index].copy()

            sheet = make_sample_sheet(
                sample_row=sample_row,
                manifest_sample=manifest_sample,
                group_name=group_name,
                lenses=args.lenses,
            )

            pm2 = safe_float(sample_row, args.pm_col)
            cars = safe_float(sample_row, args.car_col)

            timestamp = str(safe_value(sample_row, "timestamp", ""))
            timestamp_safe = (
                timestamp.replace(":", "-")
                .replace(" ", "_")
                .replace("/", "-")
            )

            sheet_name = (
                f"sample_{sample_index:04d}_"
                f"cars_{int(cars):03d}_"
                f"pm2_{pm2:.1f}_"
                f"{timestamp_safe}.jpg"
            )

            sheet_path = group_dir / sheet_name
            cv2.imwrite(str(sheet_path), sheet)
            image_paths.append(sheet_path)

            index_rows.append({
                "group": group_name,
                "sample_index": sample_index,
                "timestamp": timestamp,
                "pm2": pm2,
                "car_count": cars,
                "auto_rickshaw_count": safe_float(sample_row, "idd_auto_rickshaw_count_sum"),
                "motorcycle_count": safe_float(sample_row, "idd_motorcycle_count_sum"),
                "truck_count": safe_float(sample_row, "idd_truck_count_sum"),
                "heavy_vehicle_count": safe_float(sample_row, "idd_heavy_vehicle_count_sum"),
                "rh": safe_float(sample_row, "rh"),
                "temp": safe_float(sample_row, "temp"),
                "road_brown_pixel_ratio_mean": safe_float(sample_row, "road_brown_pixel_ratio_mean"),
                "road_gray_dry_pixel_ratio_mean": safe_float(sample_row, "road_gray_dry_pixel_ratio_mean"),
                "road_area_ratio_mean": safe_float(sample_row, "road_area_ratio_mean"),
                "sheet_path": str(sheet_path),
                "group_overview_path": "",
            })

        overview_path = group_dir / f"{group_name}_overview.jpg"
        created_overview = create_group_overview(
            group_dir=group_dir,
            group_name=group_name,
            image_paths=image_paths,
            output_path=overview_path,
            cols=2,
        )

        if created_overview is not None:
            for row in index_rows:
                if row["group"] == group_name:
                    row["group_overview_path"] = str(created_overview)

    index_df = pd.DataFrame(index_rows)

    index_csv = output_dir / "high_traffic_visual_audit_index.csv"
    index_df.to_csv(index_csv, index=False)

    html_path = output_dir / "high_traffic_visual_audit_report.html"
    make_html_report(
        index_df=index_df,
        output_html=html_path,
        output_dir=output_dir,
    )

    print("\nSaved index:", index_csv)
    print("Saved HTML report:", html_path)

    print("\nGroup counts:")
    print(index_df["group"].value_counts(dropna=False))

    show_cols = [
        "group",
        "sample_index",
        "timestamp",
        "car_count",
        "pm2",
        "auto_rickshaw_count",
        "truck_count",
        "road_brown_pixel_ratio_mean",
        "road_gray_dry_pixel_ratio_mean",
        "sheet_path",
    ]

    print("\nPreview:")
    print(index_df[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
