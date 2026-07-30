from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


CLASS_COLORS = {
    "car": (255, 0, 0),
    "auto_rickshaw": (0, 165, 255),
    "truck": (0, 0, 255),
    "bus": (128, 0, 128),
    "motorcycle": (0, 255, 255),
    "bicycle": (0, 255, 0),
    "unknown_vehicle": (180, 180, 180),
}


def safe_num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def draw_detections(img, objects):
    out = img.copy()

    for _, r in objects.iterrows():
        cls = str(r.get("pm_class_name", r.get("idd_class_name", "object")))
        conf = safe_num(r.get("confidence", np.nan))

        x1 = int(safe_num(r["x1"], 0))
        y1 = int(safe_num(r["y1"], 0))
        x2 = int(safe_num(r["x2"], 0))
        y2 = int(safe_num(r["y2"], 0))

        color = CLASS_COLORS.get(cls, (255, 255, 255))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} {conf:.2f}" if np.isfinite(conf) else cls
        y_text = max(18, y1 - 6)

        cv2.putText(
            out,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return out


def resize_keep_aspect(img, target_h=420):
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)


def make_text_panel(width, height, lines):
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255
    y = 35

    for line in lines:
        cv2.putText(
            panel,
            str(line),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 32

    return panel


def pad_to(img, h, w):
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[: img.shape[0], : img.shape[1]] = img
    return canvas


def build_audit_for_row(row, frame_df, obj_df, output_dir, category, rank):
    sample_index = int(row["sample_index"])

    frames = frame_df[frame_df["sample_index"].astype(int) == sample_index].copy()

    if frames.empty:
        print(f"[WARN] No frames for sample_index={sample_index}")
        return None

    # Prefer lenses 1, 4, 6 if present
    frames["lens_order"] = frames["lens_id"].map({1: 0, 4: 1, 6: 2}).fillna(99)
    frames = frames.sort_values(["lens_order", "lens_id"]).head(3)

    panels = []

    for _, fr in frames.iterrows():
        img_path = Path(str(fr["processed_frame_path"]))

        if not img_path.exists():
            print(f"[WARN] Missing image: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        key = str(fr["processed_frame_key"])
        lens = int(fr["lens_id"])

        objects = obj_df[
            (obj_df["sample_index"].astype(int) == sample_index)
            & (obj_df["lens_id"].astype(int) == lens)
        ].copy()

        if "processed_frame_key" in obj_df.columns:
            objects_key = obj_df[obj_df["processed_frame_key"].astype(str) == key].copy()
            if len(objects_key) > 0:
                objects = objects_key

        annotated = draw_detections(img, objects)
        annotated = resize_keep_aspect(annotated, target_h=420)

        cv2.putText(
            annotated,
            f"sample={sample_index} lens={lens}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"sample={sample_index} lens={lens}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        panels.append(annotated)

    if not panels:
        return None

    lines = [
        f"Category: {category}",
        f"Rank in category: {rank}",
        f"sample_index: {sample_index}",
        f"timestamp: {row.get('timestamp', row.get('sensor_timestamp', 'NA'))}",
        "",
        f"PM2 value.sPM2: {safe_num(row.get('value.sPM2')):.3f}",
        f"car count: {safe_num(row.get('idd_car_count_sum')):.0f}",
        f"auto-rickshaw count: {safe_num(row.get('idd_auto_rickshaw_count_sum')):.0f}",
        f"truck count: {safe_num(row.get('idd_truck_count_sum')):.0f}",
        f"bus count: {safe_num(row.get('idd_bus_count_sum')):.0f}",
        f"motorcycle count: {safe_num(row.get('idd_motorcycle_count_sum')):.0f}",
        f"total vehicle count: {safe_num(row.get('idd_total_vehicle_count_sum')):.0f}",
        "",
        "Box colors:",
        "car=blue, auto=orange, truck=red",
        "bus=purple, motorcycle=yellow",
    ]

    text_panel = make_text_panel(width=620, height=420, lines=lines)
    panels.append(text_panel)

    max_h = max(p.shape[0] for p in panels)
    max_w = max(p.shape[1] for p in panels)

    padded = [pad_to(p, max_h, max_w) for p in panels]

    # 2x2 grid
    while len(padded) < 4:
        padded.append(np.ones((max_h, max_w, 3), dtype=np.uint8) * 255)

    top = np.hstack(padded[:2])
    bottom = np.hstack(padded[2:4])
    sheet = np.vstack([top, bottom])

    cat_dir = output_dir / category.replace(" ", "_").replace("+", "plus")
    cat_dir.mkdir(parents=True, exist_ok=True)

    out_path = cat_dir / f"{rank:02d}_sample_{sample_index:05d}_{category.replace(' ', '_').replace('+', 'plus')}.jpg"
    cv2.imwrite(str(out_path), sheet)

    return out_path


def select_categories(df, n_per_category):
    work = df.copy()

    for c in [
        "value.sPM2",
        "idd_car_count_sum",
        "idd_auto_rickshaw_count_sum",
        "idd_truck_count_sum",
        "idd_total_vehicle_count_sum",
    ]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=["value.sPM2", "idd_car_count_sum"])

    # Define scores instead of hard thresholds, so we always get examples.
    work["auto_truck_count"] = (
        work.get("idd_auto_rickshaw_count_sum", 0).fillna(0)
        + work.get("idd_truck_count_sum", 0).fillna(0)
    )

    selections = {}

    # high car + low PM2 = high car rank, low PM rank
    temp = work.copy()
    temp["score"] = temp["idd_car_count_sum"].rank(pct=True) + (1 - temp["value.sPM2"].rank(pct=True))
    selections["High car + low PM2"] = temp.sort_values("score", ascending=False).head(n_per_category)

    # low car + high PM2
    temp = work.copy()
    temp["score"] = (1 - temp["idd_car_count_sum"].rank(pct=True)) + temp["value.sPM2"].rank(pct=True)
    selections["Low car + high PM2"] = temp.sort_values("score", ascending=False).head(n_per_category)

    # high car + high PM2
    temp = work.copy()
    temp["score"] = temp["idd_car_count_sum"].rank(pct=True) + temp["value.sPM2"].rank(pct=True)
    selections["High car + high PM2"] = temp.sort_values("score", ascending=False).head(n_per_category)

    # high auto/truck + high PM2
    temp = work.copy()
    temp["score"] = temp["auto_truck_count"].rank(pct=True) + temp["value.sPM2"].rank(pct=True)
    selections["High auto-truck + high PM2"] = temp.sort_values("score", ascending=False).head(n_per_category)

    return selections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling-csv", required=True)
    parser.add_argument("--frame-csv", required=True)
    parser.add_argument("--object-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-per-category", type=int, default=5)
    args = parser.parse_args()

    modeling = pd.read_csv(args.modeling_csv)
    frame_df = pd.read_csv(args.frame_csv)
    obj_df = pd.read_csv(args.object_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "sample_index" not in modeling.columns:
        raise SystemExit("modeling CSV must contain sample_index")

    selections = select_categories(modeling, args.n_per_category)

    selection_rows = []

    for category, rows in selections.items():
        print("\n" + "=" * 80)
        print(category)
        print(rows[[
            "sample_index",
            "timestamp" if "timestamp" in rows.columns else "value.sPM2",
            "value.sPM2",
            "idd_car_count_sum",
            "idd_auto_rickshaw_count_sum",
            "idd_truck_count_sum",
            "idd_total_vehicle_count_sum",
        ]].to_string(index=False))

        for rank, (_, row) in enumerate(rows.iterrows(), start=1):
            out_path = build_audit_for_row(row, frame_df, obj_df, output_dir, category, rank)

            rec = row.to_dict()
            rec["category"] = category
            rec["category_rank"] = rank
            rec["audit_image_path"] = str(out_path) if out_path else ""
            selection_rows.append(rec)

            if out_path:
                print("Saved:", out_path)

    selection_df = pd.DataFrame(selection_rows)
    selection_csv = output_dir / "selected_visual_audit_rows.csv"
    selection_df.to_csv(selection_csv, index=False)

    print("\nSaved selection table:", selection_csv)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
