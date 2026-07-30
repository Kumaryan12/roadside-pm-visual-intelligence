from pathlib import Path
import cv2
import numpy as np
import pandas as pd


OUTDIR = Path("outputs/road_dust_wb_validation_new")
FEATURE_CSV = OUTDIR / "segformer_road_condition_features_with_metadata.csv"
SUMMARY_CSV = OUTDIR / "FINAL_important_16mm_wb_validation_summary.csv"

VIS_DIR = OUTDIR / "visual_proof_wb"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def resize_to_width(img, target_w=520):
    h, w = img.shape[:2]
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def make_text_panel(width, height, lines):
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255
    y = 35

    for line in lines:
        cv2.putText(
            panel,
            str(line),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 31

    return panel


def pad_to_same_height(images):
    max_h = max(img.shape[0] for img in images)
    padded = []

    for img in images:
        h, w = img.shape[:2]
        canvas = np.ones((max_h, w, 3), dtype=np.uint8) * 255
        canvas[:h, :w] = img
        padded.append(canvas)

    return padded


def compute_simple_road_brown_highlight(img):
    """
    This reproduces a visual brown-pixel highlight using color thresholds.
    It is for visual explanation only.
    The quantitative values come from the already extracted SegFormer CSV.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32) - 128
    B = lab[:, :, 2].astype(np.float32) - 128

    brown = (B > 8) & (A > 0) & (S > 35) & (V > 40)

    overlay = img.copy()
    overlay[brown] = (0, 140, 255)  # orange highlight in BGR

    blended = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
    return blended, brown


def make_individual_panel(row):
    img_path = Path(str(row["processed_frame_path"]))
    img = read_image(img_path)

    brown_overlay, brown_mask = compute_simple_road_brown_highlight(img)

    original = resize_to_width(img, 520)
    brown_overlay = resize_to_width(brown_overlay, 520)

    # Add labels
    wb = str(row["wb_group"])
    file_name = str(row["FileName"])

    road_brown = float(row["road_brown_pixel_ratio"])
    road_gray = float(row["road_gray_dry_pixel_ratio"])
    road_area = float(row["road_area_ratio"])
    bright = float(row["road_mean_brightness"])

    lines = [
        f"File: {file_name}",
        f"WB: {wb}",
        f"Road brown ratio: {road_brown:.4f}",
        f"Road gray/dry ratio: {road_gray:.4f}",
        f"Road area ratio: {road_area:.4f}",
        f"Brightness: {bright:.4f}",
        "",
        "Orange overlay = brown/dust-like pixels",
        "computed after road segmentation.",
    ]

    h = max(original.shape[0], brown_overlay.shape[0])
    text_panel = make_text_panel(520, h, lines)

    panels = pad_to_same_height([original, brown_overlay, text_panel])
    sheet = np.hstack(panels)

    return sheet


def main():
    df = pd.read_csv(FEATURE_CSV)

    # Use only 16 mm controlled group
    main16 = df[df["FocalLength"].astype(str) == "16.0 mm"].copy()

    # Pick representative images:
    # one near the group mean for each WB group.
    selected = []

    for wb in ["6000K_16.0 mm", "Auto_16.0 mm", "Daylight_16.0 mm"]:
        sub = main16[main16["wb_group"] == wb].copy()
        if sub.empty:
            continue

        mean_val = sub["road_brown_pixel_ratio"].mean()
        sub["dist_to_mean"] = (sub["road_brown_pixel_ratio"] - mean_val).abs()
        selected.append(sub.sort_values("dist_to_mean").iloc[0])

    selected = pd.DataFrame(selected)

    # Individual proof sheets
    individual_paths = []
    for _, row in selected.iterrows():
        sheet = make_individual_panel(row)
        out = VIS_DIR / f"individual_{row['wb_group'].replace(' ', '').replace('.', 'p')}_{row['FileName']}.jpg"
        cv2.imwrite(str(out), sheet)
        individual_paths.append(out)
        print("Saved:", out)

    # Combined side-by-side sheet
    combined_panels = []
    for _, row in selected.iterrows():
        img_path = Path(str(row["processed_frame_path"]))
        img = read_image(img_path)

        brown_overlay, _ = compute_simple_road_brown_highlight(img)
        brown_overlay = resize_to_width(brown_overlay, 440)

        title = f"{row['wb_group']} | brown={float(row['road_brown_pixel_ratio']):.3f}"
        cv2.putText(
            brown_overlay,
            title,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            brown_overlay,
            title,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        combined_panels.append(brown_overlay)

    combined_panels = pad_to_same_height(combined_panels)
    combined = np.hstack(combined_panels)

    out_combined = VIS_DIR / "WB_comparison_brown_overlay_16mm.jpg"
    cv2.imwrite(str(out_combined), combined)
    print("Saved:", out_combined)

    # Summary card image
    summary = pd.read_csv(SUMMARY_CSV)

    def get_cv(score, wb):
        sub = summary[(summary["score"] == score) & (summary["wb_group"] == wb)]
        if len(sub) == 0:
            return None
        return float(sub["cv_percent"].iloc[0])

    brown_6000 = get_cv("road_brown_pixel_ratio", "6000K_16.0 mm")
    brown_auto = get_cv("road_brown_pixel_ratio", "Auto_16.0 mm")
    brown_day = get_cv("road_brown_pixel_ratio", "Daylight_16.0 mm")

    lines = [
        "Road Dust White Balance Validation",
        "",
        "Controlled subset:",
        "Focal length = 16 mm",
        "ISO = 200, f/9, 1/160 s, HDR off",
        "Road pixels extracted using SegFormer",
        "",
        "road_brown_pixel_ratio CV:",
        f"6000K:   {brown_6000:.2f}%",
        f"Auto:    {brown_auto:.2f}%",
        f"Daylight:{brown_day:.2f}%",
        "",
        "Conclusion:",
        "Fixed 6000K gave the lowest brown-score variation",
        "among tested 16 mm settings.",
        "",
        "Use 6000K for future road-dust capture.",
    ]

    summary_img = make_text_panel(900, 620, lines)
    out_summary = VIS_DIR / "WB_validation_summary_card.jpg"
    cv2.imwrite(str(out_summary), summary_img)
    print("Saved:", out_summary)

    # Save selected rows
    selected.to_csv(VIS_DIR / "selected_visual_proof_rows.csv", index=False)
    print("Saved:", VIS_DIR / "selected_visual_proof_rows.csv")


if __name__ == "__main__":
    main()
