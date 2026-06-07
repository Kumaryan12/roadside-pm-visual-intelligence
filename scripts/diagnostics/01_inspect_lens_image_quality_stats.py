from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd


DEFAULT_MANIFEST = Path("outputs/features/processed_frame_manifest_preprocessed_v2.csv")
DEFAULT_OUTPUT_CSV = Path("outputs/diagnostics/lens_image_quality_stats_v2.csv")
DEFAULT_SUMMARY_CSV = Path("outputs/diagnostics/lens_image_quality_summary_v2.csv")


def resolve_path(path_value: str) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return Path.cwd() / p


def safe_ratio(a, b):
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b


def compute_image_stats(image_bgr):
    h, w = image_bgr.shape[:2]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    h_ch = hsv[:, :, 0].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32) / 255.0
    v_ch = hsv[:, :, 2].astype(np.float32) / 255.0

    r_mean = float(r.mean())
    g_mean = float(g.mean())
    b_mean = float(b.mean())

    gray_world_mean = np.mean([r_mean, g_mean, b_mean])

    # Approximate gains needed to bring each channel toward neutral gray.
    wb_gain_r = safe_ratio(gray_world_mean, r_mean)
    wb_gain_g = safe_ratio(gray_world_mean, g_mean)
    wb_gain_b = safe_ratio(gray_world_mean, b_mean)

    black_ratio_v25 = float((hsv[:, :, 2] < 25).mean())
    very_dark_ratio_v50 = float((hsv[:, :, 2] < 50).mean())
    glare_ratio_v240 = float((hsv[:, :, 2] > 240).mean())

    stats = {
        "image_width": w,
        "image_height": h,
        "image_area": h * w,

        "rgb_r_mean": r_mean,
        "rgb_g_mean": g_mean,
        "rgb_b_mean": b_mean,
        "rgb_r_over_g": safe_ratio(r_mean, g_mean),
        "rgb_b_over_g": safe_ratio(b_mean, g_mean),

        "gray_world_wb_gain_r_est": wb_gain_r,
        "gray_world_wb_gain_g_est": wb_gain_g,
        "gray_world_wb_gain_b_est": wb_gain_b,

        "hsv_hue_mean": float(h_ch.mean()),
        "hsv_saturation_mean": float(s_ch.mean()),
        "hsv_value_mean": float(v_ch.mean()),
        "hsv_value_median": float(np.median(v_ch)),
        "hsv_value_p05": float(np.quantile(v_ch, 0.05)),
        "hsv_value_p95": float(np.quantile(v_ch, 0.95)),

        "gray_mean": float(gray.mean() / 255.0),
        "gray_std_contrast": float(gray.std() / 255.0),

        "black_ratio_v_lt_25": black_ratio_v25,
        "very_dark_ratio_v_lt_50": very_dark_ratio_v50,
        "glare_ratio_v_gt_240": glare_ratio_v240,
    }

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required = ["processed_frame_path", "lens_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    if "preprocess_status" in df.columns:
        df = df[df["preprocess_status"] == "success"].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    rows = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        frame_path = resolve_path(row["processed_frame_path"])
        lens_id = row["lens_id"]

        print(f"[{i}/{len(df)}] lens={lens_id} path={frame_path}")

        result = {
            "lens_id": lens_id,
            "processed_frame_key": row.get("processed_frame_key", ""),
            "sample_index": row.get("sample_index", np.nan),
            "sensor_timestamp": row.get("sensor_timestamp", row.get("timestamp", "")),
            "processed_frame_path": str(frame_path),
            "status": "failed",
            "error": "",
        }

        if not frame_path.exists():
            result["error"] = f"frame_not_found: {frame_path}"
            rows.append(result)
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            result["error"] = f"could_not_read: {frame_path}"
            rows.append(result)
            continue

        try:
            stats = compute_image_stats(image)
            result.update(stats)
            result["status"] = "success"
        except Exception as exc:
            result["error"] = str(exc)

        rows.append(result)

    out = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    success = out[out["status"] == "success"].copy()

    numeric_cols = success.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["sample_index"]]

    summary = success.groupby("lens_id")[numeric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    summary.to_csv(summary_csv, index=False)

    print("\nSaved frame stats:", output_csv)
    print("Saved summary:", summary_csv)

    print("\nStatus:")
    print(out["status"].value_counts(dropna=False))

    print("\nKey lens summary:")
    key_cols = [
        "lens_id",
        "hsv_value_mean_mean",
        "hsv_saturation_mean_mean",
        "gray_std_contrast_mean",
        "black_ratio_v_lt_25_mean",
        "glare_ratio_v_gt_240_mean",
        "rgb_r_over_g_mean",
        "rgb_b_over_g_mean",
        "gray_world_wb_gain_r_est_mean",
        "gray_world_wb_gain_b_est_mean",
    ]

    key_cols = [c for c in key_cols if c in summary.columns]
    print(summary[key_cols].to_string(index=False))


if __name__ == "__main__":
    main()