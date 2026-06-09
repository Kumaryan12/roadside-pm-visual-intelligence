from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_image_id(x) -> str:
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
    except Exception:
        pass
    return str(x).strip()


def find_traqid_root(raw_dir: Path) -> Path:
    if (raw_dir / "TRAQID.csv").exists():
        return raw_dir

    matches = list(raw_dir.rglob("TRAQID.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find TRAQID.csv under {raw_dir}")

    return matches[0].parent


def extract_numeric_tokens(path: Path) -> list[str]:
    return re.findall(r"\d+", path.stem)


def build_view_index(view_dir: Path, view_name: str) -> pd.DataFrame:
    files = [
        p for p in view_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    rows = []
    for p in files:
        tokens = extract_numeric_tokens(p)

        # Use the last numeric token as the most likely image id.
        # This handles filenames like img_000123.png.
        image_id = None
        if tokens:
            image_id = normalize_image_id(tokens[-1])

        rows.append(
            {
                f"{view_name}_path": str(p),
                f"{view_name}_file": p.name,
                f"{view_name}_image_id_from_name": image_id,
                f"{view_name}_size_kb": p.stat().st_size / 1024,
            }
        )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(f"No images found in {view_dir}")

    dup_ids = df[f"{view_name}_image_id_from_name"].duplicated().sum()
    if dup_ids:
        print(f"WARNING: duplicate inferred image ids in {view_name}: {dup_ids}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        default="experiments/traqid_pretraining_v1/data/raw",
    )

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/data/processed",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    traqid_root = find_traqid_root(raw_dir)
    csv_path = traqid_root / "TRAQID.csv"
    front_dir = traqid_root / "front"
    rear_dir = traqid_root / "rear"

    if not front_dir.exists():
        raise FileNotFoundError(f"Front folder not found: {front_dir}")
    if not rear_dir.exists():
        raise FileNotFoundError(f"Rear folder not found: {rear_dir}")

    labels = pd.read_csv(csv_path)
    labels = labels.copy()
    labels["row_id"] = range(len(labels))
    labels["image_id"] = labels["Image"].apply(normalize_image_id)

    front = build_view_index(front_dir, "front")
    rear = build_view_index(rear_dir, "rear")

    manifest = labels.merge(
        front,
        left_on="image_id",
        right_on="front_image_id_from_name",
        how="left",
    ).merge(
        rear,
        left_on="image_id",
        right_on="rear_image_id_from_name",
        how="left",
    )

    manifest["front_exists"] = manifest["front_path"].apply(
        lambda p: Path(str(p)).exists() if pd.notna(p) else False
    )
    manifest["rear_exists"] = manifest["rear_path"].apply(
        lambda p: Path(str(p)).exists() if pd.notna(p) else False
    )

    manifest["created_at_parsed"] = pd.to_datetime(
        manifest["created_at"],
        errors="coerce",
    )

    # Useful grouping columns for leakage-safe splits.
    manifest["date"] = manifest["created_at_parsed"].dt.date.astype(str)
    manifest["hour"] = manifest["created_at_parsed"].dt.floor("h").astype(str)

    # Basic environmental sanity flags.
    manifest["temperature_plausible"] = manifest["Temperature"].between(-10, 60)
    manifest["humidity_plausible"] = manifest["Humidity"].between(0, 100)

    # Log targets are useful because PM/AQI are right-skewed.
    for col in ["PM2.5", "PM10", "aqi"]:
        if col in manifest.columns:
            manifest[f"log1p_{col}"] = manifest[col].apply(lambda x: pd.NA if pd.isna(x) or x < 0 else __import__("math").log1p(x))

    summary = {
        "traqid_root": str(traqid_root),
        "csv_rows": int(len(labels)),
        "front_images": int(len(front)),
        "rear_images": int(len(rear)),
        "manifest_rows": int(len(manifest)),
        "front_missing": int((~manifest["front_exists"]).sum()),
        "rear_missing": int((~manifest["rear_exists"]).sum()),
        "created_at_parse_success": int(manifest["created_at_parsed"].notna().sum()),
        "unique_image_ids": int(manifest["image_id"].nunique()),
        "unique_timestamps": int(manifest["created_at_parsed"].nunique()),
        "unique_dates": int(manifest["date"].nunique()),
        "unique_hours": int(manifest["hour"].nunique()),
        "temperature_implausible_rows": int((~manifest["temperature_plausible"]).sum()),
        "humidity_implausible_rows": int((~manifest["humidity_plausible"]).sum()),
        "aqi_cat_counts": manifest["aqi_cat"].value_counts(dropna=False).to_dict()
        if "aqi_cat" in manifest.columns else {},
        "day_or_night_counts": manifest["Day_or_Night"].value_counts(dropna=False).to_dict()
        if "Day_or_Night" in manifest.columns else {},
        "season_counts": manifest["Season"].value_counts(dropna=False).to_dict()
        if "Season" in manifest.columns else {},
    }

    manifest_out = out_dir / "traqid_paired_manifest.csv"
    summary_out = report_dir / "traqid_paired_manifest_summary.json"

    manifest.to_csv(manifest_out, index=False)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 90)
    print("TRAQID PAIRED MANIFEST BUILT")
    print("=" * 90)
    print("Saved manifest:", manifest_out)
    print("Saved summary:", summary_out)
    print(json.dumps(summary, indent=2))

    print("\nPreview:")
    preview_cols = [
        "row_id",
        "created_at",
        "Image",
        "image_id",
        "front_file",
        "rear_file",
        "PM2.5",
        "PM10",
        "aqi",
        "Temperature",
        "Humidity",
        "Season",
        "Day_or_Night",
        "aqi_cat",
        "front_exists",
        "rear_exists",
        "temperature_plausible",
        "humidity_plausible",
    ]
    preview_cols = [c for c in preview_cols if c in manifest.columns]
    print(manifest[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()