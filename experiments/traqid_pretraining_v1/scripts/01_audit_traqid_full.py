from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_traqid_root(raw_dir: Path) -> Path:
    """
    Finds the folder containing TRAQID.csv.
    Works for:
      data/raw/TRAQID.csv
      data/raw/traqid/TRAQID.csv
    """
    if (raw_dir / "TRAQID.csv").exists():
        return raw_dir

    candidates = []
    for path in raw_dir.rglob("TRAQID.csv"):
        candidates.append(path.parent)

    if not candidates:
        raise FileNotFoundError(f"Could not find TRAQID.csv inside: {raw_dir}")

    return candidates[0]


def find_image_roots(traqid_root: Path) -> list[Path]:
    """
    Handles both possible layouts:
      traqid_root/Images/...
      traqid_root/front + traqid_root/rear
    """
    roots = []

    images_dir = traqid_root / "Images"
    if images_dir.exists():
        roots.append(images_dir)

    front_dir = traqid_root / "front"
    rear_dir = traqid_root / "rear"

    if front_dir.exists():
        roots.append(front_dir)

    if rear_dir.exists():
        roots.append(rear_dir)

    if not roots:
        # Fallback: scan whole TRAQID root.
        roots.append(traqid_root)

    return roots


def infer_view(path: Path) -> str:
    joined = " ".join(path.parts).lower()
    name = path.name.lower()

    if "front" in joined or "front" in name:
        return "front"

    if "rear" in joined or "rear" in name or "back" in joined or "back" in name:
        return "rear"

    return "unknown"


def extract_numeric_tokens(path: Path) -> list[str]:
    return re.findall(r"\d+", path.stem)


def build_image_index(image_roots: list[Path]) -> pd.DataFrame:
    all_files = []

    for root in image_roots:
        all_files.extend(
            [
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

    all_files = sorted(set(all_files))

    rows = []

    for path in all_files:
        tokens = extract_numeric_tokens(path)

        rows.append(
            {
                "image_path": str(path),
                "file_name": path.name,
                "stem": path.stem,
                "extension": path.suffix.lower(),
                "parent": str(path.parent),
                "view": infer_view(path),
                "numeric_tokens": "|".join(tokens),
                "size_kb": path.stat().st_size / 1024,
            }
        )

    return pd.DataFrame(rows)


def normalize_image_id(x) -> str:
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
    except Exception:
        pass

    return str(x).strip()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        default="experiments/traqid_pretraining_v1/data/raw",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    traqid_root = find_traqid_root(raw_dir)
    csv_path = traqid_root / "TRAQID.csv"
    image_roots = find_image_roots(traqid_root)

    df = pd.read_csv(csv_path)
    image_index = build_image_index(image_roots)

    print("=" * 90)
    print("TRAQID FULL AUDIT")
    print("=" * 90)
    print("Raw dir:", raw_dir)
    print("TRAQID root:", traqid_root)
    print("CSV path:", csv_path)
    print("Image roots:")
    for root in image_roots:
        print(" -", root)

    print("\nCSV shape:", df.shape)

    print("\nColumns:")
    for col in df.columns:
        print(" -", col)

    print("\nImage files found:", len(image_index))

    if len(image_index):
        print("\nImage view counts:")
        print(image_index["view"].value_counts(dropna=False))

        print("\nImage extension counts:")
        print(image_index["extension"].value_counts(dropna=False))

        print("\nImage size KB summary:")
        print(image_index["size_kb"].describe().to_string())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    target_cols = [
        c for c in ["PM2.5", "PM10", "aqi", "Temperature", "Humidity"]
        if c in df.columns
    ]

    if target_cols:
        print("\nTarget/weather numeric summary:")
        print(df[target_cols].describe().T.to_string())

    for col in ["Season", "Day_or_Night", "aqi_cat", "Sequence"]:
        if col in df.columns:
            print(f"\n{col} counts:")
            print(df[col].value_counts(dropna=False).to_string())

    # Basic image-id matching check.
    match_summary = {}

    if "Image" in df.columns and len(image_index):
        df_ids = set(df["Image"].apply(normalize_image_id).astype(str))

        image_index["token_match_found"] = image_index["numeric_tokens"].apply(
            lambda s: any(
                normalize_image_id(tok) in df_ids
                for tok in str(s).split("|")
                if tok.strip()
            )
        )

        match_summary = {
            "csv_unique_image_ids": int(len(df_ids)),
            "image_files_with_numeric_token_match": int(image_index["token_match_found"].sum()),
            "image_files_without_numeric_token_match": int((~image_index["token_match_found"]).sum()),
        }

        print("\nImage-token-to-CSV matching check:")
        print(json.dumps(match_summary, indent=2))

    summary = {
        "raw_dir": str(raw_dir),
        "traqid_root": str(traqid_root),
        "csv_path": str(csv_path),
        "image_roots": [str(p) for p in image_roots],
        "csv_shape": list(df.shape),
        "columns": list(df.columns),
        "image_file_count": int(len(image_index)),
        "image_view_counts": image_index["view"].value_counts(dropna=False).to_dict()
        if len(image_index)
        else {},
        "image_extension_counts": image_index["extension"].value_counts(dropna=False).to_dict()
        if len(image_index)
        else {},
        "missing_values": df.isna().sum().sort_values(ascending=False).to_dict(),
        "match_summary": match_summary,
    }

    if target_cols:
        target_summary = df[target_cols].describe().T
        target_summary.to_csv(report_dir / "traqid_target_weather_summary.csv")

    image_index.to_csv(report_dir / "traqid_full_image_index.csv", index=False)
    df.head(100).to_csv(report_dir / "traqid_full_csv_head100.csv", index=False)

    with open(report_dir / "traqid_full_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(" -", report_dir / "traqid_full_image_index.csv")
    print(" -", report_dir / "traqid_full_csv_head100.csv")
    print(" -", report_dir / "traqid_full_audit_summary.json")
    if target_cols:
        print(" -", report_dir / "traqid_target_weather_summary.csv")


if __name__ == "__main__":
    main()