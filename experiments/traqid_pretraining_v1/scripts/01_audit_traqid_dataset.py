from pathlib import Path
import argparse
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_traqid_root(raw_dir: Path) -> Path:
    candidates = []

    for path in raw_dir.rglob("*"):
        if path.is_dir() and (path / "TRAQID.csv").exists():
            candidates.append(path)

    if (raw_dir / "TRAQID.csv").exists():
        candidates.append(raw_dir)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find TRAQID.csv inside {raw_dir}. "
            "Extract the downloaded TRAQID ZIP inside data/raw first."
        )

    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default="experiments/traqid_pretraining_v1/data/raw",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/reports",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traqid_root = find_traqid_root(raw_dir)
    csv_path = traqid_root / "TRAQID.csv"
    images_dir = traqid_root / "Images"

    print("=" * 90)
    print("TRAQID DATASET AUDIT")
    print("=" * 90)
    print("TRAQID root:", traqid_root)
    print("CSV path:", csv_path)
    print("Images dir exists:", images_dir.exists())

    df = pd.read_csv(csv_path)

    print("\nCSV shape:", df.shape)
    print("\nColumns:")
    for col in df.columns:
        print(" -", col)

    image_files = []
    if images_dir.exists():
        image_files = [
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    print("\nImage files found:", len(image_files))

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).head(30))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        print("\nNumeric summary:")
        print(df[numeric_cols].describe().T.to_string())

    possible_targets = [
        "PM2.5", "PM25", "PM_2_5", "pm2.5", "pm25",
        "PM10", "pm10",
        "AQI", "aqi",
        "Temperature", "temperature", "Temp", "temp",
        "Humidity", "humidity", "RH", "rh",
    ]

    matched_targets = [c for c in possible_targets if c in df.columns]

    print("\nMatched likely target/weather columns:")
    print(matched_targets)

    report_path = out_dir / "traqid_audit_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRAQID DATASET AUDIT\n")
        f.write("=" * 90 + "\n")
        f.write(f"TRAQID root: {traqid_root}\n")
        f.write(f"CSV path: {csv_path}\n")
        f.write(f"CSV shape: {df.shape}\n")
        f.write(f"Image files found: {len(image_files)}\n\n")

        f.write("Columns:\n")
        for col in df.columns:
            f.write(f" - {col}\n")

        f.write("\nMissing values:\n")
        f.write(df.isna().sum().sort_values(ascending=False).head(50).to_string())

        if numeric_cols:
            f.write("\n\nNumeric summary:\n")
            f.write(df[numeric_cols].describe().T.to_string())

        f.write("\n\nMatched likely target/weather columns:\n")
        f.write(str(matched_targets))

    df.head(50).to_csv(out_dir / "traqid_csv_head50.csv", index=False)

    image_index_path = out_dir / "traqid_image_file_index.csv"
    image_index = pd.DataFrame(
        {
            "image_path": [str(p) for p in image_files],
            "file_name": [p.name for p in image_files],
            "parent": [str(p.parent) for p in image_files],
            "view_guess": [
                "Front" if "Front" in p.parts else "Rear" if "Rear" in p.parts else ""
                for p in image_files
            ],
        }
    )
    image_index.to_csv(image_index_path, index=False)

    print("\nSaved:")
    print(" -", report_path)
    print(" -", out_dir / "traqid_csv_head50.csv")
    print(" -", image_index_path)


if __name__ == "__main__":
    main()