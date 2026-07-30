from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


DAY_TO_DATE = {
    "7_24_data": "2019-07-24",
    "10_19_data": "2019-10-19",
    "11_10_data": "2019-11-10",
}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def parse_image_time_from_name(path: Path, date_str: str):
    """
    Extract image time from filename.

    Handles:
    - 152338.JPG                  -> 15:23:38
    - 2019-10-19 104440.JPG       -> 10:44:40
    - IMG_20191110_103543.jpg     -> 10:35:43

    Rule:
    Use the last 6 digits in the filename as HHMMSS.
    """
    stem = path.stem
    digits = re.sub(r"\D", "", stem)

    if len(digits) < 6:
        return pd.NaT

    hhmmss = digits[-6:]

    hh = int(hhmmss[0:2])
    mm = int(hhmmss[2:4])
    ss = int(hhmmss[4:6])

    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        return pd.NaT

    return pd.to_datetime(f"{date_str} {hh:02d}:{mm:02d}:{ss:02d}", errors="coerce")

def load_location_csv(csv_path: Path, day_folder: str, date_str: str):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["pm2.5", "pm10", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    # 10_19 and 11_10 have full datetime.
    # 7_24 has only HH:MM:SS.
    raw_time = df["time"].astype(str)

    if raw_time.str.contains(r"\d{4}-\d{2}-\d{2}", regex=True).any():
        df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(date_str + " " + raw_time, errors="coerce")

    df["day_folder"] = day_folder
    df["date"] = date_str
    df["location_file"] = csv_path.name

    m = re.search(r"location(\d+)", csv_path.stem.lower())
    df["location_id"] = int(m.group(1)) if m else -1

    # Add missing temperature/humidity as NaN for 7_24.
    if "temperature" not in df.columns:
        df["temperature"] = np.nan
    if "humidity" not in df.columns:
        df["humidity"] = np.nan

    keep = [
        "day_folder",
        "date",
        "location_file",
        "location_id",
        "timestamp",
        "pm2.5",
        "pm10",
        "temperature",
        "humidity",
    ]

    return df[keep].dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def nearest_sensor_row(sensor_df: pd.DataFrame, image_time: pd.Timestamp):
    if sensor_df.empty or pd.isna(image_time):
        return None, None

    ts = sensor_df["timestamp"].values.astype("datetime64[ns]")
    target = np.datetime64(image_time.to_datetime64())

    pos = np.searchsorted(ts, target)

    candidates = []
    if pos > 0:
        candidates.append(pos - 1)
    if pos < len(ts):
        candidates.append(pos)

    if not candidates:
        return None, None

    best_idx = min(
        candidates,
        key=lambda i: abs((sensor_df.iloc[i]["timestamp"] - image_time).total_seconds()),
    )

    row = sensor_df.iloc[best_idx]
    delta_sec = abs((row["timestamp"] - image_time).total_seconds())

    return row, delta_sec


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        default="experiments/hvaq_vgg16_lstm_v1/data/raw",
    )

    parser.add_argument(
        "--max-match-seconds",
        type=float,
        default=5.0,
        help="Maximum absolute time difference allowed between image timestamp and sensor row.",
    )

    parser.add_argument(
        "--out-manifest",
        default="experiments/hvaq_vgg16_lstm_v1/data/processed/hvaq_image_label_manifest.csv",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/hvaq_vgg16_lstm_v1/reports",
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_manifest = Path(args.out_manifest)
    report_dir = Path(args.report_dir)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    day_summaries = []

    for day_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        day_folder = day_dir.name

        if day_folder not in DAY_TO_DATE:
            print("Skipping unknown day folder:", day_folder)
            continue

        date_str = DAY_TO_DATE[day_folder]

        pictures_dir = day_dir / "pictures"
        if not pictures_dir.exists():
            print("Missing pictures dir:", pictures_dir)
            continue

        image_files = sorted([
            p for p in pictures_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        csv_files = sorted(day_dir.glob("location*.csv"))

        print("\n" + "=" * 90)
        print("Day:", day_folder, "date:", date_str)
        print("Images:", len(image_files))
        print("Location CSVs:", len(csv_files))

        sensor_by_location = {}
        for csv_path in csv_files:
            sensor_df = load_location_csv(csv_path, day_folder, date_str)
            loc_id = int(sensor_df["location_id"].iloc[0])
            sensor_by_location[loc_id] = sensor_df

        for img_path in tqdm(image_files, desc=f"Matching {day_folder}"):
            image_time = parse_image_time_from_name(img_path, date_str)

            image_meta = {
                "image_path": str(img_path),
                "image_file": img_path.name,
                "image_time": image_time,
                "day_folder": day_folder,
                "date": date_str,
            }

            try:
                with Image.open(img_path) as im:
                    image_meta["width"] = im.width
                    image_meta["height"] = im.height
            except Exception:
                image_meta["width"] = np.nan
                image_meta["height"] = np.nan

            for loc_id, sensor_df in sensor_by_location.items():
                sensor_row, delta_sec = nearest_sensor_row(sensor_df, image_time)

                if sensor_row is None:
                    continue

                if delta_sec is None or delta_sec > args.max_match_seconds:
                    continue

                row = dict(image_meta)
                row.update(
                    {
                        "location_id": loc_id,
                        "location_file": sensor_row["location_file"],
                        "sensor_time": sensor_row["timestamp"],
                        "time_delta_sec": delta_sec,
                        "PM2.5": float(sensor_row["pm2.5"]),
                        "PM10": float(sensor_row["pm10"]),
                        "Temperature": float(sensor_row["temperature"])
                        if pd.notna(sensor_row["temperature"])
                        else np.nan,
                        "Humidity": float(sensor_row["humidity"])
                        if pd.notna(sensor_row["humidity"])
                        else np.nan,
                    }
                )

                rows.append(row)

        day_summaries.append(
            {
                "day_folder": day_folder,
                "date": date_str,
                "images": len(image_files),
                "location_csvs": len(csv_files),
            }
        )

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError("No image-label matches created. Try increasing --max-match-seconds.")

    manifest = manifest.sort_values(["date", "location_id", "image_time"]).reset_index(drop=True)
    manifest["row_id"] = np.arange(len(manifest))

    manifest.to_csv(out_manifest, index=False)

    summary = {
        "raw_root": str(raw_root),
        "out_manifest": str(out_manifest),
        "rows": int(len(manifest)),
        "unique_images": int(manifest["image_path"].nunique()),
        "unique_dates": int(manifest["date"].nunique()),
        "unique_locations": int(manifest["location_id"].nunique()),
        "max_match_seconds": args.max_match_seconds,
        "day_summaries": day_summaries,
        "rows_per_date": manifest["date"].value_counts().sort_index().to_dict(),
        "rows_per_location": manifest["location_id"].value_counts().sort_index().to_dict(),
        "target_summary": manifest[["PM2.5", "PM10", "Temperature", "Humidity"]].describe().to_dict(),
        "missing_values": manifest.isna().sum().to_dict(),
    }

    summary_path = report_dir / "hvaq_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 90)
    print("HVAQ MANIFEST CREATED")
    print("=" * 90)
    print("Rows:", len(manifest))
    print("Unique images:", manifest["image_path"].nunique())
    print("Unique dates:", manifest["date"].nunique())
    print("Unique locations:", manifest["location_id"].nunique())
    print("\nRows per date:")
    print(manifest["date"].value_counts().sort_index())
    print("\nRows per location:")
    print(manifest["location_id"].value_counts().sort_index())
    print("\nTarget summary:")
    print(manifest[["PM2.5", "PM10", "Temperature", "Humidity"]].describe())
    print("\nMissing:")
    print(manifest.isna().sum())
    print("\nSaved:")
    print(" -", out_manifest)
    print(" -", summary_path)


if __name__ == "__main__":
    main()