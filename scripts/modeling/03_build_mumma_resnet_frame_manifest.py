import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--sensor-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame_path = Path(args.frame_manifest)
    sensor_path = Path(args.sensor_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = pd.read_csv(frame_path)
    sensor = pd.read_csv(sensor_path)

    frames = frames.copy()
    sensor = sensor.copy()

    # The processed frame manifest already has sample_index matching the sensor-row index.
    sensor["sample_index"] = range(len(sensor))

    frames["sensor_timestamp_dt"] = pd.to_datetime(
        frames["sensor_timestamp"], dayfirst=True, errors="coerce"
    )
    sensor["timestamp_dt"] = pd.to_datetime(
        sensor["timestamp"], dayfirst=True, errors="coerce"
    )

    merged = frames.merge(
        sensor,
        on="sample_index",
        how="left",
        validate="many_to_one",
        suffixes=("", "_sensor"),
    )

    merged["timestamp_match"] = (
        pd.to_datetime(merged["sensor_timestamp"], dayfirst=True, errors="coerce")
        == pd.to_datetime(merged["timestamp"], dayfirst=True, errors="coerce")
    )

    merged["image_exists"] = merged["processed_frame_path"].apply(lambda p: Path(str(p)).exists())

    rename_map = {
        "processed_frame_path": "image_path",
        "value.lat": "lat",
        "value.long": "long",
        "temp": "temperature",
        "rh": "humidity",
        "value.sPM1": "PM1",
        "value.sPM2": "PM2.5",
        "value.sPM4": "PM4",
        "value.sPM10": "PM10",
        "value.co_ppb": "CO_ppb",
        "value.no2_ppb": "NO2_ppb",
        "value.so2_ppb": "SO2_ppb",
        "value.o3_ppb_compensated": "O3_ppb",
    }

    merged = merged.rename(columns={k: v for k, v in rename_map.items() if k in merged.columns})

    merged = merged.sort_values(["sample_index", "lens_id"]).reset_index(drop=True)
    merged["mumma_frame_row_id"] = range(len(merged))

    merged.to_csv(out_path, index=False)

    print("=" * 90)
    print("MUMMA RESNET FRAME MANIFEST")
    print("=" * 90)
    print("Frame manifest:", frame_path)
    print("Sensor CSV:", sensor_path)
    print("Output:", out_path)
    print()
    print("Frame rows:", len(frames))
    print("Sensor rows:", len(sensor))
    print("Merged rows:", len(merged))
    print()
    print("Lens counts:")
    print(merged["lens_id"].value_counts(dropna=False).to_string())
    print()
    print("Timestamp match:")
    print(merged["timestamp_match"].value_counts(dropna=False).to_string())
    print()
    print("Image exists:")
    print(merged["image_exists"].value_counts(dropna=False).to_string())
    print()
    print("Target summary:")
    for col in ["PM2.5", "PM10", "PM1", "temperature", "humidity", "CO_ppb", "NO2_ppb"]:
        if col in merged.columns:
            print(f"\n{col}")
            print(merged[col].describe().to_string())

    print()
    show_cols = [
        "mumma_frame_row_id",
        "sample_index",
        "sensor_timestamp",
        "timestamp",
        "timestamp_match",
        "lens_id",
        "image_path",
        "image_exists",
        "PM2.5",
        "PM10",
        "temperature",
        "humidity",
    ]
    show_cols = [c for c in show_cols if c in merged.columns]
    print(merged[show_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()