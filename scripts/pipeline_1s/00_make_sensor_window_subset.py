from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-sensor-csv",
        default="data/sensor/MC1S_window_115430_124150.csv",
    )

    parser.add_argument(
        "--output-sensor-csv",
        default="data/sensor/MC1S_best_10min_window.csv",
    )

    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start time in format: DD-MM-YYYY HH:MM, example: '23-02-2026 12:05'",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End time in format: DD-MM-YYYY HH:MM, example: '23-02-2026 12:15'",
    )

    args = parser.parse_args()

    input_path = Path(args.input_sensor_csv)
    output_path = Path(args.output_sensor_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input sensor CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column not found: {args.timestamp_col}")

    ts = pd.to_datetime(
        df[args.timestamp_col],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    if ts.isna().any():
        bad = df.loc[ts.isna(), args.timestamp_col].head(10).tolist()
        raise ValueError(f"Could not parse timestamps. Examples: {bad}")

    start = pd.to_datetime(args.start, format="%d-%m-%Y %H:%M")
    end = pd.to_datetime(args.end, format="%d-%m-%Y %H:%M")

    if end <= start:
        raise ValueError("End time must be after start time.")

    mask = (ts >= start) & (ts <= end)
    out = df[mask].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print("\nSaved sensor subset:")
    print(output_path)
    print("Input shape:", df.shape)
    print("Output shape:", out.shape)

    print("\nSelected time range:")
    print("Start:", args.start)
    print("End:", args.end)

    if len(out) > 0:
        print("\nSubset timestamp counts:")
        print(out[args.timestamp_col].value_counts().sort_index())

        print("\nHead:")
        print(out.head().to_string(index=False))

        print("\nTail:")
        print(out.tail().to_string(index=False))
    else:
        print("\nWARNING: No rows selected. Check your start/end timestamps.")


if __name__ == "__main__":
    main()