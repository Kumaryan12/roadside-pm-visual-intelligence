#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def pick_first_existing(cols, candidates, required=True):
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"None of these columns found: {candidates}")
    return None


def make_cyclic_hour(dt):
    hour_float = dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0
    hour_sin = np.sin(2 * np.pi * hour_float / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_float / 24.0)
    return hour_float, hour_sin, hour_cos


def main():
    ap = argparse.ArgumentParser(
        description="Build T1-history/T2-forecast sliding-window manifest from paired timestamped rows."
    )
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--history-len", type=int, default=12)
    ap.add_argument("--forecast-len", type=int, default=12)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--group-col", default=None, help="Optional column to prevent windows crossing dates/groups.")
    ap.add_argument("--timestamp-col", default=None)
    ap.add_argument("--row-id-col", default=None)
    ap.add_argument("--pm25-col", default=None)
    ap.add_argument("--pm10-col", default=None)
    ap.add_argument("--aqi-col", default=None)
    ap.add_argument("--make-random-split", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--make-twofold", action="store_true")
    ap.add_argument("--make-chrono", action="store_true")
    args = ap.parse_args()

    base_path = Path(args.base_manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_path)
    cols = set(df.columns)

    timestamp_col = args.timestamp_col or pick_first_existing(
        cols,
        ["target_created_at", "created_at", "timestamp", "datetime", "time", "date_time", "target_time"],
        required=True,
    )

    pm25_col = args.pm25_col or pick_first_existing(cols, ["PM2.5", "PM25", "pm25", "target_PM2.5"], required=True)
    pm10_col = args.pm10_col or pick_first_existing(cols, ["PM10", "pm10", "target_PM10"], required=True)
    aqi_col = args.aqi_col or pick_first_existing(cols, ["aqi", "AQI", "target_aqi"], required=False)

    if args.row_id_col and args.row_id_col in df.columns:
        row_ids = df[args.row_id_col].tolist()
        row_id_col = args.row_id_col
    elif "row_id" in df.columns:
        row_ids = df["row_id"].tolist()
        row_id_col = "row_id"
    else:
        row_ids = list(range(len(df)))
        row_id_col = "__index__"
        df[row_id_col] = row_ids

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).copy()

    df = df.sort_values(timestamp_col).reset_index(drop=True)

    if args.group_col and args.group_col in df.columns:
        group_col = args.group_col
    else:
        df["__date_group__"] = df[timestamp_col].dt.date.astype(str)
        group_col = "__date_group__"

    # Useful time features
    df["forecast_base_time"] = df[timestamp_col]
    df["date"] = df[timestamp_col].dt.date.astype(str)
    df["hour_float"], df["hour_sin"], df["hour_cos"] = make_cyclic_hour(df[timestamp_col])

    H = args.history_len
    F = args.forecast_len
    stride = args.stride

    rows = []
    seq_id = 0

    for group_value, g in df.groupby(group_col, sort=False):
        g = g.sort_values(timestamp_col).reset_index(drop=True)
        n = len(g)

        # Need H input rows + F future target rows
        max_start = n - H - F
        if max_start < 0:
            continue

        for start in range(0, max_start + 1, stride):
            input_block = g.iloc[start:start + H]
            target_block = g.iloc[start + H:start + H + F]

            input_ids = input_block[row_id_col].tolist()
            target_ids = target_block[row_id_col].tolist()

            target_pm25_values = target_block[pm25_col].astype(float).tolist()
            target_pm10_values = target_block[pm10_col].astype(float).tolist()

            if aqi_col:
                target_aqi_values = target_block[aqi_col].astype(float).tolist()
            else:
                target_aqi_values = []

            rows.append({
                "forecast_sequence_id": seq_id,
                "group": group_value,
                "date": str(group_value),
                "history_len": H,
                "forecast_len": F,
                "stride": stride,
                "input_row_ids": "|".join(map(str, input_ids)),
                "target_row_ids": "|".join(map(str, target_ids)),
                "input_start_time": input_block[timestamp_col].iloc[0],
                "input_end_time": input_block[timestamp_col].iloc[-1],
                "target_start_time": target_block[timestamp_col].iloc[0],
                "target_end_time": target_block[timestamp_col].iloc[-1],
                "target_PM2.5_mean": float(np.mean(target_pm25_values)),
                "target_PM10_mean": float(np.mean(target_pm10_values)),
                "target_PM2.5_last": float(target_pm25_values[-1]),
                "target_PM10_last": float(target_pm10_values[-1]),
                "target_PM2.5_values": "|".join(map(lambda x: f"{x:.6f}", target_pm25_values)),
                "target_PM10_values": "|".join(map(lambda x: f"{x:.6f}", target_pm10_values)),
                "hour_float": float(input_block["hour_float"].iloc[-1]),
                "hour_sin": float(input_block["hour_sin"].iloc[-1]),
                "hour_cos": float(input_block["hour_cos"].iloc[-1]),
            })

            if aqi_col:
                rows[-1].update({
                    "target_aqi_mean": float(np.mean(target_aqi_values)),
                    "target_aqi_last": float(target_aqi_values[-1]),
                    "target_aqi_values": "|".join(map(lambda x: f"{x:.6f}", target_aqi_values)),
                })

            # carry context columns from input end if available
            for c in ["Temperature", "Humidity", "Season", "Day_or_Night"]:
                if c in g.columns:
                    rows[-1][c] = input_block[c].iloc[-1]

            seq_id += 1

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError("No forecasting sequences created. Check grouping and sequence lengths.")

    # Random split over generated forecasting windows
    if args.make_random_split:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(out))
        rng.shuffle(idx)

        n = len(out)
        n_train = int(round(args.train_frac * n))
        n_val = int(round(args.val_frac * n))
        train_idx = set(idx[:n_train])
        val_idx = set(idx[n_train:n_train + n_val])
        test_idx = set(idx[n_train + n_val:])

        split = []
        for i in range(n):
            if i in train_idx:
                split.append("train")
            elif i in val_idx:
                split.append("val")
            else:
                split.append("test")
        out["split_random_forecast"] = split

    # Two-fold random split
    if args.make_twofold:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(out))
        rng.shuffle(idx)
        half = len(out) // 2
        fold1_train = set(idx[:half])
        split = ["fold1_train" if i in fold1_train else "fold1_test" for i in range(len(out))]
        out["split_twofold_forecast"] = split

    # Chronological split by target_start_time
    if args.make_chrono:
        out = out.sort_values("target_start_time").reset_index(drop=True)
        n = len(out)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        split = []
        for i in range(n):
            if i < n_train:
                split.append("train")
            elif i < n_train + n_val:
                split.append("val")
            else:
                split.append("test")
        out["split_chrono_forecast"] = split

    out.to_csv(out_path, index=False)

    print("=" * 90)
    print("FORECASTING MANIFEST BUILT")
    print("=" * 90)
    print("Base:", base_path)
    print("Output:", out_path)
    print("Shape:", out.shape)
    print("Timestamp col:", timestamp_col)
    print("Row id col:", row_id_col)
    print("PM2.5 col:", pm25_col)
    print("PM10 col:", pm10_col)
    print("AQI col:", aqi_col)
    print("Group col:", group_col)
    print("History len:", H)
    print("Forecast len:", F)
    print("Stride:", stride)

    print("\nColumns:")
    print(list(out.columns))

    for c in ["split_random_forecast", "split_twofold_forecast", "split_chrono_forecast"]:
        if c in out.columns:
            print(f"\n{c}:")
            print(out[c].value_counts())

    print("\nHead:")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
