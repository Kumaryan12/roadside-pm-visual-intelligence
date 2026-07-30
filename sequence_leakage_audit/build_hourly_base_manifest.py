#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def mode_or_first(s):
    s = s.dropna()
    if len(s) == 0:
        return None
    m = s.mode()
    if len(m) > 0:
        return m.iloc[0]
    return s.iloc[0]


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate repeated image rows into one timestamp/hour-level base manifest."
    )
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timestamp-col", default="created_at_parsed")
    ap.add_argument("--row-id-col", default="row_id")
    args = ap.parse_args()

    in_path = Path(args.base_manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if args.timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column not found: {args.timestamp_col}")

    if args.row_id_col not in df.columns:
        raise KeyError(f"Row ID column not found: {args.row_id_col}")

    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], errors="coerce")
    df = df.dropna(subset=[args.timestamp_col]).copy()

    df = df.sort_values([args.timestamp_col, args.row_id_col]).reset_index(drop=True)

    rows = []

    for i, (ts, g) in enumerate(df.groupby(args.timestamp_col, sort=True)):
        row = {
            "hourly_row_id": i,
            "created_at_parsed": ts,
            "date": ts.date().isoformat(),
            "hour": int(ts.hour),
            "source_row_ids": "|".join(map(str, g[args.row_id_col].tolist())),
            "num_images_at_time": int(len(g)),
            "PM2.5": float(g["PM2.5"].mean()) if "PM2.5" in g.columns else np.nan,
            "PM10": float(g["PM10"].mean()) if "PM10" in g.columns else np.nan,
            "aqi": float(g["aqi"].mean()) if "aqi" in g.columns else np.nan,
            "Temperature": float(g["Temperature"].mean()) if "Temperature" in g.columns else np.nan,
            "Humidity": float(g["Humidity"].mean()) if "Humidity" in g.columns else np.nan,
            "Season": mode_or_first(g["Season"]) if "Season" in g.columns else None,
            "Day_or_Night": mode_or_first(g["Day_or_Night"]) if "Day_or_Night" in g.columns else None,
        }

        for c in ["split_date_chrono", "date_fold_id"]:
            if c in g.columns:
                row[c] = mode_or_first(g[c])

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    print("=" * 90)
    print("HOURLY/TIMESTAMP BASE MANIFEST BUILT")
    print("=" * 90)
    print("Input:", in_path)
    print("Output:", out_path)
    print("Input rows:", len(df))
    print("Unique timestamps:", len(out_df))
    print()
    print("Columns:")
    print(list(out_df.columns))
    print()
    print("Images per timestamp summary:")
    print(out_df["num_images_at_time"].describe().to_string())
    print()
    print("Rows per date:")
    print(out_df["date"].value_counts().sort_index().to_string())
    print()
    print("Head:")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
