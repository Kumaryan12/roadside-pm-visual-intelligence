import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["PM2.5", "PM10", "aqi"]


TABULAR_COLS = [
    "Temperature",
    "Humidity",
    "Season",
    "Day_or_Night",
]


ID_COLS = [
    "row_id",
    "image_id",
    "created_at",
    "created_at_parsed",
    "split_date_chrono",
    "aqi_cat",
]


def one_hot_tabular(df):
    df = df.copy()

    for c in ["Season", "Day_or_Night", "aqi_cat"]:
        if c in df.columns:
            dummies = pd.get_dummies(df[c], prefix=c, dummy_na=False)
            df = pd.concat([df.drop(columns=[c]), dummies], axis=1)

    return df


def numeric_feature_cols(df):
    blocked = set(TARGETS + ID_COLS)
    blocked.update([
        "front_path",
        "rear_path",
        "front_file",
        "rear_file",
        "created_at",
        "date",
        "hour",
        "Image",
        "image_id",
        "row_id",
        "split_date_chrono",
        "created_at_parsed",
    ])

    cols = []

    for c in df.columns:
        if c in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool:
            cols.append(c)

    return sorted(set(cols))


def build_t_sequences(df, feature_cols, t_steps):
    df = df.sort_values(["created_at_parsed", "row_id"]).reset_index(drop=True)

    rows = []

    values = df[feature_cols].to_numpy(dtype=float)
    targets = df[TARGETS].to_numpy(dtype=float)

    for end_idx in range(t_steps - 1, len(df)):
        start_idx = end_idx - t_steps + 1

        block = df.iloc[start_idx:end_idx + 1]

        # Do not allow sequence to cross date-fold boundary if present.
        if "date_fold_id" in block.columns:
            if block["date_fold_id"].nunique() > 1:
                continue

        # Avoid mixing across very large time jumps.
        times = pd.to_datetime(block["created_at_parsed"], errors="coerce")
        if times.isna().any():
            continue

        max_gap_min = times.diff().dt.total_seconds().dropna().max() / 60.0
        if pd.notna(max_gap_min) and max_gap_min > 180:
            continue

        seq_values = values[start_idx:end_idx + 1].reshape(-1)

        out = {
            "sequence_id": len(rows),
            "target_row_id": int(df.loc[end_idx, "row_id"]),
            "target_image_id": int(df.loc[end_idx, "image_id"]) if "image_id" in df.columns else np.nan,
            "target_created_at": df.loc[end_idx, "created_at"],
            "target_created_at_parsed": df.loc[end_idx, "created_at_parsed"],
            "target_split_date_chrono": df.loc[end_idx, "split_date_chrono"] if "split_date_chrono" in df.columns else np.nan,
            "seq_start_row_id": int(df.loc[start_idx, "row_id"]),
            "seq_end_row_id": int(df.loc[end_idx, "row_id"]),
            "seq_row_ids": "|".join(str(int(x)) for x in block["row_id"].values),
            "T": t_steps,
        }

        for j, target in enumerate(TARGETS):
            out[target] = float(targets[end_idx, j])

        for step in range(t_steps):
            lag = t_steps - 1 - step
            for k, col in enumerate(feature_cols):
                out[f"lag{lag}__{col}"] = seq_values[step * len(feature_cols) + k]

        rows.append(out)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )
    parser.add_argument(
        "--yolo",
        default="experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_advanced_features_front.csv",
    )
    parser.add_argument(
        "--road",
        default="experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_road_features_front.csv",
    )
    parser.add_argument(
        "--out",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_engineered_T7_sequence_table.csv",
    )
    parser.add_argument("--t-steps", type=int, default=7)
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    yolo = pd.read_csv(args.yolo)
    road = pd.read_csv(args.road)

    base["created_at_parsed"] = pd.to_datetime(base["created_at"], errors="coerce")
    yolo["created_at_parsed"] = pd.to_datetime(yolo["created_at"], errors="coerce")
    road["created_at_parsed"] = pd.to_datetime(road["sensor_timestamp"], errors="coerce")

    # Road file uses sample_index equivalent to row_id.
    road = road.rename(columns={"sample_index": "row_id"})

    # Keep useful road columns only.
    road_keep = [
        "row_id",
        "road_area_pixels",
        "road_area_ratio",
        "road_mean_brightness",
        "road_mean_saturation",
        "road_contrast_std",
        "road_shadow_ratio",
        "road_glare_ratio",
        "road_brown_pixel_ratio",
        "road_gray_dry_pixel_ratio",
        "road_edge_density",
        "road_laplacian_std",
        "road_haze_flatness_proxy",
    ]
    road_keep = [c for c in road_keep if c in road.columns]
    road = road[road_keep].copy()

    # Start from YOLO because it already has targets + tabular + advanced YOLO.
    df = yolo.merge(road, on="row_id", how="left", validate="one_to_one")

    # Bring AQI and fold metadata from base.
    base_keep = [
        "row_id",
        "aqi",
        "date_fold_id",
        "split_date_chrono",
        "created_at_parsed",
    ]
    base_keep = [c for c in base_keep if c in base.columns]

    df = df.merge(
        base[base_keep],
        on="row_id",
        how="left",
        suffixes=("", "_base"),
        validate="one_to_one",
    )

    if "created_at_parsed_base" in df.columns:
        df["created_at_parsed"] = df["created_at_parsed_base"].combine_first(df["created_at_parsed"])
        df = df.drop(columns=["created_at_parsed_base"])

    if "split_date_chrono_base" in df.columns:
        df["split_date_chrono"] = df["split_date_chrono"].combine_first(df["split_date_chrono_base"])
        df = df.drop(columns=["split_date_chrono_base"])

    df = df.dropna(subset=TARGETS + ["created_at_parsed"]).copy()
    df = one_hot_tabular(df)

    feature_cols = numeric_feature_cols(df)

    # Remove target leakage columns explicitly.
    leakage_terms = [
        "PM2.5",
        "PM10",
        "aqi",
        "log1p_PM2.5",
        "log1p_PM10",
        "log1p_aqi",
        "target",
        "y_true",
        "y_pred",
    ]
    feature_cols = [
        c for c in feature_cols
        if not any(term.lower() in c.lower() for term in leakage_terms)
    ]

    seq = build_t_sequences(df, feature_cols, args.t_steps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seq.to_csv(out, index=False)

    summary_path = out.with_suffix(".summary.txt")
    summary_path.write_text(
        "\n".join([
            "TRAQID T-sequence engineered table",
            f"Base rows: {len(base)}",
            f"YOLO rows: {len(yolo)}",
            f"Road rows: {len(road)}",
            f"Merged rows: {len(df)}",
            f"T steps: {args.t_steps}",
            f"Feature columns per step: {len(feature_cols)}",
            f"Output sequence rows: {len(seq)}",
            f"Output columns: {len(seq.columns)}",
            "",
            "Feature columns:",
            "\n".join(feature_cols),
        ]),
        encoding="utf-8",
    )

    print("=" * 90)
    print("TRAQID ENGINEERED T-SEQUENCE TABLE BUILT")
    print("=" * 90)
    print("Merged rows:", len(df))
    print("T steps:", args.t_steps)
    print("Feature columns per step:", len(feature_cols))
    print("Sequence rows:", len(seq))
    print("Output:", out)
    print("Summary:", summary_path)
    print()
    print(seq.head(3).to_string(index=False))


if __name__ == "__main__":
    main()