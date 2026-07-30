import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engineered-table", required=True)
    parser.add_argument("--resnet-features", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    engineered_path = Path(args.engineered_table)
    resnet_path = Path(args.resnet_features)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engineered = pd.read_csv(engineered_path)
    resnet = pd.read_csv(resnet_path)

    if "sample_index" not in engineered.columns:
        raise ValueError("engineered table missing sample_index")

    if "sample_index" not in resnet.columns:
        raise ValueError("resnet feature table missing sample_index")

    resnet_cols = [c for c in resnet.columns if c.startswith("resnet50_")]

    if not resnet_cols:
        raise ValueError("No resnet50_* feature columns found in ResNet feature file")

    keep_resnet_cols = ["sample_index"] + resnet_cols + ["lens_count"]
    keep_resnet_cols = [c for c in keep_resnet_cols if c in resnet.columns]

    resnet_small = resnet[keep_resnet_cols].copy()

    duplicate_engineered = engineered["sample_index"].duplicated().sum()
    duplicate_resnet = resnet_small["sample_index"].duplicated().sum()

    print("=" * 90)
    print("MUMMA ENGINEERED + RESNET50 MERGE")
    print("=" * 90)
    print("Engineered table:", engineered_path)
    print("ResNet features:", resnet_path)
    print("Output:", out_path)
    print()
    print("Engineered shape:", engineered.shape)
    print("ResNet shape:", resnet.shape)
    print("ResNet feature cols:", len(resnet_cols))
    print("Duplicate sample_index in engineered:", duplicate_engineered)
    print("Duplicate sample_index in resnet:", duplicate_resnet)

    if duplicate_engineered > 0:
        print()
        print("WARNING: engineered table has duplicate sample_index values.")
        print("This may be expected if it is frame-level, but for sensor-level modelling it should usually be unique.")

    if duplicate_resnet > 0:
        raise ValueError("ResNet feature table should be unique by sample_index.")

    merged = engineered.merge(
        resnet_small,
        on="sample_index",
        how="left",
        validate="many_to_one",
    )

    merged["has_resnet50_features"] = merged[resnet_cols].notna().all(axis=1)

    # Add clean aliases if useful.
    rename_map = {
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

    for old, new in rename_map.items():
        if old in merged.columns and new not in merged.columns:
            merged[new] = merged[old]

    # Sort chronologically by sample_index.
    merged = merged.sort_values("sample_index").reset_index(drop=True)

    merged.to_csv(out_path, index=False)

    print()
    print("Merged shape:", merged.shape)
    print("Rows with ResNet50 features:")
    print(merged["has_resnet50_features"].value_counts(dropna=False).to_string())

    print()
    print("Key target summary:")
    target_cols = [c for c in ["PM2.5", "PM10", "PM1", "temperature", "humidity", "CO_ppb", "NO2_ppb", "SO2_ppb", "O3_ppb"] if c in merged.columns]
    if target_cols:
        print(merged[target_cols].describe().T.to_string())
    else:
        print("No clean target aliases found.")

    print()
    print("Feature-family counts:")
    print("resnet50_*:", len([c for c in merged.columns if c.startswith("resnet50_")]))
    print("idd_*:", len([c for c in merged.columns if c.startswith("idd_")]))
    print("road_*:", len([c for c in merged.columns if c.startswith("road_")]))
    print("osm_*:", len([c for c in merged.columns if c.startswith("osm_")]))
    print("vehicle_*:", len([c for c in merged.columns if c.startswith("vehicle_")]))

    print()
    show_cols = [
        "sample_index",
        "timestamp",
        "PM2.5",
        "PM10",
        "temperature",
        "humidity",
        "lens_count",
        "has_resnet50_features",
    ]
    show_cols = [c for c in show_cols if c in merged.columns]
    print(merged[show_cols].head(12).to_string(index=False))

    print()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()