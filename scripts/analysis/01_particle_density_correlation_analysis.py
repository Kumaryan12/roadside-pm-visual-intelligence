from pathlib import Path
import argparse
import pandas as pd


DEFAULT_INPUT_CSV = Path("outputs/features/particle_density_modeling_table_v2.csv")
DEFAULT_OUTPUT_CSV = Path("outputs/features/particle_density_feature_correlations_v2.csv")


TARGET_COLS = [
    "value.sPM1",
    "value.sPM2",
    "value.sPM4",
    "value.sPM10",
    "value.sNPM1",
    "nPM2",
    "value.sNPM4",
    "value.sNPM10",
]


FEATURE_PREFIXES = [
    "idd_",
    "road_",
    "vehicle_resuspension_x_",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if "image_features_available" in df.columns:
        df = df[df["image_features_available"] == True].copy()

    target_cols = [c for c in TARGET_COLS if c in df.columns]

    feature_cols = [
        c for c in df.columns
        if any(c.startswith(prefix) for prefix in FEATURE_PREFIXES)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    feature_cols = [
        c for c in feature_cols
        if "status" not in c.lower()
        and "error" not in c.lower()
        and not c.endswith("_available")
    ]

    rows = []

    for feature in feature_cols:
        for target in target_cols:
            valid = df[[feature, target]].dropna()

            if len(valid) < 5:
                corr = None
            else:
                corr = valid[feature].corr(valid[target], method="spearman")

            rows.append({
                "feature": feature,
                "target": target,
                "spearman_corr": corr,
                "abs_spearman_corr": abs(corr) if pd.notna(corr) else None,
                "n": len(valid),
            })

    out = pd.DataFrame(rows)
    out = out.sort_values("abs_spearman_corr", ascending=False)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Rows:", len(out))

    print("\nTop correlations:")
    print(out.head(40).to_string(index=False))


if __name__ == "__main__":
    main()