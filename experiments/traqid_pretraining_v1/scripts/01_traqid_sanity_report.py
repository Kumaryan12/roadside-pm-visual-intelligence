from pathlib import Path
import pandas as pd

MANIFEST = Path("experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv")
OUTDIR = Path("experiments/traqid_pretraining_v1/reports/pipeline_validation")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

print("Shape:", df.shape)
print("\nSplits:")
print(df["split_date_chrono"].value_counts(dropna=False))

print("\nAQI categories:")
print(df["aqi_cat"].value_counts(dropna=False))

print("\nDay/Night:")
print(df["Day_or_Night"].value_counts(dropna=False))

print("\nSeason:")
print(df["Season"].value_counts(dropna=False))

cols = ["PM2.5", "PM10", "aqi", "Temperature", "Humidity"]
print("\nNumeric summary:")
print(df[cols].describe().T)

print("\nImage existence:")
print("front_exists:", df["front_exists"].mean())
print("rear_exists:", df["rear_exists"].mean())

report_path = OUTDIR / "traqid_sanity_summary.csv"
df[cols + ["split_date_chrono", "Season", "Day_or_Night", "aqi_cat"]].to_csv(report_path, index=False)
print("\nSaved:", report_path)