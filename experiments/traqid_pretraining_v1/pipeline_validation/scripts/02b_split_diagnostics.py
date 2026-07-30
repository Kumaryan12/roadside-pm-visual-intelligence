from pathlib import Path
import pandas as pd

MANIFEST = Path("experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv")
OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

df = df[
    (df["front_exists"] == True)
    & (df["rear_exists"] == True)
    & (df["env_plausible"] == True)
].copy()

df["created_at_parsed"] = pd.to_datetime(df["created_at_parsed"])
df["month"] = df["created_at_parsed"].dt.to_period("M").astype(str)
df["date"] = df["created_at_parsed"].dt.date.astype(str)
df["hour_num"] = df["created_at_parsed"].dt.hour

num_cols = ["PM2.5", "PM10", "aqi", "Temperature", "Humidity"]

print("\nRows by split:")
print(df["split_date_chrono"].value_counts())

print("\nTarget summary by split:")
print(df.groupby("split_date_chrono")[num_cols].describe().to_string())

print("\nPM2.5 quantiles by split:")
print(
    df.groupby("split_date_chrono")["PM2.5"]
    .quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    .unstack()
    .to_string()
)

print("\nAQI category by split:")
print(pd.crosstab(df["split_date_chrono"], df["aqi_cat"], normalize="index").round(3).to_string())

print("\nSeason by split:")
print(pd.crosstab(df["split_date_chrono"], df["Season"], normalize="index").round(3).to_string())

print("\nDay/Night by split:")
print(pd.crosstab(df["split_date_chrono"], df["Day_or_Night"], normalize="index").round(3).to_string())

print("\nMonthly PM2.5 summary:")
monthly = (
    df.groupby(["month", "split_date_chrono"])
    .agg(
        n=("PM2.5", "size"),
        pm25_mean=("PM2.5", "mean"),
        pm25_median=("PM2.5", "median"),
        pm25_std=("PM2.5", "std"),
        temp_mean=("Temperature", "mean"),
        humidity_mean=("Humidity", "mean"),
    )
    .reset_index()
)

print(monthly.to_string(index=False))

out = OUTDIR / "split_diagnostics_monthly_summary.csv"
monthly.to_csv(out, index=False)

split_summary = (
    df.groupby("split_date_chrono")
    .agg(
        n=("PM2.5", "size"),
        pm25_mean=("PM2.5", "mean"),
        pm25_median=("PM2.5", "median"),
        pm25_std=("PM2.5", "std"),
        pm25_min=("PM2.5", "min"),
        pm25_max=("PM2.5", "max"),
        temp_mean=("Temperature", "mean"),
        humidity_mean=("Humidity", "mean"),
    )
    .reset_index()
)

split_summary.to_csv(OUTDIR / "split_diagnostics_split_summary.csv", index=False)

print("\nSaved:")
print(out)
print(OUTDIR / "split_diagnostics_split_summary.csv")