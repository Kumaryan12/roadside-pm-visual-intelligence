from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

manifest_in = Path("experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv")
manifest_out = Path("experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_random_debug_split.csv")

df = pd.read_csv(manifest_in)

# Keep paired observation as unit: one row = front+rear pair.
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True,
    stratify=df["aqi_cat"] if "aqi_cat" in df.columns else None,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True,
    stratify=temp_df["aqi_cat"] if "aqi_cat" in temp_df.columns else None,
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split_date_chrono"] = "train"
val_df["split_date_chrono"] = "val"
test_df["split_date_chrono"] = "test"

out = pd.concat([train_df, val_df, test_df], ignore_index=True)
out.to_csv(manifest_out, index=False)

print("Saved:", manifest_out)
print(out["split_date_chrono"].value_counts())
print("\nDate distribution by split:")
print(pd.crosstab(out["date"], out["split_date_chrono"]))