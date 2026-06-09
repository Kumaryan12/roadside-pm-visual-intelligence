from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

inp = Path("experiments/traqid_pretraining_v1/data/processed/traqid_T7_front_sequence_manifest.csv")
out = Path("experiments/traqid_pretraining_v1/data/processed/traqid_T7_front_sequence_manifest_random_debug_split.csv")

df = pd.read_csv(inp)

stratify_col = "aqi_cat" if "aqi_cat" in df.columns else None

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True,
    stratify=df[stratify_col] if stratify_col else None,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True,
    stratify=temp_df[stratify_col] if stratify_col else None,
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split_date_chrono"] = "train"
val_df["split_date_chrono"] = "val"
test_df["split_date_chrono"] = "test"

final = pd.concat([train_df, val_df, test_df], ignore_index=True)
final.to_csv(out, index=False)

print("Saved:", out)
print(final["split_date_chrono"].value_counts())

print("\nDate × split table:")
print(pd.crosstab(final["date"], final["split_date_chrono"]))