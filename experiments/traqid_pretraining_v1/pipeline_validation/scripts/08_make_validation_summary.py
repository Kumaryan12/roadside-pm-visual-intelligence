from pathlib import Path
import pandas as pd

REPORTS = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUT = REPORTS / "traqid_pipeline_validation_summary.csv"

files = [
    ("tabular_chrono", REPORTS / "tabular_baseline_results.csv"),
    ("tabular_random", REPORTS / "random_split_tabular_baseline_results.csv"),
    ("embedding", REPORTS / "embedding_baseline_results.csv"),
    ("vehicle", REPORTS / "vehicle_feature_baseline_results.csv"),
    ("full_fusion", REPORTS / "full_fusion_baseline_results.csv"),
]

dfs = []

for name, path in files:
    if not path.exists():
        print("Missing:", path)
        continue

    df = pd.read_csv(path)
    df["source_result_file"] = name
    dfs.append(df)

all_results = pd.concat(dfs, ignore_index=True, sort=False)

keep_cols = [
    "source_result_file",
    "split",
    "model",
    "features",
    "embedding",
    "target",
    "n_train",
    "n_test",
    "MAE",
    "RMSE",
    "R2",
    "Spearman",
]

keep_cols = [c for c in keep_cols if c in all_results.columns]
summary = all_results[keep_cols].copy()

summary = summary.sort_values(
    by=["split", "RMSE"],
    ascending=[True, True],
    na_position="last",
)

summary.to_csv(OUT, index=False)

print("Saved:", OUT)
print(summary.to_string(index=False))