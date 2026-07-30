from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

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
df["hour_num"] = df["created_at_parsed"].dt.hour

target = "PM2.5"

numeric_features = ["Temperature", "Humidity", "hour_num"]
categorical_features = ["Season", "Day_or_Night"]

train_val, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["aqi_cat"],
)

train, val = train_test_split(
    train_val,
    test_size=0.20,
    random_state=42,
    stratify=train_val["aqi_cat"],
)

print("Rows:", {"train": len(train), "val": len(val), "test": len(test)})

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

models = {
    "mean_baseline": None,
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    "hist_gradient_boosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        random_state=42,
    ),
}

def get_metrics(y_true, y_pred):
    if np.std(y_pred) == 0:
        sp = np.nan
    else:
        sp = spearmanr(y_true, y_pred).correlation

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Spearman": sp,
    }

X_train = train[numeric_features + categorical_features]
y_train = train[target].values

X_test = test[numeric_features + categorical_features]
y_test = test[target].values

results = []

for name, model in models.items():
    if model is None:
        pred = np.full(shape=len(y_test), fill_value=np.mean(y_train), dtype=float)
    else:
        pipe = Pipeline(
            [
                ("preprocess", preprocess),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

    row = {
        "model": name,
        "target": target,
        "split": "random_stratified_by_aqi_cat",
        "features": "Temperature+Humidity+hour+Season+Day_or_Night",
        "n_train": len(train),
        "n_test": len(test),
    }
    row.update(get_metrics(y_test, pred))
    results.append(row)

res = pd.DataFrame(results).sort_values("RMSE")
print("\nResults:")
print(res.to_string(index=False))

out = OUTDIR / "random_split_tabular_baseline_results.csv"
res.to_csv(out, index=False)
print("\nSaved:", out)