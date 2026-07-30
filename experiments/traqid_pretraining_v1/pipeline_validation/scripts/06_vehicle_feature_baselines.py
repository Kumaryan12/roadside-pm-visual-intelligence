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

YOLO_CSV = Path("experiments/traqid_pretraining_v1/pipeline_validation/outputs/traqid_yolo_vehicle_features_front.csv")
OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "PM2.5"

vehicle_features = [
    "car_count",
    "bus_count",
    "truck_count",
    "auto_count",
    "motorcycle_count",
    "bicycle_count",
    "person_count",
    "total_vehicle_count",
    "heavy_vehicle_count",
    "two_wheeler_count",
    "detections_total_raw",
]

tabular_numeric = ["Temperature", "Humidity", "hour_num"]
tabular_categorical = ["Season", "Day_or_Night"]

df = pd.read_csv(YOLO_CSV)
df = df[df["yolo_status"] == "success"].copy().reset_index(drop=True)
df["created_at_parsed"] = pd.to_datetime(df["created_at"])
df["hour_num"] = df["created_at_parsed"].dt.hour

for c in vehicle_features:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

def get_metrics(y_true, y_pred):
    sp = np.nan if np.std(y_pred) == 0 else spearmanr(y_true, y_pred).correlation
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Spearman": sp,
    }

def make_random_split(df):
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
    return train.index.values, val.index.values, test.index.values

def make_chrono_split(df):
    train_idx = df.index[df["split_date_chrono"] == "train"].values
    val_idx = df.index[df["split_date_chrono"] == "val"].values
    test_idx = df.index[df["split_date_chrono"] == "test"].values
    return train_idx, val_idx, test_idx

def make_preprocessor(feature_set):
    transformers = []

    if "vehicle" in feature_set:
        transformers.append(("vehicle", StandardScaler(), vehicle_features))

    if "tabular" in feature_set:
        transformers.append(("tab_num", StandardScaler(), tabular_numeric))
        transformers.append(("tab_cat", OneHotEncoder(handle_unknown="ignore"), tabular_categorical))

    return ColumnTransformer(transformers=transformers)

def fit_predict_model(df, y, train_idx, test_idx, feature_set, model_name):
    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            random_state=42,
        )
    else:
        raise ValueError(model_name)

    features = []
    if "vehicle" in feature_set:
        features += vehicle_features
    if "tabular" in feature_set:
        features += tabular_numeric + tabular_categorical

    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(feature_set)),
            ("model", model),
        ]
    )

    pipe.fit(df.iloc[train_idx][features], y[train_idx])
    pred = pipe.predict(df.iloc[test_idx][features])
    return pred

y = df[TARGET].values.astype(float)

splits = {
    "random_stratified": make_random_split(df),
    "chronological": make_chrono_split(df),
}

experiments = [
    ("mean_baseline", [], None),
    ("vehicle_ridge", ["vehicle"], "ridge"),
    ("vehicle_rf", ["vehicle"], "random_forest"),
    ("vehicle_hgb", ["vehicle"], "hgb"),
    ("tabular_hgb", ["tabular"], "hgb"),
    ("vehicle_plus_tabular_hgb", ["vehicle", "tabular"], "hgb"),
    ("vehicle_plus_tabular_rf", ["vehicle", "tabular"], "random_forest"),
]

results = []

for split_name, (train_idx, val_idx, test_idx) in splits.items():
    print("\n" + "=" * 100)
    print("Split:", split_name)
    print("Rows:", {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)})

    for exp_name, feature_set, model_name in experiments:
        if exp_name == "mean_baseline":
            pred = np.full(len(test_idx), np.mean(y[train_idx]), dtype=float)
        else:
            pred = fit_predict_model(df, y, train_idx, test_idx, feature_set, model_name)

        row = {
            "split": split_name,
            "model": exp_name,
            "features": "+".join(feature_set) if feature_set else "none",
            "target": TARGET,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        row.update(get_metrics(y[test_idx], pred))
        results.append(row)

res = pd.DataFrame(results).sort_values(["split", "RMSE"])
print("\nResults:")
print(res.to_string(index=False))

out = OUTDIR / "vehicle_feature_baseline_results.csv"
res.to_csv(out, index=False)
print("\nSaved:", out)

corr_cols = vehicle_features + [TARGET, "PM10", "Temperature", "Humidity"]
corr = df[corr_cols].corr(numeric_only=True)
corr_out = OUTDIR / "vehicle_feature_correlations.csv"
corr.to_csv(corr_out)
print("Saved:", corr_out)

print("\nCorrelation with PM2.5:")
print(corr[TARGET].sort_values(ascending=False).to_string())