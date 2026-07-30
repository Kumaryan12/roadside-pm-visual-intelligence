from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

MANIFEST = Path("experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv")
EMB_DIR = Path("experiments/traqid_pretraining_v1/embeddings")
OUTDIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "PM2.5"

EMBEDDING_FILES = {
    "front": "traqid_mobilenetv2_torch_front_embeddings.npy",
    "rear": "traqid_mobilenetv2_torch_rear_embeddings.npy",
    "mean": "traqid_mobilenetv2_torch_front_rear_mean_embeddings.npy",
    "concat": "traqid_mobilenetv2_torch_front_rear_concat_embeddings.npy",
}

df = pd.read_csv(MANIFEST)

df = df[
    (df["front_exists"] == True)
    & (df["rear_exists"] == True)
    & (df["env_plausible"] == True)
].copy().reset_index(drop=True)

df["created_at_parsed"] = pd.to_datetime(df["created_at_parsed"])
df["hour_num"] = df["created_at_parsed"].dt.hour

numeric_features = ["Temperature", "Humidity", "hour_num"]
categorical_features = ["Season", "Day_or_Night"]

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

def make_random_split(df):
    from sklearn.model_selection import train_test_split

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

def fit_embedding_ridge(X_emb, y, train_idx, test_idx, pca_dim=128, alpha=100.0):
    n_components = min(pca_dim, X_emb.shape[1], len(train_idx) - 1)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )

    pipe.fit(X_emb[train_idx], y[train_idx])
    pred = pipe.predict(X_emb[test_idx])
    return pred

def fit_tabular_plus_embedding(X_emb, df, y, train_idx, test_idx, pca_dim=128, alpha=100.0):
    n_components = min(pca_dim, X_emb.shape[1], len(train_idx) - 1)

    tab_preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    tab_train = tab_preprocess.fit_transform(df.iloc[train_idx][numeric_features + categorical_features])
    tab_test = tab_preprocess.transform(df.iloc[test_idx][numeric_features + categorical_features])

    emb_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )

    emb_train = emb_pipe.fit_transform(X_emb[train_idx])
    emb_test = emb_pipe.transform(X_emb[test_idx])

    if hasattr(tab_train, "toarray"):
        tab_train = tab_train.toarray()
        tab_test = tab_test.toarray()

    X_train = np.hstack([tab_train, emb_train])
    X_test = np.hstack([tab_test, emb_test])

    model = Ridge(alpha=alpha)
    model.fit(X_train, y[train_idx])
    pred = model.predict(X_test)
    return pred

def fit_tabular_hgb(df, y, train_idx, test_idx):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    random_state=42,
                ),
            ),
        ]
    )

    X_train = df.iloc[train_idx][numeric_features + categorical_features]
    X_test = df.iloc[test_idx][numeric_features + categorical_features]

    model.fit(X_train, y[train_idx])
    pred = model.predict(X_test)
    return pred

y = df[TARGET].values.astype(float)

results = []

splits = {
    "random_stratified": make_random_split(df),
    "chronological": make_chrono_split(df),
}

for split_name, (train_idx, val_idx, test_idx) in splits.items():
    print("\n" + "=" * 100)
    print("Split:", split_name)
    print("Rows:", {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)})

    mean_pred = np.full(len(test_idx), np.mean(y[train_idx]), dtype=float)
    row = {
        "split": split_name,
        "model": "mean_baseline",
        "embedding": "none",
        "features": "none",
        "target": TARGET,
    }
    row.update(get_metrics(y[test_idx], mean_pred))
    results.append(row)

    tab_pred = fit_tabular_hgb(df, y, train_idx, test_idx)
    row = {
        "split": split_name,
        "model": "hgb_tabular",
        "embedding": "none",
        "features": "Temperature+Humidity+hour+Season+Day_or_Night",
        "target": TARGET,
    }
    row.update(get_metrics(y[test_idx], tab_pred))
    results.append(row)

    for emb_name, emb_file in EMBEDDING_FILES.items():
        X_emb_all = np.load(EMB_DIR / emb_file)

        if X_emb_all.shape[0] != len(pd.read_csv(MANIFEST)):
            raise ValueError(
                f"Embedding rows {X_emb_all.shape[0]} do not match original manifest rows {len(pd.read_csv(MANIFEST))}"
            )

        X_emb = X_emb_all[df["row_id"].values]

        pred_emb = fit_embedding_ridge(X_emb, y, train_idx, test_idx, pca_dim=128, alpha=100.0)
        row = {
            "split": split_name,
            "model": "ridge_pca128_embedding_only",
            "embedding": emb_name,
            "features": "image_embedding",
            "target": TARGET,
        }
        row.update(get_metrics(y[test_idx], pred_emb))
        results.append(row)

        pred_fusion = fit_tabular_plus_embedding(X_emb, df, y, train_idx, test_idx, pca_dim=128, alpha=100.0)
        row = {
            "split": split_name,
            "model": "ridge_pca128_tabular_plus_embedding",
            "embedding": emb_name,
            "features": "tabular+image_embedding",
            "target": TARGET,
        }
        row.update(get_metrics(y[test_idx], pred_fusion))
        results.append(row)

res = pd.DataFrame(results).sort_values(["split", "RMSE"])
print("\nFinal results:")
print(res.to_string(index=False))

out = OUTDIR / "embedding_baseline_results.csv"
res.to_csv(out, index=False)
print("\nSaved:", out)