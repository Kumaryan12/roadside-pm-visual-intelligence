from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


warnings.filterwarnings("ignore")

ROOT = Path("experiments/mumma_281_pipeline_v1")
DATA = ROOT / "data/processed/final_feature_table.csv"

EMB_NPY = ROOT / "embeddings/mumma281_resnet50_gap_embeddings.npy"
EMB_INDEX = ROOT / "embeddings/mumma281_resnet50_gap_index.csv"

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

OUT_RESULTS = REPORTS / "pm25_resnet_t7_random2fold_lstm_fair_results.csv"
OUT_PREDS = REPORTS / "pm25_resnet_t7_random2fold_lstm_fair_predictions.csv"

TARGET = "value.sPM2"

T = 7
RANDOM_STATE = 42
N_SPLITS = 2

EPOCHS = 250
PATIENCE = 35
BATCH_SIZE = 32

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    sp = np.nan if np.std(y_pred) == 0 else spearmanr(y_true, y_pred).correlation

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(sp) if sp == sp else np.nan,
        "Bias": float(np.mean(y_pred - y_true)),
    }


def add_time_features(df):
    df = df.copy()

    dt = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df["timestamp_parsed"] = dt

    return df


def build_sequences(features, y, valid_mask, t=7):
    """
    Creates rolling T-length sequences.

    Important:
    We only keep a sequence if all T frames in that window have valid ResNet embeddings.
    This avoids silently dropping failed embedding rows and creating fake temporal continuity.
    """
    X_seq = []
    y_seq = []
    source_indices = []

    for i in range(t - 1, len(features)):
        window_valid = valid_mask[i - t + 1:i + 1]

        if not np.all(window_valid):
            continue

        X_seq.append(features[i - t + 1:i + 1])
        y_seq.append(y[i])
        source_indices.append(i)

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(source_indices, dtype=int),
    )


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.15):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.15):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def make_deep_model(model_type, input_dim, hidden_dim, dropout):
    if model_type == "gru":
        return GRURegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    if model_type == "lstm":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def train_one_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    model_type="gru",
    hidden_dim=32,
    dropout=0.15,
    lr=1e-3,
    weight_decay=1e-3,
):
    train_ds = SeqDataset(X_train, y_train)
    val_ds = SeqDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = make_deep_model(
        model_type=model_type,
        input_dim=X_train.shape[-1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    preds = []

    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(pred)

    return np.concatenate(preds)


def run_deep_random2fold(
    X_seq,
    y_seq,
    model_type,
    hidden_dim,
    dropout,
    log_target,
):
    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pred_oof = np.zeros(len(y_seq), dtype=float)

    target = np.log1p(y_seq) if log_target else y_seq

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_seq), start=1):
        print(
            f"Fold {fold}/{N_SPLITS} | "
            f"{model_type} | hidden={hidden_dim} | "
            f"dropout={dropout} | log_target={log_target}"
        )

        pred = train_one_fold(
            X_train=X_seq[train_idx],
            y_train=target[train_idx],
            X_val=X_seq[val_idx],
            y_val=target[val_idx],
            model_type=model_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        if log_target:
            pred = np.expm1(pred)

        pred_oof[val_idx] = np.clip(pred, 0, None)

    return pred_oof


def run_flattened_tree_random2fold(
    X_seq,
    y_seq,
    model_name,
    log_target,
):
    X_flat = X_seq.reshape(X_seq.shape[0], -1)

    target = np.log1p(y_seq) if log_target else y_seq

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    if model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=700,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    pred = cross_val_predict(
        pipe,
        X_flat,
        target,
        cv=cv,
    )

    if log_target:
        pred = np.expm1(pred)

    return np.clip(pred, 0, None)


def save_checkpoint(records, pred_frames):
    results = pd.DataFrame(records).sort_values("RMSE")
    results.to_csv(OUT_RESULTS, index=False)

    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(OUT_PREDS, index=False)


def main():
    set_seed(RANDOM_STATE)

    print("Device:", DEVICE)
    print("Target:", TARGET)
    print("T:", T)
    print("CV: random 2-fold sequence CV")

    df = pd.read_csv(DATA)
    df = add_time_features(df)

    emb = np.load(EMB_NPY)
    idx = pd.read_csv(EMB_INDEX)

    print("\nLoaded:")
    print("Feature table:", df.shape)
    print("Embedding:", emb.shape)
    print("Index:", idx.shape)

    if len(df) != len(emb):
        raise ValueError(f"Row mismatch: df={len(df)}, emb={len(emb)}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    target_valid = df[TARGET].notna().values
    emb_valid = np.isfinite(emb).all(axis=1)
    valid_mask = target_valid & emb_valid

    df["_original_row_id"] = np.arange(len(df))

    sort_cols = ["timestamp_parsed"]

    if "sample_index" in df.columns:
        sort_cols.append("sample_index")

    order = df.sort_values(sort_cols).index.values

    df = df.iloc[order].reset_index(drop=True)
    emb = emb[order]
    valid_mask = valid_mask[order]

    y = df[TARGET].values.astype(float)

    print("\nAfter temporal sort:")
    print("Rows:", len(df))
    print("Valid target rows:", int(target_valid.sum()), "/", len(target_valid))
    print("Valid embedding rows:", int(emb_valid.sum()), "/", len(emb_valid))
    print("Valid target + embedding rows:", int(valid_mask.sum()), "/", len(valid_mask))
    print("Timestamp range:", df["timestamp_parsed"].min(), "to", df["timestamp_parsed"].max())
    print("Duplicate timestamps:", df["timestamp_parsed"].duplicated().sum())

    print("\nTarget summary:")
    print(pd.Series(y).describe().to_string())

    if EMB_INDEX.exists():
        print("\nEmbedding status:")
        if "resnet_status" in idx.columns:
            print(idx["resnet_status"].value_counts().to_string())

    pca_dims = [8, 16, 32, 64]

    records = []
    pred_frames = []

    for pca_dim in pca_dims:
        print("\n" + "=" * 100)
        print("PCA dim:", pca_dim)

        # Fit scaler/PCA only on rows with valid ResNet embeddings.
        # Failed rows contain NaNs, so PCA cannot see them.
        # We still keep the full 281-row order, but invalid rows are only mean-filled
        # so that transform() can run. They are excluded later by valid_mask when
        # building T=7 sequences.
        valid_emb = emb[valid_mask]

        if len(valid_emb) < pca_dim + 1:
            print(f"Not enough valid embeddings for PCA dim {pca_dim}. Skipping.")
            continue

        scaler = StandardScaler()
        scaler.fit(valid_emb)

        emb_scaled_full = np.empty_like(emb, dtype=np.float32)

        valid_scaled = scaler.transform(valid_emb).astype(np.float32)
        emb_scaled_full[valid_mask] = valid_scaled

        # For invalid rows, use zero after scaling, which corresponds roughly to
        # the mean valid embedding in standardized space.
        emb_scaled_full[~valid_mask] = 0.0

        pca = PCA(
            n_components=pca_dim,
            random_state=RANDOM_STATE,
        )

        pca.fit(valid_scaled)

        emb_pca = pca.transform(emb_scaled_full).astype(np.float32)
        explained = float(pca.explained_variance_ratio_.sum())

        X_seq, y_seq, source_indices = build_sequences(
            features=emb_pca,
            y=y,
            valid_mask=valid_mask,
            t=T,
        )

        print("Sequence shape:", X_seq.shape)
        print("Target sequence shape:", y_seq.shape)
        print("Valid T=7 sequences:", len(y_seq))
        print("Explained variance:", explained)

        if len(y_seq) < 20:
            print("Too few valid T=7 sequences. Skipping PCA dim:", pca_dim)
            continue

        for model_name in ["extra_trees", "random_forest"]:
            for log_target in [False, True]:
                exp = (
                    f"resnet50_pca{pca_dim}_T{T}_random2fold_"
                    f"flattened_{model_name}_"
                    f"{'log1p' if log_target else 'raw'}"
                )

                print("\nSTART", exp)

                pred = run_flattened_tree_random2fold(
                    X_seq=X_seq,
                    y_seq=y_seq,
                    model_name=model_name,
                    log_target=log_target,
                )

                row = {
                    "experiment": exp,
                    "feature_set": f"resnet50_pca{pca_dim}_T{T}",
                    "model": f"flattened_{model_name}",
                    "T": T,
                    "cv": "random_2fold_sequence_cv",
                    "pca_dim": pca_dim,
                    "n_rows": len(y_seq),
                    "n_features_per_timestep": pca_dim,
                    "n_flattened_features": pca_dim * T,
                    "explained_variance": explained,
                    "log_target": log_target,
                    "valid_embedding_rows": int(valid_mask.sum()),
                    "total_rows": len(valid_mask),
                }

                row.update(metrics(y_seq, pred))
                records.append(row)

                pred_frames.append(
                    pd.DataFrame(
                        {
                            "source_row_index": source_indices,
                            "experiment": exp,
                            "model": f"flattened_{model_name}",
                            "pca_dim": pca_dim,
                            "y_true_pm25": y_seq,
                            "y_pred_pm25": pred,
                            "residual_pred_minus_true": pred - y_seq,
                        }
                    )
                )

                save_checkpoint(records, pred_frames)

                print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman", "Bias"]})

        configs = [
            ("gru", 8, 0.10),
            ("gru", 16, 0.15),
            ("gru", 32, 0.20),
            ("lstm", 8, 0.10),
            ("lstm", 16, 0.15),
            ("lstm", 32, 0.20),
        ]

        for model_type, hidden_dim, dropout in configs:
            for log_target in [False, True]:
                exp = (
                    f"resnet50_pca{pca_dim}_T{T}_random2fold_"
                    f"{model_type}_h{hidden_dim}_"
                    f"{'log1p' if log_target else 'raw'}"
                )

                print("\nSTART", exp)

                pred = run_deep_random2fold(
                    X_seq=X_seq,
                    y_seq=y_seq,
                    model_type=model_type,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    log_target=log_target,
                )

                row = {
                    "experiment": exp,
                    "feature_set": f"resnet50_pca{pca_dim}_T{T}",
                    "model": f"{model_type}_h{hidden_dim}",
                    "T": T,
                    "cv": "random_2fold_sequence_cv",
                    "pca_dim": pca_dim,
                    "n_rows": len(y_seq),
                    "n_features_per_timestep": pca_dim,
                    "n_flattened_features": pca_dim * T,
                    "explained_variance": explained,
                    "log_target": log_target,
                    "valid_embedding_rows": int(valid_mask.sum()),
                    "total_rows": len(valid_mask),
                }

                row.update(metrics(y_seq, pred))
                records.append(row)

                pred_frames.append(
                    pd.DataFrame(
                        {
                            "source_row_index": source_indices,
                            "experiment": exp,
                            "model": f"{model_type}_h{hidden_dim}",
                            "pca_dim": pca_dim,
                            "y_true_pm25": y_seq,
                            "y_pred_pm25": pred,
                            "residual_pred_minus_true": pred - y_seq,
                        }
                    )
                )

                save_checkpoint(records, pred_frames)

                print({k: row[k] for k in ["MAE", "RMSE", "R2", "Spearman", "Bias"]})

    if not records:
        print("\nNo results produced. Check valid embedding rows and T=7 sequence availability.")
        return

    results = pd.DataFrame(records).sort_values("RMSE")
    results.to_csv(OUT_RESULTS, index=False)

    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(OUT_PREDS, index=False)

    print("\nFinal ResNet50 T7 random 2-fold fair results:")
    print(results.head(60).to_string(index=False))

    print("\nSaved:")
    print(OUT_RESULTS)
    print(OUT_PREDS)


if __name__ == "__main__":
    main()