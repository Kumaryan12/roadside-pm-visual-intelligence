import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


TARGET = "PM2.5"


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


class T7PM25Dataset(Dataset):
    def __init__(self, seq_df, embedding_array, y_scaled=None):
        self.seq_df = seq_df.reset_index(drop=True)
        self.embedding_array = embedding_array
        self.y_scaled = y_scaled

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]

        start_id = int(row["seq_start_row_id"])
        end_id = int(row["seq_end_row_id"])
        ids = np.arange(start_id, end_id + 1)

        if len(ids) != 7:
            raise ValueError(f"Expected T=7 but got {len(ids)} for idx={idx}")

        x = self.embedding_array[ids].astype(np.float32)

        if self.y_scaled is None:
            y = np.array([0.0], dtype=np.float32)
        else:
            y = np.array([self.y_scaled[idx]], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


class GRUPM25Regressor(nn.Module):
    def __init__(
        self,
        input_dim,
        projection_dim=512,
        hidden_dim=256,
        num_layers=2,
        dropout=0.25,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.projector = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.projector(x)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def train_base_model(
    train_df,
    val_df,
    embeddings,
    device,
    seed,
    batch_size,
    epochs,
    patience,
    lr,
    weight_decay,
    projection_dim,
    hidden_dim,
    num_layers,
    dropout,
):
    seed_everything(seed)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(train_df[[TARGET]].values).reshape(-1)
    y_val_scaled = y_scaler.transform(val_df[[TARGET]].values).reshape(-1)

    train_ds = T7PM25Dataset(train_df, embeddings, y_train_scaled)
    val_ds = T7PM25Dataset(val_df, embeddings, y_val_scaled)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = GRUPM25Regressor(
        input_dim=embeddings.shape[1],
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.squeeze(-1).to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.squeeze(-1).to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    return model, y_scaler, pd.DataFrame(history), best_epoch, best_val


def predict_base_model(model, y_scaler, seq_df, embeddings, device, batch_size):
    ds = T7PM25Dataset(seq_df, embeddings, y_scaled=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    preds_scaled = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds_scaled.append(pred)

    preds_scaled = np.concatenate(preds_scaled).reshape(-1, 1)
    preds = y_scaler.inverse_transform(preds_scaled).reshape(-1)

    return preds


def normalize_row_id(df, fallback_name=None):
    df = df.copy()

    if "row_id" in df.columns:
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
        df = df.dropna(subset=["row_id"]).copy()
        df["row_id"] = df["row_id"].astype(int)
        return df

    if fallback_name and fallback_name in df.columns:
        df = df.rename(columns={fallback_name: "row_id"})
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
        df = df.dropna(subset=["row_id"]).copy()
        df["row_id"] = df["row_id"].astype(int)
        return df

    raise ValueError("Could not find row_id column.")


def add_time_features(df, time_col="created_at"):
    df = df.copy()

    if time_col not in df.columns:
        return df

    dt = pd.to_datetime(df[time_col], errors="coerce")

    hour = dt.dt.hour.fillna(0).astype(float)
    month = dt.dt.month.fillna(1).astype(float)
    dayofweek = dt.dt.dayofweek.fillna(0).astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    return df


def is_forbidden_input_column(col):
    c = col.lower()

    forbidden_exact = {
        "pm2.5",
        "pm10",
        "aqi",
        "log1p_pm2.5",
        "log1p_pm10",
        "log1p_aqi",
        "target_pm2.5",
        "target_pm10",
        "target_aqi",
        "y_true_pm25",
        "y_pred_pm25",
    }

    forbidden_contains = [
        "pm2.5",
        "pm25",
        "pm_2_5",
        "pm10",
        "aqi",
        "y_true",
        "y_pred",
        "target",
        "split",
        "date_chrono",
        "front_path",
        "rear_path",
        "image",
        "path",
        "file",
        "error",
        "status",
    ]

    if c in forbidden_exact:
        return True

    return any(x in c for x in forbidden_contains)


def build_row_feature_table(manifest_path, yolo_path, road_path):
    manifest = safe_read_csv(manifest_path)
    yolo = safe_read_csv(yolo_path)
    road = safe_read_csv(road_path)

    manifest = normalize_row_id(manifest)
    yolo = normalize_row_id(yolo)
    road = normalize_row_id(road, fallback_name="sample_index")

    manifest = add_time_features(manifest, "created_at")

    base_cols = [
        "row_id",
        "created_at",
        "Temperature",
        "Humidity",
        "Season",
        "Day_or_Night",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dayofweek_sin",
        "dayofweek_cos",
    ]

    base_cols = [c for c in base_cols if c in manifest.columns]
    base = manifest[base_cols].copy()

    yolo_keep = [
        c for c in yolo.columns
        if c == "row_id"
        or (
            pd.api.types.is_numeric_dtype(yolo[c])
            and not is_forbidden_input_column(c)
        )
    ]

    road_keep = [
        c for c in road.columns
        if c == "row_id"
        or (
            pd.api.types.is_numeric_dtype(road[c])
            and not is_forbidden_input_column(c)
        )
    ]

    yolo = yolo[yolo_keep].copy()
    road = road[road_keep].copy()

    merged = base.merge(yolo, on="row_id", how="left", suffixes=("", "_yolo"))
    merged = merged.merge(road, on="row_id", how="left", suffixes=("", "_road"))

    cat_cols = [c for c in ["Season", "Day_or_Night"] if c in merged.columns]
    if cat_cols:
        merged = pd.get_dummies(merged, columns=cat_cols, dummy_na=True)

    merged = merged.sort_values("row_id").reset_index(drop=True)

    feature_cols = []
    for c in merged.columns:
        if c == "row_id":
            continue
        if is_forbidden_input_column(c):
            continue
        if pd.api.types.is_numeric_dtype(merged[c]) or merged[c].dtype == bool:
            feature_cols.append(c)

    return merged, sorted(set(feature_cols))


def make_feature_matrix(row_features, feature_cols):
    row_features = row_features.dropna(subset=["row_id"]).copy()
    row_features["row_id"] = row_features["row_id"].astype(int)

    max_row_id = int(row_features["row_id"].max())

    mat = np.full(
        (max_row_id + 1, len(feature_cols)),
        np.nan,
        dtype=np.float32,
    )

    row_ids = row_features["row_id"].values
    mat[row_ids] = row_features[feature_cols].values.astype(np.float32)

    return mat


def aggregate_one_sequence(seq):
    seq = np.asarray(seq, dtype=np.float32)

    with np.errstate(all="ignore"):
        last = seq[-1]
        mean = np.nanmean(seq, axis=0)
        std = np.nanstd(seq, axis=0)
        minv = np.nanmin(seq, axis=0)
        maxv = np.nanmax(seq, axis=0)

    out = np.concatenate([last, mean, std, minv, maxv], axis=0)
    out[~np.isfinite(out)] = np.nan

    return out


def build_sequence_features(seq_df, feature_matrix, T=7):
    X = []
    keep = []

    for i, row in seq_df.iterrows():
        start = int(row["seq_start_row_id"])
        end = int(row["seq_end_row_id"])
        ids = np.arange(start, end + 1)

        if len(ids) != T:
            continue
        if ids.min() < 0:
            continue
        if ids.max() >= feature_matrix.shape[0]:
            continue

        seq = feature_matrix[ids]
        agg = aggregate_one_sequence(seq)

        X.append(agg)
        keep.append(i)

    X = np.asarray(X, dtype=np.float32)
    kept_df = seq_df.iloc[keep].reset_index(drop=True)

    return X, kept_df


def clean_feature_columns(X_train, X_test, names):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    observed_count = np.isfinite(X_train).sum(axis=0)
    keep_mask = observed_count > 0

    X_train = X_train[:, keep_mask]
    X_test = X_test[:, keep_mask]
    names = [n for n, keep in zip(names, keep_mask) if keep]

    return X_train, X_test, names


def evaluate(actual, pred):
    return {
        "R2": float(r2_score(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAE": float(mean_absolute_error(actual, pred)),
    }


def make_residual_models(seed):
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),

        "extra_trees_conservative_plus_basepred": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=900,
                random_state=seed,
                n_jobs=-1,
                max_features=0.35,
                min_samples_leaf=4,
                bootstrap=False,
            )),
        ]),

        "extra_trees_default_plus_basepred": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=700,
                random_state=seed,
                n_jobs=-1,
                max_features=0.5,
                min_samples_leaf=2,
                bootstrap=False,
            )),
        ]),

        "extra_trees_smooth_plus_basepred": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=1000,
                random_state=seed,
                n_jobs=-1,
                max_features=0.5,
                min_samples_leaf=8,
                bootstrap=False,
            )),
        ]),

        "random_forest_smooth_plus_basepred": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=900,
                random_state=seed,
                n_jobs=-1,
                max_features=0.5,
                min_samples_leaf=6,
                bootstrap=True,
            )),
        ]),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence-table", required=True)
    parser.add_argument("--embedding-npy", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--yolo-features", required=True)
    parser.add_argument("--road-features", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--split-mode", choices=["random"], default="random")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)

    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    seq = safe_read_csv(args.sequence_table)
    seq = seq.dropna(subset=[TARGET]).copy()
    seq = seq[seq["T"] == 7].copy() if "T" in seq.columns else seq.copy()
    seq = seq.reset_index(drop=True)

    embeddings = np.load(args.embedding_npy, mmap_mode="r")
    print("Embedding shape:", embeddings.shape)

    max_needed_row = int(seq["seq_end_row_id"].max())
    if max_needed_row >= embeddings.shape[0]:
        raise ValueError(
            f"Sequence table needs row {max_needed_row}, but embeddings have {embeddings.shape[0]} rows."
        )

    all_idx = np.arange(len(seq))

    trainval_idx, test_idx = train_test_split(
        all_idx,
        train_size=args.train_frac,
        random_state=args.seed,
        shuffle=True,
    )

    trainval_df = seq.iloc[trainval_idx].reset_index(drop=True)
    test_df = seq.iloc[test_idx].reset_index(drop=True)

    print("Trainval rows:", len(trainval_df))
    print("Test rows:", len(test_df))

    oof_base_pred = np.full(len(trainval_df), np.nan, dtype=np.float32)
    test_base_preds = []

    fold_summaries = []
    all_histories = []

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(trainval_df), start=1):
        print("\n" + "=" * 90)
        print(f"OOF BASE MODEL FOLD {fold}/{args.folds}")
        print("=" * 90)

        fold_train = trainval_df.iloc[tr_idx].reset_index(drop=True)
        fold_val = trainval_df.iloc[va_idx].reset_index(drop=True)

        model, y_scaler, history, best_epoch, best_val = train_base_model(
            train_df=fold_train,
            val_df=fold_val,
            embeddings=embeddings,
            device=device,
            seed=args.seed + fold,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            projection_dim=args.projection_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

        fold_val_pred = predict_base_model(
            model=model,
            y_scaler=y_scaler,
            seq_df=fold_val,
            embeddings=embeddings,
            device=device,
            batch_size=args.batch_size,
        )

        oof_base_pred[va_idx] = fold_val_pred.astype(np.float32)

        fold_test_pred = predict_base_model(
            model=model,
            y_scaler=y_scaler,
            seq_df=test_df,
            embeddings=embeddings,
            device=device,
            batch_size=args.batch_size,
        )

        test_base_preds.append(fold_test_pred)

        fold_actual = fold_val[TARGET].values.astype(float)
        fold_metrics = evaluate(fold_actual, fold_val_pred)

        fold_summary = {
            "fold": fold,
            "train_rows": len(fold_train),
            "val_rows": len(fold_val),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "oof_R2": fold_metrics["R2"],
            "oof_RMSE": fold_metrics["RMSE"],
            "oof_MAE": fold_metrics["MAE"],
        }
        fold_summaries.append(fold_summary)

        history["fold"] = fold
        all_histories.append(history)

        print("Fold OOF metrics:", fold_summary)

    if np.isnan(oof_base_pred).any():
        raise RuntimeError("Some OOF base predictions are missing.")

    test_base_pred = np.mean(np.vstack(test_base_preds), axis=0)

    trainval_actual = trainval_df[TARGET].values.astype(float)
    test_actual = test_df[TARGET].values.astype(float)

    oof_residual = trainval_actual - oof_base_pred

    oof_base_metrics = evaluate(trainval_actual, oof_base_pred)
    test_base_metrics = evaluate(test_actual, test_base_pred)

    print("\nOOF base metrics on trainval:", oof_base_metrics)
    print("Base ensemble metrics on test:", test_base_metrics)

    row_features, raw_feature_cols = build_row_feature_table(
        manifest_path=args.manifest,
        yolo_path=args.yolo_features,
        road_path=args.road_features,
    )

    feature_matrix = make_feature_matrix(row_features, raw_feature_cols)

    X_trainval, trainval_kept = build_sequence_features(trainval_df, feature_matrix, T=7)
    X_test, test_kept = build_sequence_features(test_df, feature_matrix, T=7)

    if len(trainval_kept) != len(trainval_df):
        raise RuntimeError("Some trainval rows were dropped during engineered feature construction.")
    if len(test_kept) != len(test_df):
        raise RuntimeError("Some test rows were dropped during engineered feature construction.")

    agg_names = []
    for prefix in ["last", "mean", "std", "min", "max"]:
        for col in raw_feature_cols:
            agg_names.append(f"{prefix}__{col}")

    X_trainval, X_test, agg_names = clean_feature_columns(X_trainval, X_test, agg_names)

    X_train_res = np.column_stack([
        X_trainval,
        oof_base_pred,
        np.abs(oof_base_pred),
        oof_base_pred ** 2,
    ])

    X_test_res = np.column_stack([
        X_test,
        test_base_pred,
        np.abs(test_base_pred),
        test_base_pred ** 2,
    ])

    feature_names = agg_names + [
        "base_pred_pm25",
        "abs_base_pred_pm25",
        "base_pred_pm25_squared",
    ]

    rows = []

    rows.append({
        "model": "oof_base_ensemble_only",
        "R2": test_base_metrics["R2"],
        "RMSE": test_base_metrics["RMSE"],
        "MAE": test_base_metrics["MAE"],
    })

    prediction_outputs = []

    residual_models = make_residual_models(args.seed)

    for model_name, residual_model in residual_models.items():
        print("Training residual model:", model_name)

        residual_model.fit(X_train_res, oof_residual)
        pred_residual_test = residual_model.predict(X_test_res)

        final_pred = test_base_pred + pred_residual_test

        m = evaluate(test_actual, final_pred)

        rows.append({
            "model": model_name,
            "R2": m["R2"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
        })

        pred_df = test_df[[
            "sequence_id",
            "target_row_id",
            "target_created_at",
            "seq_start_row_id",
            "seq_end_row_id",
        ]].copy()

        pred_df["actual_PM2.5"] = test_actual
        pred_df["base_pred_PM2.5"] = test_base_pred
        pred_df["predicted_residual_PM2.5"] = pred_residual_test
        pred_df["final_predicted_PM2.5"] = final_pred
        pred_df["final_residual_PM2.5"] = final_pred - test_actual
        pred_df["residual_model"] = model_name

        prediction_outputs.append(pred_df)

    results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    all_predictions = pd.concat(prediction_outputs, ignore_index=True)
    fold_summary_df = pd.DataFrame(fold_summaries)
    histories = pd.concat(all_histories, ignore_index=True)

    oof_pred_df = trainval_df[[
        "sequence_id",
        "target_row_id",
        "target_created_at",
        "seq_start_row_id",
        "seq_end_row_id",
    ]].copy()
    oof_pred_df["actual_PM2.5"] = trainval_actual
    oof_pred_df["oof_base_pred_PM2.5"] = oof_base_pred
    oof_pred_df["oof_residual_PM2.5"] = oof_residual

    results_path = out_dir / "pm25_oof_residual_results.csv"
    results_md_path = out_dir / "pm25_oof_residual_results.md"
    predictions_path = out_dir / "pm25_oof_residual_predictions.csv"
    fold_summary_path = out_dir / "pm25_oof_fold_summary.csv"
    history_path = out_dir / "pm25_oof_training_history.csv"
    oof_pred_path = out_dir / "pm25_oof_trainval_predictions.csv"
    feature_path = out_dir / "pm25_oof_residual_feature_columns.txt"
    config_path = out_dir / "config.json"

    results.to_csv(results_path, index=False)
    results_md_path.write_text(results.to_markdown(index=False), encoding="utf-8")
    all_predictions.to_csv(predictions_path, index=False)
    fold_summary_df.to_csv(fold_summary_path, index=False)
    histories.to_csv(history_path, index=False)
    oof_pred_df.to_csv(oof_pred_path, index=False)
    feature_path.write_text("\n".join(feature_names), encoding="utf-8")

    config = vars(args)
    config["device"] = str(device)
    config["n_sequence_rows"] = int(len(seq))
    config["trainval_rows"] = int(len(trainval_df))
    config["test_rows"] = int(len(test_df))
    config["raw_engineered_features"] = int(len(raw_feature_cols))
    config["aggregated_features_after_cleaning"] = int(len(agg_names))
    config["residual_features_with_basepred"] = int(len(feature_names))

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("PM2.5 OOF RESIDUAL CORRECTION COMPLETE")
    print("=" * 90)
    print("Folds:", args.folds)
    print("Trainval rows:", len(trainval_df))
    print("Test rows:", len(test_df))
    print("Raw engineered features:", len(raw_feature_cols))
    print("Aggregated features after cleaning:", len(agg_names))
    print("Residual features with basepred:", len(feature_names))
    print()
    print(results.to_string(index=False))
    print()
    print("Saved:", results_path)
    print("Saved:", results_md_path)
    print("Saved:", predictions_path)
    print("Saved:", fold_summary_path)
    print("Saved:", history_path)
    print("Saved:", oof_pred_path)
    print("Saved:", feature_path)
    print("Saved:", config_path)


if __name__ == "__main__":
    main()