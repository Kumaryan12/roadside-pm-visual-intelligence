import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


TARGETS = ["PM2.5", "PM10", "aqi"]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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


def add_time_features(df, time_col="created_at"):
    df = df.copy()
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
        "actual_pm2.5",
        "actual_pm10",
        "actual_aqi",
        "predicted_pm2.5",
        "predicted_pm10",
        "predicted_aqi",
    }

    forbidden_contains = [
        "pm2.5",
        "pm25",
        "pm_2_5",
        "pm10",
        "aqi",
        "y_true",
        "y_pred",
        "actual_",
        "predicted_",
        "target",
        "split",
        "date_chrono",
        "created_at",
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


def build_row_level_feature_table(manifest_path, yolo_path, road_path):
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
        "PM2.5",
        "PM10",
        "aqi",
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

    return merged


def select_engineered_feature_cols(df):
    cols = []

    for c in df.columns:
        if c in ["row_id"] + TARGETS:
            continue

        if is_forbidden_input_column(c):
            continue

        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool:
            cols.append(c)

    return sorted(set(cols))


class T7GatedFusionDataset(Dataset):
    def __init__(self, X_img, X_eng, y):
        self.X_img = torch.tensor(X_img, dtype=torch.float32)
        self.X_eng = torch.tensor(X_eng, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_img[idx], self.X_eng[idx], self.y[idx]


class T7GatedMultimodalFusion(nn.Module):
    def __init__(
        self,
        image_dim,
        engineered_dim,
        image_projection_dim=512,
        engineered_projection_dim=128,
        image_hidden_dim=256,
        engineered_hidden_dim=128,
        fusion_hidden_dim=256,
        num_layers=2,
        dropout=0.25,
        output_dim=3,
        rnn_type="gru",
        bidirectional=False,
    ):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional

        self.image_project = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, image_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.engineered_project = nn.Sequential(
            nn.LayerNorm(engineered_dim),
            nn.Linear(engineered_dim, engineered_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM

        self.image_rnn = rnn_cls(
            input_size=image_projection_dim,
            hidden_size=image_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.engineered_rnn = rnn_cls(
            input_size=engineered_projection_dim,
            hidden_size=engineered_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        image_out_dim = image_hidden_dim * (2 if bidirectional else 1)
        engineered_out_dim = engineered_hidden_dim * (2 if bidirectional else 1)

        self.image_to_fusion = nn.Sequential(
            nn.LayerNorm(image_out_dim),
            nn.Linear(image_out_dim, fusion_hidden_dim),
            nn.ReLU(),
        )

        self.engineered_to_fusion = nn.Sequential(
            nn.LayerNorm(engineered_out_dim),
            nn.Linear(engineered_out_dim, fusion_hidden_dim),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.Sigmoid(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_hidden_dim),
            nn.Linear(fusion_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x_img, x_eng):
        img = self.image_project(x_img)
        eng = self.engineered_project(x_eng)

        img_out, _ = self.image_rnn(img)
        eng_out, _ = self.engineered_rnn(eng)

        img_last = img_out[:, -1, :]
        eng_last = eng_out[:, -1, :]

        img_repr = self.image_to_fusion(img_last)
        eng_repr = self.engineered_to_fusion(eng_last)

        combined = torch.cat([img_repr, eng_repr], dim=1)
        gate = self.gate(combined)

        fused = gate * img_repr + (1.0 - gate) * eng_repr

        return self.head(fused)


def make_random_split(n, split_seed=42, train_frac=0.80, val_frac_of_train=0.15):
    rng = np.random.default_rng(split_seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    test_size = int(round(n * (1 - train_frac)))
    test_idx = idx[:test_size]
    trainval_idx = idx[test_size:]

    val_size = int(round(len(trainval_idx) * val_frac_of_train))
    val_idx = trainval_idx[:val_size]
    train_idx = trainval_idx[val_size:]

    return train_idx, val_idx, test_idx


def make_chrono_split(seq_df):
    split_col = "target_split_date_chrono"

    if split_col not in seq_df.columns:
        raise ValueError(f"Missing {split_col} for chrono split.")

    train_idx = seq_df.index[seq_df[split_col] == "train"].to_numpy()
    val_idx = seq_df.index[seq_df[split_col] == "val"].to_numpy()
    test_idx = seq_df.index[seq_df[split_col] == "test"].to_numpy()

    return train_idx, val_idx, test_idx


def build_sequence_arrays(seq_df, row_features, feature_cols, embeddings, T=7):
    row_features = row_features.dropna(subset=["row_id"]).copy()
    row_features["row_id"] = row_features["row_id"].astype(int)

    max_row_id = int(row_features["row_id"].max())

    feature_matrix = np.full(
        (max_row_id + 1, len(feature_cols)),
        np.nan,
        dtype=np.float32,
    )

    row_ids = row_features["row_id"].values
    feature_matrix[row_ids] = row_features[feature_cols].values.astype(np.float32)

    X_img = []
    X_eng = []
    y = []
    keep_rows = []

    for i, r in seq_df.iterrows():
        start = int(r["seq_start_row_id"])
        end = int(r["seq_end_row_id"])
        ids = np.arange(start, end + 1)

        if len(ids) != T:
            continue

        if ids.min() < 0:
            continue

        if ids.max() >= len(embeddings):
            continue

        if ids.max() >= feature_matrix.shape[0]:
            continue

        target_values = r[TARGETS].values.astype(float)
        if not np.isfinite(target_values).all():
            continue

        img_seq = embeddings[ids]
        eng_seq = feature_matrix[ids]

        X_img.append(img_seq)
        X_eng.append(eng_seq)
        y.append(target_values)
        keep_rows.append(i)

    X_img = np.asarray(X_img, dtype=np.float32)
    X_eng = np.asarray(X_eng, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    seq_kept = seq_df.loc[keep_rows].reset_index(drop=True)

    return X_img, X_eng, y, seq_kept


def impute_and_scale_3d(X, train_idx):
    n, T, D = X.shape

    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = np.nan

    train_flat = X[train_idx].reshape(-1, D)

    col_median = np.nanmedian(train_flat, axis=0)
    col_median[~np.isfinite(col_median)] = 0.0

    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_median, inds[2])

    scaler = StandardScaler()
    scaler.fit(X[train_idx].reshape(-1, D))

    X_scaled = scaler.transform(X.reshape(-1, D)).reshape(n, T, D).astype(np.float32)
    X_scaled[~np.isfinite(X_scaled)] = 0.0

    return X_scaled, scaler


def scale_targets(y, train_idx):
    y = np.asarray(y, dtype=np.float32)
    y[~np.isfinite(y)] = np.nan

    y_median = np.nanmedian(y[train_idx], axis=0)
    y_median[~np.isfinite(y_median)] = 0.0

    y_inds = np.where(np.isnan(y))
    y[y_inds] = np.take(y_median, y_inds[1])

    scaler = StandardScaler()
    scaler.fit(y[train_idx])

    y_scaled = scaler.transform(y).astype(np.float32)
    y_scaled[~np.isfinite(y_scaled)] = 0.0

    return y_scaled, scaler


def evaluate(y_true, y_pred):
    rows = []

    for i, target in enumerate(TARGETS):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        rows.append({
            "target": target,
            "R2": float(r2_score(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)),
        })

    rows.append({
        "target": "Average",
        "R2": float(np.mean([r["R2"] for r in rows])),
        "RMSE": float(np.mean([r["RMSE"] for r in rows])),
        "MAE": float(np.mean([r["MAE"] for r in rows])),
    })

    return rows


def train_model(model, train_loader, val_loader, device, epochs, lr, patience):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb_img, xb_eng, yb in train_loader:
            xb_img = xb_img.to(device)
            xb_eng = xb_eng.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb_img, xb_eng)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb_img, xb_eng, yb in val_loader:
                xb_img = xb_img.to(device)
                xb_eng = xb_eng.to(device)
                yb = yb.to(device)

                pred = model(xb_img, xb_eng)
                loss = loss_fn(pred, yb)

                if torch.isfinite(loss):
                    val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed: validation loss was never finite.")

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history), best_epoch, best_val


def predict_model(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for xb_img, xb_eng, _ in loader:
            xb_img = xb_img.to(device)
            xb_eng = xb_eng.to(device)
            pred = model(xb_img, xb_eng)
            preds.append(pred.detach().cpu().numpy())

    return np.vstack(preds)


def make_prediction_df(seq_kept, idx, y_true, pred, split_name, model_seed, split_seed):
    pred_df = seq_kept.iloc[idx][[
        "sequence_id",
        "target_row_id",
        "target_created_at",
        "seq_start_row_id",
        "seq_end_row_id",
    ]].copy()

    pred_df["split"] = split_name
    pred_df["model_seed"] = model_seed
    pred_df["split_seed"] = split_seed

    for i, target in enumerate(TARGETS):
        pred_df[f"actual_{target}"] = y_true[:, i]
        pred_df[f"predicted_{target}"] = pred[:, i]
        pred_df[f"residual_{target}"] = pred[:, i] - y_true[:, i]

    return pred_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence-table", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--yolo-features", required=True)
    parser.add_argument("--road-features", required=True)
    parser.add_argument("--embedding-npy", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--split-mode", choices=["random", "chrono"], default="random")
    parser.add_argument("--T", type=int, default=7)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac-of-train", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=12)

    parser.add_argument("--image-projection-dim", type=int, default=512)
    parser.add_argument("--engineered-projection-dim", type=int, default=128)
    parser.add_argument("--image-hidden-dim", type=int, default=256)
    parser.add_argument("--engineered-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)

    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    seq_df = safe_read_csv(args.sequence_table)
    seq_df = seq_df.dropna(subset=TARGETS).reset_index(drop=True)

    row_features = build_row_level_feature_table(
        manifest_path=args.manifest,
        yolo_path=args.yolo_features,
        road_path=args.road_features,
    )

    feature_cols = select_engineered_feature_cols(row_features)

    embeddings = np.load(args.embedding_npy, mmap_mode="r")

    X_img, X_eng, y, seq_kept = build_sequence_arrays(
        seq_df=seq_df,
        row_features=row_features,
        feature_cols=feature_cols,
        embeddings=embeddings,
        T=args.T,
    )

    if args.split_mode == "random":
        train_idx, val_idx, test_idx = make_random_split(
            len(seq_kept),
            split_seed=args.split_seed,
            train_frac=args.train_frac,
            val_frac_of_train=args.val_frac_of_train,
        )
    else:
        train_idx, val_idx, test_idx = make_chrono_split(seq_kept)

    X_img_scaled, img_scaler = impute_and_scale_3d(X_img, train_idx)
    X_eng_scaled, eng_scaler = impute_and_scale_3d(X_eng, train_idx)
    y_scaled, y_scaler = scale_targets(y, train_idx)

    print("X_img shape:", X_img_scaled.shape)
    print("X_eng shape:", X_eng_scaled.shape)
    print("y shape:", y_scaled.shape)
    print("X_img finite:", np.isfinite(X_img_scaled).mean())
    print("X_eng finite:", np.isfinite(X_eng_scaled).mean())
    print("y finite:", np.isfinite(y_scaled).mean())

    train_ds = T7GatedFusionDataset(
        X_img_scaled[train_idx],
        X_eng_scaled[train_idx],
        y_scaled[train_idx],
    )

    val_ds = T7GatedFusionDataset(
        X_img_scaled[val_idx],
        X_eng_scaled[val_idx],
        y_scaled[val_idx],
    )

    test_ds = T7GatedFusionDataset(
        X_img_scaled[test_idx],
        X_eng_scaled[test_idx],
        y_scaled[test_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    image_dim = X_img_scaled.shape[2]
    engineered_dim = X_eng_scaled.shape[2]

    model = T7GatedMultimodalFusion(
        image_dim=image_dim,
        engineered_dim=engineered_dim,
        image_projection_dim=args.image_projection_dim,
        engineered_projection_dim=args.engineered_projection_dim,
        image_hidden_dim=args.image_hidden_dim,
        engineered_hidden_dim=args.engineered_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=len(TARGETS),
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
    ).to(device)

    arch_name = f"gated_{args.rnn_type}{'_bidir' if args.bidirectional else ''}"

    print("=" * 90)
    print("TRAQID T7 GATED MULTIMODAL FUSION NO-PM INPUTS")
    print("=" * 90)
    print("Architecture:", arch_name)
    print("Sequence rows:", len(seq_kept))
    print("Split mode:", args.split_mode)
    print("Model seed:", args.seed)
    print("Split seed:", args.split_seed if args.split_mode == "random" else "chrono_split_column")
    print("Train/Val/Test:", len(train_idx), len(val_idx), len(test_idx))
    print("Image dim:", image_dim)
    print("Engineered dim:", engineered_dim)
    print("Feature columns:", len(feature_cols))
    print("Targets:", TARGETS)
    print("=" * 90)

    model, history, best_epoch, best_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    test_pred_scaled = predict_model(model, test_loader, device)
    val_pred_scaled = predict_model(model, val_loader, device)

    test_pred = y_scaler.inverse_transform(test_pred_scaled)
    val_pred = y_scaler.inverse_transform(val_pred_scaled)

    test_true = y[test_idx]
    val_true = y[val_idx]

    metric_rows = evaluate(test_true, test_pred)

    metrics = pd.DataFrame(metric_rows)
    metrics["split_mode"] = args.split_mode
    metrics["model"] = f"t7_{arch_name}_resnet_yolo_road_tabular_no_pm_inputs"
    metrics["image_dim"] = image_dim
    metrics["engineered_dim"] = engineered_dim
    metrics["train_rows"] = len(train_idx)
    metrics["val_rows"] = len(val_idx)
    metrics["test_rows"] = len(test_idx)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = best_val
    metrics["model_seed"] = args.seed
    metrics["split_seed"] = args.split_seed if args.split_mode == "random" else -1

    test_pred_df = make_prediction_df(
        seq_kept=seq_kept,
        idx=test_idx,
        y_true=test_true,
        pred=test_pred,
        split_name="test",
        model_seed=args.seed,
        split_seed=args.split_seed if args.split_mode == "random" else -1,
    )

    val_pred_df = make_prediction_df(
        seq_kept=seq_kept,
        idx=val_idx,
        y_true=val_true,
        pred=val_pred,
        split_name="val",
        model_seed=args.seed,
        split_seed=args.split_seed if args.split_mode == "random" else -1,
    )

    metrics_path = out_dir / "metrics.csv"
    test_pred_path = out_dir / "predictions.csv"
    val_pred_path = out_dir / "val_predictions.csv"
    hist_path = out_dir / "training_history.csv"
    feature_path = out_dir / "engineered_feature_columns.txt"
    config_path = out_dir / "config.json"
    split_path = out_dir / "split_indices.npz"

    metrics.to_csv(metrics_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)
    val_pred_df.to_csv(val_pred_path, index=False)
    history.to_csv(hist_path, index=False)
    feature_path.write_text("\n".join(feature_cols), encoding="utf-8")

    np.savez(
        split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    config = vars(args)
    config["architecture"] = arch_name
    config["image_dim"] = int(image_dim)
    config["engineered_dim"] = int(engineered_dim)
    config["n_sequence_rows"] = int(len(seq_kept))
    config["n_engineered_features"] = int(len(feature_cols))
    config["targets"] = TARGETS

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("TRAQID T7 GATED MULTIMODAL FUSION COMPLETE")
    print("=" * 90)
    print(metrics.to_string(index=False))
    print()
    print("Saved:", metrics_path)
    print("Saved:", test_pred_path)
    print("Saved:", val_pred_path)
    print("Saved:", hist_path)
    print("Saved:", feature_path)
    print("Saved:", config_path)
    print("Saved:", split_path)


if __name__ == "__main__":
    main()