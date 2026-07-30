from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer: str) -> torch.device:
    prefer = prefer.lower()

    if prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("WARNING: MPS requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError("device must be auto, mps, or cpu")


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_index_sequence(s: str) -> list[int]:
    return [int(x) for x in str(s).split("|") if str(x).strip()]


def encode_target(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if mode == "raw":
        return y

    if mode == "log1p":
        return np.log1p(y)

    raise ValueError(mode)


def inverse_target(
    y: np.ndarray,
    mode: str,
    clip_log_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Converts model-space prediction back to PM2.5 units.

    Important:
    For log1p target mode, unbounded neural outputs can explode after expm1().
    Example: expm1(12) ≈ 162,753.

    clip_log_range should be computed using training labels only.
    """
    y = np.asarray(y, dtype=np.float32)

    if mode == "raw":
        return y

    if mode == "log1p":
        if clip_log_range is not None:
            lo, hi = clip_log_range
            y = np.clip(y, lo, hi)
        return np.expm1(y)

    raise ValueError(mode)


def compute_clip_log_range(
    y_train_raw: np.ndarray,
    target_mode: str,
    enabled: bool,
    low_pct: float,
    high_pct: float,
) -> tuple[float, float] | None:
    if not enabled:
        return None

    if target_mode != "log1p":
        return None

    y_train_raw = np.asarray(y_train_raw, dtype=float)
    train_log = np.log1p(y_train_raw)

    lo = float(np.percentile(train_log, low_pct))
    hi = float(np.percentile(train_log, high_pct))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Invalid clipping range computed from training labels.")

    if lo >= hi:
        raise ValueError(f"Invalid clipping range: lo={lo}, hi={hi}")

    return lo, hi


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if finite_mask.sum() == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
            "Pearson": float("nan"),
            "Spearman": float("nan"),
            "finite_fraction": 0.0,
        }

    y_true_f = y_true[finite_mask]
    y_pred_f = y_pred[finite_mask]

    out = {
        "MAE": float(mean_absolute_error(y_true_f, y_pred_f)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true_f, y_pred_f))),
        "R2": float(r2_score(y_true_f, y_pred_f)),
        "finite_fraction": float(finite_mask.mean()),
    }

    if len(np.unique(y_true_f)) > 1 and len(np.unique(y_pred_f)) > 1:
        out["Pearson"] = float(pearsonr(y_true_f, y_pred_f).statistic)
        out["Spearman"] = float(spearmanr(y_true_f, y_pred_f).statistic)
    else:
        out["Pearson"] = float("nan")
        out["Spearman"] = float("nan")

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "target_created_at" in df.columns:
        dt = pd.to_datetime(df["target_created_at"], errors="coerce")
    elif "created_at" in df.columns:
        dt = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        dt = pd.Series(pd.NaT, index=df.index)

    hour = dt.dt.hour.fillna(0).astype(float)

    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    return df


def build_tabular_preprocessor(numeric_cols: list[str], categorical_cols: list[str]):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


# ============================================================
# Dataset
# ============================================================

class T7FusionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        tabular_array: np.ndarray,
        target_mode: str,
        emb_mean: np.ndarray,
        emb_std: np.ndarray,
    ):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.tabular_array = tabular_array.astype(np.float32)
        self.target_mode = target_mode

        self.emb_mean = emb_mean.astype(np.float32)
        self.emb_std = emb_std.astype(np.float32)

        self.seq_indices = [
            parse_index_sequence(s)
            for s in self.df["seq_embedding_indices"].tolist()
        ]

        self.y_raw = self.df["target_value"].astype(float).to_numpy(dtype=np.float32)
        self.y_model = encode_target(self.y_raw, target_mode).astype(np.float32)

        self.sequence_ids = self.df["sequence_id"].astype(int).to_numpy()
        self.target_row_ids = self.df["target_row_id"].astype(int).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idxs = self.seq_indices[idx]

        x_seq = np.asarray(self.embeddings[idxs], dtype=np.float32)
        x_seq = (x_seq - self.emb_mean) / self.emb_std

        x_tab = self.tabular_array[idx]

        return {
            "x_seq": torch.tensor(x_seq, dtype=torch.float32),
            "x_tab": torch.tensor(x_tab, dtype=torch.float32),
            "y": torch.tensor([self.y_model[idx]], dtype=torch.float32),
            "y_raw": torch.tensor([self.y_raw[idx]], dtype=torch.float32),
            "sequence_id": torch.tensor(self.sequence_ids[idx], dtype=torch.long),
            "target_row_id": torch.tensor(self.target_row_ids[idx], dtype=torch.long),
        }


# ============================================================
# Model
# ============================================================

class GRUTabularFusionRegressor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        tabular_dim: int,
        hidden_dim: int = 128,
        tab_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.25,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_dim, tab_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim + tab_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, x_seq, x_tab):
        seq_out, _ = self.gru(x_seq)
        seq_vec = seq_out[:, -1, :]

        tab_vec = self.tab_mlp(x_tab)

        fused = torch.cat([seq_vec, tab_vec], dim=1)
        y = self.head(fused)

        return y


# ============================================================
# Training helpers
# ============================================================

def compute_train_embedding_scaler(train_df: pd.DataFrame, embeddings: np.ndarray):
    used = set()

    for s in train_df["seq_embedding_indices"]:
        used.update(parse_index_sequence(s))

    used = sorted(used)

    x = np.asarray(embeddings[used], dtype=np.float32)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm: float | None):
    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x_seq = batch["x_seq"].to(device)
        x_tab = batch["x_tab"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(x_seq, x_tab)
        loss = criterion(pred, y)

        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        bs = x_seq.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.inference_mode()
def predict(
    model,
    loader,
    device,
    target_mode: str,
    clip_log_range: tuple[float, float] | None = None,
):
    model.eval()

    preds_model = []
    actual_raw = []
    sequence_ids = []
    target_row_ids = []

    for batch in loader:
        x_seq = batch["x_seq"].to(device)
        x_tab = batch["x_tab"].to(device)

        pred_model = model(x_seq, x_tab).detach().cpu().numpy().reshape(-1)
        y_raw = batch["y_raw"].cpu().numpy().reshape(-1)

        preds_model.append(pred_model)
        actual_raw.append(y_raw)
        sequence_ids.append(batch["sequence_id"].cpu().numpy().reshape(-1))
        target_row_ids.append(batch["target_row_id"].cpu().numpy().reshape(-1))

    preds_model = np.concatenate(preds_model)
    actual_raw = np.concatenate(actual_raw)
    sequence_ids = np.concatenate(sequence_ids)
    target_row_ids = np.concatenate(target_row_ids)

    preds_raw = inverse_target(
        preds_model,
        target_mode,
        clip_log_range=clip_log_range,
    )
    preds_raw = np.maximum(preds_raw, 0)

    return actual_raw, preds_raw, sequence_ids, target_row_ids, preds_model


def summarize_predictions(name: str, y_true, y_pred, y_model_pred=None):
    print(f"\n{name} prediction summary:")

    df = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )

    if y_model_pred is not None:
        df["pred_model_space"] = np.asarray(y_model_pred, dtype=float)

    print(
        df.describe(
            percentiles=[0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
        ).to_string()
    )


def save_scatter(y_true, y_pred, out_path: Path, title: str):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)

    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))

    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.title(title)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_T7_front_sequence_manifest_purged_block_split.csv",
    )

    parser.add_argument(
        "--embeddings",
        default="experiments/traqid_pretraining_v1/embeddings/traqid_supervised_pm25_random_front_rear_mean_embeddings.npy",
    )

    parser.add_argument("--target-mode", default="log1p", choices=["raw", "log1p"])
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tab-hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument(
        "--numeric-cols",
        default="Temperature,Humidity,hour_sin,hour_cos",
    )

    parser.add_argument(
        "--categorical-cols",
        default="Season,Day_or_Night",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Use 0 to disable.",
    )

    # Prediction clipping is enabled by default for log1p mode.
    parser.add_argument(
        "--clip-log-pred",
        dest="clip_log_pred",
        action="store_true",
        default=True,
        help="Clip log-space predictions using training-label percentiles before expm1(). Enabled by default.",
    )

    parser.add_argument(
        "--no-clip-log-pred",
        dest="clip_log_pred",
        action="store_false",
        help="Disable log prediction clipping.",
    )

    parser.add_argument("--clip-low-percentile", type=float, default=0.5)
    parser.add_argument("--clip-high-percentile", type=float, default=99.5)

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/models/t7_supervised_front_rear_mean_gru_tabular_fusion_purged_block",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/t7_supervised_front_rear_mean_gru_tabular_fusion_purged_block",
    )

    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures/t7_supervised_front_rear_mean_gru_tabular_fusion_purged_block",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    sequence_manifest = Path(args.sequence_manifest)
    embeddings_path = Path(args.embeddings)

    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    fig_dir = Path(args.fig_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seq_df = pd.read_csv(sequence_manifest)
    seq_df.columns = [str(c).strip() for c in seq_df.columns]
    seq_df = add_time_features(seq_df)

    embeddings = np.load(embeddings_path, mmap_mode="r")

    required = [
        "sequence_id",
        "split_date_chrono",
        "seq_embedding_indices",
        "target_value",
        "target_row_id",
    ]

    missing = [c for c in required if c not in seq_df.columns]
    if missing:
        raise ValueError(f"Sequence manifest missing columns: {missing}")

    # Ignore purged rows if this is a purged-block manifest.
    seq_df = seq_df[seq_df["split_date_chrono"].isin(["train", "val", "test"])].copy()

    numeric_cols = [c.strip() for c in args.numeric_cols.split(",") if c.strip()]
    categorical_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]

    for c in numeric_cols + categorical_cols:
        if c not in seq_df.columns:
            raise ValueError(f"Requested context column not found: {c}")

    train_df = seq_df[seq_df["split_date_chrono"] == "train"].copy()
    val_df = seq_df[seq_df["split_date_chrono"] == "val"].copy()
    test_df = seq_df[seq_df["split_date_chrono"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"Empty split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    device = get_device(args.device)

    print("=" * 90)
    print("TRAQID T=7 GRU + TABULAR CONTEXT FUSION")
    print("=" * 90)
    print("Sequence manifest:", sequence_manifest)
    print("Embeddings:", embeddings_path)
    print("Embedding shape:", embeddings.shape)
    print("Sequences:", len(seq_df))
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    print("Target mode:", args.target_mode)
    print("Numeric context:", numeric_cols)
    print("Categorical context:", categorical_cols)
    print("Device:", device)
    print("Gradient clipping norm:", args.grad_clip_norm)
    print("Clip log predictions:", args.clip_log_pred)

    print("\nSplit dates:")
    print("train:", sorted(train_df["date"].unique().tolist()) if "date" in train_df.columns else "")
    print("val  :", sorted(val_df["date"].unique().tolist()) if "date" in val_df.columns else "")
    print("test :", sorted(test_df["date"].unique().tolist()) if "date" in test_df.columns else "")

    # ============================================================
    # Baseline and clipping range
    # ============================================================

    y_train = train_df["target_value"].astype(float).to_numpy()
    y_val = val_df["target_value"].astype(float).to_numpy()
    y_test = test_df["target_value"].astype(float).to_numpy()

    clip_log_range = compute_clip_log_range(
        y_train_raw=y_train,
        target_mode=args.target_mode,
        enabled=args.clip_log_pred,
        low_pct=args.clip_low_percentile,
        high_pct=args.clip_high_percentile,
    )

    print("\nTarget distribution:")
    print("train target min/max:", float(np.min(y_train)), float(np.max(y_train)))
    print("val target min/max  :", float(np.min(y_val)), float(np.max(y_val)))
    print("test target min/max :", float(np.min(y_test)), float(np.max(y_test)))

    print("\nPrediction clipping:")
    print("clip_log_range:", clip_log_range)
    if clip_log_range is not None:
        lo, hi = clip_log_range
        print("raw PM range after clip:", float(np.expm1(lo)), float(np.expm1(hi)))

    if args.target_mode == "log1p":
        mean_model_value = np.log1p(y_train).mean()
        val_mean_pred = np.expm1(np.full_like(y_val, mean_model_value, dtype=float))
        test_mean_pred = np.expm1(np.full_like(y_test, mean_model_value, dtype=float))
    else:
        mean_model_value = y_train.mean()
        val_mean_pred = np.full_like(y_val, mean_model_value, dtype=float)
        test_mean_pred = np.full_like(y_test, mean_model_value, dtype=float)

    baseline = {
        "val": compute_metrics(y_val, val_mean_pred),
        "test": compute_metrics(y_test, test_mean_pred),
    }

    print("\nMean baseline:")
    print(json.dumps(baseline, indent=2))

    # ============================================================
    # Tabular preprocessing fitted only on train
    # ============================================================

    preprocessor = build_tabular_preprocessor(numeric_cols, categorical_cols)

    X_tab_train = preprocessor.fit_transform(train_df[numeric_cols + categorical_cols])
    X_tab_val = preprocessor.transform(val_df[numeric_cols + categorical_cols])
    X_tab_test = preprocessor.transform(test_df[numeric_cols + categorical_cols])

    X_tab_train = np.asarray(X_tab_train, dtype=np.float32)
    X_tab_val = np.asarray(X_tab_val, dtype=np.float32)
    X_tab_test = np.asarray(X_tab_test, dtype=np.float32)

    print("\nTabular feature shape:")
    print("train:", X_tab_train.shape)
    print("val  :", X_tab_val.shape)
    print("test :", X_tab_test.shape)

    print("\nComputing train embedding scaler...")
    emb_mean, emb_std = compute_train_embedding_scaler(train_df, embeddings)

    train_ds = T7FusionDataset(train_df, embeddings, X_tab_train, args.target_mode, emb_mean, emb_std)
    val_ds = T7FusionDataset(val_df, embeddings, X_tab_val, args.target_mode, emb_mean, emb_std)
    test_ds = T7FusionDataset(test_df, embeddings, X_tab_test, args.target_mode, emb_mean, emb_std)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    embedding_dim = embeddings.shape[1]
    tabular_dim = X_tab_train.shape[1]

    model = GRUTabularFusionRegressor(
        embedding_dim=embedding_dim,
        tabular_dim=tabular_dim,
        hidden_dim=args.hidden_dim,
        tab_hidden_dim=args.tab_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_path = out_dir / "best_t7_gru_tabular_fusion.pt"

    history = []

    # ============================================================
    # Training
    # ============================================================

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip_norm=args.grad_clip_norm,
        )

        val_actual, val_pred, _, _, val_pred_model = predict(
            model,
            val_loader,
            device,
            args.target_mode,
            clip_log_range=clip_log_range,
        )

        test_actual, test_pred, _, _, test_pred_model = predict(
            model,
            test_loader,
            device,
            args.target_mode,
            clip_log_range=clip_log_range,
        )

        val_metrics = compute_metrics(val_actual, val_pred)
        test_metrics = compute_metrics(test_actual, test_pred)

        row = {
            "epoch": epoch,
            "train_loss": loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "val_pred_model_min": float(np.min(val_pred_model)),
            "val_pred_model_max": float(np.max(val_pred_model)),
            "test_pred_model_min": float(np.min(test_pred_model)),
            "test_pred_model_max": float(np.max(test_pred_model)),
            "clip_log_low": None if clip_log_range is None else clip_log_range[0],
            "clip_log_high": None if clip_log_range is None else clip_log_range[1],
        }

        history.append(row)

        print(
            f"[epoch {epoch:02d}/{args.epochs}] "
            f"loss={loss:.4f} | "
            f"VAL RMSE={val_metrics['RMSE']:.3f}, Spearman={val_metrics['Spearman']:.3f} | "
            f"TEST RMSE={test_metrics['RMSE']:.3f}, Spearman={test_metrics['Spearman']:.3f} | "
            f"test_log_pred=[{row['test_pred_model_min']:.3f}, {row['test_pred_model_max']:.3f}]"
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "baseline": baseline,
                    "embedding_mean": emb_mean,
                    "embedding_std": emb_std,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols,
                    "tabular_dim": int(tabular_dim),
                    "clip_log_range": clip_log_range,
                    "clip_log_pred": args.clip_log_pred,
                },
                best_path,
            )

    # ============================================================
    # Final evaluation from best checkpoint
    # ============================================================

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_actual, val_pred, val_seq_ids, val_row_ids, val_pred_model = predict(
        model,
        val_loader,
        device,
        args.target_mode,
        clip_log_range=clip_log_range,
    )

    test_actual, test_pred, test_seq_ids, test_row_ids, test_pred_model = predict(
        model,
        test_loader,
        device,
        args.target_mode,
        clip_log_range=clip_log_range,
    )

    summarize_predictions("VAL", val_actual, val_pred, val_pred_model)
    summarize_predictions("TEST", test_actual, test_pred, test_pred_model)

    final_metrics = {
        "baseline": baseline,
        "best_checkpoint": {
            "path": str(best_path),
            "epoch": checkpoint["epoch"],
        },
        "val": compute_metrics(val_actual, val_pred),
        "test": compute_metrics(test_actual, test_pred),
        "clip_log_range": clip_log_range,
        "clip_log_pred": args.clip_log_pred,
    }

    print("\nFINAL BEST MODEL METRICS:")
    print(json.dumps(final_metrics, indent=2))

    # ============================================================
    # Save outputs
    # ============================================================

    history_path = report_dir / "training_history_t7_gru_tabular_fusion.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    pred_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "sequence_id": val_seq_ids,
                    "target_row_id": val_row_ids,
                    "split": "val",
                    "actual_PM25": val_actual,
                    "predicted_PM25": val_pred,
                    "pred_model_space": val_pred_model,
                }
            ),
            pd.DataFrame(
                {
                    "sequence_id": test_seq_ids,
                    "target_row_id": test_row_ids,
                    "split": "test",
                    "actual_PM25": test_actual,
                    "predicted_PM25": test_pred,
                    "pred_model_space": test_pred_model,
                }
            ),
        ],
        ignore_index=True,
    )

    predictions_path = report_dir / "predictions_t7_gru_tabular_fusion.csv"
    pred_df.to_csv(predictions_path, index=False)

    metrics_path = report_dir / "metrics_t7_gru_tabular_fusion.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    config_path = report_dir / "config_t7_gru_tabular_fusion.json"
    config = {
        "args": vars(args),
        "sequence_manifest": str(sequence_manifest),
        "embeddings": str(embeddings_path),
        "embedding_shape": list(embeddings.shape),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "tabular_dim": int(tabular_dim),
        "clip_log_range": clip_log_range,
        "clip_log_pred": args.clip_log_pred,
        "rows": {
            "sequences": int(len(seq_df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    save_scatter(
        val_actual,
        val_pred,
        fig_dir / "scatter_val_t7_gru_tabular_fusion.png",
        "TRAQID T=7 GRU + Tabular Fusion VAL",
    )

    save_scatter(
        test_actual,
        test_pred,
        fig_dir / "scatter_test_t7_gru_tabular_fusion.png",
        "TRAQID T=7 GRU + Tabular Fusion TEST",
    )

    print("\nSaved:")
    print(" - best model:", best_path)
    print(" - history:", history_path)
    print(" - predictions:", predictions_path)
    print(" - metrics:", metrics_path)
    print(" - config:", config_path)
    print(" - figures:", fig_dir)


if __name__ == "__main__":
    main() 