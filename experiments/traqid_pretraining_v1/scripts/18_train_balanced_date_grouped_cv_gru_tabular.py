
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import joblib
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


TARGET = "PM2.5"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str) -> torch.device:
    prefer = prefer.lower()

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but is not available.")
        return torch.device("mps")

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available.")
        return torch.device("cuda")

    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError("device must be auto, mps, cuda, or cpu")


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_index_sequence(value: str) -> list[int]:
    return [int(x) for x in str(value).split("|") if str(x).strip()]


def encode_target(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if mode == "raw":
        return y

    if mode == "log1p":
        return np.log1p(np.maximum(y, 0))

    raise ValueError(f"Unknown target mode: {mode}")


def inverse_target(
    y: np.ndarray,
    mode: str,
    clip_log_range: tuple[float, float] | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if mode == "raw":
        return y

    if mode == "log1p":
        if clip_log_range is not None:
            lo, hi = clip_log_range
            y = np.clip(y, lo, hi)
        return np.expm1(y)

    raise ValueError(f"Unknown target mode: {mode}")


def compute_clip_log_range(
    y_train_raw: np.ndarray,
    target_mode: str,
    enabled: bool,
    low_pct: float,
    high_pct: float,
) -> tuple[float, float] | None:
    if not enabled or target_mode != "log1p":
        return None

    train_log = np.log1p(np.maximum(np.asarray(y_train_raw, dtype=float), 0))

    lo = float(np.percentile(train_log, low_pct))
    hi = float(np.percentile(train_log, high_pct))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite prediction-clipping bounds.")

    if lo >= hi:
        raise ValueError(f"Invalid prediction-clipping range: {lo}, {hi}")

    return lo, hi


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)

    if finite.sum() == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
            "Pearson": float("nan"),
            "Spearman": float("nan"),
            "finite_fraction": 0.0,
        }

    yt = y_true[finite]
    yp = y_pred[finite]

    result = {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(math.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)),
        "finite_fraction": float(finite.mean()),
    }

    if len(np.unique(yt)) > 1 and len(np.unique(yp)) > 1:
        result["Pearson"] = float(pearsonr(yt, yp).statistic)
        result["Spearman"] = float(spearmanr(yt, yp).statistic)
    else:
        result["Pearson"] = float("nan")
        result["Spearman"] = float("nan")

    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "target_created_at" in df.columns:
        dt = pd.to_datetime(df["target_created_at"], errors="coerce")
    elif "created_at" in df.columns:
        dt = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        dt = pd.Series(pd.NaT, index=df.index)

    hour = dt.dt.hour.fillna(0).astype(float)
    dayofweek = dt.dt.dayofweek.fillna(0).astype(float)
    month = dt.dt.month.fillna(1).astype(float)

    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["dayofweek_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
    df["dayofweek_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)
    df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    return df


def build_tabular_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ]
    )

    transformers = []

    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("At least one numeric or categorical context column is required.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def save_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)

    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))

    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title(title)
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_training_curve(history: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_RMSE"], label="Validation RMSE")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ============================================================
# Input preparation
# ============================================================

def prepare_sequence_table(
    sequence_table_path: Path,
    context_manifest_path: Path,
) -> pd.DataFrame:
    sequences = pd.read_csv(sequence_table_path)
    sequences.columns = [str(c).strip() for c in sequences.columns]

    required_sequence_columns = {
        "sequence_id",
        "cv_fold",
        "cv_split",
        "date",
        "target_row_id",
        "target_created_at",
        "seq_start_row_id",
        "seq_end_row_id",
        TARGET,
    }

    missing = required_sequence_columns - set(sequences.columns)

    if missing:
        raise ValueError(
            f"Balanced sequence table is missing columns: {sorted(missing)}"
        )

    sequences["cv_fold"] = pd.to_numeric(
        sequences["cv_fold"], errors="raise"
    ).astype(int)

    sequences["target_row_id"] = pd.to_numeric(
        sequences["target_row_id"], errors="raise"
    ).astype(int)

    sequences["seq_start_row_id"] = pd.to_numeric(
        sequences["seq_start_row_id"], errors="raise"
    ).astype(int)

    sequences["seq_end_row_id"] = pd.to_numeric(
        sequences["seq_end_row_id"], errors="raise"
    ).astype(int)

    sequences["target_created_at"] = pd.to_datetime(
        sequences["target_created_at"], errors="coerce"
    )

    sequences = sequences.dropna(
        subset=["target_created_at", TARGET]
    ).copy()

    if "seq_embedding_indices" not in sequences.columns:
        sequences["seq_embedding_indices"] = sequences.apply(
            lambda row: "|".join(
                str(i)
                for i in range(
                    int(row["seq_start_row_id"]),
                    int(row["seq_end_row_id"]) + 1,
                )
            ),
            axis=1,
        )

    invalid_lengths = sequences["seq_embedding_indices"].map(
        lambda value: len(parse_index_sequence(value)) != 7
    )

    if invalid_lengths.any():
        examples = sequences.loc[
            invalid_lengths,
            [
                "sequence_id",
                "seq_start_row_id",
                "seq_end_row_id",
                "seq_embedding_indices",
            ],
        ].head(10)

        raise ValueError(
            "Some balanced sequences do not contain exactly seven indices.\n"
            + examples.to_string(index=False)
        )

    context = pd.read_csv(context_manifest_path)
    context.columns = [str(c).strip() for c in context.columns]

    if "row_id" not in context.columns:
        context["row_id"] = np.arange(len(context))

    context["row_id"] = pd.to_numeric(
        context["row_id"], errors="coerce"
    )

    context = context.dropna(subset=["row_id"]).copy()
    context["row_id"] = context["row_id"].astype(int)

    if context["row_id"].duplicated().any():
        duplicated = int(context["row_id"].duplicated().sum())
        print(
            f"WARNING: context manifest contains {duplicated} duplicate row_id values. "
            "Keeping the first row for each row_id."
        )
        context = context.drop_duplicates("row_id", keep="first")

    protected = {
        "row_id",
        "PM2.5",
        "PM10",
        "aqi",
        "target_value",
        "target_row_id",
        "target_created_at",
        "seq_start_row_id",
        "seq_end_row_id",
        "sequence_id",
        "cv_fold",
        "cv_split",
        "date",
    }

    context_columns = [
        column
        for column in context.columns
        if column not in protected
    ]

    context_for_merge = context[
        ["row_id", *context_columns]
    ].copy()

    context_for_merge = context_for_merge.rename(
        columns={"row_id": "target_row_id"}
    )

    merged = sequences.merge(
        context_for_merge,
        on="target_row_id",
        how="left",
        validate="many_to_one",
    )

    merged["target_value"] = pd.to_numeric(
        merged[TARGET], errors="coerce"
    )

    merged = merged.dropna(subset=["target_value"]).copy()
    merged = add_time_features(merged)

    return merged.reset_index(drop=True)


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
        self.tabular_array = np.asarray(tabular_array, dtype=np.float32)
        self.target_mode = target_mode
        self.emb_mean = np.asarray(emb_mean, dtype=np.float32)
        self.emb_std = np.asarray(emb_std, dtype=np.float32)

        self.seq_indices = [
            parse_index_sequence(value)
            for value in self.df["seq_embedding_indices"].tolist()
        ]

        self.y_raw = self.df["target_value"].astype(float).to_numpy(
            dtype=np.float32
        )

        self.y_model = encode_target(
            self.y_raw,
            self.target_mode,
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        indices = self.seq_indices[idx]

        x_seq = np.asarray(
            self.embeddings[indices],
            dtype=np.float32,
        )

        x_seq = (x_seq - self.emb_mean) / self.emb_std
        x_tab = self.tabular_array[idx]

        return {
            "x_seq": torch.tensor(x_seq, dtype=torch.float32),
            "x_tab": torch.tensor(x_tab, dtype=torch.float32),
            "y": torch.tensor([self.y_model[idx]], dtype=torch.float32),
            "y_raw": torch.tensor([self.y_raw[idx]], dtype=torch.float32),
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

        gru_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, tab_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                gru_output_dim + tab_hidden_dim,
                fusion_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_tab: torch.Tensor,
    ) -> torch.Tensor:
        sequence_output, _ = self.gru(x_seq)
        sequence_vector = sequence_output[:, -1, :]

        tabular_vector = self.tabular_mlp(x_tab)
        fused = torch.cat([sequence_vector, tabular_vector], dim=1)

        return self.head(fused)


# ============================================================
# Training helpers
# ============================================================

def compute_train_embedding_scaler(
    train_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used_indices: set[int] = set()

    for value in train_df["seq_embedding_indices"]:
        used_indices.update(parse_index_sequence(value))

    used = sorted(used_indices)

    if not used:
        raise ValueError("No training embedding indices were found.")

    if min(used) < 0 or max(used) >= embeddings.shape[0]:
        raise IndexError(
            f"Training sequences require embedding rows {min(used)} to {max(used)}, "
            f"but embedding array contains {embeddings.shape[0]} rows."
        )

    values = np.asarray(
        embeddings[used],
        dtype=np.float32,
    )

    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float | None,
) -> float:
    model.train()

    total_loss = 0.0
    total_rows = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x_seq = batch["x_seq"].to(device)
        x_tab = batch["x_tab"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(x_seq, x_tab)
        loss = criterion(prediction, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite training loss.")

        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
            )

        optimizer.step()

        batch_size = x_seq.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size

    return total_loss / max(total_rows, 1)


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mode: str,
    clip_log_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    model_space_predictions = []
    actual_raw = []

    for batch in loader:
        x_seq = batch["x_seq"].to(device)
        x_tab = batch["x_tab"].to(device)

        prediction = model(x_seq, x_tab)

        model_space_predictions.append(
            prediction.detach().cpu().numpy().reshape(-1)
        )

        actual_raw.append(
            batch["y_raw"].cpu().numpy().reshape(-1)
        )

    model_space_predictions = np.concatenate(
        model_space_predictions
    )

    actual_raw = np.concatenate(actual_raw)

    predicted_raw = inverse_target(
        model_space_predictions,
        target_mode,
        clip_log_range=clip_log_range,
    )

    predicted_raw = np.maximum(predicted_raw, 0)

    return actual_raw, predicted_raw, model_space_predictions


def make_loader(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    tabular_array: np.ndarray,
    target_mode: str,
    emb_mean: np.ndarray,
    emb_std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = T7FusionDataset(
        df=df,
        embeddings=embeddings,
        tabular_array=tabular_array,
        target_mode=target_mode,
        emb_mean=emb_mean,
        emb_std=emb_std,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def geometric_mean_baseline(
    y_train: np.ndarray,
    output_length: int,
    target_mode: str,
) -> np.ndarray:
    if target_mode == "log1p":
        value = float(
            np.expm1(np.log1p(np.maximum(y_train, 0)).mean())
        )
    else:
        value = float(np.mean(y_train))

    return np.full(output_length, value, dtype=float)


# ============================================================
# Fold training
# ============================================================

def train_fold(
    fold: int,
    fold_df: pd.DataFrame,
    embeddings: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    device: torch.device,
    args: argparse.Namespace,
    fold_model_dir: Path,
    fold_report_dir: Path,
    fold_figure_dir: Path,
) -> dict[str, pd.DataFrame | dict]:
    fold_seed = args.seed + fold
    set_seed(fold_seed)

    train_df = fold_df[
        fold_df["cv_split"] == "train"
    ].copy().reset_index(drop=True)

    val_df = fold_df[
        fold_df["cv_split"] == "val"
    ].copy().reset_index(drop=True)

    test_df = fold_df[
        fold_df["cv_split"] == "test"
    ].copy().reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"Fold {fold} has an empty split: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    fold_model_dir.mkdir(parents=True, exist_ok=True)
    fold_report_dir.mkdir(parents=True, exist_ok=True)
    fold_figure_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"BALANCED DATE-GROUPED CV FOLD {fold}")
    print("=" * 100)
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print("Train dates:", sorted(train_df["date"].astype(str).unique()))
    print("Validation dates:", sorted(val_df["date"].astype(str).unique()))
    print("Test dates:", sorted(test_df["date"].astype(str).unique()))

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

    baseline_val_pred = geometric_mean_baseline(
        y_train,
        len(y_val),
        args.target_mode,
    )

    baseline_test_pred = geometric_mean_baseline(
        y_train,
        len(y_test),
        args.target_mode,
    )

    baseline = {
        "val": compute_metrics(y_val, baseline_val_pred),
        "test": compute_metrics(y_test, baseline_test_pred),
    }

    preprocessor = build_tabular_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    context_columns = [*numeric_cols, *categorical_cols]

    x_tab_train = preprocessor.fit_transform(
        train_df[context_columns]
    )

    x_tab_val = preprocessor.transform(
        val_df[context_columns]
    )

    x_tab_test = preprocessor.transform(
        test_df[context_columns]
    )

    x_tab_train = np.asarray(x_tab_train, dtype=np.float32)
    x_tab_val = np.asarray(x_tab_val, dtype=np.float32)
    x_tab_test = np.asarray(x_tab_test, dtype=np.float32)

    emb_mean, emb_std = compute_train_embedding_scaler(
        train_df,
        embeddings,
    )

    train_loader = make_loader(
        df=train_df,
        embeddings=embeddings,
        tabular_array=x_tab_train,
        target_mode=args.target_mode,
        emb_mean=emb_mean,
        emb_std=emb_std,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        df=val_df,
        embeddings=embeddings,
        tabular_array=x_tab_val,
        target_mode=args.target_mode,
        emb_mean=emb_mean,
        emb_std=emb_std,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = make_loader(
        df=test_df,
        embeddings=embeddings,
        tabular_array=x_tab_test,
        target_mode=args.target_mode,
        emb_mean=emb_mean,
        emb_std=emb_std,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = GRUTabularFusionRegressor(
        embedding_dim=embeddings.shape[1],
        tabular_dim=x_tab_train.shape[1],
        hidden_dim=args.hidden_dim,
        tab_hidden_dim=args.tab_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    criterion = nn.HuberLoss(delta=args.huber_delta)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    best_checkpoint_path = (
        fold_model_dir / "best_gru_tabular_fusion.pt"
    )

    history_rows = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=args.grad_clip_norm,
        )

        val_actual, val_predicted, val_model_space = predict(
            model=model,
            loader=val_loader,
            device=device,
            target_mode=args.target_mode,
            clip_log_range=clip_log_range,
        )

        val_metrics = compute_metrics(
            val_actual,
            val_predicted,
        )

        history_rows.append(
            {
                "cv_fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                **{
                    f"val_{key}": value
                    for key, value in val_metrics.items()
                },
                "val_pred_model_min": float(np.min(val_model_space)),
                "val_pred_model_max": float(np.max(val_model_space)),
            }
        )

        print(
            f"Fold {fold} | Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.5f} | "
            f"val_RMSE={val_metrics['RMSE']:.4f} | "
            f"val_MAE={val_metrics['MAE']:.4f} | "
            f"val_R2={val_metrics['R2']:.4f}"
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "cv_fold": fold,
                    "fold_seed": fold_seed,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "baseline": baseline,
                    "embedding_mean": emb_mean,
                    "embedding_std": emb_std,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols,
                    "tabular_dim": int(x_tab_train.shape[1]),
                    "clip_log_range": clip_log_range,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print(
                f"Fold {fold}: early stopping after epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    val_actual, val_predicted, val_model_space = predict(
        model=model,
        loader=val_loader,
        device=device,
        target_mode=args.target_mode,
        clip_log_range=clip_log_range,
    )

    test_actual, test_predicted, test_model_space = predict(
        model=model,
        loader=test_loader,
        device=device,
        target_mode=args.target_mode,
        clip_log_range=clip_log_range,
    )

    val_metrics = compute_metrics(
        val_actual,
        val_predicted,
    )

    test_metrics = compute_metrics(
        test_actual,
        test_predicted,
    )

    metric_rows = [
        {
            "cv_fold": fold,
            "split": "val",
            **val_metrics,
            "best_epoch": best_epoch,
            "best_val_RMSE": best_val_rmse,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_PM2.5_mean": float(np.mean(y_train)),
            "val_PM2.5_mean": float(np.mean(y_val)),
            "test_PM2.5_mean": float(np.mean(y_test)),
            "train_dates": "|".join(
                sorted(train_df["date"].astype(str).unique())
            ),
            "val_dates": "|".join(
                sorted(val_df["date"].astype(str).unique())
            ),
            "test_dates": "|".join(
                sorted(test_df["date"].astype(str).unique())
            ),
            "fold_seed": fold_seed,
        },
        {
            "cv_fold": fold,
            "split": "test",
            **test_metrics,
            "best_epoch": best_epoch,
            "best_val_RMSE": best_val_rmse,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_PM2.5_mean": float(np.mean(y_train)),
            "val_PM2.5_mean": float(np.mean(y_val)),
            "test_PM2.5_mean": float(np.mean(y_test)),
            "train_dates": "|".join(
                sorted(train_df["date"].astype(str).unique())
            ),
            "val_dates": "|".join(
                sorted(val_df["date"].astype(str).unique())
            ),
            "test_dates": "|".join(
                sorted(test_df["date"].astype(str).unique())
            ),
            "fold_seed": fold_seed,
        },
    ]

    metrics_df = pd.DataFrame(metric_rows)
    history_df = pd.DataFrame(history_rows)

    def make_prediction_frame(
        source_df: pd.DataFrame,
        split_name: str,
        actual: np.ndarray,
        predicted: np.ndarray,
        model_space: np.ndarray,
    ) -> pd.DataFrame:
        output = source_df[
            [
                "sequence_id",
                "cv_fold",
                "cv_split",
                "date",
                "target_row_id",
                "target_created_at",
                "seq_start_row_id",
                "seq_end_row_id",
            ]
        ].copy()

        output["split"] = split_name
        output["actual_PM2.5"] = actual
        output["predicted_PM2.5"] = predicted
        output["pred_model_space"] = model_space
        output["prediction_error"] = predicted - actual
        output["absolute_error"] = np.abs(
            output["prediction_error"]
        )

        return output

    val_predictions_df = make_prediction_frame(
        source_df=val_df,
        split_name="val",
        actual=val_actual,
        predicted=val_predicted,
        model_space=val_model_space,
    )

    test_predictions_df = make_prediction_frame(
        source_df=test_df,
        split_name="test",
        actual=test_actual,
        predicted=test_predicted,
        model_space=test_model_space,
    )

    metrics_df.to_csv(
        fold_report_dir / "metrics.csv",
        index=False,
    )

    history_df.to_csv(
        fold_report_dir / "training_history.csv",
        index=False,
    )

    val_predictions_df.to_csv(
        fold_report_dir / "val_predictions.csv",
        index=False,
    )

    test_predictions_df.to_csv(
        fold_report_dir / "test_predictions.csv",
        index=False,
    )

    joblib.dump(
        preprocessor,
        fold_model_dir / "tabular_preprocessor.joblib",
    )

    np.savez_compressed(
        fold_model_dir / "embedding_scaler.npz",
        mean=emb_mean,
        std=emb_std,
    )

    fold_config = {
        "cv_fold": fold,
        "fold_seed": fold_seed,
        "device": str(device),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "tabular_dim": int(x_tab_train.shape[1]),
        "embedding_dim": int(embeddings.shape[1]),
        "clip_log_range": clip_log_range,
        "best_epoch": best_epoch,
        "best_val_RMSE": best_val_rmse,
        "baseline": baseline,
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "dates": {
            "train": sorted(
                train_df["date"].astype(str).unique().tolist()
            ),
            "val": sorted(
                val_df["date"].astype(str).unique().tolist()
            ),
            "test": sorted(
                test_df["date"].astype(str).unique().tolist()
            ),
        },
    }

    (
        fold_report_dir / "config.json"
    ).write_text(
        json.dumps(fold_config, indent=2),
        encoding="utf-8",
    )

    save_scatter(
        y_true=val_actual,
        y_pred=val_predicted,
        out_path=fold_figure_dir / "scatter_val.png",
        title=f"Fold {fold} Validation",
    )

    save_scatter(
        y_true=test_actual,
        y_pred=test_predicted,
        out_path=fold_figure_dir / "scatter_test.png",
        title=f"Fold {fold} Test",
    )

    save_training_curve(
        history=history_df,
        out_path=fold_figure_dir / "training_curve.png",
        title=f"Fold {fold} Training",
    )

    print("\nFold metrics:")
    print(metrics_df.to_string(index=False))

    return {
        "metrics": metrics_df,
        "history": history_df,
        "val_predictions": val_predictions_df,
        "test_predictions": test_predictions_df,
        "baseline": baseline,
    }


# ============================================================
# Aggregate reporting
# ============================================================

def make_date_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (fold, date), group in predictions.groupby(
        ["cv_fold", "date"],
        sort=True,
    ):
        metrics = compute_metrics(
            group["actual_PM2.5"].to_numpy(),
            group["predicted_PM2.5"].to_numpy(),
        )

        rows.append(
            {
                "cv_fold": int(fold),
                "date": str(date),
                "rows": int(len(group)),
                "actual_mean": float(
                    group["actual_PM2.5"].mean()
                ),
                "predicted_mean": float(
                    group["predicted_PM2.5"].mean()
                ),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def make_aggregate_metrics(
    fold_test_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for metric in [
        "MAE",
        "RMSE",
        "R2",
        "Pearson",
        "Spearman",
    ]:
        values = fold_test_metrics[metric].astype(float)

        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
                "folds": int(len(values)),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-table",
        default=(
            "experiments/traqid_pretraining_v1/"
            "data/processed/balanced_date_grouped_T7_cv/"
            "traqid_balanced_date_T7_all_folds.csv"
        ),
    )

    parser.add_argument(
        "--context-manifest",
        default=(
            "experiments/traqid_pretraining_v1/"
            "data/processed/traqid_paired_manifest_with_splits.csv"
        ),
    )

    parser.add_argument(
        "--embeddings",
        default=(
            "experiments/traqid_pretraining_v1/"
            "embeddings/paper_style_cnn/"
            "traqid_paper_resnet50_front_rear_concat_gap_features.npy"
        ),
    )

    parser.add_argument(
        "--target-mode",
        default="log1p",
        choices=["raw", "log1p"],
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tab-hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--huber-delta", type=float, default=1.0)

    parser.add_argument(
        "--numeric-cols",
        default=(
            "Temperature,Humidity,hour_sin,hour_cos,"
            "dayofweek_sin,dayofweek_cos,month_sin,month_cos"
        ),
    )

    parser.add_argument(
        "--categorical-cols",
        default="Season,Day_or_Night",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument(
        "--clip-log-pred",
        dest="clip_log_pred",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no-clip-log-pred",
        dest="clip_log_pred",
        action="store_false",
    )

    parser.add_argument(
        "--clip-low-percentile",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--clip-high-percentile",
        type=float,
        default=99.5,
    )

    parser.add_argument(
        "--model-dir",
        default=(
            "experiments/traqid_pretraining_v1/"
            "models/pm25_balanced_date_grouped_T7_gru_tabular_cv"
        ),
    )

    parser.add_argument(
        "--report-dir",
        default=(
            "experiments/traqid_pretraining_v1/"
            "reports/pm25_balanced_date_grouped_T7_gru_tabular_cv"
        ),
    )

    parser.add_argument(
        "--figure-dir",
        default=(
            "experiments/traqid_pretraining_v1/"
            "figures/pm25_balanced_date_grouped_T7_gru_tabular_cv"
        ),
    )

    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("At least two folds are required.")

    set_seed(args.seed)
    device = get_device(args.device)

    sequence_table_path = Path(args.sequence_table)
    context_manifest_path = Path(args.context_manifest)
    embeddings_path = Path(args.embeddings)

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    figure_dir = Path(args.figure_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_sequence_table(
        sequence_table_path=sequence_table_path,
        context_manifest_path=context_manifest_path,
    )

    embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    maximum_index = max(
        max(parse_index_sequence(value))
        for value in data["seq_embedding_indices"]
    )

    if maximum_index >= embeddings.shape[0]:
        raise IndexError(
            f"Sequences require embedding index {maximum_index}, "
            f"but embedding array has {embeddings.shape[0]} rows."
        )

    numeric_cols = [
        column.strip()
        for column in args.numeric_cols.split(",")
        if column.strip()
    ]

    categorical_cols = [
        column.strip()
        for column in args.categorical_cols.split(",")
        if column.strip()
    ]

    missing_context = [
        column
        for column in [*numeric_cols, *categorical_cols]
        if column not in data.columns
    ]

    if missing_context:
        raise ValueError(
            f"Requested context columns are missing: {missing_context}"
        )

    available_folds = sorted(
        data["cv_fold"].astype(int).unique().tolist()
    )

    expected_folds = list(range(1, args.folds + 1))

    if available_folds != expected_folds:
        raise ValueError(
            f"Expected folds {expected_folds}, found {available_folds}."
        )

    print("=" * 100)
    print("BALANCED DATE-GROUPED T=7 GRU + TABULAR FUSION CV")
    print("=" * 100)
    print("Sequence table:", sequence_table_path)
    print("Context manifest:", context_manifest_path)
    print("Embeddings:", embeddings_path)
    print("Embedding shape:", embeddings.shape)
    print("Prepared sequence rows:", len(data))
    print("Folds:", available_folds)
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)
    print("Target mode:", args.target_mode)
    print("Device:", device)

    metric_parts = []
    history_parts = []
    val_prediction_parts = []
    test_prediction_parts = []
    baseline_records = []

    for fold in expected_folds:
        fold_data = data[
            data["cv_fold"] == fold
        ].copy()

        result = train_fold(
            fold=fold,
            fold_df=fold_data,
            embeddings=embeddings,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            device=device,
            args=args,
            fold_model_dir=model_dir / f"fold_{fold}",
            fold_report_dir=report_dir / f"fold_{fold}",
            fold_figure_dir=figure_dir / f"fold_{fold}",
        )

        metric_parts.append(result["metrics"])
        history_parts.append(result["history"])
        val_prediction_parts.append(
            result["val_predictions"]
        )
        test_prediction_parts.append(
            result["test_predictions"]
        )

        baseline_records.append(
            {
                "cv_fold": fold,
                "val": result["baseline"]["val"],
                "test": result["baseline"]["test"],
            }
        )

        if device.type == "mps":
            torch.mps.empty_cache()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_metrics = pd.concat(
        metric_parts,
        ignore_index=True,
    )

    all_history = pd.concat(
        history_parts,
        ignore_index=True,
    )

    all_val_predictions = pd.concat(
        val_prediction_parts,
        ignore_index=True,
    )

    all_test_predictions = pd.concat(
        test_prediction_parts,
        ignore_index=True,
    )

    fold_test_metrics = all_metrics[
        all_metrics["split"] == "test"
    ].copy().sort_values("cv_fold")

    aggregate_metrics = make_aggregate_metrics(
        fold_test_metrics
    )

    pooled_metrics = compute_metrics(
        all_test_predictions["actual_PM2.5"].to_numpy(),
        all_test_predictions["predicted_PM2.5"].to_numpy(),
    )

    pooled_metrics_df = pd.DataFrame(
        [
            {
                "evaluation": "pooled_held_out_date_predictions",
                "rows": int(len(all_test_predictions)),
                "unique_dates": int(
                    all_test_predictions["date"].nunique()
                ),
                **pooled_metrics,
            }
        ]
    )

    date_metrics = make_date_metrics(
        all_test_predictions
    )

    duplicate_test_sequences = int(
        all_test_predictions["sequence_id"].duplicated().sum()
    )

    test_coverage = {
        "rows": int(len(all_test_predictions)),
        "unique_sequence_ids": int(
            all_test_predictions["sequence_id"].nunique()
        ),
        "duplicate_sequence_ids": duplicate_test_sequences,
        "unique_dates": int(
            all_test_predictions["date"].nunique()
        ),
    }

    all_metrics.to_csv(
        report_dir / "all_fold_metrics.csv",
        index=False,
    )

    aggregate_metrics.to_csv(
        report_dir / "aggregate_test_metrics.csv",
        index=False,
    )

    pooled_metrics_df.to_csv(
        report_dir / "pooled_test_metrics.csv",
        index=False,
    )

    all_history.to_csv(
        report_dir / "all_fold_training_history.csv",
        index=False,
    )

    all_val_predictions.to_csv(
        report_dir / "all_fold_val_predictions.csv",
        index=False,
    )

    all_test_predictions.to_csv(
        report_dir / "all_fold_test_predictions.csv",
        index=False,
    )

    date_metrics.to_csv(
        report_dir / "test_metrics_by_date.csv",
        index=False,
    )

    (
        report_dir / "baseline_metrics.json"
    ).write_text(
        json.dumps(baseline_records, indent=2),
        encoding="utf-8",
    )

    final_summary = {
        "model": "balanced_date_grouped_T7_GRU_tabular_fusion",
        "target": TARGET,
        "folds": args.folds,
        "aggregate_test_metrics": aggregate_metrics.to_dict(
            orient="records"
        ),
        "pooled_test_metrics": pooled_metrics,
        "test_coverage": test_coverage,
        "notes": [
            "Complete acquisition dates were assigned to grouped folds.",
            "Sequences were created independently inside train, validation, and test date groups.",
            "Tabular preprocessing and embedding normalization were fitted using training rows only.",
            "Test data were evaluated only after selecting the best validation checkpoint.",
        ],
    }

    (
        report_dir / "final_summary.json"
    ).write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )

    markdown = [
        "# Balanced Date-Grouped T7 GRU + Tabular Fusion",
        "",
        "## Per-fold held-out test metrics",
        "",
        fold_test_metrics.to_markdown(index=False),
        "",
        "## Mean and standard deviation across folds",
        "",
        aggregate_metrics.to_markdown(index=False),
        "",
        "## Pooled held-out-date metrics",
        "",
        pooled_metrics_df.to_markdown(index=False),
        "",
        "## Held-out date-level metrics",
        "",
        date_metrics.to_markdown(index=False),
        "",
        "## Evaluation protocol",
        "",
        "- Complete acquisition dates were used as grouping units.",
        "- T=7 sequences never crossed date or split boundaries.",
        "- Train, validation, and test sequence rows have zero overlap.",
        "- Tabular and embedding scalers were fitted on training data only.",
        "- Test predictions were generated only from the best validation checkpoint.",
        "",
    ]

    (
        report_dir / "final_summary.md"
    ).write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    save_scatter(
        y_true=all_test_predictions["actual_PM2.5"].to_numpy(),
        y_pred=all_test_predictions["predicted_PM2.5"].to_numpy(),
        out_path=figure_dir / "scatter_pooled_test.png",
        title="Balanced Date-Grouped CV: Pooled Held-Out Test",
    )

    print("\n" + "=" * 100)
    print("BALANCED DATE-GROUPED T7 GRU + TABULAR CV COMPLETE")
    print("=" * 100)

    print("\nPER-FOLD TEST METRICS")
    print(
        fold_test_metrics[
            [
                "cv_fold",
                "R2",
                "RMSE",
                "MAE",
                "Pearson",
                "Spearman",
                "test_rows",
                "test_PM2.5_mean",
                "best_epoch",
            ]
        ].to_string(index=False)
    )

    print("\nAGGREGATE TEST METRICS")
    print(aggregate_metrics.to_string(index=False))

    print("\nPOOLED TEST METRICS")
    print(pooled_metrics_df.to_string(index=False))

    print("\nTEST COVERAGE")
    print(json.dumps(test_coverage, indent=2))

    print("\nSaved:")
    print(" - all fold metrics:", report_dir / "all_fold_metrics.csv")
    print(" - aggregate metrics:", report_dir / "aggregate_test_metrics.csv")
    print(" - pooled metrics:", report_dir / "pooled_test_metrics.csv")
    print(" - test predictions:", report_dir / "all_fold_test_predictions.csv")
    print(" - date metrics:", report_dir / "test_metrics_by_date.csv")
    print(" - final summary:", report_dir / "final_summary.md")
    print(" - pooled scatter:", figure_dir / "scatter_pooled_test.png")


if __name__ == "__main__":
    main()
