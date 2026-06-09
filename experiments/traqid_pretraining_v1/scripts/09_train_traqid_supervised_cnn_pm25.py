from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)

    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        out["Pearson"] = float(pearsonr(y_true, y_pred).statistic)
        out["Spearman"] = float(spearmanr(y_true, y_pred).statistic)
    else:
        out["Pearson"] = float("nan")
        out["Spearman"] = float("nan")

    return out


def inverse_target(y: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "raw":
        return y
    if target_mode == "log1p":
        return np.expm1(y)
    raise ValueError(f"Unknown target_mode: {target_mode}")


def resolve_manifest_path(manifest_arg: str) -> Path:
    """
    Makes the script robust against accidentally passing the unsplit manifest.

    Required column:
        split_date_chrono

    If the given file does not contain it, this function checks whether
    traqid_paired_manifest_with_splits.csv exists in the same folder.
    """
    requested_path = Path(manifest_arg)

    if not requested_path.exists():
        raise FileNotFoundError(f"Manifest not found: {requested_path}")

    try:
        head = pd.read_csv(requested_path, nrows=5)
    except Exception as exc:
        raise RuntimeError(f"Could not read manifest: {requested_path}\nError: {exc}") from exc

    head.columns = [str(c).strip() for c in head.columns]

    if "split_date_chrono" in head.columns:
        return requested_path

    fallback_path = requested_path.parent / "traqid_paired_manifest_with_splits.csv"

    if fallback_path.exists():
        fallback_head = pd.read_csv(fallback_path, nrows=5)
        fallback_head.columns = [str(c).strip() for c in fallback_head.columns]

        if "split_date_chrono" in fallback_head.columns:
            print("\nWARNING:")
            print("Requested manifest does not contain `split_date_chrono`:")
            print(" ", requested_path)
            print("Automatically switching to:")
            print(" ", fallback_path)
            return fallback_path

    raise ValueError(
        "\nThe manifest does not contain `split_date_chrono`.\n"
        f"Requested path: {requested_path.resolve()}\n\n"
        "Use this file:\n"
        "experiments/traqid_pretraining_v1/data/processed/"
        "traqid_paired_manifest_with_splits.csv\n\n"
        "Or rerun:\n"
        "python experiments/traqid_pretraining_v1/scripts/04_create_traqid_splits.py\n"
    )


def load_manifest(manifest_arg: str, debug: bool = True) -> tuple[pd.DataFrame, Path]:
    manifest_path = resolve_manifest_path(manifest_arg)

    if debug:
        print("\nDEBUG manifest path:", manifest_path)
        print("DEBUG absolute path:", manifest_path.resolve())
        print("DEBUG exists:", manifest_path.exists())

    df = pd.read_csv(manifest_path)
    df.columns = [str(c).strip() for c in df.columns]

    if debug:
        print("DEBUG loaded columns:")
        print(list(df.columns))

    if "split_date_chrono" not in df.columns:
        raise ValueError(
            "Internal error: manifest was resolved, but `split_date_chrono` is still missing."
        )

    return df, manifest_path


# ============================================================
# Dataset
# ============================================================

class TRAQIDImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str,
        target_col: str,
        transform,
        target_mode: str = "log1p",
    ):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.target_col = target_col
        self.transform = transform
        self.target_mode = target_mode

    def __len__(self):
        return len(self.df)

    def encode_target(self, y: float) -> float:
        if self.target_mode == "raw":
            return float(y)
        if self.target_mode == "log1p":
            return float(np.log1p(y))
        raise ValueError(f"Unknown target_mode: {self.target_mode}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = row[self.image_col]
        y_raw = float(row[self.target_col])
        y = self.encode_target(y_raw)

        try:
            img = Image.open(path).convert("RGB")
            x = self.transform(img)
            ok = True
        except Exception:
            x = torch.zeros(3, 224, 224)
            ok = False

        return {
            "x": x,
            "y": torch.tensor([y], dtype=torch.float32),
            "y_raw": torch.tensor([y_raw], dtype=torch.float32),
            "ok": ok,
            "row_id": int(row["row_id"]),
        }


# ============================================================
# Model
# ============================================================

class CNNRegressor(nn.Module):
    def __init__(self, backbone_name: str = "mobilenetv2", dropout: float = 0.25):
        super().__init__()

        backbone_name = backbone_name.lower()

        if backbone_name == "mobilenetv2":
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
            self.transform = weights.transforms()
            self.features = model.features
            feature_dim = 1280

        elif backbone_name == "efficientnetb0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
            self.transform = weights.transforms()
            self.features = model.features
            feature_dim = 1280

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            self.transform = weights.transforms()
            self.features = nn.Sequential(*list(model.children())[:-2])
            feature_dim = 2048

        else:
            raise ValueError("backbone must be mobilenetv2, efficientnetb0, or resnet50")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        z = self.features(x)
        z = self.pool(z)
        y = self.head(z)
        return y


def set_backbone_trainable(model: CNNRegressor, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def unfreeze_last_mobilenet_blocks(model: CNNRegressor, n_last_blocks: int) -> None:
    for p in model.features.parameters():
        p.requires_grad = False

    children = list(model.features.children())

    n_last_blocks = max(1, min(n_last_blocks, len(children)))

    for block in children[-n_last_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# Training / Prediction
# ============================================================

@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    preds_model = []
    actual_raw = []
    row_ids = []

    for batch in loader:
        x = batch["x"].to(device)
        y_raw = batch["y_raw"].cpu().numpy().reshape(-1)
        row_id = batch["row_id"].cpu().numpy().reshape(-1)

        pred = model(x).detach().cpu().numpy().reshape(-1)

        preds_model.append(pred)
        actual_raw.append(y_raw)
        row_ids.append(row_id)

    preds_model = np.concatenate(preds_model)
    actual_raw = np.concatenate(actual_raw)
    row_ids = np.concatenate(row_ids)

    preds_raw = inverse_target(preds_model, target_mode)
    preds_raw = np.maximum(preds_raw, 0)

    return actual_raw, preds_raw, row_ids


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def save_scatter(y_true, y_pred, out_path: Path, title: str) -> None:
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
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )

    parser.add_argument("--view", default="front", choices=["front", "rear"])
    parser.add_argument("--target", default="PM2.5")
    parser.add_argument("--target-mode", default="log1p", choices=["raw", "log1p"])

    parser.add_argument(
        "--backbone",
        default="mobilenetv2",
        choices=["mobilenetv2", "efficientnetb0", "resnet50"],
    )

    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=6)

    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=4)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-env-plausible", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/models/supervised_cnn_pm25",
    )
    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/supervised_cnn_pm25",
    )
    parser.add_argument(
        "--fig-dir",
        default="experiments/traqid_pretraining_v1/figures/supervised_cnn_pm25",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    fig_dir = Path(args.fig_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df, resolved_manifest_path = load_manifest(args.manifest, debug=True)

    image_col = f"{args.view}_path"
    exists_col = f"{args.view}_exists"

    required = [
        "row_id",
        "created_at",
        "date",
        "split_date_chrono",
        image_col,
        exists_col,
        args.target,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    df = df[df[exists_col] == True].copy()

    if args.require_env_plausible:
        df = df[
            df["Temperature"].between(-10, 60)
            & df["Humidity"].between(0, 100)
        ].copy()

    if args.limit is not None:
        # This is only for sanity tests, not real evaluation.
        per_split = max(1, args.limit // 3)
        df = (
            df.groupby("split_date_chrono", group_keys=False)
            .head(per_split)
            .reset_index(drop=True)
        )

    train_df = df[df["split_date_chrono"] == "train"].copy()
    val_df = df[df["split_date_chrono"] == "val"].copy()
    test_df = df[df["split_date_chrono"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            "One of train/val/test splits is empty. "
            f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

    device = get_device(args.device)

    print("=" * 90)
    print("TRAQID SUPERVISED CNN PM2.5 TRAINING")
    print("=" * 90)
    print("Resolved manifest:", resolved_manifest_path)
    print("Rows:", len(df))
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    print("View:", args.view)
    print("Image column:", image_col)
    print("Target:", args.target)
    print("Target mode:", args.target_mode)
    print("Backbone:", args.backbone)
    print("Device:", device)
    print("Batch size:", args.batch_size)
    print("Head epochs:", args.head_epochs)
    print("Finetune epochs:", args.finetune_epochs)

    print("\nSplit dates:")
    print("train:", sorted(train_df["date"].unique().tolist()))
    print("val  :", sorted(val_df["date"].unique().tolist()))
    print("test :", sorted(test_df["date"].unique().tolist()))

    # ============================================================
    # Mean baseline
    # ============================================================

    train_y = train_df[args.target].astype(float).to_numpy()
    val_y = val_df[args.target].astype(float).to_numpy()
    test_y = test_df[args.target].astype(float).to_numpy()

    if args.target_mode == "log1p":
        mean_model_value = np.log1p(train_y).mean()
        val_mean_pred = np.expm1(np.full_like(val_y, mean_model_value, dtype=float))
        test_mean_pred = np.expm1(np.full_like(test_y, mean_model_value, dtype=float))
    else:
        mean_model_value = train_y.mean()
        val_mean_pred = np.full_like(val_y, mean_model_value, dtype=float)
        test_mean_pred = np.full_like(test_y, mean_model_value, dtype=float)

    baseline = {
        "val": compute_metrics(val_y, val_mean_pred),
        "test": compute_metrics(test_y, test_mean_pred),
    }

    print("\nMean baseline:")
    print(json.dumps(baseline, indent=2))

    # ============================================================
    # Model and loaders
    # ============================================================

    model = CNNRegressor(backbone_name=args.backbone, dropout=args.dropout)
    transform = model.transform
    model.to(device)

    total_params, trainable_params = count_trainable_params(model)
    print("\nInitial params:")
    print("Total params:", total_params)
    print("Trainable params before freezing:", trainable_params)

    train_ds = TRAQIDImageDataset(
        train_df,
        image_col,
        args.target,
        transform,
        args.target_mode,
    )
    val_ds = TRAQIDImageDataset(
        val_df,
        image_col,
        args.target,
        transform,
        args.target_mode,
    )
    test_ds = TRAQIDImageDataset(
        test_df,
        image_col,
        args.target,
        transform,
        args.target_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    criterion = nn.HuberLoss(delta=1.0)

    history = []
    best_val_rmse = float("inf")

    safe_target_name = args.target.replace(".", "_")
    best_state_path = out_dir / f"best_{args.backbone}_{args.view}_{safe_target_name}.pt"

    start_time = time.time()

    # ============================================================
    # Phase 1: train regression head only
    # ============================================================

    if args.head_epochs > 0:
        print("\nPHASE 1: Train regression head only")

        set_backbone_trainable(model, False)

        total_params, trainable_params = count_trainable_params(model)
        print("Trainable params after freezing backbone:", trainable_params, "/", total_params)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.head_lr,
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.head_epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

            val_actual, val_pred, _ = predict(model, val_loader, device, args.target_mode)
            test_actual, test_pred, _ = predict(model, test_loader, device, args.target_mode)

            val_metrics = compute_metrics(val_actual, val_pred)
            test_metrics = compute_metrics(test_actual, test_pred)

            row = {
                "phase": "head",
                "epoch": epoch,
                "train_loss": loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            history.append(row)

            print(
                f"[head {epoch:02d}/{args.head_epochs}] "
                f"loss={loss:.4f} | "
                f"VAL RMSE={val_metrics['RMSE']:.3f}, "
                f"Spearman={val_metrics['Spearman']:.3f} | "
                f"TEST RMSE={test_metrics['RMSE']:.3f}, "
                f"Spearman={test_metrics['Spearman']:.3f}"
            )

            if val_metrics["RMSE"] < best_val_rmse:
                best_val_rmse = val_metrics["RMSE"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "resolved_manifest": str(resolved_manifest_path),
                        "epoch": epoch,
                        "phase": "head",
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                    },
                    best_state_path,
                )

    # ============================================================
    # Phase 2: fine-tune last backbone blocks
    # ============================================================

    if args.finetune_epochs > 0:
        print("\nPHASE 2: Fine-tune last backbone blocks")

        if args.backbone == "mobilenetv2":
            unfreeze_last_mobilenet_blocks(model, args.unfreeze_last_blocks)
        else:
            set_backbone_trainable(model, True)

        total_params, trainable_params = count_trainable_params(model)
        print("Trainable params during fine-tune:", trainable_params, "/", total_params)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.finetune_lr,
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.finetune_epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

            val_actual, val_pred, _ = predict(model, val_loader, device, args.target_mode)
            test_actual, test_pred, _ = predict(model, test_loader, device, args.target_mode)

            val_metrics = compute_metrics(val_actual, val_pred)
            test_metrics = compute_metrics(test_actual, test_pred)

            row = {
                "phase": "finetune",
                "epoch": epoch,
                "train_loss": loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            history.append(row)

            print(
                f"[fine {epoch:02d}/{args.finetune_epochs}] "
                f"loss={loss:.4f} | "
                f"VAL RMSE={val_metrics['RMSE']:.3f}, "
                f"Spearman={val_metrics['Spearman']:.3f} | "
                f"TEST RMSE={test_metrics['RMSE']:.3f}, "
                f"Spearman={test_metrics['Spearman']:.3f}"
            )

            if val_metrics["RMSE"] < best_val_rmse:
                best_val_rmse = val_metrics["RMSE"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "resolved_manifest": str(resolved_manifest_path),
                        "epoch": epoch,
                        "phase": "finetune",
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                    },
                    best_state_path,
                )

    elapsed = time.time() - start_time

    if not best_state_path.exists():
        raise RuntimeError("No checkpoint was saved. Check whether epochs are set to zero.")

    # ============================================================
    # Final evaluation from best checkpoint
    # ============================================================

    checkpoint = torch.load(best_state_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_actual, val_pred, val_row_ids = predict(model, val_loader, device, args.target_mode)
    test_actual, test_pred, test_row_ids = predict(model, test_loader, device, args.target_mode)

    final_metrics = {
        "baseline": baseline,
        "best_checkpoint": {
            "path": str(best_state_path),
            "phase": checkpoint["phase"],
            "epoch": checkpoint["epoch"],
        },
        "val": compute_metrics(val_actual, val_pred),
        "test": compute_metrics(test_actual, test_pred),
    }

    print("\nFINAL BEST MODEL METRICS:")
    print(json.dumps(final_metrics, indent=2))

    # ============================================================
    # Save outputs
    # ============================================================

    hist_df = pd.DataFrame(history)
    hist_path = report_dir / f"training_history_{args.backbone}_{args.view}.csv"
    hist_df.to_csv(hist_path, index=False)

    val_pred_df = pd.DataFrame(
        {
            "row_id": val_row_ids,
            "split": "val",
            "actual_PM25": val_actual,
            "predicted_PM25": val_pred,
        }
    )

    test_pred_df = pd.DataFrame(
        {
            "row_id": test_row_ids,
            "split": "test",
            "actual_PM25": test_actual,
            "predicted_PM25": test_pred,
        }
    )

    pred_df = pd.concat([val_pred_df, test_pred_df], ignore_index=True)
    pred_path = report_dir / f"predictions_{args.backbone}_{args.view}.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = report_dir / f"metrics_{args.backbone}_{args.view}.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    config_path = report_dir / f"config_{args.backbone}_{args.view}.json"
    config = {
        "args": vars(args),
        "resolved_manifest": str(resolved_manifest_path),
        "elapsed_sec": elapsed,
        "rows": {
            "all": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    save_scatter(
        val_actual,
        val_pred,
        fig_dir / f"scatter_val_{args.backbone}_{args.view}.png",
        f"TRAQID VAL {args.view} {args.backbone}",
    )

    save_scatter(
        test_actual,
        test_pred,
        fig_dir / f"scatter_test_{args.backbone}_{args.view}.png",
        f"TRAQID TEST {args.view} {args.backbone}",
    )

    print("\nSaved:")
    print(" - best model:", best_state_path)
    print(" - history:", hist_path)
    print(" - predictions:", pred_path)
    print(" - metrics:", metrics_path)
    print(" - config:", config_path)
    print(" - figures:", fig_dir)


if __name__ == "__main__":
    main()