"""Adapt a PM25Vision ResNet50 encoder to MUMMA roadside imagery.

Checkpoint selection uses one complete MUMMA date.  After selecting the epoch,
the encoder is reinitialized from the supplied checkpoint and refitted on all
five dates for exactly that number of epochs.  Only images and the declared
target are used; TRAQID is never read by this program.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-col", default="processed_frame_path")
    parser.add_argument("--target-col", default="sPM2")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--lens-col", default="lens_id")
    parser.add_argument("--lens", default="6")
    parser.add_argument("--status-col", default="preprocess_status")
    parser.add_argument("--selection-date", default="2026-02-05")
    parser.add_argument("--background-csv")
    parser.add_argument("--background-col", default="background_merra2_pm25_ug_m3")
    parser.add_argument(
        "--target-mode",
        choices=["direct", "local_increment"],
        default="local_increment",
    )
    parser.add_argument("--unfreeze", choices=["layer4", "layer3_layer4"], default="layer4")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def select_device(name: str):
    import torch

    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_encoder(path: Path, unfreeze: str):
    import torch
    from torch import nn
    from torchvision import models

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "encoder_state_dict" not in payload:
        raise ValueError(f"{path} does not contain encoder_state_dict")
    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for parameter in encoder.layer4.parameters():
        parameter.requires_grad = True
    if unfreeze == "layer3_layer4":
        for parameter in encoder.layer3.parameters():
            parameter.requires_grad = True
    return encoder, payload


def transforms():
    from torchvision import models, transforms as T

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    mean, std = weights.transforms().mean, weights.transforms().std
    train = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return train, weights.transforms()


def make_loader(frame, transform, target_mean, target_std, args, shuffle):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class Images(Dataset):
        def __len__(self):
            return len(frame)

        def __getitem__(self, index):
            row = frame.iloc[index]
            image = transform(Image.open(str(row[args.image_col])).convert("RGB"))
            target = (float(row["_adaptation_target"]) - target_mean) / target_std
            return image, torch.tensor(target, dtype=torch.float32)

    return DataLoader(
        Images(),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def train_model(train_frame, validation_frame, args, *, epochs, early_stop, seed):
    import torch
    from torch import nn

    seed_everything(seed)
    device = select_device(args.device)
    encoder, source = load_encoder(Path(args.init_checkpoint), args.unfreeze)
    head = nn.Sequential(nn.Dropout(0.25), nn.Linear(2048, 1))
    model = nn.Sequential(encoder, head).to(device)
    target_mean = float(train_frame["_adaptation_target"].mean())
    target_std = float(train_frame["_adaptation_target"].std(ddof=0))
    if not np.isfinite(target_std) or target_std <= 0:
        raise ValueError("Adaptation target has zero or invalid standard deviation")
    train_transform, evaluation_transform = transforms()
    train_loader = make_loader(
        train_frame, train_transform, target_mean, target_std, args, True
    )
    validation_loader = None
    if validation_frame is not None:
        validation_loader = make_loader(
            validation_frame,
            evaluation_transform,
            target_mean,
            target_std,
            args,
            False,
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.SmoothL1Loss(beta=1.0)
    best_state = None
    best_rmse = float("inf")
    best_epoch = epochs
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images.to(device)).squeeze(1)
            loss = loss_function(prediction, targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if validation_loader is not None:
            model.eval()
            predictions, truths = [], []
            with torch.inference_mode():
                for images, targets in validation_loader:
                    prediction = model(images.to(device)).squeeze(1).cpu().numpy()
                    predictions.extend(prediction * target_std + target_mean)
                    truths.extend(targets.numpy() * target_std + target_mean)
            predictions = np.asarray(predictions)
            truths = np.asarray(truths)
            rmse = float(mean_squared_error(truths, predictions) ** 0.5)
            row.update(
                {
                    "val_mae": float(mean_absolute_error(truths, predictions)),
                    "val_rmse": rmse,
                    "val_r2": float(r2_score(truths, predictions)),
                }
            )
            print(
                f"epoch={epoch:02d} train_loss={row['train_loss']:.5f} "
                f"val_MAE={row['val_mae']:.4f} val_RMSE={rmse:.4f} "
                f"val_R2={row['val_r2']:.4f}",
                flush=True,
            )
            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch
                stale = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                stale += 1
                if early_stop and stale >= args.patience:
                    print(f"Early stopping after epoch {epoch}.", flush=True)
                    history.append(row)
                    break
        else:
            print(
                f"refit_epoch={epoch:02d} train_loss={row['train_loss']:.5f}",
                flush=True,
            )
        history.append(row)
    if validation_loader is not None:
        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected")
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, target_mean, target_std, source


def main() -> int:
    import torch

    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, low_memory=False)
    required = {
        "sample_id", args.image_col, args.target_col, args.date_col,
        args.lens_col, args.status_col,
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Manifest is missing columns: {missing}")
    frame = frame[
        frame[args.lens_col].astype(str).eq(str(args.lens))
        & frame[args.status_col].astype(str).eq("success")
    ].copy()
    if frame["sample_id"].duplicated().any():
        raise ValueError("Expected one selected-lens image per sample_id")
    frame["_adaptation_target"] = pd.to_numeric(
        frame[args.target_col], errors="coerce"
    )
    if args.target_mode == "local_increment":
        if not args.background_csv:
            raise ValueError("--background-csv is required for local_increment")
        background = pd.read_csv(args.background_csv)
        required_background = {"sample_id", args.background_col}
        if missing := sorted(required_background - set(background.columns)):
            raise ValueError(f"Background table is missing columns: {missing}")
        background = background[["sample_id", args.background_col]].copy()
        frame = frame.merge(
            background, on="sample_id", how="left", validate="one_to_one"
        )
        frame["_adaptation_target"] -= pd.to_numeric(
            frame[args.background_col], errors="coerce"
        )
    frame = frame.dropna(subset=["_adaptation_target"]).reset_index(drop=True)
    dates = sorted(frame[args.date_col].astype(str).unique())
    if args.selection_date not in dates:
        raise ValueError(
            f"Selection date {args.selection_date} not found; available={dates}"
        )
    selection_mask = frame[args.date_col].astype(str).eq(args.selection_date)
    development = frame.loc[~selection_mask].reset_index(drop=True)
    validation = frame.loc[selection_mask].reset_index(drop=True)
    _, history, best_epoch, _, _, source = train_model(
        development,
        validation,
        args,
        epochs=args.epochs,
        early_stop=True,
        seed=args.seed,
    )
    history.to_csv(output / "selection_history.csv", index=False)
    final_model, refit_history, _, target_mean, target_std, _ = train_model(
        frame,
        None,
        args,
        epochs=best_epoch,
        early_stop=False,
        seed=args.seed + 1000,
    )
    refit_history.to_csv(output / "refit_history.csv", index=False)
    encoder = final_model[0]
    checkpoint = {
        "architecture": "resnet50",
        "pretraining": "ImageNet -> PM25Vision -> MUMMA",
        "source_checkpoint": str(args.init_checkpoint),
        "source_pretraining": source.get("pretraining"),
        "target": (
            args.target_col
            if args.target_mode == "direct"
            else f"{args.target_col} - {args.background_col}"
        ),
        "selection_date": args.selection_date,
        "selection_used_traqid": False,
        "refit_dates": dates,
        "refit_rows": len(frame),
        "unfreeze": args.unfreeze,
        "best_epoch": best_epoch,
        "target_mean": target_mean,
        "target_std": target_std,
        "encoder_state_dict": encoder.state_dict(),
    }
    torch.save(checkpoint, output / "best_checkpoint.pt")
    run = {
        key: value
        for key, value in checkpoint.items()
        if key != "encoder_state_dict"
    }
    run["warning"] = (
        "MUMMA target labels were used for encoder adaptation and must be "
        "disclosed in every downstream TRAQID report."
    )
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    print(f"Saved checkpoint: {output / 'best_checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
