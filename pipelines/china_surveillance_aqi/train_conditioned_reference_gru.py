"""Train a leakage-safe reference-context plus background-conditioned image GRU.

The context branch predicts the local increment with ExtraTrees.  Training-row
context predictions are cross-fitted by training site before defining the
neural correction target.  A FiLM-conditioned GRU then learns the remaining
error from T=7 image embeddings and background/time context.  The neural
correction is bounded and receives a scalar shrinkage weight selected on the
outer validation site only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from roadside_pm.features.images.resnet import select_device


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
        "pearson": float(pearsonr(actual, predicted)[0]),
        "spearman": float(spearmanr(actual, predicted)[0]),
    }


def context_features(frame: pd.DataFrame) -> np.ndarray:
    timestamp = pd.to_datetime(frame["target_time"], errors="raise")
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    day = timestamp.dt.dayofyear
    return np.column_stack(
        [
            frame["background_reference_pm25"].to_numpy(dtype=float),
            frame["background_reference_site_count"].to_numpy(dtype=float),
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * day / 365.25),
            np.cos(2 * np.pi * day / 365.25),
        ]
    ).astype("float32")


def extra_trees(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=10,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )


def crossfit_context(
    train: pd.DataFrame, features: np.ndarray, target: np.ndarray, seed: int
) -> np.ndarray:
    predictions = np.full(len(train), np.nan, dtype="float32")
    sites = train["site"].astype(str).to_numpy()
    unique_sites = sorted(set(sites))
    if len(unique_sites) < 3:
        raise ValueError("Training-site cross-fitting requires at least three training sites")
    for index, site in enumerate(unique_sites):
        held = sites == site
        model = extra_trees(seed + index)
        model.fit(features[~held], target[~held])
        predictions[held] = model.predict(features[held]).astype("float32")
    if np.isnan(predictions).any():
        raise RuntimeError("Incomplete inner site-cross-fitted predictions")
    return predictions


class SequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        embeddings: np.ndarray,
        context: np.ndarray,
        correction_target: np.ndarray,
        base_pm25: np.ndarray,
    ):
        self.frame = frame.reset_index(drop=True)
        self.embeddings = embeddings
        self.context = context.astype("float32")
        self.target = correction_target.astype("float32")
        self.base = base_pm25.astype("float32")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        positions = [int(value) for value in str(self.frame.iloc[index]["embedding_rows"]).split("|")]
        return (
            torch.from_numpy(self.embeddings[positions].astype("float32")),
            torch.from_numpy(self.context[index]),
            torch.tensor(self.target[index], dtype=torch.float32),
            torch.tensor(self.base[index], dtype=torch.float32),
            torch.tensor(float(self.frame.iloc[index]["target_pm25"]), dtype=torch.float32),
            torch.tensor(int(self.frame.iloc[index]["sequence_id"]), dtype=torch.long),
        )


class ConditionedGRU(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        max_correction: float,
    ):
        super().__init__()
        self.max_correction = max_correction
        self.image_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 64), nn.GELU()
        )
        self.film = nn.Linear(64, projection_dim * 2)
        self.gru = nn.GRU(projection_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 64),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + 64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        image = self.image_projection(images)
        encoded_context = self.context_encoder(context)
        gamma, beta = self.film(encoded_context).chunk(2, dim=-1)
        image = image * (1.0 + 0.1 * torch.tanh(gamma[:, None, :])) + 0.1 * beta[:, None, :]
        sequence, _ = self.gru(image)
        raw = self.head(torch.cat([sequence[:, -1], encoded_context], dim=-1)).squeeze(-1)
        return self.max_correction * torch.tanh(raw / self.max_correction)


def choose_alpha(
    actual: np.ndarray, base: np.ndarray, correction: np.ndarray, step: float
) -> tuple[float, float]:
    grid = np.arange(0.0, 1.0 + step / 2, step)
    errors = [mean_squared_error(actual, base + alpha * correction) ** 0.5 for alpha in grid]
    best = int(np.argmin(errors))
    return float(grid[best]), float(errors[best])


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    corrections, bases, actuals, ids = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for image, context, _, base, actual, sequence_id in loader:
            corrections.append(model(image.to(device), context.to(device)).cpu().numpy())
            bases.append(base.numpy())
            actuals.append(actual.numpy())
            ids.append(sequence_id.numpy())
    return tuple(np.concatenate(values) for values in (corrections, bases, actuals, ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-correction", type=float, default=15.0)
    parser.add_argument("--correction-l2", type=float, default=0.005)
    parser.add_argument("--huber-beta", type=float, default=3.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.02)
    parser.add_argument(
        "--protocol-label",
        default="partition_holdout_with_training_partition_crossfit_context",
        help="Descriptive evaluation protocol stored in metrics.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = pd.read_csv(args.sequence_manifest)
    embeddings = np.load(args.embeddings, mmap_mode="r")
    parts = {name: frame.loc[frame["split"].eq(name)].reset_index(drop=True) for name in ("train", "val", "test")}
    if min(map(len, parts.values())) == 0:
        raise ValueError({name: len(value) for name, value in parts.items()})

    features = {name: context_features(part) for name, part in parts.items()}
    train_increment = parts["train"]["target_local_increment"].to_numpy(dtype="float32")
    train_context_oof = crossfit_context(parts["train"], features["train"], train_increment, args.seed)
    context_model = extra_trees(args.seed)
    context_model.fit(features["train"], train_increment)
    context_prediction = {
        "train": train_context_oof,
        "val": context_model.predict(features["val"]).astype("float32"),
        "test": context_model.predict(features["test"]).astype("float32"),
    }
    correction_target = {
        name: parts[name]["target_local_increment"].to_numpy(dtype="float32") - context_prediction[name]
        for name in parts
    }
    base_pm25 = {
        name: parts[name]["background_reference_pm25"].to_numpy(dtype="float32")
        + context_prediction[name]
        for name in parts
    }
    datasets = {
        name: SequenceDataset(
            part, embeddings, features[name], correction_target[name], base_pm25[name]
        )
        for name, part in parts.items()
    }
    training_sites = parts["train"]["site"].astype(str)
    site_counts = training_sites.value_counts()
    weights = training_sites.map(lambda site: 1.0 / site_counts[site]).to_numpy(
        dtype="float64", copy=True
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), len(weights), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }

    context_mean = features["train"].mean(axis=0)
    context_std = np.maximum(features["train"].std(axis=0), 1e-6)
    # Normalize in-place after constructing datasets, which hold these arrays.
    for name in datasets:
        datasets[name].context[:] = (datasets[name].context - context_mean) / context_std

    device = select_device(args.device)
    model = ConditionedGRU(
        embeddings.shape[1], features["train"].shape[1], args.projection_dim,
        args.hidden_dim, args.dropout, args.max_correction,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)
    best_rmse = math.inf
    best_state = None
    best_alpha = 0.0
    bad_epochs = 0
    history = []
    val_actual = parts["val"]["target_pm25"].to_numpy(dtype=float)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, context, target, _, _, _ in loaders["train"]:
            optimizer.zero_grad()
            predicted = model(image.to(device), context.to(device))
            target = target.to(device)
            loss = loss_fn(predicted, target) + args.correction_l2 * predicted.square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_correction, val_base, _, _ = predict(model, loaders["val"], device)
        grid = np.arange(0.0, args.alpha_max + args.alpha_step / 2, args.alpha_step)
        errors = [
            mean_squared_error(val_actual, val_base + alpha * val_correction) ** 0.5
            for alpha in grid
        ]
        selected = int(np.argmin(errors))
        alpha, val_rmse = float(grid[selected]), float(errors[selected])
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_rmse": val_rmse, "alpha": alpha}
        )
        if val_rmse < best_rmse - 1e-8:
            best_rmse = val_rmse
            best_alpha = alpha
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args),
            "context_mean": context_mean.tolist(),
            "context_std": context_std.tolist(),
            "validation_selected_alpha": best_alpha,
        },
        output / "best_model.pt",
    )

    report = {
        "protocol": args.protocol_label,
        "context_model": "ExtraTreesRegressor",
        "image_model": "FiLM_background_conditioned_GRU",
        "validation_selected_alpha": best_alpha,
        "validation_selected_checkpoint_rmse": best_rmse,
        "test_partition_target_used_for_training_or_selection": False,
    }
    for name in ("val", "test"):
        correction, base, actual, sequence_ids = predict(model, loaders[name], device)
        reference = parts[name]["background_reference_pm25"].to_numpy(dtype=float)
        conditioned = base + best_alpha * correction
        report[name] = {
            "reference_only": score(actual, reference),
            "reference_context": score(actual, base),
            "reference_context_plus_conditioned_image": score(actual, conditioned),
        }
        pd.DataFrame(
            {
                "sequence_id": sequence_ids.astype(int),
                "actual_pm25": actual,
                "predicted_reference_only": reference,
                "predicted_reference_context": base,
                "predicted_conditioned_image_correction": correction,
                "validation_selected_alpha": best_alpha,
                "predicted_reference_context_plus_conditioned_image": conditioned,
            }
        ).to_csv(output / f"predictions_{name}.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
