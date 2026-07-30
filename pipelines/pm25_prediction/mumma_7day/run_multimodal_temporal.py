"""Train gated multi-lens ResNet + tabular temporal PM2.5 models.

The random protocol constructs overlapping windows first and shuffles the
windows afterward. It is deliberately leakage-contaminated and exists only as
a replication diagnostic. The date protocol holds out an entire date and
selects validation runs from the remaining dates.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from roadside_pm.features.images.resnet import select_device
from roadside_pm.features.images.sequences import build_grouped_sequences
from roadside_pm.modeling.multimodal_temporal import GatedMultiViewTemporalRegressor


PROXY_PATTERN = re.compile(r"(^|_)(spm|snpm|stps|opc|pm1|pm2|pm4|pm10)(_|$)", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-table", required=True)
    parser.add_argument("--feature-groups", required=True)
    parser.add_argument("--feature-set", default="sensor_plus_visual")
    parser.add_argument("--embedding-index", required=True)
    parser.add_argument("--lens1-embeddings", required=True)
    parser.add_argument("--lens2-embeddings", required=True)
    parser.add_argument("--lens6-embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", choices=["random", "date"], default="random")
    parser.add_argument("--test-date", help="Required with --protocol date")
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--target-window", type=int, default=3,
                        help="Causal trailing target mean in rows; 3 rows = 30 seconds at 10 s")
    parser.add_argument("--image-hidden-dim", type=int, default=128)
    parser.add_argument("--tabular-hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def add_causal_target(frame: pd.DataFrame, target: str, window: int) -> tuple[pd.DataFrame, str]:
    if window < 1:
        raise ValueError("target-window must be positive")
    result = frame.sort_values(["run_id", "sample_timestamp", "sample_id"], kind="stable").copy()
    name = target if window == 1 else f"{target}_trailing_{window}rows"
    result[name] = (
        result.groupby("run_id", sort=False)[target]
        .transform(lambda values: values.rolling(window, min_periods=window).mean())
    )
    return result[result[name].notna()].copy(), name


def assign_date_splits(frame: pd.DataFrame, test_date: str, seed: int) -> pd.DataFrame:
    result = frame.copy()
    result["model_split"] = "train"
    result.loc[result["date"].astype(str).eq(test_date), "model_split"] = "test"
    training = result[result["model_split"].eq("train")]
    runs = training["run_id"].drop_duplicates().to_numpy()
    if len(runs) < 2 or not result["model_split"].eq("test").any():
        raise ValueError("Date protocol requires a populated test date and at least two training runs")
    rng = np.random.default_rng(seed)
    rng.shuffle(runs)
    validation_runs = set(runs[:max(1, round(0.2 * len(runs)))])
    result.loc[result["run_id"].isin(validation_runs), "model_split"] = "val"
    return result


def make_sequences(frame: pd.DataFrame, args: argparse.Namespace, target_name: str) -> pd.DataFrame:
    work = frame.copy()
    if args.protocol == "date":
        if not args.test_date:
            raise ValueError("--test-date is required with --protocol date")
        work = assign_date_splits(work, args.test_date, args.seed)
        return build_grouped_sequences(
            work, sequence_length=args.sequence_length, id_column="sample_id",
            embedding_row_column="embedding_row", timestamp_column="sample_timestamp",
            group_columns=["run_id"], split_column="model_split", target_columns=[target_name],
        )

    work["sequence_pool"] = "all"
    sequences = build_grouped_sequences(
        work, sequence_length=args.sequence_length, id_column="sample_id",
        embedding_row_column="embedding_row", timestamp_column="sample_timestamp",
        group_columns=["run_id"], split_column="sequence_pool", target_columns=[target_name],
    )
    train_val, test = train_test_split(
        sequences.index.to_numpy(), test_size=0.2, random_state=args.seed, shuffle=True,
    )
    train, val = train_test_split(
        train_val, test_size=0.25, random_state=args.seed, shuffle=True,
    )
    sequences["model_split"] = ""
    sequences.loc[train, "model_split"] = "train"
    sequences.loc[val, "model_split"] = "val"
    sequences.loc[test, "model_split"] = "test"
    return sequences


def overlap_audit(sequences: pd.DataFrame) -> dict[str, object]:
    exploded = sequences[["model_split", "sample_ids"]].copy()
    exploded["sample_id"] = exploded["sample_ids"].str.split("|")
    exploded = exploded.explode("sample_id")
    memberships = exploded.groupby("sample_id")["model_split"].nunique()
    sets = {name: set(part["sample_id"]) for name, part in exploded.groupby("model_split")}
    train_context = sets.get("train", set())
    test_targets = set(sequences.loc[sequences.model_split.eq("test"), "target_sample_id"])
    return {
        "raw_samples_in_multiple_splits": int((memberships > 1).sum()),
        "unique_raw_samples": int(len(memberships)),
        "train_test_raw_overlap": int(len(sets.get("train", set()) & sets.get("test", set()))),
        "test_target_frames_seen_as_train_context": int(len(test_targets & train_context)),
        "test_target_frame_context_leakage_fraction": float(len(test_targets & train_context) / max(len(test_targets), 1)),
        "leakage_contaminated": bool((memberships > 1).any()),
    }


class MultimodalSequences(Dataset):
    def __init__(self, rows, image_arrays, tabular, target_name, target_mean, target_std):
        self.rows = rows.reset_index(drop=True)
        self.images = image_arrays
        self.tabular = tabular
        self.target_name = target_name
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows.iloc[item]
        positions = np.fromstring(str(row["embedding_rows"]), dtype=int, sep="|")
        images = np.stack([array[positions] for array in self.images], axis=1).astype("float32")
        tabular = self.tabular[positions].astype("float32")
        target = (float(row[self.target_name]) - self.target_mean) / self.target_std
        return torch.from_numpy(images), torch.from_numpy(tabular), torch.tensor(target, dtype=torch.float32), int(row.sequence_id)


def metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "R2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.modeling_table)
    index = pd.read_csv(args.embedding_index)
    index = index[index.embedding_status.eq("success")].sort_values("embedding_row")
    if index.sample_id.duplicated().any() or table.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique in the embedding index and modeling table")
    table, target_name = add_causal_target(table, args.target, args.target_window)
    aligned = index.merge(table, on="sample_id", how="inner", validate="one_to_one")
    if len(aligned) != len(table):
        raise ValueError(f"Only {len(aligned)}/{len(table)} target rows have embeddings")

    groups = json.loads(Path(args.feature_groups).read_text())
    if args.feature_set not in groups:
        raise ValueError(f"Unknown feature set {args.feature_set!r}; available={sorted(groups)}")
    feature_columns = list(groups[args.feature_set]["columns"])
    rejected = [column for column in feature_columns if PROXY_PATTERN.search(column)]
    if rejected:
        raise ValueError(f"PM/OPC target proxies are forbidden: {rejected}")

    sequences = make_sequences(aligned, args, target_name)
    sequences.to_csv(output / "sequences.csv", index=False)
    audit = overlap_audit(sequences)
    (output / "overlap_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    counts = sequences.model_split.value_counts()
    if not all(counts.get(name, 0) for name in ("train", "val", "test")):
        raise ValueError(f"Empty split: {counts.to_dict()}")

    # Create an embedding-row-aligned tabular array. Imputation and scaling are
    # fitted only on raw samples used by training sequences.
    max_row = int(index.embedding_row.max())
    raw_tabular = np.full((max_row + 1, len(feature_columns)), np.nan, dtype="float32")
    values = aligned[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    raw_tabular[aligned.embedding_row.to_numpy(dtype=int)] = values
    train_positions = np.unique(np.concatenate([
        np.fromstring(value, dtype=int, sep="|")
        for value in sequences.loc[sequences.model_split.eq("train"), "embedding_rows"]
    ]))
    median = np.nanmedian(raw_tabular[train_positions], axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    imputed = np.where(np.isfinite(raw_tabular), raw_tabular, median)
    mean = imputed[train_positions].mean(axis=0)
    std = imputed[train_positions].std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    tabular = ((imputed - mean) / std).astype("float32")

    image_arrays = [
        np.load(path, mmap_mode="r")
        for path in (args.lens1_embeddings, args.lens2_embeddings, args.lens6_embeddings)
    ]
    if len({array.shape for array in image_arrays}) != 1:
        raise ValueError(f"Lens embedding shapes differ: {[array.shape for array in image_arrays]}")
    if image_arrays[0].shape[0] <= max_row:
        raise ValueError("Embedding arrays do not cover the embedding index")

    train_rows = sequences[sequences.model_split.eq("train")]
    target_mean = float(train_rows[target_name].mean())
    target_std = max(float(train_rows[target_name].std(ddof=0)), 1e-6)
    datasets = {
        name: MultimodalSequences(
            sequences[sequences.model_split.eq(name)], image_arrays, tabular,
            target_name, target_mean, target_std,
        )
        for name in ("train", "val", "test")
    }
    loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train")
        for name, dataset in datasets.items()
    }
    device = select_device(args.device)
    model = GatedMultiViewTemporalRegressor(
        image_arrays[0].shape[1], len(feature_columns), n_views=3,
        image_hidden_dim=args.image_hidden_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        temporal_hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    best_loss, best_state, bad_epochs, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for images, features, target, _ in loaders["train"]:
            optimizer.zero_grad()
            prediction = model(images.to(device), features.to(device))
            loss = loss_fn(prediction, target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for images, features, target, _ in loaders["val"]:
                val_losses.append(loss_fn(model(images.to(device), features.to(device)), target.to(device)).item())
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={history[-1]['train_loss']:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"early_stopping epoch={epoch} best_val_loss={best_loss:.6f}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    torch.save({
        "state_dict": model.state_dict(), "config": vars(args),
        "feature_columns": feature_columns, "tabular_median": median.tolist(),
        "tabular_mean": mean.tolist(), "tabular_std": std.tolist(),
        "target_name": target_name, "target_mean": target_mean, "target_std": target_std,
    }, output / "best_model.pt")

    reports, attention_rows = {}, []
    for split in ("val", "test"):
        actuals, predictions, sequence_ids = [], [], []
        attention_sum = np.zeros(3, dtype=float)
        attention_count = 0
        model.eval()
        with torch.inference_mode():
            for images, features, target, ids in loaders[split]:
                normalized, attention = model(images.to(device), features.to(device), return_attention=True)
                predictions.extend((normalized.cpu().numpy() * target_std + target_mean).tolist())
                actuals.extend((target.numpy() * target_std + target_mean).tolist())
                sequence_ids.extend(ids.numpy().tolist())
                attention_sum += attention.cpu().numpy().sum(axis=(0, 1))
                attention_count += attention.shape[0] * attention.shape[1]
        actual = np.asarray(actuals)
        predicted = np.asarray(predictions)
        reports[split] = metric_row(actual, predicted)
        pd.DataFrame({
            "sequence_id": sequence_ids, "actual": actual, "predicted": predicted,
        }).to_csv(output / f"predictions_{split}.csv", index=False)
        for lens, weight in zip(("lens1", "lens2", "lens6"), attention_sum / attention_count):
            attention_rows.append({"split": split, "lens": lens, "mean_attention": float(weight)})
    pd.DataFrame(attention_rows).to_csv(output / "lens_attention.csv", index=False)

    run = {
        "protocol": args.protocol,
        "reportable_as_generalization": args.protocol == "date" and not audit["leakage_contaminated"],
        "target": target_name,
        "target_definition": f"causal trailing mean of {args.target} over {args.target_window} rows",
        "sequence_length": args.sequence_length,
        "feature_set": args.feature_set,
        "n_features": len(feature_columns),
        "pm_opc_target_proxies_used": False,
        "split_counts": {key: int(value) for key, value in counts.items()},
        "overlap_audit": audit,
        "metrics": reports,
    }
    (output / "metrics.json").write_text(json.dumps(reports, indent=2) + "\n")
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
