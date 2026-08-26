"""Build frozen leakage-free public China surveillance benchmarks.

The released paper-style sequence splits shuffle overlapping T=7 windows.  This
builder instead assigns raw observations to partitions first and constructs
windows only within a partition and a continuous camera episode.

Protocols
---------
* Taiwan: six outer leave-one-site-out folds.  The next site in sorted order is
  used as validation and the remaining four sites are training sites.
* Nanjing/Shanghai: one chronological fold using complete dates.  The earliest
  60% of dates train, the next 20% validate, and the latest 20% test.

No target values are consulted when creating any split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from roadside_pm.features.images.sequences import build_grouped_sequences


TARGETS = {
    "taiwan": ("pm25", "pm10", "aqi"),
    "nanjing": ("pm25", "pm10", "aqi"),
    "shanghai": ("aqi",),
}


def chronological_date_partitions(dates: pd.Series) -> dict[str, str]:
    ordered = sorted(pd.Series(dates).dropna().astype(str).unique())
    if len(ordered) < 5:
        raise ValueError(f"Chronological evaluation requires at least five dates, found {len(ordered)}")
    train_end = max(1, math.floor(0.60 * len(ordered)))
    val_end = max(train_end + 1, math.floor(0.80 * len(ordered)))
    val_end = min(val_end, len(ordered) - 1)
    return {
        date: "train" if index < train_end else "val" if index < val_end else "test"
        for index, date in enumerate(ordered)
    }


def add_continuity_episodes(frame: pd.DataFrame, max_gap_hours: float) -> pd.DataFrame:
    pieces = []
    for (site, split), group in frame.groupby(["site", "split"], sort=True, dropna=False):
        group = group.sort_values(["timestamp", "row_id"], kind="stable").copy()
        gap_hours = group["timestamp"].diff().dt.total_seconds().div(3600)
        new_episode = gap_hours.isna() | gap_hours.le(0) | gap_hours.gt(max_gap_hours)
        group["episode_id"] = (
            str(site) + "_" + str(split) + "_" + new_episode.cumsum().astype(str)
        )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def raw_sets(sequences: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for split, group in sequences.groupby("split"):
        values: set[str] = set()
        for packed in group["sample_ids"].astype(str):
            values.update(packed.split("|"))
        result[str(split)] = values
    return result


def overlap_audit(sequences: pd.DataFrame) -> dict[str, object]:
    sets = raw_sets(sequences)
    train = sets.get("train", set())
    val = sets.get("val", set())
    test = sets.get("test", set())
    train_context: set[str] = set()
    for packed in sequences.loc[sequences["split"].eq("train"), "sample_ids"].astype(str):
        train_context.update(packed.split("|"))
    test_targets = set(
        sequences.loc[sequences["split"].eq("test"), "target_sample_id"].astype(str)
    )
    audit = {
        "raw_rows_by_split": {key: len(value) for key, value in sorted(sets.items())},
        "train_val_raw_overlap": len(train & val),
        "train_test_raw_overlap": len(train & test),
        "val_test_raw_overlap": len(val & test),
        "test_targets_seen_as_train_context": len(test_targets & train_context),
        "test_target_context_leakage_fraction": (
            len(test_targets & train_context) / len(test_targets) if test_targets else 0.0
        ),
    }
    if any(
        audit[key]
        for key in ("train_val_raw_overlap", "train_test_raw_overlap", "val_test_raw_overlap")
    ):
        raise RuntimeError(f"Raw-row leakage detected: {audit}")
    return audit


def build_fold(
    raw: pd.DataFrame,
    *,
    dataset: str,
    fold_name: str,
    partition: pd.Series,
    sequence_length: int,
    max_gap_hours: float,
    output_dir: Path,
    split_definition: dict[str, object],
) -> dict[str, object]:
    frame = raw.copy()
    frame["split"] = partition.reindex(frame.index)
    if frame["split"].isna().any():
        raise ValueError(f"Fold {fold_name} left raw observations without a partition")
    frame = add_continuity_episodes(frame, max_gap_hours)
    target_columns = [f"target_{name}" for name in TARGETS[dataset]]
    for source, target in zip(TARGETS[dataset], target_columns):
        frame[target] = pd.to_numeric(frame[source], errors="coerce")
    frame = frame.dropna(subset=target_columns).copy()
    sequences = build_grouped_sequences(
        frame,
        sequence_length=sequence_length,
        id_column="row_id",
        embedding_row_column="embedding_row",
        timestamp_column="timestamp",
        group_columns=["dataset", "site", "episode_id"],
        split_column="split",
        target_columns=target_columns,
    )
    if sequences.empty:
        raise ValueError(f"Fold {fold_name} produced no sequences")
    counts = sequences["split"].value_counts().to_dict()
    missing = {"train", "val", "test"} - set(counts)
    if missing:
        raise ValueError(f"Fold {fold_name} has no sequences for partitions: {sorted(missing)}")
    audit = overlap_audit(sequences)
    fold_dir = output_dir / dataset / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    sequences.to_csv(fold_dir / "sequences.csv", index=False)
    report = {
        "dataset": dataset,
        "fold": fold_name,
        "protocol": split_definition,
        "sequence_length": sequence_length,
        "maximum_within_sequence_gap_hours": max_gap_hours,
        "raw_rows": int(len(frame)),
        "sequence_counts": {str(key): int(value) for key, value in counts.items()},
        "date_counts_by_split": {
            str(split): int(group["target_time"].astype(str).str[:10].nunique())
            for split, group in sequences.groupby("split")
        },
        "site_counts_by_split": {
            str(split): int(group["site"].nunique())
            for split, group in sequences.groupby("split")
        },
        "overlap_audit": audit,
    }
    (fold_dir / "split_audit.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(TARGETS), default=sorted(TARGETS)
    )
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--max-gap-hours", type=float, default=3.0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, low_memory=False)
    required = {"row_id", "dataset", "site", "timestamp", "image_path", *set().union(*TARGETS.values())}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    manifest["timestamp"] = pd.to_datetime(manifest["timestamp"], errors="raise")
    manifest["date"] = manifest["timestamp"].dt.date.astype(str)
    manifest["embedding_row"] = pd.to_numeric(manifest["row_id"], errors="raise").astype(int)
    if manifest["row_id"].duplicated().any():
        raise ValueError("row_id must be unique across the manifest")
    expected = list(range(len(manifest)))
    if manifest["embedding_row"].tolist() != expected:
        raise ValueError(
            "This benchmark expects embedding array positions to match the contiguous manifest row_id"
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for dataset in args.datasets:
        raw = manifest.loc[manifest["dataset"].eq(dataset)].copy()
        raw = raw.loc[raw["image_path"].notna()].copy()
        if dataset == "taiwan":
            sites = sorted(raw["site"].astype(str).unique())
            if len(sites) < 3:
                raise ValueError("Taiwan leave-one-site-out needs at least three sites")
            site_values = raw["site"].astype(str)
            for index, test_site in enumerate(sites):
                val_site = sites[(index + 1) % len(sites)]
                partition = pd.Series("train", index=raw.index)
                partition.loc[site_values.eq(val_site)] = "val"
                partition.loc[site_values.eq(test_site)] = "test"
                reports.append(
                    build_fold(
                        raw,
                        dataset=dataset,
                        fold_name=f"site_{test_site}",
                        partition=partition,
                        sequence_length=args.sequence_length,
                        max_gap_hours=args.max_gap_hours,
                        output_dir=output,
                        split_definition={
                            "name": "leave_one_site_out",
                            "test_site": test_site,
                            "validation_site": val_site,
                            "training_sites": [s for s in sites if s not in {test_site, val_site}],
                            "target_values_used_for_split": False,
                        },
                    )
                )
        else:
            mapping = chronological_date_partitions(raw["date"])
            partition = raw["date"].map(mapping)
            reports.append(
                build_fold(
                    raw,
                    dataset=dataset,
                    fold_name="chronological",
                    partition=partition,
                    sequence_length=args.sequence_length,
                    max_gap_hours=args.max_gap_hours,
                    output_dir=output,
                    split_definition={
                        "name": "chronological_complete_date_60_20_20",
                        "training_dates": [date for date, split in mapping.items() if split == "train"],
                        "validation_dates": [date for date, split in mapping.items() if split == "val"],
                        "test_dates": [date for date, split in mapping.items() if split == "test"],
                        "target_values_used_for_split": False,
                    },
                )
            )

    summary = {
        "manifest": args.manifest,
        "output_dir": str(output),
        "benchmark_status": "frozen_v1",
        "protocol_warning": (
            "Do not modify sites, dates, gap threshold, sequence length, or validation choices "
            "after inspecting outer-test results."
        ),
        "folds": reports,
    }
    (output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    rows = []
    for report in reports:
        row = {"dataset": report["dataset"], "fold": report["fold"]}
        row.update({f"sequences_{key}": value for key, value in report["sequence_counts"].items()})
        row.update(report["overlap_audit"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "benchmark_summary.csv", index=False)
    print(json.dumps({"folds": len(reports), "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
