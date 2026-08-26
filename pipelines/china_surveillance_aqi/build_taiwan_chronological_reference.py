"""Build a leakage-free Taiwan future-period synchronized-reference benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from pipelines.china_surveillance_aqi.build_synchronized_reference import reference_for_rows
from roadside_pm.features.images.sequences import build_grouped_sequences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--max-gap-hours", type=float, default=3.0)
    parser.add_argument("--minimum-reference-sites", type=int, default=1)
    args = parser.parse_args()

    raw = pd.read_csv(args.manifest, low_memory=False)
    raw = raw.loc[raw["dataset"].eq("taiwan")].copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="raise")
    raw["date"] = raw["timestamp"].dt.date.astype(str)
    raw["site"] = raw["site"].astype(str)
    raw["pm25"] = pd.to_numeric(raw["pm25"], errors="coerce")
    raw["embedding_row"] = pd.to_numeric(raw["row_id"], errors="raise").astype(int)
    sites = sorted(raw["site"].unique())
    dates = sorted(raw["date"].unique())
    if len(sites) < 3 or len(dates) < 10:
        raise ValueError(f"Insufficient sites/dates: sites={len(sites)}, dates={len(dates)}")
    train_end = max(1, math.floor(0.60 * len(dates)))
    val_end = min(max(train_end + 1, math.floor(0.80 * len(dates))), len(dates) - 1)
    mapping = {
        date: "train" if index < train_end else "val" if index < val_end else "test"
        for index, date in enumerate(dates)
    }
    raw["split"] = raw["date"].map(mapping)

    pieces = []
    for (site, split), group in raw.groupby(["site", "split"], sort=True):
        group = group.sort_values(["timestamp", "row_id"], kind="stable").copy()
        gap = group["timestamp"].diff().dt.total_seconds().div(3600)
        boundary = gap.isna() | gap.le(0) | gap.gt(args.max_gap_hours)
        group["episode_id"] = site + "_" + split + "_" + boundary.cumsum().astype(str)
        group["target_pm25"] = group["pm25"]
        pieces.append(group)
    work = pd.concat(pieces, ignore_index=True).dropna(subset=["target_pm25"])
    sequences = build_grouped_sequences(
        work,
        sequence_length=args.sequence_length,
        id_column="row_id",
        embedding_row_column="embedding_row",
        timestamp_column="timestamp",
        group_columns=["dataset", "site", "episode_id"],
        split_column="split",
        target_columns=["target_pm25"],
    )

    raw_sets = {}
    for split, group in sequences.groupby("split"):
        values = set()
        for packed in group["sample_ids"].astype(str):
            values.update(packed.split("|"))
        raw_sets[str(split)] = values
    overlap = {
        "train_val": len(raw_sets.get("train", set()) & raw_sets.get("val", set())),
        "train_test": len(raw_sets.get("train", set()) & raw_sets.get("test", set())),
        "val_test": len(raw_sets.get("val", set()) & raw_sets.get("test", set())),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Chronological raw overlap: {overlap}")

    wide = raw.pivot_table(index="timestamp", columns="site", values="pm25", aggfunc="median")
    background, sources = reference_for_rows(sequences, wide, sites)
    sequences["background_reference_pm25"] = background
    sequences["background_reference_sites"] = sources
    sequences["background_reference_site_count"] = sequences[
        "background_reference_sites"
    ].map(lambda value: len(str(value).split("|")) if str(value) else 0)
    sequences["target_local_increment"] = (
        sequences["target_pm25"] - sequences["background_reference_pm25"]
    )
    usable = (
        sequences["background_reference_pm25"].notna()
        & sequences["background_reference_site_count"].ge(args.minimum_reference_sites)
    )
    sequences = sequences.loc[usable].copy()
    self_reference = sequences.apply(
        lambda row: row["site"] in str(row["background_reference_sites"]).split("|"), axis=1
    )
    if self_reference.any():
        raise RuntimeError("A target site was used in its own synchronized reference")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequences.to_csv(output / "sequences_reference.csv", index=False)
    audit = {
        "protocol": "chronological_complete_date_60_20_20",
        "dates": len(dates),
        "sites": sites,
        "training_dates": [date for date in dates if mapping[date] == "train"],
        "validation_dates": [date for date in dates if mapping[date] == "val"],
        "test_dates": [date for date in dates if mapping[date] == "test"],
        "sequence_counts": {str(k): int(v) for k, v in sequences["split"].value_counts().items()},
        "raw_overlap": overlap,
        "self_reference_rows": int(self_reference.sum()),
        "background_at_inference": "contemporaneous median of other sites only",
        "target_site_pm25_used_in_own_background": False,
    }
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
