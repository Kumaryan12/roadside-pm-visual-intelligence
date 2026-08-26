"""Attach a leakage-safe synchronized regional PM2.5 reference to Taiwan folds.

For each outer fold, only sites designated as training/reference sites may
contribute to the background.  When the target row itself belongs to a training
site, that site is excluded from its own reference median.  Validation and test
site targets are therefore never used to construct their background value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def reference_for_rows(
    rows: pd.DataFrame,
    reference_wide: pd.DataFrame,
    training_sites: list[str],
) -> tuple[np.ndarray, list[str]]:
    values = []
    source_sets = []
    for row in rows.itertuples(index=False):
        timestamp = pd.Timestamp(row.target_time)
        target_site = str(row.site)
        allowed = [site for site in training_sites if site != target_site]
        if timestamp not in reference_wide.index:
            values.append(np.nan)
            source_sets.append("")
            continue
        available = reference_wide.loc[timestamp, allowed].dropna()
        values.append(float(available.median()) if len(available) else np.nan)
        source_sets.append("|".join(map(str, available.index)))
    return np.asarray(values, dtype=float), source_sets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--minimum-reference-sites", type=int, default=1)
    args = parser.parse_args()

    raw = pd.read_csv(args.manifest, low_memory=False)
    raw = raw.loc[raw["dataset"].eq("taiwan")].copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="raise")
    raw["site"] = raw["site"].astype(str)
    raw["pm25"] = pd.to_numeric(raw["pm25"], errors="coerce")
    wide = raw.pivot_table(index="timestamp", columns="site", values="pm25", aggfunc="median")

    root = Path(args.benchmark_root) / "taiwan"
    summaries = []
    for fold in sorted(root.glob("site_*")):
        audit_path = fold / "split_audit.json"
        sequence_path = fold / "sequences.csv"
        if not audit_path.exists() or not sequence_path.exists():
            continue
        audit = json.loads(audit_path.read_text())
        training_sites = [str(value) for value in audit["protocol"]["training_sites"]]
        test_site = str(audit["protocol"]["test_site"])
        validation_site = str(audit["protocol"]["validation_site"])
        if test_site in training_sites or validation_site in training_sites:
            raise RuntimeError(f"Fold {fold.name} has invalid site roles")

        sequences = pd.read_csv(sequence_path)
        sequences["target_time"] = pd.to_datetime(sequences["target_time"], errors="raise")
        sequences["site"] = sequences["site"].astype(str)
        background, source_sets = reference_for_rows(sequences, wide, training_sites)
        sequences["background_reference_pm25"] = background
        sequences["background_reference_sites"] = source_sets
        sequences["background_reference_site_count"] = sequences[
            "background_reference_sites"
        ].map(lambda value: len(str(value).split("|")) if str(value) else 0)
        sequences["target_local_increment"] = (
            pd.to_numeric(sequences["target_pm25"], errors="raise")
            - sequences["background_reference_pm25"]
        )
        usable = (
            sequences["background_reference_pm25"].notna()
            & sequences["background_reference_site_count"].ge(args.minimum_reference_sites)
        )
        output = sequences.loc[usable].copy()

        # Direct string-membership audit: a target site may never contribute to
        # its own synchronized background.
        self_reference = output.apply(
            lambda row: str(row["site"]) in str(row["background_reference_sites"]).split("|"),
            axis=1,
        )
        if self_reference.any():
            raise RuntimeError(f"Self-reference leakage detected in {fold.name}")
        forbidden = {test_site, validation_site}
        used_sites = set()
        for packed in output["background_reference_sites"].astype(str):
            used_sites.update(value for value in packed.split("|") if value)
        if used_sites & forbidden:
            raise RuntimeError(
                f"Outer validation/test sites used as references in {fold.name}: {used_sites & forbidden}"
            )

        output.to_csv(fold / "sequences_reference.csv", index=False)
        coverage = {
            str(split): {
                "rows_total": int(len(group)),
                "rows_usable": int(usable.loc[group.index].sum()),
                "coverage_fraction": float(usable.loc[group.index].mean()),
            }
            for split, group in sequences.groupby("split")
        }
        report = {
            "fold": fold.name,
            "protocol": "synchronized_training_site_reference",
            "training_reference_sites": training_sites,
            "validation_site_excluded": validation_site,
            "test_site_excluded": test_site,
            "training_target_site_excluded_from_own_reference": True,
            "target_values_used_to_choose_reference_sites": False,
            "minimum_reference_sites": args.minimum_reference_sites,
            "coverage": coverage,
            "self_reference_rows": int(self_reference.sum()),
            "forbidden_reference_sites_used": sorted(used_sites & forbidden),
        }
        (fold / "reference_audit.json").write_text(json.dumps(report, indent=2))
        summaries.append(report)

    if not summaries:
        raise FileNotFoundError(f"No Taiwan fold directories found below {root}")
    (root / "reference_summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps({"folds": len(summaries), "benchmark_root": str(root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
