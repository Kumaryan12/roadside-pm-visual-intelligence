"""Build nested training-site folds for conservative visual-weight selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipelines.china_surveillance_aqi.build_synchronized_reference import reference_for_rows


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
    reports = []
    for outer in sorted(root.glob("site_*")):
        outer_audit = json.loads((outer / "split_audit.json").read_text())
        outer_training_sites = [str(value) for value in outer_audit["protocol"]["training_sites"]]
        outer_validation_site = str(outer_audit["protocol"]["validation_site"])
        outer_test_site = str(outer_audit["protocol"]["test_site"])
        base = pd.read_csv(outer / "sequences.csv")
        base["site"] = base["site"].astype(str)
        base["target_time"] = pd.to_datetime(base["target_time"], errors="raise")

        for inner_test_site in outer_training_sites:
            inner_training_sites = [site for site in outer_training_sites if site != inner_test_site]
            keep_sites = set(inner_training_sites + [outer_validation_site, inner_test_site])
            sequences = base.loc[base["site"].isin(keep_sites)].copy()
            sequences["split"] = "train"
            sequences.loc[sequences["site"].eq(outer_validation_site), "split"] = "val"
            sequences.loc[sequences["site"].eq(inner_test_site), "split"] = "test"
            background, sources = reference_for_rows(sequences, wide, inner_training_sites)
            sequences["background_reference_pm25"] = background
            sequences["background_reference_sites"] = sources
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
            sequences = sequences.loc[usable].copy()

            forbidden = {outer_validation_site, inner_test_site, outer_test_site}
            used = set()
            for packed in sequences["background_reference_sites"].astype(str):
                used.update(value for value in packed.split("|") if value)
            if used & forbidden:
                raise RuntimeError(
                    f"Forbidden reference in outer={outer.name}, inner={inner_test_site}: {used & forbidden}"
                )
            self_reference = sequences.apply(
                lambda row: row["site"] in str(row["background_reference_sites"]).split("|"), axis=1
            )
            if self_reference.any():
                raise RuntimeError(f"Self-reference in outer={outer.name}, inner={inner_test_site}")
            counts = sequences["split"].value_counts().to_dict()
            if {"train", "val", "test"} - set(counts):
                raise ValueError(f"Empty inner partition: outer={outer.name}, inner={inner_test_site}")

            destination = outer / "inner_alpha_cv" / f"held_{inner_test_site}"
            destination.mkdir(parents=True, exist_ok=True)
            sequences.to_csv(destination / "sequences_reference.csv", index=False)
            report = {
                "outer_test_site": outer_test_site,
                "outer_validation_site": outer_validation_site,
                "inner_test_site": inner_test_site,
                "inner_training_reference_sites": inner_training_sites,
                "outer_test_site_rows_present": int(sequences["site"].eq(outer_test_site).sum()),
                "target_or_validation_sites_used_as_reference": sorted(used & forbidden),
                "self_reference_rows": int(self_reference.sum()),
                "sequence_counts": {str(key): int(value) for key, value in counts.items()},
            }
            (destination / "split_audit.json").write_text(json.dumps(report, indent=2))
            reports.append(report)

    (root / "inner_alpha_cv_summary.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps({"inner_folds": len(reports), "taiwan_root": str(root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
