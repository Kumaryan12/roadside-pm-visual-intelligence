"""Create a timestamp/location table for TRAQID atmospheric extraction.

The public processed TRAQID manifest contains timestamps but not the GPS track
described in the dataset paper.  This adapter therefore requires an explicit
representative latitude/longitude and records that approximation in every row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "experiments/traqid_pretraining_v1/data/processed/"
            "traqid_paired_manifest_with_splits.csv"
        ),
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument(
        "--location-note",
        default="user-supplied representative coordinate",
    )
    args = parser.parse_args()

    source = pd.read_csv(
        args.manifest,
        usecols=["row_id", "created_at_parsed"],
    )
    if source["row_id"].duplicated().any():
        raise ValueError("TRAQID row_id must be unique")
    timestamp = pd.to_datetime(source["created_at_parsed"], errors="raise")
    result = pd.DataFrame(
        {
            "sample_id": source["row_id"].astype(str),
            "sample_timestamp": timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "lat": float(args.latitude),
            "long": float(args.longitude),
            "location_provenance": args.location_note,
        }
    )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    output.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "rows": len(result),
                "unique_dates": int(timestamp.dt.date.nunique()),
                "minimum_timestamp": str(timestamp.min()),
                "maximum_timestamp": str(timestamp.max()),
                "representative_coordinate": {
                    "latitude": args.latitude,
                    "longitude": args.longitude,
                },
                "location_note": args.location_note,
                "warning": (
                    "The processed TRAQID release has no per-row GPS. "
                    "This coordinate is a regional approximation."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(result)} TRAQID atmospheric input rows: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

