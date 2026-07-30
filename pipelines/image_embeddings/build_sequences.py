"""Build temporal windows only within preassigned groups and splits."""

import argparse
from pathlib import Path

import pandas as pd

from roadside_pm.features.images.sequences import build_grouped_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--embedding-index", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--id-col", default="sample_index")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--split-col", required=True)
    parser.add_argument("--group-cols", nargs="+", required=True)
    parser.add_argument("--target-cols", nargs="+", required=True)
    parser.add_argument("--sequence-length", type=int, default=7)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    index = pd.read_csv(args.embedding_index)
    index = index[index["embedding_status"] == "success"].copy()
    merged = manifest.merge(index[[args.id_col, "embedding_row"]], on=args.id_col, how="inner", validate="one_to_one")
    sequences = build_grouped_sequences(merged, sequence_length=args.sequence_length, id_column=args.id_col, embedding_row_column="embedding_row", timestamp_column=args.timestamp_col, group_columns=args.group_cols, split_column=args.split_col, target_columns=args.target_cols)
    output = Path(args.output_csv); output.parent.mkdir(parents=True, exist_ok=True)
    sequences.to_csv(output, index=False)
    print(f"Saved {len(sequences)} leakage-safe sequences: {output}")


if __name__ == "__main__": main()

