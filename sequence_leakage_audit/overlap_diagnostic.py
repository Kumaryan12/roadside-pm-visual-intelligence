import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_id_set(x):
    """
    Parses row-id strings like:
      "0|1|2|3"
      "0,1,2,3"
      "0 1 2 3"
      123
    into a Python set of ints/strings.

    Keeps non-integer IDs as strings if integer conversion fails.
    """
    if pd.isna(x):
        return set()

    if isinstance(x, (int, np.integer)):
        return {int(x)}

    if isinstance(x, float):
        if math.isfinite(x):
            return {int(x)}
        return set()

    s = str(x).strip()
    if not s:
        return set()

    for sep in ["|", ",", " "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            break
    else:
        parts = [s]

    out = set()
    for p in parts:
        try:
            out.add(int(float(p)))
        except Exception:
            out.add(p)
    return out


def build_inverted_index(sets):
    inv = defaultdict(list)
    for i, ids in enumerate(sets):
        for rid in ids:
            inv[rid].append(i)
    return inv


def max_overlap_for_one(eval_input, eval_target, train_inputs, train_targets, train_combined, inv_combined):
    """
    Finds the training sequence with maximum combined overlap.
    Also reports input overlap and target overlap with that best train sequence.

    For speed, candidate training sequences are retrieved through an inverted index.
    """
    eval_combined = eval_input | eval_target

    candidate_idxs = set()
    for rid in eval_combined:
        candidate_idxs.update(inv_combined.get(rid, []))

    if not candidate_idxs:
        return {
            "nearest_train_local_idx": -1,
            "max_input_overlap": 0,
            "max_target_overlap": 0,
            "max_combined_overlap": 0,
        }

    best_idx = -1
    best_combined = -1
    best_input = 0
    best_target = 0

    for idx in candidate_idxs:
        input_ov = len(eval_input & train_inputs[idx])
        target_ov = len(eval_target & train_targets[idx])
        combined_ov = len(eval_combined & train_combined[idx])

        if combined_ov > best_combined:
            best_combined = combined_ov
            best_input = input_ov
            best_target = target_ov
            best_idx = idx

    return {
        "nearest_train_local_idx": best_idx,
        "max_input_overlap": best_input,
        "max_target_overlap": best_target,
        "max_combined_overlap": best_combined,
    }


def fraction_ge(series, k):
    if len(series) == 0:
        return np.nan
    return float((series >= k).mean())


def summarize(details, thresholds):
    row = {}
    for col in ["max_input_overlap", "max_target_overlap", "max_combined_overlap"]:
        vals = details[col].astype(float)
        row[f"{col}_mean"] = vals.mean()
        row[f"{col}_median"] = vals.median()
        row[f"{col}_max"] = vals.max()

        for k in thresholds:
            row[f"{col}_frac_ge_{k}"] = fraction_ge(vals, k)

        row[f"{col}_frac_eq_0"] = float((vals == 0).mean())

    return row


def main():
    ap = argparse.ArgumentParser(
        description="Audit train/eval overlap leakage in sliding-window sequence manifests."
    )

    ap.add_argument("--manifest", required=True, help="Path to sequence manifest CSV.")
    ap.add_argument("--split-col", required=True, help="Column containing split labels.")
    ap.add_argument("--train-name", required=True, help="Split value used for training.")
    ap.add_argument("--eval-names", nargs="+", required=True, help="One or more split values to audit, e.g. val test.")
    ap.add_argument("--input-col", required=True, help="Column containing input row/frame ids, e.g. seq_row_ids.")
    ap.add_argument("--target-col", default=None, help="Optional column containing target window row ids for forecasting.")
    ap.add_argument("--target-id-col", default=None, help="Optional single target row id column, e.g. target_row_id.")
    ap.add_argument("--sequence-id-col", default="sequence_id", help="Sequence id column.")
    ap.add_argument("--thresholds", default="1,3,5,6", help="Comma-separated thresholds for overlap fractions.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")

    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]

    print("=" * 90)
    print("SEQUENCE OVERLAP DIAGNOSTIC")
    print("=" * 90)
    print(f"Manifest: {manifest_path}")
    print(f"Split column: {args.split_col}")
    print(f"Train split: {args.train_name}")
    print(f"Eval splits: {args.eval_names}")
    print(f"Input column: {args.input_col}")
    print(f"Target window column: {args.target_col}")
    print(f"Target id column: {args.target_id_col}")
    print(f"Output: {out_dir}")

    df = pd.read_csv(manifest_path)
    print(f"\nLoaded shape: {df.shape}")

    required = [args.split_col, args.input_col]
    if args.sequence_id_col in df.columns:
        required.append(args.sequence_id_col)
    if args.target_col:
        required.append(args.target_col)
    if args.target_id_col:
        required.append(args.target_id_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    train_df = df[df[args.split_col] == args.train_name].copy()
    if train_df.empty:
        raise ValueError(f"No train rows found for {args.split_col} == {args.train_name}")

    print("\nSplit counts:")
    print(df[args.split_col].value_counts(dropna=False))

    train_inputs = [parse_id_set(x) for x in train_df[args.input_col].values]

    train_targets = []
    for _, r in train_df.iterrows():
        tgt = set()
        if args.target_col:
            tgt |= parse_id_set(r[args.target_col])
        if args.target_id_col:
            tgt |= parse_id_set(r[args.target_id_col])
        train_targets.append(tgt)

    train_combined = [a | b for a, b in zip(train_inputs, train_targets)]
    inv_combined = build_inverted_index(train_combined)

    if args.sequence_id_col in train_df.columns:
        train_seq_ids = train_df[args.sequence_id_col].tolist()
    else:
        train_seq_ids = train_df.index.tolist()

    all_details = []
    summary_rows = []

    for eval_name in args.eval_names:
        eval_df = df[df[args.split_col] == eval_name].copy()

        if eval_df.empty:
            print(f"\nWarning: no rows found for eval split {eval_name}; skipping.")
            continue

        print("\n" + "-" * 90)
        print(f"Auditing eval split: {eval_name}")
        print(f"Train sequences: {len(train_df)}")
        print(f"Eval sequences : {len(eval_df)}")

        eval_inputs = [parse_id_set(x) for x in eval_df[args.input_col].values]

        eval_targets = []
        for _, r in eval_df.iterrows():
            tgt = set()
            if args.target_col:
                tgt |= parse_id_set(r[args.target_col])
            if args.target_id_col:
                tgt |= parse_id_set(r[args.target_id_col])
            eval_targets.append(tgt)

        details_rows = []

        for local_i, (idx, r) in enumerate(eval_df.iterrows()):
            res = max_overlap_for_one(
                eval_inputs[local_i],
                eval_targets[local_i],
                train_inputs,
                train_targets,
                train_combined,
                inv_combined,
            )

            nearest_local = res["nearest_train_local_idx"]
            nearest_seq_id = None if nearest_local < 0 else train_seq_ids[nearest_local]

            seq_id = r[args.sequence_id_col] if args.sequence_id_col in eval_df.columns else idx

            row = {
                "eval_split": eval_name,
                "eval_index": idx,
                "eval_sequence_id": seq_id,
                "nearest_train_sequence_id": nearest_seq_id,
                "input_len": len(eval_inputs[local_i]),
                "target_len": len(eval_targets[local_i]),
                **res,
            }
            details_rows.append(row)

        details = pd.DataFrame(details_rows)

        summary = summarize(details, thresholds)
        summary.update({
            "manifest": str(manifest_path),
            "split_col": args.split_col,
            "train_name": args.train_name,
            "eval_name": eval_name,
            "train_sequences": len(train_df),
            "eval_sequences": len(eval_df),
            "input_col": args.input_col,
            "target_col": args.target_col if args.target_col else "",
            "target_id_col": args.target_id_col if args.target_id_col else "",
        })

        summary_rows.append(summary)
        all_details.append(details)

        print("\nSummary:")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

    if not summary_rows:
        raise RuntimeError("No eval splits were audited.")

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.concat(all_details, ignore_index=True)

    summary_csv = out_dir / "overlap_summary.csv"
    details_csv = out_dir / "overlap_by_eval_sequence.csv"
    report_md = out_dir / "overlap_report.md"

    summary_df.to_csv(summary_csv, index=False)
    details_df.to_csv(details_csv, index=False)

    md = []
    md.append("# Sequence Overlap Diagnostic Report\n")
    md.append(f"- Manifest: `{manifest_path}`")
    md.append(f"- Split column: `{args.split_col}`")
    md.append(f"- Train split: `{args.train_name}`")
    md.append(f"- Eval splits: `{', '.join(args.eval_names)}`")
    md.append(f"- Input column: `{args.input_col}`")
    md.append(f"- Target column: `{args.target_col if args.target_col else ''}`")
    md.append(f"- Target id column: `{args.target_id_col if args.target_id_col else ''}`")
    md.append("")
    md.append("## Summary\n")
    md.append(summary_df.to_markdown(index=False))
    md.append("")
    md.append("## Interpretation guide\n")
    md.append("- `max_input_overlap`: maximum number of input frames/rows shared with any training sequence.")
    md.append("- `max_target_overlap`: maximum number of target-window rows shared with any training sequence. This matters for forecasting.")
    md.append("- `max_combined_overlap`: maximum overlap across input and target ids combined.")
    md.append("- High fractions near the sequence length indicate severe sliding-window leakage risk.")
    md.append("- Zero overlap does not guarantee full statistical independence; it only rules out direct row/frame overlap.")
    report_md.write_text("\n".join(md))

    print("\n" + "=" * 90)
    print("DONE")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved details : {details_csv}")
    print(f"Saved report  : {report_md}")


if __name__ == "__main__":
    main()
