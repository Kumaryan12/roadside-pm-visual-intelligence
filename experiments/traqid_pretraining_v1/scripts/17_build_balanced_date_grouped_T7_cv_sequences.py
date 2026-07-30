import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["PM2.5", "PM10", "aqi"]


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_manifest(df):
    df = df.copy()

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    if "created_at" not in df.columns:
        raise ValueError("created_at column missing.")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    df = df.dropna(
        subset=["row_id", "created_at", *TARGETS]
    ).copy()

    df["row_id"] = df["row_id"].astype(int)
    df["date"] = df["created_at"].dt.date.astype(str)

    df = df.sort_values(
        ["created_at", "row_id"]
    ).reset_index(drop=True)

    return df


def make_date_stats(df):
    stats = (
        df.groupby("date", as_index=False)
        .agg(
            rows=("row_id", "size"),
            start_time=("created_at", "min"),
            end_time=("created_at", "max"),
            pm25_mean=("PM2.5", "mean"),
            pm25_std=("PM2.5", "std"),
            pm25_min=("PM2.5", "min"),
            pm25_median=("PM2.5", "median"),
            pm25_max=("PM2.5", "max"),
        )
    )

    stats["pm25_std"] = stats["pm25_std"].fillna(0.0)

    return stats


def assign_balanced_folds(date_stats, n_folds):
    """
    Deterministic balanced assignment.

    1. Sort dates by PM2.5 mean from highest to lowest.
    2. Divide them into batches of n_folds.
    3. Assign each batch in alternating forward/reverse order.
    4. Within each batch, map larger dates to currently smaller folds.

    This guarantees:
    - every fold receives dates;
    - high/medium/low PM2.5 dates are distributed across folds;
    - row counts remain reasonably balanced.
    """
    stats = date_stats.copy()

    if len(stats) < n_folds:
        raise ValueError(
            f"Only {len(stats)} dates are available for {n_folds} folds."
        )

    stats = stats.sort_values(
        ["pm25_mean", "rows"],
        ascending=[False, False],
    ).reset_index(drop=True)

    fold_rows = np.zeros(n_folds, dtype=int)
    fold_dates = [[] for _ in range(n_folds)]
    assignments = []

    for batch_start in range(0, len(stats), n_folds):
        batch = stats.iloc[batch_start:batch_start + n_folds].copy()

        # Put larger dates first so they can be assigned to smaller folds.
        batch = batch.sort_values(
            ["rows", "pm25_mean"],
            ascending=[False, False],
        ).reset_index(drop=True)

        # Folds currently containing the fewest rows receive larger dates.
        available_folds = sorted(
            range(n_folds),
            key=lambda f: (fold_rows[f], len(fold_dates[f]), f),
        )

        # Alternate direction to avoid systematic fold bias.
        batch_number = batch_start // n_folds
        if batch_number % 2 == 1:
            available_folds = list(reversed(available_folds))

        for position, (_, row) in enumerate(batch.iterrows()):
            chosen_fold = available_folds[position]

            fold_rows[chosen_fold] += int(row["rows"])
            fold_dates[chosen_fold].append(row["date"])

            assignments.append({
                "date": row["date"],
                "group_fold": chosen_fold + 1,
                "rows": int(row["rows"]),
                "pm25_mean": float(row["pm25_mean"]),
                "pm25_median": float(row["pm25_median"]),
                "pm25_min": float(row["pm25_min"]),
                "pm25_max": float(row["pm25_max"]),
            })

    assignment_df = pd.DataFrame(assignments)

    counts = assignment_df["group_fold"].value_counts()
    missing_folds = [
        fold
        for fold in range(1, n_folds + 1)
        if fold not in counts.index
    ]

    if missing_folds:
        raise RuntimeError(
            f"Fold assignment failed. Empty folds: {missing_folds}"
        )

    return assignment_df
def build_sequences_for_dates(
    df,
    selected_dates,
    split_name,
    cv_fold,
    T,
):
    selected_dates = set(selected_dates)
    part = df[df["date"].isin(selected_dates)].copy()

    output_rows = []

    for date, day_df in part.groupby("date"):
        day_df = day_df.sort_values(
            ["created_at", "row_id"]
        ).reset_index(drop=True)

        if len(day_df) < T:
            continue

        for end_pos in range(T - 1, len(day_df)):
            window = day_df.iloc[
                end_pos - T + 1:end_pos + 1
            ]
            target = day_df.iloc[end_pos]

            row_ids = window["row_id"].to_numpy()

            # Avoid crossing missing raw rows.
            if not np.all(np.diff(row_ids) == 1):
                continue

            sequence_row = {
                "sequence_id": (
                    f"fold{cv_fold}_{split_name}_"
                    f"{date}_{int(target['row_id'])}"
                ),
                "cv_fold": cv_fold,
                "cv_split": split_name,
                "date": date,
                "target_row_id": int(target["row_id"]),
                "target_created_at": target["created_at"],
                "seq_start_row_id": int(row_ids[0]),
                "seq_end_row_id": int(row_ids[-1]),
                "T": T,
            }

            for target_col in TARGETS:
                sequence_row[target_col] = float(
                    target[target_col]
                )

            output_rows.append(sequence_row)

    return pd.DataFrame(output_rows)


def make_fold_summary(
    manifest,
    sequences,
    fold_assignment,
    n_folds,
):
    rows = []

    for cv_fold in range(1, n_folds + 1):
        for split_name in ["train", "val", "test"]:
            seq_part = sequences[
                (sequences["cv_fold"] == cv_fold)
                & (sequences["cv_split"] == split_name)
            ]

            dates = sorted(
                seq_part["date"].unique().tolist()
            )

            source_rows = manifest[
                manifest["date"].isin(dates)
            ]

            rows.append({
                "cv_fold": cv_fold,
                "split": split_name,
                "dates": "|".join(dates),
                "n_dates": len(dates),
                "raw_rows": len(source_rows),
                "sequences": len(seq_part),
                "PM2.5_mean": (
                    seq_part["PM2.5"].mean()
                    if len(seq_part) else np.nan
                ),
                "PM2.5_std": (
                    seq_part["PM2.5"].std()
                    if len(seq_part) else np.nan
                ),
                "PM2.5_median": (
                    seq_part["PM2.5"].median()
                    if len(seq_part) else np.nan
                ),
                "PM2.5_min": (
                    seq_part["PM2.5"].min()
                    if len(seq_part) else np.nan
                ),
                "PM2.5_max": (
                    seq_part["PM2.5"].max()
                    if len(seq_part) else np.nan
                ),
            })

    return pd.DataFrame(rows)


def verify_no_overlap(sequences, n_folds):
    checks = []

    for cv_fold in range(1, n_folds + 1):
        fold_df = sequences[
            sequences["cv_fold"] == cv_fold
        ]

        split_rows = {}

        for split_name in ["train", "val", "test"]:
            used_rows = set()

            part = fold_df[
                fold_df["cv_split"] == split_name
            ]

            for _, row in part.iterrows():
                used_rows.update(
                    range(
                        int(row["seq_start_row_id"]),
                        int(row["seq_end_row_id"]) + 1,
                    )
                )

            split_rows[split_name] = used_rows

        train_val_overlap = len(
            split_rows["train"] & split_rows["val"]
        )
        train_test_overlap = len(
            split_rows["train"] & split_rows["test"]
        )
        val_test_overlap = len(
            split_rows["val"] & split_rows["test"]
        )

        checks.append({
            "cv_fold": cv_fold,
            "train_val_row_overlap": train_val_overlap,
            "train_test_row_overlap": train_test_overlap,
            "val_test_row_overlap": val_test_overlap,
            "passed": (
                train_val_overlap == 0
                and train_test_overlap == 0
                and val_test_overlap == 0
            ),
        })

    return pd.DataFrame(checks)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default=(
            "experiments/traqid_pretraining_v1/"
            "data/processed/"
            "traqid_paired_manifest_with_splits.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "experiments/traqid_pretraining_v1/"
            "data/processed/"
            "balanced_date_grouped_T7_cv"
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--T", type=int, default=7)

    args = parser.parse_args()

    if args.folds < 3:
        raise ValueError(
            "At least 3 folds are required for "
            "train/validation/test rotation."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = normalize_manifest(
        safe_read_csv(args.manifest)
    )

    date_stats = make_date_stats(manifest)

    fold_assignment = assign_balanced_folds(
        date_stats,
        n_folds=args.folds,
    )
    assigned_folds = set(fold_assignment["group_fold"].unique())
    expected_folds = set(range(1, args.folds + 1))

    if assigned_folds != expected_folds:
        raise RuntimeError(
            f"Invalid fold assignment. "
            f"Expected {sorted(expected_folds)}, "
            f"found {sorted(assigned_folds)}."
        )

    dates_per_fold = fold_assignment.groupby("group_fold")["date"].nunique()

    if (dates_per_fold < 2).any():
        raise RuntimeError(
            "Every fold must contain at least two dates. "
            f"Observed:\n{dates_per_fold}"
        )

    date_to_group = dict(
        zip(
            fold_assignment["date"],
            fold_assignment["group_fold"],
        )
    )

    manifest["group_fold"] = (
        manifest["date"]
        .map(date_to_group)
        .astype(int)
    )

    all_sequence_parts = []

    for cv_fold in range(1, args.folds + 1):
        test_group = cv_fold

        # Cyclic validation group.
        val_group = (
            cv_fold % args.folds
        ) + 1

        train_groups = [
            fold
            for fold in range(1, args.folds + 1)
            if fold not in {test_group, val_group}
        ]

        test_dates = fold_assignment.loc[
            fold_assignment["group_fold"] == test_group,
            "date",
        ].tolist()

        val_dates = fold_assignment.loc[
            fold_assignment["group_fold"] == val_group,
            "date",
        ].tolist()

        train_dates = fold_assignment.loc[
            fold_assignment["group_fold"].isin(
                train_groups
            ),
            "date",
        ].tolist()

        print("\n" + "=" * 90)
        print(f"BUILDING CV FOLD {cv_fold}")
        print("=" * 90)
        print("Train groups:", train_groups)
        print("Validation group:", val_group)
        print("Test group:", test_group)
        print("Train dates:", train_dates)
        print("Validation dates:", val_dates)
        print("Test dates:", test_dates)

        train_seq = build_sequences_for_dates(
            manifest,
            selected_dates=train_dates,
            split_name="train",
            cv_fold=cv_fold,
            T=args.T,
        )

        val_seq = build_sequences_for_dates(
            manifest,
            selected_dates=val_dates,
            split_name="val",
            cv_fold=cv_fold,
            T=args.T,
        )

        test_seq = build_sequences_for_dates(
            manifest,
            selected_dates=test_dates,
            split_name="test",
            cv_fold=cv_fold,
            T=args.T,
        )

        fold_sequences = pd.concat(
            [train_seq, val_seq, test_seq],
            ignore_index=True,
        )

        fold_path = (
            out_dir
            / f"traqid_balanced_date_T{args.T}_fold{cv_fold}.csv"
        )
        fold_sequences.to_csv(fold_path, index=False)

        all_sequence_parts.append(fold_sequences)

        print(
            "Sequences:",
            {
                "train": len(train_seq),
                "val": len(val_seq),
                "test": len(test_seq),
            },
        )
        print("Saved:", fold_path)

    all_sequences = pd.concat(
        all_sequence_parts,
        ignore_index=True,
    )

    summary = make_fold_summary(
        manifest,
        all_sequences,
        fold_assignment,
        args.folds,
    )

    overlap_check = verify_no_overlap(
        all_sequences,
        args.folds,
    )

    assignment_path = (
        out_dir / "balanced_date_fold_assignment.csv"
    )
    summary_path = (
        out_dir / "balanced_date_cv_summary.csv"
    )
    overlap_path = (
        out_dir / "balanced_date_overlap_check.csv"
    )
    all_sequences_path = (
        out_dir
        / f"traqid_balanced_date_T{args.T}_all_folds.csv"
    )
    config_path = out_dir / "config.json"

    fold_assignment.sort_values(
        ["group_fold", "pm25_mean"]
    ).to_csv(assignment_path, index=False)

    summary.to_csv(summary_path, index=False)
    overlap_check.to_csv(overlap_path, index=False)
    all_sequences.to_csv(
        all_sequences_path,
        index=False,
    )

    config = {
        "manifest": args.manifest,
        "folds": args.folds,
        "T": args.T,
        "unique_dates": int(manifest["date"].nunique()),
        "raw_rows": int(len(manifest)),
        "total_sequence_rows_across_cv_folds": int(
            len(all_sequences)
        ),
        "validation_rotation": (
            "validation_group = test_group + 1 cyclically"
        ),
    }

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("BALANCED DATE-GROUPED T7 CV BUILD COMPLETE")
    print("=" * 90)

    print("\nDATE ASSIGNMENT")
    print(
        fold_assignment.sort_values(
            ["group_fold", "pm25_mean"]
        ).to_string(index=False)
    )

    print("\nCV SUMMARY")
    print(summary.to_string(index=False))

    print("\nOVERLAP CHECK")
    print(overlap_check.to_string(index=False))

    print("\nSaved:")
    print(assignment_path)
    print(summary_path)
    print(overlap_path)
    print(all_sequences_path)
    print(config_path)


if __name__ == "__main__":
    main()