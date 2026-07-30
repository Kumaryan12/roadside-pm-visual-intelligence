import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


TARGET_COLS = ["PM2.5", "PM10", "aqi"]


def parse_row_ids(value):
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if "|" in value:
            return [int(x) for x in value.split("|") if x != ""]
        if "," in value:
            return [int(x) for x in value.split(",") if x != ""]
        return [int(value)]
    return [int(value)]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true, y_pred, target_cols):
    rows = []

    r2s = []
    rmses = []
    maes = []

    for i, col in enumerate(target_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        r2 = float(r2_score(yt, yp))
        r = rmse(yt, yp)
        mae = float(mean_absolute_error(yt, yp))

        rows.append({
            "target": col,
            "R2": r2,
            "RMSE": r,
            "MAE": mae,
        })

        r2s.append(r2)
        rmses.append(r)
        maes.append(mae)

    rows.append({
        "target": "Average",
        "R2": float(np.mean(r2s)),
        "RMSE": float(np.mean(rmses)),
        "MAE": float(np.mean(maes)),
    })

    return rows


def build_y_from_target_ids(seq_df, base_by_row_id, target_id_col, target_cols):
    target_ids = seq_df[target_id_col].astype(int).values
    y = []
    for rid in target_ids:
        y.append(base_by_row_id.loc[rid, target_cols].astype(float).values)
    return np.asarray(y, dtype=float)


def build_persistence_from_sequences(seq_df, base_by_row_id, input_col, target_cols):
    preds = []

    for value in seq_df[input_col].values:
        row_ids = parse_row_ids(value)

        if len(row_ids) < 2:
            raise ValueError(
                "Persistence baseline requires at least 2 input rows per sequence. "
                f"Got sequence: {value}"
            )

        # IMPORTANT:
        # The final row in a T=7 estimation sequence is the target timestamp.
        # Therefore, using the final row as persistence would leak the target.
        # We use the penultimate row as a previous-step persistence baseline.
        prev_row_id = row_ids[-2]

        preds.append(base_by_row_id.loc[prev_row_id, target_cols].astype(float).values)

    return np.asarray(preds, dtype=float)


def evaluate_protocol(
    seq_df,
    base_by_row_id,
    split_col,
    train_name,
    eval_name,
    input_col,
    target_id_col,
    target_cols,
):
    train_df = seq_df[seq_df[split_col] == train_name].copy()
    eval_df = seq_df[seq_df[split_col] == eval_name].copy()

    if train_df.empty:
        raise ValueError(f"No train rows found for {split_col} == {train_name}")
    if eval_df.empty:
        raise ValueError(f"No eval rows found for {split_col} == {eval_name}")

    y_train = build_y_from_target_ids(train_df, base_by_row_id, target_id_col, target_cols)
    y_eval = build_y_from_target_ids(eval_df, base_by_row_id, target_id_col, target_cols)

    train_mean = np.mean(y_train, axis=0, keepdims=True)
    pred_train_mean = np.repeat(train_mean, repeats=len(eval_df), axis=0)

    pred_persistence = build_persistence_from_sequences(
        eval_df,
        base_by_row_id,
        input_col,
        target_cols,
    )

    all_rows = []

    for model_name, pred in [
        ("train_mean", pred_train_mean),
        ("previous_step_persistence", pred_persistence),
    ]:
        metric_rows = compute_metrics(y_eval, pred, target_cols)

        for r in metric_rows:
            out = {
                "model": model_name,
                "split_col": split_col,
                "train_name": train_name,
                "eval_name": eval_name,
                "train_sequences": len(train_df),
                "eval_sequences": len(eval_df),
                **r,
            }
            all_rows.append(out)

    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate naive baselines for T=7 estimation sequence manifests."
    )

    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--split-col", required=True)
    parser.add_argument("--train-name", required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--input-col", default="seq_row_ids")
    parser.add_argument("--target-id-col", default="target_row_id")
    parser.add_argument("--row-id-col", default="row_id")
    parser.add_argument("--target-cols", nargs="+", default=TARGET_COLS)
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    seq_path = Path(args.sequence_manifest)
    base_path = Path(args.base_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_df = pd.read_csv(seq_path)
    base_df = pd.read_csv(base_path)

    missing_seq_cols = [
        c for c in [args.split_col, args.input_col, args.target_id_col]
        if c not in seq_df.columns
    ]
    if missing_seq_cols:
        raise ValueError(f"Missing sequence-manifest columns: {missing_seq_cols}")

    missing_base_cols = [
        c for c in [args.row_id_col] + args.target_cols
        if c not in base_df.columns
    ]
    if missing_base_cols:
        raise ValueError(f"Missing base-manifest columns: {missing_base_cols}")

    base_df = base_df.copy()
    base_df[args.row_id_col] = base_df[args.row_id_col].astype(int)
    base_by_row_id = base_df.set_index(args.row_id_col)

    rows = evaluate_protocol(
        seq_df=seq_df,
        base_by_row_id=base_by_row_id,
        split_col=args.split_col,
        train_name=args.train_name,
        eval_name=args.eval_name,
        input_col=args.input_col,
        target_id_col=args.target_id_col,
        target_cols=args.target_cols,
    )

    results = pd.DataFrame(rows)

    csv_path = out_dir / f"estimation_baselines_{args.split_col}_{args.eval_name}.csv"
    md_path = out_dir / f"estimation_baselines_{args.split_col}_{args.eval_name}.md"

    results.to_csv(csv_path, index=False)
    md_path.write_text(results.to_markdown(index=False))

    avg = results[results["target"] == "Average"].copy()

    print("=" * 90)
    print("ESTIMATION BASELINES")
    print("=" * 90)
    print("Sequence manifest:", seq_path)
    print("Base manifest:", base_path)
    print("Split:", args.split_col)
    print("Train:", args.train_name)
    print("Eval:", args.eval_name)
    print("Targets:", args.target_cols)
    print()
    print(avg.to_string(index=False))
    print()
    print("Saved:", csv_path)
    print("Saved:", md_path)


if __name__ == "__main__":
    main()