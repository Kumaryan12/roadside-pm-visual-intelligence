"""Matched random-window TRAQID direct/CAMS/MERRA-2/GEOS-CF diagnostic.

All variants reuse the same T=7 sequence manifest, ``split_random`` assignment,
ResNet50 embeddings, seed, and non-PM residual predictors.  Random overlapping
windows are intentionally retained only to compare with historical TRAQID
benchmarks; the resulting metrics are not generalization estimates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.pm25_prediction.mumma_7day.model_current_data import estimators
from pipelines.pm25_prediction.mumma_7day.run_residual_fusion_current_data import (
    metric_row,
)


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "direct": None,
    "cams": "background_cams_pm25_ug_m3",
    "merra2": "background_merra2_pm25_ug_m3",
}
GEOS_CF = "background_geos_cf_pm25_ug_m3"


def overlap_audit(sequences: pd.DataFrame) -> dict[str, object]:
    exploded = sequences[["split_random", "seq_row_ids"]].copy()
    exploded["row_id"] = exploded["seq_row_ids"].astype(str).str.split("|")
    exploded = exploded.explode("row_id")
    memberships = exploded.groupby("row_id")["split_random"].nunique()
    split_sets = {
        split: set(group["row_id"].astype(str))
        for split, group in exploded.groupby("split_random")
    }
    train_context = split_sets.get("train", set())
    test_targets = set(
        sequences.loc[
            sequences["split_random"].eq("test"), "target_row_id"
        ].astype(str)
    )
    leaked_targets = test_targets & train_context
    return {
        "raw_rows_in_multiple_splits": int((memberships > 1).sum()),
        "unique_raw_rows": int(len(memberships)),
        "train_val_raw_overlap": int(
            len(split_sets.get("train", set()) & split_sets.get("val", set()))
        ),
        "train_test_raw_overlap": int(
            len(split_sets.get("train", set()) & split_sets.get("test", set()))
        ),
        "val_test_raw_overlap": int(
            len(split_sets.get("val", set()) & split_sets.get("test", set()))
        ),
        "test_targets_seen_as_train_context": int(len(leaked_targets)),
        "test_target_context_leakage_fraction": float(
            len(leaked_targets) / max(len(test_targets), 1)
        ),
        "leakage_contaminated": True,
    }


def residual_columns(table: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in table.select_dtypes(include=[np.number]).columns:
        name = column.lower()
        if column in {
            "sequence_id",
            "target_row_id",
            "target_image_id",
            "seq_start_row_id",
            "seq_end_row_id",
            "T",
            "PM2.5",
            "PM10",
            "aqi",
        }:
            continue
        if any(
            token in name
            for token in ("pm2.5", "pm10", "aqi", "row_id", "image_id", "date_fold")
        ):
            continue
        columns.append(column)
    if not columns:
        raise ValueError("No non-PM numeric residual features were selected")
    return columns


def attach(
    path: Path,
    manifest: pd.DataFrame,
    engineered: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    result = predictions.merge(
        manifest[
            [
                "sequence_id",
                "target_row_id",
                "target_PM2.5",
                "_background",
            ]
        ],
        on="sequence_id",
        validate="one_to_one",
    )
    return result.merge(
        engineered[["sequence_id", *features]],
        on="sequence_id",
        validate="one_to_one",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-manifest",
        default=(
            "experiments/traqid_pretraining_v1/data/processed/"
            "paper_style_sequences/"
            "traqid_paper_style_T7_front_sequence_manifest_time_balanced_purged.csv"
        ),
    )
    parser.add_argument(
        "--embeddings",
        default=(
            "experiments/traqid_pretraining_v1/embeddings/paper_style_cnn/"
            "traqid_paper_resnet50_front_rear_concat_gap_features.npy"
        ),
    )
    parser.add_argument(
        "--engineered-table",
        default=(
            "experiments/traqid_pretraining_v1/data/processed/"
            "traqid_engineered_T7_sequence_table.csv"
        ),
    )
    parser.add_argument("--background-csv", required=True)
    parser.add_argument(
        "--geos-cf-csv",
        help=(
            "Optional sample-level GEOS-CF table. When supplied, a matched "
            "GEOS-CF local-increment variant is added."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequences = pd.read_csv(args.sequence_manifest)
    required = {
        "sequence_id",
        "split_random",
        "seq_row_ids",
        "target_row_id",
        "target_PM2.5",
        "embedding_rows",
    }
    missing = sorted(required - set(sequences.columns))
    if missing:
        if "embedding_rows" in missing and "seq_row_ids" in sequences:
            sequences["embedding_rows"] = sequences["seq_row_ids"]
            missing.remove("embedding_rows")
        if missing:
            raise ValueError(f"TRAQID sequence manifest is missing: {missing}")
    if not {"train", "val", "test"}.issubset(set(sequences["split_random"])):
        raise ValueError("split_random must contain train, val, and test")

    background = pd.read_csv(args.background_csv)
    background["sample_id"] = background["sample_id"].astype(str)
    variants = dict(VARIANTS)
    if args.geos_cf_csv:
        geos_cf = pd.read_csv(args.geos_cf_csv)
        geos_cf["sample_id"] = geos_cf["sample_id"].astype(str)
        if geos_cf["sample_id"].duplicated().any():
            raise ValueError("GEOS-CF sample_id must be unique")
        if GEOS_CF not in geos_cf.columns:
            raise ValueError(f"GEOS-CF table is missing: {GEOS_CF}")
        background = background.merge(
            geos_cf[["sample_id", GEOS_CF]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        variants["geos_cf"] = GEOS_CF
    needed_background = {
        "sample_id",
        "background_status",
        *[column for column in variants.values() if column is not None],
    }
    missing_background = sorted(needed_background - set(background.columns))
    if missing_background:
        raise ValueError(f"Background table is missing: {missing_background}")
    if not background["background_status"].eq("success").all():
        raise ValueError("Background table contains unsuccessful rows")
    background = background.set_index("sample_id")

    engineered = pd.read_csv(args.engineered_table)
    if engineered["sequence_id"].duplicated().any():
        raise ValueError("Engineered table sequence_id must be unique")
    features = residual_columns(engineered)
    (output / "residual_feature_columns.txt").write_text(
        "\n".join(features) + "\n"
    )
    audit = overlap_audit(sequences)
    (output / "overlap_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    all_metrics: list[pd.DataFrame] = []
    for variant, background_column in variants.items():
        variant_dir = output / variant
        model_dir = variant_dir / f"{args.cell}_T7"
        variant_dir.mkdir(parents=True, exist_ok=True)
        manifest = sequences.copy()
        manifest["target_row_id"] = manifest["target_row_id"].astype(str)
        if background_column is None:
            manifest["_background"] = 0.0
            target = "target_PM2.5"
        else:
            values = manifest["target_row_id"].map(background[background_column])
            if values.isna().any():
                raise ValueError(
                    f"{int(values.isna().sum())} sequence targets lack "
                    f"{background_column}"
                )
            manifest["_background"] = values.to_numpy(dtype=float)
            target = f"target_PM2.5_{variant}_local_increment"
            manifest[target] = (
                manifest["target_PM2.5"].to_numpy(dtype=float)
                - manifest["_background"].to_numpy(dtype=float)
            )
        manifest_path = variant_dir / "sequence_manifest_random.csv"
        manifest.to_csv(manifest_path, index=False)
        command = [
            sys.executable,
            str(ROOT / "pipelines/image_embeddings/train_rnn.py"),
            "--sequence-manifest",
            str(manifest_path),
            "--embeddings",
            args.embeddings,
            "--split-col",
            "split_random",
            "--target-cols",
            target,
            "--cell",
            args.cell,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--output-dir",
            str(model_dir),
        ]
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        if not (args.resume and (model_dir / "metrics.json").is_file()):
            subprocess.run(command, cwd=ROOT, env=environment, check=True)

        validation = attach(
            model_dir / "predictions_val.csv", manifest, engineered, features
        )
        test = attach(
            model_dir / "predictions_test.csv", manifest, engineered, features
        )
        actual_column = f"actual_{target}"
        predicted_column = f"predicted_{target}"
        validation_residual = (
            validation[actual_column].to_numpy(dtype=float)
            - validation[predicted_column].to_numpy(dtype=float)
        )
        test_truth = (
            test[actual_column].to_numpy(dtype=float)
            + test["_background"].to_numpy(dtype=float)
        )
        base_prediction = (
            test[predicted_column].to_numpy(dtype=float)
            + test["_background"].to_numpy(dtype=float)
        )
        rows = [
            {
                "variant": variant,
                "correction_model": "image_base",
                "n_features": 0,
                "n_test": len(test),
                **metric_row(test_truth, base_prediction),
            }
        ]
        prediction_tables = []
        for model_name, model in estimators(args.seed).items():
            model.fit(validation[features], validation_residual)
            corrected = base_prediction + model.predict(test[features])
            rows.append(
                {
                    "variant": variant,
                    "correction_model": model_name,
                    "n_features": len(features),
                    "n_test": len(test),
                    **metric_row(test_truth, corrected),
                }
            )
            prediction_tables.append(
                pd.DataFrame(
                    {
                        "sequence_id": test["sequence_id"],
                        "target_row_id": test["target_row_id"],
                        "actual_PM2.5": test_truth,
                        "base_prediction_PM2.5": base_prediction,
                        "corrected_prediction_PM2.5": corrected,
                        "variant": variant,
                        "correction_model": model_name,
                    }
                )
            )
        metrics = pd.DataFrame(rows).sort_values("rmse")
        metrics.to_csv(variant_dir / "metrics.csv", index=False)
        pd.concat(prediction_tables, ignore_index=True).to_csv(
            variant_dir / "predictions.csv", index=False
        )
        all_metrics.append(metrics)
        print(f"\n{variant.upper()}\n{metrics.to_string(index=False)}", flush=True)

    if not args.dry_run:
        combined = pd.concat(all_metrics, ignore_index=True).sort_values("rmse")
        combined.to_csv(output / "metrics_comparison.csv", index=False)
        run = {
            "dataset": "TRAQID",
            "protocol": "random_overlapping_T7_windows",
            "reportable_as_generalization": False,
            "variants": list(variants),
            "cell": args.cell,
            "seed": args.seed,
            "residual_feature_count": len(features),
            "background_csv": args.background_csv,
            "overlap_audit": audit,
            "warning": (
                "Random overlapping sequences share raw images across splits. "
                "Metrics are historical diagnostics only."
            ),
        }
        (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        print("\nMATCHED COMPARISON\n" + combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
