"""Reviewer-requested grouped and sensitivity metrics for preserved TRAQID tests.

This script does not fit or select a model.  It reads outer-test predictions
already produced by the declared experiments, then reports metrics by date,
after within-date centering, and after deterministic T=7 temporal thinning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lodo-predictions",
        default="artifacts/runs/traqid_T7_lodo_multisource_hybrid_external_v2/predictions.csv",
    )
    parser.add_argument(
        "--reference-predictions",
        default="artifacts/runs/traqid_T7_forced_localref_context_innercv_lodo_v2/predictions.csv",
    )
    parser.add_argument(
        "--calibration-predictions",
        default="artifacts/runs/traqid_T7_reference_context_chrono10_calibration_v1/predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/runs/traqid_reviewer_requested_audit_v1",
    )
    return parser.parse_args()


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
    }


def summarize(frame: pd.DataFrame, label: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {"analysis": label}
    result.update(metrics(frame.actual.to_numpy(), frame.predicted.to_numpy()))

    by_date = []
    for _, group in frame.groupby("date", sort=True):
        row = metrics(group.actual.to_numpy(), group.predicted.to_numpy())
        by_date.append(row)
    date_metrics = pd.DataFrame(by_date)
    result.update(
        {
            "dates": int(len(date_metrics)),
            "mean_date_mae": float(date_metrics.mae.mean()),
            "mean_date_rmse": float(date_metrics.rmse.mean()),
            "mean_date_r2": float(date_metrics.r2.mean()),
            "median_date_r2": float(date_metrics.r2.median()),
            "positive_date_r2_count": int((date_metrics.r2 > 0).sum()),
        }
    )

    centered = frame.copy()
    centered["actual_centered"] = centered.actual - centered.groupby("date").actual.transform("mean")
    centered["predicted_centered"] = centered.predicted - centered.groupby("date").predicted.transform("mean")
    result["date_centered_r2"] = float(
        r2_score(centered.actual_centered, centered.predicted_centered)
    )
    return result


def thin(frame: pd.DataFrame, stride: int = 7) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("date", sort=True):
        order = "target_created_at" if "target_created_at" in group else "sequence_id"
        parts.append(group.sort_values([order, "sequence_id"], kind="stable").iloc[::stride])
    return pd.concat(parts, ignore_index=True)


def normalized_predictions(
    path: str,
    method_col: str,
    method: str,
    prediction_col: str,
    protocol: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame[method_col].eq(method)].copy()
    if protocol is not None:
        frame = frame.loc[frame["protocol"].eq(protocol)].copy()
    return frame.rename(
        columns={"actual_PM2.5": "actual", prediction_col: "predicted"}
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    lodo = normalized_predictions(
        args.lodo_predictions,
        "method",
        "nonvisual_plus_images_local",
        "predicted_PM2.5",
        protocol="lodo",
    )
    reference = normalized_predictions(
        args.reference_predictions,
        "method",
        "reference_context_local",
        "predicted_PM2.5",
    )
    calibration_all = pd.read_csv(args.calibration_predictions)
    calibration_frames = {}
    for name in ["zero_shot", "meta_ridge_median_offset"]:
        frame = calibration_all.loc[calibration_all.calibration.eq(name)].copy()
        calibration_frames[name] = frame.rename(
            columns={"actual_PM2.5": "actual", "calibrated_PM2.5": "predicted"}
        )

    frames = {
        "20-date zero-shot LODO": lodo,
        "16-date reference-assisted LODO": reference,
        "future-90% zero-shot": calibration_frames["zero_shot"],
        "future-90% meta-calibrated": calibration_frames["meta_ridge_median_offset"],
    }
    grouped = pd.DataFrame([summarize(frame, name) for name, frame in frames.items()])
    grouped.to_csv(output / "grouped_metrics.csv", index=False)

    thinned_rows = []
    for name, frame in frames.items():
        selected = thin(frame)
        row = summarize(selected, name)
        row["stride"] = 7
        thinned_rows.append(row)
    thinned = pd.DataFrame(thinned_rows)
    thinned.to_csv(output / "temporally_thinned_metrics.csv", index=False)

    bins = [-np.inf, 35, 60, 90, 150, np.inf]
    labels = ["<=35", "35-60", "60-90", "90-150", ">150"]
    strata_rows = []
    for name, frame in calibration_frames.items():
        work = frame.copy()
        work["concentration_band"] = pd.cut(work.actual, bins=bins, labels=labels)
        for band, group in work.groupby("concentration_band", observed=True):
            row = {"calibration": name, "concentration_band": str(band)}
            row.update(metrics(group.actual.to_numpy(), group.predicted.to_numpy()))
            strata_rows.append(row)
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(output / "calibration_concentration_strata.csv", index=False)

    merged = calibration_frames["zero_shot"][
        ["date", "sequence_id", "actual", "predicted"]
    ].rename(columns={"predicted": "zero_shot"}).merge(
        calibration_frames["meta_ridge_median_offset"][
            ["date", "sequence_id", "predicted"]
        ].rename(columns={"predicted": "calibrated"}),
        on=["date", "sequence_id"],
        validate="one_to_one",
    )
    influence_rows = []
    for excluded in ["none", *sorted(merged.date.astype(str).unique())]:
        use = merged if excluded == "none" else merged.loc[merged.date.astype(str).ne(excluded)]
        for prediction in ["zero_shot", "calibrated"]:
            row = {"excluded_date": excluded, "method": prediction}
            row.update(metrics(use.actual.to_numpy(), use[prediction].to_numpy()))
            influence_rows.append(row)
    influence = pd.DataFrame(influence_rows)
    influence.to_csv(output / "calibration_leave_one_date_influence.csv", index=False)

    date_rows = []
    for date, group in merged.groupby("date", sort=True):
        zero = metrics(group.actual.to_numpy(), group.zero_shot.to_numpy())
        calibrated = metrics(group.actual.to_numpy(), group.calibrated.to_numpy())
        date_rows.append(
            {
                "date": date,
                "zero_shot_rmse": zero["rmse"],
                "calibrated_rmse": calibrated["rmse"],
                "rmse_reduction": zero["rmse"] - calibrated["rmse"],
            }
        )
    date_effects = pd.DataFrame(date_rows)
    date_effects.to_csv(output / "calibration_date_effects.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    x = np.arange(len(labels))
    width = 0.36
    for offset, (name, color) in zip([-width / 2, width / 2], [("zero_shot", "#6b7280"), ("meta_ridge_median_offset", "#1f5f8b")]):
        sub = strata.loc[strata.calibration.eq(name)].set_index("concentration_band").reindex(labels)
        axes[0].bar(x + offset, sub.rmse, width, label=name.replace("_", " "), color=color)
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Measured PM$_{2.5}$ band ($\mu$g m$^{-3}$)")
    axes[0].set_ylabel("RMSE ($\mu$g m$^{-3}$)")
    axes[0].set_title("Error by concentration")
    axes[0].legend(frameon=False, fontsize=8)

    ordered = date_effects.sort_values("rmse_reduction")
    colors = np.where(ordered.rmse_reduction >= 0, "#2a7f86", "#b45309")
    axes[1].barh(ordered.date.astype(str), ordered.rmse_reduction, color=colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("RMSE reduction after calibration")
    axes[1].set_title("Date-level calibration effect")
    axes[1].tick_params(axis="y", labelsize=6.5)
    fig.tight_layout()
    fig.savefig(output / "traqid_grouped_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "traqid_grouped_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("GROUPED METRICS")
    print(grouped.to_string(index=False))
    print("\nTEMPORALLY THINNED METRICS")
    print(thinned.to_string(index=False))
    print("\nCONCENTRATION STRATA")
    print(strata.to_string(index=False))
    print("\nCALIBRATION DATE EFFECTS")
    print(date_effects.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
