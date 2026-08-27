"""Generate Taiwan baselines and grouped uncertainty audits.

This analysis does not refit or alter the frozen conditioned-image model.  It
evaluates transparent monitoring-network baselines on the identical 54-date
chronological test partition, bootstraps paired model differences by complete
date, and simulates the loss of one supporting station at inference time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    keep = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[keep]
    predicted = predicted[keep]
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
        "coverage_fraction": float(keep.mean()),
    }


def other_site_summary(
    frame: pd.DataFrame,
    wide: pd.DataFrame,
    sites: list[str],
    lag_hours: int = 0,
    removed_site: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    times = pd.DatetimeIndex(frame["target_time"] - pd.to_timedelta(lag_hours, unit="h"))
    values = wide.reindex(times).loc[:, sites].to_numpy(dtype=float, copy=True)
    targets = frame["site"].astype(str).to_numpy()
    for index, site in enumerate(sites):
        values[targets == site, index] = np.nan
        if removed_site == site:
            values[:, index] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmean(values, axis=1), np.nanmedian(values, axis=1)


def paired_date_bootstrap(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    comparisons: list[tuple[str, str]],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = frame["date"].astype(str).to_numpy()
    unique_dates = np.array(sorted(set(dates)))
    actual = frame["actual_pm25"].to_numpy(dtype=float)
    rows: list[dict[str, float | str | int]] = []
    samples: list[dict[str, float | str | int]] = []
    indices = {date: np.flatnonzero(dates == date) for date in unique_dates}
    for baseline, candidate in comparisons:
        base = predictions[baseline]
        cand = predictions[candidate]
        boot = {"mae_reduction": [], "rmse_reduction": [], "r2_gain": []}
        for replicate in range(replicates):
            sampled = rng.choice(unique_dates, size=len(unique_dates), replace=True)
            selected = np.concatenate([indices[date] for date in sampled])
            base_score = metrics(actual[selected], base[selected])
            cand_score = metrics(actual[selected], cand[selected])
            values = {
                "mae_reduction": base_score["mae"] - cand_score["mae"],
                "rmse_reduction": base_score["rmse"] - cand_score["rmse"],
                "r2_gain": cand_score["r2"] - base_score["r2"],
            }
            for metric, value in values.items():
                boot[metric].append(value)
                samples.append(
                    {
                        "baseline": baseline,
                        "candidate": candidate,
                        "replicate": replicate,
                        "metric": metric,
                        "value": value,
                    }
                )
        for metric, values in boot.items():
            array = np.asarray(values)
            rows.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "median": float(np.median(array)),
                    "ci_low_95": float(np.quantile(array, 0.025)),
                    "ci_high_95": float(np.quantile(array, 0.975)),
                    "probability_improvement": float(np.mean(array > 0)),
                    "bootstrap_unit": "complete test date",
                    "replicates": replicates,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--frozen-test-predictions", required=True)
    parser.add_argument("--exploratory-test-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw_manifest, low_memory=False)
    raw = raw.loc[raw["dataset"].eq("taiwan"), ["site", "timestamp", "pm25"]].copy()
    raw["site"] = raw["site"].astype(str)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="raise")
    raw["pm25"] = pd.to_numeric(raw["pm25"], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "pm25"]).drop_duplicates(["site", "timestamp"])
    sites = sorted(raw["site"].unique())
    wide = raw.pivot(index="timestamp", columns="site", values="pm25").sort_index()

    sequences = pd.read_csv(args.sequence_manifest)
    sequences["site"] = sequences["site"].astype(str)
    sequences["target_time"] = pd.to_datetime(sequences["target_time"], errors="raise")
    sequences["date"] = sequences["target_time"].dt.date.astype(str)
    frozen = pd.read_csv(args.frozen_test_predictions)
    exploratory = pd.read_csv(args.exploratory_test_predictions)
    test = sequences.loc[sequences["split"].eq("test")].copy()
    test = test.merge(frozen, on="sequence_id", how="inner", validate="one_to_one")
    test = test.merge(
        exploratory.drop(columns=["actual_pm25"]),
        on="sequence_id",
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    if len(test) != len(frozen):
        raise RuntimeError(f"Test merge lost rows: {len(test)} != {len(frozen)}")
    if not np.allclose(test["target_pm25"], test["actual_pm25"]):
        raise RuntimeError("Target mismatch after prediction merge")

    mean_current, median_current = other_site_summary(test, wide, sites)
    _, median_lag1 = other_site_summary(test, wide, sites, lag_hours=1)
    if not np.allclose(
        median_current,
        test["predicted_reference_only"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise RuntimeError("Reconstructed network median does not match frozen prediction")

    train = sequences.loc[sequences["split"].eq("train")].copy()
    train["target_time"] = pd.to_datetime(train["target_time"], errors="raise")
    train["hour"] = train["target_time"].dt.hour
    test["hour"] = test["target_time"].dt.hour
    site_hour = train.groupby(["site", "hour"])["target_pm25"].mean()
    site_mean = train.groupby("site")["target_pm25"].mean()
    global_mean = float(train["target_pm25"].mean())
    climatology = np.array(
        [
            site_hour.get((site, hour), site_mean.get(site, global_mean))
            for site, hour in zip(test["site"], test["hour"])
        ],
        dtype=float,
    )

    predictions = {
        "global_training_mean": np.full(len(test), global_mean),
        "target_site_hour_climatology": climatology,
        "other_site_median_lag1h": median_lag1,
        "other_site_mean_current": mean_current,
        "other_site_median_current": median_current,
        "context_only_frozen": test["predicted_reference_context"].to_numpy(dtype=float),
        "conditioned_image_frozen": test[
            "predicted_reference_context_plus_conditioned_image"
        ].to_numpy(dtype=float),
        "learned_spatial_background_posthoc": test[
            "predicted_learned_spatial_background"
        ].to_numpy(dtype=float),
        "lagged_spatiotemporal_background_posthoc": test[
            "predicted_lagged_spatiotemporal_background"
        ].to_numpy(dtype=float),
        "validation_ensemble_posthoc": test[
            "predicted_validation_selected_ensemble"
        ].to_numpy(dtype=float),
    }
    actual = test["actual_pm25"].to_numpy(dtype=float)
    baseline_rows = []
    for name, predicted in predictions.items():
        baseline_rows.append({"model": name, **metrics(actual, predicted)})
    baseline_table = pd.DataFrame(baseline_rows).sort_values("rmse")
    baseline_table.to_csv(output / "chronological_baselines.csv", index=False)

    comparisons = [
        ("other_site_median_current", "context_only_frozen"),
        ("other_site_median_current", "conditioned_image_frozen"),
        ("context_only_frozen", "conditioned_image_frozen"),
        ("other_site_median_current", "learned_spatial_background_posthoc"),
    ]
    summary, samples = paired_date_bootstrap(
        test, predictions, comparisons, args.bootstrap_replicates, args.seed
    )
    summary.to_csv(output / "paired_date_bootstrap_summary.csv", index=False)
    samples.to_csv(output / "paired_date_bootstrap_samples.csv", index=False)

    date_rows = []
    for date, group in test.groupby("date", sort=True):
        index = group.index.to_numpy()
        for name in [
            "other_site_median_current",
            "context_only_frozen",
            "conditioned_image_frozen",
            "learned_spatial_background_posthoc",
        ]:
            date_rows.append({"date": date, "model": name, **metrics(actual[index], predictions[name][index])})
    date_table = pd.DataFrame(date_rows)
    date_table.to_csv(output / "datewise_metrics.csv", index=False)
    date_summary = (
        date_table.groupby("model")
        .agg(
            dates=("date", "nunique"),
            mean_date_r2=("r2", "mean"),
            median_date_r2=("r2", "median"),
            positive_date_r2_count=("r2", lambda values: int((values > 0).sum())),
            mean_date_rmse=("rmse", "mean"),
            worst_date_rmse=("rmse", "max"),
        )
        .reset_index()
    )
    date_summary.to_csv(output / "datewise_summary.csv", index=False)

    outage_rows = [{"removed_support_site": "none", **metrics(actual, median_current)}]
    for removed in sites:
        _, predicted = other_site_summary(test, wide, sites, removed_site=removed)
        outage_rows.append({"removed_support_site": removed, **metrics(actual, predicted)})
    outage = pd.DataFrame(outage_rows)
    outage.to_csv(output / "support_station_outage_sensitivity.csv", index=False)

    plot_order = [
        "global_training_mean",
        "target_site_hour_climatology",
        "other_site_median_lag1h",
        "other_site_mean_current",
        "other_site_median_current",
        "conditioned_image_frozen",
    ]
    labels = [
        "Train\nmean",
        "Site-hour\nclimatology",
        "Network median\n(previous hour)",
        "Network mean\n(current hour)",
        "Network median\n(current hour)",
        "Complete\nmodel",
    ]
    values = [float(baseline_table.set_index("model").loc[name, "rmse"]) for name in plot_order]
    colors = ["#8C939D"] * 5 + ["#3B6EA8"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.5))
    axes[0].bar(np.arange(len(values)), values, color=colors)
    axes[0].set_xticks(np.arange(len(values)), labels, rotation=0, ha="center")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("RMSE ($\\mu$g m$^{-3}$)")
    axes[0].set_title("Identical 54-date chronological test partition")
    axes[0].grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.15, f"{value:.2f}", ha="center", fontsize=8)

    outage_values = outage["rmse"].to_numpy(dtype=float)
    outage_labels = outage["removed_support_site"].astype(str).tolist()
    outage_colors = ["#3B6EA8"] + ["#8C939D"] * (len(outage_values) - 1)
    axes[1].bar(np.arange(len(outage_values)), outage_values, color=outage_colors)
    axes[1].set_xticks(np.arange(len(outage_values)), outage_labels)
    axes[1].set_xlabel("Supporting station removed")
    axes[1].set_ylabel("Network-median RMSE ($\\mu$g m$^{-3}$)")
    axes[1].set_title("Single-support-station outage sensitivity")
    axes[1].grid(axis="y", alpha=0.25)
    for i, value in enumerate(outage_values):
        axes[1].text(i, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "reviewer_baselines_and_outages.pdf", bbox_inches="tight")
    fig.savefig(output / "reviewer_baselines_and_outages.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "protocol": "chronological_complete_date_60_20_20",
        "test_rows": int(len(test)),
        "test_dates": int(test["date"].nunique()),
        "target_site_current_pm25_used_as_predictor": False,
        "other_site_contemporaneous_pm25_required_at_inference": True,
        "frozen_model_refit": False,
        "new_baselines_status": "retrospective baseline sensitivity analysis on previously inspected test",
        "posthoc_models": [
            "learned_spatial_background_posthoc",
            "lagged_spatiotemporal_background_posthoc",
            "validation_ensemble_posthoc",
        ],
    }
    (output / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(baseline_table.to_string(index=False))
    print("\nPAIRED DATE BOOTSTRAP\n", summary.to_string(index=False))
    print("\nDATEWISE SUMMARY\n", date_summary.to_string(index=False))
    print("\nOUTAGE SENSITIVITY\n", outage.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
