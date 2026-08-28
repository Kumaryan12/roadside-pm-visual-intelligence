#!/usr/bin/env python3
"""Rebuild the journal figures from preserved data and model outputs.

The script intentionally uses only deterministic plotting primitives and
stored observations/predictions.  It does not use image-generation models.
The illustrative dataset-scene montage is not regenerated here because the
six exact Taiwan source JPEGs recorded in its provenance note are not present
in the current workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/environmental_modelling_software_submission_v1"
    / "figures_rebuilt_code_v1"
)

TRAQID_MANIFEST = ROOT / "experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv"
TAIWAN_SEQUENCES = ROOT / "artifacts/runs/taiwan_chronological_reference_v1/sequences_reference.csv"
TAIWAN_PREDICTIONS = ROOT / "artifacts/runs/taiwan_chronological_reference_v1/models/conservative_conditioned_resnet50_gru_T7/predictions_test.csv"
TAIWAN_SPLIT = ROOT / "artifacts/runs/taiwan_chronological_reference_v1/split_audit.json"
TAIWAN_AUDIT = ROOT / "artifacts/runs/taiwan_reviewer_requested_audit_submission_v1"
TRAQID_CAL = ROOT / "artifacts/runs/traqid_T7_reference_context_chrono10_calibration_v1"
TRAQID_AUDIT = ROOT / "artifacts/runs/traqid_reviewer_requested_audit_v1"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[keep], predicted[keep]
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "bias": float(np.mean(predicted - actual)),
    }


def load_taiwan() -> pd.DataFrame:
    seq = pd.read_csv(TAIWAN_SEQUENCES)
    seq["target_time"] = pd.to_datetime(seq["target_time"])
    seq["date"] = seq["target_time"].dt.date.astype(str)
    pred = pd.read_csv(TAIWAN_PREDICTIONS)
    test = seq.loc[seq["split"].eq("test")].merge(
        pred, on="sequence_id", how="inner", validate="one_to_one"
    )
    if len(test) != len(pred):
        raise RuntimeError("Taiwan test merge lost rows")
    if not np.allclose(test.target_pm25, test.actual_pm25):
        raise RuntimeError("Taiwan target mismatch")
    return test.sort_values(["target_time", "site", "sequence_id"]).reset_index(drop=True)


def figure_01_traqid_distribution(output: Path) -> dict[str, object]:
    df = pd.read_csv(TRAQID_MANIFEST, low_memory=False)
    df["date"] = df["date"].astype(str)
    dates = sorted(df.date.unique())
    counts = df.groupby("date").size().reindex(dates)
    seasons = df.groupby("date").Season.first().reindex(dates)
    season_hatch = {"Monsoon": "", "Winter": "//", "Summer": "xx"}

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    x = np.arange(len(dates))
    for season in ["Monsoon", "Winter", "Summer"]:
        mask = seasons.eq(season).to_numpy()
        axes[0].bar(x[mask], counts.to_numpy()[mask], color="#666666", hatch=season_hatch[season], label=season, edgecolor="white", linewidth=0.4)
    axes[0].set_ylabel("T=7 sequences")
    axes[0].set_title("TRAQID coverage across 20 complete collection dates")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")

    values = [df.loc[df.date.eq(date), "PM2.5"].dropna().to_numpy() for date in dates]
    bp = axes[1].boxplot(values, positions=x, widths=0.62, patch_artist=True, showfliers=False)
    for box, date in zip(bp["boxes"], dates):
        box.set(facecolor="#b5b5b5", edgecolor="#555555", linewidth=0.7)
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set(color="#444444", linewidth=0.75)
    axes[1].set_ylabel(r"Roadside PM$_{2.5}$ ($\mu$g m$^{-3}$)")
    axes[1].set_xticks(x, dates, rotation=50, ha="right", fontsize=6.5)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, output, "Figure_01_TRAQID_coverage_and_target_distribution")
    return {"rows": int(len(df)), "dates": len(dates), "date_counts": counts.to_dict()}


def box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fontsize: float = 8.0) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor="#fafafa", edgecolor="#222222", linewidth=0.8)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=0.9, color="#222222", shrinkA=2, shrinkB=2))


def panel(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor="white", edgecolor="#333333", linewidth=0.9))
    ax.text(x + 0.018, y + h - 0.035, title, ha="left", va="top", fontsize=9.2, fontweight="bold")


def figure_03_architecture(output: Path) -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(13.0, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel(ax, (0.02, 0.70), (0.70, 0.27), "A. BACKGROUND SIGNAL")
    panel(ax, (0.02, 0.38), (0.70, 0.27), "B. VISUAL LOCAL-INCREMENT BRANCH")
    panel(ax, (0.02, 0.06), (0.70, 0.27), "C. STRUCTURED CONTEXT BRANCH")
    panel(ax, (0.75, 0.06), (0.23, 0.91), "D. VALIDATION-SELECTED\nFUSION & OUTPUT")

    box(ax, (0.06, 0.785), (0.16, 0.10), "Timestamp + location\n$(t,s)$")
    box(ax, (0.29, 0.785), (0.19, 0.10), "External atmospheric product\nor synchronized monitoring network", 7.2)
    box(ax, (0.55, 0.785), (0.13, 0.10), "Background estimate\n$B(t,s)$")
    arrow(ax, (0.22, 0.835), (0.29, 0.835)); arrow(ax, (0.48, 0.835), (0.55, 0.835))
    ax.text(0.37, 0.735, "Independent of the target observation being predicted", ha="center", fontsize=7.0)

    box(ax, (0.045, 0.455), (0.12, 0.10), "$T=7$ image sequence")
    box(ax, (0.215, 0.455), (0.14, 0.10), "ResNet50\nvisual embeddings")
    box(ax, (0.405, 0.455), (0.14, 0.10), "Temporal encoder\nGRU / conditioned GRU", 7.2)
    box(ax, (0.595, 0.455), (0.105, 0.10), "Visual local estimate\n$\widehat{\Delta}_{visual}$", 7.2)
    arrow(ax, (0.165, 0.505), (0.215, 0.505)); arrow(ax, (0.355, 0.505), (0.405, 0.505)); arrow(ax, (0.545, 0.505), (0.595, 0.505))

    box(ax, (0.045, 0.135), (0.18, 0.10), "Available atmospheric, temporal,\ntraffic, road and site context", 7.0)
    box(ax, (0.31, 0.135), (0.14, 0.10), "Tree-based\nlocal model")
    box(ax, (0.535, 0.135), (0.165, 0.10), "Context local estimate\n$\widehat{\Delta}_{context}$")
    arrow(ax, (0.225, 0.185), (0.31, 0.185)); arrow(ax, (0.45, 0.185), (0.535, 0.185))

    box(ax, (0.785, 0.755), (0.16, 0.10), "Validation-selected fusion\n$F(\widehat{\Delta}_{visual},\widehat{\Delta}_{context})$", 7.1)
    box(ax, (0.785, 0.575), (0.16, 0.09), "Fused local increment\n$\widehat{\Delta}_{local}$")
    box(ax, (0.785, 0.395), (0.16, 0.09), "Add background\n$B+\widehat{\Delta}_{local}$")
    box(ax, (0.785, 0.19), (0.16, 0.10), "Final PM$_{2.5}$ prediction\n$\widehat{y}(t,s)$")
    arrow(ax, (0.865, 0.755), (0.865, 0.665)); arrow(ax, (0.865, 0.575), (0.865, 0.485)); arrow(ax, (0.865, 0.395), (0.865, 0.29))
    arrow(ax, (0.68, 0.835), (0.785, 0.805)); arrow(ax, (0.70, 0.505), (0.785, 0.795)); arrow(ax, (0.70, 0.185), (0.785, 0.785))
    save(fig, output, "Figure_03_framework_architecture")
    return {"equation": "y_hat(t,s) = B(t,s) + Delta_hat_local(t,s)", "generative_ai_used": False}


def figure_04_protocol(output: Path) -> dict[str, object]:
    sequences = pd.read_csv(TAIWAN_SEQUENCES, usecols=["split", "target_time"])
    sequences["date"] = pd.to_datetime(sequences.target_time).dt.date.astype(str)
    retained = {
        split: pd.to_datetime(sorted(sequences.loc[sequences.split.eq(split), "date"].unique()))
        for split in ["train", "val", "test"]
    }
    train, val, test = retained["train"], retained["val"], retained["test"]
    fig = plt.figure(figsize=(11.8, 4.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.7, 1], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    for dates, grey, label in [(train, "#595959", f"Train: {len(train)} dates"), (val, "#969696", f"Validation: {len(val)} dates"), (test, "#d0d0d0", f"Test: {len(test)} dates")]:
        ax.bar(dates, np.ones(len(dates)), width=0.92, color=grey, edgecolor=grey, label=label)
    ax.set_ylim(0, 1.08); ax.set_yticks([]); ax.set_title("Chronological complete-date protocol", loc="left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d")); ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35)); ax.grid(False)
    stats = [("27,084", "T=7 sequences"), ("6", "fixed monitoring sites"), ("0", "raw frames shared\nacross partitions"), ("0", "target-site readings in\nits own background"), ("7,180", "chronological test\nsequences"), ("No", "test-target use for\ntraining or selection")]
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    for i, (value, label) in enumerate(stats):
        x0 = i / len(stats) + 0.006
        w = 1 / len(stats) - 0.012
        ax2.add_patch(FancyBboxPatch((x0, 0.08), w, 0.80, boxstyle="round,pad=0.01", facecolor="#f5f5f5", edgecolor="#dddddd", linewidth=0.6))
        ax2.text(x0 + w / 2, 0.62, value, ha="center", fontweight="bold", fontsize=13)
        ax2.text(x0 + w / 2, 0.27, label, ha="center", va="center", fontsize=6.6)
    save(fig, output, "Figure_04_protocol_and_leakage_audit")
    return {"training_dates": len(train), "validation_dates": len(val), "test_dates": len(test), "raw_frame_overlap": 0}


def figure_05_measured_predicted(output: Path, test: pd.DataFrame) -> dict[str, object]:
    actual = test.actual_pm25.to_numpy()
    panels = [("predicted_reference_only", "Monitoring-network background"), ("predicted_reference_context_plus_conditioned_image", "Background + context + conditioned image")]
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.8), sharex=True, sharey=True)
    limits = (0, max(110, float(np.nanpercentile(actual, 99.8))))
    out = {}
    for ax, (column, title) in zip(axes, panels):
        pred = test[column].to_numpy()
        score = scores(actual, pred); out[column] = score
        ax.hexbin(actual, pred, gridsize=55, cmap="Greys", mincnt=1, linewidths=0, bins="log")
        ax.plot(limits, limits, "--", color="#333333", linewidth=0.9)
        ax.set(xlim=limits, ylim=limits, title=title, xlabel=r"Measured PM$_{2.5}$ ($\mu$g m$^{-3}$)")
        ax.text(0.05, 0.93, f"n = {score['n']:,}\n$R^2$ = {score['r2']:.3f}\nRMSE = {score['rmse']:.2f}\nMAE = {score['mae']:.2f}", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84}, fontsize=7.5)
    axes[0].set_ylabel(r"Predicted PM$_{2.5}$ ($\mu$g m$^{-3}$)")
    fig.suptitle("Measured versus predicted PM$_{2.5}$ on the chronological test partition", fontweight="semibold")
    fig.tight_layout()
    save(fig, output, "Figure_05_measured_vs_predicted")
    return out


def figure_06_baselines_outages(output: Path) -> dict[str, object]:
    base = pd.read_csv(TAIWAN_AUDIT / "chronological_baselines.csv")
    outage = pd.read_csv(TAIWAN_AUDIT / "support_station_outage_sensitivity.csv")
    order = ["global_training_mean", "target_site_hour_climatology", "other_site_median_lag1h", "other_site_mean_current", "other_site_median_current", "conditioned_image_frozen"]
    labels = ["Train\nmean", "Site-hour\nclimatology", "Network median\n(previous hour)", "Network\nmean", "Network\nmedian", "Complete\nmodel"]
    sub = base.set_index("model").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
    bars = axes[0].bar(np.arange(len(sub)), sub.rmse, color="#858585", edgecolor="#444444", linewidth=0.5)
    axes[0].set_xticks(np.arange(len(sub)), labels, fontsize=6.5); axes[0].set_ylabel(r"RMSE ($\mu$g m$^{-3}$)"); axes[0].set_title("Identical 54-date chronological test partition")
    axes[0].bar_label(bars, fmt="%.2f", fontsize=6, padding=2)
    bars2 = axes[1].bar(outage.removed_support_site.astype(str), outage.rmse, color="#8f8f8f", edgecolor="#444444", linewidth=0.5)
    axes[1].set_ylabel(r"Network median RMSE ($\mu$g m$^{-3}$)"); axes[1].set_xlabel("Supporting station removed"); axes[1].set_title("Single-support-station outage sensitivity"); axes[1].bar_label(bars2, fmt="%.2f", fontsize=6, padding=2)
    fig.tight_layout(); save(fig, output, "Figure_06_baselines_and_outages")
    return {"baselines": base.to_dict("records"), "outages": outage.to_dict("records")}


def figure_07_datewise(output: Path) -> dict[str, object]:
    df = pd.read_csv(TAIWAN_AUDIT / "datewise_metrics.csv")
    base = df.loc[df.model.eq("other_site_median_current")].sort_values("date")
    final = df.loc[df.model.eq("conditioned_image_frozen")].sort_values("date")
    merged = base[["date", "r2", "rmse"]].merge(final[["date", "r2", "rmse"]], on="date", suffixes=("_base", "_final"), validate="one_to_one")
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.0), gridspec_kw={"height_ratios": [1, 1.4]})
    axes[0].boxplot([merged.r2_base, merged.r2_final], tick_labels=["Background", "Final model"], patch_artist=True, boxprops={"facecolor": "#bcbcbc"}, medianprops={"color": "black"}, showfliers=False)
    rng = np.random.default_rng(42)
    for i, vals in enumerate([merged.r2_base, merged.r2_final], start=1): axes[0].scatter(rng.normal(i, 0.035, len(vals)), vals, s=9, color="#555555", alpha=0.65)
    axes[0].axhline(0, color="black", linewidth=0.7, linestyle="--"); axes[0].set_ylabel("Date-wise $R^2$"); axes[0].set_title("Performance heterogeneity across 54 chronological test dates")
    reduction = merged.rmse_base - merged.rmse_final
    axes[1].bar(np.arange(len(merged)), reduction, color=np.where(reduction >= 0, "#666666", "#b8b8b8"), width=0.8)
    axes[1].axhline(0, color="black", linewidth=0.7); axes[1].set_ylabel(r"RMSE reduction ($\mu$g m$^{-3}$)"); axes[1].set_xlabel("Chronological test date")
    tick = np.arange(0, len(merged), 6); axes[1].set_xticks(tick, merged.date.iloc[tick], rotation=50, ha="right", fontsize=6)
    fig.tight_layout(); save(fig, output, "Figure_07_datewise_robustness")
    return {"dates": len(merged), "dates_improved": int((reduction > 0).sum()), "median_rmse_reduction": float(reduction.median())}


def figure_08_pollution_range(output: Path, test: pd.DataFrame) -> dict[str, object]:
    bins = [-np.inf, 15, 35, 55, np.inf]; labels = ["<=15", "15-35", "35-55", ">55"]
    work = test.copy(); work["band"] = pd.cut(work.actual_pm25, bins=bins, labels=labels)
    rows = []
    for band, group in work.groupby("band", observed=True):
        for name, col in [("Background", "predicted_reference_only"), ("Final model", "predicted_reference_context_plus_conditioned_image")]:
            s = scores(group.actual_pm25, group[col]); rows.append({"band": str(band), "model": name, **s})
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.8))
    x = np.arange(len(labels)); width = 0.36
    for ax, metric, title in zip(axes, ["mae", "rmse"], ["Mean absolute error", "Root mean squared error"]):
        for j, (model, grey) in enumerate([("Background", "#aaaaaa"), ("Final model", "#555555")]):
            sub = table.loc[table.model.eq(model)].set_index("band").reindex(labels)
            ax.bar(x + (j - 0.5) * width, sub[metric], width, label=model, color=grey)
        counts = table.loc[table.model.eq("Background")].set_index("band").reindex(labels).n
        for xi, n in zip(x, counts): ax.text(xi, ax.get_ylim()[1] * 0.96, f"n={int(n):,}", ha="center", va="top", fontsize=6)
        ax.set_xticks(x, labels); ax.set_xlabel(r"Measured PM$_{2.5}$ range ($\mu$g m$^{-3}$)"); ax.set_ylabel(metric.upper()); ax.set_title(title)
    axes[0].legend(frameon=False); fig.suptitle("Error by measured pollution range", fontweight="semibold"); fig.tight_layout(); save(fig, output, "Figure_08_pollution_range_performance")
    return {"strata": table.to_dict("records")}


def figure_09_bootstrap(output: Path) -> dict[str, object]:
    table = pd.read_csv(TAIWAN_AUDIT / "paired_date_bootstrap_summary.csv")
    use = table.loc[(table.baseline.eq("other_site_median_current")) & (table.candidate.eq("conditioned_image_frozen"))].set_index("metric").loc[["mae_reduction", "rmse_reduction", "r2_gain"]]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.9))
    names = ["MAE reduction", "RMSE reduction", "$R^2$ gain"]
    for ax, (metric, row), label in zip(axes, use.iterrows(), names):
        x = row["median"]; lo = row["ci_low_95"]; hi = row["ci_high_95"]
        ax.errorbar(x, 0, xerr=[[x - lo], [hi - x]], fmt="o", color="#555555", capsize=3)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8); ax.set_yticks([]); ax.set_xlabel(label)
        ax.set_title(f"{x:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]", fontsize=8.5)
        ax.text(0.98, 0.05, f"P(improvement)={row['probability_improvement']:.3f}", transform=ax.transAxes, ha="right", fontsize=6.5, color="#555555")
    fig.suptitle("Paired complete-date bootstrap of the conditioned-image gain", fontweight="semibold")
    fig.text(0.5, -0.01, f"{int(use.replicates.iloc[0]):,} resamples of the 54 test dates; positive values favour the final model.", ha="center", fontsize=7)
    fig.tight_layout(); save(fig, output, "Figure_09_date_block_bootstrap_gain")
    return {"comparison": use.reset_index().to_dict("records")}


def load_traqid_calibration() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(TRAQID_CAL / "predictions.csv")
    params = pd.read_csv(TRAQID_CAL / "calibration_parameters.csv")
    date_metrics = pd.read_csv(TRAQID_CAL / "metrics_by_date.csv")
    return pred, params, date_metrics


def figure_10_calibration_diagnostics(output: Path) -> dict[str, object]:
    pred, params, metrics = load_traqid_calibration()
    p = params.loc[params.calibration.eq("zero_shot")].copy()
    z = pred.loc[pred.calibration.eq("zero_shot")].copy()
    future = (
        z.groupby("date")
        .apply(
            lambda g: pd.Series(
                {
                    "future_mean_residual": float(
                        np.mean(g["actual_PM2.5"] - g["calibrated_PM2.5"])
                    )
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    relation = p[["date", "adaptation_median_residual"]].merge(future, on="date", validate="one_to_one")
    corr = float(relation.adaptation_median_residual.corr(relation.future_mean_residual))
    coef = np.polyfit(relation.adaptation_median_residual, relation.future_mean_residual, 1)
    xline = np.linspace(relation.adaptation_median_residual.min(), relation.adaptation_median_residual.max(), 100)
    fig_a, ax_a = plt.subplots(1, 1, figsize=(5.7, 4.0))
    ax_a.scatter(relation.adaptation_median_residual, relation.future_mean_residual, color="#606060", s=27)
    ax_a.plot(xline, np.polyval(coef, xline), color="#222222")
    ax_a.axhline(0, color="#888888", linewidth=0.6); ax_a.axvline(0, color="#888888", linewidth=0.6)
    ax_a.set_xlabel(r"Median residual in first 10% ($\mu$g m$^{-3}$)"); ax_a.set_ylabel(r"Mean residual in future 90% ($\mu$g m$^{-3}$)"); ax_a.set_title(f"Early bias predicts later bias ($r$={corr:.2f})")
    fig_a.tight_layout(); save(fig_a, output, "Figure_10a_TRAQID_bias_relation")

    m = metrics.loc[metrics.calibration.isin(["zero_shot", "meta_ridge_median_offset"])].pivot(index="date", columns="calibration", values="rmse").sort_index()
    x = np.arange(len(m)); width = 0.37
    fig_b, ax_b = plt.subplots(1, 1, figsize=(6.5, 4.0))
    ax_b.bar(x - width / 2, m.zero_shot, width, label="Zero-shot", color="#a6a6a6")
    ax_b.bar(x + width / 2, m.meta_ridge_median_offset, width, label="10% meta-calibrated", color="#555555")
    ax_b.set_xticks(x, m.index, rotation=50, ha="right", fontsize=6); ax_b.set_ylabel(r"RMSE ($\mu$g m$^{-3}$)"); ax_b.set_title("Matched future-90% error by held-out date"); ax_b.legend(frameon=False)
    fig_b.tight_layout(); save(fig_b, output, "Figure_10b_TRAQID_calibration_RMSE")
    return {"early_future_residual_correlation": corr, "dates": len(relation), "relation": relation.to_dict("records")}


def figure_11_grouped_sensitivity(output: Path) -> dict[str, object]:
    strata = pd.read_csv(TRAQID_AUDIT / "calibration_concentration_strata.csv")
    effects = pd.read_csv(TRAQID_AUDIT / "calibration_date_effects.csv").sort_values("rmse_reduction")
    labels = ["<=35", "35-60", "60-90", "90-150", ">150"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    x = np.arange(len(labels)); width = 0.36
    for j, (name, grey) in enumerate([("zero_shot", "#a8a8a8"), ("meta_ridge_median_offset", "#555555")]):
        sub = strata.loc[strata.calibration.eq(name)].set_index("concentration_band").reindex(labels)
        axes[0].bar(x + (j - 0.5) * width, sub.rmse, width, label=name.replace("_", " "), color=grey)
    axes[0].set_xticks(x, labels); axes[0].set_xlabel(r"Measured PM$_{2.5}$ band ($\mu$g m$^{-3}$)"); axes[0].set_ylabel(r"RMSE ($\mu$g m$^{-3}$)"); axes[0].set_title("Error by concentration"); axes[0].legend(frameon=False)
    axes[1].barh(effects.date.astype(str), effects.rmse_reduction, color=np.where(effects.rmse_reduction >= 0, "#666666", "#b8b8b8"))
    axes[1].axvline(0, color="black", linewidth=0.7); axes[1].set_xlabel("RMSE reduction after calibration"); axes[1].set_title("Date-level calibration effect"); axes[1].tick_params(axis="y", labelsize=6)
    fig.tight_layout(); save(fig, output, "Figure_11_TRAQID_grouped_sensitivity")
    return {"concentration_strata": strata.to_dict("records"), "date_effects": effects.to_dict("records")}


def figure_12_timeseries(output: Path) -> dict[str, object]:
    pred, _, _ = load_traqid_calibration()
    zero = pred.loc[pred.calibration.eq("zero_shot"), ["date", "sequence_id", "target_created_at", "actual_PM2.5", "calibrated_PM2.5"]].rename(columns={"calibrated_PM2.5": "zero_shot"})
    cal = pred.loc[pred.calibration.eq("meta_ridge_median_offset"), ["date", "sequence_id", "calibrated_PM2.5"]].rename(columns={"calibrated_PM2.5": "calibrated"})
    work = zero.merge(cal, on=["date", "sequence_id"], validate="one_to_one")
    work["target_created_at"] = pd.to_datetime(work.target_created_at)
    date_rows = []
    for date, g in work.groupby("date"):
        z = scores(g["actual_PM2.5"], g.zero_shot)
        c = scores(g["actual_PM2.5"], g.calibrated)
        date_rows.append({"date": date, "gain": z["rmse"] - c["rmse"]})
    effects = pd.DataFrame(date_rows).sort_values("gain")
    selected = [
        effects.iloc[-1].date,
        effects.iloc[(len(effects) - 1) // 2].date,
        effects.iloc[0].date,
    ]
    descriptors = ["largest improvement", "median improvement", "largest degradation"]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.0))
    for ax, date, desc in zip(axes, selected, descriptors):
        g = work.loc[work.date.eq(date)].sort_values("target_created_at")
        gain = float(effects.loc[effects.date.eq(date), "gain"].iloc[0])
        ax.plot(g.target_created_at, g["actual_PM2.5"], color="#111111", linewidth=0.85, label="Measured")
        ax.plot(g.target_created_at, g.zero_shot, color="#9a9a9a", linewidth=0.8, label="Zero-shot")
        ax.plot(g.target_created_at, g.calibrated, color="#555555", linewidth=0.8, linestyle="--", label="Meta-calibrated")
        ax.set_title(f"{date}: {desc}, RMSE change {gain:+.1f}", fontsize=8.2); ax.set_ylabel(r"PM$_{2.5}$"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].legend(frameon=False, ncol=3, loc="upper right"); axes[-1].set_xlabel("Time"); fig.tight_layout(); save(fig, output, "Figure_12_TRAQID_measured_predicted_examples")
    return {"selected_dates": selected, "selection_rule": descriptors, "rmse_gains": effects.set_index("date").gain.to_dict()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    configure_style()
    required = [
        TRAQID_MANIFEST,
        TAIWAN_SEQUENCES,
        TAIWAN_PREDICTIONS,
        TAIWAN_SPLIT,
        TAIWAN_AUDIT / "chronological_baselines.csv",
        TAIWAN_AUDIT / "datewise_metrics.csv",
        TAIWAN_AUDIT / "paired_date_bootstrap_summary.csv",
        TAIWAN_AUDIT / "support_station_outage_sensitivity.csv",
        TRAQID_CAL / "predictions.csv",
        TRAQID_CAL / "calibration_parameters.csv",
        TRAQID_CAL / "metrics_by_date.csv",
        TRAQID_AUDIT / "calibration_concentration_strata.csv",
        TRAQID_AUDIT / "calibration_date_effects.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError("Missing required sources:\n" + "\n".join(missing))

    test = load_taiwan()
    report = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "method": "deterministic Python/matplotlib from preserved observations and predictions",
        "generative_ai_used": False,
        "figures": {},
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "excluded": {"Figure_02_dataset_scene_examples": "Exact six Taiwan source JPEGs are absent; existing montage was not reverse-engineered."},
    }
    report["figures"]["Figure_01"] = figure_01_traqid_distribution(output)
    report["figures"]["Figure_03"] = figure_03_architecture(output)
    report["figures"]["Figure_04"] = figure_04_protocol(output)
    report["figures"]["Figure_05"] = figure_05_measured_predicted(output, test)
    report["figures"]["Figure_06"] = figure_06_baselines_outages(output)
    report["figures"]["Figure_07"] = figure_07_datewise(output)
    report["figures"]["Figure_08"] = figure_08_pollution_range(output, test)
    report["figures"]["Figure_09"] = figure_09_bootstrap(output)
    report["figures"]["Figure_10"] = figure_10_calibration_diagnostics(output)
    report["figures"]["Figure_11"] = figure_11_grouped_sensitivity(output)
    report["figures"]["Figure_12"] = figure_12_timeseries(output)
    report["generated_artifacts"] = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.suffix.lower() in {".pdf", ".png"}
    }
    (output / "rebuild_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"output_dir": str(output), "figures_rebuilt": len(report["figures"]), "excluded": report["excluded"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
