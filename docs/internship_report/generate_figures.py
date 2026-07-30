#!/usr/bin/env python3
"""Generate reproducible figures for the internship report.

All values are either read from committed run artifacts or are fixed audit
summaries documented in docs/pm25_experiment_consolidation.md.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#102A56"
BLUE = "#2F66B0"
CYAN = "#1597A5"
GREEN = "#2E7D32"
ORANGE = "#E67E22"
RED = "#C0392B"
PURPLE = "#6C4AB6"
GREY = "#5D6675"
LIGHT = "#F4F7FB"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": "#CAD3E0",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def rounded_box(ax, x, y, w, h, text, face, edge, fontsize=9, weight="normal"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NAVY,
        weight=weight,
        wrap=True,
    )
    return box


def arrow(ax, p1, p2, color=GREY, lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            connectionstyle="arc3",
        )
    )


def figure_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.96,
        "End-to-end roadside PM intelligence workflow",
        ha="center",
        va="top",
        fontsize=18,
        weight="bold",
        color=NAVY,
    )
    items = [
        ("Raw videos +\nsensor logs", "#EAF2FF", BLUE),
        ("10 s alignment +\nframe extraction", "#EDF8FA", CYAN),
        ("Lens-specific\npreprocessing", "#EDF8FA", CYAN),
        ("YOLO traffic +\nSegFormer road", "#FFF2E5", ORANGE),
        ("OSM + AlphaEarth +\natmospheric context", "#F2ECFF", PURPLE),
        ("Temporal and\ntabular models", "#EAF6EA", GREEN),
        ("Leakage audit +\nwhole-date evaluation", "#FDECEC", RED),
    ]
    xs = np.linspace(0.02, 0.86, len(items))
    w, h, y = 0.12, 0.42, 0.30
    for i, ((text, face, edge), x) in enumerate(zip(items, xs)):
        rounded_box(ax, x, y, w, h, text, face, edge, fontsize=9, weight="bold")
        ax.text(x + w / 2, y - 0.06, f"{i+1}", ha="center", color=edge, weight="bold")
        if i < len(items) - 1:
            arrow(ax, (x + w, y + h / 2), (xs[i + 1] - 0.008, y + h / 2))
    ax.text(
        0.5,
        0.11,
        "Reproducible artifacts at every stage: manifests, features, embeddings, split audits, predictions and metrics",
        ha="center",
        color=GREY,
        fontsize=10,
    )
    save(fig, "project_pipeline.png")


def figure_dataset():
    dates = ["Feb 1", "Feb 2", "Feb 3", "Feb 4", "Feb 5"]
    rows = np.array([1265, 1736, 902, 1542, 1428])
    runs = np.array([10, 15, 4, 16, 10])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.7, 4.2))
    bars = ax1.bar(dates, rows, color=[BLUE, CYAN, PURPLE, ORANGE, GREEN])
    ax1.set_title("Aligned 10-second observations by date", weight="bold", color=NAVY)
    ax1.set_ylabel("Rows")
    ax1.set_ylim(0, 1900)
    for b, v in zip(bars, rows):
        ax1.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:,}", ha="center", weight="bold")
    bars2 = ax2.bar(dates, runs, color=[BLUE, CYAN, PURPLE, ORANGE, GREEN])
    ax2.set_title("Collection runs by date", weight="bold", color=NAVY)
    ax2.set_ylabel("Runs")
    ax2.set_ylim(0, 18)
    for b, v in zip(bars2, runs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.35, str(v), ha="center", weight="bold")
    fig.suptitle(
        "Canonical five-day MUMMA dataset: 6,873 samples, 55 runs, 20,619 images",
        fontsize=15,
        weight="bold",
        color=NAVY,
        y=1.03,
    )
    fig.tight_layout()
    save(fig, "mumma_dataset_summary.png")


def figure_triptych():
    manifest = ROOT / "artifacts/runs/mumma_5day_combined_lenses_1_2_6_v1/manifests/preprocessed_frames.csv"
    sample = "run_20260205_124917_1693_000020"
    table = pd.read_csv(manifest, low_memory=False)
    rows = table.loc[table["sample_id"].eq(sample)].sort_values("lens_id")
    imgs = []
    labels = []
    for _, row in rows.iterrows():
        path = ROOT / str(row["processed_frame_path"])
        imgs.append(Image.open(path).convert("RGB"))
        labels.append(f"Lens {int(row['lens_id'])}")
    target_h = 760
    resized = []
    for im in imgs:
        scale = target_h / im.height
        resized.append(im.resize((int(im.width * scale), target_h), Image.Resampling.LANCZOS))
    gap, header = 22, 78
    canvas = Image.new("RGB", (sum(x.width for x in resized) + gap * 4, target_h + header), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    x = gap
    for im, label in zip(resized, labels):
        canvas.paste(im, (x, header))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (im.width - (box[2] - box[0])) / 2, 20), label, font=font, fill=NAVY)
        x += im.width + gap
    canvas.save(OUT / "mumma_multiview_sample.jpg", quality=92, optimize=True)


def figure_yolo_counts():
    labels = ["Motorcycle", "Car", "Auto-rickshaw", "Truck", "Bus", "Bicycle", "Fallback"]
    values = [50196, 45193, 20621, 9355, 7826, 2382, 961]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, values, color=[BLUE, CYAN, ORANGE, PURPLE, GREEN, "#8C6D31", GREY])
    ax.set_yticks(y, labels)
    ax.set_xlabel("Detected objects")
    ax.set_title("PM-relevant road users detected across 20,619 frames", weight="bold", color=NAVY)
    ax.grid(axis="y", visible=False)
    for b, v in zip(bars, values):
        ax.text(v + 700, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=9)
    ax.set_xlim(0, 56000)
    save(fig, "yolo_object_counts.png")


def figure_validation():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.96, "Why validation protocol changed the scientific conclusion", ha="center",
            fontsize=17, weight="bold", color=NAVY)
    rounded_box(ax, 0.04, 0.20, 0.42, 0.62,
                "RANDOM SEQUENCE DIAGNOSTIC\n\nSliding T=7 windows are built first\nand shuffled second.\n\n6,451 / 6,873 raw samples occur\nin multiple partitions.\n\n5,142 raw samples overlap train/test.\n\n98.32% of test targets already\nappeared as training context.",
                "#FDECEC", RED, fontsize=11, weight="bold")
    rounded_box(ax, 0.54, 0.20, 0.42, 0.62,
                "WHOLE-DATE HOLDOUT\n\nAn entire collection date is withheld.\n\nSequences remain inside run_id.\n\nNo sequence crosses a run or date.\n\nThe test day is not used for fitting,\nfeature selection or fusion weighting.\n\nThis is the principal fair stress test.",
                "#EAF6EA", GREEN, fontsize=11, weight="bold")
    arrow(ax, (0.47, 0.51), (0.53, 0.51), color=NAVY, lw=2.0)
    ax.text(0.5, 0.12, "High random R² is a same-distribution diagnostic, not evidence of future-date generalization.",
            ha="center", color=NAVY, fontsize=11, weight="bold")
    save(fig, "validation_protocols.png")


def figure_random_fair():
    labels = [
        "TRAQID OOF\nrandom",
        "MUMMA smoothed\nrandom",
        "MUMMA ConvNeXt\nrandom",
        "Direct fair\nvisual+geo",
        "CAMS nested\nfair",
        "MERRA-2 nested\nfair",
    ]
    r2 = [0.966, 0.958, 0.867, 0.163, 0.457, 0.538]
    colors = [RED, RED, RED, BLUE, CYAN, GREEN]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(np.arange(len(labels)), r2, color=colors)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Random diagnostics and whole-date results must not be mixed", weight="bold", color=NAVY)
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.3f}", ha="center", weight="bold")
    ax.text(0.8, 1.0, "overlap-contaminated", color=RED, ha="center", fontsize=9)
    ax.text(4.5, 0.96, "whole-date pooled", color=GREEN, ha="center", fontsize=9)
    save(fig, "random_vs_fair_r2.png")


def figure_architecture():
    fig, ax = plt.subplots(figsize=(16, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.975, "Background-Aware Roadside PM$_{2.5}$ Prediction",
            ha="center", va="top", fontsize=23, weight="bold", color=NAVY)
    ax.text(0.5, 0.915,
            "External atmosphere establishes the regional baseline; visual and contextual branches estimate only the local roadside departure.",
            ha="center", color=GREY, fontsize=11.5)

    # Three horizontal scientific lanes plus a separate fusion/output column.
    rounded_box(ax, 0.025, 0.705, 0.95, 0.155, "", "#ECF3FF", BLUE)
    rounded_box(ax, 0.025, 0.405, 0.735, 0.245, "", "#EAF8F8", CYAN)
    rounded_box(ax, 0.025, 0.125, 0.735, 0.235, "", "#FFF3E8", ORANGE)
    rounded_box(ax, 0.785, 0.075, 0.19, 0.575, "", "#F3EEFF", PURPLE)

    ax.text(0.045, 0.825, "A. REGIONAL BACKGROUND", color=BLUE, weight="bold", fontsize=11)
    ax.text(0.045, 0.615, "B. TEMPORAL LOCAL-INCREMENT BRANCH", color=CYAN, weight="bold", fontsize=11)
    ax.text(0.045, 0.325, "C. TABULAR / GEOSPATIAL LOCAL-INCREMENT BRANCH",
            color=ORANGE, weight="bold", fontsize=11)
    ax.text(0.88, 0.615, "D. FUSION & OUTPUT", color=PURPLE, weight="bold",
            fontsize=11, ha="center")

    # Lane A: the background remains separate until the final addition.
    rounded_box(ax, 0.205, 0.745, 0.145, 0.075, "Timestamp + location",
                "white", BLUE, 9.5, "bold")
    rounded_box(ax, 0.405, 0.735, 0.18, 0.095, "CAMS or MERRA-2\nbackground product",
                "white", BLUE, 9.5, "bold")
    rounded_box(ax, 0.64, 0.745, 0.11, 0.075, "Regional baseline\n" + r"$B(t,\mathbf{s})$",
                "white", BLUE, 9, "bold")
    arrow(ax, (0.35, 0.782), (0.405, 0.782), color=BLUE)
    arrow(ax, (0.585, 0.782), (0.64, 0.782), color=BLUE)
    ax.text(0.49, 0.712, "External input; never fitted to the roadside test target",
            ha="center", color=BLUE, fontsize=8.5)

    # Lane B: all arrows travel left to right; correction covariates enter vertically.
    temporal_boxes = [
        (0.155, 0.505, 0.11, 0.078, "Lens-6 images\nT = 7"),
        (0.285, 0.505, 0.10, 0.078, "ResNet50\n2,048-D"),
        (0.405, 0.505, 0.09, 0.078, "GRU\nhidden = 128"),
        (0.515, 0.505, 0.105, 0.078, r"Base estimate" + "\n" + r"$\hat{\Delta}_{temp}$"),
        (0.64, 0.485, 0.095, 0.118, "Cross-fitted RF\nresidual correction\n"
                                    + r"$\hat{\Delta}_{temp}^{corr}$"),
    ]
    for x, y, w, h, text_value in temporal_boxes:
        rounded_box(ax, x, y, w, h, text_value, "white", CYAN, 8.7, "bold")
    for left, right in zip(temporal_boxes[:-1], temporal_boxes[1:]):
        arrow(ax, (left[0] + left[2], left[1] + left[3] / 2),
              (right[0], right[1] + right[3] / 2), color=CYAN)
    rounded_box(ax, 0.64, 0.415, 0.095, 0.052, "Sensor + YOLO + road",
                "#FFF8F1", ORANGE, 7.5, "bold")
    arrow(ax, (0.6875, 0.467), (0.6875, 0.485), color=ORANGE, lw=1.4)

    # Lane C: a distinct, auditable context model.
    rounded_box(ax, 0.155, 0.205, 0.195, 0.085, "YOLO traffic + road appearance\n"
                                                "OSM 250 m + AlphaEarth 64-D",
                "white", ORANGE, 8.7, "bold")
    rounded_box(ax, 0.405, 0.205, 0.13, 0.085, "ExtraTrees\nlocal model",
                "white", ORANGE, 9, "bold")
    rounded_box(ax, 0.59, 0.205, 0.145, 0.085, "Tabular estimate\n"
                                             + r"$\hat{\Delta}_{tab}$",
                "white", ORANGE, 9, "bold")
    arrow(ax, (0.35, 0.2475), (0.405, 0.2475), color=ORANGE)
    arrow(ax, (0.535, 0.2475), (0.59, 0.2475), color=ORANGE)

    # Fusion: first combine local estimates, then add the independent background.
    rounded_box(ax, 0.81, 0.50, 0.14, 0.082, "Validation-selected blend\n"
                                             + r"$\lambda\hat{\Delta}_{temp}^{corr}"
                                             + r"+(1-\lambda)\hat{\Delta}_{tab}$",
                "white", PURPLE, 8.5, "bold")
    rounded_box(ax, 0.81, 0.385, 0.14, 0.072, "Fused local increment\n"
                                              + r"$\hat{\Delta}_{local}$",
                "white", PURPLE, 8.8, "bold")
    rounded_box(ax, 0.81, 0.27, 0.14, 0.072, "Add regional baseline\n"
                                             + r"$B+\hat{\Delta}_{local}$",
                "white", PURPLE, 8.8, "bold")
    rounded_box(ax, 0.81, 0.145, 0.14, 0.078, "Final roadside PM$_{2.5}$\n"
                                              + r"$\hat y(t,\mathbf{s})$",
                "#EAF6EA", GREEN, 9.5, "bold")

    # Temporal input to fusion: direct and horizontal.
    arrow(ax, (0.735, 0.544), (0.81, 0.544), color=CYAN, lw=1.8)
    # Tabular input follows the panel boundary, avoiding all model boxes.
    ax.plot([0.735, 0.77, 0.77], [0.2475, 0.2475, 0.52], color=ORANGE, lw=1.8)
    arrow(ax, (0.77, 0.52), (0.81, 0.52), color=ORANGE, lw=1.8)
    arrow(ax, (0.88, 0.50), (0.88, 0.457), color=PURPLE)
    arrow(ax, (0.88, 0.385), (0.88, 0.342), color=PURPLE)
    arrow(ax, (0.88, 0.27), (0.88, 0.223), color=GREEN)

    # Background bus stays on the outer boundary and enters only at the add step.
    ax.plot([0.75, 0.965, 0.965], [0.782, 0.782, 0.306], color=BLUE, lw=1.8)
    arrow(ax, (0.965, 0.306), (0.95, 0.306), color=BLUE, lw=1.8)

    ax.text(
        0.5,
        0.04,
        r"$\hat y = B + \lambda\hat{\Delta}_{temp}^{corr} + (1-\lambda)\hat{\Delta}_{tab}$",
        ha="center",
        color=NAVY,
        fontsize=13,
        weight="bold",
    )
    ax.text(0.5, 0.012,
            "One concentration is predicted; traffic, road and geospatial variables are covariates, not separately added source masses.",
            ha="center", color=GREY, fontsize=9)
    save(fig, "current_architecture.png")


def figure_background_comparison():
    labels = ["Direct visual/geo", "CAMS + local", "MERRA-2 + local", "MERRA-2 nested\nvalidation-selected",
              "MERRA-2 nested\n50:50 diagnostic"]
    r2 = [0.163, 0.425, 0.550, 0.538, 0.566]
    rmse = [47.740, 39.560, 35.007, 35.689, 34.589]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))
    x = np.arange(len(labels))
    colors = [GREY, CYAN, GREEN, PURPLE, ORANGE]
    b1 = ax1.bar(x, r2, color=colors)
    ax1.set_xticks(x, labels, rotation=18, ha="right")
    ax1.set_ylabel("Pooled $R^2$")
    ax1.set_ylim(0, 0.64)
    ax1.set_title("Explained variance", weight="bold", color=NAVY)
    for b, v in zip(b1, r2):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, weight="bold")
    b2 = ax2.bar(x, rmse, color=colors)
    ax2.set_xticks(x, labels, rotation=18, ha="right")
    ax2.set_ylabel("RMSE ($\\mu$g m$^{-3}$)")
    ax2.set_ylim(30, 52)
    ax2.set_title("Prediction error", weight="bold", color=NAVY)
    for b, v in zip(b2, rmse):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.45, f"{v:.1f}", ha="center", fontsize=9, weight="bold")
    fig.suptitle("Five-day whole-date results: atmospheric background materially improved pooled performance",
                 fontsize=14, weight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "background_model_comparison.png")


def figure_per_date():
    dates = ["Feb 1", "Feb 2", "Feb 3", "Feb 4", "Feb 5"]
    selected = [-0.790929, -0.467233, -0.283294, 0.276622, 0.231704]
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    colors = [RED if v < 0 else GREEN for v in selected]
    bars = ax.bar(dates, selected, color=colors)
    ax.axhline(0, color="#222222", lw=1)
    ax.set_ylabel("Per-date $R^2$")
    ax.set_ylim(-0.95, 0.42)
    ax.set_title("MERRA-2 nested ensemble: pooled improvement did not eliminate date instability",
                 weight="bold", color=NAVY)
    for b, v in zip(bars, selected):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.08),
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", weight="bold")
    ax.text(4.45, -0.85, "Pooled $R^2$ = 0.538\nMean fold $R^2$ = -0.207",
            ha="right", color=NAVY, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECF3FF", edgecolor=BLUE))
    save(fig, "merra_per_date_r2.png")


def figure_architecture2():
    metrics = pd.read_csv(ROOT / "artifacts/runs/mumma_5day_architecture2_stage1_fusion_fair_v1/metrics_aggregate.csv")
    order = ["image_only", "atmosphere_only", "early_fusion"]
    x = np.arange(3)
    sub = metrics.set_index("variant").loc[order]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3))
    labels = ["Image only", "Atmosphere only", "Early fusion"]
    colors = [BLUE, GREEN, PURPLE]
    b1 = ax1.bar(x, sub["rmse"], color=colors)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("RMSE ($\\mu$g m$^{-3}$)")
    ax1.set_ylim(40, 60)
    ax1.set_title("MUMMA whole-date error", weight="bold", color=NAVY)
    for b, v in zip(b1, sub["rmse"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}", ha="center", weight="bold")
    b2 = ax2.bar(x, sub["r2"], color=colors)
    ax2.axhline(0, color="#333", lw=0.9)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Pooled $R^2$")
    ax2.set_ylim(-0.2, 0.3)
    ax2.set_title("MUMMA whole-date explained variance", weight="bold", color=NAVY)
    for b, v in zip(b2, sub["r2"]):
        ax2.text(b.get_x() + b.get_width() / 2, v + (0.012 if v >= 0 else -0.025), f"{v:.3f}",
                 ha="center", va="bottom" if v >= 0 else "top", weight="bold")
    fig.suptitle("Architecture 2: early image-atmosphere fusion did not beat atmosphere-only",
                 fontsize=14, weight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "architecture2_results.png")


def figure_uncertainty():
    labels = ["Temporal", "Selected\nensemble", "Equal 50:50", "Tabular"]
    coverage = [0.7605, 0.7764, 0.7905, 0.8499]
    width = [63.75, 63.14, 64.96, 82.75]
    fig, ax1 = plt.subplots(figsize=(9.7, 4.6))
    x = np.arange(len(labels))
    bars = ax1.bar(x - 0.18, coverage, width=0.36, color=BLUE, label="Empirical coverage")
    ax1.axhline(0.90, color=RED, linestyle="--", lw=1.6, label="Nominal 90%")
    ax1.set_ylim(0.65, 0.94)
    ax1.set_ylabel("Coverage")
    ax1.set_xticks(x, labels)
    ax2 = ax1.twinx()
    lines = ax2.plot(x + 0.18, width, marker="o", color=ORANGE, lw=2.2, label="Mean interval width")
    ax2.set_ylim(50, 92)
    ax2.set_ylabel("Mean width ($\\mu$g m$^{-3}$)")
    ax1.set_title("Conformal uncertainty intervals under-covered the nominal 90% level",
                  weight="bold", color=NAVY)
    handles = [bars, ax1.lines[0], lines[0]]
    ax1.legend(handles, ["Empirical coverage", "Nominal 90%", "Mean interval width"],
               loc="lower right", frameon=False)
    save(fig, "uncertainty_diagnostics.png")


def figure_source_proxy():
    targets = ["Fine mass\nfraction", "Effective\ndiameter", "Density\nproxy", "Number\nproxy"]
    random_r2 = [0.744, 0.648, 0.544, 0.844]
    fair_r2 = [-72.0, -6.16, -2.13, -5.12]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.3, 4.5))
    x = np.arange(len(targets))
    ax1.bar(x, random_r2, color=[CYAN, BLUE, PURPLE, ORANGE])
    ax1.set_xticks(x, targets)
    ax1.set_ylim(0, 0.95)
    ax1.set_ylabel("$R^2$")
    ax1.set_title("Random-row diagnostic", weight="bold", color=NAVY)
    for i, v in enumerate(random_r2):
        ax1.text(i, v + 0.025, f"{v:.3f}", ha="center", weight="bold")
    ax2.bar(x, fair_r2, color=RED)
    ax2.axhline(0, color="#222", lw=0.8)
    ax2.set_xticks(x, targets)
    ax2.set_ylabel("$R^2$")
    ax2.set_title("Held-out Feb 1 stress test", weight="bold", color=NAVY)
    for i, v in enumerate(fair_r2):
        ax2.text(i, v - 1.2 if v < -10 else v - 0.2, f"{v:.2f}", ha="center", va="top", weight="bold")
    fig.suptitle("Particle-regime targets appeared predictable randomly but did not transfer by date",
                 fontsize=14, weight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "particle_regime_results.png")


def figure_chronology():
    fig, ax = plt.subplots(figsize=(12.2, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Data audit", "raw SSD, timestamps,\nlenses, alignment"),
        ("Two-day pilot", "features, CNN-RNN,\nleakage audit"),
        ("Five-day scale-up", "6,873 rows,\n20,619 images"),
        ("Geospatial context", "OSM + AlphaEarth\n+ road depth"),
        ("Fair modelling", "whole-date CV,\nCAMS increment"),
        ("Refinement", "MERRA-2, nested\nensemble, uncertainty"),
        ("Scientific audit", "source limits,\nvariogram, spikes"),
    ]
    xs = np.linspace(0.06, 0.94, len(steps))
    ax.plot([xs[0], xs[-1]], [0.55, 0.55], color="#B7C3D4", lw=3)
    colors = [BLUE, CYAN, PURPLE, ORANGE, GREEN, NAVY, RED]
    for i, ((title, detail), x, color) in enumerate(zip(steps, xs, colors)):
        ax.scatter([x], [0.55], s=450, color=color, zorder=3, edgecolor="white", linewidth=2)
        ax.text(x, 0.55, str(i + 1), ha="center", va="center", color="white", weight="bold")
        y = 0.75 if i % 2 == 0 else 0.25
        ax.text(x, y, title, ha="center", va="center", color=color, weight="bold", fontsize=10)
        ax.text(x, y - (0.10 if i % 2 == 0 else -0.10), detail, ha="center", va="center",
                color=GREY, fontsize=8.5)
    ax.set_title("Two-month technical chronology", fontsize=17, weight="bold", color=NAVY, pad=10)
    save(fig, "internship_chronology.png")


def copy_existing_architecture():
    source = ROOT / "docs/deliverables/pm25_candidate_architecture_clean.png"
    if source.exists():
        shutil.copy2(source, OUT / "cams_candidate_architecture_legacy.png")


def main():
    figure_pipeline()
    figure_dataset()
    figure_triptych()
    figure_yolo_counts()
    figure_validation()
    figure_random_fair()
    figure_architecture()
    figure_background_comparison()
    figure_per_date()
    figure_architecture2()
    figure_uncertainty()
    figure_source_proxy()
    figure_chronology()
    copy_existing_architecture()
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
