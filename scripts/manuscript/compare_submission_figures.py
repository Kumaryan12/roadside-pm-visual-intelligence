#!/usr/bin/env python3
"""Compare deterministic figure rebuilds with the legacy submission artwork.

The reported similarities measure appearance only.  Numerical fidelity is
established by ``rebuild_report.json`` and the source tables, not by pixels.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "docs/environmental_modelling_software_submission_v1"


PAIRS = [
    ("Figure 1", "Figure_01_TRAQID_coverage_and_target_distribution.png", "Figure_01_TRAQID_coverage_and_target_distribution.png", None),
    ("Figure 3", "Figure_02_framework_architecture.png", "Figure_03_framework_architecture.png", None),
    ("Figure 4", "Figure_03_protocol_and_leakage_audit.pdf", "Figure_04_protocol_and_leakage_audit.pdf", None),
    ("Figure 5", "Figure_04_measured_vs_predicted.pdf", "Figure_05_measured_vs_predicted.pdf", None),
    ("Figure 6", "Figure_05_baselines_and_outages.pdf", "Figure_06_baselines_and_outages.pdf", None),
    ("Figure 7", "Figure_06_datewise_robustness.pdf", "Figure_07_datewise_robustness.pdf", None),
    ("Figure 8", "Figure_07_pollution_range_performance.pdf", "Figure_08_pollution_range_performance.pdf", None),
    ("Figure 9", "Figure_08_date_block_bootstrap_gain.pdf", "Figure_09_date_block_bootstrap_gain.pdf", None),
    ("Figure 10a", "Figure_09_TRAQID_bias_relation.png", "Figure_10_TRAQID_calibration_diagnostics.pdf", "left"),
    ("Figure 10b", "Figure_10_TRAQID_calibration_RMSE.png", "Figure_10_TRAQID_calibration_diagnostics.pdf", "right"),
    ("Figure 11", "Figure_11_TRAQID_grouped_sensitivity.pdf", "Figure_11_TRAQID_grouped_sensitivity.pdf", None),
    ("Figure 12", "Figure_12_TRAQID_measured_predicted_examples.png", "Figure_12_TRAQID_measured_predicted_examples.png", None),
]


def load_artwork(path: Path, temporary: Path) -> Image.Image:
    if path.suffix.lower() != ".pdf":
        return Image.open(path).convert("RGB")
    stem = temporary / (path.stem + "_render")
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "160", str(path), str(stem)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return Image.open(stem.with_suffix(".png")).convert("RGB")


def normalise(image: Image.Image, size: tuple[int, int] = (900, 500)) -> Image.Image:
    image = ImageOps.grayscale(image)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", size, 255)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def crop_panel(image: Image.Image, panel: str | None) -> Image.Image:
    if panel is None:
        return image
    w, h = image.size
    if panel == "left":
        return image.crop((0, 0, int(w * 0.51), h))
    return image.crop((int(w * 0.49), 0, w, h))


def metrics(a: Image.Image, b: Image.Image) -> dict[str, float]:
    x = np.asarray(normalise(a), dtype=float) / 255.0
    y = np.asarray(normalise(b), dtype=float) / 255.0
    correlation = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    pixel_mae = float(np.mean(np.abs(x - y)))
    ex = np.asarray(normalise(a).filter(ImageFilter.FIND_EDGES), dtype=float)
    ey = np.asarray(normalise(b).filter(ImageFilter.FIND_EDGES), dtype=float)
    bx = ex > np.quantile(ex, 0.84)
    by = ey > np.quantile(ey, 0.84)
    union = np.logical_or(bx, by).sum()
    edge_iou = float(np.logical_and(bx, by).sum() / union) if union else 1.0
    return {"pixel_correlation": correlation, "pixel_mae": pixel_mae, "edge_iou": edge_iou}


def labelled(image: Image.Image, heading: str, target: tuple[int, int] = (920, 530)) -> Image.Image:
    panel = Image.new("RGB", target, "white")
    view = image.copy()
    view.thumbnail((target[0] - 20, target[1] - 52), Image.Resampling.LANCZOS)
    panel.paste(view, ((target[0] - view.width) // 2, 42 + (target[1] - 52 - view.height) // 2))
    ImageDraw.Draw(panel).text((12, 12), heading, fill="black")
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, default=SUBMISSION / "figures_submission_grayscale")
    parser.add_argument("--rebuilt-dir", type=Path, default=SUBMISSION / "figures_rebuilt_code_v1")
    parser.add_argument("--output-dir", type=Path, default=SUBMISSION / "figure_comparison_v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sheets = []
    with tempfile.TemporaryDirectory(prefix="figure_compare_") as tmp:
        temp = Path(tmp)
        for label, legacy_name, rebuilt_name, crop in PAIRS:
            legacy_path = args.legacy_dir / legacy_name
            rebuilt_path = args.rebuilt_dir / rebuilt_name
            if not legacy_path.exists() or not rebuilt_path.exists():
                rows.append({"figure": label, "status": "missing", "legacy": str(legacy_path), "rebuilt": str(rebuilt_path)})
                continue
            legacy = load_artwork(legacy_path, temp)
            rebuilt = crop_panel(load_artwork(rebuilt_path, temp), crop)
            row = {"figure": label, "status": "compared", "legacy": legacy_name, "rebuilt": rebuilt_name, **metrics(legacy, rebuilt)}
            rows.append(row)
            pair = Image.new("RGB", (1840, 570), "#e6e6e6")
            pair.paste(labelled(legacy, f"{label}: legacy submission artwork"), (0, 0))
            pair.paste(labelled(rebuilt, f"{label}: deterministic code rebuild"), (920, 0))
            sheets.append(pair)

    with (args.output_dir / "appearance_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)

    contact = Image.new("RGB", (1840, 570 * len(sheets)), "#dddddd")
    for index, sheet in enumerate(sheets):
        contact.paste(sheet, (0, index * 570))
    contact.save(args.output_dir / "legacy_vs_deterministic_contact_sheet.png", optimize=True)
    compared = [row for row in rows if row["status"] == "compared"]
    print(f"Compared {len(compared)} figure panels")
    for row in compared:
        print(f"{row['figure']:>10}: correlation={row['pixel_correlation']:.3f}, edge IoU={row['edge_iou']:.3f}, pixel MAE={row['pixel_mae']:.3f}")
    print(f"Contact sheet: {args.output_dir / 'legacy_vs_deterministic_contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
