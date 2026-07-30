"""Rotate raw lens frames consistently and build six-lens comparison sheets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input_manifest)
    source = source[source["preprocess_status"].eq("success")].copy()
    output_root = Path(args.output_root)
    rotated_root = output_root / "rotated"
    sheet_root = output_root / "contact_sheets"
    rotated_root.mkdir(parents=True, exist_ok=True)
    sheet_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in source.to_dict("records"):
        source_path = Path(record["processed_frame_path"])
        lens = int(record["lens_id"])
        destination = rotated_root / f"lens{lens}" / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            rotated = image.convert("RGB").transpose(Image.Transpose.ROTATE_90)
            rotated.save(destination, quality=94)
        row = dict(record)
        row.update(
            {
                "source_frame_path": str(source_path),
                "processed_frame_path": str(destination),
                "orientation_audit_rotation": "rot90_counterclockwise",
            }
        )
        rows.append(row)

    manifest = pd.DataFrame(rows)
    for sample_id, group in manifest.groupby("sample_id", sort=True):
        tiles = []
        for record in group.sort_values("lens_id").to_dict("records"):
            lens = int(record["lens_id"])
            with Image.open(record["processed_frame_path"]) as image:
                tile = ImageOps.fit(image.convert("RGB"), (300, 400))
            canvas = Image.new("RGB", (300, 432), "white")
            canvas.paste(tile, (0, 32))
            ImageDraw.Draw(canvas).text((10, 9), f"Lens {lens}", fill="black")
            tiles.append(canvas)
        sheet = Image.new("RGB", (900, 864), "white")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % 3) * 300, (index // 3) * 432))
        sheet.save(sheet_root / f"{sample_id}_all_lenses.jpg", quality=94)

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    print(f"Saved {len(manifest)} rotated frames and {manifest.sample_id.nunique()} sheets")
    print(f"Manifest: {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
