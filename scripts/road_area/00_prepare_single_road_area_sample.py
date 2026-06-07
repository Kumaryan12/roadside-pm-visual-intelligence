from pathlib import Path
import argparse
import shutil
import pandas as pd


def resolve_path(p, project_root):
    p = Path(str(p))
    if p.is_absolute():
        return p
    return project_root / p


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="outputs/features/processed_frame_manifest_preprocessed_v2.csv",
    )
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--lens-id", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/road_area_inputs")

    args = parser.parse_args()

    project_root = Path(".").resolve()
    manifest_path = project_root / args.manifest
    output_dir = project_root / args.output_dir

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df = df[df["preprocess_status"] == "success"].copy()
    df = df[df["lens_id"] == args.lens_id].copy()

    if args.sample_index is not None:
        df = df[df["sample_index"] == args.sample_index].copy()

    if df.empty:
        raise ValueError("No matching successful frame found.")

    row = df.iloc[0]

    image_path = resolve_path(row["processed_frame_path"], project_root)

    if not image_path.exists():
        raise FileNotFoundError(f"Processed image not found: {image_path}")

    sample_index = int(row["sample_index"])
    lens_id = int(row["lens_id"])

    out_name = f"sample_{sample_index:05d}_lens_{lens_id}_processed.jpg"
    out_image_path = image_dir / out_name

    shutil.copy2(image_path, out_image_path)

    meta_path = output_dir / f"sample_{sample_index:05d}_lens_{lens_id}_metadata.csv"
    pd.DataFrame([row.to_dict()]).to_csv(meta_path, index=False)

    print("\nPrepared road-area input image:")
    print(out_image_path)

    print("\nSaved metadata:")
    print(meta_path)

    print("\nUse this image path in the next scripts:")
    print(out_image_path)


if __name__ == "__main__":
    main()