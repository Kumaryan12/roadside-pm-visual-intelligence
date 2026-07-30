from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_tensorflow_model(model_name: str, image_size: int):
    import tensorflow as tf

    model_name = model_name.lower()

    if model_name == "mobilenetv2":
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        base = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        embedding_dim = 1280

    elif model_name == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input

        base = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        embedding_dim = 1280

    elif model_name == "resnet50":
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input

        base = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
        embedding_dim = 2048

    else:
        raise ValueError(
            f"Unknown model_name={model_name}. "
            "Use mobilenetv2, efficientnetb0, or resnet50."
        )

    base.trainable = False

    return base, preprocess_input, embedding_dim


def read_image(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
    return arr


def load_batch(
    paths: List[str],
    image_size: int,
) -> Tuple[np.ndarray, List[bool], List[str]]:
    images = []
    ok_flags = []
    errors = []

    for path in paths:
        try:
            arr = read_image(path, image_size)
            images.append(arr)
            ok_flags.append(True)
            errors.append("")
        except Exception as exc:
            images.append(np.zeros((image_size, image_size, 3), dtype=np.float32))
            ok_flags.append(False)
            errors.append(str(exc))

    batch = np.stack(images, axis=0)
    return batch, ok_flags, errors


def make_memmap(path: Path, shape: tuple, overwrite: bool):
    if path.exists() and overwrite:
        path.unlink()

    if path.exists() and not overwrite:
        arr = np.load(path, mmap_mode="r+")
        if arr.shape != shape:
            raise ValueError(
                f"Existing array has wrong shape: {path}, "
                f"found={arr.shape}, expected={shape}. "
                "Use --overwrite to recreate."
            )
        return arr

    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype="float32",
        shape=shape,
    )


def extract_view_embeddings(
    *,
    df: pd.DataFrame,
    view: str,
    model,
    preprocess_input,
    image_size: int,
    batch_size: int,
    embedding_dim: int,
    output_path: Path,
    status_df: pd.DataFrame,
    overwrite: bool,
) -> pd.DataFrame:
    path_col = f"{view}_path"
    status_col = f"{view}_embed_status"
    error_col = f"{view}_embed_error"

    if path_col not in df.columns:
        raise ValueError(f"Missing column in manifest: {path_col}")

    n = len(df)
    emb = make_memmap(output_path, (n, embedding_dim), overwrite=overwrite)

    if status_col not in status_df.columns:
        status_df[status_col] = "pending"
    if error_col not in status_df.columns:
        status_df[error_col] = ""

    n_batches = math.ceil(n / batch_size)
    start_time = time.time()

    print(f"\nExtracting {view} embeddings")
    print("Output:", output_path)
    print("Rows:", n)
    print("Batch size:", batch_size)
    print("Batches:", n_batches)

    for batch_idx, start in enumerate(range(0, n, batch_size), start=1):
        end = min(start + batch_size, n)

        # Resume behavior: skip if all rows already marked success.
        current_status = status_df.loc[start:end - 1, status_col].astype(str).values
        if np.all(current_status == "success"):
            if batch_idx % 25 == 0 or batch_idx == n_batches:
                print(f"[{view}] batch {batch_idx}/{n_batches} skipped/resumed")
            continue

        paths = df.iloc[start:end][path_col].astype(str).tolist()

        batch, ok_flags, errors = load_batch(paths, image_size=image_size)
        batch = preprocess_input(batch)

        preds = model.predict(batch, verbose=0)
        preds = np.asarray(preds, dtype=np.float32)

        emb[start:end, :] = preds

        for local_i, ok in enumerate(ok_flags):
            row_i = start + local_i
            if ok:
                status_df.at[row_i, status_col] = "success"
                status_df.at[row_i, error_col] = ""
            else:
                status_df.at[row_i, status_col] = "image_load_failed"
                status_df.at[row_i, error_col] = errors[local_i]

        if batch_idx % 10 == 0 or batch_idx == 1 or batch_idx == n_batches:
            elapsed = time.time() - start_time
            rate = end / max(elapsed, 1e-9)
            print(
                f"[{view}] batch {batch_idx:5d}/{n_batches} | "
                f"rows {end:6d}/{n} | "
                f"{rate:.2f} images/sec"
            )

    emb.flush()

    success_count = int((status_df[status_col] == "success").sum())
    fail_count = int((status_df[status_col] != "success").sum())

    print(f"\n{view} complete:")
    print(" success:", success_count)
    print(" failed :", fail_count)

    return status_df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="experiments/traqid_pretraining_v1/data/processed/traqid_paired_manifest_with_splits.csv",
    )

    parser.add_argument(
        "--out-dir",
        default="experiments/traqid_pretraining_v1/embeddings",
    )

    parser.add_argument(
        "--report-dir",
        default="experiments/traqid_pretraining_v1/reports/embedding_extraction",
    )

    parser.add_argument(
        "--model",
        default="mobilenetv2",
        choices=["mobilenetv2", "efficientnetb0", "resnet50"],
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument(
        "--views",
        default="front,rear",
        help="Comma-separated views to extract: front,rear",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional small test limit, e.g. --limit 256",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding arrays.",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    required_cols = [
        "row_id",
        "image_id",
        "created_at",
        "front_path",
        "rear_path",
        "split_date_chrono",
        "PM2.5",
        "PM10",
        "aqi",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Keep only valid image pairs.
    if "front_exists" in df.columns:
        df = df[df["front_exists"] == True].copy()
    if "rear_exists" in df.columns:
        df = df[df["rear_exists"] == True].copy()

    df = df.reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).copy().reset_index(drop=True)

    views = [v.strip().lower() for v in args.views.split(",") if v.strip()]
    valid_views = {"front", "rear"}
    bad_views = [v for v in views if v not in valid_views]
    if bad_views:
        raise ValueError(f"Invalid views: {bad_views}. Use front,rear.")

    print("=" * 90)
    print("TRAQID IMAGE EMBEDDING EXTRACTION")
    print("=" * 90)
    print("Manifest:", manifest_path)
    print("Rows:", len(df))
    print("Model:", args.model)
    print("Image size:", args.image_size)
    print("Batch size:", args.batch_size)
    print("Views:", views)

    print("\nSplit counts:")
    print(df["split_date_chrono"].value_counts().to_string())

    model, preprocess_input, embedding_dim = load_tensorflow_model(
        args.model,
        args.image_size,
    )

    status_df = df[
        [
            "row_id",
            "image_id",
            "created_at",
            "date",
            "split_date_chrono",
            "PM2.5",
            "PM10",
            "aqi",
            "Temperature",
            "Humidity",
            "Season",
            "Day_or_Night",
            "aqi_cat",
            "front_path",
            "rear_path",
        ]
    ].copy()

    prefix = f"traqid_{args.model}_{args.image_size}"

    output_files = {}

    for view in views:
        output_path = out_dir / f"{prefix}_{view}_embeddings.npy"

        status_df = extract_view_embeddings(
            df=df,
            view=view,
            model=model,
            preprocess_input=preprocess_input,
            image_size=args.image_size,
            batch_size=args.batch_size,
            embedding_dim=embedding_dim,
            output_path=output_path,
            status_df=status_df,
            overwrite=args.overwrite,
        )

        output_files[f"{view}_embeddings"] = str(output_path)

    # Optional fused mean embedding if both front and rear were extracted.
    if "front" in views and "rear" in views:
        front_path = out_dir / f"{prefix}_front_embeddings.npy"
        rear_path = out_dir / f"{prefix}_rear_embeddings.npy"
        mean_path = out_dir / f"{prefix}_front_rear_mean_embeddings.npy"
        concat_path = out_dir / f"{prefix}_front_rear_concat_embeddings.npy"

        print("\nCreating fused front/rear embeddings...")

        front = np.load(front_path, mmap_mode="r")
        rear = np.load(rear_path, mmap_mode="r")

        mean_emb = make_memmap(
            mean_path,
            (len(df), embedding_dim),
            overwrite=args.overwrite,
        )

        concat_emb = make_memmap(
            concat_path,
            (len(df), embedding_dim * 2),
            overwrite=args.overwrite,
        )

        chunk = 2048
        for start in range(0, len(df), chunk):
            end = min(start + chunk, len(df))
            mean_emb[start:end, :] = (front[start:end, :] + rear[start:end, :]) / 2.0
            concat_emb[start:end, :embedding_dim] = front[start:end, :]
            concat_emb[start:end, embedding_dim:] = rear[start:end, :]

        mean_emb.flush()
        concat_emb.flush()

        output_files["front_rear_mean_embeddings"] = str(mean_path)
        output_files["front_rear_concat_embeddings"] = str(concat_path)

    index_out = out_dir / f"{prefix}_embedding_index.csv"
    status_df.to_csv(index_out, index=False)

    summary = {
        "manifest": str(manifest_path),
        "rows": int(len(df)),
        "model": args.model,
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "embedding_dim_per_view": int(embedding_dim),
        "views": views,
        "output_files": output_files,
        "index_csv": str(index_out),
        "split_counts": df["split_date_chrono"].value_counts().to_dict(),
        "warning": (
            "Embeddings are image representations only. Do not evaluate by random image split. "
            "Use split_date_chrono/date groups from the manifest."
        ),
    }

    for view in views:
        status_col = f"{view}_embed_status"
        summary[f"{view}_status_counts"] = (
            status_df[status_col].value_counts(dropna=False).to_dict()
        )

    summary_out = report_dir / f"{prefix}_embedding_extraction_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    for k, v in output_files.items():
        print(f" - {k}: {v}")
    print(" - index:", index_out)
    print(" - summary:", summary_out)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()