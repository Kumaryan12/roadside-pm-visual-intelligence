from pathlib import Path
import argparse
import json
import subprocess
import shutil

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic",
    ".mp4", ".mov", ".avi", ".mkv", ".m4v",
}


EXIF_KEYS = [
    "Make",
    "Model",
    "LensModel",
    "LensID",
    "FocalLength",
    "FocalLengthIn35mmFormat",
    "FieldOfView",
    "GPSLatitude",
    "GPSLongitude",
    "GPSAltitude",
    "CreateDate",
    "DateTimeOriginal",
    "MediaCreateDate",
    "TrackCreateDate",
    "ModifyDate",
    "WhiteBalance",
    "ColorTemperature",
    "ExposureTime",
    "ShutterSpeed",
    "ISO",
    "Aperture",
    "FNumber",
    "ImageWidth",
    "ImageHeight",
    "Rotation",
    "Orientation",
]


def run_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as exc:
        return "", str(exc), 1


def extract_exiftool_metadata(file_path):
    if shutil.which("exiftool") is None:
        return {"exiftool_error": "exiftool_not_installed"}

    stdout, stderr, code = run_command([
        "exiftool",
        "-j",
        "-a",
        "-G1",
        "-s",
        str(file_path),
    ])

    if code != 0 or not stdout.strip():
        return {"exiftool_error": stderr.strip()}

    try:
        data = json.loads(stdout)[0]
    except Exception as exc:
        return {"exiftool_error": str(exc)}

    out = {}

    for key in EXIF_KEYS:
        matches = []

        for full_key, value in data.items():
            plain_key = full_key.split(":")[-1]
            if plain_key == key:
                matches.append(str(value))

        out[f"exif_{key}"] = " | ".join(matches) if matches else ""

    return out


def extract_ffprobe_metadata(file_path):
    if shutil.which("ffprobe") is None:
        return {"ffprobe_error": "ffprobe_not_installed"}

    stdout, stderr, code = run_command([
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ])

    if code != 0 or not stdout.strip():
        return {"ffprobe_error": stderr.strip()}

    try:
        data = json.loads(stdout)
    except Exception as exc:
        return {"ffprobe_error": str(exc)}

    out = {}

    fmt = data.get("format", {})
    out["ffprobe_duration_sec"] = fmt.get("duration", "")
    out["ffprobe_bit_rate"] = fmt.get("bit_rate", "")
    out["ffprobe_format_name"] = fmt.get("format_name", "")

    tags = fmt.get("tags", {})
    out["ffprobe_creation_time"] = tags.get("creation_time", "")

    video_streams = [
        s for s in data.get("streams", [])
        if s.get("codec_type") == "video"
    ]

    if video_streams:
        s = video_streams[0]

        out["ffprobe_video_codec"] = s.get("codec_name", "")
        out["ffprobe_width"] = s.get("width", "")
        out["ffprobe_height"] = s.get("height", "")
        out["ffprobe_avg_frame_rate"] = s.get("avg_frame_rate", "")
        out["ffprobe_r_frame_rate"] = s.get("r_frame_rate", "")
        out["ffprobe_nb_frames"] = s.get("nb_frames", "")

        stream_tags = s.get("tags", {})
        out["ffprobe_stream_creation_time"] = stream_tags.get("creation_time", "")
        out["ffprobe_rotate"] = stream_tags.get("rotate", "")

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
        help="Root folder of external hard disk or media folder.",
    )

    parser.add_argument(
        "--output-csv",
        default="outputs/diagnostics/external_media_metadata_scan.csv",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if args.limit is not None:
        files = files[:args.limit]

    print("Root:", root)
    print("Files found:", len(files))

    rows = []

    for i, file_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {file_path}")

        row = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "extension": file_path.suffix.lower(),
            "parent_folder": str(file_path.parent),
            "size_mb": file_path.stat().st_size / (1024 * 1024),
        }

        exif = extract_exiftool_metadata(file_path)
        ffprobe = extract_ffprobe_metadata(file_path)

        row.update(exif)
        row.update(ffprobe)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\nSaved:", output_csv)
    print("Shape:", df.shape)

    print("\nExtensions:")
    print(df["extension"].value_counts(dropna=False))

    key_cols = [
        "extension",
        "exif_Make",
        "exif_Model",
        "exif_LensModel",
        "exif_FocalLength",
        "exif_FieldOfView",
        "exif_GPSLatitude",
        "exif_GPSLongitude",
        "exif_WhiteBalance",
        "exif_ColorTemperature",
        "ffprobe_width",
        "ffprobe_height",
        "ffprobe_avg_frame_rate",
        "ffprobe_duration_sec",
        "ffprobe_creation_time",
    ]

    key_cols = [c for c in key_cols if c in df.columns]

    print("\nPreview:")
    print(df[key_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()