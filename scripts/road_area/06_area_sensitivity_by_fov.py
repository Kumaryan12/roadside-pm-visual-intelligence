from pathlib import Path
import argparse
import subprocess
import math
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--depth-npy", required=True)
    parser.add_argument("--image-width", type=float, required=True)
    parser.add_argument("--fovs", nargs="+", type=float, default=[50, 60, 70, 80, 90, 100])
    parser.add_argument("--max-depth-m", type=float, default=30.0)
    parser.add_argument("--output-dir", default="outputs/road_area_inputs/fov_sensitivity")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for fov in args.fovs:
        fx = args.image_width / (2 * math.tan(math.radians(fov / 2)))

        out_csv = output_dir / f"road_area_hfov_{int(fov)}.csv"

        cmd = [
            "python",
            "scripts/road_validation/05_estimate_road_area_from_metric_depth.py",
            "--image", args.image,
            "--mask", args.mask,
            "--depth-npy", args.depth_npy,
            "--fx", str(fx),
            "--fy", str(fx),
            "--max-depth-m", str(args.max_depth_m),
            "--output-csv", str(out_csv),
        ]

        subprocess.run(cmd, check=True)

        df = pd.read_csv(out_csv)
        row = df.iloc[0].to_dict()
        row["horizontal_fov_deg"] = fov
        row["derived_fx_px"] = fx
        rows.append(row)

    summary = pd.DataFrame(rows)

    summary_path = output_dir / "road_area_fov_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    cols = [
        "horizontal_fov_deg",
        "derived_fx_px",
        "estimated_road_area_m2",
        "road_pixels_used_for_area",
        "valid_road_quads_used",
        "max_depth_m",
    ]

    print(summary[cols].to_string(index=False))
    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()