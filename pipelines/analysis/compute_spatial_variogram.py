"""Compute exact empirical spatial variograms from latitude/longitude data.

The implementation uses Haversine distance and blockwise pair accumulation so
it does not materialize the full N-by-N distance matrix.  It reports classical
semivariance and the Cressie-Hawkins robust estimator, both pooled and within
groups such as collection date.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--lat-col", default="lat")
    parser.add_argument("--lon-col", default="long")
    parser.add_argument("--group-col", default="date")
    parser.add_argument("--bin-width-km", type=float, default=0.5)
    parser.add_argument("--max-distance-km", type=float, default=50.0)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _accumulate_pairs(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = len(edges) - 1
    count = np.zeros(bins, dtype=np.int64)
    distance_sum = np.zeros(bins, dtype=float)
    semivariance_sum = np.zeros(bins, dtype=float)
    sqrt_difference_sum = np.zeros(bins, dtype=float)
    n = len(values)

    def add(distance: np.ndarray, difference: np.ndarray) -> None:
        flat_distance = distance.ravel()
        flat_difference = difference.ravel()
        index = np.searchsorted(edges, flat_distance, side="right") - 1
        valid = (index >= 0) & (index < bins) & np.isfinite(flat_difference)
        index = index[valid]
        flat_distance = flat_distance[valid]
        flat_difference = flat_difference[valid]
        count[:] += np.bincount(index, minlength=bins)
        distance_sum[:] += np.bincount(index, weights=flat_distance, minlength=bins)
        semivariance_sum[:] += np.bincount(
            index, weights=0.5 * flat_difference**2, minlength=bins
        )
        sqrt_difference_sum[:] += np.bincount(
            index, weights=np.sqrt(np.abs(flat_difference)), minlength=bins
        )

    for start_i in range(0, n, block_size):
        stop_i = min(start_i + block_size, n)
        lat_i = lat[start_i:stop_i]
        lon_i = lon[start_i:stop_i]
        value_i = values[start_i:stop_i]
        for start_j in range(start_i, n, block_size):
            stop_j = min(start_j + block_size, n)
            lat_j = lat[start_j:stop_j]
            lon_j = lon[start_j:stop_j]
            value_j = values[start_j:stop_j]
            delta_lat = lat_j[None, :] - lat_i[:, None]
            delta_lon = lon_j[None, :] - lon_i[:, None]
            a = (
                np.sin(delta_lat / 2.0) ** 2
                + np.cos(lat_i[:, None]) * np.cos(lat_j[None, :])
                * np.sin(delta_lon / 2.0) ** 2
            )
            distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
            difference = value_i[:, None] - value_j[None, :]
            if start_i == start_j:
                upper = np.triu_indices(stop_i - start_i, k=1)
                add(distance[upper], difference[upper])
            else:
                add(distance, difference)
    return count, distance_sum, semivariance_sum, sqrt_difference_sum


def empirical_variogram(
    frame: pd.DataFrame,
    *,
    target: str,
    lat_col: str,
    lon_col: str,
    edges: np.ndarray,
    block_size: int,
) -> pd.DataFrame:
    lat = np.radians(frame[lat_col].to_numpy(dtype=float))
    lon = np.radians(frame[lon_col].to_numpy(dtype=float))
    values = frame[target].to_numpy(dtype=float)
    count, distance_sum, semivariance_sum, root_sum = _accumulate_pairs(
        lat, lon, values, edges, block_size
    )
    result = pd.DataFrame({
        "distance_bin_start_km": edges[:-1],
        "distance_bin_end_km": edges[1:],
        "pair_count": count,
    })
    nonempty = count > 0
    result["mean_distance_km"] = np.nan
    result.loc[nonempty, "mean_distance_km"] = distance_sum[nonempty] / count[nonempty]
    result["semivariance_classical"] = np.nan
    result.loc[nonempty, "semivariance_classical"] = (
        semivariance_sum[nonempty] / count[nonempty]
    )
    correction = 0.457 + 0.494 / np.maximum(count, 1) + 0.045 / np.maximum(count, 1) ** 2
    result["semivariance_cressie_hawkins"] = np.nan
    result.loc[nonempty, "semivariance_cressie_hawkins"] = (
        0.5 * (root_sum[nonempty] / count[nonempty]) ** 4 / correction[nonempty]
    )
    return result


def main() -> int:
    args = parse_args()
    if args.bin_width_km <= 0 or args.max_distance_km <= 0:
        raise ValueError("Distance limits must be positive")
    required = [args.target, args.lat_col, args.lon_col, args.group_col]
    frame = pd.read_csv(args.input_csv, usecols=required)
    for column in [args.target, args.lat_col, args.lon_col]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    original_rows = len(frame)
    frame = frame.dropna(subset=[args.target, args.lat_col, args.lon_col]).reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("At least two complete observations are required")
    edges = np.arange(0.0, args.max_distance_km + args.bin_width_km, args.bin_width_km)
    if edges[-1] < args.max_distance_km:
        edges = np.append(edges, args.max_distance_km)

    print(f"Computing pooled variogram for {len(frame)} rows", flush=True)
    pooled = empirical_variogram(
        frame, target=args.target, lat_col=args.lat_col, lon_col=args.lon_col,
        edges=edges, block_size=args.block_size,
    )
    pooled.insert(0, "scope", "pooled")

    accumulators = []
    for group, part in frame.groupby(args.group_col, sort=True):
        print(f"Computing within-{args.group_col} pairs for {group}: {len(part)} rows", flush=True)
        item = empirical_variogram(
            part, target=args.target, lat_col=args.lat_col, lon_col=args.lon_col,
            edges=edges, block_size=args.block_size,
        )
        item.insert(0, "scope", f"within_{args.group_col}")
        item.insert(1, args.group_col, str(group))
        accumulators.append(item)
    by_group = pd.concat(accumulators, ignore_index=True)

    # Aggregate within-group sufficient statistics by recomputing exact pairs
    # per group and combining the resulting pair-weighted estimators.
    within = by_group.groupby(
        ["distance_bin_start_km", "distance_bin_end_km"], as_index=False
    ).apply(
        lambda part: pd.Series({
            "pair_count": int(part["pair_count"].sum()),
            "mean_distance_km": np.average(
                part.loc[part["pair_count"] > 0, "mean_distance_km"],
                weights=part.loc[part["pair_count"] > 0, "pair_count"],
            ) if part["pair_count"].sum() else np.nan,
            "semivariance_classical": np.average(
                part.loc[part["pair_count"] > 0, "semivariance_classical"],
                weights=part.loc[part["pair_count"] > 0, "pair_count"],
            ) if part["pair_count"].sum() else np.nan,
            # This is a pair-weighted summary of group-level robust estimates,
            # deliberately labelled rather than treated as a pooled CH estimate.
            "semivariance_cressie_hawkins": np.average(
                part.loc[part["pair_count"] > 0, "semivariance_cressie_hawkins"],
                weights=part.loc[part["pair_count"] > 0, "pair_count"],
            ) if part["pair_count"].sum() else np.nan,
        }),
        include_groups=False,
    ).reset_index(drop=True)
    within.insert(0, "scope", f"within_{args.group_col}_aggregate")
    combined = pd.concat([pooled, within], ignore_index=True, sort=False)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "empirical_variogram.csv", index=False)
    by_group.to_csv(output / f"empirical_variogram_by_{args.group_col}.csv", index=False)

    try:
        if args.no_plot:
            raise ImportError
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for scope, part in combined.groupby("scope", sort=False):
            label = scope.replace("_", " ")
            axes[0].plot(part["mean_distance_km"], part["semivariance_classical"], label=label)
            axes[1].plot(
                part["mean_distance_km"], part["semivariance_cressie_hawkins"], label=label
            )
        axes[0].set_title("Classical empirical variogram")
        axes[1].set_title("Robust empirical variogram")
        for axis in axes:
            axis.set_xlabel("Haversine distance (km)")
            axis.set_ylabel("Semivariance ((µg/m³)²)")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(output / "empirical_variogram.png", dpi=180)
        plt.close(fig)
    except ImportError:
        print("plot disabled or matplotlib unavailable; skipped plot", flush=True)

    first = within[within["pair_count"] > 0].iloc[0]
    summary = {
        "input_csv": args.input_csv,
        "target": args.target,
        "distance": "Haversine great-circle distance",
        "rows_input": original_rows,
        "rows_complete": len(frame),
        "groups": int(frame[args.group_col].nunique()),
        "bin_width_km": args.bin_width_km,
        "max_distance_km": args.max_distance_km,
        "target_summary": {
            "mean": float(frame[args.target].mean()),
            "variance_population": float(frame[args.target].var(ddof=0)),
            "minimum": float(frame[args.target].min()),
            "maximum": float(frame[args.target].max()),
        },
        "first_within_group_bin": first.to_dict(),
        "interpretation_warning": (
            "Spatial semivariance is confounded by temporal autocorrelation, repeated routes, "
            "nonstationary day effects, and unequal spatial sampling. Within-date estimates "
            "reduce but do not remove those effects."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved variogram outputs to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
