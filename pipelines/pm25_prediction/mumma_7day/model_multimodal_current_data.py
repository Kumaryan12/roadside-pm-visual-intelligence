"""Benchmark current MUMMA sensor and engineered visual features.

Whole-date holdout is the primary protocol. Random-row results are retained as
non-reportable diagnostics. PM/OPC channels related to the target are excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pipelines.pm25_prediction.mumma_7day.model_current_data import (
    MET_GAS,
    MOBILITY,
    TIME,
    aggregate_metrics,
    engineer_features,
    evaluate_fold,
)
from roadside_pm.features.geospatial.alphaearth import BANDS as ALPHAEARTH_BANDS


YOLO_SUM = [
    "idd_total_vehicle_count",
    "idd_heavy_vehicle_count",
    "idd_motor_vehicle_count",
    "idd_auto_rickshaw_count",
    "idd_bicycle_count",
    "idd_bus_count",
    "idd_car_count",
    "idd_motorcycle_count",
    "idd_truck_count",
    "idd_unknown_vehicle_count",
    "idd_exhaust_proxy_initial",
    "idd_resuspension_vehicle_proxy_initial",
]
YOLO_MEAN_MAX = [
    "idd_vehicle_box_area_ratio",
    "idd_average_confidence",
    "idd_max_confidence",
]
ROAD_FEATURES = [
    "road_area_ratio",
    "road_mean_brightness",
    "road_mean_saturation",
    "road_contrast_std",
    "road_shadow_ratio",
    "road_glare_ratio",
    "road_brown_pixel_ratio",
    "road_gray_dry_pixel_ratio",
    "road_edge_density",
    "road_laplacian_std",
    "road_haze_flatness_proxy",
]
ROAD_AREA_FEATURES = [
    "raw_road_pixels_before_cleaning",
    "cleaned_road_pixels_after_cleaning",
    "removed_road_fraction_by_cleaning",
    "road_depth_p5_m",
    "road_depth_p25_m",
    "road_depth_p50_m",
    "road_depth_p75_m",
    "road_depth_p95_m",
    "road_depth_std_m",
    "estimated_road_area_m2",
    "road_pixels_used_for_area",
    "valid_road_quads_used",
    "road_area_ratio_px",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-csv", required=True)
    parser.add_argument("--yolo-csv", required=True)
    parser.add_argument("--road-csv")
    parser.add_argument("--road-area-csv")
    parser.add_argument("--osm-csv")
    parser.add_argument("--alphaearth-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="sPM2")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def aggregate_yolo(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    frame = frame[frame["idd_detection_status"] == "success"].copy()
    required = ["sample_id", "lens_id", *YOLO_SUM, *YOLO_MEAN_MAX]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"YOLO table is missing columns: {missing}")
    for column in YOLO_SUM + YOLO_MEAN_MAX:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    named = {}
    for column in YOLO_SUM:
        named[f"{column}_all_lenses_sum"] = (column, "sum")
    for column in YOLO_MEAN_MAX:
        named[f"{column}_all_lenses_mean"] = (column, "mean")
        named[f"{column}_all_lenses_max"] = (column, "max")
    named["yolo_lens_count"] = ("lens_id", "nunique")
    aggregated = frame.groupby("sample_id", sort=False).agg(**named).reset_index()

    per_lens_source = YOLO_SUM + ["idd_vehicle_box_area_ratio"]
    pivot = frame.pivot(index="sample_id", columns="lens_id", values=per_lens_source)
    pivot.columns = [f"{column}_lens{int(lens)}" for column, lens in pivot.columns]
    pivot = pivot.reset_index()
    aggregated = aggregated.merge(pivot, on="sample_id", validate="one_to_one")
    features = [column for column in aggregated.columns if column != "sample_id"]
    return aggregated, features


def aggregate_road(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    frame = frame[frame["road_condition_status"] == "success"].copy()
    if "sample_id" not in frame.columns:
        frame["sample_id"] = frame["processed_frame_key"].astype(str).str.replace(
            r"_lens\d+_preprocessed_lens\d+$", "", regex=True,
        )
        if frame["sample_id"].eq(frame["processed_frame_key"].astype(str)).any():
            raise ValueError("Could not recover sample_id from some road processed_frame_key values")
    available = [column for column in ROAD_FEATURES if column in frame.columns]
    if not available:
        raise ValueError("Road table has no expected road feature columns")
    for column in available:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    named = {}
    for column in available:
        named[f"{column}_all_lenses_mean"] = (column, "mean")
        named[f"{column}_all_lenses_max"] = (column, "max")
    named["road_lens_count"] = ("lens_id", "nunique")
    aggregated = frame.groupby("sample_id", sort=False).agg(**named).reset_index()
    pivot = frame.pivot(index="sample_id", columns="lens_id", values=available)
    pivot.columns = [f"{column}_lens{int(lens)}" for column, lens in pivot.columns]
    aggregated = aggregated.merge(pivot.reset_index(), on="sample_id", validate="one_to_one")
    features = [column for column in aggregated.columns if column != "sample_id"]
    return aggregated, features


def aggregate_road_area(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    required = {"sample_id", "area_status", "area_quality_flag"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Road-area table is missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Road-area table contains duplicate sample_id values")
    available = [column for column in ROAD_AREA_FEATURES if column in frame.columns]
    if not available:
        raise ValueError("Road-area table has no expected numeric features")
    result = frame[["sample_id", "area_status", "area_quality_flag", *available]].copy()
    for column in available:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["road_area_available"] = result["area_status"].astype(str).eq("success").astype(int)
    result["road_area_low_mask_flag"] = result["area_quality_flag"].astype(str).eq(
        "bad_low_road_mask_area"
    ).astype(int)
    result["road_area_needs_review_flag"] = result["area_quality_flag"].astype(str).eq(
        "needs_visual_review"
    ).astype(int)
    features = [
        *available,
        "road_area_available",
        "road_area_low_mask_flag",
        "road_area_needs_review_flag",
    ]
    return result[["sample_id", *features]], features


def validated_osm(path: Path, expected_sample_ids: set[str]) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    if "sample_id" not in frame or "osm_status" not in frame:
        raise ValueError("OSM table must contain sample_id and osm_status")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError("OSM table contains duplicate sample_id values")
    observed = set(frame["sample_id"])
    missing = expected_sample_ids - observed
    extra = observed - expected_sample_ids
    if missing or extra:
        raise ValueError(f"OSM sample coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    failed = ~frame["osm_status"].astype(str).str.lower().eq("success")
    if failed.any():
        raise ValueError(f"OSM table contains {int(failed.sum())} failed rows; rerun extraction")
    excluded = {
        "sample_id", "sample_index", "osm_location_key", "osm_status", "osm_error",
        "osm_latitude_rounded", "osm_longitude_rounded",
    }
    features = []
    for column in frame.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            frame[column] = converted
            features.append(column)
    if not features:
        raise ValueError("OSM table contains no complete numeric context features")
    return frame[["sample_id", *features]].copy(), features


def validated_alphaearth(
    path: Path, expected_sample_ids: set[str]
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    required = {"sample_id", "alphaearth_status", *ALPHAEARTH_BANDS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"AlphaEarth table is missing columns: {missing_columns}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError("AlphaEarth table contains duplicate sample_id values")
    observed = set(frame["sample_id"])
    missing = expected_sample_ids - observed
    extra = observed - expected_sample_ids
    if missing or extra:
        raise ValueError(
            f"AlphaEarth sample coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    failed = ~frame["alphaearth_status"].astype(str).str.lower().eq("success")
    if failed.any():
        raise ValueError(
            f"AlphaEarth table contains {int(failed.sum())} failed rows; rerun extraction"
        )
    for column in ALPHAEARTH_BANDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_values = int(frame[ALPHAEARTH_BANDS].isna().sum().sum())
    if missing_values:
        raise ValueError(f"AlphaEarth embedding bands contain {missing_values} missing values")
    features = [f"alphaearth_{band}" for band in ALPHAEARTH_BANDS]
    result = frame[["sample_id", *ALPHAEARTH_BANDS]].rename(
        columns=dict(zip(ALPHAEARTH_BANDS, features, strict=True))
    )
    return result, features


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = engineer_features(pd.read_csv(args.sensor_csv))
    if not frame["sample_id"].is_unique:
        raise ValueError("Sensor sample_id must be unique")
    frame["sample_id"] = frame["sample_id"].astype(str)

    yolo, yolo_features = aggregate_yolo(Path(args.yolo_csv))
    frame = frame.merge(yolo, on="sample_id", how="left", validate="one_to_one")
    if frame[yolo_features].isna().all(axis=1).any():
        raise ValueError("Some sensor samples have no YOLO feature row")

    road_features: list[str] = []
    if args.road_csv:
        road, road_features = aggregate_road(Path(args.road_csv))
        frame = frame.merge(road, on="sample_id", how="left", validate="one_to_one")
        frame["road_features_available"] = ~frame[road_features].isna().all(axis=1)

    road_area_features: list[str] = []
    if args.road_area_csv:
        road_area, road_area_features = aggregate_road_area(Path(args.road_area_csv))
        frame = frame.merge(road_area, on="sample_id", how="left", validate="one_to_one")

    osm_features: list[str] = []
    if args.osm_csv:
        osm, osm_features = validated_osm(Path(args.osm_csv), set(frame["sample_id"]))
        frame = frame.merge(osm, on="sample_id", how="left", validate="one_to_one")

    alphaearth_features: list[str] = []
    if args.alphaearth_csv:
        alphaearth, alphaearth_features = validated_alphaearth(
            Path(args.alphaearth_csv), set(frame["sample_id"])
        )
        frame = frame.merge(alphaearth, on="sample_id", how="left", validate="one_to_one")

    sensor_features = MET_GAS + TIME + MOBILITY
    visual_features = yolo_features + road_features
    visual_area_features = visual_features + road_area_features
    groups = {
        "sensor_met_gas_time_mobility": {
            "columns": sensor_features,
            "reportable": True,
            "note": "Non-PM sensor, time, and mobility features.",
        },
        "visual_yolo" + ("_road" if road_features else ""): {
            "columns": visual_features,
            "reportable": True,
            "note": "Fine-tuned IDD traffic" + (" and SegFormer road" if road_features else "") + " features.",
        },
        "sensor_plus_visual": {
            "columns": sensor_features + visual_features,
            "reportable": True,
            "note": "Residual-fusion candidate table without PM/OPC target proxies.",
        },
    }
    if road_area_features:
        groups["visual_yolo_road_area_depth"] = {
            "columns": visual_area_features,
            "reportable": False,
            "note": "YOLO, road appearance, and provisional metric depth/area features; camera calibration and area masks require validation.",
        }
        groups["sensor_plus_visual_area_depth"] = {
            "columns": sensor_features + visual_area_features,
            "reportable": False,
            "note": "Non-PM sensor plus visual and provisional metric depth/area features.",
        }
    if osm_features:
        groups["visual_yolo_road_osm"] = {
            "columns": visual_features + osm_features,
            "reportable": True,
            "note": "Fine-tuned IDD traffic, SegFormer road, and validated OSM context features.",
        }
        groups["sensor_plus_visual_osm"] = {
            "columns": sensor_features + visual_features + osm_features,
            "reportable": True,
            "note": "Non-PM sensor plus visual and validated OSM context features.",
        }
        if road_area_features:
            groups["visual_yolo_road_area_depth_osm"] = {
                "columns": visual_area_features + osm_features,
                "reportable": False,
                "note": "Visual, provisional area/depth, and validated OSM context features.",
            }
            groups["sensor_plus_visual_area_depth_osm"] = {
                "columns": sensor_features + visual_area_features + osm_features,
                "reportable": False,
                "note": "Non-PM sensor, visual, provisional area/depth, and validated OSM context features.",
            }
    if alphaearth_features:
        groups["alphaearth"] = {
            "columns": alphaearth_features,
            "reportable": True,
            "note": "Fixed pretrained 2025 AlphaEarth point-embedding bands; no target-fitted preprocessing.",
        }
        groups["sensor_plus_alphaearth"] = {
            "columns": sensor_features + alphaearth_features,
            "reportable": True,
            "note": "Non-PM sensor context plus fixed pretrained AlphaEarth point embeddings.",
        }
        groups["visual_yolo_road_alphaearth"] = {
            "columns": visual_features + alphaearth_features,
            "reportable": True,
            "note": "YOLO/road context plus fixed pretrained AlphaEarth point embeddings.",
        }
        groups["sensor_plus_visual_alphaearth"] = {
            "columns": sensor_features + visual_features + alphaearth_features,
            "reportable": True,
            "note": "Non-PM sensor, YOLO/road, and fixed pretrained AlphaEarth point embeddings.",
        }
        if osm_features:
            groups["visual_yolo_road_osm_alphaearth"] = {
                "columns": visual_features + osm_features + alphaearth_features,
                "reportable": True,
                "note": "YOLO/road, validated OSM, and fixed pretrained AlphaEarth context.",
            }
            groups["sensor_plus_visual_osm_alphaearth"] = {
                "columns": sensor_features + visual_features + osm_features + alphaearth_features,
                "reportable": True,
                "note": "Complete non-PM sensor, visual, OSM, and AlphaEarth context.",
            }

    dates = sorted(frame["date"].unique())
    if len(dates) < 2:
        raise ValueError(f"At least two dates are required for leave-one-day-out evaluation; found {dates}")
    metric_rows = []
    prediction_tables = []
    for test_date in dates:
        test_index = frame.index[frame["date"] == test_date].to_numpy()
        train_index = frame.index[frame["date"] != test_date].to_numpy()
        rows, tables = evaluate_fold(
            frame, train_index, test_index, protocol="leave_one_day_out",
            fold=f"test_{test_date}", groups=groups, seed=args.random_state,
            target=args.target,
        )
        metric_rows.extend(rows)
        prediction_tables.extend(tables)

    train_index, test_index = train_test_split(
        frame.index.to_numpy(), test_size=0.2, random_state=args.random_state,
    )
    rows, tables = evaluate_fold(
        frame, train_index, test_index, protocol="random_row",
        fold=f"seed_{args.random_state}", groups=groups,
        seed=args.random_state, target=args.target,
    )
    metric_rows.extend(rows)
    prediction_tables.extend(tables)

    folds = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    aggregate = aggregate_metrics(predictions)
    frame.to_csv(output / "modeling_table.csv", index=False)
    folds.to_csv(output / "metrics_by_fold.csv", index=False)
    aggregate.to_csv(output / "metrics_aggregate.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "feature_groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    run = {
        "sensor_csv": args.sensor_csv,
        "yolo_csv": args.yolo_csv,
        "road_csv": args.road_csv,
        "road_area_csv": args.road_area_csv,
        "osm_csv": args.osm_csv,
        "alphaearth_csv": args.alphaearth_csv,
        "alphaearth_feature_count": len(alphaearth_features),
        "rows": len(frame),
        "dates": dates,
        "primary_protocol": "leave_one_day_out",
        "leave_one_day_out_folds": len(dates),
        "random_row_reportable": False,
        "pm_opc_target_proxies_used": False,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(aggregate.to_string(index=False))
    print(f"Saved results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
