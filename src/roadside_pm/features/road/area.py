"""Canonical normalization for metric-depth road-area features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_depth_gated_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Map depth-gated v3 outputs to stable, version-neutral columns."""
    required = [
        "visible_road_area_m2_depth_est",
        "road_area_m2_occlusion_adjusted_conservative_v3",
        "road_area_vehicle_occlusion_fraction_v3",
        "vehicle_occlusion_quality_v3",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing depth-gated v3 road-area columns: {missing}")

    out = df.copy()
    visible = pd.to_numeric(out["visible_road_area_m2_depth_est"], errors="coerce")
    effective = pd.to_numeric(
        out["road_area_m2_occlusion_adjusted_conservative_v3"], errors="coerce"
    )
    occluded = (effective - visible).clip(lower=0.0)

    out["road_area_segmented_unoccluded_m2"] = visible
    out["road_area_vehicle_occluded_m2"] = occluded
    out["road_area_effective_visible_m2"] = effective
    out["road_area_vehicle_occlusion_fraction"] = pd.to_numeric(
        out["road_area_vehicle_occlusion_fraction_v3"], errors="coerce"
    )
    out["road_area_occlusion_quality"] = out["vehicle_occlusion_quality_v3"].astype(str)
    out["road_area_feature_version"] = "depth_gated_vehicle_footprint_v3"
    out["road_area_features_available"] = (
        np.isfinite(visible) & np.isfinite(effective) & (visible >= 0) & (effective >= visible)
    )
    return out

