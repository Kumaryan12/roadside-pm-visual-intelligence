"""Compatibility import for the historical module path."""

try:
    from roadside_pm.features.road.clip_classifier import (
        ROAD_CLASS_MAP,
        ROAD_DUST_SCORE_MAP,
        ROAD_LABELS,
        RoadClassifier,
    )
except ModuleNotFoundError:  # Direct execution from an uninstalled checkout.
    from src.roadside_pm.features.road.clip_classifier import (
        ROAD_CLASS_MAP,
        ROAD_DUST_SCORE_MAP,
        ROAD_LABELS,
        RoadClassifier,
    )

__all__ = ["ROAD_CLASS_MAP", "ROAD_DUST_SCORE_MAP", "ROAD_LABELS", "RoadClassifier"]
