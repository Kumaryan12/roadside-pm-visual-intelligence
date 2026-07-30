"""Compatibility import for the historical module path."""

try:
    from roadside_pm.features.road.opencv import clamp, extract_basic_image_features
except ModuleNotFoundError:  # Direct execution from an uninstalled checkout.
    from src.roadside_pm.features.road.opencv import clamp, extract_basic_image_features

__all__ = ["clamp", "extract_basic_image_features"]
