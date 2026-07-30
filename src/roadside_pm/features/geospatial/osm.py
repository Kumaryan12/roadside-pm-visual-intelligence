"""OpenStreetMap feature extraction through the Overpass API."""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any


POI_TAGS = {
    "fuel_station": ("amenity", "fuel"),
    "restaurant": ("amenity", "restaurant"),
    "bus_stop": ("highway", "bus_stop"),
    "parking": ("amenity", "parking"),
    "construction": ("landuse", "construction"),
    "industrial": ("landuse", "industrial"),
    "factory": ("man_made", "works"),
    "warehouse": ("building", "warehouse"),
    "marketplace": ("amenity", "marketplace"),
    "park": ("leisure", "park"),
    "school_college": ("amenity", "school|college|university"),
    "hospital": ("amenity", "hospital|clinic"),
    "commercial": ("landuse", "commercial"),
    "retail": ("landuse", "retail"),
}

ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service", "living_street", "unclassified"]


def build_overpass_query(latitude: float, longitude: float, radius_m: int) -> str:
    parts = ["[out:json][timeout:60];("]
    for _, (key, values) in POI_TAGS.items():
        if "|" in values:
            parts.append(f'nwr(around:{radius_m},{latitude},{longitude})["{key}"~"^({values})$"];')
        else:
            parts.append(f'nwr(around:{radius_m},{latitude},{longitude})["{key}"="{values}"];')
    parts.append(f'way(around:{radius_m},{latitude},{longitude})["highway"];')
    parts.append(");out tags center geom;")
    return "".join(parts)


def build_overpass_bbox_query(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    timeout_s: int = 180,
) -> str:
    """Build one Overpass query for a buffered route tile.

    The returned elements are filtered to the exact per-sample radius locally.
    Fetching a few occupied route tiles is substantially kinder to public
    Overpass services than issuing one request for every moving coordinate.
    """
    bbox = f"({south},{west},{north},{east})"
    parts = [f"[out:json][timeout:{timeout_s}];("]
    for _, (key, values) in POI_TAGS.items():
        if "|" in values:
            parts.append(f'nwr["{key}"~"^({values})$"]{bbox};')
        else:
            parts.append(f'nwr["{key}"="{values}"]{bbox};')
    parts.append(f'way["highway"]{bbox};')
    parts.append(");out tags center geom;")
    return "".join(parts)


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def _point_segment_distance_m(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Approximate point-to-segment distance in a local metric projection."""
    lat0 = math.radians(point[0])
    scale_y = 111_320.0
    scale_x = scale_y * math.cos(lat0)
    px = (point[1] - point[1]) * scale_x
    py = (point[0] - point[0]) * scale_y
    ax = (start[1] - point[1]) * scale_x
    ay = (start[0] - point[0]) * scale_y
    bx = (end[1] - point[1]) * scale_x
    by = (end[0] - point[0]) * scale_y
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.hypot(px - ax, py - ay)
    fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    nearest_x = ax + fraction * dx
    nearest_y = ay + fraction * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def element_distance_m(element: dict[str, Any], point: tuple[float, float]) -> float:
    """Return the nearest available element geometry distance to ``point``."""
    geometry = [
        (float(item["lat"]), float(item["lon"]))
        for item in element.get("geometry", [])
        if "lat" in item and "lon" in item
    ]
    distances: list[float] = []
    if geometry:
        distances.extend(haversine_m(point, item) for item in geometry)
        distances.extend(
            _point_segment_distance_m(point, start, end)
            for start, end in zip(geometry, geometry[1:])
        )
    if "lat" in element and "lon" in element:
        distances.append(haversine_m(point, (float(element["lat"]), float(element["lon"]))))
    center = element.get("center", {})
    if "lat" in center and "lon" in center:
        distances.append(haversine_m(point, (float(center["lat"]), float(center["lon"]))))
    return min(distances, default=float("inf"))


def summarize_nearby_elements(
    elements: list[dict[str, Any]],
    point: tuple[float, float],
    radius_m: int,
) -> dict[str, Any]:
    """Summarize tile elements falling within one sample's exact radius."""
    nearby = [element for element in elements if element_distance_m(element, point) <= radius_m]
    return summarize_elements(nearby, radius_m)


def summarize_elements(elements: list[dict[str, Any]], radius_m: int) -> dict[str, Any]:
    counts = Counter()
    road_counts = Counter()
    total_road_length = 0.0
    road_segments = 0
    for element in elements:
        tags = element.get("tags", {})
        for name, (key, values) in POI_TAGS.items():
            if key in tags and tags[key] in values.split("|"):
                counts[name] += 1
        highway = tags.get("highway")
        if highway:
            road_segments += 1
            if highway in ROAD_CLASSES:
                road_counts[highway] += 1
            geometry = element.get("geometry", [])
            points = [(float(p["lat"]), float(p["lon"])) for p in geometry if "lat" in p and "lon" in p]
            total_road_length += sum(haversine_m(a, b) for a, b in zip(points, points[1:]))
    row = {f"{name}_count_{radius_m}m": int(counts[name]) for name in POI_TAGS}
    row.update({
        f"road_segment_count_{radius_m}m": road_segments,
        f"total_road_length_{radius_m}m": total_road_length,
    })
    for road_class in ROAD_CLASSES:
        column = road_class if road_class == "motorway" else f"{road_class}_road"
        row[f"{column}_count_{radius_m}m"] = int(road_counts[road_class])
    return row


def fetch_overpass(endpoint: str, query: str, *, timeout_s: int = 90, retries: int = 3) -> list[dict[str, Any]]:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(endpoint, data=body, headers={"User-Agent": "roadside-pm-visual-intelligence/0.1"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))["elements"]
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    return []
