"""CLIP-based road condition and visual-dust classifier.

Migrated from ``src.road_features.clip_road_classifier``. Model loading remains
lazy at class construction, matching the historical behavior.
"""

ROAD_LABELS = [
    "clean paved asphalt road with no visible dust",
    "slightly dusty paved road with a thin layer of road dust",
    "dry unpaved dirt road with loose soil and gravel",
    "wet road surface after rain with water reflections",
    "road near construction site with sand dust and debris",
    "road covered with heavy loose dust and dry soil",
    "hazy polluted urban road with poor air visibility",
    "rough broken road with potholes dirt patches and uneven surface",
]

ROAD_CLASS_MAP = {
    "clean paved asphalt road with no visible dust": "clean_paved_road",
    "slightly dusty paved road with a thin layer of road dust": "dusty_paved_road",
    "dry unpaved dirt road with loose soil and gravel": "unpaved_dirt_road",
    "wet road surface after rain with water reflections": "wet_road",
    "road near construction site with sand dust and debris": "construction_dust_road",
    "road covered with heavy loose dust and dry soil": "heavy_loose_dust_road",
    "hazy polluted urban road with poor air visibility": "hazy_polluted_road",
    "rough broken road with potholes dirt patches and uneven surface": "rough_dusty_road",
}

ROAD_DUST_SCORE_MAP = {
    "clean paved asphalt road with no visible dust": 0.10,
    "wet road surface after rain with water reflections": 0.15,
    "slightly dusty paved road with a thin layer of road dust": 0.45,
    "hazy polluted urban road with poor air visibility": 0.55,
    "rough broken road with potholes dirt patches and uneven surface": 0.65,
    "dry unpaved dirt road with loose soil and gravel": 0.80,
    "road near construction site with sand dust and debris": 0.90,
    "road covered with heavy loose dust and dry soil": 1.00,
}


class RoadClassifier:
    def __init__(self):
        # Keep heavyweight native ML imports out of module discovery and CLI
        # registry commands. They are loaded only when this feature is used.
        from transformers import pipeline

        self.classifier = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
        )

    def classify(self, image_path: str) -> dict:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        predictions = self.classifier(image, candidate_labels=ROAD_LABELS)

        best_prediction = predictions[0]
        top_label = best_prediction["label"]
        top_confidence = float(best_prediction["score"])
        simplified_condition = ROAD_CLASS_MAP.get(top_label, "unknown_road")

        weighted_dust_score = 0.0
        total_score = 0.0
        for prediction in predictions:
            label = prediction["label"]
            score = float(prediction["score"])
            weighted_dust_score += score * ROAD_DUST_SCORE_MAP.get(label, 0.50)
            total_score += score

        if total_score > 0:
            road_dust_score = weighted_dust_score / total_score
        else:
            road_dust_score = ROAD_DUST_SCORE_MAP.get(top_label, 0.50)

        if road_dust_score <= 0.30:
            dust_level = "low"
        elif road_dust_score <= 0.65:
            dust_level = "moderate"
        else:
            dust_level = "high"

        return {
            "road_condition": simplified_condition,
            "road_condition_full_label": top_label,
            "road_condition_confidence": top_confidence,
            "clip_road_dust_score": road_dust_score,
            "road_dust_score": road_dust_score,
            "road_dust_level": dust_level,
        }


__all__ = [
    "ROAD_CLASS_MAP",
    "ROAD_DUST_SCORE_MAP",
    "ROAD_LABELS",
    "RoadClassifier",
]
