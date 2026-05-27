import cv2
import numpy as np


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(float(value), max_value))


def extract_basic_image_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "brightness_mean": 0.0,
            "contrast_std": 0.0,
            "brown_pixel_ratio": 0.0,
            "edge_density": 0.0,
            "haze_score": 0.0,
            "visual_dust_score": 0.0
        }

    image = cv2.resize(image, (640, 360))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Approximate brown/yellow dust color range in HSV.
    # This catches soil, sand, dry road dust, and loose brown particles.
    lower_brown = np.array([8, 30, 35])
    upper_brown = np.array([40, 255, 240])

    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_pixel_ratio = float(np.sum(brown_mask > 0) / brown_mask.size)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.sum(edges > 0) / edges.size)

    # Haze-like images often have lower contrast.
    brightness_norm = brightness_mean / 255.0
    contrast_norm = clamp(contrast_std / 80.0)

    haze_score = clamp((1.0 - contrast_norm) * brightness_norm)

    # Convert brown-pixel ratio into stronger 0-1 signal.
    brown_component = clamp(brown_pixel_ratio * 3.0)

    # Edge density can indicate texture/roughness, but too much edge may also
    # come from vehicles/buildings. So use it lightly.
    texture_component = clamp(edge_density * 5.0)

    visual_dust_score = clamp(
        0.55 * brown_component
        + 0.30 * haze_score
        + 0.15 * texture_component
    )

    return {
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "brown_pixel_ratio": brown_pixel_ratio,
        "edge_density": edge_density,
        "haze_score": haze_score,
        "visual_dust_score": visual_dust_score
    }