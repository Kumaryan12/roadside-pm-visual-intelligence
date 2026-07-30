from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


DEFAULT_MODEL_PATH = (
    "models/road_segmentation/"
    "best_segformer_b0_idd_binary_road/"
    "best_segformer_b0_idd_binary_road"
)

FALLBACK_PROCESSOR = "nvidia/segformer-b0-finetuned-cityscapes-768-768"


# Based on your measured lens statistics.
LENS_REFERENCE_STATS = {
    "lens1": {
        "target_v_mean": 0.343834,
        "target_s_mean": 0.231930,
    },
    "lens6": {
        "target_v_mean": 0.402573,
        "target_s_mean": 0.177737,
    },
    "lens1_lens6_average": {
        "target_v_mean": (0.343834 + 0.402573) / 2,
        "target_s_mean": (0.231930 + 0.177737) / 2,
    },
}


st.set_page_config(
    page_title="Road Feature Normalization Validator",
    layout="wide",
)


def pil_to_bgr(pil_image: Image.Image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@st.cache_resource
def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = SegformerImageProcessor.from_pretrained(model_path)
    except Exception:
        processor = SegformerImageProcessor.from_pretrained(FALLBACK_PROCESSOR)

    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return processor, model, model.config.id2label, device


def get_road_class_ids(id2label, num_labels):
    road_ids = []

    for class_id, label in id2label.items():
        if str(label).lower() == "road":
            road_ids.append(int(class_id))

    # For binary fine-tuned model: 0 = background, 1 = road
    if not road_ids and num_labels == 2:
        road_ids = [1]

    if not road_ids:
        raise ValueError(f"Could not identify road class from id2label={id2label}")

    return road_ids


def segment_road(image_bgr, processor, model, id2label, device):
    image_rgb = bgr_to_rgb(image_bgr)
    pil_image = Image.fromarray(image_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )

    pred = upsampled_logits.argmax(dim=1)[0].detach().cpu().numpy()

    road_ids = get_road_class_ids(
        id2label=id2label,
        num_labels=model.config.num_labels,
    )

    road_mask = np.isin(pred, road_ids).astype(np.uint8) * 255

    return road_mask


def clean_road_mask(image_bgr, road_mask, black_threshold=25, morph_kernel_size=5):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]

    non_black_mask = (value_channel > black_threshold).astype(np.uint8) * 255
    cleaned = cv2.bitwise_and(road_mask, non_black_mask)

    if morph_kernel_size > 0:
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


def gray_world_white_balance(image_bgr, strength=1.0):
    """
    Mild gray-world white balance.
    strength=0 gives original image.
    strength=1 gives full gray-world correction.
    """
    image = image_bgr.astype(np.float32)

    b_mean = image[:, :, 0].mean()
    g_mean = image[:, :, 1].mean()
    r_mean = image[:, :, 2].mean()

    gray_mean = np.mean([b_mean, g_mean, r_mean])

    gain_b = gray_mean / max(b_mean, 1e-6)
    gain_g = gray_mean / max(g_mean, 1e-6)
    gain_r = gray_mean / max(r_mean, 1e-6)

    # Blend gains with 1.0 for mild correction
    gain_b = 1.0 + strength * (gain_b - 1.0)
    gain_g = 1.0 + strength * (gain_g - 1.0)
    gain_r = 1.0 + strength * (gain_r - 1.0)

    image[:, :, 0] *= gain_b
    image[:, :, 1] *= gain_g
    image[:, :, 2] *= gain_r

    return np.clip(image, 0, 255).astype(np.uint8)


def match_hsv_reference(
    image_bgr,
    road_mask,
    target_v_mean,
    target_s_mean,
    strength=1.0,
    max_scale_change=0.35,
):
    """
    Match road-region brightness and saturation to reference values.
    Applies scale only to S and V channels.
    """
    if road_mask is None or (road_mask > 0).sum() == 0:
        return image_bgr.copy()

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    road_region = road_mask > 0

    current_s_mean = hsv[:, :, 1][road_region].mean() / 255.0
    current_v_mean = hsv[:, :, 2][road_region].mean() / 255.0

    s_scale = target_s_mean / max(current_s_mean, 1e-6)
    v_scale = target_v_mean / max(current_v_mean, 1e-6)

    # Avoid extreme correction
    min_scale = 1.0 - max_scale_change
    max_scale = 1.0 + max_scale_change

    s_scale = float(np.clip(s_scale, min_scale, max_scale))
    v_scale = float(np.clip(v_scale, min_scale, max_scale))

    # Blend with no correction
    s_scale = 1.0 + strength * (s_scale - 1.0)
    v_scale = 1.0 + strength * (v_scale - 1.0)

    hsv[:, :, 1] *= s_scale
    hsv[:, :, 2] *= v_scale

    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out


def normalize_for_feature_extraction(
    image_bgr,
    road_mask,
    mode,
    reference_name,
    wb_strength,
    hsv_strength,
):
    out = image_bgr.copy()

    if mode in ["gray_world", "gray_world_plus_hsv_reference"]:
        out = gray_world_white_balance(out, strength=wb_strength)

    if mode in ["hsv_reference", "gray_world_plus_hsv_reference"]:
        ref = LENS_REFERENCE_STATS[reference_name]
        out = match_hsv_reference(
            image_bgr=out,
            road_mask=road_mask,
            target_v_mean=ref["target_v_mean"],
            target_s_mean=ref["target_s_mean"],
            strength=hsv_strength,
        )

    return out


def make_overlay(image_bgr, mask, color=(0, 255, 0), alpha=0.40):
    overlay = image_bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)


def extract_lens_quality_stats(image_bgr, road_mask=None):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    stats = {
        "image_value_mean": float(v.mean() / 255.0),
        "image_saturation_mean": float(s.mean() / 255.0),
        "image_contrast_std": float(gray.std() / 255.0),
        "black_ratio_v_lt_25": float((v < 25).mean()),
        "glare_ratio_v_gt_240": float((v > 240).mean()),
        "rgb_r_over_g": float(r.mean() / max(g.mean(), 1e-6)),
        "rgb_b_over_g": float(b.mean() / max(g.mean(), 1e-6)),
    }

    if road_mask is not None and (road_mask > 0).sum() > 0:
        road_region = road_mask > 0
        stats.update({
            "road_value_mean": float(v[road_region].mean() / 255.0),
            "road_saturation_mean": float(s[road_region].mean() / 255.0),
            "road_contrast_std": float(gray[road_region].std() / 255.0),
        })

    return stats


def extract_road_features(
    image_bgr,
    road_mask,
    brown_h_min,
    brown_h_max,
    brown_s_min,
    brown_s_max,
    brown_v_min,
    brown_v_max,
    gray_s_max,
    gray_v_min,
    gray_v_max,
):
    if road_mask is None or (road_mask > 0).sum() == 0:
        return None, None, None, None

    h_img, w_img = image_bgr.shape[:2]
    total_pixels = h_img * w_img
    road_region = road_mask > 0

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    roi_hsv = hsv[road_region]
    roi_gray = gray[road_region]

    h_vals = roi_hsv[:, 0].astype(np.float32)
    s_vals = roi_hsv[:, 1].astype(np.float32) / 255.0
    v_vals = roi_hsv[:, 2].astype(np.float32) / 255.0

    brown_condition = (
        (h_vals >= brown_h_min)
        & (h_vals <= brown_h_max)
        & (s_vals >= brown_s_min)
        & (s_vals <= brown_s_max)
        & (v_vals >= brown_v_min)
        & (v_vals <= brown_v_max)
    )

    gray_dry_condition = (
        (s_vals <= gray_s_max)
        & (v_vals >= gray_v_min)
        & (v_vals <= gray_v_max)
    )

    brown_mask = np.zeros(road_mask.shape, dtype=np.uint8)
    gray_mask = np.zeros(road_mask.shape, dtype=np.uint8)

    road_indices = np.where(road_region)
    brown_mask[road_indices] = brown_condition.astype(np.uint8) * 255
    gray_mask[road_indices] = gray_dry_condition.astype(np.uint8) * 255

    edges = cv2.Canny(gray, 80, 160)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    road_area_pixels = int(road_region.sum())
    road_area_ratio = road_area_pixels / total_pixels

    road_mean_brightness = float(v_vals.mean())
    road_mean_saturation = float(s_vals.mean())
    road_contrast_std = float(roi_gray.std() / 255.0)

    road_brown_pixel_ratio = float(brown_condition.mean())
    road_gray_dry_pixel_ratio = float(gray_dry_condition.mean())

    road_edge_density = float((edges[road_region] > 0).mean())
    road_laplacian_std = float(lap[road_region].std() / 255.0)

    road_haze_flatness_proxy = float(
        road_mean_brightness * (1.0 - road_contrast_std)
    )

    texture_score = np.clip(
        0.5 * (road_edge_density / 0.06)
        + 0.5 * (road_laplacian_std / 0.08),
        0,
        1,
    )

    haze_score = np.clip(road_haze_flatness_proxy / 0.60, 0, 1)

    visual_dust_score = float(
        100.0
        * (
            0.45 * road_brown_pixel_ratio
            + 0.25 * road_gray_dry_pixel_ratio
            + 0.20 * texture_score
            + 0.10 * haze_score
        )
    )

    features = {
        "road_area_pixels": road_area_pixels,
        "road_area_ratio": road_area_ratio,
        "road_mean_brightness": road_mean_brightness,
        "road_mean_saturation": road_mean_saturation,
        "road_contrast_std": road_contrast_std,
        "road_brown_pixel_ratio": road_brown_pixel_ratio,
        "road_gray_dry_pixel_ratio": road_gray_dry_pixel_ratio,
        "road_edge_density": road_edge_density,
        "road_laplacian_std": road_laplacian_std,
        "road_haze_flatness_proxy": road_haze_flatness_proxy,
        "texture_score_0_1": float(texture_score),
        "haze_score_0_1": float(haze_score),
        "visual_dust_score_0_100": visual_dust_score,
    }

    return features, brown_mask, gray_mask, edges


def make_comparison_df(raw_features, normalized_features):
    rows = []

    for key in raw_features.keys():
        raw_val = raw_features.get(key, np.nan)
        norm_val = normalized_features.get(key, np.nan)

        rows.append({
            "feature": key,
            "raw": raw_val,
            "normalized": norm_val,
            "delta": norm_val - raw_val,
            "percent_change": (
                ((norm_val - raw_val) / raw_val) * 100
                if raw_val not in [0, None] and not pd.isna(raw_val)
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


st.title("Road Feature Normalization Validator")
st.caption(
    "Use this to check whether brown/gray/texture scores are stable or biased by lens brightness, saturation, and white balance."
)

with st.sidebar:
    st.header("Model")

    model_path = st.text_input(
        "SegFormer model path",
        value=DEFAULT_MODEL_PATH,
    )

    st.header("Road Mask Cleaning")

    black_threshold = st.slider(
        "Remove pixels with V below",
        min_value=0,
        max_value=80,
        value=25,
        step=1,
    )

    morph_kernel_size = st.select_slider(
        "Morphology kernel size",
        options=[0, 3, 5, 7, 9],
        value=5,
    )

    st.header("Normalization")

    norm_mode = st.selectbox(
        "Normalization mode",
        [
            "none",
            "gray_world",
            "hsv_reference",
            "gray_world_plus_hsv_reference",
        ],
        index=3,
    )

    reference_name = st.selectbox(
        "HSV reference",
        list(LENS_REFERENCE_STATS.keys()),
        index=2,
    )

    wb_strength = st.slider(
        "Gray-world strength",
        0.0,
        1.0,
        0.5,
        0.05,
    )

    hsv_strength = st.slider(
        "HSV reference strength",
        0.0,
        1.0,
        0.7,
        0.05,
    )

    st.header("Brown / Soil-like HSV Thresholds")

    brown_h_min = st.slider("Brown Hue min", 0, 179, 8)
    brown_h_max = st.slider("Brown Hue max", 0, 179, 35)
    brown_s_min = st.slider("Brown Saturation min", 0.0, 1.0, 0.10, 0.01)
    brown_s_max = st.slider("Brown Saturation max", 0.0, 1.0, 0.80, 0.01)
    brown_v_min = st.slider("Brown Brightness min", 0.0, 1.0, 0.18, 0.01)
    brown_v_max = st.slider("Brown Brightness max", 0.0, 1.0, 0.95, 0.01)

    st.header("Gray-Dry HSV Thresholds")

    gray_s_max = st.slider("Gray-dry Saturation max", 0.0, 1.0, 0.28, 0.01)
    gray_v_min = st.slider("Gray-dry Brightness min", 0.0, 1.0, 0.25, 0.01)
    gray_v_max = st.slider("Gray-dry Brightness max", 0.0, 1.0, 0.85, 0.01)


uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload an image to start validation.")
    st.stop()


try:
    processor, model, id2label, device = load_model(model_path)
except Exception as exc:
    st.error(f"Could not load model from {model_path}")
    st.exception(exc)
    st.stop()


pil_image = Image.open(uploaded_file)
image_bgr = pil_to_bgr(pil_image)

try:
    raw_road_mask = segment_road(
        image_bgr=image_bgr,
        processor=processor,
        model=model,
        id2label=id2label,
        device=device,
    )

    clean_mask = clean_road_mask(
        image_bgr=image_bgr,
        road_mask=raw_road_mask,
        black_threshold=black_threshold,
        morph_kernel_size=morph_kernel_size,
    )

except Exception as exc:
    st.error("Road segmentation failed.")
    st.exception(exc)
    st.stop()


normalized_bgr = normalize_for_feature_extraction(
    image_bgr=image_bgr,
    road_mask=clean_mask,
    mode=norm_mode,
    reference_name=reference_name,
    wb_strength=wb_strength,
    hsv_strength=hsv_strength,
)


raw_features, raw_brown_mask, raw_gray_mask, raw_edges = extract_road_features(
    image_bgr=image_bgr,
    road_mask=clean_mask,
    brown_h_min=brown_h_min,
    brown_h_max=brown_h_max,
    brown_s_min=brown_s_min,
    brown_s_max=brown_s_max,
    brown_v_min=brown_v_min,
    brown_v_max=brown_v_max,
    gray_s_max=gray_s_max,
    gray_v_min=gray_v_min,
    gray_v_max=gray_v_max,
)

norm_features, norm_brown_mask, norm_gray_mask, norm_edges = extract_road_features(
    image_bgr=normalized_bgr,
    road_mask=clean_mask,
    brown_h_min=brown_h_min,
    brown_h_max=brown_h_max,
    brown_s_min=brown_s_min,
    brown_s_max=brown_s_max,
    brown_v_min=brown_v_min,
    brown_v_max=brown_v_max,
    gray_s_max=gray_s_max,
    gray_v_min=gray_v_min,
    gray_v_max=gray_v_max,
)

if raw_features is None or norm_features is None:
    st.warning("No valid road pixels found after segmentation and cleaning.")
    st.image(bgr_to_rgb(image_bgr), caption="Uploaded Image", use_container_width=True)
    st.stop()


raw_quality = extract_lens_quality_stats(image_bgr, clean_mask)
norm_quality = extract_lens_quality_stats(normalized_bgr, clean_mask)

comparison_df = make_comparison_df(raw_features, norm_features)

st.subheader("Model / Processing Info")
st.write({
    "device": device,
    "model_num_labels": model.config.num_labels,
    "id2label": id2label,
    "normalization_mode": norm_mode,
    "reference": reference_name,
})


st.subheader("Key Score Comparison")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Visual Dust Score",
    f"{norm_features['visual_dust_score_0_100']:.2f}/100",
    delta=f"{norm_features['visual_dust_score_0_100'] - raw_features['visual_dust_score_0_100']:.2f}",
)

c2.metric(
    "Brown Score",
    f"{norm_features['road_brown_pixel_ratio'] * 100:.2f}%",
    delta=f"{(norm_features['road_brown_pixel_ratio'] - raw_features['road_brown_pixel_ratio']) * 100:.2f}%",
)

c3.metric(
    "Gray-Dry Score",
    f"{norm_features['road_gray_dry_pixel_ratio'] * 100:.2f}%",
    delta=f"{(norm_features['road_gray_dry_pixel_ratio'] - raw_features['road_gray_dry_pixel_ratio']) * 100:.2f}%",
)

c4.metric(
    "Road Area",
    f"{raw_features['road_area_ratio'] * 100:.2f}%",
)


st.subheader("Original, Road Mask, and Normalized Image")

road_overlay = make_overlay(image_bgr, clean_mask, color=(0, 255, 0), alpha=0.40)

col1, col2, col3 = st.columns(3)

with col1:
    st.image(
        bgr_to_rgb(image_bgr),
        caption="Original Image",
        use_container_width=True,
    )

with col2:
    st.image(
        bgr_to_rgb(road_overlay),
        caption="Road Mask Overlay",
        use_container_width=True,
    )

with col3:
    st.image(
        bgr_to_rgb(normalized_bgr),
        caption="Normalized for Feature Extraction",
        use_container_width=True,
    )


st.subheader("Raw vs Normalized Brown Pixel Detection")

raw_brown_overlay = make_overlay(image_bgr, raw_brown_mask, color=(0, 140, 255), alpha=0.55)
norm_brown_overlay = make_overlay(normalized_bgr, norm_brown_mask, color=(0, 140, 255), alpha=0.55)

col4, col5 = st.columns(2)

with col4:
    st.image(
        bgr_to_rgb(raw_brown_overlay),
        caption="Raw Brown / Soil-like Pixels",
        use_container_width=True,
    )

with col5:
    st.image(
        bgr_to_rgb(norm_brown_overlay),
        caption="Normalized Brown / Soil-like Pixels",
        use_container_width=True,
    )


st.subheader("Raw vs Normalized Gray-Dry Pixel Detection")

raw_gray_overlay = make_overlay(image_bgr, raw_gray_mask, color=(180, 180, 180), alpha=0.55)
norm_gray_overlay = make_overlay(normalized_bgr, norm_gray_mask, color=(180, 180, 180), alpha=0.55)

col6, col7 = st.columns(2)

with col6:
    st.image(
        bgr_to_rgb(raw_gray_overlay),
        caption="Raw Gray-Dry Pixels",
        use_container_width=True,
    )

with col7:
    st.image(
        bgr_to_rgb(norm_gray_overlay),
        caption="Normalized Gray-Dry Pixels",
        use_container_width=True,
    )


st.subheader("Raw vs Normalized Feature Table")

st.dataframe(
    comparison_df,
    use_container_width=True,
)

csv = comparison_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download raw-vs-normalized feature comparison CSV",
    data=csv,
    file_name="raw_vs_normalized_road_features.csv",
    mime="text/csv",
)


st.subheader("Lens / Image Quality Diagnostics")

quality_df = pd.DataFrame([
    {"version": "raw", **raw_quality},
    {"version": "normalized", **norm_quality},
])

st.dataframe(quality_df, use_container_width=True)


st.warning(
    "Use this dashboard for validation. Brown score and gray-dry score are visual road-surface proxies, "
    "not direct dust measurements. If normalization changes the score drastically while the image meaning "
    "does not change, the feature is lens/color sensitive and should be treated carefully."
)