from pathlib import Path
import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# Change this if your model folder is different.
DEFAULT_MODEL_PATH = (
    "models/road_segmentation/"
    "best_segformer_b0_idd_binary_road/"
    "best_segformer_b0_idd_binary_road"
)

FALLBACK_PROCESSOR = "nvidia/segformer-b0-finetuned-cityscapes-768-768"


st.set_page_config(
    page_title="Road Dust Visual Feature Validator",
    layout="wide",
)


def pil_to_bgr(pil_image: Image.Image):
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@st.cache_resource
def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Processor may not exist in your locally saved fine-tuned folder.
    # So first try local, then fall back to the original SegFormer processor.
    try:
        processor = SegformerImageProcessor.from_pretrained(model_path)
    except Exception:
        processor = SegformerImageProcessor.from_pretrained(FALLBACK_PROCESSOR)

    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()

    id2label = model.config.id2label

    return processor, model, id2label, device


def get_road_class_ids(id2label, num_labels):
    road_ids = []

    for class_id, label in id2label.items():
        if str(label).lower() == "road":
            road_ids.append(int(class_id))

    # For our fine-tuned binary IDD model:
    # 0 = background, 1 = road
    if not road_ids and num_labels == 2:
        road_ids = [1]

    if not road_ids:
        raise ValueError(f"Could not identify road class from labels: {id2label}")

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

    return road_mask, pred


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


def make_overlay(image_bgr, mask, color=(0, 255, 0), alpha=0.40):
    overlay = image_bgr.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return blended


def extract_road_dust_features(
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
        return None, None, None

    h_img, w_img = image_bgr.shape[:2]
    total_pixels = h_img * w_img

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    road_region = road_mask > 0

    roi_hsv = hsv[road_region]
    roi_gray = gray[road_region]

    h_vals = roi_hsv[:, 0].astype(np.float32)
    s_vals = roi_hsv[:, 1].astype(np.float32) / 255.0
    v_vals = roi_hsv[:, 2].astype(np.float32) / 255.0

    road_area_pixels = int(road_region.sum())
    road_area_ratio = road_area_pixels / total_pixels

    road_mean_brightness = float(v_vals.mean())
    road_mean_saturation = float(s_vals.mean())
    road_contrast_std = float(roi_gray.std() / 255.0)

    # Brown / soil-like road pixels
    brown_condition = (
        (h_vals >= brown_h_min)
        & (h_vals <= brown_h_max)
        & (s_vals >= brown_s_min)
        & (s_vals <= brown_s_max)
        & (v_vals >= brown_v_min)
        & (v_vals <= brown_v_max)
    )

    # Gray-dry road pixels
    gray_dry_condition = (
        (s_vals <= gray_s_max)
        & (v_vals >= gray_v_min)
        & (v_vals <= gray_v_max)
    )

    road_brown_pixel_ratio = float(brown_condition.mean())
    road_gray_dry_pixel_ratio = float(gray_dry_condition.mean())

    # Create full-size masks for visualization
    brown_mask = np.zeros(road_mask.shape, dtype=np.uint8)
    gray_dry_mask = np.zeros(road_mask.shape, dtype=np.uint8)

    road_indices = np.where(road_region)
    brown_mask[road_indices] = brown_condition.astype(np.uint8) * 255
    gray_dry_mask[road_indices] = gray_dry_condition.astype(np.uint8) * 255

    # Edge density
    edges = cv2.Canny(gray, 80, 160)
    road_edge_density = float((edges[road_region] > 0).mean())

    # Laplacian texture
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    road_laplacian_std = float(lap[road_region].std() / 255.0)

    # Haze / flatness proxy
    road_haze_flatness_proxy = float(
        road_mean_brightness * (1.0 - road_contrast_std)
    )

    # Experimental visual dust score.
    # This is NOT PM concentration.
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

    return features, brown_mask, gray_dry_mask


def dataframe_download_button(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download feature CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


st.title("Road Dust Visual Feature Validator")
st.caption(
    "Upload a road image and inspect road segmentation, brown-road score, "
    "gray-dry score, texture features, and experimental visual dust score."
)

with st.sidebar:
    st.header("Model Settings")

    model_path = st.text_input(
        "SegFormer model path",
        value=DEFAULT_MODEL_PATH,
    )

    st.header("Mask Cleaning")
    black_threshold = st.slider(
        "Remove near-black pixels below V threshold",
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
    st.info("Upload an image to begin.")
    st.stop()

try:
    processor, model, id2label, device = load_model(model_path)
except Exception as exc:
    st.error(f"Could not load model from: {model_path}")
    st.exception(exc)
    st.stop()

pil_image = Image.open(uploaded_file)
image_bgr = pil_to_bgr(pil_image)

st.subheader("Model Info")
st.write(
    {
        "device": device,
        "num_labels": model.config.num_labels,
        "id2label": id2label,
    }
)

try:
    raw_road_mask, pred = segment_road(
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
    st.error("Segmentation failed.")
    st.exception(exc)
    st.stop()

features, brown_mask, gray_dry_mask = extract_road_dust_features(
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

if features is None:
    st.warning("No valid road pixels found after segmentation and cleaning.")
    st.image(bgr_to_rgb(image_bgr), caption="Uploaded image", use_container_width=True)
    st.stop()

road_overlay = make_overlay(image_bgr, clean_mask, color=(0, 255, 0), alpha=0.40)
brown_overlay = make_overlay(image_bgr, brown_mask, color=(0, 140, 255), alpha=0.55)
gray_overlay = make_overlay(image_bgr, gray_dry_mask, color=(180, 180, 180), alpha=0.55)

st.subheader("Key Scores")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Visual Dust Score",
    f"{features['visual_dust_score_0_100']:.2f}/100",
)

c2.metric(
    "Brown Road Score",
    f"{features['road_brown_pixel_ratio'] * 100:.2f}%",
)

c3.metric(
    "Gray-Dry Score",
    f"{features['road_gray_dry_pixel_ratio'] * 100:.2f}%",
)

c4.metric(
    "Road Area",
    f"{features['road_area_ratio'] * 100:.2f}%",
)

st.subheader("Road Segmentation and Dust Feature Masks")

col1, col2 = st.columns(2)

with col1:
    st.image(
        bgr_to_rgb(image_bgr),
        caption="Original Uploaded Image",
        use_container_width=True,
    )

with col2:
    st.image(
        bgr_to_rgb(road_overlay),
        caption="Cleaned Road Mask Overlay",
        use_container_width=True,
    )

col3, col4 = st.columns(2)

with col3:
    st.image(
        bgr_to_rgb(brown_overlay),
        caption="Brown / Soil-like Road Pixels",
        use_container_width=True,
    )

with col4:
    st.image(
        bgr_to_rgb(gray_overlay),
        caption="Gray-Dry Road Pixels",
        use_container_width=True,
    )

st.subheader("Detailed Feature Table")

feature_df = pd.DataFrame([features])
st.dataframe(feature_df.T.rename(columns={0: "value"}), use_container_width=True)

dataframe_download_button(
    feature_df,
    filename="road_dust_visual_features.csv",
)

st.warning(
    "Scientific caution: the visual dust score is a heuristic road-surface "
    "appearance score. It is not PM2.5 or PM10 concentration. Validate it using "
    "ground truth labels or PM sensor correlation before making claims."
)