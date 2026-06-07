from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

DEFAULT_ROAD_AREA_CSV = "outputs/road_area_lens1_batch/lens1_final_road_area_features_v1.csv"
DEFAULT_VEHICLE_DETAIL_CSV = "outputs/road_area_lens1_batch/lens1_bbox_nms_conservative_vehicle_occlusion_details.csv"
DEFAULT_OUTPUT_LABEL_CSV = "outputs/road_area_lens1_batch/manual_road_area_validation_labels_v1.csv"


# ============================================================
# HELPERS
# ============================================================

def load_image(path):
    path = Path(str(path))
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def load_mask_as_rgb(path):
    path = Path(str(path))
    if not path.exists():
        return None

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[mask > 0] = [255, 255, 255]

    return Image.fromarray(rgb)


def draw_vehicle_boxes(image_path, vehicle_df):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    img = img_bgr.copy()

    for _, r in vehicle_df.iterrows():
        x1 = int(r["bbox_x1"])
        y1 = int(r["bbox_y1"])
        x2 = int(r["bbox_x2"])
        y2 = int(r["bbox_y2"])

        cls = str(r["class_name"])
        hidden = float(r.get("hidden_road_area_m2_est_conservative", 0.0))
        contact = float(r.get("road_contact_confidence", 0.0))
        vis = float(r.get("visibility_factor", 0.0))

        # Green = contributed hidden road area
        # Red = rejected / no contribution
        color = (0, 255, 0) if hidden > 0 else (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} | hidden={hidden:.1f} | rc={contact:.2f} | vis={vis:.2f}"

        y_text = max(20, y1 - 8)
        cv2.putText(
            img,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def initialize_labels(df):
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "sample_index": r.get("sample_index"),
            "matched_run_id": r.get("matched_run_id", ""),
            "lens_id": r.get("lens_id", 1),
            "manual_visible_road_area_quality": "",
            "manual_vehicle_occlusion_quality": "",
            "manual_final_area_decision": "",
            "manual_notes": "",
        })
    return pd.DataFrame(rows)


def get_label_row(labels_df, sample_index, matched_run_id, lens_id):
    mask = (
        (labels_df["sample_index"].astype(str) == str(sample_index))
        & (labels_df["matched_run_id"].astype(str) == str(matched_run_id))
        & (labels_df["lens_id"].astype(str) == str(lens_id))
    )

    if mask.any():
        return labels_df[mask].iloc[0].to_dict()

    return {
        "sample_index": sample_index,
        "matched_run_id": matched_run_id,
        "lens_id": lens_id,
        "manual_visible_road_area_quality": "",
        "manual_vehicle_occlusion_quality": "",
        "manual_final_area_decision": "",
        "manual_notes": "",
    }


def update_label_row(labels_df, updated):
    sample_index = updated["sample_index"]
    matched_run_id = updated["matched_run_id"]
    lens_id = updated["lens_id"]

    mask = (
        (labels_df["sample_index"].astype(str) == str(sample_index))
        & (labels_df["matched_run_id"].astype(str) == str(matched_run_id))
        & (labels_df["lens_id"].astype(str) == str(lens_id))
    )

    if mask.any():
        idx = labels_df[mask].index[0]
        for k, v in updated.items():
            labels_df.loc[idx, k] = v
    else:
        labels_df = pd.concat([labels_df, pd.DataFrame([updated])], ignore_index=True)

    return labels_df


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Road Area Validation Dashboard",
    layout="wide",
)

st.title("Road Area Validation Dashboard")
st.caption("Manual validation for visible road area + vehicle-footprint occlusion-adjusted road area")

with st.sidebar:
    st.header("Input files")

    road_area_csv = st.text_input(
        "Final road-area CSV",
        DEFAULT_ROAD_AREA_CSV,
    )

    vehicle_detail_csv = st.text_input(
        "Vehicle detail CSV",
        DEFAULT_VEHICLE_DETAIL_CSV,
    )

    output_label_csv = st.text_input(
        "Manual label output CSV",
        DEFAULT_OUTPUT_LABEL_CSV,
    )

    st.divider()

    st.header("Filters")

    status_filter = st.multiselect(
        "Final road-area status",
        [
            "use_primary",
            "use_sensitivity_only",
            "exclude_or_manual_review",
        ],
        default=[
            "use_primary",
            "use_sensitivity_only",
            "exclude_or_manual_review",
        ],
    )

    sort_by = st.selectbox(
        "Sort by",
        [
            "sample_index",
            "road_area_vehicle_occlusion_fraction_v1",
            "road_area_m2_occlusion_adjusted_conservative_v1",
            "road_area_m2_depth_est_visible_v1",
            "road_area_occlusion_added_ratio_vs_visible",
        ],
        index=1,
    )

    sort_desc = st.checkbox("Sort descending", value=True)

    show_only_problematic = st.checkbox(
        "Show only sensitivity/exclude frames",
        value=False,
    )


# Load data
road_path = Path(road_area_csv)
veh_path = Path(vehicle_detail_csv)
label_path = Path(output_label_csv)

if not road_path.exists():
    st.error(f"Road-area CSV not found: {road_path}")
    st.stop()

if not veh_path.exists():
    st.error(f"Vehicle detail CSV not found: {veh_path}")
    st.stop()

df = pd.read_csv(road_path)
veh_df = pd.read_csv(veh_path)

if "final_road_area_feature_status" not in df.columns:
    st.error("final_road_area_feature_status column missing.")
    st.stop()

filtered = df[df["final_road_area_feature_status"].isin(status_filter)].copy()

if show_only_problematic:
    filtered = filtered[
        filtered["final_road_area_feature_status"].isin(
            ["use_sensitivity_only", "exclude_or_manual_review"]
        )
    ].copy()

if sort_by in filtered.columns:
    filtered = filtered.sort_values(sort_by, ascending=not sort_desc)

filtered = filtered.reset_index(drop=True)

if len(filtered) == 0:
    st.warning("No samples after filtering.")
    st.stop()

# Load or initialize label file
if label_path.exists():
    labels_df = pd.read_csv(label_path)
else:
    labels_df = initialize_labels(df)

# Sidebar sample selector
with st.sidebar:
    st.divider()
    st.header("Sample navigation")

    sample_options = [
        f"{i}: sample={int(r.sample_index)} | status={r.final_road_area_feature_status}"
        for i, r in filtered.iterrows()
    ]

    selected_option = st.selectbox("Select sample", sample_options)
    selected_i = int(selected_option.split(":")[0])

row = filtered.iloc[selected_i]

sample_index = row["sample_index"]
matched_run_id = row.get("matched_run_id", "")
lens_id = row.get("lens_id", 1)

sample_vehicles = veh_df[
    (veh_df["sample_index"].astype(str) == str(sample_index))
    & (veh_df["lens_id"].astype(str) == str(lens_id))
].copy()

# ============================================================
# TOP METRICS
# ============================================================

st.subheader(f"Sample {sample_index} | Lens {lens_id}")

metric_cols = st.columns(5)

visible_area = safe_float(row.get("road_area_m2_depth_est_visible_v1"))
hidden_area = safe_float(row.get("road_area_m2_vehicle_occluded_conservative_v1"))
final_area = safe_float(row.get("road_area_m2_occlusion_adjusted_conservative_v1"))
occ_frac = safe_float(row.get("road_area_vehicle_occlusion_fraction_v1"))
status = row.get("final_road_area_feature_status", "")

metric_cols[0].metric("Visible road area", f"{visible_area:.2f} m²" if visible_area is not None else "NA")
metric_cols[1].metric("Vehicle-hidden area", f"{hidden_area:.2f} m²" if hidden_area is not None else "NA")
metric_cols[2].metric("Final road area", f"{final_area:.2f} m²" if final_area is not None else "NA")
metric_cols[3].metric("Vehicle occlusion fraction", f"{occ_frac:.3f}" if occ_frac is not None else "NA")
metric_cols[4].metric("Status", str(status))

st.write(
    {
        "road_mask_area_quality": row.get("road_mask_area_quality", ""),
        "occlusion_adjustment_quality": row.get("occlusion_adjustment_quality_conservative", ""),
        "feature_version": row.get("road_area_feature_version", ""),
        "vehicle_count_detected": row.get("vehicle_count_detected_for_occlusion", ""),
        "vehicle_count_used": row.get("vehicle_count_used_for_occlusion_conservative", row.get("vehicle_count_used_for_occlusion", "")),
    }
)

# ============================================================
# IMAGES
# ============================================================

image_path = Path(str(row.get("processed_frame_path", "")))
mask_path = Path(str(row.get("saved_road_mask_path", "")))

original_img = load_image(image_path)
mask_img = load_mask_as_rgb(mask_path)
box_img = draw_vehicle_boxes(image_path, sample_vehicles)

image_cols = st.columns(3)

with image_cols[0]:
    st.markdown("### Original Image")
    if original_img is not None:
        st.image(original_img, use_container_width=True)
    else:
        st.warning(f"Could not load original image: {image_path}")

with image_cols[1]:
    st.markdown("### Road Mask")
    if mask_img is not None:
        st.image(mask_img, use_container_width=True)
    else:
        st.warning(f"Could not load road mask: {mask_path}")

with image_cols[2]:
    st.markdown("### Vehicle Boxes")
    st.caption("Green = contributes hidden road area; Red = rejected/no contribution")
    if box_img is not None:
        st.image(box_img, use_container_width=True)
    else:
        st.warning("Could not draw vehicle boxes.")

# ============================================================
# VEHICLE TABLE
# ============================================================

st.subheader("Vehicle-level occlusion details")

vehicle_display_cols = [
    "class_name",
    "detection_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "standard_footprint_m2",
    "visibility_factor",
    "road_contact_confidence",
    "hidden_road_area_m2_est_full_footprint",
    "conservative_factor",
    "hidden_road_area_m2_est_conservative",
    "road_contact_reason",
    "visibility_reason",
]

vehicle_display_cols = [c for c in vehicle_display_cols if c in sample_vehicles.columns]

if len(sample_vehicles) > 0:
    st.dataframe(sample_vehicles[vehicle_display_cols], use_container_width=True)
else:
    st.info("No vehicle details for this sample.")

# ============================================================
# MANUAL LABEL FORM
# ============================================================

st.subheader("Manual validation labels")

existing_label = get_label_row(labels_df, sample_index, matched_run_id, lens_id)

with st.form("manual_validation_form"):
    col1, col2, col3 = st.columns(3)

    visible_quality = col1.selectbox(
        "Manual visible road/mask quality",
        ["", "good", "medium", "bad", "uncertain"],
        index=["", "good", "medium", "bad", "uncertain"].index(
            existing_label.get("manual_visible_road_area_quality", "")
            if existing_label.get("manual_visible_road_area_quality", "") in ["", "good", "medium", "bad", "uncertain"]
            else ""
        ),
    )

    occlusion_quality = col2.selectbox(
        "Manual vehicle occlusion quality",
        ["", "good", "medium", "bad", "overestimated", "underestimated", "uncertain"],
        index=["", "good", "medium", "bad", "overestimated", "underestimated", "uncertain"].index(
            existing_label.get("manual_vehicle_occlusion_quality", "")
            if existing_label.get("manual_vehicle_occlusion_quality", "") in ["", "good", "medium", "bad", "overestimated", "underestimated", "uncertain"]
            else ""
        ),
    )

    final_decision = col3.selectbox(
        "Manual final decision",
        ["", "accept_primary", "accept_sensitivity", "reject", "needs_review"],
        index=["", "accept_primary", "accept_sensitivity", "reject", "needs_review"].index(
            existing_label.get("manual_final_area_decision", "")
            if existing_label.get("manual_final_area_decision", "") in ["", "accept_primary", "accept_sensitivity", "reject", "needs_review"]
            else ""
        ),
    )

    notes = st.text_area(
        "Manual notes",
        value=str(existing_label.get("manual_notes", "")) if pd.notna(existing_label.get("manual_notes", "")) else "",
        height=100,
    )

    submitted = st.form_submit_button("Save manual label")

    if submitted:
        updated = {
            "sample_index": sample_index,
            "matched_run_id": matched_run_id,
            "lens_id": lens_id,
            "manual_visible_road_area_quality": visible_quality,
            "manual_vehicle_occlusion_quality": occlusion_quality,
            "manual_final_area_decision": final_decision,
            "manual_notes": notes,
        }

        labels_df = update_label_row(labels_df, updated)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(label_path, index=False)

        st.success(f"Saved manual label to {label_path}")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "This dashboard validates the v1 road-area feature: visible depth-estimated road area "
    "+ conservative vehicle-footprint hidden-road correction. It is not ground-truth road area."
)