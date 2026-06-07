from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import cv2


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_V1_ROAD_AREA_CSV = "outputs/road_area_lens1_batch/lens1_final_road_area_features_v1.csv"
DEFAULT_V1_VEHICLE_DETAIL_CSV = "outputs/road_area_lens1_batch/lens1_bbox_nms_conservative_vehicle_occlusion_details.csv"

DEFAULT_V2_ROAD_AREA_CSV = "outputs/road_area_lens1_batch/lens1_bbox_visibility_only_occlusion_adjusted_road_area_v2.csv"
DEFAULT_V2_VEHICLE_DETAIL_CSV = "outputs/road_area_lens1_batch/lens1_bbox_visibility_only_vehicle_occlusion_details_v2.csv"

DEFAULT_V3_ROAD_AREA_CSV = "outputs/road_area_lens1_batch/lens1_depth_gated_occlusion_adjusted_road_area_v3.csv"
DEFAULT_V3_VEHICLE_DETAIL_CSV = "outputs/road_area_lens1_batch/lens1_depth_gated_vehicle_occlusion_details_v3.csv"

DEFAULT_LABEL_OUTPUT_CSV = "outputs/road_area_lens1_batch/manual_full_validation_labels_v1_v2_v3.csv"


# ============================================================
# LABEL COLUMNS
# ============================================================

LABEL_COLUMNS = [
    "sample_index",
    "matched_run_id",
    "lens_id",
    "validation_method",

    "manual_vehicle_detection_quality",
    "manual_road_mask_quality",
    "manual_depth_area_quality",
    "manual_vehicle_occlusion_quality",
    "manual_final_decision",

    "manual_vehicle_false_positive_count",
    "manual_vehicle_false_negative_count",
    "manual_vehicle_wrong_class_count",
    "manual_vehicle_duplicate_count",

    "manual_failure_road_mask_undersegmentation",
    "manual_failure_road_mask_oversegmentation",
    "manual_failure_vehicle_false_negative",
    "manual_failure_vehicle_false_positive",
    "manual_failure_vehicle_wrong_class",
    "manual_failure_duplicate_boxes",
    "manual_failure_occlusion_overestimated",
    "manual_failure_occlusion_underestimated",
    "manual_failure_depth_suspicious",
    "manual_failure_glare_shadow_uncertain",
    "manual_failure_heavy_occlusion",

    "manual_notes",
]


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_rgb_image(path):
    path = Path(str(path))
    if not path.exists():
        return None

    img = cv2.imread(str(path))
    if img is None:
        return None

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask_rgb(path):
    path = Path(str(path))
    if not path.exists():
        return None

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[mask > 0] = [255, 255, 255]
    return rgb


def overlay_mask_on_image(image_rgb, mask_rgb, alpha=0.45):
    if image_rgb is None or mask_rgb is None:
        return None

    if image_rgb.shape[:2] != mask_rgb.shape[:2]:
        mask_rgb = cv2.resize(
            mask_rgb,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = image_rgb.copy()
    road = mask_rgb[:, :, 0] > 0
    overlay[road] = [0, 255, 255]

    return cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)


def get_vehicle_hidden_area(row, method):
    if method == "v1":
        return float(row.get("hidden_road_area_m2_est_conservative", 0.0))
    if method == "v2":
        return float(row.get("hidden_road_area_m2_conservative_v2", 0.0))
    if method == "v3":
        return float(row.get("hidden_road_area_m2_conservative_v3", 0.0))
    return 0.0


def get_vehicle_visibility(row, method):
    if method == "v1":
        return row.get("visibility_factor", np.nan)
    if method == "v2":
        return row.get("bbox_visibility_factor_v2", np.nan)
    if method == "v3":
        return row.get("bbox_visibility_factor_v3", np.nan)
    return np.nan


def draw_vehicle_boxes(image_rgb, vehicle_df, method):
    if image_rgb is None:
        return None

    img = image_rgb.copy()

    for _, r in vehicle_df.iterrows():
        x1 = int(round(r.get("bbox_x1", 0)))
        y1 = int(round(r.get("bbox_y1", 0)))
        x2 = int(round(r.get("bbox_x2", 0)))
        y2 = int(round(r.get("bbox_y2", 0)))

        cls = str(r.get("class_name", "vehicle"))
        conf = r.get("detection_confidence", np.nan)

        try:
            conf = float(conf)
        except Exception:
            conf = np.nan

        hidden = get_vehicle_hidden_area(r, method)
        vis = get_vehicle_visibility(r, method)
        contact = r.get("road_contact_confidence", np.nan)

        color = (0, 255, 0) if hidden > 0 else (255, 0, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        conf_txt = f"{conf:.2f}" if np.isfinite(conf) else "NA"
        vis_txt = f"{float(vis):.2f}" if pd.notna(vis) else "NA"

        if method == "v1":
            rc_txt = f"{float(contact):.2f}" if pd.notna(contact) else "NA"
            label = f"{cls} | c={conf_txt} | h={hidden:.1f} | rc={rc_txt} | vis={vis_txt}"

        elif method == "v2":
            label = f"{cls} | c={conf_txt} | h={hidden:.1f} | vis={vis_txt}"

        elif method == "v3":
            depth = r.get("vehicle_depth_m_v3", np.nan)
            depth_factor = r.get("vehicle_depth_factor_v3", np.nan)

            depth_txt = f"{float(depth):.1f}m" if pd.notna(depth) else "NA"
            df_txt = f"{float(depth_factor):.2f}" if pd.notna(depth_factor) else "NA"

            label = f"{cls} | c={conf_txt} | h={hidden:.1f} | d={depth_txt} | df={df_txt}"

        else:
            label = f"{cls} | c={conf_txt} | h={hidden:.1f}"

        y_text = max(25, y1 - 8)
        cv2.putText(
            img,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

    return img


def make_side_by_side(images, captions):
    cols = st.columns(len(images))
    for col, img, cap in zip(cols, images, captions):
        with col:
            st.markdown(f"### {cap}")
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                st.warning("Image not available")


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_road_df(df, method):
    out = df.copy()

    if method == "v1":
        out["visible_area_m2_display"] = out.get(
            "road_area_m2_depth_est_visible_v1",
            out.get("visible_road_area_m2_depth_est"),
        )
        out["vehicle_hidden_area_m2_display"] = out.get(
            "road_area_m2_vehicle_occluded_conservative_v1",
            out.get("vehicle_occluded_road_area_m2_est_conservative"),
        )
        out["final_area_m2_display"] = out.get(
            "road_area_m2_occlusion_adjusted_conservative_v1",
            out.get("occlusion_adjusted_road_area_m2_est_conservative"),
        )
        out["occlusion_fraction_display"] = out.get(
            "road_area_vehicle_occlusion_fraction_v1",
            out.get("vehicle_occlusion_fraction_conservative"),
        )
        out["auto_status_display"] = out.get("final_road_area_feature_status")
        out["occlusion_quality_display"] = out.get("occlusion_adjustment_quality_conservative")
        out["method_version_display"] = out.get("road_area_feature_version", "v1_road_contact")
        out["validation_method"] = "v1"

    elif method == "v2":
        out["visible_area_m2_display"] = out.get("visible_road_area_m2_depth_est")
        out["vehicle_hidden_area_m2_display"] = out.get("vehicle_occluded_road_area_m2_conservative_v2")
        out["final_area_m2_display"] = out.get("road_area_m2_occlusion_adjusted_conservative_v2")
        out["occlusion_fraction_display"] = out.get("road_area_vehicle_occlusion_fraction_v2")
        out["auto_status_display"] = out.get("final_road_area_feature_status_v2")
        out["occlusion_quality_display"] = out.get("vehicle_occlusion_quality_v2")
        out["method_version_display"] = out.get("road_area_feature_version_v2", "v2_bbox_visibility_only")
        out["validation_method"] = "v2"

    elif method == "v3":
        out["visible_area_m2_display"] = out.get("visible_road_area_m2_depth_est")
        out["vehicle_hidden_area_m2_display"] = out.get("vehicle_occluded_road_area_m2_conservative_v3")
        out["final_area_m2_display"] = out.get("road_area_m2_occlusion_adjusted_conservative_v3")
        out["occlusion_fraction_display"] = out.get("road_area_vehicle_occlusion_fraction_v3")
        out["auto_status_display"] = out.get("final_road_area_feature_status_v3")
        out["occlusion_quality_display"] = out.get("vehicle_occlusion_quality_v3")
        out["method_version_display"] = out.get("road_area_feature_version_v3", "v3_depth_gated")
        out["validation_method"] = "v3"

    else:
        raise ValueError(f"Unknown method: {method}")

    return out


def fmt_m2(x):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.2f} m²"
    except Exception:
        return "NA"


def fmt_num(x, nd=3):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


# ============================================================
# LABEL HELPERS
# ============================================================

def ensure_label_df(path, base_df):
    path = Path(path)

    if path.exists():
        labels = pd.read_csv(path)
        for c in LABEL_COLUMNS:
            if c not in labels.columns:
                labels[c] = ""
        return labels[LABEL_COLUMNS].copy()

    rows = []
    for _, r in base_df.iterrows():
        rows.append({
            "sample_index": r.get("sample_index", ""),
            "matched_run_id": r.get("matched_run_id", ""),
            "lens_id": r.get("lens_id", 1),
            "validation_method": r.get("validation_method", ""),

            "manual_vehicle_detection_quality": "",
            "manual_road_mask_quality": "",
            "manual_depth_area_quality": "",
            "manual_vehicle_occlusion_quality": "",
            "manual_final_decision": "",

            "manual_vehicle_false_positive_count": 0,
            "manual_vehicle_false_negative_count": 0,
            "manual_vehicle_wrong_class_count": 0,
            "manual_vehicle_duplicate_count": 0,

            "manual_failure_road_mask_undersegmentation": False,
            "manual_failure_road_mask_oversegmentation": False,
            "manual_failure_vehicle_false_negative": False,
            "manual_failure_vehicle_false_positive": False,
            "manual_failure_vehicle_wrong_class": False,
            "manual_failure_duplicate_boxes": False,
            "manual_failure_occlusion_overestimated": False,
            "manual_failure_occlusion_underestimated": False,
            "manual_failure_depth_suspicious": False,
            "manual_failure_glare_shadow_uncertain": False,
            "manual_failure_heavy_occlusion": False,

            "manual_notes": "",
        })

    return pd.DataFrame(rows)


def get_existing_label(labels_df, sample_index, matched_run_id, lens_id, method):
    mask = (
        (labels_df["sample_index"].astype(str) == str(sample_index))
        & (labels_df["matched_run_id"].astype(str) == str(matched_run_id))
        & (labels_df["lens_id"].astype(str) == str(lens_id))
        & (labels_df["validation_method"].astype(str) == str(method))
    )

    if mask.any():
        return labels_df[mask].iloc[0].to_dict()

    return {c: "" for c in LABEL_COLUMNS}


def update_label(labels_df, updated):
    mask = (
        (labels_df["sample_index"].astype(str) == str(updated["sample_index"]))
        & (labels_df["matched_run_id"].astype(str) == str(updated["matched_run_id"]))
        & (labels_df["lens_id"].astype(str) == str(updated["lens_id"]))
        & (labels_df["validation_method"].astype(str) == str(updated["validation_method"]))
    )

    if mask.any():
        idx = labels_df[mask].index[0]
        for k, v in updated.items():
            labels_df.loc[idx, k] = v
    else:
        labels_df = pd.concat([labels_df, pd.DataFrame([updated])], ignore_index=True)

    return labels_df


def safe_select_index(options, value):
    if value in options:
        return options.index(value)
    return 0


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Road Area Validation Dashboard",
    layout="wide",
)

st.title("Road Area + Vehicle Occlusion Validation Dashboard")
st.caption(
    "Manual validation for road-area estimation. "
    "v1 = road-contact method, v2 = bbox-visibility-only, v3 = vehicle-depth-gated."
)

with st.sidebar:
    st.header("Validation method")

    method_mode = st.radio(
        "Choose method",
        ["v1", "v2", "v3"],
        index=2,
        help=(
            "v1 uses road-contact confidence. "
            "v2 assumes valid detected vehicles are road vehicles and uses bbox visibility only. "
            "v3 uses bbox visibility plus vehicle-depth cutoff."
        ),
    )

    st.divider()
    st.header("Files")

    if method_mode == "v1":
        road_area_csv = st.text_input("Road-area feature CSV", DEFAULT_V1_ROAD_AREA_CSV)
        vehicle_detail_csv = st.text_input("Vehicle detail CSV", DEFAULT_V1_VEHICLE_DETAIL_CSV)
    elif method_mode == "v2":
        road_area_csv = st.text_input("Road-area feature CSV", DEFAULT_V2_ROAD_AREA_CSV)
        vehicle_detail_csv = st.text_input("Vehicle detail CSV", DEFAULT_V2_VEHICLE_DETAIL_CSV)
    else:
        road_area_csv = st.text_input("Road-area feature CSV", DEFAULT_V3_ROAD_AREA_CSV)
        vehicle_detail_csv = st.text_input("Vehicle detail CSV", DEFAULT_V3_VEHICLE_DETAIL_CSV)

    label_output_csv = st.text_input("Manual label output CSV", DEFAULT_LABEL_OUTPUT_CSV)

    st.divider()
    st.header("Filters")

    status_options = [
        "use_primary",
        "use_sensitivity_only",
        "exclude_or_manual_review",
        "manual_review",
    ]

    selected_statuses = st.multiselect(
        "Automatic status",
        status_options,
        default=["use_primary", "use_sensitivity_only", "exclude_or_manual_review"],
    )

    validation_focus = st.selectbox(
        "Validation focus",
        [
            "problematic_first",
            "exclude_only",
            "sensitivity_only",
            "primary_random_review",
            "all",
        ],
        index=0,
    )

    sort_by = st.selectbox(
        "Sort by",
        [
            "occlusion_fraction_display",
            "final_area_m2_display",
            "visible_area_m2_display",
            "vehicle_hidden_area_m2_display",
            "sample_index",
        ],
        index=0,
    )

    sort_desc = st.checkbox("Sort descending", value=True)

    st.divider()
    st.header("Display")

    show_overlay = st.checkbox("Show road-mask overlay", value=True)
    show_vehicle_table = st.checkbox("Show vehicle table", value=True)
    show_raw_paths = st.checkbox("Show file paths", value=False)


# ============================================================
# LOAD DATA
# ============================================================

road_path = Path(road_area_csv)
veh_path = Path(vehicle_detail_csv)
label_path = Path(label_output_csv)

if not road_path.exists():
    st.error(f"Road-area CSV not found: {road_path}")
    st.stop()

if not veh_path.exists():
    st.error(f"Vehicle detail CSV not found: {veh_path}")
    st.stop()

raw_df = pd.read_csv(road_path)
df = normalize_road_df(raw_df, method_mode)

veh_df = pd.read_csv(veh_path)

if "lens_id" not in df.columns:
    df["lens_id"] = 1

if "lens_id" not in veh_df.columns:
    veh_df["lens_id"] = 1

labels_df = ensure_label_df(label_path, df)


# ============================================================
# FILTER DATA
# ============================================================

work = df.copy()

if "auto_status_display" in work.columns:
    work = work[work["auto_status_display"].isin(selected_statuses)].copy()

if validation_focus == "exclude_only":
    work = work[work["auto_status_display"] == "exclude_or_manual_review"].copy()

elif validation_focus == "sensitivity_only":
    work = work[work["auto_status_display"] == "use_sensitivity_only"].copy()

elif validation_focus == "problematic_first":
    work["_focus_rank"] = work["auto_status_display"].map({
        "exclude_or_manual_review": 0,
        "manual_review": 1,
        "use_sensitivity_only": 2,
        "use_primary": 3,
    }).fillna(4)

    work = work.sort_values(
        ["_focus_rank", "occlusion_fraction_display"],
        ascending=[True, False],
    ).copy()

elif validation_focus == "primary_random_review":
    work = work[work["auto_status_display"] == "use_primary"].copy()
    work = work.sample(frac=1.0, random_state=42).copy()

if validation_focus not in ["problematic_first", "primary_random_review"]:
    if sort_by in work.columns:
        work = work.sort_values(sort_by, ascending=not sort_desc).copy()

work = work.reset_index(drop=True)

if len(work) == 0:
    st.warning("No samples match filters.")
    st.stop()


# ============================================================
# SUMMARY
# ============================================================

st.subheader("Dataset summary")

summary_cols = st.columns(6)
summary_cols[0].metric("Method", method_mode)
summary_cols[1].metric("Samples shown", len(work))
summary_cols[2].metric("Total samples", len(df))

primary_count = int((df["auto_status_display"] == "use_primary").sum())
sensitivity_count = int((df["auto_status_display"] == "use_sensitivity_only").sum())
exclude_count = int((df["auto_status_display"] == "exclude_or_manual_review").sum())

summary_cols[3].metric("Primary", primary_count)
summary_cols[4].metric("Sensitivity", sensitivity_count)
summary_cols[5].metric("Exclude/review", exclude_count)

with st.expander("Show automatic status counts"):
    st.write(df["auto_status_display"].value_counts(dropna=False))


# ============================================================
# SAMPLE SELECTOR
# ============================================================

st.sidebar.divider()
st.sidebar.header("Sample navigation")

sample_options = []
for i, r in work.iterrows():
    sample_options.append(
        f"{i}: sample={int(r['sample_index'])} | "
        f"status={r.get('auto_status_display', '')} | "
        f"occ={fmt_num(r.get('occlusion_fraction_display'))}"
    )

selected = st.sidebar.selectbox("Choose sample", sample_options)
selected_i = int(selected.split(":")[0])

row = work.iloc[selected_i]

sample_index = row["sample_index"]
matched_run_id = row.get("matched_run_id", "")
lens_id = row.get("lens_id", 1)

st.header(f"Sample {sample_index} | Lens {lens_id} | Method {method_mode}")

sample_vehicles = veh_df[
    (veh_df["sample_index"].astype(str) == str(sample_index))
    & (veh_df["lens_id"].astype(str) == str(lens_id))
].copy()


# ============================================================
# METRICS
# ============================================================

mcols = st.columns(6)

mcols[0].metric("Visible road area", fmt_m2(row.get("visible_area_m2_display")))
mcols[1].metric("Vehicle-hidden area", fmt_m2(row.get("vehicle_hidden_area_m2_display")))
mcols[2].metric("Final road area", fmt_m2(row.get("final_area_m2_display")))
mcols[3].metric("Occlusion fraction", fmt_num(row.get("occlusion_fraction_display")))
mcols[4].metric("Auto status", str(row.get("auto_status_display", "")))
mcols[5].metric("Occ. quality", str(row.get("occlusion_quality_display", "")))

st.write({
    "road_mask_area_quality": row.get(
        "road_mask_area_quality",
        row.get("road_mask_area_quality_v2", row.get("road_mask_area_quality_v3", "")),
    ),
    "area_quality_flag": row.get("area_quality_flag", ""),
    "method_version": row.get("method_version_display", ""),
})


# ============================================================
# IMAGES
# ============================================================

image_path = Path(str(row.get("processed_frame_path", "")))
mask_path = Path(str(row.get("saved_road_mask_path", "")))

original = load_rgb_image(image_path)
mask_rgb = load_mask_rgb(mask_path)
overlay = overlay_mask_on_image(original, mask_rgb)
boxes = draw_vehicle_boxes(original, sample_vehicles, method_mode)

if show_overlay:
    make_side_by_side(
        [original, mask_rgb, overlay, boxes],
        ["Original", "Road Mask", "Mask Overlay", "Vehicle Boxes"],
    )
else:
    make_side_by_side(
        [original, mask_rgb, boxes],
        ["Original", "Road Mask", "Vehicle Boxes"],
    )

if show_raw_paths:
    st.write({
        "processed_frame_path": str(image_path),
        "saved_road_mask_path": str(mask_path),
        "road_area_csv": str(road_path),
        "vehicle_detail_csv": str(veh_path),
        "label_output_csv": str(label_path),
    })


# ============================================================
# VEHICLE DETAILS
# ============================================================

if show_vehicle_table:
    st.subheader("Vehicle detection / occlusion detail")

    if method_mode == "v1":
        vehicle_cols = [
            "class_name",
            "detection_confidence",
            "bbox_width_px",
            "bbox_height_px",
            "standard_footprint_m2",
            "visibility_factor",
            "road_contact_confidence",
            "hidden_road_area_m2_est_full_footprint",
            "conservative_factor",
            "hidden_road_area_m2_est_conservative",
            "road_contact_reason",
            "visibility_reason",
        ]
        hidden_col = "hidden_road_area_m2_est_conservative"

    elif method_mode == "v2":
        vehicle_cols = [
            "class_name",
            "detection_confidence",
            "bbox_width_px",
            "bbox_height_px",
            "bbox_area_ratio",
            "standard_footprint_m2",
            "bbox_visibility_factor_v2",
            "conservative_factor_v2",
            "hidden_road_area_m2_full_footprint_v2",
            "hidden_road_area_m2_conservative_v2",
            "visibility_reason_v2",
            "bbox_touches_edge_count_v2",
        ]
        hidden_col = "hidden_road_area_m2_conservative_v2"

    else:
        vehicle_cols = [
            "class_name",
            "detection_confidence",
            "bbox_width_px",
            "bbox_height_px",
            "bbox_area_ratio",
            "standard_footprint_m2",
            "bbox_visibility_factor_v3",
            "vehicle_depth_m_v3",
            "vehicle_depth_p25_m_v3",
            "vehicle_depth_p75_m_v3",
            "vehicle_depth_factor_v3",
            "vehicle_depth_factor_reason_v3",
            "conservative_factor_v3",
            "hidden_road_area_m2_full_footprint_v3",
            "hidden_road_area_m2_conservative_v3",
            "visibility_reason_v3",
            "depth_reason_v3",
            "used_for_occlusion_v3",
        ]
        hidden_col = "hidden_road_area_m2_conservative_v3"

    vehicle_cols = [c for c in vehicle_cols if c in sample_vehicles.columns]

    if len(sample_vehicles) > 0:
        st.dataframe(sample_vehicles[vehicle_cols], use_container_width=True)

        if hidden_col in sample_vehicles.columns:
            st.write("Vehicle hidden-area summary:")
            st.write(
                sample_vehicles.groupby("class_name")[hidden_col]
                .sum()
                .sort_values(ascending=False)
            )

        if method_mode == "v3" and "vehicle_depth_factor_reason_v3" in sample_vehicles.columns:
            st.write("Vehicle depth-factor reasons:")
            st.write(sample_vehicles["vehicle_depth_factor_reason_v3"].value_counts(dropna=False))
    else:
        st.info("No vehicle detections found for this sample.")


# ============================================================
# MANUAL VALIDATION FORM
# ============================================================

st.subheader("Manual validation")

existing = get_existing_label(
    labels_df=labels_df,
    sample_index=sample_index,
    matched_run_id=matched_run_id,
    lens_id=lens_id,
    method=method_mode,
)

quality_options = ["", "good", "medium", "bad", "uncertain"]
occlusion_options = ["", "good", "medium", "bad", "overestimated", "underestimated", "uncertain"]
decision_options = ["", "accept_primary", "accept_sensitivity", "reject", "needs_review"]

with st.form("validation_form"):
    qcols = st.columns(5)

    vehicle_quality = qcols[0].selectbox(
        "Vehicle detection quality",
        quality_options,
        index=safe_select_index(quality_options, existing.get("manual_vehicle_detection_quality", "")),
    )

    road_mask_quality = qcols[1].selectbox(
        "Road mask quality",
        quality_options,
        index=safe_select_index(quality_options, existing.get("manual_road_mask_quality", "")),
    )

    depth_quality = qcols[2].selectbox(
        "Depth/visible area quality",
        quality_options,
        index=safe_select_index(quality_options, existing.get("manual_depth_area_quality", "")),
    )

    vehicle_occ_quality = qcols[3].selectbox(
        "Vehicle occlusion quality",
        occlusion_options,
        index=safe_select_index(occlusion_options, existing.get("manual_vehicle_occlusion_quality", "")),
    )

    final_decision = qcols[4].selectbox(
        "Final manual decision",
        decision_options,
        index=safe_select_index(decision_options, existing.get("manual_final_decision", "")),
    )

    st.markdown("### Manual vehicle error counts")

    ccols = st.columns(4)

    fp_count = ccols[0].number_input(
        "False positives",
        min_value=0,
        step=1,
        value=int(existing.get("manual_vehicle_false_positive_count", 0) or 0),
    )

    fn_count = ccols[1].number_input(
        "False negatives",
        min_value=0,
        step=1,
        value=int(existing.get("manual_vehicle_false_negative_count", 0) or 0),
    )

    wrong_class_count = ccols[2].number_input(
        "Wrong class",
        min_value=0,
        step=1,
        value=int(existing.get("manual_vehicle_wrong_class_count", 0) or 0),
    )

    duplicate_count = ccols[3].number_input(
        "Duplicate boxes",
        min_value=0,
        step=1,
        value=int(existing.get("manual_vehicle_duplicate_count", 0) or 0),
    )

    st.markdown("### Failure tags")

    tag_cols = st.columns(3)

    road_underseg = tag_cols[0].checkbox(
        "Road mask under-segmentation",
        value=bool(existing.get("manual_failure_road_mask_undersegmentation", False)),
    )

    road_overseg = tag_cols[0].checkbox(
        "Road mask over-segmentation",
        value=bool(existing.get("manual_failure_road_mask_oversegmentation", False)),
    )

    vehicle_fn = tag_cols[0].checkbox(
        "Vehicle false negatives",
        value=bool(existing.get("manual_failure_vehicle_false_negative", False)),
    )

    vehicle_fp = tag_cols[1].checkbox(
        "Vehicle false positives",
        value=bool(existing.get("manual_failure_vehicle_false_positive", False)),
    )

    vehicle_wrong = tag_cols[1].checkbox(
        "Vehicle wrong class",
        value=bool(existing.get("manual_failure_vehicle_wrong_class", False)),
    )

    duplicate_boxes = tag_cols[1].checkbox(
        "Duplicate boxes remain",
        value=bool(existing.get("manual_failure_duplicate_boxes", False)),
    )

    occ_over = tag_cols[2].checkbox(
        "Occlusion overestimated",
        value=bool(existing.get("manual_failure_occlusion_overestimated", False)),
    )

    occ_under = tag_cols[2].checkbox(
        "Occlusion underestimated",
        value=bool(existing.get("manual_failure_occlusion_underestimated", False)),
    )

    depth_suspicious = tag_cols[2].checkbox(
        "Depth suspicious / depth-gate wrong",
        value=bool(existing.get("manual_failure_depth_suspicious", False)),
    )

    glare_shadow = st.checkbox(
        "Glare/shadow uncertainty",
        value=bool(existing.get("manual_failure_glare_shadow_uncertain", False)),
    )

    heavy_occ = st.checkbox(
        "Heavy vehicle occlusion",
        value=bool(existing.get("manual_failure_heavy_occlusion", False)),
    )

    notes = st.text_area(
        "Manual notes",
        value=str(existing.get("manual_notes", "")) if pd.notna(existing.get("manual_notes", "")) else "",
        height=120,
    )

    submitted = st.form_submit_button("Save validation label")

    if submitted:
        updated = {
            "sample_index": sample_index,
            "matched_run_id": matched_run_id,
            "lens_id": lens_id,
            "validation_method": method_mode,

            "manual_vehicle_detection_quality": vehicle_quality,
            "manual_road_mask_quality": road_mask_quality,
            "manual_depth_area_quality": depth_quality,
            "manual_vehicle_occlusion_quality": vehicle_occ_quality,
            "manual_final_decision": final_decision,

            "manual_vehicle_false_positive_count": int(fp_count),
            "manual_vehicle_false_negative_count": int(fn_count),
            "manual_vehicle_wrong_class_count": int(wrong_class_count),
            "manual_vehicle_duplicate_count": int(duplicate_count),

            "manual_failure_road_mask_undersegmentation": bool(road_underseg),
            "manual_failure_road_mask_oversegmentation": bool(road_overseg),
            "manual_failure_vehicle_false_negative": bool(vehicle_fn),
            "manual_failure_vehicle_false_positive": bool(vehicle_fp),
            "manual_failure_vehicle_wrong_class": bool(vehicle_wrong),
            "manual_failure_duplicate_boxes": bool(duplicate_boxes),
            "manual_failure_occlusion_overestimated": bool(occ_over),
            "manual_failure_occlusion_underestimated": bool(occ_under),
            "manual_failure_depth_suspicious": bool(depth_suspicious),
            "manual_failure_glare_shadow_uncertain": bool(glare_shadow),
            "manual_failure_heavy_occlusion": bool(heavy_occ),

            "manual_notes": notes,
        }

        labels_df = update_label(labels_df, updated)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(label_path, index=False)

        st.success(f"Saved label to: {label_path}")


# ============================================================
# VALIDATION PROGRESS
# ============================================================

st.divider()
st.subheader("Validation progress")

labelled = labels_df[
    labels_df["manual_final_decision"].fillna("").astype(str).str.len() > 0
].copy()

st.write(f"Labelled samples: {len(labelled)} / {len(labels_df)}")

if len(labelled) > 0:
    st.write("Manual final decision counts:")
    st.write(pd.crosstab(labelled["validation_method"], labelled["manual_final_decision"]))

    st.write("Failure tag counts:")
    failure_cols = [c for c in LABEL_COLUMNS if c.startswith("manual_failure_")]
    failure_counts = labelled[failure_cols].sum().sort_values(ascending=False)
    st.write(failure_counts)

st.caption(
    "Reminder: this validates a proxy feature, not ground-truth physical road area. "
    "v1 uses road-contact confidence. v2 uses bbox visibility only and may overestimate in dense traffic. "
    "v3 uses vehicle-depth gating to reject far vehicles beyond the depth cutoff."
)