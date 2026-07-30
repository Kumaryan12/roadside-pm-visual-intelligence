from pathlib import Path
import pandas as pd
import numpy as np

REPORT_DIR = Path("experiments/traqid_pretraining_v1/pipeline_validation/reports")

IMAGE_RESULTS = REPORT_DIR / "image_dominant_feature_model_results.csv"
SELECTED_YOLO_RESULTS = REPORT_DIR / "selected_advanced_yolo_fusion_results.csv"

OUT = REPORT_DIR / "final_traqid_clean_summary_table.csv"


def pick_best(df, feature_set=None, experiment=None, model=None):
    temp = df.copy()

    if feature_set is not None:
        temp = temp[temp["feature_set"] == feature_set]

    if experiment is not None and "experiment" in temp.columns:
        temp = temp[temp["experiment"] == experiment]

    if model is not None:
        temp = temp[temp["model"] == model]

    if len(temp) == 0:
        return None

    return temp.sort_values("RMSE").iloc[0].to_dict()


def add_row(rows, label, source, row, interpretation):
    if row is None:
        rows.append({
            "Label": label,
            "Source": source,
            "Model": "NOT FOUND",
            "Feature_Set": "NOT FOUND",
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Spearman": np.nan,
            "Interpretation": "Missing result row. Check source CSV."
        })
        return

    rows.append({
        "Label": label,
        "Source": source,
        "Model": row.get("model", ""),
        "Feature_Set": row.get("feature_set", ""),
        "MAE": row.get("MAE", np.nan),
        "RMSE": row.get("RMSE", np.nan),
        "R2": row.get("R2", np.nan),
        "Spearman": row.get("Spearman", np.nan),
        "Interpretation": interpretation
    })


def main():
    print("Loading:")
    print(" -", IMAGE_RESULTS)
    print(" -", SELECTED_YOLO_RESULTS)

    image = pd.read_csv(IMAGE_RESULTS)
    selected = pd.read_csv(SELECTED_YOLO_RESULTS)

    rows = []

    # ------------------------------------------------------------------
    # Image-dominant experiment rows
    # ------------------------------------------------------------------

    add_row(
        rows,
        "Mean baseline",
        "image_dominant",
        pick_best(image, feature_set="mean_baseline"),
        "Naive baseline; predicts train-set mean PM2.5."
    )

    add_row(
        rows,
        "YOLO counts only",
        "image_dominant",
        pick_best(image, feature_set="yolo_counts_only"),
        "Simple vehicle counts alone; weak and unstable."
    )

    add_row(
        rows,
        "Advanced YOLO only",
        "image_dominant",
        pick_best(image, feature_set="advanced_yolo_only"),
        "Spatial/area/proximity YOLO features without road/ResNet."
    )

    add_row(
        rows,
        "Road only",
        "image_dominant",
        pick_best(image, feature_set="road_only"),
        "Road segmentation and road-surface descriptors only."
    )

    # Best ResNet-only among PCA variants
    resnet_only = image[
        image["feature_set"].isin([
            "resnet50_only_pca16",
            "resnet50_only_pca32",
            "resnet50_only_pca64",
        ])
    ].sort_values("RMSE").iloc[0].to_dict()

    add_row(
        rows,
        "ResNet50 only",
        "image_dominant",
        resnet_only,
        "Generic scene embeddings only; captures appearance/haze/scene context."
    )

    add_row(
        rows,
        "YOLO + road",
        "image_dominant",
        pick_best(image, feature_set="yolo_plus_road"),
        "Interpretable traffic + road-surface source-proxy features."
    )

    # Best strict image model among image_all_resnet variants
    strict_image = image[
        image["feature_set"].isin([
            "image_all_resnet_pca32",
            "image_all_resnet_pca64",
        ])
    ].sort_values("RMSE").iloc[0].to_dict()

    add_row(
        rows,
        "YOLO + road + ResNet50",
        "image_dominant",
        strict_image,
        "Strict image-dominant setup without temp/humidity/day-night/time shortcuts."
    )

    add_row(
        rows,
        "Image + Temp/Humidity",
        "image_dominant",
        pick_best(image, feature_set="image_all_plus_temp_humidity"),
        "Image features with weak meteorological support only."
    )

    add_row(
        rows,
        "Image + Temp/Humidity + Day/Night",
        "image_dominant",
        pick_best(image, feature_set="image_all_plus_temp_humidity_daynight"),
        "Best image-focused multimodal result; no hour/month/season/PM10/AQI."
    )

    # ------------------------------------------------------------------
    # Fair full-prediction / residual-fusion rows
    # ------------------------------------------------------------------

    add_row(
        rows,
        "Full tabular RF baseline",
        "selected_yolo_residual",
        pick_best(selected, experiment="tabular_only", model="random_forest"),
        "Strongest fair tabular baseline with weather/time/season context."
    )

    # Best selected YOLO residual fusion
    residual_rows = selected[selected["feature_set"] == "selected_yolo_residual"]
    best_residual = residual_rows.sort_values("RMSE").iloc[0].to_dict()

    add_row(
        rows,
        "Selected YOLO residual fusion",
        "selected_yolo_residual",
        best_residual,
        "Best fair result; selected YOLO features provide small residual correction."
    )

    # Best early fusion, for contrast
    early_rows = selected[selected["feature_set"] == "tabular_plus_selected_yolo"]
    best_early = early_rows.sort_values("RMSE").iloc[0].to_dict()

    add_row(
        rows,
        "Selected YOLO early fusion",
        "selected_yolo_residual",
        best_early,
        "Directly adding selected YOLO features; weaker than residual fusion."
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    summary = pd.DataFrame(rows)

    metric_cols = ["MAE", "RMSE", "R2", "Spearman"]
    for c in metric_cols:
        summary[c] = pd.to_numeric(summary[c], errors="coerce")

    summary = summary[
        [
            "Label",
            "Source",
            "Model",
            "Feature_Set",
            "MAE",
            "RMSE",
            "R2",
            "Spearman",
            "Interpretation",
        ]
    ]

    summary.to_csv(OUT, index=False)

    print("\nFinal clean TRAQID summary:")
    print(summary.to_string(index=False))

    print("\nSaved:", OUT)

    print("\nSuggested PPT headline rows:")
    ppt_rows = summary[
        summary["Label"].isin([
            "YOLO + road + ResNet50",
            "Image + Temp/Humidity + Day/Night",
            "Full tabular RF baseline",
            "Selected YOLO residual fusion",
        ])
    ]

    print(ppt_rows.to_string(index=False))


if __name__ == "__main__":
    main()