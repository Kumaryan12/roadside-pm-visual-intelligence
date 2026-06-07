from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES_CSV = Path(
    "outputs/features/particle_density_modeling_table_idd_finetuned_osm_v2.csv"
)

DEFAULT_OUTPUT_CSV = Path(
    "outputs/features/particle_density_modeling_table_idd_finetuned_osm_density_v2.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/modeling/ablation_effective_density_idd_finetuned_osm_v2"
)


# ------------------------------------------------------------
# Column helpers
# ------------------------------------------------------------

def first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_datetime_series(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def rmse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def safe_spearman(y_true, y_pred):
    try:
        corr, _ = spearmanr(y_true, y_pred)
        if np.isnan(corr):
            return np.nan
        return float(corr)
    except Exception:
        return np.nan


def unique_keep_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


# ------------------------------------------------------------
# Effective density calculation
# ------------------------------------------------------------

def add_effective_density(df, assumed_diameter_um=None):
    """
    Calculates effective particle density using:

    effective_density =
    PM2.5_mass_kg_m3 /
    [nPM2_particles_m3 * spherical_particle_volume_m3]

    Assumptions:
    - PM2.5 mass is in microgram/m3
    - nPM2 is in particles/cm3
    - particle is approximated as a sphere
    - if sTPS/effective_diameter_um is missing, assumed_diameter_um is used
    """

    pm25_col = first_existing_col(
        df,
        [
            "value.sPM2",
            "sPM2",
            "pm25_predicted",
            "PM2.5",
            "PM25",
        ],
    )

    npm2_col = first_existing_col(
        df,
        [
            "nPM2",
            "value.nPM2",
            "value.sNPM2",
            "sNPM2",
        ],
    )

    diameter_col = first_existing_col(
        df,
        [
            "sTPS",
            "effective_diameter_um",
            "effective_particle_diameter_um",
        ],
    )

    if pm25_col is None:
        raise ValueError(
            "Could not find PM2.5 mass column. Expected one of: "
            "value.sPM2, sPM2, pm25_predicted"
        )

    if npm2_col is None:
        raise ValueError(
            "Could not find nPM2 number concentration column. Expected one of: "
            "nPM2, value.nPM2, value.sNPM2, sNPM2"
        )

    df[pm25_col] = pd.to_numeric(df[pm25_col], errors="coerce")
    df[npm2_col] = pd.to_numeric(df[npm2_col], errors="coerce")

    # ------------------------------------------------------------
    # Diameter handling
    # ------------------------------------------------------------
    if diameter_col is not None:
        print(f"Using diameter column: {diameter_col}")
        df[diameter_col] = pd.to_numeric(df[diameter_col], errors="coerce")
        df["effective_diameter_um"] = df[diameter_col]
        df["effective_diameter_source"] = "sensor_or_existing_column"

    else:
        if assumed_diameter_um is None:
            raise ValueError(
                "Could not find sTPS/effective_diameter_um column. "
                "Since no diameter column exists, pass --assumed-diameter-um, "
                "for example: --assumed-diameter-um 0.40"
            )

        print(
            "\nWARNING: No sTPS/effective diameter column found."
            f"\nUsing assumed constant spherical particle diameter = {assumed_diameter_um} µm"
            "\nThis is an assumed-diameter effective density proxy, not true measured density.\n"
        )

        df["effective_diameter_um"] = float(assumed_diameter_um)
        df["effective_diameter_source"] = "assumed_constant_diameter"

    # ------------------------------------------------------------
    # Density calculation
    # ------------------------------------------------------------

    # diameter in µm -> radius in meters
    df["effective_radius_m"] = (df["effective_diameter_um"] / 2.0) * 1e-6

    # spherical volume of one particle
    df["single_particle_volume_m3"] = (
        (4.0 / 3.0) * np.pi * (df["effective_radius_m"] ** 3)
    )

    # PM2.5 mass: µg/m3 -> kg/m3
    df["pm25_mass_kg_m3"] = df[pm25_col] * 1e-9

    # nPM2: particles/cm3 -> particles/m3
    df["nPM2_particles_m3"] = df[npm2_col] * 1e6

    # total particle volume per 1 m3 of air
    df["total_particle_volume_m3_per_m3_air"] = (
        df["nPM2_particles_m3"] * df["single_particle_volume_m3"]
    )

    df["effective_density_kg_m3"] = (
        df["pm25_mass_kg_m3"] / df["total_particle_volume_m3_per_m3_air"]
    )

    df["effective_density_kg_m3"] = df["effective_density_kg_m3"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # weaker comparison proxy
    df["pm25_mass_to_number_proxy"] = df[pm25_col] / df[npm2_col]
    df["pm25_mass_to_number_proxy"] = df["pm25_mass_to_number_proxy"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print("\nEffective density calculation columns:")
    print("PM2.5 mass column:", pm25_col)
    print("nPM2 number column:", npm2_col)
    print("diameter source:", df["effective_diameter_source"].iloc[0])
    print("\nEffective diameter used, µm:")
    print(df["effective_diameter_um"].describe())

    return df


# ------------------------------------------------------------
# Merge density source with image feature table
# ------------------------------------------------------------

def merge_density_source(features_df, density_df, tolerance_seconds=30):
    """
    If density_df is supplied separately, merge density-related columns into feature table.

    Priority:
    1. sample_index/sample_id merge if possible
    2. timestamp nearest merge if possible
    """

    density_df = density_df.copy()

    useful_cols = []

    for c in density_df.columns:
        c_low = c.lower()
        if (
            c in ["sample_index", "sample_id", "timestamp", "sensor_timestamp"]
            or "spm2" in c_low
            or "npm2" in c_low
            or c == "nPM2"
            or "stps" in c_low
            or "diameter" in c_low
            or "density" in c_low
            or c in ["temp", "rh"]
        ):
            useful_cols.append(c)

    density_small = density_df[unique_keep_order(useful_cols)].copy()

    if "sample_index" in features_df.columns:
        features_df["sample_index"] = pd.to_numeric(
            features_df["sample_index"],
            errors="coerce",
        ).astype("Int64")

    if "sample_id" in density_small.columns and "sample_index" not in density_small.columns:
        density_small = density_small.rename(columns={"sample_id": "sample_index"})

    if "sample_index" in features_df.columns and "sample_index" in density_small.columns:
        density_small["sample_index"] = pd.to_numeric(
            density_small["sample_index"],
            errors="coerce",
        ).astype("Int64")

        merged = features_df.merge(
            density_small,
            on="sample_index",
            how="left",
            suffixes=("", "_density_source"),
            validate="one_to_one",
        )

        print("Merged density source using sample_index.")
        return merged

    feature_time_col = first_existing_col(features_df, ["timestamp", "sensor_timestamp"])
    density_time_col = first_existing_col(density_small, ["timestamp", "sensor_timestamp"])

    if feature_time_col is None or density_time_col is None:
        raise ValueError(
            "Could not merge density source. Need either sample_index/sample_id or timestamp."
        )

    f = features_df.copy()
    d = density_small.copy()

    f["_merge_time"] = parse_datetime_series(f[feature_time_col])
    d["_merge_time"] = parse_datetime_series(d[density_time_col])

    f = f.sort_values("_merge_time")
    d = d.sort_values("_merge_time")

    merged = pd.merge_asof(
        f,
        d,
        on="_merge_time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        suffixes=("", "_density_source"),
    )

    merged = merged.drop(columns=["_merge_time"])

    print(f"Merged density source using nearest timestamp within {tolerance_seconds} seconds.")
    return merged


# ------------------------------------------------------------
# Feature groups for ablation
# ------------------------------------------------------------

def is_leakage_or_bad_column(col, target):
    c = col.lower()

    if col == target:
        return True

    # Prevent leakage:
    # effective_density is calculated from PM mass, nPM2, diameter, and volume terms.
    leakage_terms = [
        "spm",
        "npm",
        "pm25",
        "pm2",
        "pm1",
        "pm4",
        "pm10",
        "stps",
        "diameter",
        "radius",
        "single_particle_volume",
        "total_particle_volume",
        "density",
        "mass_to_number",
    ]

    if any(term in c for term in leakage_terms):
        return True

    bad_terms = [
        "path",
        "key",
        "status",
        "error",
        "model",
        "timestamp",
        "time",
        "date",
        "sample_index",
        "sample_id",
        "run_id",
        "unix",
        "offset",
        "latitude",
        "longitude",
        "location_key",
        "available",
    ]

    if any(term in c for term in bad_terms):
        return True

    return False


def build_feature_groups(df, target):
    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()

    candidate_cols = [
        c for c in numeric_cols
        if not is_leakage_or_bad_column(c, target)
    ]

    env_cols = [
        c for c in candidate_cols
        if (
            c.lower() in ["rh", "temp", "dry_air_fraction"]
            or "humidity" in c.lower()
            or "temperature" in c.lower()
        )
    ]

    gas_cols = [
        c for c in candidate_cols
        if (
            "co_ppb" in c.lower()
            or "no2_ppb" in c.lower()
            or "so2_ppb" in c.lower()
            or "o3_ppb" in c.lower()
        )
    ]

    vehicle_cols = [
        c for c in candidate_cols
        if c.startswith("idd_")
    ]

    road_cols = [
        c for c in candidate_cols
        if c.startswith("road_")
    ]

    osm_cols = [
        c for c in candidate_cols
        if (
            c.startswith("osm_")
            or c.endswith("_250m")
            or "road_segment_count" in c
            or "total_road_length" in c
            or "road_count" in c
        )
    ]

    interaction_cols = [
        c for c in candidate_cols
        if (
            "_x_" in c
            or c.startswith("vehicle_resuspension")
            or c.startswith("resuspension_x")
            or c.startswith("exhaust_x")
        )
    ]

    groups = {
        "environment_only": env_cols,
        "gas_only": gas_cols,
        "vehicle_only": vehicle_cols,
        "road_only": road_cols,
        "osm_only": osm_cols,

        "vehicle_plus_road": unique_keep_order(
            vehicle_cols + road_cols
        ),

        "vehicle_road_interactions": unique_keep_order(
            vehicle_cols + road_cols + interaction_cols
        ),

        "vehicle_road_env_interactions": unique_keep_order(
            env_cols + vehicle_cols + road_cols + interaction_cols
        ),

        "full_image_osm_context": unique_keep_order(
            env_cols + gas_cols + vehicle_cols + road_cols + osm_cols + interaction_cols
        ),
    }

    groups = {k: v for k, v in groups.items() if len(v) > 0}

    return groups


def make_models(random_state=42):
    return {
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        random_state=random_state,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_dummy(y, splits):
    rows = []

    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(np.zeros((len(y_train), 1)), y_train)
        pred = dummy.predict(np.zeros((len(y_test), 1)))

        rows.append({
            "feature_group": "dummy_baseline",
            "model": "dummy_mean",
            "split": split_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_features": 0,
            "mae": mean_absolute_error(y_test, pred),
            "rmse": rmse(y_test, pred),
            "r2": r2_score(y_test, pred),
            "spearman_pred_vs_true": safe_spearman(y_test, pred),
        })

    return rows


def evaluate_group(df, y, group_name, feature_cols, models, splits):
    rows = []
    X = df[feature_cols].copy()

    for model_name, model in models.items():
        for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            pred = model.predict(X_test)

            rows.append({
                "feature_group": group_name,
                "model": model_name,
                "split": split_id,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features": len(feature_cols),
                "mae": mean_absolute_error(y_test, pred),
                "rmse": rmse(y_test, pred),
                "r2": r2_score(y_test, pred),
                "spearman_pred_vs_true": safe_spearman(y_test, pred),
            })

    return rows


def save_full_rf_importance(df, y, feature_cols, output_path):
    X = df[feature_cols].copy()

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=800,
                    random_state=42,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X, y)
    rf = model.named_steps["model"]

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    imp.to_csv(output_path, index=False)
    return imp


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV))
    parser.add_argument("--density-csv", default=None)
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", default="effective_density_kg_m3")

    parser.add_argument(
        "--assumed-diameter-um",
        type=float,
        default=None,
        help=(
            "Use this constant effective spherical particle diameter in micrometers "
            "if sTPS/effective_diameter_um is missing. Example: 0.40"
        ),
    )

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--timestamp-tolerance-sec", type=int, default=30)
    parser.add_argument("--require-image", action="store_true")
    parser.add_argument("--require-osm", action="store_true")

    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    df = pd.read_csv(features_csv)
    print("Loaded features:", features_csv)
    print("Features shape:", df.shape)

    if args.density_csv is not None:
        density_csv = Path(args.density_csv)

        if not density_csv.exists():
            raise FileNotFoundError(f"Density CSV not found: {density_csv}")

        density_df = pd.read_csv(density_csv)
        print("Loaded density source:", density_csv)
        print("Density source shape:", density_df.shape)

        df = merge_density_source(
            features_df=df,
            density_df=density_df,
            tolerance_seconds=args.timestamp_tolerance_sec,
        )

    df = add_effective_density(
        df,
        assumed_diameter_um=args.assumed_diameter_um,
    )

    df.to_csv(output_csv, index=False)

    print("\nSaved density-enhanced modeling table:")
    print(output_csv)

    print("\nEffective density stats:")
    print(df["effective_density_kg_m3"].describe())

    print("\nTarget preview:")
    preview_cols = [
        "timestamp",
        "sample_index",
        "value.sPM2",
        "sPM2",
        "pm25_predicted",
        "nPM2",
        "sTPS",
        "effective_diameter_um",
        "effective_diameter_source",
        "effective_density_kg_m3",
        "pm25_mass_to_number_proxy",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(10).to_string(index=False))

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    target = args.target

    if target not in df.columns:
        raise ValueError(f"Target column not found after density calculation: {target}")

    if args.require_image and "image_features_available" in df.columns:
        before = len(df)
        df = df[df["image_features_available"] == True].copy()
        print(f"\nFiltered image_features_available: {before} -> {len(df)}")

    if args.require_osm and "osm_features_available" in df.columns:
        before = len(df)
        df = df[df["osm_features_available"] == True].copy()
        print(f"Filtered osm_features_available: {before} -> {len(df)}")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].notna()].copy()

    # Remove extreme top 1% to reduce instability.
    # This is still broad because target is an effective proxy, not literal density.
    upper = df[target].quantile(0.99)
    df = df[
        (df[target] > 0)
        & (df[target] < upper)
    ].copy()

    if "sample_index" in df.columns:
        df = df.sort_values("sample_index").reset_index(drop=True)
    else:
        time_col = first_existing_col(df, ["timestamp", "sensor_timestamp"])
        if time_col:
            df["_time_sort"] = parse_datetime_series(df[time_col])
            df = df.sort_values("_time_sort").reset_index(drop=True)

    print("\nRows used for training:", len(df))
    print("Final target stats:")
    print(df[target].describe())

    if len(df) < 40:
        raise ValueError("Too few rows for TimeSeriesSplit training.")

    n_splits = min(args.n_splits, max(2, len(df) // 40))
    print("TimeSeriesSplit n_splits:", n_splits)

    splits = list(TimeSeriesSplit(n_splits=n_splits).split(df))
    feature_groups = build_feature_groups(df, target)

    print("\nFeature groups:")
    for name, cols in feature_groups.items():
        print(f"{name}: {len(cols)} features")

    with open(output_dir / "density_ablation_feature_groups.json", "w") as f:
        json.dump(feature_groups, f, indent=2)

    models = make_models()

    rows = []
    rows.extend(evaluate_dummy(df[target], splits))

    for group_name, cols in feature_groups.items():
        print(f"\nEvaluating {group_name} ({len(cols)} features)")
        rows.extend(
            evaluate_group(
                df=df,
                y=df[target],
                group_name=group_name,
                feature_cols=cols,
                models=models,
                splits=splits,
            )
        )

    results = pd.DataFrame(rows)
    results_path = output_dir / "density_ablation_results_by_split.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results
        .groupby(["feature_group", "model"])
        .agg(
            n_features=("n_features", "max"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            spearman_mean=("spearman_pred_vs_true", "mean"),
            spearman_std=("spearman_pred_vs_true", "std"),
        )
        .reset_index()
        .sort_values(["rmse_mean", "mae_mean"], ascending=[True, True])
    )

    summary_path = output_dir / "density_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print("Density-enhanced table:", output_csv)
    print("Results by split:", results_path)
    print("Summary:", summary_path)

    print("\nDensity ablation summary sorted by RMSE:")
    print(summary.to_string(index=False))

    importance_group = None
    for candidate in [
        "full_image_osm_context",
        "vehicle_road_env_interactions",
        "vehicle_road_interactions",
        "vehicle_only",
    ]:
        if candidate in feature_groups:
            importance_group = candidate
            break

    if importance_group:
        importance_path = output_dir / f"rf_feature_importance_{importance_group}.csv"
        imp = save_full_rf_importance(
            df=df,
            y=df[target],
            feature_cols=feature_groups[importance_group],
            output_path=importance_path,
        )

        print("\nSaved RF feature importance:", importance_path)
        print("\nTop RF importances:")
        print(imp.head(30).to_string(index=False))


if __name__ == "__main__":
    main()