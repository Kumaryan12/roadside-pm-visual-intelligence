from pathlib import Path
import numpy as np
import pandas as pd


INPUT_CSV = Path("outputs/features/particle_density_modeling_table_idd_finetuned_osm_v2.csv")
OUTPUT_CSV = Path("outputs/features/particle_density_modeling_table_idd_finetuned_osm_density_v2.csv")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print("Loaded:", INPUT_CSV)
    print("Shape:", df.shape)

    pm25_col = find_col(df, [
        "value.sPM2",
        "sPM2",
        "pm25_predicted",
        "PM2.5",
        "PM25",
    ])

    npm2_col = find_col(df, [
        "nPM2",
        "value.nPM2",
        "value.sNPM2",
        "sNPM2",
    ])

    stps_col = find_col(df, [
        "sTPS",
        "effective_diameter_um",
        "effective_particle_diameter_um",
    ])

    print("\nDetected columns:")
    print("PM2.5 mass column:", pm25_col)
    print("nPM2 number column:", npm2_col)
    print("effective diameter / sTPS column:", stps_col)

    if pm25_col is None:
        raise ValueError(
            "Could not find PM2.5 mass column. "
            "Expected value.sPM2 or sPM2."
        )

    if npm2_col is None:
        raise ValueError(
            "Could not find nPM2 number concentration column. "
            "Expected nPM2."
        )

    if stps_col is None:
        print("\nERROR:")
        print("Your CSV does not contain sTPS/effective diameter.")
        print("So proper effective_density_kg_m3 cannot be calculated yet.")
        print("\nYou have two options:")
        print("1. Get MC1S sensor data for the same 11:54–12:41 window with sTPS.")
        print("2. Temporarily create only pm25_mass_to_number_proxy = PM2.5 / nPM2.")
        print("\nCreating mass-to-number proxy only...")

        df[pm25_col] = pd.to_numeric(df[pm25_col], errors="coerce")
        df[npm2_col] = pd.to_numeric(df[npm2_col], errors="coerce")

        df["pm25_mass_to_number_proxy"] = df[pm25_col] / df[npm2_col]
        df["pm25_mass_to_number_proxy"] = df["pm25_mass_to_number_proxy"].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)

        print("\nSaved:", OUTPUT_CSV)
        print("\nProxy stats:")
        print(df["pm25_mass_to_number_proxy"].describe())
        return

    # Convert to numeric
    df[pm25_col] = pd.to_numeric(df[pm25_col], errors="coerce")
    df[npm2_col] = pd.to_numeric(df[npm2_col], errors="coerce")
    df[stps_col] = pd.to_numeric(df[stps_col], errors="coerce")

    # Assumptions:
    # PM2.5 mass: microgram/m3
    # nPM2: particles/cm3
    # sTPS/effective diameter: micrometers
    df["effective_diameter_um"] = df[stps_col]

    df["effective_radius_m"] = (df["effective_diameter_um"] / 2.0) * 1e-6

    df["single_particle_volume_m3"] = (
        (4.0 / 3.0)
        * np.pi
        * (df["effective_radius_m"] ** 3)
    )

    df["pm25_mass_kg_m3"] = df[pm25_col] * 1e-9

    df["nPM2_particles_m3"] = df[npm2_col] * 1e6

    df["total_particle_volume_m3_per_m3_air"] = (
        df["nPM2_particles_m3"]
        * df["single_particle_volume_m3"]
    )

    df["effective_density_kg_m3"] = (
        df["pm25_mass_kg_m3"]
        / df["total_particle_volume_m3_per_m3_air"]
    )

    df["effective_density_kg_m3"] = df["effective_density_kg_m3"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Also create weaker comparison proxy
    df["pm25_mass_to_number_proxy"] = df[pm25_col] / df[npm2_col]
    df["pm25_mass_to_number_proxy"] = df["pm25_mass_to_number_proxy"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved:", OUTPUT_CSV)

    print("\nEffective density stats:")
    print(df["effective_density_kg_m3"].describe())

    print("\nPreview:")
    show_cols = [
        "timestamp",
        "sample_index",
        pm25_col,
        npm2_col,
        stps_col,
        "effective_diameter_um",
        "effective_density_kg_m3",
        "pm25_mass_to_number_proxy",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()