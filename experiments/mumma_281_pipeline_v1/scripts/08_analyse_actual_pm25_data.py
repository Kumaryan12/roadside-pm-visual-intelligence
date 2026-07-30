from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path("experiments/mumma_281_pipeline_v1")
DATA = ROOT / "data/processed/final_feature_table.csv"

REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures/actual_pm25_analysis"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = REPORTS / "actual_pm25_data_analysis_summary.csv"
OUT_LAG = REPORTS / "actual_pm25_lag_analysis.csv"
OUT_SPIKES = REPORTS / "actual_pm25_spike_rows.csv"
OUT_DUPLICATES = REPORTS / "actual_pm25_duplicate_timestamp_analysis.csv"


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    df = pd.read_csv(DATA)

    target = "value.sPM2"
    if target not in df.columns:
        raise ValueError(f"{target} not found")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")

    sort_cols = ["timestamp_parsed"]
    if "sample_index" in df.columns:
        sort_cols.append("sample_index")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    y = df[target].values.astype(float)

    print("=" * 100)
    print("BASIC DATA CHECK")
    print("Shape:", df.shape)
    print("Timestamp range:", df["timestamp_parsed"].min(), "to", df["timestamp_parsed"].max())
    print("Unique timestamps:", df["timestamp_parsed"].nunique())
    print("Duplicate timestamps:", df["timestamp_parsed"].duplicated().sum())
    print("Target missing:", df[target].isna().sum())

    print("\nPM2.5 summary:")
    print(df[target].describe().to_string())

    # Time gaps
    df["time_gap_sec"] = df["timestamp_parsed"].diff().dt.total_seconds()
    print("\nTime gap summary:")
    print(df["time_gap_sec"].describe().to_string())

    # Adjacent differences
    df["pm25_prev"] = df[target].shift(1)
    df["pm25_next"] = df[target].shift(-1)
    df["pm25_diff"] = df[target].diff()
    df["pm25_abs_diff"] = df["pm25_diff"].abs()
    df["pm25_pct_change"] = 100 * df["pm25_diff"] / (df["pm25_prev"].replace(0, np.nan))

    print("\nAdjacent PM2.5 absolute difference summary:")
    print(df["pm25_abs_diff"].describe().to_string())

    # Spike rows
    spike_thresholds = [5, 10, 20, 50]
    spike_records = []
    for th in spike_thresholds:
        n = int((df["pm25_abs_diff"] > th).sum())
        pct = 100 * n / max(1, len(df) - 1)
        spike_records.append({
            "threshold_abs_change": th,
            "n_spikes": n,
            "percent_of_transitions": pct,
        })

    spike_summary = pd.DataFrame(spike_records)
    print("\nSpike transition counts:")
    print(spike_summary.to_string(index=False))

    spike_rows = df[df["pm25_abs_diff"] > 10].copy()
    spike_rows.to_csv(OUT_SPIKES, index=False)

    # Lag predictability: y(t-k) predicts y(t)
    lag_rows = []
    for lag in range(1, 31):
        true = df[target].iloc[lag:].values
        pred = df[target].shift(lag).iloc[lag:].values

        valid = np.isfinite(true) & np.isfinite(pred)
        true = true[valid]
        pred = pred[valid]

        if len(true) == 0:
            continue

        lag_rows.append({
            "lag_rows": lag,
            "approx_lag_seconds_if_10s": lag * 10,
            "n": len(true),
            "MAE": mean_absolute_error(true, pred),
            "RMSE": rmse(true, pred),
            "R2": r2_score(true, pred),
            "Spearman": spearmanr(true, pred).correlation,
            "mean_abs_change": np.mean(np.abs(true - pred)),
            "median_abs_change": np.median(np.abs(true - pred)),
        })

    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(OUT_LAG, index=False)

    print("\nLag predictability:")
    print(lag_df.head(15).to_string(index=False))

    # Duplicate timestamp analysis
    dup = (
        df.groupby("timestamp_parsed")[target]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    dup["range"] = dup["max"] - dup["min"]
    dup = dup.sort_values(["count", "range"], ascending=[False, False])
    dup.to_csv(OUT_DUPLICATES, index=False)

    print("\nDuplicate timestamp PM2.5 variation:")
    print(dup.head(20).to_string(index=False))

    # Same-minute aggregation diagnostic
    minute_df = (
        df.set_index("timestamp_parsed")
        .resample("1min")[target]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    minute_df["range"] = minute_df["max"] - minute_df["min"]

    print("\nMinute-level PM2.5 variation:")
    print(minute_df.describe().to_string())

    # Persistence baselines using previous row and rolling mean
    baseline_rows = []

    for name, pred_series in {
        "previous_row": df[target].shift(1),
        "rolling_3_prev_mean": df[target].shift(1).rolling(3).mean(),
        "rolling_5_prev_mean": df[target].shift(1).rolling(5).mean(),
        "rolling_7_prev_mean": df[target].shift(1).rolling(7).mean(),
        "rolling_10_prev_mean": df[target].shift(1).rolling(10).mean(),
    }.items():
        true = df[target].values
        pred = pred_series.values
        valid = np.isfinite(true) & np.isfinite(pred)

        true_v = true[valid]
        pred_v = pred[valid]

        baseline_rows.append({
            "baseline": name,
            "n": len(true_v),
            "MAE": mean_absolute_error(true_v, pred_v),
            "RMSE": rmse(true_v, pred_v),
            "R2": r2_score(true_v, pred_v),
            "Spearman": spearmanr(true_v, pred_v).correlation,
            "Bias": float(np.mean(pred_v - true_v)),
        })

    baseline_df = pd.DataFrame(baseline_rows)
    print("\nPersistence baselines:")
    print(baseline_df.to_string(index=False))

    # Save combined summary
    summary = pd.DataFrame([
        {"metric": "n_rows", "value": len(df)},
        {"metric": "unique_timestamps", "value": df["timestamp_parsed"].nunique()},
        {"metric": "duplicate_timestamps", "value": df["timestamp_parsed"].duplicated().sum()},
        {"metric": "pm25_mean", "value": np.nanmean(y)},
        {"metric": "pm25_std", "value": np.nanstd(y)},
        {"metric": "pm25_min", "value": np.nanmin(y)},
        {"metric": "pm25_max", "value": np.nanmax(y)},
        {"metric": "adjacent_abs_diff_mean", "value": df["pm25_abs_diff"].mean()},
        {"metric": "adjacent_abs_diff_median", "value": df["pm25_abs_diff"].median()},
        {"metric": "adjacent_abs_diff_75p", "value": df["pm25_abs_diff"].quantile(0.75)},
        {"metric": "adjacent_abs_diff_max", "value": df["pm25_abs_diff"].max()},
    ])
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved:")
    print(OUT_SUMMARY)
    print(OUT_LAG)
    print(OUT_SPIKES)
    print(OUT_DUPLICATES)


if __name__ == "__main__":
    main()