#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def parse_values(s):
    if pd.isna(s):
        return []
    return [float(x) for x in str(s).split("|") if str(x).strip()]


def parse_ids(s):
    if pd.isna(s):
        return []
    return [int(float(x)) for x in str(s).split("|") if str(x).strip()]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def per_target_metrics(y_true, y_pred, horizon):
    out = {}

    pm25_true = y_true[:, :horizon]
    pm25_pred = y_pred[:, :horizon]
    pm10_true = y_true[:, horizon:]
    pm10_pred = y_pred[:, horizon:]

    out["PM2.5_all_horizons"] = regression_metrics(pm25_true.reshape(-1), pm25_pred.reshape(-1))
    out["PM10_all_horizons"] = regression_metrics(pm10_true.reshape(-1), pm10_pred.reshape(-1))
    out["all_outputs"] = regression_metrics(y_true.reshape(-1), y_pred.reshape(-1))

    out["PM2.5_by_horizon"] = []
    out["PM10_by_horizon"] = []

    for h in range(horizon):
        out["PM2.5_by_horizon"].append({
            "horizon_step": h + 1,
            **regression_metrics(pm25_true[:, h], pm25_pred[:, h])
        })
        out["PM10_by_horizon"].append({
            "horizon_step": h + 1,
            **regression_metrics(pm10_true[:, h], pm10_pred[:, h])
        })

    out["average_R2"] = float((out["PM2.5_all_horizons"]["R2"] + out["PM10_all_horizons"]["R2"]) / 2)
    out["average_RMSE"] = float((out["PM2.5_all_horizons"]["RMSE"] + out["PM10_all_horizons"]["RMSE"]) / 2)

    return out


def build_xy(df, hourly_df, horizon):
    """
    Features:
    - historical PM2.5 sequence, length 12
    - historical PM10 sequence, length 12
    - Temperature, Humidity, hour_sin, hour_cos, Season, Day_or_Night at input end
    Targets:
    - future PM2.5 sequence, length 12
    - future PM10 sequence, length 12
    """
    hourly_df = hourly_df.copy()
    hourly_df["hourly_row_id"] = hourly_df["hourly_row_id"].astype(int)
    hourly = hourly_df.set_index("hourly_row_id")

    rows = []
    y_rows = []

    for _, r in df.iterrows():
        input_ids = parse_ids(r["input_row_ids"])
        if len(input_ids) != horizon:
            continue

        hist_pm25 = hourly.loc[input_ids, "PM2.5"].astype(float).values
        hist_pm10 = hourly.loc[input_ids, "PM10"].astype(float).values

        fut_pm25 = parse_values(r["target_PM2.5_values"])
        fut_pm10 = parse_values(r["target_PM10_values"])

        if len(fut_pm25) != horizon or len(fut_pm10) != horizon:
            continue

        feat = {}

        for i, v in enumerate(hist_pm25):
            feat[f"hist_PM2.5_tminus_{horizon - i}"] = v

        for i, v in enumerate(hist_pm10):
            feat[f"hist_PM10_tminus_{horizon - i}"] = v

        for c in ["Temperature", "Humidity", "hour_sin", "hour_cos"]:
            feat[c] = r[c] if c in r else np.nan

        for c in ["Season", "Day_or_Night"]:
            feat[c] = r[c] if c in r else "Unknown"

        rows.append(feat)
        y_rows.append(fut_pm25 + fut_pm10)

    X = pd.DataFrame(rows)
    y = np.array(y_rows, dtype=np.float32)

    return X, y


def train_mean_predict(y_train, n):
    return np.tile(y_train.mean(axis=0, keepdims=True), (n, 1))


def persistence_predict(X, horizon):
    """
    Simple persistence:
    repeat latest observed PM2.5 and PM10 across all 12 future steps.
    """
    latest_pm25_col = "hist_PM2.5_tminus_1"
    latest_pm10_col = "hist_PM10_tminus_1"

    pm25_latest = X[latest_pm25_col].astype(float).values.reshape(-1, 1)
    pm10_latest = X[latest_pm10_col].astype(float).values.reshape(-1, 1)

    pm25_pred = np.tile(pm25_latest, (1, horizon))
    pm10_pred = np.tile(pm10_latest, (1, horizon))

    return np.concatenate([pm25_pred, pm10_pred], axis=1)


def make_model(name):
    if name == "ridge":
        return Ridge(alpha=1.0)

    numeric_cols = None

    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

    raise ValueError(f"Unknown model: {name}")


def fit_predict_model(model_name, X_train, y_train, X_test):
    numeric_cols = [c for c in X_train.columns if c not in ["Season", "Day_or_Night"]]
    categorical_cols = [c for c in ["Season", "Day_or_Night"] if c in X_train.columns]

    if model_name == "ridge":
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ],
            remainder="drop",
        )
        model = Pipeline([
            ("preprocess", pre),
            ("model", Ridge(alpha=1.0)),
        ])
    else:
        pre = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ],
            remainder="drop",
        )
        base = make_model(model_name)
        model = Pipeline([
            ("preprocess", pre),
            ("model", base),
        ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred


def evaluate_split(df, hourly_df, split_col, train_name, eval_name, horizon, models):
    train_df = df[df[split_col] == train_name].copy()
    eval_df = df[df[split_col] == eval_name].copy()

    if len(train_df) == 0 or len(eval_df) == 0:
        raise ValueError(f"Empty split for {split_col}: train={len(train_df)}, eval={len(eval_df)}")

    print("\n" + "=" * 90)
    print(f"EVALUATING SPLIT: {split_col} | train={train_name} | eval={eval_name}")
    print("=" * 90)
    print("Train rows:", len(train_df))
    print("Eval rows :", len(eval_df))

    X_train, y_train = build_xy(train_df, hourly_df, horizon)
    X_eval, y_eval = build_xy(eval_df, hourly_df, horizon)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_eval :", X_eval.shape, "y_eval :", y_eval.shape)

    results = []

    # train-mean baseline
    pred_mean = train_mean_predict(y_train, len(y_eval))
    m = per_target_metrics(y_eval, pred_mean, horizon)
    results.append({
        "model": "train_mean",
        "split_col": split_col,
        "train_name": train_name,
        "eval_name": eval_name,
        "PM2.5_R2": m["PM2.5_all_horizons"]["R2"],
        "PM10_R2": m["PM10_all_horizons"]["R2"],
        "Avg_R2": m["average_R2"],
        "PM2.5_RMSE": m["PM2.5_all_horizons"]["RMSE"],
        "PM10_RMSE": m["PM10_all_horizons"]["RMSE"],
        "Avg_RMSE": m["average_RMSE"],
        "details": m,
    })

    # persistence baseline
    pred_persist = persistence_predict(X_eval, horizon)
    m = per_target_metrics(y_eval, pred_persist, horizon)
    results.append({
        "model": "persistence",
        "split_col": split_col,
        "train_name": train_name,
        "eval_name": eval_name,
        "PM2.5_R2": m["PM2.5_all_horizons"]["R2"],
        "PM10_R2": m["PM10_all_horizons"]["R2"],
        "Avg_R2": m["average_R2"],
        "PM2.5_RMSE": m["PM2.5_all_horizons"]["RMSE"],
        "PM10_RMSE": m["PM10_all_horizons"]["RMSE"],
        "Avg_RMSE": m["average_RMSE"],
        "details": m,
    })

    # ML models
    for model_name in models:
        print(f"\nTraining model: {model_name}")
        pred = fit_predict_model(model_name, X_train, y_train, X_eval)
        m = per_target_metrics(y_eval, pred, horizon)
        results.append({
            "model": model_name,
            "split_col": split_col,
            "train_name": train_name,
            "eval_name": eval_name,
            "PM2.5_R2": m["PM2.5_all_horizons"]["R2"],
            "PM10_R2": m["PM10_all_horizons"]["R2"],
            "Avg_R2": m["average_R2"],
            "PM2.5_RMSE": m["PM2.5_all_horizons"]["RMSE"],
            "PM10_RMSE": m["PM10_all_horizons"]["RMSE"],
            "Avg_RMSE": m["average_RMSE"],
            "details": m,
        })

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-manifest", required=True)
    ap.add_argument("--hourly-base", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--models", nargs="+", default=["ridge", "rf", "extratrees"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.forecast_manifest)
    hourly_df = pd.read_csv(args.hourly_base)

    all_results = []

    split_specs = [
        ("split_random_forecast", "train", "test"),
        ("split_twofold_forecast", "fold1_train", "fold1_test"),
        ("split_chrono_forecast", "train", "test"),
    ]

    for split_col, train_name, eval_name in split_specs:
        res = evaluate_split(
            df=df,
            hourly_df=hourly_df,
            split_col=split_col,
            train_name=train_name,
            eval_name=eval_name,
            horizon=args.horizon,
            models=args.models,
        )
        all_results.extend(res)

    summary_rows = []
    detail_json = {}

    for r in all_results:
        key = f"{r['split_col']}__{r['eval_name']}__{r['model']}"
        detail_json[key] = r["details"]

        summary_rows.append({
            "model": r["model"],
            "split_col": r["split_col"],
            "train_name": r["train_name"],
            "eval_name": r["eval_name"],
            "PM2.5_R2": r["PM2.5_R2"],
            "PM10_R2": r["PM10_R2"],
            "Avg_R2": r["Avg_R2"],
            "PM2.5_RMSE": r["PM2.5_RMSE"],
            "PM10_RMSE": r["PM10_RMSE"],
            "Avg_RMSE": r["Avg_RMSE"],
        })

    summary = pd.DataFrame(summary_rows)

    csv_path = out_dir / "forecast_numerical_baseline_summary.csv"
    md_path = out_dir / "forecast_numerical_baseline_summary.md"
    json_path = out_dir / "forecast_numerical_baseline_details.json"

    summary.to_csv(csv_path, index=False)
    md_path.write_text(summary.to_markdown(index=False))
    json_path.write_text(json.dumps(detail_json, indent=2))

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(csv_path)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
