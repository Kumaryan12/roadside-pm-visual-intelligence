from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("experiments/traqid_pretraining_v1/reports/subset1_paper_replication")
FIG_DIR = Path("experiments/traqid_pretraining_v1/figures/subset1_paper_replication")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics_json(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def add_row_from_json(
    rows,
    *,
    experiment_id,
    model_family,
    input_type,
    split_protocol,
    leakage_risk,
    json_path,
    fallback,
    interpretation,
):
    metrics = load_metrics_json(json_path)

    if metrics is None:
        row = fallback.copy()
        row["source"] = "manual_fallback_from_logs"
    else:
        row = {
            "mean_val_rmse": safe_get(metrics, "baseline", "val", "RMSE"),
            "mean_test_rmse": safe_get(metrics, "baseline", "test", "RMSE"),
            "model_val_rmse": safe_get(metrics, "val", "RMSE"),
            "model_test_rmse": safe_get(metrics, "test", "RMSE"),
            "model_val_mae": safe_get(metrics, "val", "MAE"),
            "model_test_mae": safe_get(metrics, "test", "MAE"),
            "model_val_r2": safe_get(metrics, "val", "R2"),
            "model_test_r2": safe_get(metrics, "test", "R2"),
            "model_val_pearson": safe_get(metrics, "val", "Pearson"),
            "model_test_pearson": safe_get(metrics, "test", "Pearson"),
            "model_val_spearman": safe_get(metrics, "val", "Spearman"),
            "model_test_spearman": safe_get(metrics, "test", "Spearman"),
            "best_epoch": safe_get(metrics, "best_checkpoint", "epoch"),
            "source": str(json_path),
        }

    row.update(
        {
            "experiment_id": experiment_id,
            "model_family": model_family,
            "input_type": input_type,
            "split_protocol": split_protocol,
            "leakage_risk": leakage_risk,
            "interpretation": interpretation,
        }
    )

    mean_test = row.get("mean_test_rmse")
    model_test = row.get("model_test_rmse")

    if mean_test is not None and model_test is not None:
        row["test_rmse_delta_vs_mean"] = model_test - mean_test
        row["test_rmse_improvement_pct_vs_mean"] = ((mean_test - model_test) / mean_test) * 100.0
    else:
        row["test_rmse_delta_vs_mean"] = None
        row["test_rmse_improvement_pct_vs_mean"] = None

    rows.append(row)


def save_markdown_table(df: pd.DataFrame, path: Path):
    display_cols = [
        "experiment_id",
        "model_family",
        "input_type",
        "split_protocol",
        "leakage_risk",
        "mean_test_rmse",
        "model_test_rmse",
        "test_rmse_improvement_pct_vs_mean",
        "model_test_spearman",
        "interpretation",
    ]

    temp = df[display_cols].copy()

    numeric_cols = [
        "mean_test_rmse",
        "model_test_rmse",
        "test_rmse_improvement_pct_vs_mean",
        "model_test_spearman",
    ]

    for col in numeric_cols:
        temp[col] = temp[col].apply(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    md = temp.to_markdown(index=False)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Subset 1: Paper-Style TRAQID Replication Results\n\n")
        f.write(md)
        f.write("\n\n")
        f.write("## Main conclusion\n\n")
        f.write(
            "Random-split performance is strong, but it is not a reliable final estimate "
            "because overlapping or highly similar temporal samples may leak across train/test. "
            "The purged-block split is the fairer within-date diagnostic. The date-safe split is "
            "the strictest generalization test across unseen dates.\n"
        )


def save_bar_chart(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["model_test_rmse"])

    labels = plot_df["experiment_id"].tolist()
    values = plot_df["model_test_rmse"].astype(float).tolist()

    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Test RMSE")
    plt.title("Subset 1 TRAQID PM2.5: Test RMSE by Experiment")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "subset1_test_rmse_comparison.png", dpi=180)
    plt.close()

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["model_test_spearman"])

    labels = plot_df["experiment_id"].tolist()
    values = plot_df["model_test_spearman"].astype(float).tolist()

    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Test Spearman")
    plt.title("Subset 1 TRAQID PM2.5: Test Spearman by Experiment")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "subset1_test_spearman_comparison.png", dpi=180)
    plt.close()


def main():
    rows = []

    # 1. Single-image CNN, date-safe split.
    add_row_from_json(
        rows,
        experiment_id="single_cnn_date_safe",
        model_family="MobileNetV2 supervised CNN",
        input_type="single front image",
        split_protocol="date-safe",
        leakage_risk="low",
        json_path="experiments/traqid_pretraining_v1/reports/supervised_cnn_pm25_date_safe_front/metrics_mobilenetv2_front.json",
        fallback={
            "mean_val_rmse": 39.111506794222066,
            "mean_test_rmse": 41.605886617128164,
            "model_val_rmse": 44.21522569439036,
            "model_test_rmse": 48.60691846262798,
            "model_val_mae": 28.008848372637516,
            "model_test_mae": 35.9133356531171,
            "model_val_r2": -0.4306207293096156,
            "model_test_r2": -0.38033941629777357,
            "model_val_pearson": 0.07500714880065429,
            "model_test_pearson": 0.024894111349179744,
            "model_val_spearman": 0.13616096074655007,
            "model_test_spearman": 0.026691551863153022,
            "best_epoch": 7,
        },
        interpretation="Fails strict unseen-date generalization.",
    )

    # 2. Single-image CNN, random split.
    # Replace fallback with your JSON if you saved it separately.
    add_row_from_json(
        rows,
        experiment_id="single_cnn_random",
        model_family="MobileNetV2 supervised CNN",
        input_type="single front image",
        split_protocol="random observation split",
        leakage_risk="medium/high",
        json_path="experiments/traqid_pretraining_v1/reports/supervised_cnn_pm25_random_front/metrics_mobilenetv2_front.json",
        fallback={
            "mean_val_rmse": 47.92,
            "mean_test_rmse": 48.71,
            "model_val_rmse": 43.30,
            "model_test_rmse": 43.99,
            "model_val_mae": None,
            "model_test_mae": None,
            "model_val_r2": None,
            "model_test_r2": None,
            "model_val_pearson": None,
            "model_test_pearson": None,
            "model_val_spearman": 0.431,
            "model_test_spearman": 0.418,
            "best_epoch": None,
        },
        interpretation="Learns within-distribution visual signal, but random split is not final evidence.",
    )

    # 3. T=7 GRU, date-safe split.
    add_row_from_json(
        rows,
        experiment_id="t7_gru_date_safe",
        model_family="MobileNetV2 embeddings + GRU",
        input_type="T=7 front image embeddings",
        split_protocol="date-safe",
        leakage_risk="low",
        json_path="experiments/traqid_pretraining_v1/reports/t7_embedding_gru/metrics_t7_embedding_gru.json",
        fallback={
            "mean_val_rmse": 39.17644352288247,
            "mean_test_rmse": 41.62366640491317,
            "model_val_rmse": 44.11733752588119,
            "model_test_rmse": 46.81823935049985,
            "model_val_mae": 28.486588246535987,
            "model_test_mae": 34.83295958245556,
            "model_val_r2": -0.42264979404425107,
            "model_test_r2": -0.2788334752772301,
            "model_val_pearson": -0.02739528051960417,
            "model_test_pearson": -0.06832864333181762,
            "model_val_spearman": 0.08899985888550685,
            "model_test_spearman": -0.03562987726007946,
            "best_epoch": 23,
        },
        interpretation="Temporal model also fails strict unseen-date generalization.",
    )

    # 4. T=7 GRU, random split.
    add_row_from_json(
        rows,
        experiment_id="t7_gru_random",
        model_family="MobileNetV2 embeddings + GRU",
        input_type="T=7 front image embeddings",
        split_protocol="random sequence split",
        leakage_risk="very high",
        json_path="experiments/traqid_pretraining_v1/reports/t7_embedding_gru_random_debug/metrics_t7_embedding_gru.json",
        fallback={
            "mean_val_rmse": 47.28811772837429,
            "mean_test_rmse": 48.13320401452184,
            "model_val_rmse": 22.372659832261167,
            "model_test_rmse": 22.718788005912867,
            "model_val_mae": 12.733208068403375,
            "model_test_mae": 12.998825506632587,
            "model_val_r2": 0.7647192826140125,
            "model_test_r2": 0.7653739628617173,
            "model_val_pearson": 0.8907719685092172,
            "model_test_pearson": 0.8921989868586658,
            "model_val_spearman": 0.8867187996124036,
            "model_test_spearman": 0.8927675365292096,
            "best_epoch": 13,
        },
        interpretation="Very strong, but likely inflated by overlapping-window leakage.",
    )

    # 5. T=7 GRU, purged block split.
    add_row_from_json(
        rows,
        experiment_id="t7_gru_purged_block",
        model_family="MobileNetV2 embeddings + GRU",
        input_type="T=7 front image embeddings",
        split_protocol="purged within-date block split",
        leakage_risk="low/medium",
        json_path="experiments/traqid_pretraining_v1/reports/t7_embedding_gru_purged_block/metrics_t7_embedding_gru.json",
        fallback={
            "mean_val_rmse": 63.94315538973628,
            "mean_test_rmse": 44.776405310144,
            "model_val_rmse": 66.1442152074536,
            "model_test_rmse": 44.53816353309236,
            "model_val_mae": 41.03343198722244,
            "model_test_mae": 29.37441537696017,
            "model_val_r2": -0.2823371524038172,
            "model_test_r2": -0.04863695082714781,
            "model_val_pearson": -0.042817598207712776,
            "model_test_pearson": 0.20760865880674745,
            "model_val_spearman": 0.017109241559537292,
            "model_test_spearman": 0.18933634678219866,
            "best_epoch": 17,
        },
        interpretation="Fairer test; weak positive ranking signal but RMSE nearly equals mean baseline.",
    )

    add_row_from_json(
    rows,
    experiment_id="t7_gru_supervised_front_purged_block",
    model_family="TRAQID-supervised MobileNetV2 embeddings + GRU",
    input_type="T=7 front PM-aware embeddings",
    split_protocol="purged within-date block split",
    leakage_risk="low/medium",
    json_path="experiments/traqid_pretraining_v1/reports/t7_supervised_front_gru_purged_block/metrics_t7_embedding_gru.json",
    fallback={
        "mean_val_rmse": 63.94315538973628,
        "mean_test_rmse": 44.776405310144,
        "model_val_rmse": 62.61416736330929,
        "model_test_rmse": 41.647467728152776,
        "model_val_mae": 38.905317382680046,
        "model_test_mae": 26.711150640053802,
        "model_val_r2": -0.14911556836976847,
        "model_test_r2": 0.08306669730587346,
        "model_val_pearson": 0.14036954995407683,
        "model_test_pearson": 0.359596593620724,
        "model_val_spearman": 0.1813350934445621,
        "model_test_spearman": 0.36123066410240273,
        "best_epoch": 17,
    },
    interpretation="Best fair result so far; PM-aware visual pretraining improves purged-block generalization.",
)
    
    add_row_from_json(
    rows,
    experiment_id="t7_gru_supervised_front_rear_mean_purged_block",
    model_family="TRAQID-supervised MobileNetV2 mean-fused embeddings + GRU",
    input_type="T=7 front+rear mean PM-aware embeddings",
    split_protocol="purged within-date block split",
    leakage_risk="low/medium",
    json_path="experiments/traqid_pretraining_v1/reports/t7_supervised_front_rear_mean_gru_purged_block/metrics_t7_embedding_gru.json",
    fallback={
        "mean_val_rmse": 63.9431553897363,
        "mean_test_rmse": 44.77640531014401,
        "model_val_rmse": 59.75270546256511,
        "model_test_rmse": 39.79804844394646,
        "model_val_mae": 35.70510350129543,
        "model_test_mae": 25.342067122613308,
        "model_val_r2": -0.046486516116898136,
        "model_test_r2": 0.1626942022267166,
        "model_val_pearson": 0.30603503825128653,
        "model_test_pearson": 0.44165486031600604,
        "model_val_spearman": 0.33294856903320424,
        "model_test_spearman": 0.3627935473435461,
        "best_epoch": 17,
    },
    interpretation="Best fair result so far; mean-fused front/rear PM-aware embeddings improve both RMSE and rank correlation.",
)

    df = pd.DataFrame(rows)

    preferred_cols = [
        "experiment_id",
        "model_family",
        "input_type",
        "split_protocol",
        "leakage_risk",
        "mean_val_rmse",
        "mean_test_rmse",
        "model_val_rmse",
        "model_test_rmse",
        "test_rmse_delta_vs_mean",
        "test_rmse_improvement_pct_vs_mean",
        "model_val_spearman",
        "model_test_spearman",
        "model_val_r2",
        "model_test_r2",
        "best_epoch",
        "interpretation",
        "source",
    ]

    df = df[preferred_cols]

    csv_path = OUT_DIR / "subset1_comparison_table.csv"
    md_path = OUT_DIR / "subset1_comparison_table.md"
    json_path = OUT_DIR / "subset1_comparison_table.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    save_markdown_table(df, md_path)
    save_bar_chart(df)

    print("=" * 90)
    print("SUBSET 1 COMPARISON TABLE CREATED")
    print("=" * 90)
    print(df.to_string(index=False))

    print("\nSaved:")
    print(" -", csv_path)
    print(" -", md_path)
    print(" -", json_path)
    print(" -", FIG_DIR / "subset1_test_rmse_comparison.png")
    print(" -", FIG_DIR / "subset1_test_spearman_comparison.png")

    print("\nMain interpretation:")
    print(
        "Random split results are strong but leakage-prone. "
        "Purged-block results are fairer and show only weak improvement. "
        "Date-safe results remain poor, indicating cross-date domain shift."
    )


if __name__ == "__main__":
    main()