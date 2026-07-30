"""Registered historical stages for the MUMMA-281 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = PROJECT_ROOT / "experiments/mumma_281_pipeline_v1/scripts"


@dataclass(frozen=True)
class Stage:
    name: str
    script: Path
    role: str
    validation_note: str


STAGES = {
    "static_ablation": Stage(
        "static_ablation",
        LEGACY_ROOT / "01_pm25_model_from_final_feature_table.py",
        "Static PM2.5 feature-family ablation.",
        "Historical KFold behavior; not yet the canonical generalized protocol.",
    ),
    "engineered_fair": Stage(
        "engineered_fair",
        LEGACY_ROOT / "02_engineered_fair_pm25_models.py",
        "Fair engineered source-proxy models.",
        "Historical KFold behavior; preserve for result equivalence.",
    ),
    "residual_fusion": Stage(
        "residual_fusion",
        LEGACY_ROOT / "03_residual_fusion_pm25.py",
        "Meteorology/gas base model plus source-proxy residual model.",
        "Historical KFold behavior; preserve for result equivalence.",
    ),
    "fair_optimization": Stage(
        "fair_optimization",
        LEGACY_ROOT / "04_fair_model_optimization.py",
        "Optimize the strongest fair residual-fusion candidates.",
        "Current reported MUMMA-281 candidate; generalized split migration pending.",
    ),
    "temporal_t7": Stage(
        "temporal_t7",
        LEGACY_ROOT / "05_temporal_t7_fair_pm25.py",
        "Historical temporal T7 feature comparison.",
        "Sequence overlap and collection grouping must be audited before promotion.",
    ),
    "temporal_t7_random2fold": Stage(
        "temporal_t7_random2fold",
        LEGACY_ROOT / "05b_temporal_t7_random2fold_fair_pm25.py",
        "Random two-fold temporal comparison.",
        "Benchmark only; not evidence of generalization.",
    ),
    "extract_resnet50": Stage(
        "extract_resnet50",
        LEGACY_ROOT / "06_extract_resnet50_embeddings.py",
        "Extract ResNet50 image embeddings.",
        "Feature producer; verify image-to-sample alignment.",
    ),
    "resnet_t7_random2fold": Stage(
        "resnet_t7_random2fold",
        LEGACY_ROOT / "07_resnet_t7_random2fold_lstm_fair_pm25.py",
        "ResNet50 T7 random two-fold comparison.",
        "Benchmark only; not evidence of generalization.",
    ),
    "audit_pm25": Stage(
        "audit_pm25",
        LEGACY_ROOT / "08_analyse_actual_pm25_data.py",
        "Audit PM2.5 duplicates, spikes, and lag structure.",
        "Diagnostic stage; should precede canonical target construction.",
    ),
}


def validate_stage_registry() -> None:
    missing = [str(stage.script) for stage in STAGES.values() if not stage.script.is_file()]
    if missing:
        raise FileNotFoundError(f"Registered MUMMA-281 scripts are missing: {missing}")

