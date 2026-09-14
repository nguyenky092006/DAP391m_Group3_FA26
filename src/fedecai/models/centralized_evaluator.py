"""Single prediction validator and evaluator for all eight centralized models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fedecai.models.b6_common import classification_metrics, write_json_atomic


EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]
ACCENT_GROUPS = ["north", "central", "south"]
PROBABILITY_COLUMNS = [f"probability_{label}" for label in EMOTION_LABELS]
PREDICTION_COLUMNS = [
    "experiment_id", "model", "seed", "split", "sample_id", "speaker_id",
    "accent", "y_true", "y_pred", *PROBABILITY_COLUMNS,
]


def fixed_test_ids(manifest_path: Path, split_version: str) -> list[str]:
    manifest = pd.read_csv(
        manifest_path, usecols=["sample_id", "split", "split_version"],
        dtype={"sample_id": "string"},
    )
    test = manifest[
        (manifest["split"] == "test")
        & (manifest["split_version"] == split_version)
    ]
    if len(test) != 665 or test["sample_id"].duplicated().any():
        raise ValueError("Fixed-test manifest must contain 665 unique sample IDs")
    return sorted(test["sample_id"].astype(str))


def validate_predictions(
    frame: pd.DataFrame,
    *,
    expected_sample_ids: Iterable[str],
    expected_split: str = "test",
    require_argmax_consistency: bool = True,
) -> None:
    missing_columns = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Prediction schema missing columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Prediction file is empty")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate sample predictions are forbidden")
    expected = set(map(str, expected_sample_ids))
    actual = set(frame["sample_id"].astype(str))
    if actual != expected or len(frame) != len(expected):
        raise ValueError(
            f"Fixed-test membership mismatch: expected {len(expected)}, got {len(frame)}; "
            f"missing={len(expected - actual)}, unexpected={len(actual - expected)}"
        )
    if set(frame["split"]) != {expected_split}:
        raise ValueError(f"Predictions must contain split={expected_split} only")
    for name in ("y_true", "y_pred"):
        invalid = set(frame[name].astype(str)) - set(EMOTION_LABELS)
        if invalid:
            raise ValueError(f"Invalid {name} labels: {sorted(invalid)}")
    if set(frame["accent"].astype(str)) != set(ACCENT_GROUPS):
        raise ValueError("Predictions must cover exactly North/Central/South accents")
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction probabilities contain NaN or infinity")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probability values must be within [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Prediction probabilities must sum to one")
    if require_argmax_consistency:
        argmax = np.asarray(EMOTION_LABELS)[probabilities.argmax(axis=1)]
        inconsistent = argmax != frame["y_pred"].astype(str).to_numpy()
        if inconsistent.any():
            raise ValueError(f"y_pred disagrees with probability argmax for {int(inconsistent.sum())} rows")


def common_metrics_schema(
    *,
    experiment_id: str,
    model: str,
    seed: int,
    dataset_version: str,
    split_version: str,
    computed: dict[str, Any],
    costs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    costs = costs or {}
    accents = computed["per_accent"]
    result = {
        "experiment_id": experiment_id,
        "model": model,
        "seed": int(seed),
        "dataset_version": dataset_version,
        "split_version": split_version,
        "overall_macro_f1": computed["overall"]["macro_f1"],
        "uar": computed["overall"]["uar"],
        "weighted_f1": computed["overall"]["weighted_f1"],
        "accuracy": computed["overall"]["accuracy"],
        "north_macro_f1": accents["north"]["macro_f1"],
        "central_macro_f1": accents["central"]["macro_f1"],
        "south_macro_f1": accents["south"]["macro_f1"],
        "worst_accent_macro_f1": computed["worst_accent_macro_f1"],
        "best_accent_macro_f1": computed["best_accent_macro_f1"],
        "accent_gap": computed["accent_macro_f1_gap"],
        "accent_std": computed["accent_macro_f1_std"],
        "total_parameters": costs.get("total_parameters"),
        "trainable_parameters": costs.get("trainable_parameters"),
        "model_size_mb": costs.get("model_size_mb"),
        "training_time_sec": costs.get("training_time_sec"),
        "feature_extraction_time_sec": costs.get("feature_extraction_time_sec"),
        "model_training_time_sec": costs.get("model_training_time_sec"),
        "end_to_end_training_pipeline_time_sec": costs.get(
            "end_to_end_training_pipeline_time_sec"
        ),
        "inference_latency_ms": costs.get("inference_latency_ms"),
        "latency_mean_ms": costs.get("latency_mean_ms"),
        "latency_median_ms": costs.get("latency_median_ms"),
        "latency_std_ms": costs.get("latency_std_ms"),
        "latency_p95_ms": costs.get("latency_p95_ms"),
        "per_class": computed["overall"]["per_class"],
        "confusion_matrix": computed["overall"]["confusion_matrix"],
        "per_accent": computed["per_accent"],
        "cost_protocol": costs.get("cost_protocol"),
    }
    return result


def evaluate_prediction_file(
    prediction_path: Path,
    output_path: Path,
    *,
    expected_sample_ids: Iterable[str],
    dataset_version: str,
    split_version: str,
    costs: dict[str, Any] | None = None,
    require_argmax_consistency: bool = True,
    expected_split: str = "test",
) -> dict[str, Any]:
    frame = pd.read_csv(
        prediction_path,
        dtype={"sample_id": "string", "speaker_id": "string"},
    )
    validate_predictions(
        frame,
        expected_sample_ids=expected_sample_ids,
        expected_split=expected_split,
        require_argmax_consistency=require_argmax_consistency,
    )
    experiment_ids = frame["experiment_id"].drop_duplicates().tolist()
    models = frame["model"].drop_duplicates().tolist()
    seeds = frame["seed"].drop_duplicates().tolist()
    if len(experiment_ids) != 1 or len(models) != 1 or len(seeds) != 1:
        raise ValueError("A prediction file must contain one experiment/model/seed")
    computed = classification_metrics(
        frame["y_true"].astype(str).to_numpy(),
        frame["y_pred"].astype(str).to_numpy(),
        frame["accent"].astype(str).to_numpy(),
        EMOTION_LABELS,
    )
    result = common_metrics_schema(
        experiment_id=str(experiment_ids[0]), model=str(models[0]),
        seed=int(seeds[0]), dataset_version=dataset_version,
        split_version=split_version, computed=computed, costs=costs,
    )
    result["prediction_file"] = str(prediction_path.resolve())
    write_json_atomic(result, output_path)
    return result


def validate_common_test_membership(prediction_paths: Iterable[Path]) -> None:
    reference: set[str] | None = None
    for path in prediction_paths:
        frame = pd.read_csv(path, usecols=["sample_id"], dtype={"sample_id": "string"})
        current = set(frame["sample_id"].astype(str))
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError(f"Fixed-test sample IDs differ in {path}")
