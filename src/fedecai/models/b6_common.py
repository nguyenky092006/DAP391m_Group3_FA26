"""Shared contracts for B6 model training."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    """Capture required experiment lineage without changing repository state."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return {"git_commit": commit, "worktree_dirty": dirty}


def register_experiment(root: Path, row: dict[str, Any]) -> None:
    """Append one completed/failed run to the shared versionable registry."""
    path = root / "logs/experiment_registry.csv"
    fields = [
        "experiment_id", "timestamp", "step", "rq", "model", "variant",
        "dataset_version", "split_version", "feature_config", "model_config",
        "seed", "git_commit", "hardware", "train_time_sec",
        "inference_latency_ms", "model_size_mb", "overall_macro_f1",
        "worst_accent_macro_f1", "accent_gap", "status", "checkpoint_path",
        "prediction_path", "notes",
    ]
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fields:
                raise ValueError("Experiment registry schema does not match B6 contract")
            existing = list(reader)
    value = {field: row.get(field, "") for field in fields}
    value["timestamp"] = value["timestamp"] or date.today().isoformat()
    existing.append(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "seed", "split_version", "emotion_order", "feature_contract", "models"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"B6 config missing fields: {sorted(missing)}")
    return value


def load_b6_table(
    feature_path: Path, manifest_path: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_parquet(feature_path)
    manifest = pd.read_csv(
        manifest_path,
        usecols=[
            "sample_id", "speaker_id", "accent_clean", "emotion", "split",
            "split_version", "cv_fold", "audio_sha256",
        ],
        dtype={"sample_id": "string", "speaker_id": "string", "cv_fold": "string"},
    )
    eligible = manifest[manifest["split"] != "excluded"].copy()
    if len(eligible) != 4992:
        raise ValueError(f"Expected 4,992 eligible manifest rows, found {len(eligible):,}")
    if set(eligible["split_version"]) != {config["split_version"]}:
        raise ValueError("Manifest split version does not match B6 config")
    if features["sample_id"].duplicated().any() or eligible["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique in features and eligible manifest")
    if set(features["sample_id"].astype(str)) != set(eligible["sample_id"].astype(str)):
        raise ValueError("B4 feature rows and B5 eligible sample IDs do not match")

    metadata = {"speaker_id", "accent", "emotion", "audio_sha256"}
    feature_base = features.drop(columns=[name for name in metadata if name in features])
    joined = eligible.merge(feature_base, on="sample_id", how="inner", validate="one_to_one")
    prefixes = tuple(config["feature_contract"]["numeric_prefixes"])
    columns = [
        name
        for name in joined.columns
        if name.startswith(prefixes) and pd.api.types.is_numeric_dtype(joined[name])
    ]
    expected = int(config["feature_contract"]["expected_features"])
    if len(columns) != expected:
        raise ValueError(f"Expected {expected} handcrafted features, found {len(columns)}")
    matrix = joined[columns].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("B6 handcrafted matrix contains NaN or infinity")
    if not joined.loc[joined["split"] != "train", "cv_fold"].isna().all():
        raise ValueError("Only training rows may have grouped CV fold labels")
    if joined.loc[joined["split"] == "train", "cv_fold"].isna().any():
        raise ValueError("Every training row must have a grouped CV fold label")
    return joined, columns


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accents: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    overall = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "uar": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {
            label: float(score)
            for label, score in zip(
                labels,
                f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
            )
        },
    }
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    overall["per_class"] = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int((y_true == label).sum()),
        }
        for index, label in enumerate(labels)
    }
    overall["confusion_matrix"] = {
        "labels": labels,
        "values": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }
    per_accent = {}
    for accent in sorted(set(accents)):
        mask = accents == accent
        per_accent[str(accent)] = {
            "samples": int(mask.sum()),
            "macro_f1": float(
                f1_score(y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0)
            ),
            "uar": float(
                recall_score(y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0)
            ),
            "per_emotion_f1": {
                label: float(score)
                for label, score in zip(
                    labels,
                    f1_score(
                        y_true[mask], y_pred[mask], labels=labels,
                        average=None, zero_division=0,
                    ),
                )
            },
        }
    accent_scores = [item["macro_f1"] for item in per_accent.values()]
    return {
        "overall": overall,
        "per_accent": per_accent,
        "worst_accent_macro_f1": float(min(accent_scores)),
        "best_accent_macro_f1": float(max(accent_scores)),
        "accent_macro_f1_gap": float(max(accent_scores) - min(accent_scores)),
        "accent_macro_f1_std": float(np.std(accent_scores, ddof=0)),
    }


def prediction_rows(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    *,
    experiment_id: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, (_, source) in enumerate(frame.iterrows()):
        row = {
            "experiment_id": experiment_id,
            "model": model,
            "seed": seed,
            "sample_id": source["sample_id"],
            "speaker_id": source["speaker_id"],
            "accent": source["accent_clean"],
            "y_true": source["emotion"],
            "y_pred": predicted[index],
            "split": source.get("split"),
        }
        row.update({f"probability_{label}": float(probabilities[index, offset]) for offset, label in enumerate(labels)})
        rows.append(row)
    return rows


def write_json_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
