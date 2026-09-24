"""Data, training, checkpoint, metric, and reporting utilities for M1-M4."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pitchfusion.features.multi_prosody import CHANNELS, load_prosody_artifact
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "seed", "split_version", "emotion_order",
        "feature_contract", "models",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"Model config missing fields: {sorted(missing)}")
    return value


def load_m1_table(
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
        raise ValueError("Manifest split version does not match the M1 config")
    if features["sample_id"].duplicated().any() or eligible["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique in features and eligible manifest")
    if set(features["sample_id"].astype(str)) != set(eligible["sample_id"].astype(str)):
        raise ValueError("M1 feature rows and eligible manifest sample IDs do not match")
    metadata = {"speaker_id", "accent", "emotion", "audio_sha256"}
    feature_base = features.drop(columns=[name for name in metadata if name in features])
    joined = eligible.merge(feature_base, on="sample_id", how="inner", validate="one_to_one")
    prefixes = tuple(config["feature_contract"]["numeric_prefixes"])
    columns = [
        name for name in joined.columns
        if name.startswith(prefixes) and pd.api.types.is_numeric_dtype(joined[name])
    ]
    expected = int(config["feature_contract"]["expected_features"])
    if len(columns) != expected:
        raise ValueError(f"Expected {expected} handcrafted features, found {len(columns)}")
    matrix = joined[columns].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("M1 handcrafted matrix contains NaN or infinity")
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
        "uar": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {
            label: float(score)
            for label, score in zip(
                labels,
                f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
            )
        },
    }
    per_accent: dict[str, dict[str, float | int]] = {}
    for accent in sorted(set(accents)):
        mask = accents == accent
        per_accent[str(accent)] = {
            "samples": int(mask.sum()),
            "macro_f1": float(f1_score(y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0)),
            "uar": float(recall_score(y_true[mask], y_pred[mask], labels=labels, average="macro", zero_division=0)),
        }
    accent_scores = [float(item["macro_f1"]) for item in per_accent.values()]
    return {
        "overall": overall,
        "per_accent": per_accent,
        "worst_accent_macro_f1": min(accent_scores),
        "accent_macro_f1_gap": max(accent_scores) - min(accent_scores),
    }


def prediction_rows(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (_, source) in enumerate(frame.iterrows()):
        row: dict[str, Any] = {
            "sample_id": source["sample_id"],
            "speaker_id": source["speaker_id"],
            "accent": source["accent_clean"],
            "true_emotion": source["emotion"],
            "predicted_emotion": predicted[index],
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_wav(value: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(value), "rb") as stream:
        if stream.getframerate() != 16_000 or stream.getnchannels() != 1:
            raise ValueError("Models require mono 16 kHz PCM audio")
        width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    if width == 2:
        signal = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        source = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        packed = (
            source[:, 0].astype(np.int32)
            | source[:, 1].astype(np.int32) << 8
            | source[:, 2].astype(np.int32) << 16
        )
        packed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        signal = packed.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {width}")
    return np.ascontiguousarray(signal)


def _interpolate_channels(values: np.ndarray, target: int) -> np.ndarray:
    if values.shape[1] == target:
        return values.astype(np.float32, copy=False)
    source_positions = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, target, dtype=np.float32)
    return np.stack(
        [np.interp(target_positions, source_positions, row) for row in values],
        axis=0,
    ).astype(np.float32)


class AudioDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        audio_bytes: list[bytes],
        labels: np.ndarray,
        *,
        maximum_samples: int,
        prosody_paths: dict[str, Path] | None = None,
        prosody_mean: np.ndarray | None = None,
        prosody_std: np.ndarray | None = None,
        prosody_channels: int = 0,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.audio_bytes = audio_bytes
        self.labels = labels
        self.maximum_samples = maximum_samples
        self.prosody_paths = prosody_paths
        self.prosody_mean = prosody_mean
        self.prosody_std = prosody_std
        self.prosody_channels = prosody_channels

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray | None, int]:
        row = self.frame.iloc[index]
        audio = decode_wav(self.audio_bytes[int(row["source_row_index"])])
        audio = audio[: self.maximum_samples]
        audio = (audio - audio.mean()) / max(float(audio.std()), 1e-6)
        prosody = None
        if self.prosody_paths is not None:
            sample_id = str(row["sample_id"])
            values = load_prosody_artifact(self.prosody_paths[sample_id])
            values = values[: self.prosody_channels]
            if self.prosody_mean is None or self.prosody_std is None:
                raise RuntimeError("Prosody normalization statistics are missing")
            mean = self.prosody_mean[: self.prosody_channels, None]
            std = self.prosody_std[: self.prosody_channels, None]
            values = (values - mean) / std
            prosody = _interpolate_channels(values, len(audio))
        return audio, prosody, int(self.labels[index])


def collate_audio(
    batch: list[tuple[np.ndarray, np.ndarray | None, int]],
) -> tuple[torch.Tensor, ...]:
    maximum = max(len(row[0]) for row in batch)
    audio = torch.zeros(len(batch), maximum, dtype=torch.float32)
    mask = torch.zeros(len(batch), maximum, dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    has_prosody = batch[0][1] is not None
    prosody = None
    if has_prosody:
        channels = int(batch[0][1].shape[0])  # type: ignore[union-attr]
        prosody = torch.zeros(len(batch), channels, maximum, dtype=torch.float32)
    for index, (audio_value, prosody_value, label) in enumerate(batch):
        length = len(audio_value)
        audio[index, :length] = torch.from_numpy(audio_value.copy())
        mask[index, :length] = 1
        labels[index] = label
        if prosody is not None and prosody_value is not None:
            prosody[index, :, :length] = torch.from_numpy(prosody_value.copy())
    if prosody is None:
        return audio, mask, labels
    return audio, prosody, mask, labels


def load_experiment_data(
    manifest_path: Path,
    raw_parquet_path: Path,
    emotion_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[bytes], LabelEncoder]:
    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": "string", "speaker_id": "string"},
    )
    frame = frame[frame["split"] != "excluded"].copy()
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Both train and validation partitions are required")
    train_speakers = set(train["speaker_id"].astype(str))
    validation_speakers = set(validation["speaker_id"].astype(str))
    if train_speakers & validation_speakers:
        raise ValueError("Speaker leakage detected between train and validation")
    audio_column = pq.read_table(raw_parquet_path, columns=["path"])["path"].to_pylist()
    audio_bytes = [row["bytes"] for row in audio_column]
    if int(frame["source_row_index"].max()) >= len(audio_bytes):
        raise ValueError("Manifest source_row_index exceeds the raw dataset")
    encoder = LabelEncoder().fit(emotion_order)
    unknown = set(frame["emotion"].astype(str)) - set(encoder.classes_)
    if unknown:
        raise ValueError(f"Unknown emotion labels: {sorted(unknown)}")
    return train, validation, audio_bytes, encoder


def load_prosody_contract(
    root: Path, index_path: Path, statistics_path: Path
) -> tuple[dict[str, Path], np.ndarray, np.ndarray]:
    index = pd.read_csv(index_path, dtype={"sample_id": "string"})
    if index["sample_id"].duplicated().any():
        raise ValueError("Prosody index sample_id must be unique")
    paths = {
        str(row.sample_id): root / str(row.artifact_path)
        for row in index.itertuples(index=False)
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prosody artifact: {missing[0]}")
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if tuple(statistics["channels"]) != CHANNELS:
        raise ValueError("Prosody normalization channel order is invalid")
    if statistics.get("fit_split") != "train" or statistics.get(
        "validation_and_test_used_for_fit"
    ):
        raise ValueError("Prosody normalization must be fitted on train only")
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)
    if mean.shape != (5,) or std.shape != (5,) or np.any(std <= 0):
        raise ValueError("Invalid prosody normalization statistics")
    return paths, mean, std


def forward_batch(
    model: nn.Module, batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    values = [value.to(device, non_blocking=True) for value in batch]
    if len(values) == 3:
        audio, mask, labels = values
        logits = model(audio, mask)
    else:
        audio, prosody, mask, labels = values
        logits = model(audio, prosody, mask)
    return logits, labels


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits, labels = forward_batch(model, batch, device)
                loss = criterion(logits, labels)
            loss_sum += float(loss.item()) * len(labels)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(probabilities),
        loss_sum / len(loader.dataset),
    )


@dataclass
class TrainingResult:
    history: list[dict[str, Any]]
    best_epoch: int
    best_score: float
    runtime_seconds: float


def train_with_resume(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    training: dict[str, Any],
    train_labels: np.ndarray,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, str],
    *,
    force: bool,
) -> TrainingResult:
    model.to(device)
    counts = np.bincount(train_labels, minlength=4).astype(np.float32)
    weights = len(train_labels) / (4 * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(training["mixed_precision"])
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    previous_runtime = 0.0
    if checkpoint_path.exists() and not force:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("metadata") != checkpoint_metadata:
            raise ValueError(
                "Checkpoint inputs/configuration changed; use --force for a clean run"
            )
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        history = saved["history"]
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved["best_score"])
        best_epoch = int(saved["best_epoch"])
        best_state = saved["best_state"]
        previous_runtime = float(saved.get("runtime_seconds", 0.0))
        random.setstate(saved["python_random_state"])
        np.random.set_state(saved["numpy_random_state"])
        torch.set_rng_state(saved["torch_random_state"])
        if device.type == "cuda" and saved.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_random_state"])
        print(
            f"resumed checkpoint {checkpoint_path} at epoch {start_epoch - 1}",
            flush=True,
        )
    accumulation = int(training["gradient_accumulation_steps"])
    started = time.perf_counter()
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            with torch.autocast(
                device_type=device.type, enabled=scaler.is_enabled()
            ):
                logits, labels = forward_batch(model, batch, device)
                loss = criterion(logits, labels) / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.item()) * accumulation * len(labels)

        predicted, _, validation_loss = evaluate(model, validation_loader, device)
        truth = np.concatenate(
            [batch[-1].numpy() for batch in validation_loader]
        )
        score = float(f1_score(truth, predicted, average="macro", zero_division=0))
        uar = float(recall_score(truth, predicted, average="macro", zero_division=0))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader.dataset),
            "validation_loss": validation_loss,
            "validation_macro_f1": score,
            "validation_uar": uar,
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "metadata": checkpoint_metadata,
                "runtime_seconds": previous_runtime + time.perf_counter() - started,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        epoch_seconds = time.perf_counter() - epoch_started
        current_run_seconds = time.perf_counter() - started
        epochs_completed_this_run = epoch - start_epoch + 1
        remaining_epochs = int(training["epochs"]) - epoch
        eta_seconds = (
            current_run_seconds / epochs_completed_this_run * remaining_epochs
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"[{checkpoint_metadata['model']}] epoch {epoch}/{training['epochs']} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={validation_loss:.4f} | macro_f1={score:.4f} | "
            f"uar={uar:.4f} | best={best_score:.4f}@{best_epoch} | "
            f"lr={learning_rate:.2e} | epoch_time={epoch_seconds:.1f}s | "
            f"elapsed={previous_runtime + current_run_seconds:.1f}s | "
            f"eta={eta_seconds:.1f}s | checkpoint={checkpoint_path}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_score=best_score,
        runtime_seconds=previous_runtime + time.perf_counter() - started,
    )


def publish_neural_result(
    *,
    root: Path,
    model_name: str,
    model: nn.Module,
    model_path: Path,
    report_path: Path,
    validation: pd.DataFrame,
    validation_loader: DataLoader,
    encoder: LabelEncoder,
    device: torch.device,
    config: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    result: TrainingResult,
    model_scope: str,
    extra_lineage: dict[str, str] | None = None,
) -> None:
    predicted_ids, probabilities, _ = evaluate(model, validation_loader, device)
    predicted = encoder.inverse_transform(predicted_ids)
    labels = list(config["emotion_order"])
    metrics = classification_metrics(
        validation["emotion"].to_numpy(),
        predicted,
        validation["accent_clean"].to_numpy(),
        labels,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": model_name,
            "classes": list(encoder.classes_),
            "config": config,
        },
        model_path,
    )
    prediction_path = report_path.with_name(
        report_path.stem.replace("_report", "_validation_predictions") + ".csv"
    )
    write_csv_atomic(
        prediction_rows(validation, predicted, probabilities, labels), prediction_path
    )
    lineage = {
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    if extra_lineage:
        lineage.update(extra_lineage)
    write_json_atomic(
        {
            "schema_version": config["schema_version"],
            "model": model_name,
            "scope": model_scope,
            "best_epoch": result.best_epoch,
            "best_validation_macro_f1": result.best_score,
            "validation": metrics,
            "history": result.history,
            "runtime_seconds": result.runtime_seconds,
            "device": str(device),
            "test_samples_used": 0,
            "model_artifact": model_path.relative_to(root).as_posix(),
            "model_artifact_sha256": sha256_file(model_path),
            "lineage": lineage,
        },
        report_path,
    )
