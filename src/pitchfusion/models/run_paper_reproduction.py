"""Cleaned-data reproduction of the official ICASSP 2024 Pitch-Fusion code.

This runner intentionally does not share checkpoints, manifests, pitch caches,
or reports with the speaker-disjoint M1-M4 experiment.  It follows the public
author implementation while using the 4,992 rows accepted by this project's
cleaning policy and an utterance-level stratified 80/10/10 split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
)


ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "m3_pitch_fusion_paper_reproduction_clean"
DEFAULT_CONFIG = "configs/models/m3_paper_reproduction_clean_v1.json"
DEFAULT_SPLIT = "data/manifests/paper_reproduction_clean_seed391.csv"
DEFAULT_SPLIT_REPORT = "reports/paper_reproduction/split_report.json"
DEFAULT_PITCH_ROOT = "data/interim/features/paper_reproduction_kaldi_pitch_v1"
DEFAULT_M4_ROOT = "data/interim/features/m4_kaldi_multi_prosody_clean_v1"
DEFAULT_KALDI_EXECUTABLE = ".kaldi-win/Library/bin/compute-kaldi-pitch-feats.exe"
DEFAULT_ARTIFACT_ROOT = "artifacts/paper_reproduction/m3_pitch_fusion"
DEFAULT_REPORT_ROOT = "reports/paper_reproduction"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _speaker_overlap(frame: pd.DataFrame) -> dict[str, int]:
    groups = {
        name: set(part["speaker_id"].astype(str))
        for name, part in frame.groupby("split")
    }
    return {
        "train_validation": len(groups["train"] & groups["validation"]),
        "train_test": len(groups["train"] & groups["test"]),
        "validation_test": len(groups["validation"] & groups["test"]),
    }


def build_clean_stratified_split(
    clean_manifest: Path,
    output_path: Path,
    report_path: Path,
    config: dict[str, Any],
    *,
    force: bool,
) -> pd.DataFrame:
    data_config = config["data"]
    source = pd.read_csv(
        clean_manifest,
        dtype={"sample_id": "string", "speaker_id": "string"},
    )
    eligible = source[_as_bool(source[data_config["eligible_column"]])].copy()
    expected = int(data_config["expected_eligible_rows"])
    if len(eligible) != expected:
        raise ValueError(f"Expected {expected:,} cleaned rows, found {len(eligible):,}")
    if eligible["sample_id"].duplicated().any():
        raise ValueError("Cleaned sample_id values must be unique")
    if eligible["audio_sha256"].duplicated().any():
        raise ValueError("Cleaned input still contains duplicate waveform hashes")
    if set(eligible["emotion"].astype(str)) != set(config["emotion_order"]):
        raise ValueError("Cleaned input emotion labels do not match the config")

    positions = np.arange(len(eligible))
    train_positions, holdout_positions = train_test_split(
        positions,
        test_size=float(data_config["validation_fraction"])
        + float(data_config["test_fraction"]),
        random_state=int(config["seed"]),
        shuffle=True,
        stratify=eligible["emotion"].to_numpy(),
    )
    validation_positions, test_positions = train_test_split(
        holdout_positions,
        test_size=float(data_config["test_fraction"])
        / (
            float(data_config["validation_fraction"])
            + float(data_config["test_fraction"])
        ),
        random_state=int(config["seed"]),
        shuffle=True,
        stratify=eligible.iloc[holdout_positions]["emotion"].to_numpy(),
    )
    split = np.empty(len(eligible), dtype=object)
    split[train_positions] = "train"
    split[validation_positions] = "validation"
    split[test_positions] = "test"
    eligible["split"] = split
    eligible["split_version"] = config["split_protocol"]
    eligible["split_seed"] = int(config["seed"])
    eligible = eligible.sort_values("source_row_index").reset_index(drop=True)

    if output_path.exists() and not force:
        previous = pd.read_csv(output_path, dtype={"sample_id": "string"})
        old = previous[["sample_id", "split"]].sort_values("sample_id").reset_index(drop=True)
        new = eligible[["sample_id", "split"]].sort_values("sample_id").reset_index(drop=True)
        if not old.equals(new):
            raise ValueError(
                "Existing reproduction split differs from the deterministic split; "
                "pass --force-split only if replacement is intentional"
            )
    else:
        atomic_csv(eligible, output_path)

    counts = (
        eligible.groupby(["split", "emotion"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "validation", "test"], columns=config["emotion_order"])
    )
    for split_name in counts.index:
        if (counts.loc[split_name] == 0).any():
            raise ValueError(f"Missing emotion in {split_name} split")
    report = {
        "schema_version": config["schema_version"],
        "split_protocol": config["split_protocol"],
        "seed": int(config["seed"]),
        "source": clean_manifest.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(clean_manifest),
        "rows": int(len(eligible)),
        "split_counts": {
            name: int(value) for name, value in eligible["split"].value_counts().items()
        },
        "emotion_counts": {
            name: {emotion: int(value) for emotion, value in row.items()}
            for name, row in counts.iterrows()
        },
        "speaker_disjoint_required": False,
        "speaker_overlap": _speaker_overlap(eligible),
        "status": "PASS",
    }
    atomic_json(report, report_path)
    return eligible


def decode_wav(value: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(value), "rb") as stream:
        if stream.getframerate() != 16_000 or stream.getnchannels() != 1:
            raise ValueError("Paper reproduction requires mono 16 kHz PCM audio")
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
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Decoded waveform is empty or non-finite")
    return np.ascontiguousarray(signal)


def write_kaldi_pcm16_wav(audio: np.ndarray, path: Path) -> None:
    """Write a mono 16 kHz PCM16 copy accepted by Kaldi's WaveData reader."""
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Cannot write invalid waveform for Kaldi")
    pcm16 = np.clip(np.rint(audio * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(pcm16.tobytes())


def load_audio_bytes(raw_parquet: Path) -> list[bytes]:
    column = pq.read_table(raw_parquet, columns=["path"])["path"].to_pylist()
    values = [row["bytes"] for row in column]
    if not values or any(value is None for value in values):
        raise ValueError("Raw parquet contains missing embedded audio")
    return values


def parse_kaldi_text_archive(text: str) -> dict[str, np.ndarray]:
    """Parse the text-matrix format emitted by Kaldi's ``ark,t:`` writer."""
    matrices: dict[str, np.ndarray] = {}
    current_key: str | None = None
    current_rows: list[list[float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if current_key is None:
            key, separator, remainder = line.partition(" ")
            if not separator or "[" not in remainder:
                raise ValueError(f"Malformed Kaldi archive at line {line_number}: {line}")
            if key in matrices:
                raise ValueError(f"Duplicate utterance key in Kaldi archive: {key}")
            current_key = key
            payload = remainder.split("[", 1)[1].strip()
        else:
            payload = line
        finished = "]" in payload
        payload = payload.replace("]", " ").strip()
        if payload:
            current_rows.append([float(value) for value in payload.split()])
        if finished:
            if not current_rows:
                raise ValueError(f"Empty Kaldi matrix for utterance: {current_key}")
            matrix = np.asarray(current_rows, dtype=np.float32)
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid Kaldi matrix for utterance: {current_key}")
            matrices[current_key] = matrix
            current_key = None
            current_rows = []
    if current_key is not None:
        raise ValueError(f"Unterminated Kaldi matrix for utterance: {current_key}")
    return matrices


def run_kaldi_feature_batch(
    rows: list[Any],
    audio_bytes: list[bytes],
    kaldi_executable: Path,
    work_root: Path,
    pitch_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Run the official Kaldi executable once for a batch of embedded WAV files."""
    if not kaldi_executable.is_file():
        raise FileNotFoundError(
            f"Kaldi pitch executable not found: {kaldi_executable}. "
            "Run .\\scripts\\setup_paper_reproduction.ps1 -InstallMiniforge first."
        )
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pitch_", dir=str(work_root)) as temporary:
        temporary_root = Path(temporary)
        audio_root = temporary_root / "audio"
        audio_root.mkdir()
        scp_lines: list[str] = []
        expected: set[str] = set()
        for row in rows:
            sample_id = str(row.sample_id)
            if sample_id in expected:
                raise ValueError(f"Duplicate sample_id in Kaldi batch: {sample_id}")
            expected.add(sample_id)
            wav_path = audio_root / f"{sample_id}.wav"
            audio = decode_wav(audio_bytes[int(row.source_row_index)])
            write_kaldi_pcm16_wav(audio, wav_path)
            scp_lines.append(f"{sample_id} audio/{sample_id}.wav")
        (temporary_root / "wav.scp").write_text(
            "\n".join(scp_lines) + "\n", encoding="utf-8"
        )
        command = [
            str(kaldi_executable),
            "--sample-frequency=16000",
            f"--frame-length={float(pitch_config['frame_length_ms']):g}",
            f"--frame-shift={float(pitch_config['frame_shift_ms']):g}",
            "scp:wav.scp",
            "ark,t:pitch.txt",
        ]
        environment = os.environ.copy()
        environment["PATH"] = str(kaldi_executable.parent) + os.pathsep + environment.get(
            "PATH", ""
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            command,
            cwd=temporary_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"Kaldi pitch extraction failed with exit code {result.returncode}: {detail}"
            )
        archive_path = temporary_root / "pitch.txt"
        if not archive_path.is_file():
            raise RuntimeError("Kaldi completed without creating pitch.txt")
        matrices = parse_kaldi_text_archive(archive_path.read_text(encoding="utf-8"))
    if set(matrices) != expected:
        missing = sorted(expected - set(matrices))
        extra = sorted(set(matrices) - expected)
        raise RuntimeError(f"Kaldi output key mismatch: missing={missing}, extra={extra}")
    return matrices


def run_kaldi_pitch_batch(
    rows: list[Any], audio_bytes: list[bytes], kaldi_executable: Path,
    work_root: Path, pitch_config: dict[str, Any]
) -> dict[str, np.ndarray]:
    matrices = run_kaldi_feature_batch(
        rows, audio_bytes, kaldi_executable, work_root, pitch_config
    )
    column = int(pitch_config["kaldi_binary_output_index"])
    return {
        key: np.asarray(matrix[:, column], dtype=np.float32).reshape(-1)
        for key, matrix in matrices.items()
    }


def prepare_pitch_cache(
    manifest: pd.DataFrame,
    audio_bytes: list[bytes],
    pitch_root: Path,
    kaldi_executable: Path,
    config: dict[str, Any],
    *,
    force: bool,
) -> None:
    pitch_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest_rows = list(manifest.itertuples(index=False))
    total = len(manifest_rows)
    missing = [
        row
        for row in manifest_rows
        if force or not (pitch_root / f"{row.sample_id}.npy").is_file()
    ]
    batch_size = int(config["pitch"].get("extraction_batch_size", 128))
    if batch_size < 1:
        raise ValueError("pitch.extraction_batch_size must be positive")
    extracted = 0
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        results = run_kaldi_pitch_batch(
            batch, audio_bytes, kaldi_executable, pitch_root / ".work", config["pitch"]
        )
        for row in batch:
            path = pitch_root / f"{row.sample_id}.npy"
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, results[str(row.sample_id)], allow_pickle=False)
            os.replace(temporary, path)
        extracted += len(batch)
        elapsed = time.perf_counter() - started
        eta = elapsed / extracted * (len(missing) - extracted) if extracted else 0.0
        print(
            f"[kaldi-pitch] extracted={extracted}/{len(missing)} "
            f"total-cache={total} | elapsed={elapsed:.1f}s | eta={eta:.1f}s",
            flush=True,
        )

    index_rows: list[dict[str, Any]] = []
    for position, row in enumerate(manifest_rows, start=1):
        path = pitch_root / f"{row.sample_id}.npy"
        pitch = np.load(path, allow_pickle=False)
        if pitch.ndim != 1 or pitch.size == 0 or not np.isfinite(pitch).all():
            raise ValueError(f"Invalid cached pitch artifact: {path}")
        index_rows.append(
            {
                "sample_id": str(row.sample_id),
                "artifact_path": path.relative_to(ROOT).as_posix(),
                "frames": int(len(pitch)),
                "sha256": sha256_file(path),
            }
        )
    atomic_csv(pd.DataFrame(index_rows), pitch_root / "index.csv")
    atomic_json(
        {
            "status": "PASS",
            "samples": total,
            "extractor": config["pitch"]["extractor"],
            "frame_length_ms": config["pitch"]["frame_length_ms"],
            "frame_shift_ms": config["pitch"]["frame_shift_ms"],
            "kaldi_binary_output_index": config["pitch"]["kaldi_binary_output_index"],
            "normalization": "none",
        },
        pitch_root / "summary.json",
    )


def interpolate_like_author(pitch: np.ndarray, waveform_length: int) -> np.ndarray:
    """Match the exact np.interp construction in the public author script."""
    return np.interp(
        np.linspace(0, waveform_length, waveform_length),
        np.linspace(0, len(pitch), len(pitch)),
        pitch,
    ).astype(np.float32)


class PaperDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        audio_bytes: list[bytes],
        pitch_root: Path,
        label_to_id: dict[str, int],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.audio_bytes = audio_bytes
        self.pitch_root = pitch_root
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, int]:
        row = self.frame.iloc[index]
        audio = decode_wav(self.audio_bytes[int(row["source_row_index"])])
        pitch = np.load(
            self.pitch_root / f"{row['sample_id']}.npy", allow_pickle=False
        ).astype(np.float32, copy=False)
        pitch = interpolate_like_author(pitch, len(audio))
        return audio, pitch, self.label_to_id[str(row["emotion"])]


def collate_paper(
    batch: list[tuple[np.ndarray, np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(row[0]) for row in batch)
    audio = torch.zeros(len(batch), maximum, dtype=torch.float32)
    pitch = torch.zeros(len(batch), maximum, dtype=torch.float32)
    labels = torch.empty(len(batch), dtype=torch.long)
    for index, (audio_value, pitch_value, label) in enumerate(batch):
        length = len(audio_value)
        audio[index, :length] = torch.from_numpy(audio_value.copy())
        pitch[index, :length] = torch.from_numpy(pitch_value.copy())
        labels[index] = label
    return audio, pitch, labels


class OfficialAttentionLayer(nn.Module):
    """Attention block copied semantically from the public author repository."""

    def __init__(self, dimension: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dimension, heads, dropout=dropout)
        self.linear1 = nn.Linear(dimension, dimension * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dimension * 4, dimension)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        src = src.transpose(0, 1)
        tar = tar.transpose(0, 1)
        src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src.transpose(0, 1)


class OfficialPitchFusion(Wav2Vec2PreTrainedModel):
    """Architecture matching ser_pitch_model.py from the official repository."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.pitch_encoder = Wav2Vec2FeatureEncoder(config)
        self.pitch_projection = Wav2Vec2FeatureProjection(config)
        self.attn_a_p = OfficialAttentionLayer(config.hidden_size, 16)
        self.attn_p_a = OfficialAttentionLayer(config.hidden_size, 16)
        self.self_attn = OfficialAttentionLayer(config.hidden_size * 2, 16)
        self.projector = nn.Linear(config.hidden_size * 2, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.post_init()

    def freeze_feature_extractor(self) -> None:
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        audio_input: torch.Tensor,
        pitch_input: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        acoustic = self.wav2vec2(audio_input, attention_mask=None)[0]
        pitch = self.pitch_encoder(pitch_input).transpose(1, 2)
        pitch, _ = self.pitch_projection(pitch)
        a_p_attn = self.attn_a_p(acoustic, pitch)
        p_a_attn = self.attn_p_a(pitch, acoustic)
        attention_output = torch.cat((a_p_attn, p_a_attn), dim=-1)
        attention_output = self.self_attn(attention_output, attention_output)
        attention_output = self.projector(attention_output).mean(dim=1)
        logits = self.classifier(attention_output)
        loss = nn.CrossEntropyLoss()(logits, label) if label is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_model(config: dict[str, Any], allow_download: bool) -> OfficialPitchFusion:
    backbone = config["backbone"]
    snapshot = resolve(backbone["local_snapshot"])
    source = str(snapshot) if snapshot.is_dir() else backbone["model_id"]
    model = OfficialPitchFusion.from_pretrained(
        source,
        revision=None if snapshot.is_dir() else backbone["revision"],
        local_files_only=not allow_download,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.1,
        final_dropout=0.1,
        mask_time_prob=0.05,
        layerdrop=0.1,
        num_labels=4,
        classifier_proj_size=256,
    )
    if backbone["freeze_feature_encoder"]:
        model.freeze_feature_extractor()
    return model


def metrics_from_ids(
    truth: np.ndarray, predicted: np.ndarray, emotion_order: list[str]
) -> dict[str, Any]:
    label_ids = list(range(len(emotion_order)))
    return {
        "samples": int(len(truth)),
        "wa": float(accuracy_score(truth, predicted)),
        "ua": float(
            recall_score(
                truth, predicted, labels=label_ids, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                truth, predicted, labels=label_ids, average="macro", zero_division=0
            )
        ),
        "per_class_f1": {
            label: float(value)
            for label, value in zip(
                emotion_order,
                f1_score(
                    truth, predicted, labels=label_ids, average=None, zero_division=0
                ),
            )
        },
    }


def evaluate(
    model: OfficialPitchFusion,
    loader: DataLoader,
    device: torch.device,
    emotion_order: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for audio, pitch, labels in loader:
            audio = audio.to(device, non_blocking=True)
            pitch = pitch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(audio, pitch).logits
            loss_sum += float(criterion(logits, labels).item()) * len(labels)
            truths.append(labels.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    truth = np.concatenate(truths)
    predicted = np.concatenate(predictions)
    probability = np.concatenate(probabilities)
    return (
        metrics_from_ids(truth, predicted, emotion_order),
        truth,
        predicted,
        probability,
        loss_sum / len(loader.dataset),
    )


def prediction_frame(
    frame: pd.DataFrame,
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    emotion_order: list[str],
) -> pd.DataFrame:
    result = frame[["sample_id", "speaker_id", "accent_clean", "emotion"]].copy()
    result = result.rename(columns={"emotion": "true_emotion", "accent_clean": "accent"})
    result["true_label_id"] = truth
    result["predicted_label_id"] = predicted
    result["predicted_emotion"] = [emotion_order[value] for value in predicted]
    for index, emotion in enumerate(emotion_order):
        result[f"probability_{emotion}"] = probabilities[:, index]
    return result


def train(
    model: OfficialPitchFusion,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    artifact_root: Path,
    metadata: dict[str, str],
    *,
    force_training: bool,
    run_name: str = MODEL_NAME,
    optimizer_parameter_groups: list[dict[str, Any]] | None = None,
    early_stopping_patience: int | None = None,
    tracked_parameter_names: tuple[str, ...] = (),
    evaluate_before_training: bool = False,
) -> tuple[list[dict[str, Any]], int, float, float]:
    training = config["training"]
    checkpoint = artifact_root / "last_checkpoint.pt"
    best_path = artifact_root / "best_validation_ua.pt"
    # Optimizers must be constructed after the final device transfer. This is
    # especially important for callers supplying explicit parameter groups.
    model.to(device)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups or (
            parameter for parameter in model.parameters() if parameter.requires_grad
        ),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / int(training["gradient_accumulation_steps"])
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training["warmup_steps"]),
        num_training_steps=steps_per_epoch * int(training["epochs"]),
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_ua = -1.0
    previous_runtime = 0.0
    if checkpoint.exists() and not force_training:
        # This checkpoint is created locally by this runner and contains
        # optimizer and NumPy RNG state in addition to tensor weights.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["metadata"] != metadata:
            raise ValueError(
                "Reproduction checkpoint inputs changed; pass --force-training "
                "only if restarting is intentional"
            )
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        history = saved["history"]
        start_epoch = int(saved["epoch"]) + 1
        best_epoch = int(saved["best_epoch"])
        best_ua = float(saved["best_ua"])
        previous_runtime = float(saved.get("runtime_seconds", 0.0))
        random.setstate(saved["python_random_state"])
        np.random.set_state(saved["numpy_random_state"])
        torch.set_rng_state(saved["torch_random_state"])
        if device.type == "cuda" and saved.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_random_state"])
        print(f"[resume] checkpoint epoch {start_epoch - 1}: {checkpoint}", flush=True)
        del saved

    if evaluate_before_training and start_epoch == 1:
        initial_metrics, _, _, _, initial_loss = evaluate(
            model, validation_loader, device, config["emotion_order"]
        )
        best_epoch = 0
        best_ua = float(initial_metrics["ua"])
        history.append(
            {
                "epoch": 0,
                "train_loss": None,
                "validation_loss": initial_loss,
                "validation_ua": initial_metrics["ua"],
                "validation_wa": initial_metrics["wa"],
                "validation_macro_f1": initial_metrics["macro_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        atomic_torch_save(
            {"epoch": 0, "model_state": model.state_dict(), "metadata": metadata},
            best_path,
        )
        print(
            f"[{run_name}] epoch 0 baseline | val_loss={initial_loss:.4f} | "
            f"val_UA={initial_metrics['ua']:.4f} | val_WA={initial_metrics['wa']:.4f} | "
            f"val_macro_f1={initial_metrics['macro_f1']:.4f}",
            flush=True,
        )

    criterion = nn.CrossEntropyLoss()
    accumulation = int(training["gradient_accumulation_steps"])
    started = time.perf_counter()
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        loss_sum = 0.0
        last_gradient_norm = float("nan")
        for step, (audio, pitch, labels) in enumerate(train_loader, start=1):
            audio = audio.to(device, non_blocking=True)
            pitch = pitch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(audio, pitch).logits
            loss = criterion(logits, labels) / accumulation
            batch_loss = float(loss.detach().item()) * accumulation
            if not math.isfinite(batch_loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}, batch {step}: "
                    f"{batch_loss}"
                )
            loss.backward()
            if step % accumulation == 0 or step == len(train_loader):
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm at epoch {epoch}, batch {step}: "
                        f"{float(gradient_norm)}"
                    )
                last_gradient_norm = float(gradient_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            loss_sum += batch_loss * len(labels)
            if step == 1 or step % 100 == 0 or step == len(train_loader):
                non_finite_parameters = [
                    name for name, parameter in model.named_parameters()
                    if parameter.requires_grad and not torch.isfinite(parameter).all()
                ]
                if non_finite_parameters:
                    raise RuntimeError(
                        "Non-finite trainable parameters after optimizer step: "
                        + ", ".join(non_finite_parameters[:5])
                    )
                elapsed_step = time.perf_counter() - epoch_started
                step_eta = elapsed_step / step * (len(train_loader) - step)
                seen = min(step * int(train_loader.batch_size), len(train_loader.dataset))
                running_loss = loss_sum / seen
                print(
                    f"[{run_name}] epoch {epoch}/{training['epochs']} | "
                    f"batch {step}/{len(train_loader)} | "
                    f"batch_loss={batch_loss:.6f} | "
                    f"running_loss={running_loss:.6f} | "
                    f"grad_norm={last_gradient_norm:.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"epoch_elapsed={elapsed_step:.1f}s | batch_eta={step_eta:.1f}s",
                    flush=True,
                )

        validation_metrics, _, _, _, validation_loss = evaluate(
            model, validation_loader, device, config["emotion_order"]
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader.dataset),
            "validation_loss": validation_loss,
            "validation_ua": validation_metrics["ua"],
            "validation_wa": validation_metrics["wa"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if tracked_parameter_names:
            named_parameters = dict(model.named_parameters())
            row["tracked_parameters"] = {
                name: named_parameters[name].detach().cpu().flatten().tolist()
                for name in tracked_parameter_names
            }
        history.append(row)
        if float(validation_metrics["ua"]) > best_ua:
            best_ua = float(validation_metrics["ua"])
            best_epoch = epoch
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "metadata": metadata,
                },
                best_path,
            )

        current_runtime = time.perf_counter() - started
        atomic_torch_save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "best_epoch": best_epoch,
                "best_ua": best_ua,
                "metadata": metadata,
                "runtime_seconds": previous_runtime + current_runtime,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            },
            checkpoint,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        completed = epoch - start_epoch + 1
        remaining = int(training["epochs"]) - epoch
        eta = current_runtime / completed * remaining
        print(
            f"[{run_name}] epoch {epoch}/{training['epochs']} | "
            f"train_loss={row['train_loss']:.4f} | val_loss={validation_loss:.4f} | "
            f"val_UA={validation_metrics['ua']:.4f} | "
            f"val_WA={validation_metrics['wa']:.4f} | "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} | "
            f"best_UA={best_ua:.4f}@{best_epoch} | "
            f"lr={row['learning_rate']:.2e} | epoch_time={epoch_seconds:.1f}s | "
            f"elapsed={previous_runtime + current_runtime:.1f}s | eta={eta:.1f}s | "
            f"checkpoint={checkpoint}",
            flush=True,
        )
        if tracked_parameter_names:
            print(
                f"[{run_name}] tracked_parameters="
                f"{json.dumps(row['tracked_parameters'], ensure_ascii=False)}",
                flush=True,
            )
        if (
            early_stopping_patience is not None
            and epoch - best_epoch >= early_stopping_patience
        ):
            print(
                f"[{run_name}] early_stop epoch={epoch} | best_epoch={best_epoch} | "
                f"patience={early_stopping_patience}",
                flush=True,
            )
            break

    if not best_path.is_file():
        raise RuntimeError("Training completed without a best-validation checkpoint")
    return (
        history,
        best_epoch,
        best_ua,
        previous_runtime + time.perf_counter() - started,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split-output", default=DEFAULT_SPLIT)
    parser.add_argument("--split-report", default=DEFAULT_SPLIT_REPORT)
    parser.add_argument("--pitch-root", default=DEFAULT_PITCH_ROOT)
    parser.add_argument("--kaldi-executable", default=DEFAULT_KALDI_EXECUTABLE)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--force-pitch", action="store_true")
    parser.add_argument("--force-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = resolve(args.split_output)
    split_report_path = resolve(args.split_report)
    clean_manifest = resolve(config["data"]["clean_manifest"])
    raw_parquet = resolve(config["data"]["raw_parquet"])
    pitch_root = resolve(args.pitch_root)
    kaldi_executable = resolve(args.kaldi_executable)
    artifact_root = resolve(args.artifact_root)
    report_root = resolve(args.report_root)

    manifest = build_clean_stratified_split(
        clean_manifest,
        split_path,
        split_report_path,
        config,
        force=args.force_split,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "split",
                "rows": len(manifest),
                "counts": manifest["split"].value_counts().to_dict(),
                "speaker_overlap_allowed": True,
                "split": split_path.relative_to(ROOT).as_posix(),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.split_only:
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for diagnostics")
    device = torch.device(args.device)
    audio_bytes = load_audio_bytes(raw_parquet)
    if int(manifest["source_row_index"].max()) >= len(audio_bytes):
        raise ValueError("Manifest source_row_index exceeds raw parquet")

    if args.preflight_only:
        probe = manifest.iloc[0]
        audio = decode_wav(audio_bytes[int(probe["source_row_index"])])
        pitch = run_kaldi_pitch_batch(
            [next(manifest.iloc[[0]].itertuples(index=False))],
            audio_bytes,
            kaldi_executable,
            pitch_root / ".work",
            config["pitch"],
        )[str(probe["sample_id"])]
        seed_everything(int(config["seed"]))
        model = build_model(config, args.allow_download).to(device).eval()
        pitch = interpolate_like_author(pitch, len(audio))
        with torch.no_grad():
            logits = model(
                torch.from_numpy(audio).unsqueeze(0).to(device),
                torch.from_numpy(pitch).unsqueeze(0).to(device),
            ).logits
        if tuple(logits.shape) != (1, 4) or not torch.isfinite(logits).all():
            raise RuntimeError(f"Invalid preflight logits: {tuple(logits.shape)}")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "stage": "preflight",
                    "torch": torch.__version__,
                    "python": sys.version.split()[0],
                    "device": str(device),
                    "logits_shape": list(logits.shape),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    prepare_pitch_cache(
        manifest,
        audio_bytes,
        pitch_root,
        kaldi_executable,
        config,
        force=args.force_pitch,
    )
    if args.prepare_only:
        return

    final_report_path = report_root / "m3_paper_reproduction_report.json"
    final_model_path = artifact_root / "m3_paper_reproduction.pt"
    if final_report_path.exists() and final_model_path.exists() and not args.force_training:
        print(
            f"[complete] Existing reproduction outputs found; skipping: {final_report_path}",
            flush=True,
        )
        return

    seed_everything(int(config["seed"]))
    label_to_id = {label: index for index, label in enumerate(config["emotion_order"])}
    partitions = {
        name: manifest[manifest["split"] == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    datasets = {
        name: PaperDataset(frame, audio_bytes, pitch_root, label_to_id)
        for name, frame in partitions.items()
    }
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_paper,
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **loader_options),
        "validation": DataLoader(
            datasets["validation"], shuffle=False, **loader_options
        ),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_options),
    }

    model = build_model(config, args.allow_download)
    metadata = {
        "model": MODEL_NAME,
        "config_sha256": sha256_file(config_path),
        "split_sha256": sha256_file(split_path),
        "raw_parquet_sha256": sha256_file(raw_parquet),
        "pitch_index_sha256": sha256_file(pitch_root / "index.csv"),
    }
    history, best_epoch, best_ua, runtime_seconds = train(
        model,
        loaders["train"],
        loaders["validation"],
        device,
        config,
        artifact_root,
        metadata,
        force_training=args.force_training,
    )

    best = torch.load(
        artifact_root / "best_validation_ua.pt",
        map_location="cpu",
        weights_only=False,
    )
    if best["metadata"] != metadata:
        raise ValueError("Best checkpoint metadata does not match this reproduction")
    model.load_state_dict(best["model_state"])
    model.to(device)
    validation_metrics, _, _, _, _ = evaluate(
        model, loaders["validation"], device, config["emotion_order"]
    )
    test_metrics, truth, predicted, probabilities, test_loss = evaluate(
        model, loaders["test"], device, config["emotion_order"]
    )
    predictions = prediction_frame(
        partitions["test"], truth, predicted, probabilities, config["emotion_order"]
    )
    atomic_csv(predictions, report_root / "m3_paper_reproduction_test_predictions.csv")
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "classes": config["emotion_order"],
            "config": config,
            "best_epoch": best_epoch,
            "metadata": metadata,
        },
        final_model_path,
    )
    target = config["published_target"]
    report = {
        "schema_version": config["schema_version"],
        "model": MODEL_NAME,
        "protocol": config["split_protocol"],
        "exact_historical_reproduction": False,
        "reproduction_boundary": config["reproduction_boundary"],
        "split_counts": {name: len(frame) for name, frame in partitions.items()},
        "speaker_overlap": _speaker_overlap(manifest),
        "best_epoch": best_epoch,
        "best_validation_ua": best_ua,
        "validation": validation_metrics,
        "test": {**test_metrics, "loss": test_loss},
        "published_target": target,
        "gap_to_published": {
            "ua": float(test_metrics["ua"] - float(target["test_ua"])),
            "wa": float(test_metrics["wa"] - float(target["test_wa"])),
        },
        "history": history,
        "runtime_seconds": runtime_seconds,
        "device": str(device),
        "test_evaluations_after_model_selection": 1,
        "model_artifact": final_model_path.relative_to(ROOT).as_posix(),
        "model_artifact_sha256": sha256_file(final_model_path),
        "lineage": metadata,
    }
    atomic_json(report, final_report_path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "best_epoch": best_epoch,
                "validation_UA": validation_metrics["ua"],
                "test_UA": test_metrics["ua"],
                "test_WA": test_metrics["wa"],
                "test_macro_f1": test_metrics["macro_f1"],
                "published_UA": target["test_ua"],
                "published_WA": target["test_wa"],
                "report": final_report_path.relative_to(ROOT).as_posix(),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
