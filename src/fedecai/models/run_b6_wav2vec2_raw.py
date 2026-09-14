"""Official raw-waveform Wav2Vec2 baseline with a frozen cached encoder."""

from __future__ import annotations

import argparse
import io
import json
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
from sklearn.preprocessing import LabelEncoder
from torch import nn
from transformers import Wav2Vec2Model

from fedecai.models.b6_common import (
    git_state, prediction_rows, register_experiment, sha256_file,
    write_csv_atomic, write_json_atomic,
)
from fedecai.models.b6_deep_models import FrozenEmbeddingHead
from fedecai.models.benchmark_costs import (
    TrainingTimer, benchmark_inference_latency, model_size, parameter_count,
)
from fedecai.models.centralized_evaluator import evaluate_prediction_file
from fedecai.models.frozen_embedding_cache import (
    validate_embedding_cache, write_embedding_cache,
)
from fedecai.models.frozen_head_training import train_frozen_head


ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "wav2vec2_raw"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def decode_wav(value: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(value), "rb") as stream:
        if stream.getframerate() != 16000 or stream.getnchannels() != 1:
            raise ValueError("Wav2Vec2 requires mono raw waveform at 16 kHz")
        width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    if width == 2:
        result = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        source = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        packed = (
            source[:, 0].astype(np.int32)
            | source[:, 1].astype(np.int32) << 8
            | source[:, 2].astype(np.int32) << 16
        )
        packed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        result = packed.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {width}")
    if not len(result) or not np.isfinite(result).all():
        raise ValueError("Decoded waveform is empty or non-finite")
    return result


def normalize_waveform(value: np.ndarray, maximum: int | None) -> np.ndarray:
    value = value if maximum is None else value[:maximum]
    return ((value - value.mean()) / max(float(value.std()), 1e-6)).astype(np.float32)


def collate_waveforms(values: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(map(len, values))
    waveforms = torch.zeros(len(values), maximum, dtype=torch.float32)
    masks = torch.zeros(len(values), maximum, dtype=torch.long)
    for index, waveform in enumerate(values):
        waveforms[index, : len(waveform)] = torch.from_numpy(waveform.copy())
        masks[index, : len(waveform)] = 1
    return waveforms, masks


class CachedWav2Vec2Classifier(nn.Module):
    """Full inference path; only the cached-embedding head is trainable."""

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder, self.head = encoder, head
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def encode(self, waveform: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        hidden = self.encoder(waveform, attention_mask=attention_mask).last_hidden_state
        valid = self.encoder._get_feature_vector_attention_mask(
            hidden.shape[1], attention_mask
        ).unsqueeze(-1).to(hidden.dtype)
        return (hidden * valid).sum(1) / valid.sum(1).clamp_min(1)

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pooled = self.encode(waveform, attention_mask)
        return self.head(pooled)


def load_frame(manifest_path: Path, feature_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(
        manifest_path,
        usecols=[
            "sample_id", "source_row_index", "speaker_id", "accent_clean",
            "emotion", "split", "split_version", "audio_sha256",
        ],
        dtype={"sample_id": "string", "speaker_id": "string"},
    )
    eligible = manifest[manifest["split"] != "excluded"].copy()
    if len(eligible) != 4992 or set(eligible["split_version"]) != {"speaker_disjoint_v2"}:
        raise ValueError("Wav2Vec2 runner requires the 4,992-row speaker_disjoint_v2 population")
    features = pd.read_parquet(feature_path, columns=["sample_id", "wav2vec_artifact_path"])
    if features["sample_id"].duplicated().any():
        raise ValueError("B4 Wav2Vec2 feature table contains duplicate sample IDs")
    joined = eligible.merge(features, on="sample_id", how="inner", validate="one_to_one")
    if len(joined) != len(eligible):
        raise ValueError("B4 Wav2Vec2 cache membership does not match B5")
    return joined


def load_audio(raw_path: Path) -> list[bytes]:
    rows = pq.read_table(raw_path, columns=["path"])["path"].to_pylist()
    return [row["bytes"] for row in rows]


def build_full_model(config: dict[str, Any]) -> CachedWav2Vec2Classifier:
    snapshot = resolve(config["model"]["local_snapshot"])
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned local Wav2Vec2 snapshot is absent: {snapshot}")
    encoder = Wav2Vec2Model.from_pretrained(str(snapshot), local_files_only=True)
    head = FrozenEmbeddingHead(
        dimension=int(config["training"]["embedding_dimension"]),
        dropout=float(config["training"]["head_dropout"]),
    )
    return CachedWav2Vec2Classifier(encoder, head)


def expected_cache_metadata(config: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    source_config = ROOT / "configs/features/visec_features_v1.json"
    source_report = ROOT / "reports/tables/b4/b4_wav2vec2_full_extraction_report.json"
    return {
        "cache_version": config["training"]["cache_version"],
        "model": MODEL_NAME,
        "dataset_version": config["dataset_version"],
        "split_version": config["split_version"],
        "fit_partitions": ["train", "validation"],
        "sample_rate_hz": 16000,
        "preprocessing": config["model"]["waveform_normalization"],
        "pretrained_model_id": config["model"]["model_id"],
        "pretrained_model_revision": config["model"]["revision"],
        "pooling": config["model"]["pooling"],
        "embedding_dimension": int(config["training"]["embedding_dimension"]),
        "encoder_policy": "fully_frozen",
        "source_cache": "B4 validated raw-waveform Wav2Vec2 embeddings",
        "source_feature_config_sha256": sha256_file(source_config),
        "source_extraction_report_sha256": sha256_file(source_report),
        "manifest_sha256": sha256_file(manifest_path),
    }


def materialize_b4_cache(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    values = []
    for number, relative in enumerate(frame["wav2vec_artifact_path"], 1):
        path = resolve(str(relative))
        if not path.is_file():
            raise FileNotFoundError(f"Missing B4 Wav2Vec2 artifact: {path}")
        value = np.load(path, allow_pickle=False)
        if value.shape != (768,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid B4 Wav2Vec2 artifact: {path}")
        values.append(value)
        if number % 1000 == 0:
            print(f"Wav2Vec2 cache validation {number}/{len(frame)}", flush=True)
    return np.stack(values).astype(np.float32), time.perf_counter() - started


def smoke_test(config: dict[str, Any], frame: pd.DataFrame, raw_path: Path, device: torch.device) -> None:
    model = build_full_model(config).to(device).eval()
    audio = load_audio(raw_path)
    sample = frame[frame["split"] == "train"].head(4).reset_index(drop=True)
    waveforms = [
        normalize_waveform(decode_wav(audio[int(row.source_row_index)]), 16000)
        for row in sample.itertuples()
    ]
    batch, mask = collate_waveforms(waveforms)
    with torch.inference_mode():
        embeddings = model.encode(batch.to(device), mask.to(device)).cpu().numpy()
    with tempfile.TemporaryDirectory() as directory:
        cache_dir = Path(directory)
        expected = {
            "cache_version": "smoke_v1", "model": MODEL_NAME,
            "dataset_version": config["dataset_version"],
            "split_version": config["split_version"],
        }
        write_embedding_cache(cache_dir, sample, embeddings, expected)
        reloaded, _ = validate_embedding_cache(cache_dir, sample, expected)
    labels = LabelEncoder().fit(config["emotion_order"]).transform(sample["emotion"])
    head_config = {
        "batch_size": 2, "num_workers": 0, "dropout": 0.2,
        "learning_rate": 0.001, "weight_decay": 0.0,
        "max_epochs": 1, "patience": 1, "min_delta": 0.0,
    }
    _, _, _, predicted, _ = train_frozen_head(
        reloaded[:2], labels[:2], reloaded[2:], labels[2:],
        config=head_config, device=device, seed=int(config["seed"]),
        prefix="wav2vec2_cached_smoke",
    )
    print(json.dumps({
        "status": "SMOKE_TEST_ONLY", "model": MODEL_NAME,
        "raw_encoder_executed": True, "cache_saved_reloaded": True,
        "embedding_dimension": reloaded.shape[1], "prediction_count": len(predicted),
        "test_samples_loaded": 0,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/b6_wav2vec2_raw_v1.json")
    parser.add_argument("--features", default="data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    parser.add_argument("--manifest", default="data/manifests/split_manifest_v2_enriched.csv")
    parser.add_argument("--raw-parquet", default="data/raw/visec_hf/train-00000-of-00001.parquet")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--force-head", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, feature_path, manifest_path, raw_path = map(
        resolve, (args.config, args.features, args.manifest, args.raw_parquet)
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["model"]["encoder_policy"] != "fully_frozen" or not config["training"]["embedding_cache"]:
        raise ValueError("Official Wav2Vec2 requires a fully frozen cached encoder")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    frame = load_frame(manifest_path, feature_path)
    development = frame[frame["split"].isin(("train", "validation"))].reset_index(drop=True)
    if args.preflight_only:
        model = build_full_model(config)
        counts = parameter_count(model)
        print(json.dumps({
            "status": "PASS", "model": MODEL_NAME,
            "input_contract": "raw waveform mono 16 kHz",
            "encoder_policy": "fully_frozen", "embedding_cache": True,
            **counts, "python_executable": sys.executable,
            "device": str(device), "test_loaded_for_training": False,
        }, indent=2))
        return
    if args.smoke_test:
        smoke_test(config, development, raw_path, device)
        return

    cache_dir = ROOT / "data/interim/features/b6_frozen_embeddings/wav2vec2_raw"
    expected = expected_cache_metadata(config, manifest_path)
    cache_reused, materialization_seconds = False, None
    if (cache_dir / "metadata.json").is_file() and not args.rebuild_cache:
        embeddings, metadata = validate_embedding_cache(cache_dir, development, expected)
        cache_reused = True
        materialization_seconds = metadata.get("cache_materialization_time_sec")
        print("Wav2Vec2 frozen embedding cache: validated, reusing", flush=True)
    else:
        embeddings, materialization_seconds = materialize_b4_cache(development)
        metadata = write_embedding_cache(
            cache_dir, development, embeddings,
            {
                **expected,
                "feature_extraction_time_sec": None,
                "cache_materialization_time_sec": materialization_seconds,
                "feature_extraction_time_note": "Historical B4 extraction duration is unavailable; no value fabricated.",
            },
        )
        embeddings, metadata = validate_embedding_cache(cache_dir, development, expected)

    encoder = LabelEncoder().fit(config["emotion_order"])
    train_mask = development["split"].eq("train").to_numpy()
    validation_mask = development["split"].eq("validation").to_numpy()
    y_train = encoder.transform(development.loc[train_mask, "emotion"])
    y_validation = encoder.transform(development.loc[validation_mask, "emotion"])
    checkpoint = ROOT / "artifacts/checkpoints" / f"b6_{MODEL_NAME}_seed{config['seed']}.pt"
    canonical_metrics = ROOT / "artifacts/metrics" / f"b6_{MODEL_NAME}_seed{config['seed']}_validation.json"
    canonical_prediction = ROOT / "artifacts/predictions" / f"b6_{MODEL_NAME}_seed{config['seed']}_validation.csv"
    report_path = ROOT / "reports/tables/b6" / f"{MODEL_NAME}_report.json"
    if all(path.is_file() for path in (checkpoint, canonical_metrics, canonical_prediction, report_path)) and not args.force_head:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("training_strategy") == "frozen_encoder_cached_embeddings":
            print("Wav2Vec2 frozen cached baseline: verified outputs exist, skipping", flush=True)
            return

    head_config = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
        "dropout": config["training"]["head_dropout"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "max_epochs": config["training"]["max_epochs"],
        "patience": config["training"]["patience"],
        "min_delta": config["training"]["min_delta"],
    }
    with TrainingTimer() as training_timer:
        head, history, best_epoch, predicted, probabilities = train_frozen_head(
            embeddings[train_mask], y_train, embeddings[validation_mask], y_validation,
            config=head_config, device=device, seed=int(config["seed"]),
            prefix="wav2vec2_raw frozen head",
        )
    validation = development.loc[validation_mask].reset_index(drop=True)
    predicted_text = encoder.inverse_transform(predicted)
    experiment_id = f"b6_{MODEL_NAME}_seed{config['seed']}"
    rows = prediction_rows(
        validation, predicted_text, probabilities, config["emotion_order"],
        experiment_id=experiment_id, model=MODEL_NAME, seed=int(config["seed"]),
    )
    write_csv_atomic(rows, canonical_prediction)
    write_csv_atomic(rows, ROOT / "reports/tables/b6" / f"{MODEL_NAME}_validation_predictions.csv")

    full_model = build_full_model(config).to(device).eval()
    full_model.head.load_state_dict(head.state_dict())
    temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state": head.state_dict(), "encoder_reference": config["model"],
        "classes": list(encoder.classes_), "cache_metadata": metadata,
    }, temporary_checkpoint)
    snapshot = resolve(config["model"]["local_snapshot"])
    size = model_size(temporary_checkpoint, externally_referenced_artifacts=[snapshot])
    size["checkpoint_artifact"] = str(checkpoint.resolve())
    audio = load_audio(raw_path)
    probe = normalize_waveform(
        decode_wav(audio[int(validation.iloc[0]["source_row_index"])]),
        int(config["training"]["max_audio_samples"]),
    )
    waveform, mask = collate_waveforms([probe])
    waveform, mask = waveform.to(device), mask.to(device)
    latency = benchmark_inference_latency(
        lambda: full_model(waveform, mask), device=device,
        **{key: config["latency"][key] for key in (
            "batch_size", "precision", "warmup_runs", "timed_runs", "preprocessing_included"
        )},
    )
    counts = parameter_count(full_model)
    costs = {
        **counts, **size, **training_timer.as_dict(), **latency,
        "feature_extraction_time_sec": metadata.get("feature_extraction_time_sec"),
        "cache_materialization_time_sec": materialization_seconds,
        "model_training_time_sec": training_timer.training_time_sec,
        "end_to_end_training_pipeline_time_sec": None,
        "cost_protocol": {
            "feature_extraction": "reused validated B4 one-time raw-waveform extraction; historical duration unavailable",
            "cache_reused": cache_reused,
            "latency_path": config["latency"]["path"],
        },
    }
    evaluated = evaluate_prediction_file(
        canonical_prediction, canonical_metrics,
        expected_sample_ids=validation["sample_id"].astype(str),
        expected_split="validation", dataset_version=config["dataset_version"],
        split_version=config["split_version"], costs=costs,
    )
    import os
    os.replace(temporary_checkpoint, checkpoint)
    report = {
        "schema_version": "b6.wav2vec2_raw.cached.v1", "model": MODEL_NAME,
        "display_name": "Wav2Vec2 Raw Waveform Frozen Cached-Embedding Baseline",
        "training_strategy": "frozen_encoder_cached_embeddings",
        "model_scope": "raw mono 16 kHz -> frozen Wav2Vec2 -> masked mean -> cached embedding -> head",
        "encoder_policy": "fully_frozen", "embedding_cache": True,
        "cache": {**metadata, "path": str(cache_dir.relative_to(ROOT)).replace("\\", "/"), "reused": cache_reused},
        "selection": {"mean_cv_macro_f1": None, "best_epoch": best_epoch},
        "validation": {
            "overall": {
                "macro_f1": evaluated["overall_macro_f1"], "uar": evaluated["uar"],
                "weighted_f1": evaluated["weighted_f1"], "accuracy": evaluated["accuracy"],
            },
            "per_class": evaluated["per_class"], "confusion_matrix": evaluated["confusion_matrix"],
            "per_accent": evaluated["per_accent"],
            "worst_accent_macro_f1": evaluated["worst_accent_macro_f1"],
            "best_accent_macro_f1": evaluated["best_accent_macro_f1"],
            "accent_gap": evaluated["accent_gap"], "accent_std": evaluated["accent_std"],
        },
        "history": history, "costs": costs, "test_samples_loaded": 0,
        "lineage": {
            "config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(manifest_path),
            "raw_parquet_sha256": sha256_file(raw_path),
            "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(checkpoint), **git_state(ROOT),
        },
    }
    write_json_atomic(report, report_path)
    register_experiment(ROOT, {
        "experiment_id": f"{experiment_id}_frozen_cached", "step": "B6", "rq": "RQ3",
        "model": MODEL_NAME, "variant": "raw_waveform_frozen_cached_embeddings",
        "dataset_version": config["dataset_version"], "split_version": config["split_version"],
        "feature_config": "configs/features/visec_features_v1.json",
        "model_config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "seed": config["seed"], "git_commit": git_state(ROOT)["git_commit"],
        "hardware": costs["gpu_name"], "train_time_sec": costs["training_time_sec"],
        "inference_latency_ms": costs["inference_latency_ms"], "model_size_mb": costs["model_size_mb"],
        "overall_macro_f1": evaluated["overall_macro_f1"],
        "worst_accent_macro_f1": evaluated["worst_accent_macro_f1"],
        "accent_gap": evaluated["accent_gap"], "status": "completed",
        "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "prediction_path": str(canonical_prediction.relative_to(ROOT)).replace("\\", "/"),
        "notes": "Development validation only; B4 raw-waveform cache reused; full encoder latency measured; fixed test untouched.",
    })
    print(json.dumps({
        "status": "PASS", "model": MODEL_NAME, "best_epoch": best_epoch,
        "validation_macro_f1": evaluated["overall_macro_f1"],
        "cache_reused": cache_reused, "test_samples_loaded": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
