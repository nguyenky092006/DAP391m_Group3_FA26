"""Train a paper-aligned raw-waveform Pitch-Fusion model with safe resume."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
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
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model

from fedecai.models.b6_common import (
    classification_metrics, git_state, load_b6_table, load_config,
    prediction_rows, register_experiment, sha256_file, write_csv_atomic,
    write_json_atomic,
)
from fedecai.models.b6_deep_models import PaperAlignedPitchFusionHead
from fedecai.models.benchmark_costs import (
    TrainingTimer, benchmark_inference_latency, model_size, parameter_count,
)
from fedecai.models.centralized_evaluator import evaluate_prediction_file

ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "pitch_fusion_paper_aligned"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_wav(value: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(value), "rb") as stream:
        if stream.getframerate() != 16000 or stream.getnchannels() != 1:
            raise ValueError("Pitch-Fusion requires mono 16 kHz PCM audio")
        width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    if width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        source = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        packed = (source[:, 0].astype(np.int32) | source[:, 1].astype(np.int32) << 8 | source[:, 2].astype(np.int32) << 16)
        packed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        audio = packed.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {width}")
    return audio


def interpolate_pitch(path: Path, length: int) -> np.ndarray:
    with np.load(path) as stored:
        key = "f0_hz" if "f0_hz" in stored else stored.files[0]
        pitch = np.asarray(stored[key], dtype=np.float32).reshape(-1)
    finite = np.isfinite(pitch) & (pitch > 0)
    if not finite.any():
        return np.zeros(length, dtype=np.float32)
    indices = np.arange(len(pitch), dtype=np.float32)
    filled = np.interp(indices, indices[finite], pitch[finite]).astype(np.float32)
    filled = np.log(np.maximum(filled, 1.0))
    filled = (filled - filled.mean()) / max(float(filled.std()), 1e-6)
    return np.interp(
        np.linspace(0, len(filled) - 1, length, dtype=np.float32), indices, filled
    ).astype(np.float32)


class PitchFusionDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, audio_bytes: list[bytes], labels: np.ndarray,
        maximum: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.audio_bytes = audio_bytes
        self.labels = labels
        self.maximum = maximum

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        audio = decode_wav(self.audio_bytes[int(row["source_row_index"])])[:self.maximum]
        audio = (audio - audio.mean()) / max(float(audio.std()), 1e-6)
        pitch = interpolate_pitch(resolve(str(row["pitch_artifact_path"])), len(audio))
        return audio, pitch, int(self.labels[index])


def collate(batch):
    maximum = max(len(row[0]) for row in batch)
    audio = torch.zeros(len(batch), maximum)
    pitch = torch.zeros(len(batch), maximum)
    mask = torch.zeros(len(batch), maximum, dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    for index, (audio_value, pitch_value, label) in enumerate(batch):
        length = len(audio_value)
        audio[index, :length] = torch.from_numpy(audio_value.copy())
        pitch[index, :length] = torch.from_numpy(pitch_value.copy())
        mask[index, :length] = 1
        labels[index] = label
    return audio, pitch, mask, labels


class PaperAlignedPitchFusion(nn.Module):
    def __init__(
        self, backbone: nn.Module, architecture: dict[str, Any],
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.head = PaperAlignedPitchFusionHead(
            hidden_size=int(architecture["hidden_size"]),
            projection_size=int(architecture["classifier_projection_size"]),
            heads=int(architecture["attention_heads"]),
            dropout=float(architecture["dropout"]),
        )

    def forward(self, audio, pitch, attention_mask):
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                acoustic = self.backbone(audio, attention_mask=attention_mask).last_hidden_state
        else:
            acoustic = self.backbone(audio, attention_mask=attention_mask).last_hidden_state
        frame_valid = self.backbone._get_feature_vector_attention_mask(
            acoustic.shape[1], attention_mask
        )
        return self.head(acoustic, pitch, ~frame_valid)


def build_model(config: dict[str, Any], local_files_only: bool) -> PaperAlignedPitchFusion:
    snapshot = resolve(config["backbone"]["local_snapshot"])
    source = str(snapshot) if snapshot.is_dir() else config["backbone"]["model_id"]
    backbone = Wav2Vec2Model.from_pretrained(
        source, revision=None if snapshot.is_dir() else config["backbone"]["revision"],
        local_files_only=local_files_only,
    )
    if config["backbone"]["freeze_feature_encoder"]:
        backbone.feature_extractor._freeze_parameters()
    return PaperAlignedPitchFusion(
        backbone, config["architecture"],
        freeze_backbone=bool(config["backbone"].get("freeze_audio_encoder", False)),
    )


def evaluate(model, loader, device):
    model.eval()
    predictions, probabilities, losses = [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for audio, pitch, mask, labels in loader:
            audio, pitch, mask, labels = (x.to(device, non_blocking=True) for x in (audio, pitch, mask, labels))
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(audio, pitch, mask)
                loss = criterion(logits, labels)
            losses += float(loss.item()) * len(labels)
            # Autocast may leave logits in float16.  Compute exported
            # probabilities in float32 so the shared evaluator's probability
            # simplex check is not tripped by half-precision rounding.
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            predictions.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities), losses / len(loader.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/b6_pitch_fusion_paper_v1.json")
    parser.add_argument("--features", default="data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    parser.add_argument("--manifest", default="data/manifests/split_manifest_v2_enriched.csv")
    parser.add_argument("--raw-parquet", default="data/raw/visec_hf/train-00000-of-00001.parquet")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, features_path, manifest_path, raw_path = map(
        resolve, (args.config, args.features, args.manifest, args.raw_parquet)
    )
    config = load_config(config_path)
    frame, _ = load_b6_table(features_path, manifest_path, config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use .venv-gpu or pass --device cpu")
    device = torch.device(args.device)
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    if len(train) != 3681 or len(validation) != 646:
        raise ValueError("Unexpected B5 train/validation sizes")
    experiment_id = f"b6_{MODEL_NAME}_seed{config['seed']}"
    report_dir = ROOT / "reports/tables/b6"
    report_path = report_dir / f"{MODEL_NAME}_report.json"
    model_path = ROOT / "artifacts/checkpoints" / f"{experiment_id}.pt"
    canonical_prediction = ROOT / "artifacts/predictions" / f"{experiment_id}_validation.csv"
    canonical_metrics = ROOT / "artifacts/metrics" / f"{experiment_id}_validation.json"
    if not args.force and not args.preflight_only and not args.smoke_test and all(
        path.is_file() for path in (report_path, model_path, canonical_prediction, canonical_metrics)
    ):
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        lineage = existing.get("lineage", {})
        if (
            lineage.get("config_sha256") == sha256_file(config_path)
            and lineage.get("checkpoint_sha256") == sha256_file(model_path)
            and int(existing.get("test_samples_loaded", -1)) == 0
        ):
            print(json.dumps({
                "status": "SKIP_COMPLETE", "model": MODEL_NAME,
                "validation_macro_f1": existing["validation"]["overall"]["macro_f1"],
                "test_untouched": True,
            }), flush=True)
            return
    pitch_paths = [resolve(str(path)) for path in frame["pitch_artifact_path"]]
    missing = [path for path in pitch_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pitch artifact: {missing[0]}")
    audio_column = pq.read_table(raw_path, columns=["path"])["path"].to_pylist()
    audio_bytes = [row["bytes"] for row in audio_column]
    maximum_index = int(frame["source_row_index"].max())
    if maximum_index >= len(audio_bytes):
        raise ValueError("Manifest source_row_index exceeds raw parquet")
    model = build_model(config, local_files_only=not args.allow_download)
    parameters = sum(value.numel() for value in model.parameters())
    print(json.dumps({
        "status": "PASS", "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train": len(train), "validation": len(validation), "test_loaded_for_training": False,
        "parameters": parameters, "paper_aligned": True,
        "exact_historical_reproduction": False,
    }), flush=True)
    if args.preflight_only or args.smoke_test:
        probe_encoder = LabelEncoder().fit(config["emotion_order"])
        probe = PitchFusionDataset(
            train.iloc[:1], audio_bytes,
            probe_encoder.transform(train.iloc[:1]["emotion"]),
            int(config["training"]["max_audio_samples"]),
        )
        audio, pitch, mask, _ = collate([probe[0]])
        model.to(device).eval()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(audio.to(device), pitch.to(device), mask.to(device))
        if tuple(logits.shape) != (1, 4) or not torch.isfinite(logits).all():
            raise RuntimeError(f"Invalid preflight logits: {tuple(logits.shape)}")
        if args.smoke_test:
            model.train()
            target = torch.tensor([0], device=device)
            optimizer = torch.optim.AdamW(
                (value for value in model.parameters() if value.requires_grad),
                lr=float(config["training"]["learning_rate"]),
            )
            optimizer.zero_grad(set_to_none=True)
            trained_logits = model(audio.to(device), pitch.to(device), mask.to(device))
            loss = nn.CrossEntropyLoss()(trained_logits, target)
            loss.backward(); optimizer.step()
            if not torch.isfinite(loss):
                raise RuntimeError("Pitch-Fusion smoke loss is not finite")
            print(json.dumps({
                "status": "SMOKE_TEST_ONLY", "forward_smoke_test": "PASS",
                "tiny_train_step": "PASS", "logits_shape": [1, 4],
                "audio_encoder_frozen": bool(config["backbone"].get("freeze_audio_encoder")),
                "test_samples_loaded": 0,
            }), flush=True)
        else:
            print(json.dumps({"forward_smoke_test": "PASS", "logits_shape": [1, 4]}), flush=True)
        return

    training = config["training"]
    encoder = LabelEncoder().fit(config["emotion_order"])
    y_train, y_validation = encoder.transform(train["emotion"]), encoder.transform(validation["emotion"])
    options = {"batch_size": int(training["batch_size"]), "num_workers": int(training["num_workers"]), "pin_memory": True, "collate_fn": collate}
    train_loader = DataLoader(PitchFusionDataset(train, audio_bytes, y_train, int(training["max_audio_samples"])), shuffle=True, **options)
    validation_loader = DataLoader(PitchFusionDataset(validation, audio_bytes, y_validation, int(training["max_audio_samples"])), shuffle=False, **options)
    counts = np.bincount(y_train, minlength=4).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(y_train) / (4 * counts), device=device))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(training["mixed_precision"]))
    artifact_dir = ROOT / "artifacts/b6/pitch_fusion_paper"
    checkpoint_path = artifact_dir / "checkpoint.pt"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    previous_report_path = report_dir / f"{MODEL_NAME}_report.json"
    previous_report = (
        json.loads(previous_report_path.read_text(encoding="utf-8"))
        if previous_report_path.is_file() else {}
    )
    history, start_epoch, best_score, best_state = [], 1, -1.0, None
    if checkpoint_path.exists() and not args.force:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        history, start_epoch, best_score, best_state = saved["history"], saved["epoch"] + 1, saved["best_score"], saved["best_state"]
        print(f"resume checkpoint: epoch {saved['epoch']}", flush=True)
    seed_everything(int(config["seed"]))
    accumulation = int(training["gradient_accumulation_steps"])
    reused_completed_checkpoint = start_epoch > int(training["epochs"])
    started = time.perf_counter()
    with TrainingTimer() as training_timer:
        stale = 0
        for epoch in range(start_epoch, int(training["epochs"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for step, (audio, pitch, mask, labels) in enumerate(train_loader, 1):
                audio, pitch, mask, labels = (x.to(device, non_blocking=True) for x in (audio, pitch, mask, labels))
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    logits = model(audio, pitch, mask)
                    loss = criterion(logits, labels) / accumulation
                scaler.scale(loss).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                total += float(loss.item()) * accumulation * len(labels)
            predicted, _, validation_loss = evaluate(model, validation_loader, device)
            score = classification_metrics(encoder.inverse_transform(y_validation), encoder.inverse_transform(predicted), validation["accent_clean"].to_numpy(), config["emotion_order"])["overall"]["macro_f1"]
            history.append({"epoch": epoch, "train_loss": total / len(train), "validation_loss": validation_loss, "validation_macro_f1": score})
            if score > best_score + float(training.get("min_delta", 0.0)):
                best_score = score
                stale = 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(), "history": history, "best_score": best_score, "best_state": best_state}, checkpoint_path.with_suffix(".tmp"))
            checkpoint_path.with_suffix(".tmp").replace(checkpoint_path)
            print(f"{MODEL_NAME} epoch {epoch}/{training['epochs']} train_loss={total / len(train):.4f} val_macro_f1={score:.4f}", flush=True)
            if bool(training.get("early_stopping")) and stale >= int(training.get("patience", 3)):
                print(f"{MODEL_NAME} early stopping at epoch {epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    predicted, probabilities, _ = evaluate(model, validation_loader, device)
    predicted_text = encoder.inverse_transform(predicted)
    metrics = classification_metrics(validation["emotion"].to_numpy(), predicted_text, validation["accent_clean"].to_numpy(), config["emotion_order"])
    torch.save({"model_state": best_state, "classes": list(encoder.classes_), "config": config}, model_path)
    parameter_evidence = parameter_count(model)
    size_evidence = model_size(model_path)
    latency_batch = tuple(item[:1] for item in next(iter(validation_loader)))
    audio_probe, pitch_probe, mask_probe, _ = (
        item.to(device) for item in latency_batch
    )
    model.eval()
    latency_evidence = benchmark_inference_latency(
        lambda: model(audio_probe, pitch_probe, mask_probe), device=device,
        batch_size=1, precision="float32", warmup_runs=10, timed_runs=50,
        preprocessing_included=False,
    )
    recorded_training_time = (
        previous_report.get("runtime_seconds")
        if reused_completed_checkpoint and previous_report.get("runtime_seconds") is not None
        else training_timer.training_time_sec
    )
    costs = {
        **parameter_evidence, **size_evidence, **training_timer.as_dict(),
        **latency_evidence, "feature_extraction_time_sec": None,
        "training_time_sec": recorded_training_time,
        "model_training_time_sec": recorded_training_time,
        "end_to_end_training_pipeline_time_sec": recorded_training_time,
        "cost_protocol": {
            "feature_extraction": "raw decode and pitch loading occur in the training data path",
            "latency_path": "model-ready waveform and pitch tensors -> Wav2Vec2 plus fusion network",
            "reused_completed_checkpoint": reused_completed_checkpoint,
            "training_time_source": (
                "persisted prior runtime_seconds" if reused_completed_checkpoint
                else "current TrainingTimer"
            ),
        },
    }
    prediction_path = report_dir / f"{MODEL_NAME}_validation_predictions.csv"
    rows = prediction_rows(
        validation, predicted_text, probabilities, config["emotion_order"],
        experiment_id=experiment_id, model=MODEL_NAME, seed=int(config["seed"]),
    )
    write_csv_atomic(rows, prediction_path)
    write_csv_atomic(rows, canonical_prediction)
    evaluated = evaluate_prediction_file(
        canonical_prediction, canonical_metrics,
        expected_sample_ids=validation["sample_id"].astype(str),
        expected_split="validation", dataset_version="visec_clean_eligible_v1",
        split_version=config["split_version"], costs=costs,
    )
    repository_state = git_state(ROOT)
    report = {
        "schema_version": config["schema_version"], "model": MODEL_NAME,
        "display_name": "Pitch-Fusion Paper-Aligned Reproduction",
        "paper_aligned": True, "is_exact_historical_reproduction": False,
        "reproduction_boundary": config["reproduction_boundary"],
        "best_epoch": max(history, key=lambda row: row["validation_macro_f1"])["epoch"],
        "validation": metrics, "history": history, "test_samples_loaded": 0,
        "training_strategy": (
            "reused_completed_paper_aligned_checkpoint" if reused_completed_checkpoint
            else "minimum_single_baseline_frozen_audio_encoder"
        ),
        "costs": costs,
        "lineage": {
            "feature_sha256": sha256_file(features_path),
            "manifest_sha256": sha256_file(manifest_path),
            "raw_parquet_sha256": sha256_file(raw_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(model_path), **repository_state,
        },
        "runtime_seconds": time.perf_counter() - started, "device": str(device),
    }
    write_json_atomic(report, report_path)
    register_experiment(ROOT, {
        "experiment_id": f"b6_{MODEL_NAME}_dev_v1_{int(time.time())}", "step": "B6",
        "rq": "RQ3", "model": MODEL_NAME,
        "variant": "paper_aligned_minimal_frozen_audio_encoder",
        "dataset_version": "visec_clean_eligible_v1", "status": "completed",
        "feature_config": "configs/features/visec_features_v1.json",
        "model_config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "seed": config["seed"], "split_version": config["split_version"],
        "git_commit": repository_state["git_commit"], "hardware": costs["gpu_name"],
        "train_time_sec": costs["training_time_sec"],
        "inference_latency_ms": costs["inference_latency_ms"],
        "model_size_mb": costs["model_size_mb"],
        "overall_macro_f1": evaluated["overall_macro_f1"],
        "worst_accent_macro_f1": evaluated["worst_accent_macro_f1"],
        "accent_gap": evaluated["accent_gap"],
        "checkpoint_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "prediction_path": str(canonical_prediction.relative_to(ROOT)).replace("\\", "/"),
        "notes": "Paper-aligned development run; known reproduction differences retained; final test remained unopened.",
    })
    print(json.dumps({"status": "PASS", "validation_macro_f1": metrics["overall"]["macro_f1"], "best_epoch": report["best_epoch"], "test_untouched": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
