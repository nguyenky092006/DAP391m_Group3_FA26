"""Execute the hard-gated ResNet18 FedECAI E2-E7 validation ladder.

The module never loads test audio/features. E1 is immutable initialization and
comparison evidence; E2-E7 write isolated artifacts under artifacts/fedecai/.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedecai.features.mel_3ch import build_mel_3ch, pad_or_crop_mel_3ch
from fedecai.models.b6_common import (
    git_state, prediction_rows, register_experiment, sha256_file,
    write_csv_atomic, write_json_atomic,
)
from fedecai.models.b6_deep_models import ResNet18EmotionAccentModel, gradient_reverse
from fedecai.models.benchmark_costs import (
    TrainingTimer, benchmark_inference_latency, model_size, parameter_count,
)
from fedecai.models.centralized_evaluator import evaluate_prediction_file


ROOT = Path(__file__).resolve().parents[3]
FEATURE_TABLE = ROOT / "data/interim/features/b4_features_v1/b4_full_feature_table.parquet"
SPLIT_MANIFEST = ROOT / "data/manifests/emotion_stratified_v1.csv"
CLIENT_MANIFEST = ROOT / "data/manifests/federated_client_assignments_emotion_stratified_v1.csv"
E1_CHECKPOINT = ROOT / "artifacts/checkpoints/b6_resnet18_mel3_finetune_seed391.pt"
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
CONFIGS = {
    "E2": "configs/models/fedecai/e2_resnet18_unconditional_grl.json",
    "E3": "configs/models/fedecai/e3_resnet18_emotion_conditioned_grl.json",
    "E4": "configs/models/fedecai/e4_fedavg_resnet18.json",
    "E5": "configs/models/fedecai/e5_fedprox_resnet18.json",
    "E6": "configs/models/fedecai/e6_fedavg_resnet18_unconditional_grl.json",
    "E7": "configs/models/fedecai/e7_fedecai_resnet18.json",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FedECAIMelDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, frames: int, emotions: list[str], accents: list[str]) -> None:
        self.frame = frame.reset_index(drop=True)
        self.frames = frames
        self.emotion_to_index = {label: index for index, label in enumerate(emotions)}
        self.accent_to_index = {label: index for index, label in enumerate(accents)}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        path = Path(str(row["log_mel_artifact_path"]))
        path = path if path.is_absolute() else ROOT / path
        mel = np.load(path, allow_pickle=False).astype(np.float32)
        if mel.ndim != 2 or mel.shape[0] != 80 or not np.isfinite(mel).all():
            raise ValueError(f"Invalid log-Mel artifact: {path}")
        value = torch.from_numpy(pad_or_crop_mel_3ch(build_mel_3ch(mel), self.frames))
        value = (value - IMAGE_MEAN) / IMAGE_STD
        return (
            value,
            torch.tensor(self.emotion_to_index[str(row["emotion"])], dtype=torch.long),
            torch.tensor(self.accent_to_index[str(row["accent_clean"])], dtype=torch.long),
        )


def load_config(experiment: str) -> tuple[Path, dict[str, Any]]:
    path = ROOT / CONFIGS[experiment]
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment", "experiment_id", "backbone", "training_policy",
        "dataset_version", "split_version", "seed", "input", "emotion_classes",
        "accent_classes", "adversary", "initial_checkpoint", "artifact_root",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if config["experiment"] != experiment or config["backbone"] != "resnet18_mel3":
        raise ValueError("Active E2-E7 path must resolve only to resnet18_mel3")
    if config["training_policy"] != "full_finetune" or config["input"] != "3-channel Mel":
        raise ValueError("Active ResNet18 training/input contract mismatch")
    if config["emotion_classes"] != ["angry", "happy", "neutral", "sad"]:
        raise ValueError("Emotion class contract mismatch")
    if config["accent_classes"] != ["north", "central", "south"]:
        raise ValueError("Accent class contract mismatch")
    if config.get("forbidden_split") != "test":
        raise ValueError("Every E2-E7 config must forbid test")
    if experiment in {"E2", "E3", "E6", "E7"} and float(config.get("lambda_adv", -1)) != 0.1:
        raise ValueError("The controlled E2/E3/E6/E7 ladder requires lambda_adv=0.10")
    return path, config


def load_development_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(
        SPLIT_MANIFEST,
        usecols=["sample_id", "speaker_id", "accent_clean", "emotion", "split", "split_version"],
        dtype={"sample_id": "string", "speaker_id": "string"},
    )
    if set(manifest["split_version"].dropna().unique()) != {"emotion_stratified_v1"}:
        raise ValueError("Unexpected split version")
    development = manifest.loc[manifest["split"].isin(["train", "validation"])].copy()
    if (development["split"] == "train").sum() != 3681 or (development["split"] == "validation").sum() != 646:
        raise ValueError("Development membership count mismatch")
    features = pd.read_parquet(FEATURE_TABLE, columns=["sample_id", "log_mel_artifact_path"])
    frame = development.merge(features, on="sample_id", how="left", validate="one_to_one")
    if frame["log_mel_artifact_path"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Incomplete or duplicate development feature join")
    return (
        frame.loc[frame["split"].eq("train")].reset_index(drop=True),
        frame.loc[frame["split"].eq("validation")].reset_index(drop=True),
    )


def target_frames(train: pd.DataFrame, maximum: int = 512) -> int:
    widths = []
    for value in train["log_mel_artifact_path"]:
        path = Path(str(value)); path = path if path.is_absolute() else ROOT / path
        widths.append(np.load(path, mmap_mode="r").shape[1])
    return min(int(np.percentile(widths, 95)), int(maximum))


def load_e1_state() -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(E1_CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment_id") != "b6_resnet18_mel3_finetune_seed391":
        raise ValueError("Wrong E1 checkpoint identity")
    if checkpoint.get("model") != "resnet18_mel3":
        raise ValueError("Wrong E1 backbone")
    return checkpoint["model_state"], checkpoint


def build_model(config: dict[str, Any]) -> ResNet18EmotionAccentModel:
    model = ResNet18EmotionAccentModel(
        adversary=str(config["adversary"]), dropout=float(config.get("dropout", 0.2)),
        adversary_hidden_dim=int(config.get("adversary_hidden_dim", 128)),
    )
    state, _ = load_e1_state()
    model.emotion_model.network.load_state_dict(state, strict=True)
    if model.feature_dim != int(model.emotion_model.network.fc[1].in_features):
        raise RuntimeError("ResNet18 feature dimension was not derived from the implementation")
    return model


def optimizer_for(model: ResNet18EmotionAccentModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    high_lr = list(model.emotion_model.network.fc.parameters())
    if model.accent_head is not None:
        high_lr += list(model.accent_head.parameters())
    high_ids = {id(value) for value in high_lr}
    backbone = [value for value in model.parameters() if id(value) not in high_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(config["backbone_learning_rate"])},
            {"params": high_lr, "lr": float(config["head_learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )


def emotion_weights(frame: pd.DataFrame, labels: list[str], device: torch.device) -> torch.Tensor:
    counts = frame["emotion"].value_counts()
    values = [len(frame) / (len(labels) * max(int(counts.get(label, 0)), 1)) for label in labels]
    return torch.tensor(values, dtype=torch.float32, device=device)


def evaluate_model(
    model: ResNet18EmotionAccentModel, loader: DataLoader, device: torch.device,
    lambda_adv: float,
) -> dict[str, Any]:
    model.eval()
    emotion_truth, emotion_predicted, probabilities = [], [], []
    accent_truth, accent_predicted, features = [], [], []
    emotion_loss_sum = accent_loss_sum = 0.0
    with torch.inference_mode():
        for values, emotions, accents in loader:
            values, emotions, accents = values.to(device), emotions.to(device), accents.to(device)
            representation = model.forward_features(values)
            emotion_logits = model.forward_emotion(representation)
            if model.accent_head is None:
                accent_logits = None
            elif model.adversary == "conditional":
                posterior = torch.softmax(emotion_logits, dim=1).detach()
                accent_logits = model.accent_head(torch.cat((representation, posterior), dim=1))
            else:
                accent_logits = model.accent_head(representation)
            emotion_loss_sum += float(nn.functional.cross_entropy(emotion_logits, emotions, reduction="sum"))
            if accent_logits is not None:
                accent_loss_sum += float(nn.functional.cross_entropy(accent_logits, accents, reduction="sum"))
                accent_predicted.append(accent_logits.argmax(1).cpu().numpy())
            emotion_truth.append(emotions.cpu().numpy())
            emotion_predicted.append(emotion_logits.argmax(1).cpu().numpy())
            probabilities.append(torch.softmax(emotion_logits.float(), dim=1).cpu().numpy())
            accent_truth.append(accents.cpu().numpy())
            features.append(representation.float().cpu().numpy())
    n = len(loader.dataset)
    return {
        "emotion_truth": np.concatenate(emotion_truth),
        "emotion_predicted": np.concatenate(emotion_predicted),
        "probabilities": np.concatenate(probabilities),
        "accent_truth": np.concatenate(accent_truth),
        "accent_predicted": np.concatenate(accent_predicted) if accent_predicted else None,
        "features": np.concatenate(features),
        "emotion_loss": emotion_loss_sum / n,
        "accent_loss": accent_loss_sum / n if accent_predicted else None,
    }


def accent_discriminator_metrics(result: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    predicted = result["accent_predicted"]
    if predicted is None:
        return {"macro_f1": None, "uar": None}
    truth = result["accent_truth"]
    indices = list(range(len(labels)))
    return {
        "macro_f1": float(f1_score(truth, predicted, labels=indices, average="macro", zero_division=0)),
        "uar": float(recall_score(truth, predicted, labels=indices, average="macro", zero_division=0)),
    }


def state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_bytes(state: dict[str, torch.Tensor]) -> int:
    return int(sum(value.numel() * value.element_size() for value in state.values()))


def registry(config: dict[str, Any], config_path: Path, status: str, **values: Any) -> None:
    repository = git_state(ROOT)
    register_experiment(ROOT, {
        "experiment_id": config["experiment_id"], "step": config["experiment"],
        "rq": "RQ4" if config["experiment"] in {"E2", "E3"} else "RQ5",
        "model": "resnet18_mel3", "variant": config["adversary"],
        "dataset_version": config["dataset_version"], "split_version": config["split_version"],
        "client_partition_version": values.get("client_partition_version", ""),
        "feature_config": "configs/features/visec_features_v1.json",
        "model_config": config_path.relative_to(ROOT).as_posix(), "seed": config["seed"],
        "git_commit": repository["git_commit"], "hardware": values.get("hardware", ""),
        "train_time_sec": values.get("training_time_sec", ""),
        "inference_latency_ms": values.get("inference_latency_ms", ""),
        "model_size_mb": values.get("model_size_mb", ""),
        "overall_macro_f1": values.get("overall_macro_f1", ""),
        "worst_accent_macro_f1": values.get("worst_accent_macro_f1", ""),
        "accent_gap": values.get("accent_gap", ""), "status": status,
        "checkpoint_path": values.get("checkpoint_path", ""),
        "prediction_path": values.get("prediction_path", ""),
        "notes": values.get("notes", "Validation-only ResNet18 E1-E7 ladder; fixed test untouched."),
    })


def save_checkpoint(path: Path, model: nn.Module, config: dict[str, Any], state: dict[str, torch.Tensor], kind: str, scenario: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model_state": state, "experiment_id": config["experiment_id"],
        "experiment": config["experiment"], "model": "resnet18_mel3",
        "adversary": config["adversary"], "kind": kind, "scenario": scenario,
        "feature_dim": int(getattr(model, "feature_dim")), "config": config,
        "initial_global_checkpoint_sha256": sha256_file(E1_CHECKPOINT),
    }, temporary)
    os.replace(temporary, path)


def make_loaders(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any], frames: int, *, smoke: bool = False) -> tuple[DataLoader, DataLoader]:
    if smoke:
        train = train.head(12).copy(); validation = validation.head(12).copy(); frames = min(frames, 96)
    train_set = FedECAIMelDataset(train, frames, config["emotion_classes"], config["accent_classes"])
    validation_set = FedECAIMelDataset(validation, frames, config["emotion_classes"], config["accent_classes"])
    batch_size = int(config.get("batch_size", config.get("local_batch_size", 16)))
    options = {"batch_size": min(batch_size, len(train_set)), "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    generator = torch.Generator().manual_seed(int(config["seed"]))
    return (
        DataLoader(train_set, shuffle=True, generator=generator, **options),
        DataLoader(validation_set, shuffle=False, batch_size=min(batch_size, len(validation_set)), num_workers=0, pin_memory=torch.cuda.is_available()),
    )


def persist_predictions(
    result: dict[str, Any], validation: pd.DataFrame, config: dict[str, Any], path: Path,
) -> None:
    labels = np.asarray(config["emotion_classes"])
    rows = prediction_rows(
        validation, labels[result["emotion_predicted"]], result["probabilities"],
        config["emotion_classes"], experiment_id=config["experiment_id"],
        model="resnet18_mel3", seed=int(config["seed"]),
    )
    write_csv_atomic(rows, path)


def run_centralized(config_path: Path, config: dict[str, Any], device: torch.device, *, smoke: bool) -> dict[str, Any]:
    root = ROOT / config["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    train, validation = load_development_frame()
    frames = target_frames(train, int(config["max_frames"]))
    train_loader, validation_loader = make_loaders(train, validation, config, frames, smoke=smoke)
    seed_everything(int(config["seed"]))
    model = build_model(config).to(device)
    if smoke:
        optimizer = optimizer_for(model, config)
        values, emotions, accents = next(iter(train_loader))
        values, emotions, accents = values.to(device), emotions.to(device), accents.to(device)
        optimizer.zero_grad(set_to_none=True)
        emotion_logits, accent_logits = model(values, float(config["lambda_adv"]))
        loss = nn.functional.cross_entropy(emotion_logits, emotions) + nn.functional.cross_entropy(accent_logits, accents)
        loss.backward()
        finite = bool(torch.isfinite(loss).item()) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
        )
        if not finite:
            raise RuntimeError("Non-finite centralized smoke gradients")
        optimizer.step()
        audit = {
            "status": "SMOKE_PASS", "experiment_id": config["experiment_id"],
            "backbone": "resnet18_mel3", "input_channels": 3,
            "feature_dim": model.feature_dim, "lambda_adv": config["lambda_adv"],
            "loss": float(loss.item()), "finite_gradients": finite, "test_samples_loaded": 0,
        }
        write_json_atomic(audit, root / "smoke_audit.json")
        registry(config, config_path, "SMOKE_PASS", notes="One-batch smoke only; no benchmark metric; fixed test untouched.")
        return audit

    write_json_atomic(config, root / "config.json")
    registry(config, config_path, "RUNNING")
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["max_epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["mixed_precision"]))
    criterion_emotion = nn.CrossEntropyLoss(weight=emotion_weights(train, config["emotion_classes"], device))
    criterion_accent = nn.CrossEntropyLoss()
    accumulation = int(config["gradient_accumulation_steps"])
    history: list[dict[str, Any]] = []
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    with TrainingTimer() as timer:
        for epoch in range(1, int(config["max_epochs"]) + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            emotion_sum = accent_sum = 0.0
            for step, (values, emotions, accents) in enumerate(train_loader, 1):
                values, emotions, accents = values.to(device), emotions.to(device), accents.to(device)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    emotion_logits, accent_logits = model(values, float(config["lambda_adv"]))
                    emotion_loss = criterion_emotion(emotion_logits, emotions)
                    accent_loss = criterion_accent(accent_logits, accents)
                    loss = (emotion_loss + accent_loss) / accumulation
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("Non-finite centralized training loss")
                scaler.scale(loss).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                emotion_sum += float(emotion_loss.item()) * len(emotions)
                accent_sum += float(accent_loss.item()) * len(accents)
            result = evaluate_model(model, validation_loader, device, float(config["lambda_adv"]))
            score = float(f1_score(result["emotion_truth"], result["emotion_predicted"], average="macro", zero_division=0))
            uar = float(recall_score(result["emotion_truth"], result["emotion_predicted"], average="macro", zero_division=0))
            row = {
                "epoch": epoch, "train_emotion_loss": emotion_sum / len(train_loader.dataset),
                "train_accent_loss": accent_sum / len(train_loader.dataset),
                "validation_emotion_loss": result["emotion_loss"],
                "validation_accent_loss": result["accent_loss"],
                "validation_macro_f1": score, "validation_uar": uar,
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
            }
            history.append(row)
            print(f"{config['experiment']} epoch={epoch} macro_f1={score:.6f} uar={uar:.6f}", flush=True)
            if score > best_score + float(config["min_delta"]):
                best_score, best_epoch, best_state, stale = score, epoch, state_copy(model), 0
            else:
                stale += 1
            scheduler.step()
            if stale >= int(config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No centralized best checkpoint")
    last_state = state_copy(model)
    best_path, last_path = root / "best.pt", root / "last.pt"
    save_checkpoint(best_path, model, config, best_state, "best")
    save_checkpoint(last_path, model, config, last_state, "last")
    model.load_state_dict(best_state); model.eval()
    result = evaluate_model(model, validation_loader, device, float(config["lambda_adv"]))
    prediction_path = root / "predictions.csv"
    persist_predictions(result, validation, config, prediction_path)
    counts = parameter_count(model); size = model_size(best_path)
    probe = next(iter(validation_loader))[0][:1].to(device)
    latency = benchmark_inference_latency(
        lambda: model(probe, float(config["lambda_adv"])), device=device,
        batch_size=1, precision="float32", warmup_runs=10, timed_runs=50,
        preprocessing_included=False,
    )
    costs = {
        **counts, **size, **timer.as_dict(), **latency,
        "feature_extraction_time_sec": None, "model_training_time_sec": timer.training_time_sec,
        "end_to_end_training_pipeline_time_sec": timer.training_time_sec,
        "cost_protocol": {"feature_extraction": "B4 deterministic Mel cache excluded", "latency_path": "model-ready 3-channel Mel -> ResNet18 -> emotion and accent heads"},
    }
    metrics_path = root / "metrics.json"
    metrics = evaluate_prediction_file(
        prediction_path, metrics_path, expected_sample_ids=validation["sample_id"].astype(str),
        expected_split="validation", dataset_version=config["dataset_version"],
        split_version=config["split_version"], costs=costs,
    )
    discriminator = accent_discriminator_metrics(result, config["accent_classes"])
    metrics["accent_discriminator_macro_f1"] = discriminator["macro_f1"]
    metrics["accent_discriminator_uar"] = discriminator["uar"]
    write_json_atomic(metrics, metrics_path)
    pd.DataFrame(history).to_csv(root / "training_history.csv", index=False)
    write_json_atomic(costs, root / "resource_metrics.json")
    checkpoint_manifest = {
        "initial_e1_checkpoint": E1_CHECKPOINT.relative_to(ROOT).as_posix(),
        "initial_e1_sha256": sha256_file(E1_CHECKPOINT),
        "best_checkpoint": "best.pt", "best_sha256": sha256_file(best_path),
        "last_checkpoint": "last.pt", "last_sha256": sha256_file(last_path),
        "best_epoch": best_epoch,
    }
    write_json_atomic(checkpoint_manifest, root / "checkpoint_manifest.json")
    replay = build_model(config).to(device); replay.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model_state"]); replay.eval()
    replay_result = evaluate_model(replay, validation_loader, device, float(config["lambda_adv"]))
    reproducible = bool(np.array_equal(replay_result["emotion_predicted"], result["emotion_predicted"]) and np.allclose(replay_result["probabilities"], result["probabilities"], atol=1e-6))
    audit = {
        "status": "PASS" if reproducible else "FAIL", "experiment_id": config["experiment_id"],
        "validation_samples": len(validation), "validation_membership_sha256": sha256_file(SPLIT_MANIFEST),
        "checkpoint_replay": reproducible, "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": sha256_file(best_path), "input_channels": 3,
        "feature_dim": model.feature_dim, "test_samples_loaded": 0,
    }
    write_json_atomic(audit, root / "audit.json")
    if not reproducible:
        raise RuntimeError("Predictions cannot be reproduced from centralized checkpoint")
    report = {
        "status": "COMPLETE", "experiment_id": config["experiment_id"], "best_epoch": best_epoch,
        "epochs_run": len(history), "metrics": metrics, "accent_discriminator": discriminator,
        "lineage": checkpoint_manifest, "audit": audit, "test_samples_loaded": 0,
    }
    write_json_atomic(report, root / "run_report.json")
    registry(config, config_path, "COMPLETE", hardware=costs["gpu_name"], training_time_sec=costs["training_time_sec"], inference_latency_ms=costs["inference_latency_ms"], model_size_mb=costs["model_size_mb"], overall_macro_f1=metrics["overall_macro_f1"], worst_accent_macro_f1=metrics["worst_accent_macro_f1"], accent_gap=metrics["accent_gap"], checkpoint_path=best_path.relative_to(ROOT).as_posix(), prediction_path=prediction_path.relative_to(ROOT).as_posix())
    return report


def federated_client_preflight() -> dict[str, Any]:
    manifest = pd.read_csv(
        SPLIT_MANIFEST, usecols=["sample_id", "speaker_id", "split", "emotion", "accent_clean"],
        dtype={"sample_id": "string", "speaker_id": "string"},
    )
    assignments = pd.read_csv(CLIENT_MANIFEST, dtype={"speaker_id": "string"})
    train = manifest.loc[manifest["split"].eq("train")].copy()
    validation_ids = set(manifest.loc[manifest["split"].eq("validation"), "sample_id"].astype(str))
    test_ids = set(manifest.loc[manifest["split"].eq("test"), "sample_id"].astype(str))
    if assignments["speaker_id"].duplicated().any():
        raise ValueError("Duplicate B5 speaker assignment")
    joined = train.merge(assignments, on="speaker_id", how="left", validate="many_to_one")
    if joined[["main_client_id", "stress_client_id"]].isna().any().any() or len(joined) != 3681:
        raise ValueError("Incomplete B5 client assignment")
    if set(joined["sample_id"].astype(str)) & validation_ids or set(joined["sample_id"].astype(str)) & test_ids:
        raise ValueError("Federated clients contain validation/test samples")
    scenarios: dict[str, Any] = {}
    for scenario, column in (("main", "main_client_id"), ("stress", "stress_client_id")):
        rows = []
        for client_id, group in joined.groupby(column, sort=True):
            rows.append({
                "client_id": str(client_id), "samples": int(len(group)),
                "emotion_distribution": group["emotion"].value_counts().sort_index().to_dict(),
                "accent_distribution": group["accent_clean"].value_counts().sort_index().to_dict(),
            })
        scenarios[scenario] = {"number_of_clients": len(rows), "clients": rows}
    report = {
        "status": "PASS", "train_samples": len(joined),
        "federated_train_intersection_validation": 0,
        "federated_train_intersection_test": 0, "duplicate_client_assignment": 0,
        "client_manifest": CLIENT_MANIFEST.relative_to(ROOT).as_posix(),
        "client_manifest_sha256": sha256_file(CLIENT_MANIFEST),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST), "scenarios": scenarios,
        "test_samples_loaded": 0,
    }
    output = ROOT / "artifacts/fedecai/federated_client_preflight.json"
    write_json_atomic(report, output)
    return report


def client_frames(train: pd.DataFrame, scenario: str) -> dict[str, pd.DataFrame]:
    assignments = pd.read_csv(CLIENT_MANIFEST, dtype={"speaker_id": "string"})
    column = f"{scenario}_client_id"
    joined = train.merge(assignments[["speaker_id", column]], on="speaker_id", how="left", validate="many_to_one")
    if joined[column].isna().any():
        raise ValueError(f"Missing {scenario} client assignment")
    return {str(client): group.reset_index(drop=True) for client, group in joined.groupby(column, sort=True)}


def aggregate_states(states: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    if not states or len(states) != len(weights) or sum(weights) <= 0:
        raise ValueError("Invalid FedAvg inputs")
    total = float(sum(weights)); output: dict[str, torch.Tensor] = {}
    for key in states[0]:
        if states[0][key].is_floating_point():
            value = sum(state[key].float() * (weight / total) for state, weight in zip(states, weights))
            output[key] = value.to(dtype=states[0][key].dtype)
        else:
            output[key] = states[0][key].clone()
    return output


def local_update(
    global_state: dict[str, torch.Tensor], client: pd.DataFrame, config: dict[str, Any],
    frames: int, device: torch.device, seed: int, global_emotion_weights: torch.Tensor,
    *, smoke: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    seed_everything(seed)
    model = build_model(config); model.load_state_dict(global_state); model.to(device).train()
    loader, _ = make_loaders(client, client.head(min(12, len(client))), config, frames, smoke=smoke)
    optimizer = optimizer_for(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["mixed_precision"]))
    emotion_sum = accent_sum = proximal_sum = 0.0; seen = 0
    reference = {key: value.detach().to(device) for key, value in global_state.items() if value.is_floating_point()}
    for _ in range(1 if smoke else int(config["local_epochs"])):
        for values, emotions, accents in loader:
            values, emotions, accents = values.to(device), emotions.to(device), accents.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                emotion_logits, accent_logits = model(values, float(config.get("lambda_adv", 0.0)))
                emotion_loss = nn.functional.cross_entropy(emotion_logits, emotions, weight=global_emotion_weights)
                accent_loss = torch.zeros((), device=device)
                if accent_logits is not None:
                    accent_loss = nn.functional.cross_entropy(accent_logits, accents)
                proximal = torch.zeros((), device=device)
                if config.get("method") == "fedprox":
                    for name, parameter in model.named_parameters():
                        proximal = proximal + torch.sum((parameter - reference[name]) ** 2)
                    proximal = 0.5 * float(config["fedprox_mu"]) * proximal
                loss = emotion_loss + accent_loss + proximal
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Non-finite federated local loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            scaler.step(optimizer); scaler.update()
            n = len(emotions); seen += n
            emotion_sum += float(emotion_loss.item()) * n
            accent_sum += float(accent_loss.item()) * n
            proximal_sum += float(proximal.item()) * n
            if smoke:
                break
        if smoke:
            break
    result = state_copy(model)
    del model
    return result, {
        "emotion_loss": emotion_sum / max(seen, 1),
        "accent_loss": accent_sum / max(seen, 1),
        "proximal_loss": proximal_sum / max(seen, 1),
    }


def run_federated(config_path: Path, config: dict[str, Any], device: torch.device, *, smoke: bool) -> dict[str, Any]:
    preflight = federated_client_preflight()
    if preflight["status"] != "PASS":
        raise RuntimeError("B5 federated client preflight failed")
    root = ROOT / config["artifact_root"]; root.mkdir(parents=True, exist_ok=True)
    train, validation = load_development_frame(); frames = target_frames(train, int(config["max_frames"]))
    _, validation_loader = make_loaders(train, validation, config, frames, smoke=False)
    global_weights = emotion_weights(train, config["emotion_classes"], device)
    seed_everything(int(config["seed"]))
    initial_model = build_model(config)
    initial_state = state_copy(initial_model)
    initial_e1_hash = sha256_file(E1_CHECKPOINT)
    if smoke:
        clients = client_frames(train, "main")
        client_id = sorted(clients)[0]
        updated, losses = local_update(initial_state, clients[client_id], config, frames, device, int(config["seed"]), global_weights, smoke=True)
        valid = all(bool(torch.isfinite(value).all()) for value in updated.values() if value.is_floating_point())
        if not valid:
            raise RuntimeError("Non-finite federated smoke state")
        audit = {
            "status": "SMOKE_PASS", "experiment_id": config["experiment_id"],
            "scenario": "main", "client": client_id, "losses": losses,
            "backbone": "resnet18_mel3", "input_channels": 3,
            "feature_dim": initial_model.feature_dim,
            "initial_global_checkpoint_sha256": initial_e1_hash,
            "finite_state": valid, "test_samples_loaded": 0,
        }
        write_json_atomic(audit, root / "smoke_audit.json")
        registry(config, config_path, "SMOKE_PASS", client_partition_version="main_mixed_accent_v2", notes="One-client one-batch federated smoke only; fixed test untouched.")
        return audit

    write_json_atomic(config, root / "config.json")
    registry(config, config_path, "RUNNING", client_partition_version="main_mixed_accent_v2+stress_accent_domain_v2")
    all_history: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    scenario_results: dict[str, Any] = {}
    total_communication = 0
    with TrainingTimer() as timer:
        for scenario_index, scenario in enumerate(config["scenarios"]):
            clients = client_frames(train, scenario)
            global_model = build_model(config); global_model.load_state_dict(initial_state); global_model.to(device)
            best_score, best_round, best_state = -1.0, 0, None
            scenario_communication = 0
            for round_index in range(1, int(config["global_rounds"]) + 1):
                client_states, weights, losses = [], [], {}
                round_started = time.perf_counter()
                for client_offset, client_id in enumerate(sorted(clients)):
                    state, client_loss = local_update(
                        state_copy(global_model), clients[client_id], config, frames, device,
                        int(config["seed"]) + scenario_index * 100_000 + round_index * 100 + client_offset,
                        global_weights, smoke=False,
                    )
                    client_states.append(state); weights.append(len(clients[client_id])); losses[client_id] = client_loss
                global_state = aggregate_states(client_states, weights)
                global_model.load_state_dict(global_state); global_model.to(device).eval()
                result = evaluate_model(global_model, validation_loader, device, float(config.get("lambda_adv", 0.0)))
                truth, predicted = result["emotion_truth"], result["emotion_predicted"]
                labels = list(range(len(config["emotion_classes"])))
                score = float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0))
                uar = float(recall_score(truth, predicted, labels=labels, average="macro", zero_division=0))
                text_true = np.asarray(config["emotion_classes"])[truth]
                text_pred = np.asarray(config["emotion_classes"])[predicted]
                accent_values = validation["accent_clean"].astype(str).to_numpy()
                accent_scores = {
                    accent: float(f1_score(text_true[accent_values == accent], text_pred[accent_values == accent], labels=config["emotion_classes"], average="macro", zero_division=0))
                    for accent in config["accent_classes"]
                }
                transferred = 2 * state_bytes(global_state) * len(client_states)
                scenario_communication += transferred; total_communication += transferred
                row = {
                    "scenario": scenario, "round": round_index,
                    "selected_clients": json.dumps(sorted(clients)),
                    "client_sample_counts": json.dumps({key: len(value) for key, value in clients.items()}, sort_keys=True),
                    "client_losses": json.dumps(losses, sort_keys=True),
                    "global_validation_macro_f1": score, "global_uar": uar,
                    "north_macro_f1": accent_scores["north"], "central_macro_f1": accent_scores["central"],
                    "south_macro_f1": accent_scores["south"], "worst_accent_macro_f1": min(accent_scores.values()),
                    "accent_gap": max(accent_scores.values()) - min(accent_scores.values()),
                    "round_time_sec": time.perf_counter() - round_started,
                    "communication_bytes": transferred,
                }
                all_history.append(row)
                print(f"{config['experiment']} scenario={scenario} round={round_index} macro_f1={score:.6f} worst={row['worst_accent_macro_f1']:.6f}", flush=True)
                if score > best_score:
                    best_score, best_round, best_state = score, round_index, state_copy(global_model)
            if best_state is None:
                raise RuntimeError(f"No best checkpoint for {scenario}")
            last_state = state_copy(global_model)
            best_path, last_path = root / f"best_{scenario}.pt", root / f"last_{scenario}.pt"
            save_checkpoint(best_path, global_model, config, best_state, "best_global", scenario)
            save_checkpoint(last_path, global_model, config, last_state, "last_global", scenario)
            global_model.load_state_dict(best_state); global_model.eval()
            best_result = evaluate_model(global_model, validation_loader, device, float(config.get("lambda_adv", 0.0)))
            scenario_prediction = root / f"predictions_{scenario}.csv"
            persist_predictions(best_result, validation, config, scenario_prediction)
            scenario_metrics_path = root / f"metrics_{scenario}.json"
            scenario_metrics = evaluate_prediction_file(
                scenario_prediction, scenario_metrics_path,
                expected_sample_ids=validation["sample_id"].astype(str), expected_split="validation",
                dataset_version=config["dataset_version"], split_version=config["split_version"], costs=None,
            )
            discriminator = accent_discriminator_metrics(best_result, config["accent_classes"])
            scenario_metrics["accent_discriminator_macro_f1"] = discriminator["macro_f1"]
            scenario_metrics["accent_discriminator_uar"] = discriminator["uar"]
            write_json_atomic(scenario_metrics, scenario_metrics_path)
            scenario_results[scenario] = {
                "best_round": best_round, "metrics": scenario_metrics,
                "prediction_path": scenario_prediction.relative_to(ROOT).as_posix(),
                "prediction_sha256": sha256_file(scenario_prediction),
                "best_checkpoint": best_path.relative_to(ROOT).as_posix(), "best_sha256": sha256_file(best_path),
                "last_checkpoint": last_path.relative_to(ROOT).as_posix(), "last_sha256": sha256_file(last_path),
                "communication_bytes": scenario_communication,
            }
            checkpoint_rows.extend([
                {"scenario": scenario, "kind": "best_global", "path": best_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(best_path)},
                {"scenario": scenario, "kind": "last_global", "path": last_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(last_path)},
            ])
    main_state = torch.load(ROOT / scenario_results["main"]["best_checkpoint"], map_location="cpu", weights_only=False)["model_state"]
    final_model = build_model(config); final_model.load_state_dict(main_state); final_model.to(device).eval()
    main_result = evaluate_model(final_model, validation_loader, device, float(config.get("lambda_adv", 0.0)))
    prediction_path = root / "predictions.csv"; persist_predictions(main_result, validation, config, prediction_path)
    best_main = ROOT / scenario_results["main"]["best_checkpoint"]
    counts = parameter_count(final_model); size = model_size(best_main)
    probe = next(iter(validation_loader))[0][:1].to(device)
    latency = benchmark_inference_latency(
        lambda: final_model(probe, float(config.get("lambda_adv", 0.0))), device=device,
        batch_size=1, precision="float32", warmup_runs=10, timed_runs=50, preprocessing_included=False,
    )
    costs = {
        **counts, **size, **timer.as_dict(), **latency,
        "communication_rounds": int(config["global_rounds"]),
        "total_communication_bytes": total_communication,
        "bytes_per_round": total_communication / (int(config["global_rounds"]) * len(config["scenarios"])),
        "total_federated_training_time_sec": timer.training_time_sec,
        "cost_protocol": {"model_size": "best main-scenario deployable checkpoint", "latency_path": "model-ready 3-channel Mel -> ResNet18 -> active heads", "communication": "measured tensor bytes for server-to-client plus client-to-server state transfer"},
    }
    metrics_path = root / "metrics.json"
    metrics = evaluate_prediction_file(
        prediction_path, metrics_path, expected_sample_ids=validation["sample_id"].astype(str),
        expected_split="validation", dataset_version=config["dataset_version"],
        split_version=config["split_version"], costs=costs,
    )
    discriminator = accent_discriminator_metrics(main_result, config["accent_classes"])
    metrics["accent_discriminator_macro_f1"] = discriminator["macro_f1"]
    metrics["accent_discriminator_uar"] = discriminator["uar"]
    write_json_atomic(metrics, metrics_path)
    pd.DataFrame(all_history).to_csv(root / "round_history.csv", index=False)
    write_json_atomic(costs, root / "resource_metrics.json")
    checkpoint_manifest = {
        "initial_global_checkpoint": E1_CHECKPOINT.relative_to(ROOT).as_posix(),
        "initial_global_checkpoint_sha256": initial_e1_hash,
        "checkpoints": checkpoint_rows,
    }
    write_json_atomic(checkpoint_manifest, root / "checkpoint_manifest.json")
    replay = build_model(config); replay.load_state_dict(torch.load(best_main, map_location="cpu", weights_only=False)["model_state"]); replay.to(device).eval()
    replay_result = evaluate_model(replay, validation_loader, device, float(config.get("lambda_adv", 0.0)))
    reproducible = bool(np.array_equal(replay_result["emotion_predicted"], main_result["emotion_predicted"]) and np.allclose(replay_result["probabilities"], main_result["probabilities"], atol=1e-6))
    audit = {
        "status": "PASS" if reproducible else "FAIL", "experiment_id": config["experiment_id"],
        "federated_client_contract_status": preflight["status"], "validation_samples": len(validation),
        "checkpoint_replay": reproducible, "initial_global_checkpoint_sha256": initial_e1_hash,
        "prediction_sha256": sha256_file(prediction_path), "test_samples_loaded": 0,
    }
    write_json_atomic(audit, root / "audit.json")
    if not reproducible:
        raise RuntimeError("Predictions cannot be reproduced from federated checkpoint")
    report = {
        "status": "COMPLETE", "experiment_id": config["experiment_id"],
        "primary_scenario": "main", "scenario_results": scenario_results,
        "metrics": metrics, "resources": costs, "lineage": checkpoint_manifest,
        "audit": audit, "test_samples_loaded": 0,
    }
    write_json_atomic(report, root / "run_report.json")
    registry(config, config_path, "COMPLETE", client_partition_version="main_mixed_accent_v2+stress_accent_domain_v2", hardware=costs["gpu_name"], training_time_sec=costs["training_time_sec"], inference_latency_ms=costs["inference_latency_ms"], model_size_mb=costs["model_size_mb"], overall_macro_f1=metrics["overall_macro_f1"], worst_accent_macro_f1=metrics["worst_accent_macro_f1"], accent_gap=metrics["accent_gap"], checkpoint_path=best_main.relative_to(ROOT).as_posix(), prediction_path=prediction_path.relative_to(ROOT).as_posix())
    return report


def e1_compatibility(device: torch.device) -> dict[str, Any]:
    _, config = load_config("E2")
    train, validation = load_development_frame(); frames = target_frames(train, int(config["max_frames"]))
    subset = validation.sort_values("sample_id").head(8).reset_index(drop=True)
    _, loader = make_loaders(train.head(12), subset, config, frames, smoke=False)
    state, checkpoint = load_e1_state()
    from torchvision import models
    old_model = models.resnet18(weights=None)
    feature_dim = int(old_model.fc.in_features)
    old_model.fc = nn.Sequential(nn.Dropout(float(checkpoint["config"]["dropout"])), nn.Linear(feature_dim, 4))
    old_model.load_state_dict(state, strict=True); old_model.to(device).eval()
    new_model = ResNet18EmotionAccentModel(adversary="none", dropout=float(checkpoint["config"]["dropout"]))
    new_model.emotion_model.network.load_state_dict(state, strict=True); new_model.to(device).eval()
    old_values, new_values, old_predictions, new_predictions = [], [], [], []
    with torch.inference_mode():
        for values, _, _ in loader:
            values = values.to(device)
            old_logits = old_model(values)
            new_logits, _ = new_model(values)
            old_values.append(old_logits.cpu()); new_values.append(new_logits.cpu())
            old_predictions.append(old_logits.argmax(1).cpu()); new_predictions.append(new_logits.argmax(1).cpu())
    old_logits = torch.cat(old_values); new_logits = torch.cat(new_values)
    equivalent = bool(torch.allclose(old_logits, new_logits, rtol=1e-5, atol=1e-6))
    same_predictions = bool(torch.equal(torch.cat(old_predictions), torch.cat(new_predictions)))
    report = {
        "status": "PASS" if equivalent and same_predictions else "FAIL",
        "resnet_feature_interface_status": "PASS" if new_model.feature_dim == feature_dim == 512 else "FAIL",
        "e1_checkpoint_compatibility_status": "PASS" if equivalent and same_predictions else "FAIL",
        "samples": len(subset), "feature_dim": new_model.feature_dim,
        "max_abs_logit_difference": float(torch.max(torch.abs(old_logits - new_logits))),
        "same_predictions": same_predictions, "initial_checkpoint_sha256": sha256_file(E1_CHECKPOINT),
        "sample_ids": subset["sample_id"].astype(str).tolist(), "test_samples_loaded": 0,
    }
    output = ROOT / "artifacts/fedecai/e1_checkpoint_compatibility.json"
    write_json_atomic(report, output)
    if report["status"] != "PASS":
        raise RuntimeError("E1 checkpoint compatibility failed")
    return report


def grl_preflight() -> dict[str, Any]:
    values = torch.tensor([1.0, -2.0], requires_grad=True)
    upstream = torch.tensor([2.0, -4.0])
    output = gradient_reverse(values, 0.25)
    forward_identity = bool(torch.equal(output, values))
    (output * upstream).sum().backward()
    expected = -0.25 * upstream
    direction_and_scale = bool(torch.allclose(values.grad, expected))
    finite = bool(torch.isfinite(values.grad).all())
    zero = torch.tensor([3.0], requires_grad=True)
    gradient_reverse(zero, 0.0).sum().backward()
    lambda_zero = float(zero.grad.item()) == 0.0
    report = {
        "status": "PASS" if all((forward_identity, direction_and_scale, finite, lambda_zero)) else "FAIL",
        "forward_identity": forward_identity,
        "gradient_direction_and_lambda_scaling": direction_and_scale,
        "lambda_zero_behavior": lambda_zero,
        "finite_gradients": finite,
        "observed_gradient": values.grad.tolist(),
        "expected_gradient": expected.tolist(),
        "loss_contract": "positive emotion CE + positive accent CE; reversal/scaling occurs only inside GRL",
        "test_samples_loaded": 0,
    }
    write_json_atomic(report, ROOT / "artifacts/fedecai/grl_test_report.json")
    if report["status"] != "PASS":
        raise RuntimeError("GRL analytical preflight failed")
    return report


def centralized_gate() -> dict[str, Any]:
    e1 = json.loads((ROOT / "artifacts/fedecai/e1_checkpoint_compatibility.json").read_text(encoding="utf-8"))
    grl = json.loads((ROOT / "artifacts/fedecai/grl_test_report.json").read_text(encoding="utf-8"))
    roots = {
        "E2": ROOT / "artifacts/fedecai/e2_resnet18_unconditional_grl",
        "E3": ROOT / "artifacts/fedecai/e3_resnet18_conditional_grl",
    }
    checks: dict[str, Any] = {
        "resnet_feature_interface_status": e1.get("resnet_feature_interface_status"),
        "e1_checkpoint_compatibility_status": e1.get("e1_checkpoint_compatibility_status"),
        "grl_test_status": grl.get("status"), "test_samples_loaded": 0,
    }
    memberships = []
    for step, root in roots.items():
        report = json.loads((root / "run_report.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(root / "predictions.csv", usecols=["sample_id"], dtype={"sample_id": "string"})
        membership = set(predictions["sample_id"].astype(str)); memberships.append(membership)
        checks[f"{step.lower()}_status"] = report.get("status")
        checks[f"{step.lower()}_artifact_integrity"] = audit.get("status")
        checks[f"{step.lower()}_prediction_rows"] = len(predictions)
        checks["test_samples_loaded"] += int(report.get("test_samples_loaded", 0)) + int(audit.get("test_samples_loaded", 0))
    checks["same_validation_membership"] = len(memberships) == 2 and memberships[0] == memberships[1] and len(memberships[0]) == 646
    passed = (
        checks["resnet_feature_interface_status"] == "PASS"
        and checks["e1_checkpoint_compatibility_status"] == "PASS"
        and checks["grl_test_status"] == "PASS"
        and checks["e2_status"] == checks["e3_status"] == "COMPLETE"
        and checks["e2_artifact_integrity"] == checks["e3_artifact_integrity"] == "PASS"
        and checks["same_validation_membership"] and checks["test_samples_loaded"] == 0
    )
    report = {"status": "PASS" if passed else "FAIL", **checks}
    write_json_atomic(report, ROOT / "artifacts/fedecai/centralized_hard_gate.json")
    if not passed:
        raise RuntimeError("Centralized hard gate failed; E4-E7 forbidden")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(CONFIGS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", choices=("e1", "grl", "central", "clients"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in DAP301m")
    device = torch.device(args.device)
    if args.preflight == "e1":
        print(json.dumps(e1_compatibility(device), indent=2)); return
    if args.preflight == "grl":
        print(json.dumps(grl_preflight(), indent=2)); return
    if args.preflight == "central":
        print(json.dumps(centralized_gate(), indent=2)); return
    if args.preflight == "clients":
        print(json.dumps(federated_client_preflight(), indent=2)); return
    if not args.experiment:
        raise ValueError("--experiment or --preflight is required")
    config_path, config = load_config(args.experiment)
    if args.experiment in {"E2", "E3"}:
        result = run_centralized(config_path, config, device, smoke=args.smoke)
    else:
        result = run_federated(config_path, config, device, smoke=args.smoke)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
