"""Performance-oriented CNN2D/ResNet18/MobileNetV3 ViSEC reruns.

These are explicitly experimental development runs.  They never load test
waveforms and never overwrite the official lightweight baseline artifacts.
"""

from __future__ import annotations

import argparse
import json
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
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from fedecai.features.mel_3ch import build_mel_3ch, pad_or_crop_mel_3ch
from fedecai.models.b6_common import (
    enforce_b6_training_lock, git_state, load_b6_table, load_config, prediction_rows,
    register_experiment, sha256_file, write_csv_atomic, write_json_atomic,
)
from fedecai.models.b6_deep_models import LogMelCNN2D
from fedecai.models.benchmark_costs import (
    TrainingTimer, benchmark_inference_latency, model_size, parameter_count,
)
from fedecai.models.centralized_evaluator import evaluate_prediction_file


ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("TORCH_HOME", str(ROOT / "data/interim/models/torch"))
SUPPORTED = ("cnn2d_logmel", "resnet18_mel3", "mobilenet_v3_mel3")
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]


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


class PerformanceMelDataset(Dataset):
    def __init__(
        self, paths: list[Path], labels: np.ndarray, frames: int,
        *, three_channel: bool, imagenet_normalization: bool,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.frames = frames
        self.three_channel = three_channel
        self.imagenet_normalization = imagenet_normalization

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        mel = np.load(self.paths[index], allow_pickle=False).astype(np.float32)
        if mel.ndim != 2 or mel.shape[0] != 80 or not np.isfinite(mel).all():
            raise ValueError(f"Invalid log-Mel artifact: {self.paths[index]}")
        if self.three_channel:
            value = torch.from_numpy(pad_or_crop_mel_3ch(build_mel_3ch(mel), self.frames))
            if self.imagenet_normalization:
                value = (value - IMAGE_MEAN) / IMAGE_STD
        else:
            mel = (mel - mel.mean()) / max(float(mel.std()), 1e-6)
            output = np.zeros((1, 80, self.frames), dtype=np.float32)
            width = min(self.frames, mel.shape[1])
            output[0, :, :width] = mel[:, :width]
            value = torch.from_numpy(output)
        return value, torch.tensor(int(self.labels[index]), dtype=torch.long)


def artifact_paths(frame) -> list[Path]:
    paths = [resolve(str(value)) for value in frame["log_mel_artifact_path"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing log-Mel artifact: {missing[0]}")
    return paths


def build_model(name: str, dropout: float) -> tuple[nn.Module, nn.Module, list[nn.Parameter], str]:
    if name == "cnn2d_logmel":
        model = LogMelCNN2D(dropout=dropout)
        return model, model.classifier, list(model.features.parameters()), "random_initialization"
    if name == "resnet18_mel3":
        weight = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weight)
        features = int(model.fc.in_features)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 4))
        return model, model.fc, [p for key, p in model.named_parameters() if not key.startswith("fc.")], weight.name
    if name == "mobilenet_v3_mel3":
        weight = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weight)
        features = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(features, 4)
        model.classifier[2] = nn.Dropout(dropout)
        return model, model.classifier, list(model.features.parameters()), weight.name
    raise ValueError(name)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool):
    model.eval()
    truth, predicted, probabilities = [], [], []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for values, labels in loader:
            values, labels = values.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(values)
                loss = criterion(logits, labels)
            loss_sum += float(loss.item()) * len(labels)
            truth.append(labels.cpu().numpy())
            predicted.append(logits.argmax(1).cpu().numpy())
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return (
        np.concatenate(truth), np.concatenate(predicted),
        np.concatenate(probabilities), loss_sum / len(loader.dataset),
    )


def optimizer_for(
    name: str, model: nn.Module, head: nn.Module, config: dict[str, Any],
) -> torch.optim.Optimizer:
    if name == "cnn2d_logmel":
        return torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
    head_ids = {id(value) for value in head.parameters()}
    backbone = [value for value in model.parameters() if id(value) not in head_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(config["backbone_learning_rate"])},
            {"params": list(head.parameters()), "lr": float(config["head_learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )


def train(
    name: str, model: nn.Module, head: nn.Module, backbone: list[nn.Parameter],
    train_loader: DataLoader, validation_loader: DataLoader, labels_train: np.ndarray,
    config: dict[str, Any], device: torch.device, *, smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], int, dict[str, Any]]:
    amp = device.type == "cuda" and bool(config["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    counts = np.bincount(labels_train, minlength=4).astype(np.float32)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(len(labels_train) / (4 * np.maximum(counts, 1)), device=device)
    )
    history: list[dict[str, Any]] = []
    freeze_schedule: dict[str, Any] = {"head_warmup_epochs": 0, "full_finetune_epochs_run": 0}

    warmup = 0 if name == "cnn2d_logmel" else int(config.get("head_warmup_epochs", 0))
    if smoke:
        warmup = min(warmup, 1)
    if warmup:
        for parameter in backbone:
            parameter.requires_grad = False
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=float(config["head_learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        for epoch in range(1, warmup + 1):
            model.train()
            for values, labels in train_loader:
                values, labels = values.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = criterion(model(values), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                if smoke:
                    break
            truth, predicted, _, validation_loss = evaluate(model, validation_loader, device, amp)
            history.append({
                "stage": "head_warmup", "epoch": epoch,
                "validation_loss": validation_loss,
                "validation_macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            })
        freeze_schedule["head_warmup_epochs"] = warmup

    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = optimizer_for(name, model, head, config)
    maximum = 1 if smoke else int(config["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum)
    accumulation = int(config["gradient_accumulation_steps"])
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, maximum + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, (values, labels) in enumerate(train_loader, 1):
            values, labels = values.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(values), labels) / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.item()) * accumulation * len(labels)
            if smoke:
                break
        truth, predicted, _, validation_loss = evaluate(model, validation_loader, device, amp)
        score = float(f1_score(truth, predicted, average="macro", zero_division=0))
        history.append({
            "stage": "full_finetune" if name != "cnn2d_logmel" else "end_to_end",
            "epoch": epoch, "train_loss": loss_sum / max(1, len(train_loader.dataset)),
            "validation_loss": validation_loss, "validation_macro_f1": score,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        })
        print(f"{name} performance epoch {epoch}/{maximum} val_macro_f1={score:.4f}", flush=True)
        if score > best_score + float(config["min_delta"]):
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if not smoke:
            scheduler.step()
        if not smoke and stale >= int(config["patience"]):
            print(f"{name} early stopping at epoch {epoch}; best={best_epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("Performance training produced no checkpoint")
    freeze_schedule["full_finetune_epochs_run"] = epoch
    freeze_schedule["final_backbone_trainable"] = True
    return history, best_state, best_epoch, freeze_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED, required=True)
    parser.add_argument("--config", default="configs/models/b6_performance_v1.json")
    parser.add_argument("--features", default="data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    parser.add_argument("--manifest", default="data/manifests/emotion_stratified_v1.csv")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_b6_training_lock(ROOT, "performance-oriented image-model training/smoke")
    config_path, feature_path, manifest_path = map(resolve, (args.config, args.features, args.manifest))
    config = load_config(config_path)
    model_config = config[args.model]
    experiment_id = model_config["experiment_id"]
    checkpoint = ROOT / "artifacts/checkpoints" / f"{experiment_id}.pt"
    prediction = ROOT / "artifacts/predictions" / f"{experiment_id}_validation.csv"
    metrics_path = ROOT / "artifacts/metrics" / f"{experiment_id}_validation.json"
    report_path = ROOT / "reports/tables/b6_performance" / f"{experiment_id}_report.json"
    if not args.smoke_test and not args.force and all(path.is_file() for path in (checkpoint, prediction, metrics_path, report_path)):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("lineage", {}).get("config_sha256") == sha256_file(config_path):
            print(json.dumps({"status": "SKIP_COMPLETE", "experiment_id": experiment_id}))
            return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in DAP301m")
    device = torch.device(args.device)
    frame, _ = load_b6_table(feature_path, manifest_path, config)
    train_frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    validation_frame = frame[frame["split"].eq("validation")].reset_index(drop=True)
    encoder = LabelEncoder().fit(config["emotion_order"])
    y_train = encoder.transform(train_frame["emotion"])
    y_validation = encoder.transform(validation_frame["emotion"])
    if args.smoke_test:
        train_frame, validation_frame = train_frame.head(4), validation_frame.head(4)
        y_train, y_validation = y_train[:4], y_validation[:4]
    widths = [np.load(path, mmap_mode="r").shape[1] for path in artifact_paths(train_frame)]
    frames = min(int(np.percentile(widths, 95)), int(model_config["max_frames"]))
    if args.smoke_test:
        frames = min(frames, 64)
    image = args.model != "cnn2d_logmel"
    train_set = PerformanceMelDataset(
        artifact_paths(train_frame), y_train, frames, three_channel=image,
        imagenet_normalization=image and bool(model_config.get("imagenet_normalization")),
    )
    validation_set = PerformanceMelDataset(
        artifact_paths(validation_frame), y_validation, frames, three_channel=image,
        imagenet_normalization=image and bool(model_config.get("imagenet_normalization")),
    )
    options = {"batch_size": min(int(model_config["batch_size"]), len(train_set)), "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **options)
    validation_loader = DataLoader(validation_set, shuffle=False, **options)
    seed_everything(int(config["seed"]))
    model, head, backbone, pretrained_source = build_model(args.model, float(model_config["dropout"]))
    model.to(device)
    with TrainingTimer() as timer:
        history, best_state, best_epoch, freeze_schedule = train(
            args.model, model, head, backbone, train_loader, validation_loader,
            y_train, model_config, device, smoke=args.smoke_test,
        )
    if args.smoke_test:
        print(json.dumps({
            "status": "SMOKE_TEST_ONLY", "model": args.model,
            "experiment_id": experiment_id, "pretrained_source": pretrained_source,
            "full_backbone_train_step": "PASS", "test_samples_loaded": 0,
        }, indent=2))
        return
    model.load_state_dict(best_state); model.to(device).eval()
    truth, predicted, probabilities, validation_loss = evaluate(
        model, validation_loader, device, amp=False,
    )
    predicted_text = encoder.inverse_transform(predicted)
    rows = prediction_rows(
        validation_frame, predicted_text, probabilities, config["emotion_order"],
        experiment_id=experiment_id, model=args.model, seed=int(config["seed"]),
    )
    write_csv_atomic(rows, prediction)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state, "model": args.model,
        "experiment_id": experiment_id, "classes": list(encoder.classes_),
        "config": model_config, "pretrained_source": pretrained_source,
        "preprocessing": {"target_frames": frames, "imagenet_normalization": image},
    }, checkpoint)
    counts = parameter_count(model)
    size = model_size(checkpoint)
    probe, _ = next(iter(validation_loader)); probe = probe[:1].to(device)
    latency = benchmark_inference_latency(
        lambda: model(probe), device=device, batch_size=1, precision="float32",
        warmup_runs=10, timed_runs=50, preprocessing_included=False,
    )
    costs = {
        **counts, **size, **timer.as_dict(), **latency,
        "feature_extraction_time_sec": None,
        "model_training_time_sec": timer.training_time_sec,
        "end_to_end_training_pipeline_time_sec": timer.training_time_sec,
        "cost_protocol": {
            "feature_extraction": "deterministic B4 log-Mel cache excluded",
            "latency_path": "model-ready Mel tensor -> neural network",
        },
    }
    evaluated = evaluate_prediction_file(
        prediction, metrics_path,
        expected_sample_ids=validation_frame["sample_id"].astype(str),
        expected_split="validation", dataset_version=config["dataset_version"],
        split_version=config["split_version"], costs=costs,
    )
    repository = git_state(ROOT)
    report = {
        "status": "PASS", "schema_version": config["schema_version"],
        "experiment_id": experiment_id, "model": args.model,
        "dataset_version": config["dataset_version"], "split_version": config["split_version"],
        "seed": config["seed"], "evaluation_protocol": config["evaluation_protocol"],
        "input_representation": model_config["input_representation"],
        "training_regime": model_config["training_regime"],
        "pretrained_source": pretrained_source, "freeze_unfreeze_schedule": freeze_schedule,
        "optimizer": "AdamW", "scheduler": model_config["scheduler"],
        "best_validation_epoch": best_epoch, "validation_loss": validation_loss,
        "history": history, "validation": evaluated, "costs": costs,
        "test_samples_loaded": 0,
        "lineage": {
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
            "feature_sha256": sha256_file(feature_path),
            "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(checkpoint), **repository,
        },
    }
    write_json_atomic(report, report_path)
    register_experiment(ROOT, {
        "experiment_id": experiment_id, "step": "B6", "rq": "RQ3",
        "model": args.model, "variant": model_config["training_regime"],
        "dataset_version": config["dataset_version"], "split_version": config["split_version"],
        "feature_config": "configs/features/visec_features_v1.json",
        "model_config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "seed": config["seed"], "git_commit": repository["git_commit"],
        "hardware": costs["gpu_name"], "train_time_sec": costs["training_time_sec"],
        "inference_latency_ms": costs["inference_latency_ms"], "model_size_mb": costs["model_size_mb"],
        "overall_macro_f1": evaluated["overall_macro_f1"],
        "worst_accent_macro_f1": evaluated["worst_accent_macro_f1"],
        "accent_gap": evaluated["accent_gap"], "status": "completed",
        "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "prediction_path": str(prediction.relative_to(ROOT)).replace("\\", "/"),
        "notes": "Performance-oriented development rerun; speaker overlap disclosed; fixed test untouched.",
    })
    print(json.dumps({
        "status": "PASS", "experiment_id": experiment_id,
        "validation_macro_f1": evaluated["overall_macro_f1"],
        "best_epoch": best_epoch, "test_untouched": True,
    }, indent=2))


if __name__ == "__main__":
    main()
