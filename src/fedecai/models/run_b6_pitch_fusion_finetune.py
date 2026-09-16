"""Performance-oriented partial Wav2Vec2 plus trainable pitch-fusion rerun."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from fedecai.models.b6_common import (
    enforce_b6_training_lock, git_state, load_b6_table, load_config, prediction_rows,
    register_experiment, sha256_file, write_csv_atomic, write_json_atomic,
)
from fedecai.models.benchmark_costs import (
    TrainingTimer, benchmark_inference_latency, model_size, parameter_count,
)
from fedecai.models.centralized_evaluator import evaluate_prediction_file
from fedecai.models.run_b6_pitch_fusion_paper import (
    PaperAlignedPitchFusion, PitchFusionDataset, collate,
)


ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "pitch_fusion_paper_aligned"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config):
    snapshot = resolve(config["local_snapshot"])
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned Wav2Vec2 snapshot absent: {snapshot}")
    backbone = Wav2Vec2Model.from_pretrained(str(snapshot), local_files_only=True)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if bool(config["feature_extractor_frozen"]):
        backbone.feature_extractor._freeze_parameters()
    layers = backbone.encoder.layers
    requested = int(config["trainable_transformer_layers"])
    if requested < 0 or requested > len(layers):
        raise ValueError(f"Invalid trainable transformer layer count: {requested}")
    if requested:
        for layer in layers[-requested:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    if requested and bool(config.get("gradient_checkpointing")):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = PaperAlignedPitchFusion(backbone, config["architecture"], freeze_backbone=requested == 0)
    schedule = {
        "feature_extractor_frozen": True,
        "audio_encoder_fully_frozen": requested == 0,
        "total_transformer_layers": len(layers),
        "trainable_transformer_layers": requested,
        "trainable_transformer_layer_indices": list(range(len(layers) - requested, len(layers))),
        "pitch_encoder_trainable": True,
        "cross_attention_trainable": True,
        "self_attention_trainable": True,
        "classifier_trainable": True,
        "gradient_checkpointing": bool(config.get("gradient_checkpointing")),
    }
    return model, schedule


def evaluate(model, loader, device, amp):
    model.eval(); truth, predicted, probabilities = [], [], []; loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for audio, pitch, mask, labels in loader:
            audio, pitch, mask, labels = (item.to(device) for item in (audio, pitch, mask, labels))
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(audio, pitch, mask); loss = criterion(logits, labels)
            truth.append(labels.cpu().numpy()); predicted.append(logits.argmax(1).cpu().numpy())
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            loss_sum += float(loss.item()) * len(labels)
    return np.concatenate(truth), np.concatenate(predicted), np.concatenate(probabilities), loss_sum / len(loader.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/b6_performance_v1.json")
    parser.add_argument("--features", default="data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    parser.add_argument("--manifest", default="data/manifests/emotion_stratified_v1.csv")
    parser.add_argument("--raw-parquet", default="data/raw/visec_hf/train-00000-of-00001.parquet")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-audio-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_b6_training_lock(ROOT, "Pitch-Fusion fine-tuning/smoke")
    config_path, features_path, manifest_path, raw_path = map(
        resolve, (args.config, args.features, args.manifest, args.raw_parquet)
    )
    root_config = load_config(config_path); config = dict(root_config[MODEL_NAME])
    if args.max_audio_samples:
        config["max_audio_samples"] = args.max_audio_samples
        config["runtime_override"] = "max_audio_samples_explicit_cli_override"
    experiment_id = config["experiment_id"]
    checkpoint = ROOT / "artifacts/checkpoints" / f"{experiment_id}.pt"
    prediction = ROOT / "artifacts/predictions" / f"{experiment_id}_validation.csv"
    metrics_path = ROOT / "artifacts/metrics" / f"{experiment_id}_validation.json"
    report_path = ROOT / "reports/tables/b6_performance" / f"{experiment_id}_report.json"
    if not args.smoke_test and not args.force and all(path.is_file() for path in (checkpoint, prediction, metrics_path, report_path)):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("lineage", {}).get("config_sha256") == sha256_file(config_path):
            print(json.dumps({"status": "SKIP_COMPLETE", "experiment_id": experiment_id})); return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in DAP301m")
    device = torch.device(args.device)
    frame, _ = load_b6_table(features_path, manifest_path, root_config)
    train_frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    validation_frame = frame[frame["split"].eq("validation")].reset_index(drop=True)
    pitch_paths = [resolve(str(path)) for path in frame["pitch_artifact_path"]]
    missing = [path for path in pitch_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pitch artifact: {missing[0]}")
    audio_rows = pq.read_table(raw_path, columns=["path"])["path"].to_pylist()
    audio = [row["bytes"] for row in audio_rows]
    encoder = LabelEncoder().fit(root_config["emotion_order"])
    y_train = encoder.transform(train_frame["emotion"]); y_validation = encoder.transform(validation_frame["emotion"])
    if args.smoke_test:
        if args.max_audio_samples is not None:
            train_index = max(range(len(train_frame)), key=lambda i: len(audio[int(train_frame.iloc[i]["source_row_index"])]))
            validation_index = max(range(len(validation_frame)), key=lambda i: len(audio[int(validation_frame.iloc[i]["source_row_index"])]))
            train_frame = train_frame.iloc[[train_index]].reset_index(drop=True)
            validation_frame = validation_frame.iloc[[validation_index]].reset_index(drop=True)
            y_train = encoder.transform(train_frame["emotion"])
            y_validation = encoder.transform(validation_frame["emotion"])
        else:
            train_frame, validation_frame = train_frame.head(1), validation_frame.head(1)
            y_train, y_validation = y_train[:1], y_validation[:1]
            config["max_audio_samples"] = min(16000, int(config["max_audio_samples"]))
    options = {"batch_size": 1 if args.smoke_test else int(config["batch_size"]), "num_workers": 0, "pin_memory": device.type == "cuda", "collate_fn": collate}
    train_loader = DataLoader(PitchFusionDataset(train_frame, audio, y_train, int(config["max_audio_samples"])), shuffle=True, **options)
    validation_loader = DataLoader(PitchFusionDataset(validation_frame, audio, y_validation, int(config["max_audio_samples"])), shuffle=False, **options)
    seed_everything(int(root_config["seed"])); model, schedule = build_model(config); model.to(device)
    backbone_parameters = [value for value in model.backbone.parameters() if value.requires_grad]
    optimizer_groups = []
    if backbone_parameters:
        optimizer_groups.append({"params": backbone_parameters, "lr": float(config["backbone_learning_rate"])})
    optimizer_groups.append({"params": model.head.parameters(), "lr": float(config["head_learning_rate"])})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=float(config["weight_decay"]))
    maximum = 1 if args.smoke_test else int(config["max_epochs"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum)
        if config["scheduler"] == "cosine_annealing"
        else torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(config["mixed_precision"]),
        init_scale=1024.0,
    )
    counts = np.bincount(y_train, minlength=4).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(y_train) / (4 * np.maximum(counts, 1)), device=device))
    accumulation = 1 if args.smoke_test else int(config["gradient_accumulation_steps"])
    best_score, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []
    progress_path = ROOT / "artifacts/b6/performance" / f"{experiment_id}_progress.pt"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch, prior_training_time = 1, 0.0
    if progress_path.is_file() and not args.force and not args.smoke_test:
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("config_sha256") != sha256_file(config_path):
            raise RuntimeError("Pitch-Fusion progress checkpoint config hash mismatch; use a new experiment ID")
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        scheduler.load_state_dict(progress["scheduler_state"])
        scaler.load_state_dict(progress["scaler_state"])
        history = progress["history"]
        best_score, best_epoch, stale = progress["best_score"], progress["best_epoch"], progress["stale"]
        start_epoch = int(progress["epoch"]) + 1
        prior_training_time = float(progress.get("accumulated_training_time_sec", 0.0))
        if checkpoint.is_file():
            best_state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
        print(f"pitch-fusion resume from epoch {start_epoch}", flush=True)
    transformer_gradient_observed = False
    session_started = time.perf_counter()
    with TrainingTimer() as timer:
        for epoch in range(start_epoch, maximum + 1):
            model.train(); optimizer.zero_grad(set_to_none=True); total = 0.0
            for step, (waveform, pitch, mask, labels) in enumerate(train_loader, 1):
                waveform, pitch, mask, labels = (item.to(device) for item in (waveform, pitch, mask, labels))
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    loss = criterion(model(waveform, pitch, mask), labels) / accumulation
                scaler.scale(loss).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    if args.smoke_test:
                        transformer_gradient_observed = any(
                            parameter.grad is not None
                            and bool(torch.isfinite(parameter.grad).all().item())
                            and bool(torch.count_nonzero(parameter.grad).item())
                            for parameter in backbone_parameters
                        )
                    nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), float(config["gradient_clip"]))
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                total += float(loss.item()) * accumulation * len(labels)
                if args.smoke_test:
                    break
            truth, predicted, _, validation_loss = evaluate(model, validation_loader, device, scaler.is_enabled())
            score = float(f1_score(truth, predicted, average="macro", zero_division=0))
            validation_uar = float(recall_score(truth, predicted, average="macro", zero_division=0))
            history.append({"epoch": epoch, "train_loss": total / max(1, len(train_loader.dataset)), "validation_loss": validation_loss, "validation_macro_f1": score, "validation_uar": validation_uar, "learning_rates": [float(group["lr"]) for group in optimizer.param_groups]})
            print(f"pitch-fusion performance epoch {epoch}/{maximum} val_macro_f1={score:.4f} val_uar={validation_uar:.4f} val_loss={validation_loss:.4f}", flush=True)
            if score > best_score + float(config["min_delta"]):
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if not args.smoke_test:
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    temporary_best = checkpoint.with_suffix(".pt.tmp")
                    torch.save({"model_state": best_state, "classes": list(encoder.classes_), "config": config, "fine_tune_schedule": schedule}, temporary_best)
                    os.replace(temporary_best, checkpoint)
            else:
                stale += 1
            if not args.smoke_test:
                scheduler.step()
                temporary_progress = progress_path.with_suffix(".tmp")
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(), "history": history,
                    "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
                    "accumulated_training_time_sec": prior_training_time + time.perf_counter() - session_started,
                    "config_sha256": sha256_file(config_path),
                }, temporary_progress)
                os.replace(temporary_progress, progress_path)
            if not args.smoke_test and stale >= int(config["patience"]):
                print(f"pitch-fusion early stopping at epoch {epoch}; best={best_epoch}", flush=True); break
    if best_state is None:
        raise RuntimeError("Pitch-Fusion fine-tuning produced no checkpoint")
    if args.smoke_test:
        if backbone_parameters and not transformer_gradient_observed:
            raise RuntimeError("Smoke test did not observe gradients in trainable Pitch-Fusion Wav2Vec2 layers")
        print(json.dumps({"status": "SMOKE_TEST_ONLY", "experiment_id": experiment_id, **schedule, "tiny_train_step": "PASS", "transformer_gradient_observed": transformer_gradient_observed if backbone_parameters else None, "test_samples_loaded": 0}, indent=2)); return
    model.load_state_dict(best_state); model.to(device).eval()
    _, predicted, probabilities, validation_loss = evaluate(model, validation_loader, device, amp=False)
    predicted_text = encoder.inverse_transform(predicted)
    rows = prediction_rows(validation_frame, predicted_text, probabilities, root_config["emotion_order"], experiment_id=experiment_id, model=MODEL_NAME, seed=int(root_config["seed"]))
    write_csv_atomic(rows, prediction)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "classes": list(encoder.classes_), "config": config, "fine_tune_schedule": schedule}, checkpoint)
    counts_evidence = parameter_count(model); size = model_size(checkpoint)
    waveform, pitch, mask, _ = next(iter(validation_loader)); waveform, pitch, mask = waveform[:1].to(device), pitch[:1].to(device), mask[:1].to(device)
    latency = benchmark_inference_latency(lambda: model(waveform, pitch, mask), device=device, batch_size=1, precision="float32", warmup_runs=10, timed_runs=50, preprocessing_included=False)
    total_training_time = prior_training_time + timer.training_time_sec
    encoder_mode = "frozen" if int(config["trainable_transformer_layers"]) == 0 else "partially fine-tuned"
    costs = {**counts_evidence, **size, **timer.as_dict(), **latency, "training_time_sec": total_training_time, "feature_extraction_time_sec": None, "model_training_time_sec": total_training_time, "end_to_end_training_pipeline_time_sec": total_training_time, "cost_protocol": {"feature_extraction": "raw decode and cached pYIN contour loading occur in training path", "latency_path": f"model-ready waveform and pitch contour -> {encoder_mode} Wav2Vec2 -> fusion network", "resume_supported": True}}
    evaluated = evaluate_prediction_file(prediction, metrics_path, expected_sample_ids=validation_frame["sample_id"].astype(str), expected_split="validation", dataset_version=root_config["dataset_version"], split_version=root_config["split_version"], costs=costs)
    repository = git_state(ROOT)
    report = {"status": "PASS", "schema_version": root_config["schema_version"], "experiment_id": experiment_id, "model": MODEL_NAME, "dataset_version": root_config["dataset_version"], "split_version": root_config["split_version"], "seed": root_config["seed"], "evaluation_protocol": root_config["evaluation_protocol"], "input_representation": config["input_representation"], "training_regime": config["training_regime"], "fine_tune_schedule": schedule, "optimizer": "AdamW", "scheduler": config["scheduler"], "best_validation_epoch": best_epoch, "validation_loss": validation_loss, "history": history, "validation": evaluated, "costs": costs, "exact_historical_reproduction": False, "test_samples_loaded": 0, "lineage": {"config_sha256": sha256_file(config_path), "manifest_sha256": sha256_file(manifest_path), "feature_sha256": sha256_file(features_path), "raw_parquet_sha256": sha256_file(raw_path), "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": sha256_file(checkpoint), **repository}}
    write_json_atomic(report, report_path)
    register_experiment(ROOT, {"experiment_id": experiment_id, "step": "B6", "rq": "RQ3", "model": MODEL_NAME, "variant": config["training_regime"], "dataset_version": root_config["dataset_version"], "split_version": root_config["split_version"], "feature_config": "raw_waveform_16khz_plus_pyin_pitch", "model_config": str(config_path.relative_to(ROOT)).replace("\\", "/"), "seed": root_config["seed"], "git_commit": repository["git_commit"], "hardware": costs["gpu_name"], "train_time_sec": costs["training_time_sec"], "inference_latency_ms": costs["inference_latency_ms"], "model_size_mb": costs["model_size_mb"], "overall_macro_f1": evaluated["overall_macro_f1"], "worst_accent_macro_f1": evaluated["worst_accent_macro_f1"], "accent_gap": evaluated["accent_gap"], "status": "completed", "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "prediction_path": str(prediction.relative_to(ROOT)).replace("\\", "/"), "notes": f"Performance-oriented {encoder_mode} Wav2Vec2 plus fusion training; speaker overlap disclosed; fixed test untouched; not exact historical reproduction."})
    print(json.dumps({"status": "PASS", "experiment_id": experiment_id, "validation_macro_f1": evaluated["overall_macro_f1"], "best_epoch": best_epoch, "test_untouched": True}, indent=2))


if __name__ == "__main__":
    main()
