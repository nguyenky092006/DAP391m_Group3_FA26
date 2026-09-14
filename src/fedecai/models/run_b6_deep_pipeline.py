"""Train the three remaining B6 neural models with grouped CV and safe resume."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# Allow this file to be executed directly from the repository root without
# requiring callers to set PYTHONPATH=src first.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from fedecai.features.mel_3ch import build_mel_3ch, pad_or_crop_mel_3ch
from fedecai.models.b6_common import (
    classification_metrics, git_state, load_b6_table, load_config,
    prediction_rows, register_experiment, sha256_file, write_csv_atomic,
    write_json_atomic,
)
from fedecai.models.b6_deep_models import (
    FrozenPitchFusion, LogMelCNN2D, MelImageClassifier, Wav2Vec2EmbeddingMLP,
)
from fedecai.models.benchmark_costs import (
    TrainingTimer, benchmark_inference_latency, model_size, parameter_count,
)
from fedecai.models.centralized_evaluator import evaluate_prediction_file


ROOT = Path(__file__).resolve().parents[3]
IMAGE_MODELS = ("resnet18_mel3", "mobilenet_v3_mel3", "vit_b16_mel3")
MODEL_ORDER = (
    "cnn2d_logmel", *IMAGE_MODELS, "wav2vec2_embedding_mlp", "pitch_fusion",
)
DISPLAY_NAMES = {
    "cnn2d_logmel": "CNN2D Log-Mel Baseline",
    "resnet18_mel3": "ResNet-18 3-channel Mel-Spectrogram",
    "mobilenet_v3_mel3": "MobileNetV3-Small 3-channel Mel-Spectrogram",
    "vit_b16_mel3": "ViT-B/16 3-channel Mel-Spectrogram",
    "wav2vec2_embedding_mlp": "Wav2Vec2 Frozen Embedding MLP",
    "pitch_fusion": "Pitch Summary Gated Fusion Ablation",
}


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LogMelDataset(Dataset):
    def __init__(
        self, paths: list[Path], labels: np.ndarray, frames: int,
        mel_3ch: bool = False,
    ) -> None:
        self.paths, self.labels, self.frames = paths, labels, frames
        self.mel_3ch = mel_3ch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = np.load(self.paths[index]).astype(np.float32)
        if value.ndim != 2 or value.shape[0] != 80:
            raise ValueError(f"Invalid log-Mel shape {value.shape}: {self.paths[index]}")
        if self.mel_3ch:
            output = pad_or_crop_mel_3ch(build_mel_3ch(value), self.frames)
        else:
            value = (value - value.mean()) / max(float(value.std()), 1e-6)
            output = np.zeros((1, 80, self.frames), dtype=np.float32)
            width = min(value.shape[1], self.frames)
            output[0, :, :width] = value[:, :width]
        return torch.from_numpy(output), torch.tensor(self.labels[index], dtype=torch.long)


class FusionDataset(Dataset):
    def __init__(self, audio: np.ndarray, pitch: np.ndarray, labels: np.ndarray) -> None:
        self.audio = torch.from_numpy(audio.astype(np.float32))
        self.pitch = torch.from_numpy(pitch.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.audio[index], self.pitch[index], self.labels[index]


def forward(model: nn.Module, batch: list[torch.Tensor] | tuple[torch.Tensor, ...], device: torch.device):
    values = [item.to(device, non_blocking=True) for item in batch]
    return (model(values[0]), values[1]) if len(values) == 2 else (model(*values[:-1]), values[-1])


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions, probabilities, losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch in loader:
            logits, labels = forward(model, batch, device)
            losses.append(float(criterion(logits, labels).item()) * len(labels))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities), sum(losses) / len(loader.dataset)


def train_model(
    model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
    device: torch.device, config: dict[str, Any], labels: np.ndarray,
    prefix: str, fixed_epochs: int | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], int]:
    model.to(device)
    counts = np.bincount(labels, minlength=4).astype(np.float32)
    weights = len(labels) / (4 * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    maximum = fixed_epochs or int(config["max_epochs"])
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, maximum + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, target = forward(model, batch, device)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)
        predicted, _, validation_loss = evaluate(model, validation_loader, device)
        truth = np.concatenate([batch[-1].numpy() for batch in validation_loader])
        score = float(f1_score(truth, predicted, average="macro", zero_division=0))
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_loader.dataset), "validation_loss": validation_loss, "validation_macro_f1": score}
        history.append(row)
        print(f"{prefix} epoch {epoch}/{maximum} train_loss={row['train_loss']:.4f} val_macro_f1={score:.4f}", flush=True)
        if fixed_epochs is not None:
            best_epoch = epoch
            continue
        if score > best_score + float(config["min_delta"]):
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if bool(config.get("early_stopping", True)) and stale >= int(config["patience"]):
                print(f"{prefix} early stopping at epoch {epoch}; best={best_epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def paths_from(frame: pd.DataFrame, column: str, root: Path) -> list[Path]:
    paths = [resolve(root, str(value)) for value in frame[column]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} artifacts; first: {missing[0]}")
    return paths


def build_loaders(
    name: str, train: pd.DataFrame, validation: pd.DataFrame, labels_train: np.ndarray,
    labels_validation: np.ndarray, model_config: dict[str, Any], root: Path,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    batch_size = int(model_config["batch_size"])
    if name == "cnn2d_logmel" or name in IMAGE_MODELS:
        train_paths = paths_from(train, "log_mel_artifact_path", root)
        validation_paths = paths_from(validation, "log_mel_artifact_path", root)
        widths = [np.load(path, mmap_mode="r").shape[1] for path in train_paths]
        frames = min(int(np.percentile(widths, 95)), int(model_config["max_frames"]))
        mel_3ch = name in IMAGE_MODELS
        train_set = LogMelDataset(train_paths, labels_train, frames, mel_3ch=mel_3ch)
        validation_set = LogMelDataset(
            validation_paths, labels_validation, frames, mel_3ch=mel_3ch
        )
        preprocessing = {
            "mel_3ch_mode": "repeat_normalized_log_mel" if mel_3ch else None,
            "normalization": "fixed_-80_0_db_to_0_1" if mel_3ch else "per_sample",
            "channels": 3 if mel_3ch else 1,
            "target_frames": frames,
            "target_frames_selection_scope": "current_training_partition_only",
            "crop": "center" if mel_3ch else "left",
            "pad": "right",
        }
    else:
        wav_columns = sorted(column for column in train if column.startswith("wav2vec2_") and pd.api.types.is_numeric_dtype(train[column]))
        pitch_columns = sorted(column for column in train if column.startswith("pitch_") and pd.api.types.is_numeric_dtype(train[column]))
        if len(wav_columns) != 768 or len(pitch_columns) != 6:
            raise ValueError(f"Expected 768 Wav2Vec2 and 6 pitch features, found {len(wav_columns)} and {len(pitch_columns)}")
        audio_scaler = StandardScaler().fit(train[wav_columns])
        train_audio = audio_scaler.transform(train[wav_columns]).astype(np.float32)
        validation_audio = audio_scaler.transform(validation[wav_columns]).astype(np.float32)
        if name == "wav2vec2_embedding_mlp":
            train_set = TensorDataset(torch.from_numpy(train_audio), torch.from_numpy(labels_train.astype(np.int64)))
            validation_set = TensorDataset(torch.from_numpy(validation_audio), torch.from_numpy(labels_validation.astype(np.int64)))
            preprocessing = {"wav2vec2_scaler_mean": audio_scaler.mean_.tolist(), "wav2vec2_scaler_scale": audio_scaler.scale_.tolist()}
        else:
            pitch_scaler = StandardScaler().fit(train[pitch_columns])
            train_pitch = pitch_scaler.transform(train[pitch_columns]).astype(np.float32)
            validation_pitch = pitch_scaler.transform(validation[pitch_columns]).astype(np.float32)
            train_set = FusionDataset(train_audio, train_pitch, labels_train)
            validation_set = FusionDataset(validation_audio, validation_pitch, labels_validation)
            preprocessing = {
                "wav2vec2_scaler_mean": audio_scaler.mean_.tolist(), "wav2vec2_scaler_scale": audio_scaler.scale_.tolist(),
                "pitch_columns": pitch_columns, "pitch_scaler_mean": pitch_scaler.mean_.tolist(), "pitch_scaler_scale": pitch_scaler.scale_.tolist(),
            }
    loader_options = {"batch_size": batch_size, "num_workers": int(model_config["num_workers"]), "pin_memory": True}
    return DataLoader(train_set, shuffle=True, **loader_options), DataLoader(validation_set, shuffle=False, **loader_options), preprocessing


def make_model(name: str, dropout: float) -> nn.Module:
    if name == "cnn2d_logmel":
        return LogMelCNN2D(dropout=dropout)
    if name == "wav2vec2_embedding_mlp":
        return Wav2Vec2EmbeddingMLP(dropout=dropout)
    if name in IMAGE_MODELS:
        backbones = {
            "resnet18_mel3": "resnet18",
            "mobilenet_v3_mel3": "mobilenet_v3_small",
            "vit_b16_mel3": "vit_b_16",
        }
        return MelImageClassifier(backbones[name], dropout=dropout)
    return FrozenPitchFusion(dropout=dropout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/b6_deep_v1.json")
    parser.add_argument("--features", default="data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    parser.add_argument("--manifest", default="data/manifests/split_manifest_v2_enriched.csv")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, feature_path, manifest_path = (resolve(ROOT, value) for value in (args.config, args.features, args.manifest))
    config = load_config(config_path)
    selected_models = args.models or list(config["models"])
    unknown_for_config = [name for name in selected_models if name not in config["models"]]
    if unknown_for_config:
        raise ValueError(
            f"Models {unknown_for_config} are not declared by {config_path.relative_to(ROOT)}"
        )
    frame, _ = load_b6_table(feature_path, manifest_path, config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run with .venv-gpu, or explicitly pass --device cpu.")
    device = torch.device(args.device)
    encoder = LabelEncoder().fit(config["emotion_order"])
    train_all = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    labels_validation = encoder.transform(validation["emotion"])
    for name in selected_models:
        column = "log_mel_artifact_path" if name == "cnn2d_logmel" or name in IMAGE_MODELS else "wav2vec_artifact_path"
        paths_from(frame, column, ROOT)
    print(json.dumps({"status": "PASS", "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "train": len(train_all), "validation": len(validation), "test_loaded_for_training": False}), flush=True)
    if args.preflight_only:
        return

    seed_everything(int(config["seed"]))
    report_dir, artifact_dir = ROOT / "reports/tables/b6", ROOT / "artifacts/b6/deep"
    checkpoint_dir = ROOT / "artifacts/checkpoints"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for position, name in enumerate(selected_models, 1):
        report_path = report_dir / f"{name}_report.json"
        model_path = checkpoint_dir / f"b6_{name}_seed{config['seed']}.pt"
        if report_path.exists() and model_path.exists() and not args.force:
            print(f"[{position}/{len(args.models)}] {name}: complete, skipping", flush=True)
            continue
        print(f"[{position}/{len(selected_models)}] {name}: grouped cross-validation", flush=True)
        base = {**config["training"], **config["models"][name]}
        progress_path = artifact_dir / f"{name}_cv_progress.json"
        progress = {} if args.force or not progress_path.exists() else json.loads(progress_path.read_text(encoding="utf-8"))
        if args.force:
            # Replace stale fold metadata before training starts, so an
            # interrupted forced rerun cannot resume from the previous run.
            write_json_atomic({"fold_results": [], "histories": {}}, progress_path)
        fold_results = progress.get("fold_results", [])
        histories = progress.get("histories", {})
        completed = {str(row["fold"]) for row in fold_results}
        started = time.perf_counter()
        training_timer = TrainingTimer().__enter__()
        for fold_index, fold in enumerate(sorted(train_all["cv_fold"].unique())):
            if str(fold) in completed:
                print(f"{name} fold {fold}: checkpoint found, skipping", flush=True)
                continue
            fold_train = train_all[train_all["cv_fold"] != fold].reset_index(drop=True)
            fold_validation = train_all[train_all["cv_fold"] == fold].reset_index(drop=True)
            y_train, y_validation = encoder.transform(fold_train["emotion"]), encoder.transform(fold_validation["emotion"])
            train_loader, validation_loader, _ = build_loaders(name, fold_train, fold_validation, y_train, y_validation, base, ROOT)
            seed_everything(int(config["seed"]) + fold_index)
            model = make_model(name, float(base["dropout"]))
            model, history, best_epoch = train_model(model, train_loader, validation_loader, device, base, y_train, f"{name} fold {fold}")
            predicted, _, _ = evaluate(model, validation_loader, device)
            metrics = classification_metrics(encoder.inverse_transform(y_validation), encoder.inverse_transform(predicted), fold_validation["accent_clean"].to_numpy(), config["emotion_order"])
            fold_results.append({"fold": str(fold), "best_epoch": best_epoch, **metrics})
            histories[str(fold)] = history
            write_json_atomic({"fold_results": fold_results, "histories": histories}, progress_path)
        selected_epochs = max(1, int(round(float(np.median([row["best_epoch"] for row in fold_results])))))
        print(f"{name}: final train for {selected_epochs} epochs (median CV best epoch)", flush=True)
        y_train = encoder.transform(train_all["emotion"])
        train_loader, validation_loader, preprocessing = build_loaders(name, train_all, validation, y_train, labels_validation, base, ROOT)
        seed_everything(int(config["seed"]))
        model = make_model(name, float(base["dropout"]))
        model, final_history, _ = train_model(model, train_loader, validation_loader, device, base, y_train, f"{name} final", fixed_epochs=selected_epochs)
        training_timer.__exit__(None, None, None)
        predicted, probabilities, _ = evaluate(model, validation_loader, device)
        predicted_text = encoder.inverse_transform(predicted)
        metrics = classification_metrics(validation["emotion"].to_numpy(), predicted_text, validation["accent_clean"].to_numpy(), config["emotion_order"])
        torch.save({"model_state": model.state_dict(), "model": name, "classes": list(encoder.classes_), "preprocessing": preprocessing, "config": base}, model_path)
        parameter_evidence = parameter_count(model)
        size_evidence = model_size(model_path)
        latency_batch = tuple(item[:1] for item in next(iter(validation_loader)))
        model.eval()
        latency_evidence = benchmark_inference_latency(
            lambda: forward(model, latency_batch, device)[0], device=device,
            batch_size=1, precision="float32", warmup_runs=10, timed_runs=50,
            preprocessing_included=False,
        )
        costs = {
            **parameter_evidence, **size_evidence, **training_timer.as_dict(),
            **latency_evidence, "feature_extraction_time_sec": None,
            "model_training_time_sec": training_timer.training_time_sec,
            "end_to_end_training_pipeline_time_sec": training_timer.training_time_sec,
            "cost_protocol": {
                "feature_extraction": "B4 Mel artifact generation excluded",
                "latency_path": "model-ready Mel tensor -> neural network",
            },
        }
        prediction_path = report_dir / f"{name}_validation_predictions.csv"
        experiment_id = f"b6_{name}_seed{config['seed']}"
        rows = prediction_rows(
            validation, predicted_text, probabilities, config["emotion_order"],
            experiment_id=experiment_id, model=name, seed=int(config["seed"]),
        )
        write_csv_atomic(rows, prediction_path)
        canonical_prediction = ROOT / "artifacts/predictions" / f"{experiment_id}_validation.csv"
        canonical_metrics = ROOT / "artifacts/metrics" / f"{experiment_id}_validation.json"
        write_csv_atomic(rows, canonical_prediction)
        evaluated = evaluate_prediction_file(
            canonical_prediction, canonical_metrics,
            expected_sample_ids=validation["sample_id"].astype(str),
            expected_split="validation", dataset_version="visec_clean_eligible_v1",
            split_version=config["split_version"], costs=costs,
        )
        repository_state = git_state(ROOT)
        report = {
            "schema_version": config["schema_version"], "model": name,
            "display_name": DISPLAY_NAMES[name],
            "model_scope": (
                "CNN over B4 log-Mel" if name == "cnn2d_logmel" else
                "image backbone over shared B4 three-channel Mel" if name in IMAGE_MODELS else
                "classifier over frozen B4 Wav2Vec2 embeddings" if name == "wav2vec2_embedding_mlp" else
                "gated fusion over frozen B4 Wav2Vec2 embeddings and six pitch summaries"
            ),
            "is_end_to_end_wav2vec2_finetuning": False,
            "is_exact_paper_pitch_fusion_reproduction": False,
            "selection": {"folds": fold_results, "mean_cv_macro_f1": float(np.mean([row["overall"]["macro_f1"] for row in fold_results])), "selected_epochs": selected_epochs},
            "validation": metrics, "final_epoch_history": final_history,
            "costs": costs,
            "lineage": {
                "feature_sha256": sha256_file(feature_path),
                "manifest_sha256": sha256_file(manifest_path),
                "config_sha256": sha256_file(config_path),
                "checkpoint_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256_file(model_path),
                **repository_state,
            },
            "runtime_seconds": time.perf_counter() - started, "device": str(device), "test_samples_loaded": 0,
        }
        write_json_atomic(report, report_path)
        register_experiment(ROOT, {
            "experiment_id": f"b6_{name}_dev_v1_{int(time.time())}", "step": "B6",
            "rq": "RQ3", "model": name, "variant": "speaker_grouped_cv_centralized",
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
            "notes": "Development model selection only; final test remained unopened.",
        })
        print(f"{name}: validation Macro-F1={metrics['overall']['macro_f1']:.4f}", flush=True)
    summaries = []
    for name in MODEL_ORDER:
        path = report_dir / f"{name}_report.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            summaries.append({"model": name, "mean_cv_macro_f1": report["selection"]["mean_cv_macro_f1"], "validation_macro_f1": report["validation"]["overall"]["macro_f1"], "validation_uar": report["validation"]["overall"]["uar"]})
    write_json_atomic({"schema_version": config["schema_version"], "models_completed": len(summaries), "results": summaries, "test_untouched": True}, report_dir / "b6_deep_summary.json")
    print(json.dumps({"status": "PASS", "models_completed": [row["model"] for row in summaries], "test_untouched": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
