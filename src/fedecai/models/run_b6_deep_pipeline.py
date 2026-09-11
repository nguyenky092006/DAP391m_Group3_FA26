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

from fedecai.models.b6_common import (
    classification_metrics, load_b6_table, load_config, prediction_rows,
    sha256_file, write_csv_atomic, write_json_atomic,
)
from fedecai.models.b6_deep_models import FrozenPitchFusion, LogMelCNN2D, Wav2Vec2EmbeddingMLP


ROOT = Path(__file__).resolve().parents[3]
MODEL_ORDER = ("cnn2d_logmel", "wav2vec2_embedding_mlp", "pitch_fusion")
DISPLAY_NAMES = {
    "cnn2d_logmel": "CNN2D Log-Mel Baseline",
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
    def __init__(self, paths: list[Path], labels: np.ndarray, frames: int) -> None:
        self.paths, self.labels, self.frames = paths, labels, frames

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = np.load(self.paths[index]).astype(np.float32)
        if value.ndim != 2 or value.shape[0] != 80:
            raise ValueError(f"Invalid log-Mel shape {value.shape}: {self.paths[index]}")
        value = (value - value.mean()) / max(float(value.std()), 1e-6)
        output = np.zeros((80, self.frames), dtype=np.float32)
        width = min(value.shape[1], self.frames)
        output[:, :width] = value[:, :width]
        return torch.from_numpy(output[None, :, :]), torch.tensor(self.labels[index], dtype=torch.long)


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
    if name == "cnn2d_logmel":
        train_paths = paths_from(train, "log_mel_artifact_path", root)
        validation_paths = paths_from(validation, "log_mel_artifact_path", root)
        widths = [np.load(path, mmap_mode="r").shape[1] for path in train_paths]
        frames = min(int(np.percentile(widths, 95)), int(model_config["max_frames"]))
        train_set = LogMelDataset(train_paths, labels_train, frames)
        validation_set = LogMelDataset(validation_paths, labels_validation, frames)
        preprocessing = {"logmel_normalization": "per sample", "target_frames": frames}
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
    return FrozenPitchFusion(dropout=dropout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/b6_deep_v1.json")
    parser.add_argument("--features", default="data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    parser.add_argument("--manifest", default="data/manifests/split_manifest_v2_enriched.csv")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, feature_path, manifest_path = (resolve(ROOT, value) for value in (args.config, args.features, args.manifest))
    config = load_config(config_path)
    frame, _ = load_b6_table(feature_path, manifest_path, config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run with .venv-gpu, or explicitly pass --device cpu.")
    device = torch.device(args.device)
    encoder = LabelEncoder().fit(config["emotion_order"])
    train_all = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    labels_validation = encoder.transform(validation["emotion"])
    for name in args.models:
        column = "log_mel_artifact_path" if name == "cnn2d_logmel" else "wav2vec_artifact_path"
        paths_from(frame, column, ROOT)
    print(json.dumps({"status": "PASS", "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "train": len(train_all), "validation": len(validation), "test_loaded_for_training": False}), flush=True)
    if args.preflight_only:
        return

    seed_everything(int(config["seed"]))
    report_dir, artifact_dir = ROOT / "reports/tables/b6", ROOT / "artifacts/b6/deep"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for position, name in enumerate(args.models, 1):
        report_path, model_path = report_dir / f"{name}_report.json", artifact_dir / f"{name}.pt"
        if report_path.exists() and model_path.exists() and not args.force:
            print(f"[{position}/{len(args.models)}] {name}: complete, skipping", flush=True)
            continue
        print(f"[{position}/{len(args.models)}] {name}: grouped cross-validation", flush=True)
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
        predicted, probabilities, _ = evaluate(model, validation_loader, device)
        predicted_text = encoder.inverse_transform(predicted)
        metrics = classification_metrics(validation["emotion"].to_numpy(), predicted_text, validation["accent_clean"].to_numpy(), config["emotion_order"])
        torch.save({"model_state": model.state_dict(), "model": name, "classes": list(encoder.classes_), "preprocessing": preprocessing, "config": base}, model_path)
        write_csv_atomic(prediction_rows(validation, predicted_text, probabilities, config["emotion_order"]), report_dir / f"{name}_validation_predictions.csv")
        report = {
            "schema_version": config["schema_version"], "model": name,
            "display_name": DISPLAY_NAMES[name],
            "model_scope": "CNN over B4 log-Mel" if name == "cnn2d_logmel" else "classifier over frozen B4 Wav2Vec2 embeddings" if name == "wav2vec2_embedding_mlp" else "gated fusion over frozen B4 Wav2Vec2 embeddings and six pitch summaries",
            "is_end_to_end_wav2vec2_finetuning": False,
            "is_exact_paper_pitch_fusion_reproduction": False,
            "selection": {"folds": fold_results, "mean_cv_macro_f1": float(np.mean([row["overall"]["macro_f1"] for row in fold_results])), "selected_epochs": selected_epochs},
            "validation": metrics, "final_epoch_history": final_history,
            "lineage": {"feature_sha256": sha256_file(feature_path), "manifest_sha256": sha256_file(manifest_path), "config_sha256": sha256_file(config_path)},
            "runtime_seconds": time.perf_counter() - started, "device": str(device), "test_samples_loaded": 0,
        }
        write_json_atomic(report, report_path)
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
