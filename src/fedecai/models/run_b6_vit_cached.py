"""Centralized frozen-pretrained ViT with one-time Mel embedding caching."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from fedecai.features.mel_3ch import build_mel_3ch, pad_or_crop_mel_3ch
from fedecai.models.b6_common import (
    git_state, load_b6_table, load_config, prediction_rows, register_experiment,
    sha256_file, write_csv_atomic, write_json_atomic,
)
from fedecai.models.benchmark_costs import (
    TrainingTimer, benchmark_inference_latency, model_size, parameter_count,
)
from fedecai.models.centralized_evaluator import evaluate_prediction_file
from fedecai.models.frozen_embedding_cache import (
    validate_embedding_cache, write_embedding_cache,
)
from fedecai.models.frozen_head_training import train_frozen_head


ROOT = Path(__file__).resolve().parents[3]
MODEL_NAME = "vit_b16_mel3"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class MelCacheDataset(Dataset):
    def __init__(self, paths: list[Path], frames: int) -> None:
        self.paths, self.frames = paths, frames

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        mel = np.load(self.paths[index]).astype(np.float32)
        return torch.from_numpy(pad_or_crop_mel_3ch(build_mel_3ch(mel), self.frames))


class FrozenViTClassifier(nn.Module):
    def __init__(self, network: nn.Module, head: nn.Module, config: dict[str, Any]) -> None:
        super().__init__()
        self.network, self.head = network, head
        for parameter in self.network.parameters():
            parameter.requires_grad = False
        self.mean = torch.tensor(config["normalization"]["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(config["normalization"]["std"]).view(1, 3, 1, 1)
        self.image_size = tuple(config["resize"]["size"])

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(values, size=self.image_size, mode="bilinear", align_corners=False)
        values = (values - self.mean.to(values.device)) / self.std.to(values.device)
        sequence = self.network._process_input(values)
        token = self.network.class_token.expand(len(values), -1, -1)
        sequence = self.network.encoder(torch.cat((token, sequence), dim=1))
        return sequence[:, 0]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            embedding = self.encode(values)
        return self.head(embedding)


def build_model(config: dict[str, Any]) -> tuple[FrozenViTClassifier, Path, Any]:
    torch_hub_dir = ROOT / "data/interim/models/torch"
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_hub_dir))
    weights = models.ViT_B_16_Weights[config["weights"]]
    network = models.vit_b_16(weights=weights, image_size=int(config["image_size"]))
    network.heads = nn.Identity()
    from fedecai.models.b6_deep_models import FrozenEmbeddingHead

    head = FrozenEmbeddingHead(
        dimension=int(config["embedding_dimension"]), dropout=float(config["dropout"])
    )
    model = FrozenViTClassifier(network, head, config)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Torchvision weight cache missing after model load: {checkpoint}")
    return model, checkpoint, weights


def cache_metadata(config: dict[str, Any], config_path: Path, manifest_path: Path) -> dict[str, Any]:
    return {
        "cache_version": config["cache_version"],
        "model": MODEL_NAME,
        "dataset_version": "visec_clean_eligible_v1",
        "split_version": "speaker_disjoint_v2",
        "fit_partitions": ["train", "validation"],
        "feature_config": "configs/features/visec_features_v1.json",
        "feature_config_sha256": sha256_file(ROOT / "configs/features/visec_features_v1.json"),
        "model_config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "mel_3ch_mode": "repeat_normalized_log_mel",
        "pretrained_model": f"torchvision.vit_b_16:{config['weights']}",
        "resize": config["resize"],
        "normalization": config["normalization"],
        "pooling": config["pooling"],
        "embedding_dimension": int(config["embedding_dimension"]),
        "encoder_policy": "fully_frozen",
    }


def extract_embeddings(
    model: FrozenViTClassifier, frame, config: dict[str, Any], device: torch.device
) -> tuple[np.ndarray, float]:
    paths = [resolve(str(value)) for value in frame["log_mel_artifact_path"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing B4 log-Mel artifact: {missing[0]}")
    loader = DataLoader(
        MelCacheDataset(paths, int(config["max_frames"])),
        batch_size=int(config["extraction_batch_size"]), shuffle=False,
        num_workers=int(config["num_workers"]), pin_memory=True,
    )
    model.to(device).eval()
    values = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            values.append(model.encode(batch.to(device, non_blocking=True)).cpu().numpy())
            if batch_number % 25 == 0:
                print(f"ViT cache extraction batch {batch_number}/{len(loader)}", flush=True)
    return np.concatenate(values).astype(np.float32), time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/b6_image_models_v1.json")
    parser.add_argument("--features", default="data/interim/features/b4_features_v1/b4_full_feature_table.parquet")
    parser.add_argument("--manifest", default="data/manifests/split_manifest_v2_enriched.csv")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--force-head", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, feature_path, manifest_path = map(
        resolve, (args.config, args.features, args.manifest)
    )
    full_config = load_config(config_path)
    config = full_config["models"][MODEL_NAME]
    if not config.get("freeze_backbone") or not config.get("cache_embeddings"):
        raise ValueError("ViT baseline requires frozen backbone and embedding cache")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    frame, _ = load_b6_table(feature_path, manifest_path, full_config)
    development = frame[frame["split"].isin(("train", "validation"))].reset_index(drop=True)
    if len(development) != 4327:
        raise ValueError(f"Expected 4,327 train+validation samples, got {len(development)}")
    expected = cache_metadata(config, config_path, manifest_path)
    cache_dir = ROOT / "data/interim/features/b6_frozen_embeddings/vit_b16_mel3"

    model, weights_path, _ = build_model(config)
    if args.smoke_test:
        sample = development[development["split"] == "train"].groupby(
            "emotion", sort=True
        ).head(2).reset_index(drop=True)
        smoke_embeddings, _ = extract_embeddings(model, sample, config, device)
        smoke_expected = {
            "cache_version": "smoke_v1", "model": MODEL_NAME,
            "dataset_version": "visec_clean_eligible_v1",
            "split_version": "speaker_disjoint_v2",
        }
        with tempfile.TemporaryDirectory() as directory:
            smoke_dir = Path(directory)
            write_embedding_cache(smoke_dir, sample, smoke_embeddings, smoke_expected)
            reloaded, _ = validate_embedding_cache(smoke_dir, sample, smoke_expected)
        smoke_labels = LabelEncoder().fit(full_config["emotion_order"]).transform(sample["emotion"])
        smoke_config = {**config, "max_epochs": 1, "patience": 1, "batch_size": 4}
        _, _, _, predicted, _ = train_frozen_head(
            reloaded[:4], smoke_labels[:4], reloaded[4:], smoke_labels[4:],
            config=smoke_config, device=device, seed=int(full_config["seed"]),
            prefix="vit_cached_smoke",
        )
        print(json.dumps({
            "status": "SMOKE_TEST_ONLY", "model": MODEL_NAME,
            "frozen_encoder_executed": True, "cache_saved_reloaded": True,
            "embedding_dimension": reloaded.shape[1],
            "prediction_count": len(predicted), "test_samples_loaded": 0,
        }, indent=2))
        return
    cache_reused, extraction_seconds = False, None
    if (cache_dir / "metadata.json").is_file() and not args.rebuild_cache:
        embeddings, metadata = validate_embedding_cache(cache_dir, development, expected)
        cache_reused = True
        extraction_seconds = metadata.get("feature_extraction_time_sec")
        print("ViT frozen embedding cache: validated, reusing", flush=True)
    else:
        embeddings, extraction_seconds = extract_embeddings(model, development, config, device)
        metadata = write_embedding_cache(
            cache_dir, development, embeddings,
            {**expected, "feature_extraction_time_sec": extraction_seconds},
        )
        embeddings, metadata = validate_embedding_cache(cache_dir, development, expected)

    encoder = LabelEncoder().fit(full_config["emotion_order"])
    train_mask = development["split"].eq("train").to_numpy()
    validation_mask = development["split"].eq("validation").to_numpy()
    y_train = encoder.transform(development.loc[train_mask, "emotion"])
    y_validation = encoder.transform(development.loc[validation_mask, "emotion"])

    checkpoint = ROOT / "artifacts/checkpoints" / f"b6_{MODEL_NAME}_seed{full_config['seed']}.pt"
    report_path = ROOT / "reports/tables/b6" / f"{MODEL_NAME}_report.json"
    canonical_metrics = ROOT / "artifacts/metrics" / f"b6_{MODEL_NAME}_seed{full_config['seed']}_validation.json"
    canonical_prediction = ROOT / "artifacts/predictions" / f"b6_{MODEL_NAME}_seed{full_config['seed']}_validation.csv"
    if report_path.is_file() and checkpoint.is_file() and canonical_metrics.is_file() and not args.force_head:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("training_strategy") == "frozen_pretrained_backbone_cached_embeddings":
            print("ViT frozen cached baseline: verified outputs exist, skipping", flush=True)
            return

    with TrainingTimer() as training_timer:
        head, history, best_epoch, predicted, probabilities = train_frozen_head(
            embeddings[train_mask], y_train, embeddings[validation_mask], y_validation,
            config=config, device=device, seed=int(full_config["seed"]),
            prefix="vit_b16_mel3 frozen head",
        )
    model.head.load_state_dict(head.state_dict())
    model.to(device).eval()
    validation = development.loc[validation_mask].reset_index(drop=True)
    predicted_text = encoder.inverse_transform(predicted)
    experiment_id = f"b6_{MODEL_NAME}_seed{full_config['seed']}"
    rows = prediction_rows(
        validation, predicted_text, probabilities, full_config["emotion_order"],
        experiment_id=experiment_id, model=MODEL_NAME, seed=int(full_config["seed"]),
    )
    write_csv_atomic(rows, canonical_prediction)
    write_csv_atomic(rows, ROOT / "reports/tables/b6" / f"{MODEL_NAME}_validation_predictions.csv")

    temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state": head.state_dict(), "backbone_reference": str(weights_path),
        "weights": config["weights"], "classes": list(encoder.classes_),
        "cache_metadata": metadata, "config": config,
    }, temporary_checkpoint)
    size = model_size(temporary_checkpoint, externally_referenced_artifacts=[weights_path])
    size["checkpoint_artifact"] = str(checkpoint.resolve())
    mel_probe = next(iter(DataLoader(
        MelCacheDataset([resolve(str(validation.iloc[0]["log_mel_artifact_path"]))], int(config["max_frames"])),
        batch_size=1,
    ))).to(device)
    latency = benchmark_inference_latency(
        lambda: model(mel_probe), device=device, batch_size=1, precision="float32",
        warmup_runs=10, timed_runs=50, preprocessing_included=False,
    )
    counts = parameter_count(model)
    end_to_end = (
        None if extraction_seconds is None
        else float(extraction_seconds) + training_timer.training_time_sec
    )
    costs = {
        **counts, **size, **training_timer.as_dict(), **latency,
        "feature_extraction_time_sec": extraction_seconds,
        "model_training_time_sec": training_timer.training_time_sec,
        "end_to_end_training_pipeline_time_sec": end_to_end,
        "cost_protocol": {
            "feature_extraction": "one-time frozen pretrained ViT extraction; cache reuse recorded",
            "cache_reused": cache_reused,
            "latency_path": "model-ready 3-channel Mel -> resize/normalize -> frozen ViT -> head",
        },
    }
    evaluated = evaluate_prediction_file(
        canonical_prediction, canonical_metrics,
        expected_sample_ids=validation["sample_id"].astype(str),
        expected_split="validation", dataset_version="visec_clean_eligible_v1",
        split_version=full_config["split_version"], costs=costs,
    )
    os.replace(temporary_checkpoint, checkpoint)
    report = {
        "schema_version": "b6.vit_frozen_cached.v1", "model": MODEL_NAME,
        "display_name": "ViT-B/16 Frozen Pretrained Cached-Embedding Baseline",
        "training_strategy": "frozen_pretrained_backbone_cached_embeddings",
        "model_scope": "3-channel Mel -> frozen pretrained ViT -> CLS embedding -> lightweight head",
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
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(checkpoint), **git_state(ROOT),
        },
    }
    write_json_atomic(report, report_path)
    register_experiment(ROOT, {
        "experiment_id": f"{experiment_id}_frozen_cached", "step": "B6", "rq": "RQ3",
        "model": MODEL_NAME, "variant": "frozen_pretrained_cached_embeddings",
        "dataset_version": "visec_clean_eligible_v1", "split_version": full_config["split_version"],
        "feature_config": "configs/features/visec_features_v1.json",
        "model_config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "seed": full_config["seed"], "git_commit": git_state(ROOT)["git_commit"],
        "hardware": costs["gpu_name"], "train_time_sec": costs["training_time_sec"],
        "inference_latency_ms": costs["inference_latency_ms"], "model_size_mb": costs["model_size_mb"],
        "overall_macro_f1": evaluated["overall_macro_f1"],
        "worst_accent_macro_f1": evaluated["worst_accent_macro_f1"],
        "accent_gap": evaluated["accent_gap"], "status": "completed",
        "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "prediction_path": str(canonical_prediction.relative_to(ROOT)).replace("\\", "/"),
        "notes": "Development validation only; pretrained ViT frozen; cache validated; fixed test untouched.",
    })
    print(json.dumps({
        "status": "PASS", "model": MODEL_NAME, "best_epoch": best_epoch,
        "validation_macro_f1": evaluated["overall_macro_f1"],
        "cache_reused": cache_reused, "test_samples_loaded": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
