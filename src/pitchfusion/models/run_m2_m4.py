"""Train M2, M3, and M4 under one fixed comparison contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from pitchfusion.models.training import (
    AudioDataset,
    collate_audio,
    load_experiment_data,
    load_prosody_contract,
    publish_neural_result,
    seed_everything,
    train_with_resume,
)
from pitchfusion.models.training import sha256_file
from pitchfusion.models.architectures import (
    PitchFusionClassifier,
    Wav2Vec2EmotionClassifier,
    freeze_feature_encoder,
)


ROOT = Path(__file__).resolve().parents[3]
MODEL_NAMES = ("m2_wav2vec2", "m3_pitch_fusion", "m4_multi_prosody")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "seed",
        "split_version",
        "emotion_order",
        "models",
        "backbone",
        "architecture",
        "training",
        "comparison_contract",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config missing fields: {sorted(missing)}")
    if tuple(config["models"]) != MODEL_NAMES:
        raise ValueError("The approved model order must remain M2, M3, M4")
    if config["comparison_contract"].get("m5_status") != (
        "not_implemented_pending_human_decision"
    ):
        raise ValueError("M5 must remain outside this implementation")
    return config


def build_backbone(config: dict[str, Any], allow_download: bool) -> Wav2Vec2Model:
    backbone_config = config["backbone"]
    snapshot = resolve(backbone_config["local_snapshot"])
    source = str(snapshot) if snapshot.is_dir() else backbone_config["model_id"]
    backbone = Wav2Vec2Model.from_pretrained(
        source,
        revision=None if snapshot.is_dir() else backbone_config["revision"],
        local_files_only=not allow_download,
    )
    if int(backbone.config.hidden_size) != int(config["architecture"]["hidden_size"]):
        raise ValueError("Wav2Vec2 hidden size does not match the locked architecture")
    if backbone_config["freeze_feature_encoder"]:
        freeze_feature_encoder(backbone)
    return backbone


def build_model(
    name: str, config: dict[str, Any], allow_download: bool
) -> torch.nn.Module:
    backbone = build_backbone(config, allow_download)
    architecture = config["architecture"]
    common = {
        "projection_size": int(architecture["classifier_projection_size"]),
        "classes": len(config["emotion_order"]),
        "dropout": float(architecture["dropout"]),
    }
    if name == "m2_wav2vec2":
        return Wav2Vec2EmotionClassifier(backbone, **common)
    channels = 1 if name == "m3_pitch_fusion" else 5
    return PitchFusionClassifier(
        backbone,
        prosody_channels=channels,
        attention_heads=int(architecture["attention_heads"]),
        **common,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/models/m2_m4_pitch_fusion_v1.json",
    )
    parser.add_argument(
        "--manifest", default="data/manifests/split_manifest_v2_enriched.csv"
    )
    parser.add_argument(
        "--raw-parquet", default="data/raw/visec_hf/train-00000-of-00001.parquet"
    )
    parser.add_argument(
        "--prosody-index",
        default="data/interim/features/multi_prosody_v1/feature_index.csv",
    )
    parser.add_argument(
        "--prosody-statistics",
        default="data/interim/features/multi_prosody_v1/normalization_train.json",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    manifest_path = resolve(args.manifest)
    raw_path = resolve(args.raw_parquet)
    config = load_config(config_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu only for a small check")
    device = torch.device(args.device)
    train, validation, audio_bytes, encoder = load_experiment_data(
        manifest_path, raw_path, config["emotion_order"]
    )
    y_train = encoder.transform(train["emotion"])
    y_validation = encoder.transform(validation["emotion"])
    prosody_paths: dict[str, Path] | None = None
    prosody_mean: np.ndarray | None = None
    prosody_std: np.ndarray | None = None
    if any(name != "m2_wav2vec2" for name in args.models):
        prosody_paths, prosody_mean, prosody_std = load_prosody_contract(
            ROOT, resolve(args.prosody_index), resolve(args.prosody_statistics)
        )
        missing_ids = (set(train["sample_id"].astype(str)) | set(validation["sample_id"].astype(str))) - set(prosody_paths)
        if missing_ids:
            raise ValueError(f"Missing prosody artifact for sample: {sorted(missing_ids)[0]}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "models": args.models,
                "device": str(device),
                "train": len(train),
                "validation": len(validation),
                "test_samples_used": 0,
                "m5_implemented": False,
            },
            indent=2,
        ),
        flush=True,
    )

    training = config["training"]
    loader_options = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_audio,
    }
    report_dir = ROOT / "reports/m1_m4"
    artifact_dir = ROOT / "artifacts/m1_m4/pitch_fusion"
    for position, name in enumerate(args.models, start=1):
        seed_everything(int(config["seed"]))
        model = build_model(name, config, args.allow_download)
        channels = 0 if name == "m2_wav2vec2" else 1 if name == "m3_pitch_fusion" else 5
        train_set = AudioDataset(
            train,
            audio_bytes,
            y_train,
            maximum_samples=int(training["max_audio_samples"]),
            prosody_paths=prosody_paths if channels else None,
            prosody_mean=prosody_mean,
            prosody_std=prosody_std,
            prosody_channels=channels,
        )
        validation_set = AudioDataset(
            validation,
            audio_bytes,
            y_validation,
            maximum_samples=int(training["max_audio_samples"]),
            prosody_paths=prosody_paths if channels else None,
            prosody_mean=prosody_mean,
            prosody_std=prosody_std,
            prosody_channels=channels,
        )
        train_loader = DataLoader(train_set, shuffle=True, **loader_options)
        validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)
        report_path = report_dir / f"{name}_report.json"
        model_path = artifact_dir / f"{name}.pt"
        if report_path.exists() and model_path.exists() and not args.force:
            print(f"[{position}/{len(args.models)}] {name}: completed, skipping")
            continue
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "model": name,
                        "parameters": sum(value.numel() for value in model.parameters()),
                        "prosody_channels": channels,
                    }
                )
            )
            continue
        checkpoint_path = artifact_dir / f"{name}_checkpoint.pt"
        checkpoint_metadata = {
            "model": name,
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
            "raw_parquet_sha256": sha256_file(raw_path),
        }
        if channels:
            checkpoint_metadata.update(
                {
                    "prosody_index_sha256": sha256_file(resolve(args.prosody_index)),
                    "prosody_statistics_sha256": sha256_file(
                        resolve(args.prosody_statistics)
                    ),
                }
            )
        result = train_with_resume(
            model,
            train_loader,
            validation_loader,
            device,
            training,
            y_train,
            checkpoint_path,
            checkpoint_metadata,
            force=args.force,
        )
        extra_lineage = {"raw_parquet_sha256": sha256_file(raw_path)}
        if channels:
            extra_lineage.update(
                {
                    "prosody_index_sha256": sha256_file(resolve(args.prosody_index)),
                    "prosody_statistics_sha256": sha256_file(
                        resolve(args.prosody_statistics)
                    ),
                }
            )
        scope = {
            "m2_wav2vec2": "M2 raw waveform Wav2Vec2-only baseline",
            "m3_pitch_fusion": "M3 pitch-only original Pitch-Fusion baseline",
            "m4_multi_prosody": "M4 five-channel multi-prosody feature-gain model",
        }[name]
        publish_neural_result(
            root=ROOT,
            model_name=name,
            model=model,
            model_path=model_path,
            report_path=report_path,
            validation=validation,
            validation_loader=validation_loader,
            encoder=encoder,
            device=device,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            result=result,
            model_scope=scope,
            extra_lineage=extra_lineage,
        )


if __name__ == "__main__":
    main()
