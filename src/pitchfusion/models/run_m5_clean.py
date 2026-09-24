"""M5: M4 multi-prosody input with a lightweight residual dilated CNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from pitchfusion.models.run_m4_clean import (
    ARTIFACT_ROOT as M4_ARTIFACT_ROOT,
    FEATURE_ROOT,
    REPORT_ROOT,
    SPLIT,
    M4Dataset,
    MultiChannelFeatureEncoder,
    collate,
    prepare_features,
)
from pitchfusion.models.run_paper_reproduction import (
    DEFAULT_KALDI_EXECUTABLE,
    OfficialPitchFusion,
    atomic_csv,
    atomic_json,
    atomic_torch_save,
    build_clean_stratified_split,
    evaluate,
    load_audio_bytes,
    prediction_frame,
    resolve,
    seed_everything,
    sha256_file,
    train,
)

CONFIG = "configs/models/m5_residual_dilated_clean_v1.json"
ARTIFACT_ROOT = "artifacts/paper_reproduction/m5_residual_dilated"
MODEL_NAME = "m5_residual_dilated_clean"


class LightweightResidualDilatedBlock(nn.Module):
    """Pre-norm bottleneck residual block preserving time and channels."""

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.reduce = nn.Conv1d(channels, bottleneck_channels, kernel_size=1)
        self.depthwise = nn.Conv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=bottleneck_channels,
        )
        self.expand = nn.Conv1d(bottleneck_channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.norm(values)
        values = self.activation(self.reduce(values))
        values = self.activation(self.depthwise(values))
        values = self.dropout(self.expand(values))
        return residual + self.residual_scale * values


class ImprovedProsodyEncoder(nn.Module):
    """Reuse M4's input adapter/CNN and append dilated residual blocks."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        base = MultiChannelFeatureEncoder(config)
        self.input_adapter = base.input_adapter
        self.feature_encoder = base.feature_encoder
        architecture = json.loads(
            resolve(CONFIG).read_text(encoding="utf-8")
        )["m5_architecture"]
        channels = int(config.conv_dim[-1])
        self.residual_blocks = nn.ModuleList(
            LightweightResidualDilatedBlock(
                channels=channels,
                bottleneck_channels=int(architecture["bottleneck_channels"]),
                dilation=int(dilation),
                dropout=float(architecture["dropout"]),
            )
            for dilation in architecture["dilations"]
        )
        self.reset_input_adapter()

    def reset_input_adapter(self) -> None:
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            self.input_adapter.weight[0, 0, 0] = 1.0
            self.input_adapter.bias.zero_()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_adapter(values)
        for conv_layer in self.feature_encoder.conv_layers:
            hidden_states = conv_layer(hidden_states)
        for block in self.residual_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class M5Model(OfficialPitchFusion):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.pitch_encoder = ImprovedProsodyEncoder(config)
        self.post_init()
        self.pitch_encoder.reset_input_adapter()


def build_loaders(
    manifest: Any,
    audio: list[bytes],
    feature_root: Path,
    stats: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, DataLoader]]:
    label_to_id = {
        name: index for index, name in enumerate(config["emotion_order"])
    }
    parts = {
        name: manifest[manifest.split == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    loaders = {
        name: DataLoader(
            M4Dataset(frame, audio, feature_root, label_to_id, stats),
            batch_size=int(config["training"]["batch_size"]),
            shuffle=name == "train",
            num_workers=int(config["training"]["num_workers"]),
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        for name, frame in parts.items()
    }
    return parts, loaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--kaldi-executable", default=DEFAULT_KALDI_EXECUTABLE)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-training", action="store_true")
    args = parser.parse_args()

    config_path = resolve(CONFIG)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = build_clean_stratified_split(
        resolve(config["data"]["clean_manifest"]),
        resolve(SPLIT),
        resolve("reports/paper_reproduction/split_report.json"),
        config,
        force=False,
    )
    audio = load_audio_bytes(resolve(config["data"]["raw_parquet"]))
    feature_root = resolve(FEATURE_ROOT)
    stats = prepare_features(
        manifest,
        audio,
        feature_root,
        resolve(args.kaldi_executable),
        config,
        args.force_features,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    seed_everything(int(config["seed"]))
    parts, loaders = build_loaders(
        manifest, audio, feature_root, stats, config, device
    )

    source = resolve(config["backbone"]["local_snapshot"])
    model = M5Model.from_pretrained(
        str(source),
        local_files_only=True,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.1,
        final_dropout=0.1,
        mask_time_prob=0.05,
        layerdrop=0.1,
        num_labels=4,
        classifier_proj_size=256,
    )
    model.pitch_encoder.reset_input_adapter()
    if config["backbone"]["freeze_feature_encoder"]:
        model.freeze_feature_extractor()

    metadata = {
        "model": MODEL_NAME,
        "architecture_version": "m4_cnn_plus_lightweight_residual_dilated_v1",
        "comparison": "M4_to_M5_architecture_gain",
        "initialization": "wav2vec2_pretrained_independent_of_m4_checkpoint",
        "config_sha256": sha256_file(config_path),
        "split_sha256": sha256_file(resolve(SPLIT)),
        "feature_index_sha256": sha256_file(feature_root / "index.csv"),
    }

    if args.preflight_only:
        model.to(device).train()
        audio_batch, prosody_batch, labels = next(iter(loaders["train"]))
        logits = model(
            audio_batch[:1].to(device), prosody_batch[:1].to(device)
        ).logits
        loss = nn.CrossEntropyLoss()(logits, labels[:1].to(device))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not torch.isfinite(logits).all() or not gradients:
            raise RuntimeError("M5 preflight produced invalid logits or gradients")
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("M5 preflight produced a non-finite gradient")
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip"])
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("M5 preflight produced a non-finite gradient norm")
        print(json.dumps({
            "status": "PASS",
            "model": MODEL_NAME,
            "input_shape": list(prosody_batch[:1].shape),
            "logits_shape": list(logits.shape),
            "loss": float(loss.detach().cpu()),
            "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            "gradient_clip": float(config["training"]["gradient_clip"]),
            "parameters": sum(value.numel() for value in model.parameters()),
            "trainable_parameters": sum(
                value.numel() for value in model.parameters() if value.requires_grad
            ),
            "m4_checkpoint_used": False,
            "feature_cache_reused": True,
        }, indent=2))
        return

    artifact_root = resolve(ARTIFACT_ROOT)
    history, best_epoch, best_ua, runtime = train(
        model,
        loaders["train"],
        loaders["validation"],
        device,
        config,
        artifact_root,
        metadata,
        force_training=args.force_training,
        run_name=MODEL_NAME,
    )
    best = torch.load(
        artifact_root / "best_validation_ua.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(best["model_state"])
    model.to(device)
    validation, *_ = evaluate(
        model, loaders["validation"], device, config["emotion_order"]
    )
    test, truth, predicted, probabilities, test_loss = evaluate(
        model, loaders["test"], device, config["emotion_order"]
    )
    report_root = resolve(REPORT_ROOT)
    atomic_csv(
        prediction_frame(
            parts["test"], truth, predicted, probabilities, config["emotion_order"]
        ),
        report_root / "m5_residual_dilated_test_predictions.csv",
    )
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "best_epoch": best_epoch,
            "metadata": metadata,
        },
        artifact_root / "m5_residual_dilated.pt",
    )
    atomic_json(
        {
            "model": MODEL_NAME,
            "protocol": config["split_protocol"],
            "split_counts": {name: len(frame) for name, frame in parts.items()},
            "best_epoch": best_epoch,
            "best_validation_ua": best_ua,
            "validation": validation,
            "test": {**test, "loss": test_loss},
            "history": history,
            "runtime_seconds": runtime,
            "lineage": metadata,
            "comparison_baseline": M4_ARTIFACT_ROOT,
        },
        report_root / "m5_residual_dilated_report.json",
    )
    print(json.dumps({
        "status": "PASS",
        "best_epoch": best_epoch,
        "test_UA": test["ua"],
        "test_WA": test["wa"],
        "test_macro_f1": test["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
