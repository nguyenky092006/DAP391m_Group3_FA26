"""Extract one revision-pinned Wav2Vec2 embedding for the B4.7 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

try:
    from .extract_feature_pilot import decode_wav, load_embedded_audio
    from .validate_feature_spec import load_feature_spec
except ImportError:
    from extract_feature_pilot import decode_wav, load_embedded_audio  # type: ignore
    from validate_feature_spec import load_feature_spec  # type: ignore


FEATURE_FAMILY = "wav2vec2_embedding"
DEFAULT_SAMPLE_ID = "visec_hf_000111"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_sample(
    pilot_manifest: list[dict[str, str]], sample_id: str
) -> dict[str, str]:
    matches = [row for row in pilot_manifest if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one pilot row for {sample_id}; found {len(matches)}")
    return matches[0]


def feature_output_lengths(
    input_lengths: torch.Tensor,
    conv_kernels: Sequence[int],
    conv_strides: Sequence[int],
) -> torch.Tensor:
    if len(conv_kernels) != len(conv_strides) or not conv_kernels:
        raise ValueError("Convolution kernels and strides must have equal non-zero length")
    lengths = input_lengths.to(dtype=torch.long)
    for kernel, stride in zip(conv_kernels, conv_strides):
        lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
    if torch.any(lengths <= 0):
        raise ValueError("Input is too short for the Wav2Vec2 feature encoder")
    return lengths


def build_feature_attention_mask(
    input_attention_mask: torch.Tensor,
    feature_frames: int,
    conv_kernels: Sequence[int],
    conv_strides: Sequence[int],
) -> torch.Tensor:
    if input_attention_mask.ndim != 2:
        raise ValueError("Input attention mask must have shape batch x samples")
    input_lengths = input_attention_mask.to(dtype=torch.long).sum(dim=1)
    feature_lengths = feature_output_lengths(input_lengths, conv_kernels, conv_strides)
    positions = torch.arange(feature_frames, device=input_attention_mask.device)
    mask = positions.unsqueeze(0) < feature_lengths.unsqueeze(1)
    if mask.shape != (input_attention_mask.shape[0], feature_frames):
        raise ValueError("Feature attention mask has an unexpected shape")
    return mask


def masked_mean_pool(
    last_hidden_state: torch.Tensor, feature_attention_mask: torch.Tensor
) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ValueError("Hidden state must have shape batch x frames x features")
    if feature_attention_mask.shape != last_hidden_state.shape[:2]:
        raise ValueError("Feature mask must match hidden-state batch and frame axes")
    weights = feature_attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    denominators = weights.sum(dim=1)
    if torch.any(denominators == 0):
        raise ValueError("Every sample must contain at least one valid feature frame")
    return (last_hidden_state * weights).sum(dim=1) / denominators


def wav2vec_spec(spec: dict[str, Any]) -> dict[str, Any]:
    matches = [
        family
        for family in spec["feature_families"]
        if family["name"] == FEATURE_FAMILY
    ]
    if len(matches) != 1:
        raise ValueError("Feature specification must contain one Wav2Vec2 family")
    family = matches[0]
    if family["status"] not in {"specified_not_downloaded", "pilot_validated"}:
        raise ValueError("Wav2Vec2 feature contract is not ready for a one-sample pilot")
    return family


def relative_to_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def run_one_sample_pilot(
    repo_root: Path,
    spec_path: Path,
    pilot_manifest_path: Path,
    parquet_path: Path,
    cache_dir: Path,
    output_root: Path,
    report_path: Path,
    sample_id: str,
    local_files_only: bool,
    threads: int,
) -> dict[str, Any]:
    family = wav2vec_spec(load_feature_spec(spec_path))
    parameters = family["parameters"]
    model_id = parameters["model_id"]
    model_revision = parameters["model_revision"]
    row = select_sample(read_csv(pilot_manifest_path), sample_id)

    audio_object = load_embedded_audio(parquet_path, [int(row["source_row_index"])])[
        int(row["source_row_index"])
    ]
    audio_bytes = audio_object["bytes"]
    actual_audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if actual_audio_hash != row["audio_sha256"]:
        raise ValueError("Pilot audio hash does not match the reviewed manifest")
    signal, sample_rate, audio_info = decode_wav(audio_bytes)
    if sample_rate != parameters["sample_rate_hz"] or audio_info["channels"] != 1:
        raise ValueError("Pilot audio violates the Wav2Vec2 audio contract")

    cache_dir.mkdir(parents=True, exist_ok=True)
    download_started = time.perf_counter()
    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=model_revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=(
                "config.json",
                "preprocessor_config.json",
                "pytorch_model.bin",
                "model.safetensors",
            ),
        )
    )
    download_seconds = time.perf_counter() - download_started

    load_started = time.perf_counter()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        snapshot_path, local_files_only=True
    )
    model = Wav2Vec2Model.from_pretrained(snapshot_path, local_files_only=True)
    model.eval()
    torch.set_num_threads(threads)
    load_seconds = time.perf_counter() - load_started

    if feature_extractor.sampling_rate != sample_rate:
        raise ValueError("Checkpoint feature extractor has an unexpected sample rate")
    if not feature_extractor.do_normalize:
        raise ValueError("Checkpoint feature extractor must normalize the waveform")
    if model.config.hidden_size != parameters["hidden_dimension"]:
        raise ValueError("Checkpoint hidden size differs from the feature contract")

    inputs = feature_extractor(
        signal,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    if "attention_mask" not in inputs:
        raise ValueError("Processor did not return the required attention mask")

    def infer() -> tuple[np.ndarray, int, int, float]:
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(
                input_values=inputs["input_values"],
                attention_mask=inputs["attention_mask"],
            )
            hidden = output.last_hidden_state
            frame_mask = build_feature_attention_mask(
                inputs["attention_mask"],
                hidden.shape[1],
                model.config.conv_kernel,
                model.config.conv_stride,
            )
            embedding = masked_mean_pool(hidden, frame_mask)
        elapsed = time.perf_counter() - started
        value = embedding[0].detach().cpu().numpy().astype(np.float32, copy=False)
        return value, int(hidden.shape[1]), int(frame_mask.sum().item()), elapsed

    first, total_frames, valid_frames, first_seconds = infer()
    second, second_total, second_valid, second_seconds = infer()
    if first.shape != (parameters["output_dimension"],):
        raise ValueError(f"Unexpected Wav2Vec2 embedding shape: {first.shape}")
    if not np.isfinite(first).all():
        raise ValueError("Wav2Vec2 embedding contains non-finite values")
    if total_frames != second_total or valid_frames != second_valid:
        raise ValueError("Wav2Vec2 frame counts changed between deterministic runs")
    maximum_repeat_difference = float(np.max(np.abs(first - second)))
    if maximum_repeat_difference != 0.0:
        raise ValueError(
            f"Repeated CPU inference changed output: {maximum_repeat_difference}"
        )

    artifact_dir = output_root / FEATURE_FAMILY
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{sample_id}.npy"
    np.save(artifact_path, first, allow_pickle=False)
    reopened = np.load(artifact_path, allow_pickle=False)
    if not np.array_equal(first, reopened):
        raise ValueError("Reopened Wav2Vec2 artifact differs from the saved embedding")

    report: dict[str, Any] = {
        "status": "PASS",
        "scope": "One reviewed utterance; CPU inference only; no training or fitted preprocessing.",
        "sample": {
            "sample_id": sample_id,
            "source_row_index": int(row["source_row_index"]),
            "speaker_id": row["speaker_id"],
            "accent": row["accent"],
            "emotion": row["emotion"],
            "duration_sec": float(row["duration_sec"]),
            "audio_sha256": actual_audio_hash,
        },
        "model": {
            "model_id": model_id,
            "revision": model_revision,
            "hidden_dimension": int(model.config.hidden_size),
            "transformer_layers": int(model.config.num_hidden_layers),
            "pooling": parameters["pooling"],
            "device": "cpu",
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
        },
        "input": {
            "sample_rate_hz": sample_rate,
            "samples": int(signal.shape[0]),
            "attention_mask_samples": int(inputs["attention_mask"].sum().item()),
            "waveform_normalized_by_processor": bool(feature_extractor.do_normalize),
        },
        "output": {
            "shape": list(first.shape),
            "dtype": str(first.dtype),
            "finite_fraction": float(np.isfinite(first).mean()),
            "total_feature_frames": total_frames,
            "valid_feature_frames": valid_frames,
            "maximum_repeat_difference": maximum_repeat_difference,
            "artifact_path": relative_to_repo(artifact_path, repo_root),
            "artifact_sha256": file_sha256(artifact_path),
        },
        "timing_seconds": {
            "snapshot_resolution_or_download": download_seconds,
            "model_load": load_seconds,
            "first_inference": first_seconds,
            "second_inference": second_seconds,
        },
        "cache": {
            "path": relative_to_repo(cache_dir, repo_root),
            "local_files_only": local_files_only,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--spec", type=Path, default=repo_root / "configs/features/visec_features_v1.json"
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=repo_root / "reports/tables/b4/balanced_12_pilot_manifest.csv",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=repo_root / "data/raw/visec_hf/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "data/interim/models/huggingface",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "data/interim/features/b4_features_v1/wav2vec2_one_sample",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_wav2vec2_one_sample_report.json",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    report = run_one_sample_pilot(
        repo_root=repo_root,
        spec_path=args.spec.resolve(),
        pilot_manifest_path=args.pilot_manifest.resolve(),
        parquet_path=args.parquet.resolve(),
        cache_dir=args.cache_dir.resolve(),
        output_root=args.output_root.resolve(),
        report_path=args.report_path.resolve(),
        sample_id=args.sample_id,
        local_files_only=args.local_files_only,
        threads=args.threads,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
