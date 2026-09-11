"""Run the B4.8 three-sample Wav2Vec2 resume and recovery pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

try:
    from .extract_feature_pilot import decode_wav, load_embedded_audio
    from .extract_wav2vec2_pilot import (
        FEATURE_FAMILY,
        build_feature_attention_mask,
        file_sha256,
        masked_mean_pool,
        relative_to_repo,
        wav2vec_spec,
    )
    from .validate_feature_spec import load_feature_spec
except ImportError:
    from extract_feature_pilot import decode_wav, load_embedded_audio  # type: ignore
    from extract_wav2vec2_pilot import (  # type: ignore
        FEATURE_FAMILY,
        build_feature_attention_mask,
        file_sha256,
        masked_mean_pool,
        relative_to_repo,
        wav2vec_spec,
    )
    from validate_feature_spec import load_feature_spec  # type: ignore


FEATURE_VERSION = "b4_features_v1"
DEFAULT_SAMPLE_IDS = (
    "visec_hf_000111",  # Central / Angry
    "visec_hf_000017",  # North / Angry
    "visec_hf_000009",  # South / Angry
)
INDEX_FIELDS = (
    "sample_id",
    "source_row_index",
    "speaker_id",
    "emotion",
    "accent",
    "audio_sha256",
    "feature_family",
    "feature_version",
    "model_revision",
    "artifact_path",
    "artifact_sha256",
    "shape",
    "dtype",
    "finite_fraction",
    "status",
    "error",
)
LINEAGE_FIELDS = (
    "sample_id",
    "source_row_index",
    "speaker_id",
    "emotion",
    "accent",
    "audio_sha256",
    "feature_family",
    "feature_version",
    "model_revision",
    "artifact_path",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def select_rows(
    pilot_manifest: list[dict[str, str]], sample_ids: tuple[str, ...]
) -> list[dict[str, str]]:
    by_id = {row["sample_id"]: row for row in pilot_manifest}
    if len(by_id) != len(pilot_manifest):
        raise ValueError("Pilot manifest contains duplicate sample IDs")
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Requested samples are absent from pilot manifest: {missing}")
    selected = [by_id[sample_id] for sample_id in sample_ids]
    if len({row["speaker_id"] for row in selected}) != len(selected):
        raise ValueError("B4.8 pilot samples must use distinct speakers")
    if {row["accent"] for row in selected} != {"central", "north", "south"}:
        raise ValueError("B4.8 pilot must contain one sample from every accent")
    return selected


def expected_record(
    row: dict[str, str], artifact_path: Path, repo_root: Path, model_revision: str
) -> dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "source_row_index": row["source_row_index"],
        "speaker_id": row["speaker_id"],
        "emotion": row["emotion"],
        "accent": row["accent"],
        "audio_sha256": row["audio_sha256"],
        "feature_family": FEATURE_FAMILY,
        "feature_version": FEATURE_VERSION,
        "model_revision": model_revision,
        "artifact_path": relative_to_repo(artifact_path, repo_root),
        "artifact_sha256": "",
        "shape": "768",
        "dtype": "float32",
        "finite_fraction": "",
        "status": "pending",
        "error": "",
    }


def inspect_resume_candidate(
    existing: dict[str, str] | None,
    expected: dict[str, str],
    repo_root: Path,
) -> tuple[bool, str]:
    if existing is None:
        return False, "missing_index_record"
    for field in LINEAGE_FIELDS:
        if existing.get(field) != expected[field]:
            return False, f"lineage_mismatch:{field}"
    if existing.get("status") != "ok" or existing.get("error"):
        return False, "index_status_not_clean"
    artifact_path = (repo_root / existing["artifact_path"]).resolve()
    try:
        artifact_path.relative_to(repo_root.resolve())
    except ValueError:
        return False, "artifact_outside_repository"
    if not artifact_path.is_file():
        return False, "artifact_missing"
    if file_sha256(artifact_path) != existing.get("artifact_sha256"):
        return False, "artifact_sha256_mismatch"
    try:
        value = np.load(artifact_path, allow_pickle=False)
    except Exception:
        return False, "artifact_unreadable"
    if value.shape != (768,):
        return False, "artifact_shape_mismatch"
    if value.dtype != np.float32:
        return False, "artifact_dtype_mismatch"
    if not np.isfinite(value).all():
        return False, "artifact_non_finite"
    return True, "verified_artifact_reused"


def atomic_save_embedding(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def load_runtime(
    family: dict[str, Any], cache_dir: Path, threads: int
) -> tuple[Wav2Vec2FeatureExtractor, Wav2Vec2Model, float]:
    parameters = family["parameters"]
    started = time.perf_counter()
    snapshot_path = Path(
        snapshot_download(
            repo_id=parameters["model_id"],
            revision=parameters["model_revision"],
            cache_dir=cache_dir,
            local_files_only=True,
            allow_patterns=(
                "config.json",
                "preprocessor_config.json",
                "pytorch_model.bin",
                "model.safetensors",
            ),
        )
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        snapshot_path, local_files_only=True
    )
    model = Wav2Vec2Model.from_pretrained(snapshot_path, local_files_only=True)
    torch.set_num_threads(threads)
    model.eval()
    if feature_extractor.sampling_rate != parameters["sample_rate_hz"]:
        raise ValueError("Checkpoint feature extractor has an unexpected sample rate")
    if not feature_extractor.do_normalize:
        raise ValueError("Checkpoint feature extractor must normalize the waveform")
    if model.config.hidden_size != parameters["hidden_dimension"]:
        raise ValueError("Checkpoint hidden size differs from the feature contract")
    return feature_extractor, model, time.perf_counter() - started


def extract_embedding(
    signal: np.ndarray,
    sample_rate: int,
    feature_extractor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
) -> tuple[np.ndarray, int]:
    inputs = feature_extractor(
        signal,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    with torch.inference_mode():
        hidden = model(
            input_values=inputs["input_values"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state
        frame_mask = build_feature_attention_mask(
            inputs["attention_mask"],
            hidden.shape[1],
            model.config.conv_kernel,
            model.config.conv_stride,
        )
        embedding = masked_mean_pool(hidden, frame_mask)[0]
    value = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
    if value.shape != (768,) or not np.isfinite(value).all():
        raise ValueError("Wav2Vec2 produced an invalid 768-value embedding")
    return value, int(frame_mask.sum().item())


def run_resume_pilot(
    repo_root: Path,
    spec_path: Path,
    pilot_manifest_path: Path,
    parquet_path: Path,
    cache_dir: Path,
    run_root: Path,
    report_path: Path,
    evidence_index_path: Path,
    sample_ids: tuple[str, ...],
    threads: int,
    selected_rows: list[dict[str, str]] | None = None,
    scope: str = "Three reviewed utterances, one per accent, all Angry; CPU and local checkpoint cache only.",
    selection_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = wav2vec_spec(load_feature_spec(spec_path))
    parameters = family["parameters"]
    selected = (
        select_rows(read_csv(pilot_manifest_path), sample_ids)
        if selected_rows is None
        else selected_rows
    )
    if tuple(row["sample_id"] for row in selected) != sample_ids:
        raise ValueError("Selected rows must match sample_ids in deterministic order")
    index_path = run_root / "feature_index.csv"
    existing_rows = {row["sample_id"]: row for row in read_csv(index_path)}

    records: dict[str, dict[str, str]] = {}
    pending: list[tuple[dict[str, str], dict[str, str], str]] = []
    resume_reasons: dict[str, str] = {}
    for row in selected:
        artifact_path = run_root / FEATURE_FAMILY / f"{row['sample_id']}.npy"
        expected = expected_record(
            row, artifact_path, repo_root, parameters["model_revision"]
        )
        reusable, reason = inspect_resume_candidate(
            existing_rows.get(row["sample_id"]), expected, repo_root
        )
        resume_reasons[row["sample_id"]] = reason
        if reusable:
            records[row["sample_id"]] = existing_rows[row["sample_id"]]
        else:
            records[row["sample_id"]] = expected
            pending.append((row, expected, reason))

    model_loaded = False
    runtime_load_seconds = 0.0
    processed = 0
    failures = 0
    if pending:
        indexes = [int(row["source_row_index"]) for row, _, _ in pending]
        audio_objects = load_embedded_audio(parquet_path, indexes)
        feature_extractor, model, runtime_load_seconds = load_runtime(
            family, cache_dir, threads
        )
        model_loaded = True
        for row, record, _ in pending:
            started = time.perf_counter()
            try:
                source_index = int(row["source_row_index"])
                audio_bytes = audio_objects[source_index]["bytes"]
                audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
                if audio_sha256 != row["audio_sha256"]:
                    raise ValueError("Audio SHA-256 differs from the reviewed manifest")
                signal, sample_rate, info = decode_wav(audio_bytes)
                if sample_rate != parameters["sample_rate_hz"] or info["channels"] != 1:
                    raise ValueError("Audio violates the 16 kHz mono contract")
                value, valid_frames = extract_embedding(
                    signal, sample_rate, feature_extractor, model
                )
                artifact_path = repo_root / record["artifact_path"]
                atomic_save_embedding(artifact_path, value)
                reopened = np.load(artifact_path, allow_pickle=False)
                if not np.array_equal(value, reopened):
                    raise ValueError("Saved embedding differs after reopening")
                record.update(
                    {
                        "artifact_sha256": file_sha256(artifact_path),
                        "shape": str(value.shape[0]),
                        "dtype": str(value.dtype),
                        "finite_fraction": f"{float(np.isfinite(value).mean()):.8f}",
                        "status": "ok",
                        "error": "",
                    }
                )
                resume_reasons[row["sample_id"]] += f";recomputed_valid_frames={valid_frames}"
                processed += 1
            except Exception as exc:
                record.update(
                    {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                failures += 1
            records[row["sample_id"]] = record
            atomic_write_csv([records[sample_id] for sample_id in sample_ids], index_path)
            resume_reasons[row["sample_id"]] += (
                f";elapsed_seconds={time.perf_counter() - started:.6f}"
            )

    ordered_records = [records[sample_id] for sample_id in sample_ids]
    atomic_write_csv(ordered_records, index_path)
    evidence_index_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(index_path, evidence_index_path)
    report = {
        "status": "PASS" if failures == 0 else "FAIL",
        "scope": scope,
        "feature_version": FEATURE_VERSION,
        "model_id": parameters["model_id"],
        "model_revision": parameters["model_revision"],
        "selected_samples": list(sample_ids),
        "selected_speakers": len({row["speaker_id"] for row in selected}),
        "selected_accents": sorted({row["accent"] for row in selected}),
        "processed": processed,
        "resumed": len(selected) - len(pending),
        "invalidated": sum(
            reason != "missing_index_record" and reason != "verified_artifact_reused"
            for _, _, reason in pending
        ),
        "failures": failures,
        "model_loaded": model_loaded,
        "runtime_load_seconds": runtime_load_seconds,
        "resume_reasons": resume_reasons,
        "index_path": relative_to_repo(index_path, repo_root),
        "evidence_index_path": relative_to_repo(evidence_index_path, repo_root),
        "artifact_hashes": {
            row["sample_id"]: row["artifact_sha256"] for row in ordered_records
        },
    }
    if selection_metadata is not None:
        report["selection"] = selection_metadata
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"B4.8 pilot completed with {failures} failures")
    return report


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--cache-dir", type=Path, default=repo_root / "data/interim/models/huggingface"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=repo_root / "data/interim/features/b4_features_v1/wav2vec2_resume_3",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_wav2vec2_resume3_run.json",
    )
    parser.add_argument(
        "--evidence-index",
        type=Path,
        default=repo_root / "reports/tables/b4/b4_wav2vec2_resume3_feature_index.csv",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    report = run_resume_pilot(
        repo_root=repo_root,
        spec_path=args.spec.resolve(),
        pilot_manifest_path=args.pilot_manifest.resolve(),
        parquet_path=args.parquet.resolve(),
        cache_dir=args.cache_dir.resolve(),
        run_root=args.run_root.resolve(),
        report_path=args.report_path.resolve(),
        evidence_index_path=args.evidence_index.resolve(),
        sample_ids=DEFAULT_SAMPLE_IDS,
        threads=args.threads,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
