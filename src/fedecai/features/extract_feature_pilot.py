"""Extract a small, leakage-safe B4 feature pilot from embedded ViSEC audio.

The pilot selects at most one reviewed utterance per accent-emotion cell. It
does not pad, crop, normalize, augment, fit statistics, or modify raw data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import wave
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import opensmile
import pyarrow.parquet as pq


ACCENTS = ("central", "north", "south")
EMOTIONS = ("angry", "happy", "neutral", "sad")
PILOT_FAMILIES = (
    "mfcc40_summary",
    "egemaps_v02",
    "log_mel_80",
    "pitch_contour",
)
FEATURE_VERSION = "b4_features_v1"


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def select_pilot_rows(
    rows: list[dict[str, str]], max_samples: int = 12
) -> list[dict[str, str]]:
    """Choose deterministic coverage samples, preferring clean and new speakers."""
    if not 1 <= max_samples <= len(ACCENTS) * len(EMOTIONS):
        raise ValueError("max_samples must be between 1 and 12")
    eligible = [row for row in rows if _as_bool(row["is_eligible_for_split"])]
    selected: list[dict[str, str]] = []
    used_speakers: set[str] = set()
    for accent in ACCENTS:
        for emotion in EMOTIONS:
            candidates = [
                row
                for row in eligible
                if row["accent_clean"] == accent and row["emotion"] == emotion
            ]
            if not candidates:
                raise ValueError(f"No eligible pilot sample for {accent}/{emotion}")
            candidates.sort(
                key=lambda row: (
                    _as_bool(row["possible_clipping"]) or _as_bool(row["high_silence_ratio"]),
                    row["speaker_id"] in used_speakers,
                    int(row["source_row_index"]),
                )
            )
            chosen = candidates[0]
            selected.append(chosen)
            used_speakers.add(chosen["speaker_id"])
            if len(selected) == max_samples:
                return selected
    return selected


def load_embedded_audio(
    parquet_path: Path, source_row_indexes: list[int]
) -> dict[int, dict[str, Any]]:
    """Read only Parquet row groups containing selected pilot samples."""
    targets = set(source_row_indexes)
    found: dict[int, dict[str, Any]] = {}
    parquet_file = pq.ParquetFile(parquet_path)
    row_offset = 0
    for row_group_index in range(parquet_file.num_row_groups):
        row_count = parquet_file.metadata.row_group(row_group_index).num_rows
        local_targets = sorted(
            index - row_offset
            for index in targets
            if row_offset <= index < row_offset + row_count
        )
        if local_targets:
            table = parquet_file.read_row_group(row_group_index, columns=["path"])
            audio_column = table.column("path")
            for local_index in local_targets:
                value = audio_column[local_index].as_py()
                if not value or not value.get("bytes"):
                    raise ValueError(f"Missing embedded audio at source row {row_offset + local_index}")
                found[row_offset + local_index] = value
        row_offset += row_count
    missing = sorted(targets - set(found))
    if missing:
        raise ValueError(f"Pilot source rows not found in Parquet: {missing}")
    return found


def _decode_pcm(frames: bytes, sample_width: int) -> tuple[np.ndarray, float]:
    if sample_width == 1:
        return np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0, 128.0
    if sample_width == 2:
        return np.frombuffer(frames, dtype="<i2").astype(np.float32), 32768.0
    if sample_width == 3:
        values = np.frombuffer(frames, dtype=np.uint8)
        values = values[: (values.size // 3) * 3].reshape(-1, 3)
        decoded = (
            values[:, 0].astype(np.int32)
            | (values[:, 1].astype(np.int32) << 8)
            | (values[:, 2].astype(np.int32) << 16)
        )
        decoded = (decoded ^ 0x800000) - 0x800000
        return decoded.astype(np.float32), 8388608.0
    if sample_width == 4:
        return np.frombuffer(frames, dtype="<i4").astype(np.float32), 2147483648.0
    raise ValueError(f"Unsupported PCM width: {sample_width} bytes")


def decode_wav(audio_bytes: bytes) -> tuple[np.ndarray, int, dict[str, int]]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)
    samples, full_scale = _decode_pcm(frames, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    signal = np.ascontiguousarray(samples / full_scale, dtype=np.float32)
    return signal, sample_rate, {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
    }


def extract_mfcc(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    matrix = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40,
        n_fft=400,
        win_length=400,
        hop_length=160,
        window="hann",
        center=True,
    )
    return np.concatenate((matrix.mean(axis=1), matrix.std(axis=1))).astype(np.float32)


def extract_log_mel(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    power = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=80,
        n_fft=400,
        win_length=400,
        hop_length=160,
        window="hann",
        fmin=20,
        fmax=8000,
        power=2.0,
        center=True,
    )
    return librosa.power_to_db(power, ref=np.max, top_db=80).astype(np.float32)


def summarize_pitch(f0: np.ndarray) -> np.ndarray:
    """Return six deterministic pitch summaries, including all-unvoiced audio."""
    voiced_f0 = f0[np.isfinite(f0)]
    if voiced_f0.size == 0:
        return np.zeros(6, dtype=np.float32)
    return np.asarray(
        [
            voiced_f0.size / f0.size,
            voiced_f0.mean(),
            voiced_f0.std(),
            np.median(voiced_f0),
            np.quantile(voiced_f0, 0.10),
            np.quantile(voiced_f0, 0.90),
        ],
        dtype=np.float32,
    )


def extract_pitch(signal: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    f0, voiced_flag, voiced_probability = librosa.pyin(
        signal,
        fmin=50,
        fmax=500,
        sr=sample_rate,
        frame_length=1024,
        hop_length=160,
        center=True,
        fill_na=np.nan,
    )
    summary = summarize_pitch(f0)
    return {
        "f0_hz": f0.astype(np.float32),
        "voiced_flag": voiced_flag.astype(bool),
        "voiced_probability": voiced_probability.astype(np.float32),
        "summary": summary,
    }


def build_smile_extractor() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_egemaps(
    signal: np.ndarray, sample_rate: int, smile: opensmile.Smile
) -> tuple[np.ndarray, list[str]]:
    frame = smile.process_signal(signal, sample_rate)
    values = frame.iloc[0].to_numpy(dtype=np.float32)
    return values, [str(column) for column in frame.columns]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _save_feature(
    family: str,
    sample_id: str,
    value: Any,
    output_root: Path,
) -> tuple[Path, str, float]:
    family_dir = output_root / family
    family_dir.mkdir(parents=True, exist_ok=True)
    if family == "pitch_contour":
        path = family_dir / f"{sample_id}.npz"
        np.savez_compressed(path, **value)
        shape = f"frames={value['f0_hz'].shape[0]};summary=6"
        finite_fraction = float(np.isfinite(value["f0_hz"]).mean())
    else:
        path = family_dir / f"{sample_id}.npy"
        np.save(path, value, allow_pickle=False)
        shape = "x".join(str(size) for size in value.shape)
        finite_fraction = float(np.isfinite(value).mean())
    return path, shape, finite_fraction


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_feature_value(family: str, value: Any) -> None:
    if family == "mfcc40_summary":
        if value.shape != (80,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid MFCC output: shape={value.shape}")
    elif family == "egemaps_v02":
        if value.shape != (88,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid eGeMAPS output: shape={value.shape}")
    elif family == "log_mel_80":
        if value.ndim != 2 or value.shape[0] != 80 or value.shape[1] == 0:
            raise ValueError(f"Invalid log-Mel output: shape={value.shape}")
        if not np.isfinite(value).all():
            raise ValueError("Log-Mel output contains non-finite values")
    elif family == "pitch_contour":
        lengths = {
            value["f0_hz"].shape[0],
            value["voiced_flag"].shape[0],
            value["voiced_probability"].shape[0],
        }
        if len(lengths) != 1 or value["summary"].shape != (6,):
            raise ValueError("Pitch arrays have inconsistent shapes")
        if not np.isfinite(value["summary"]).all():
            raise ValueError("Pitch summary contains non-finite values")
    else:
        raise ValueError(f"Unsupported pilot feature family: {family}")


def run_pilot(
    manifest_path: Path,
    parquet_path: Path,
    output_root: Path,
    max_samples: int,
    families: tuple[str, ...],
    repo_root: Path,
) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    selected = select_pilot_rows(rows, max_samples=max_samples)
    indexes = [int(row["source_row_index"]) for row in selected]
    audio_objects = load_embedded_audio(parquet_path, indexes)
    smile = build_smile_extractor() if "egemaps_v02" in families else None

    pilot_manifest = []
    feature_index = []
    egemaps_names: list[str] | None = None
    failures = 0
    for row in selected:
        source_index = int(row["source_row_index"])
        audio_bytes = audio_objects[source_index]["bytes"]
        actual_hash = hashlib.sha256(audio_bytes).hexdigest()
        if actual_hash != row["audio_sha256"]:
            raise ValueError(f"Audio SHA-256 mismatch for {row['sample_id']}")
        signal, sample_rate, audio_info = decode_wav(audio_bytes)
        if sample_rate != 16_000 or audio_info["channels"] != 1:
            raise ValueError(
                f"Unexpected audio contract for {row['sample_id']}: "
                f"sample_rate={sample_rate}, channels={audio_info['channels']}"
            )
        pilot_manifest.append(
            {
                "sample_id": row["sample_id"],
                "source_row_index": source_index,
                "speaker_id": row["speaker_id"],
                "accent": row["accent_clean"],
                "emotion": row["emotion"],
                "duration_sec": row["duration_sec_measured"],
                "possible_clipping": row["possible_clipping"],
                "high_silence_ratio": row["high_silence_ratio"],
                "audio_sha256": actual_hash,
            }
        )

        for family in families:
            record = {
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "emotion": row["emotion"],
                "accent": row["accent_clean"],
                "feature_family": family,
                "feature_version": FEATURE_VERSION,
                "artifact_path": "",
                "artifact_sha256": "",
                "shape": "",
                "dtype": "float32",
                "finite_fraction": "",
                "status": "error",
                "error": "",
            }
            try:
                if family == "mfcc40_summary":
                    value = extract_mfcc(signal, sample_rate)
                elif family == "egemaps_v02":
                    if smile is None:
                        raise RuntimeError("openSMILE extractor was not initialized")
                    value, names = extract_egemaps(signal, sample_rate, smile)
                    if egemaps_names is None:
                        egemaps_names = names
                    elif egemaps_names != names:
                        raise ValueError("eGeMAPS column order changed between samples")
                elif family == "log_mel_80":
                    value = extract_log_mel(signal, sample_rate)
                elif family == "pitch_contour":
                    value = extract_pitch(signal, sample_rate)
                else:
                    raise ValueError(f"Unsupported family: {family}")
                validate_feature_value(family, value)
                artifact_path, shape, finite_fraction = _save_feature(
                    family, row["sample_id"], value, output_root
                )
                record.update(
                    {
                        "artifact_path": _relative_path(artifact_path, repo_root),
                        "artifact_sha256": file_sha256(artifact_path),
                        "shape": shape,
                        "finite_fraction": f"{finite_fraction:.8f}",
                        "status": "ok",
                    }
                )
            except Exception as exc:
                failures += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
            feature_index.append(record)

    write_csv(pilot_manifest, output_root / "pilot_manifest.csv")
    write_csv(feature_index, output_root / "feature_index.csv")
    if egemaps_names is not None:
        (output_root / "egemaps_feature_names.json").write_text(
            json.dumps(egemaps_names, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    family_results = {}
    for family in families:
        records = [row for row in feature_index if row["feature_family"] == family]
        family_results[family] = {
            "ok": sum(row["status"] == "ok" for row in records),
            "error": sum(row["status"] == "error" for row in records),
            "shapes": sorted({row["shape"] for row in records if row["shape"]}),
        }
    report = {
        "feature_version": FEATURE_VERSION,
        "scope": "Per-sample deterministic pilot only; no fitting, padding, cropping, normalization, or augmentation.",
        "selected_samples": len(selected),
        "selected_speakers": len({row["speaker_id"] for row in pilot_manifest}),
        "selected_cells": sorted(
            {f"{row['accent']}:{row['emotion']}" for row in pilot_manifest}
        ),
        "quality_flagged_selected": sum(
            _as_bool(row["possible_clipping"]) or _as_bool(row["high_silence_ratio"])
            for row in pilot_manifest
        ),
        "families": family_results,
        "feature_records": len(feature_index),
        "failures": failures,
        "output_root": _relative_path(output_root, repo_root),
    }
    (output_root / "pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError(
            f"Pilot completed with {failures} feature failures; inspect feature_index.csv"
        )
    return report


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data/manifests/clean_manifest.csv",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=repo_root / "data/raw/visec_hf/train-00000-of-00001.parquet",
    )
    parser.add_argument("--run-name", default="balanced_12")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=PILOT_FAMILIES,
        default=list(PILOT_FAMILIES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    output_root = repo_root / "data/interim/features" / FEATURE_VERSION / args.run_name
    report = run_pilot(
        args.manifest.resolve(),
        args.parquet.resolve(),
        output_root.resolve(),
        args.max_samples,
        tuple(args.families),
        repo_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
