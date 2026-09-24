"""Extract M3/M4 prosody artifacts and fit normalization on train only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import wave
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pitchfusion.features.multi_prosody import (
    CHANNELS,
    extract_multi_prosody,
    load_prosody_artifact,
    save_prosody_artifact,
)


ROOT = Path(__file__).resolve().parents[3]


def decode_wav(value: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(value), "rb") as stream:
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    if channels != 1:
        raise ValueError(f"Expected mono audio, received {channels} channels")
    if width == 2:
        signal = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        source = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        packed = (
            source[:, 0].astype(np.int32)
            | source[:, 1].astype(np.int32) << 8
            | source[:, 2].astype(np.int32) << 16
        )
        packed = np.where(packed & 0x800000, packed - 0x1000000, packed)
        signal = packed.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {width}")
    return np.ascontiguousarray(signal), sample_rate


def write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty feature index")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fit_train_statistics(
    manifest: pd.DataFrame, artifact_by_sample: dict[str, Path]
) -> dict[str, Any]:
    train_ids = manifest.loc[manifest["split"] == "train", "sample_id"].astype(str)
    sums = np.zeros(len(CHANNELS), dtype=np.float64)
    squared = np.zeros(len(CHANNELS), dtype=np.float64)
    counts = np.zeros(len(CHANNELS), dtype=np.int64)
    for sample_id in train_ids:
        values = load_prosody_artifact(artifact_by_sample[sample_id]).astype(np.float64)
        sums += values.sum(axis=1)
        squared += np.square(values).sum(axis=1)
        counts += values.shape[1]
    if np.any(counts == 0):
        raise ValueError("At least one prosody channel has no training frames")
    means = sums / counts
    variances = np.maximum(squared / counts - np.square(means), 1e-12)
    return {
        "channels": list(CHANNELS),
        "mean": means.tolist(),
        "std": np.sqrt(variances).tolist(),
        "frame_counts": counts.tolist(),
        "fit_split": "train",
        "validation_and_test_used_for_fit": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifests/split_manifest_v2_enriched.csv",
    )
    parser.add_argument(
        "--raw-parquet",
        type=Path,
        default=ROOT / "data/raw/visec_hf/train-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/interim/features/multi_prosody_v1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"sample_id": "string"})
    manifest = manifest[manifest["split"] != "excluded"].copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    if manifest["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    audio_objects = pq.read_table(args.raw_parquet, columns=["path"])["path"].to_pylist()
    artifact_dir = args.output_dir / "artifacts"
    rows: list[dict[str, Any]] = []
    artifact_by_sample: dict[str, Path] = {}
    for number, source in enumerate(manifest.itertuples(index=False), start=1):
        sample_id = str(source.sample_id)
        source_index = int(source.source_row_index)
        if source_index >= len(audio_objects):
            raise IndexError(f"source_row_index out of range: {source_index}")
        audio_bytes = audio_objects[source_index]["bytes"]
        audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
        if audio_sha256 != str(source.audio_sha256):
            raise ValueError(f"Audio SHA-256 mismatch for {sample_id}")
        artifact_path = artifact_dir / f"{sample_id}.npz"
        if args.force or not artifact_path.is_file():
            signal, sample_rate = decode_wav(audio_bytes)
            values = extract_multi_prosody(signal, sample_rate)
            artifact_sha256 = save_prosody_artifact(artifact_path, values)
        else:
            values = load_prosody_artifact(artifact_path)
            artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        artifact_by_sample[sample_id] = artifact_path
        try:
            relative_artifact = artifact_path.relative_to(ROOT).as_posix()
        except ValueError:
            relative_artifact = str(artifact_path.resolve())
        rows.append(
            {
                "sample_id": sample_id,
                "split": source.split,
                "source_row_index": source_index,
                "frames": values.shape[1],
                "channels": values.shape[0],
                "artifact_path": relative_artifact,
                "artifact_sha256": artifact_sha256,
                "audio_sha256": audio_sha256,
            }
        )
        if number % 100 == 0 or number == len(manifest):
            print(f"prosody {number}/{len(manifest)}", flush=True)

    statistics = fit_train_statistics(manifest, artifact_by_sample)
    write_csv_atomic(rows, args.output_dir / "feature_index.csv")
    statistics_path = args.output_dir / "normalization_train.json"
    statistics_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "samples": len(rows),
                "channels": list(CHANNELS),
                "normalization_fit_split": "train",
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
