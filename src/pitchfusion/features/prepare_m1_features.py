"""Build the 174-feature table used by the fixed M1 SVM-RBF baseline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NUMBA_CACHE_DIR = ROOT / "data/interim/numba_cache"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import librosa
import numpy as np
import opensmile
import pandas as pd
import pyarrow.parquet as pq


MFCC_COLUMNS = [
    *(f"mfcc_mean_{index:02d}" for index in range(1, 41)),
    *(f"mfcc_std_{index:02d}" for index in range(1, 41)),
]
EGEMAPS_COLUMNS = [f"egemaps_{index:03d}" for index in range(88)]
PITCH_COLUMNS = [
    "pitch_voiced_fraction",
    "pitch_f0_mean_hz",
    "pitch_f0_std_hz",
    "pitch_f0_median_hz",
    "pitch_f0_q10_hz",
    "pitch_f0_q90_hz",
]
FEATURE_COLUMNS = MFCC_COLUMNS + EGEMAPS_COLUMNS + PITCH_COLUMNS


def decode_wav(value: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(value), "rb") as stream:
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    if channels != 1 or sample_rate != 16_000:
        raise ValueError(
            f"M1 requires mono 16 kHz audio, received {channels} channel(s) at {sample_rate} Hz"
        )
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
    return np.concatenate((matrix.mean(axis=1), matrix.std(axis=1))).astype(
        np.float32
    )


def extract_pitch_summary(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    f0_hz, _, _ = librosa.pyin(
        signal,
        fmin=50.0,
        fmax=500.0,
        sr=sample_rate,
        frame_length=1024,
        hop_length=160,
        center=True,
        fill_na=np.nan,
    )
    voiced = np.asarray(f0_hz, dtype=np.float32)
    voiced = voiced[np.isfinite(voiced) & (voiced > 0)]
    if voiced.size == 0:
        return np.zeros(6, dtype=np.float32)
    return np.asarray(
        [
            voiced.size / len(f0_hz),
            voiced.mean(),
            voiced.std(),
            np.median(voiced),
            np.quantile(voiced, 0.10),
            np.quantile(voiced, 0.90),
        ],
        dtype=np.float32,
    )


def build_smile() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def validate_table(frame: pd.DataFrame, manifest: pd.DataFrame) -> None:
    if len(frame) != len(manifest):
        raise ValueError(f"Expected {len(manifest)} rows, found {len(frame)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("M1 feature table contains duplicate sample_id values")
    if set(frame["sample_id"].astype(str)) != set(manifest["sample_id"].astype(str)):
        raise ValueError("M1 feature IDs do not match the eligible manifest")
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"M1 feature table is missing columns: {sorted(missing)[:3]}")
    matrix = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if matrix.shape[1] != 174 or not np.isfinite(matrix).all():
        raise ValueError("M1 feature matrix must contain 174 finite values per sample")


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
        "--output",
        type=Path,
        default=ROOT / "data/interim/features/m1_features_v1.parquet",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"sample_id": "string"})
    manifest = manifest[manifest["split"] != "excluded"].copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    if args.output.is_file() and not args.force:
        existing = pd.read_parquet(args.output)
        validate_table(existing, manifest)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "resume_existing",
                    "rows": len(existing),
                    "features": len(FEATURE_COLUMNS),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return

    audio_objects = pq.read_table(args.raw_parquet, columns=["path"])["path"].to_pylist()
    smile = build_smile()
    records: list[dict[str, object]] = []
    for number, source in enumerate(manifest.itertuples(index=False), start=1):
        source_index = int(source.source_row_index)
        audio_bytes = audio_objects[source_index]["bytes"]
        if hashlib.sha256(audio_bytes).hexdigest() != str(source.audio_sha256):
            raise ValueError(f"Audio SHA-256 mismatch for {source.sample_id}")
        signal, sample_rate = decode_wav(audio_bytes)
        mfcc = extract_mfcc(signal, sample_rate)
        egemaps = smile.process_signal(signal, sample_rate).iloc[0].to_numpy(
            dtype=np.float32
        )
        pitch = extract_pitch_summary(signal, sample_rate)
        values = np.concatenate((mfcc, egemaps, pitch)).astype(np.float32)
        if values.shape != (174,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid M1 features for {source.sample_id}")
        record: dict[str, object] = {"sample_id": str(source.sample_id)}
        record.update(zip(FEATURE_COLUMNS, values.tolist()))
        records.append(record)
        if number % 100 == 0 or number == len(manifest):
            print(f"M1 features {number}/{len(manifest)}", flush=True)

    frame = pd.DataFrame.from_records(records)
    validate_table(frame, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": "extracted",
                "rows": len(frame),
                "features": len(FEATURE_COLUMNS),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
