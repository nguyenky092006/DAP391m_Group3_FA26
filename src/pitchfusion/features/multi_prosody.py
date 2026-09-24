"""Frame-level multi-prosody extraction for the approved Pitch-Fusion study.

The output channel order is fixed throughout M3 and M4:
log-F0, voicing confidence, log-energy, delta log-F0, delta log-energy.
No dataset-level normalization is performed here. Normalization statistics must
be fitted later from training samples only.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NUMBA_CACHE_DIR = PROJECT_ROOT / "data/interim/numba_cache"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import librosa
import numpy as np


CHANNELS = (
    "log_f0",
    "voicing_confidence",
    "log_energy",
    "delta_log_f0",
    "delta_log_energy",
)


def extract_multi_prosody(
    signal: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 1024,
    hop_length: int = 160,
    fmin_hz: float = 50.0,
    fmax_hz: float = 500.0,
) -> np.ndarray:
    """Return a finite float32 array with shape ``[5, frames]``.

    Unvoiced log-F0 values are interpolated between voiced frames. Leading and
    trailing unvoiced regions use the closest voiced value. Completely
    unvoiced audio receives a zero log-F0 contour. Voicing confidence remains
    explicit, so the model can distinguish interpolation from voiced evidence.
    """
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16 kHz audio, received {sample_rate} Hz")
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Audio must be non-empty and finite")

    f0_hz, _, voiced_probability = librosa.pyin(
        signal,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        fill_na=np.nan,
    )
    rms = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    frames = min(len(f0_hz), len(voiced_probability), len(rms))
    if frames == 0:
        raise ValueError("Feature extraction produced zero frames")

    f0_hz = np.asarray(f0_hz[:frames], dtype=np.float32)
    confidence = np.nan_to_num(
        np.asarray(voiced_probability[:frames], dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    confidence = np.clip(confidence, 0.0, 1.0)
    voiced = np.isfinite(f0_hz) & (f0_hz > 0.0)
    if voiced.any():
        positions = np.arange(frames, dtype=np.float32)
        log_f0 = np.interp(
            positions,
            positions[voiced],
            np.log(np.maximum(f0_hz[voiced], 1e-6)),
        ).astype(np.float32)
    else:
        log_f0 = np.zeros(frames, dtype=np.float32)

    log_energy = np.log(np.maximum(rms[:frames], 1e-8)).astype(np.float32)
    delta_width = min(9, frames if frames % 2 == 1 else frames - 1)
    if delta_width >= 3:
        delta_log_f0 = librosa.feature.delta(
            log_f0, width=delta_width, mode="nearest"
        ).astype(np.float32)
        delta_log_energy = librosa.feature.delta(
            log_energy, width=delta_width, mode="nearest"
        ).astype(np.float32)
    else:
        delta_log_f0 = np.zeros_like(log_f0)
        delta_log_energy = np.zeros_like(log_energy)

    features = np.stack(
        (log_f0, confidence, log_energy, delta_log_f0, delta_log_energy),
        axis=0,
    ).astype(np.float32)
    if features.shape != (len(CHANNELS), frames) or not np.isfinite(features).all():
        raise ValueError(f"Invalid multi-prosody output: {features.shape}")
    return features


def save_prosody_artifact(path: Path, values: np.ndarray) -> str:
    """Save one artifact and return its SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, values=values, channels=np.asarray(CHANNELS))
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prosody_artifact(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as stored:
        values = np.asarray(stored["values"], dtype=np.float32)
        channels = tuple(str(value) for value in stored["channels"].tolist())
    if channels != CHANNELS:
        raise ValueError(f"Unexpected channel order in {path}: {channels}")
    if values.ndim != 2 or values.shape[0] != len(CHANNELS):
        raise ValueError(f"Invalid prosody shape in {path}: {values.shape}")
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid prosody values in {path}")
    return values
