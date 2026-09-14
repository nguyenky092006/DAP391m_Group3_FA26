"""Deterministic three-channel adapter for existing B4 log-Mel artifacts."""

from __future__ import annotations

import numpy as np


MEL_3CH_MODE = "repeat_normalized_log_mel"
DB_MIN = -80.0
DB_MAX = 0.0


def build_mel_3ch(
    log_mel: np.ndarray,
    *,
    mode: str = MEL_3CH_MODE,
    db_min: float = DB_MIN,
    db_max: float = DB_MAX,
) -> np.ndarray:
    """Return channels-first float32 Mel without fitting population statistics."""
    value = np.asarray(log_mel)
    if mode != MEL_3CH_MODE:
        raise ValueError(f"Unsupported mel_3ch_mode: {mode}")
    if value.ndim != 2 or value.shape[0] != 80 or value.shape[1] == 0:
        raise ValueError(f"Expected log-Mel shape (80, frames), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Log-Mel input contains non-finite values")
    if not db_min < db_max:
        raise ValueError("db_min must be smaller than db_max")
    normalized = (np.clip(value, db_min, db_max) - db_min) / (db_max - db_min)
    return np.repeat(normalized[np.newaxis, :, :], 3, axis=0).astype(np.float32, copy=False)


def pad_or_crop_mel_3ch(mel_3ch: np.ndarray, target_frames: int) -> np.ndarray:
    """Apply the shared center-crop/right-pad policy after train-only size selection."""
    value = np.asarray(mel_3ch, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 3 or value.shape[1] != 80:
        raise ValueError(f"Expected Mel tensor shape (3, 80, frames), got {value.shape}")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    frames = value.shape[2]
    if frames > target_frames:
        start = (frames - target_frames) // 2
        return np.ascontiguousarray(value[:, :, start : start + target_frames])
    if frames < target_frames:
        return np.pad(
            value,
            ((0, 0), (0, 0), (0, target_frames - frames)),
            mode="constant",
            constant_values=0.0,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(value)
