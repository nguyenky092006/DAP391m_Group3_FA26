from __future__ import annotations

import numpy as np

from pitchfusion.features.multi_prosody import CHANNELS, extract_multi_prosody


def test_extract_multi_prosody_contract() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    signal = (0.2 * np.sin(2 * np.pi * 180.0 * time)).astype(np.float32)
    values = extract_multi_prosody(signal, sample_rate)
    assert values.ndim == 2
    assert values.shape[0] == len(CHANNELS) == 5
    assert values.shape[1] > 0
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    assert np.all((values[1] >= 0.0) & (values[1] <= 1.0))


def test_extract_multi_prosody_rejects_wrong_sample_rate() -> None:
    signal = np.zeros(16_000, dtype=np.float32)
    try:
        extract_multi_prosody(signal, 8_000)
    except ValueError as error:
        assert "16 kHz" in str(error)
    else:
        raise AssertionError("Expected a sample-rate contract failure")
