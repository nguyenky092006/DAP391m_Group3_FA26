import io
import math
import unittest
import wave

import numpy as np

from fedecai.features.extract_feature_pilot import (
    build_smile_extractor,
    decode_wav,
    extract_egemaps,
    extract_log_mel,
    extract_mfcc,
    summarize_pitch,
    extract_pitch,
    validate_feature_value,
)


def sine_wav(duration_sec=1.2, frequency_hz=220.0, sample_rate=16_000):
    time = np.arange(int(duration_sec * sample_rate), dtype=np.float64) / sample_rate
    signal = 0.25 * np.sin(2 * math.pi * frequency_hz * time)
    pcm = np.round(signal * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


class TestFeaturePilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signal, cls.sample_rate, cls.audio_info = decode_wav(sine_wav())

    def test_wav_decode_preserves_audio_contract(self):
        self.assertEqual(self.sample_rate, 16_000)
        self.assertEqual(self.audio_info["channels"], 1)
        self.assertEqual(self.signal.dtype, np.float32)
        self.assertLessEqual(float(np.abs(self.signal).max()), 1.0)

    def test_librosa_feature_shapes(self):
        mfcc = extract_mfcc(self.signal, self.sample_rate)
        log_mel = extract_log_mel(self.signal, self.sample_rate)
        pitch = extract_pitch(self.signal, self.sample_rate)
        validate_feature_value("mfcc40_summary", mfcc)
        validate_feature_value("log_mel_80", log_mel)
        validate_feature_value("pitch_contour", pitch)
        self.assertEqual(mfcc.shape, (80,))
        self.assertEqual(log_mel.shape[0], 80)
        self.assertEqual(pitch["summary"].shape, (6,))

    def test_all_unvoiced_pitch_summary_uses_zero_sentinel(self):
        summary = summarize_pitch(np.full(10, np.nan, dtype=np.float32))
        np.testing.assert_array_equal(summary, np.zeros(6, dtype=np.float32))

    def test_egemaps_has_88_functionals(self):
        values, names = extract_egemaps(
            self.signal, self.sample_rate, build_smile_extractor()
        )
        validate_feature_value("egemaps_v02", values)
        self.assertEqual(values.shape, (88,))
        self.assertEqual(len(names), 88)


if __name__ == "__main__":
    unittest.main()
