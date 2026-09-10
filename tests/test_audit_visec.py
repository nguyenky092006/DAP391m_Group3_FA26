import io
import struct
import unittest
import wave

from fedecai.data.audit_visec import (
    ACCENT_MAP,
    EMOTION_ID_MAP,
    inspect_wav,
    write_invalid_audio_report,
)


class TestAuditViSEC(unittest.TestCase):
    def test_expected_label_mappings(self) -> None:
        self.assertEqual(ACCENT_MAP["mid"], "central")
        self.assertEqual(EMOTION_ID_MAP, {"happy": 0, "neutral": 1, "sad": 2, "angry": 3})

    def test_wav_header_is_read_without_extraction(self) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(b"\x00\x00" * 16_000)

        result = inspect_wav(buffer.getvalue())

        self.assertEqual(result["sample_rate_hz"], 16_000)
        self.assertEqual(result["num_channels"], 1)
        self.assertEqual(result["sample_width_bytes"], 2)
        self.assertEqual(result["num_frames"], 16_000)
        self.assertEqual(result["duration_sec_measured"], 1.0)
        self.assertEqual(result["peak_amplitude_ratio"], 0.0)
        self.assertEqual(result["silent_frame_ratio"], 1.0)
        self.assertFalse(result["possible_clipping"])
        self.assertTrue(result["high_silence_ratio"])

    def test_clipping_screen_detects_sustained_full_scale_audio(self) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(struct.pack("<h", 32_767) * 16_000)

        result = inspect_wav(buffer.getvalue())

        self.assertEqual(result["clipped_sample_ratio"], 1.0)
        self.assertTrue(result["possible_clipping"])
        self.assertFalse(result["high_silence_ratio"])

    def test_24_bit_pcm_is_supported(self) -> None:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(3)
            wav_file.setframerate(16_000)
            wav_file.writeframes(b"\xff\xff\x7f" * 1_600)

        result = inspect_wav(buffer.getvalue())

        self.assertEqual(result["sample_width_bytes"], 3)
        self.assertTrue(result["possible_clipping"])
        self.assertLessEqual(result["peak_amplitude_ratio"], 1.0)

    def test_empty_invalid_audio_report_keeps_header(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid_audio.csv"
            write_invalid_audio_report([], output)
            self.assertEqual(
                output.read_text(encoding="utf-8").strip(),
                "sample_id,source_row_index,audio_path_raw,speaker_id,audit_issue",
            )


if __name__ == "__main__":
    unittest.main()
