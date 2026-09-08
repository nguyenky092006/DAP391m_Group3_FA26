from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from pvc_fedorion.data.manifest import build_manifest, parse_vesc_filename


def _write_silence(path: Path, sample_rate: int = 16_000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = sample_rate // 10
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count * channels)


class ParseFilenameTests(unittest.TestCase):
    def test_parses_processed_child_view(self) -> None:
        parsed = parse_vesc_filename("happy_p04_eb_F30_10006_10010_vocals.wav")
        self.assertEqual(parsed.view, "processed")
        self.assertEqual(parsed.utterance_id, "happy_p04_eb_f30_10006_10010")
        self.assertEqual(parsed.source_id, "p04")
        self.assertEqual(parsed.speaker_id, "p04:F30")
        self.assertTrue(parsed.is_child)
        self.assertEqual(parsed.issues, ())

    def test_preserves_label_typo_as_an_issue(self) -> None:
        parsed = parse_vesc_filename("auxiety_p07_F24_12923_12929.wav")
        self.assertEqual(parsed.filename_label_raw, "auxiety")
        self.assertEqual(parsed.filename_label, "anxiety")
        self.assertIn("filename_label_alias", parsed.issues)

    def test_missing_source_is_explicit(self) -> None:
        parsed = parse_vesc_filename("happy_M28_551_554.wav")
        self.assertEqual(parsed.source_id, "unknown")
        self.assertEqual(parsed.speaker_id, "unknown:M28")
        self.assertIn("missing_source_id", parsed.issues)


class ManifestAuditTests(unittest.TestCase):
    def test_detects_pair_and_speaker_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "train"
            test = root / "test"
            output = root / "output"

            _write_silence(train / "Happy" / "happy_p01_F1_100_102.wav")
            _write_silence(test / "Happy" / "happy_p01_F1_100_102_vocals.wav", 24_200)
            _write_silence(test / "Sad" / "sad_p02_M3_200_202.wav")

            manifest_path, audit_path, audit = build_manifest(train, test, output)

            self.assertTrue(manifest_path.is_file())
            self.assertTrue(audit_path.is_file())
            self.assertEqual(audit["total_files"], 3)
            self.assertEqual(audit["utterance_family_count"], 2)
            self.assertEqual(audit["paired_family_count"], 1)
            self.assertEqual(audit["cross_original_split_pair_count"], 1)
            self.assertEqual(audit["pair_sample_rate_mismatch_count"], 1)
            self.assertEqual(audit["test_files_with_train_speaker_count"], 1)


if __name__ == "__main__":
    unittest.main()
