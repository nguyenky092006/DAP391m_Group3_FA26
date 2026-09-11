import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fedecai.features.extract_wav2vec2_resume_pilot import (
    atomic_save_embedding,
    expected_record,
    file_sha256,
    inspect_resume_candidate,
    select_rows,
)


class TestWav2Vec2ResumePilot(unittest.TestCase):
    def test_repository_first_run_and_replay_evidence(self):
        repo_root = Path(__file__).resolve().parents[1]
        evidence_dir = repo_root / "reports/tables/b4"
        first = json.loads(
            (evidence_dir / "b4_wav2vec2_resume3_first_run.json").read_text(
                encoding="utf-8"
            )
        )
        replay = json.loads(
            (evidence_dir / "b4_wav2vec2_resume3_replay.json").read_text(
                encoding="utf-8"
            )
        )
        with (evidence_dir / "b4_wav2vec2_resume3_feature_index.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            index = list(csv.DictReader(stream))
        self.assertEqual(first["status"], "PASS")
        self.assertEqual((first["processed"], first["resumed"]), (3, 0))
        self.assertTrue(first["model_loaded"])
        self.assertEqual(replay["status"], "PASS")
        self.assertEqual((replay["processed"], replay["resumed"]), (0, 3))
        self.assertFalse(replay["model_loaded"])
        self.assertEqual(first["artifact_hashes"], replay["artifact_hashes"])
        self.assertEqual(len(index), 3)
        self.assertTrue(all(row["status"] == "ok" for row in index))
        self.assertTrue(all(len(row["artifact_sha256"]) == 64 for row in index))

    def _row(self, sample_id: str, speaker: str, accent: str) -> dict[str, str]:
        return {
            "sample_id": sample_id,
            "source_row_index": sample_id[-1],
            "speaker_id": speaker,
            "accent": accent,
            "emotion": "angry",
            "audio_sha256": "a" * 64,
        }

    def test_selection_requires_three_accents_and_distinct_speakers(self):
        rows = [
            self._row("sample1", "speaker1", "central"),
            self._row("sample2", "speaker2", "north"),
            self._row("sample3", "speaker3", "south"),
        ]
        selected = select_rows(rows, ("sample1", "sample2", "sample3"))
        self.assertEqual([row["accent"] for row in selected], ["central", "north", "south"])

    def test_verified_artifact_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            row = self._row("sample1", "speaker1", "central")
            artifact = repo_root / "data/interim/sample1.npy"
            value = np.arange(768, dtype=np.float32)
            atomic_save_embedding(artifact, value)
            record = expected_record(row, artifact, repo_root, "revision")
            record.update(
                {
                    "artifact_sha256": file_sha256(artifact),
                    "finite_fraction": "1.00000000",
                    "status": "ok",
                }
            )
            reusable, reason = inspect_resume_candidate(record, record.copy(), repo_root)
            self.assertTrue(reusable)
            self.assertEqual(reason, "verified_artifact_reused")

    def test_corrupt_artifact_is_invalidated_for_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            row = self._row("sample1", "speaker1", "central")
            artifact = repo_root / "data/interim/sample1.npy"
            atomic_save_embedding(artifact, np.ones(768, dtype=np.float32))
            record = expected_record(row, artifact, repo_root, "revision")
            record.update(
                {
                    "artifact_sha256": file_sha256(artifact),
                    "finite_fraction": "1.00000000",
                    "status": "ok",
                }
            )
            atomic_save_embedding(artifact, np.zeros(768, dtype=np.float32))
            reusable, reason = inspect_resume_candidate(record, record.copy(), repo_root)
            self.assertFalse(reusable)
            self.assertEqual(reason, "artifact_sha256_mismatch")

    def test_lineage_change_forces_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            row = self._row("sample1", "speaker1", "central")
            artifact = repo_root / "data/interim/sample1.npy"
            atomic_save_embedding(artifact, np.ones(768, dtype=np.float32))
            record = expected_record(row, artifact, repo_root, "revision")
            record.update(
                {
                    "artifact_sha256": file_sha256(artifact),
                    "finite_fraction": "1.00000000",
                    "status": "ok",
                }
            )
            changed = record.copy()
            changed["model_revision"] = "different"
            reusable, reason = inspect_resume_candidate(record, changed, repo_root)
            self.assertFalse(reusable)
            self.assertEqual(reason, "lineage_mismatch:model_revision")


if __name__ == "__main__":
    unittest.main()
