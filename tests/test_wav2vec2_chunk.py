import csv
import json
import unittest
from pathlib import Path

from fedecai.features.extract_wav2vec2_chunk import (
    select_eligible_chunk,
    selection_sha256,
)


class TestWav2Vec2Chunk(unittest.TestCase):
    def test_repository_preflight_and_replay_evidence(self):
        repo_root = Path(__file__).resolve().parents[1]
        evidence_dir = repo_root / "reports/tables/b4"
        first = json.loads(
            (evidence_dir / "b4_wav2vec2_chunk_o000000_n000012_first_run.json").read_text(
                encoding="utf-8"
            )
        )
        replay = json.loads(
            (evidence_dir / "b4_wav2vec2_chunk_o000000_n000012_replay.json").read_text(
                encoding="utf-8"
            )
        )
        with (evidence_dir / "b4_wav2vec2_chunk_o000000_n000012_feature_index.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            index = list(csv.DictReader(stream))
        self.assertEqual((first["processed"], first["resumed"]), (12, 0))
        self.assertTrue(first["model_loaded"])
        self.assertEqual((replay["processed"], replay["resumed"]), (0, 12))
        self.assertFalse(replay["model_loaded"])
        self.assertEqual(first["artifact_hashes"], replay["artifact_hashes"])
        self.assertEqual(first["selection"], replay["selection"])
        self.assertEqual(first["selection"]["eligible_rows"], 4992)
        self.assertEqual(len(index), 12)
        self.assertTrue(all(row["status"] == "ok" for row in index))

    def _row(
        self, source_index: int, eligible: str = "True", accent: str = "south"
    ) -> dict[str, str]:
        return {
            "sample_id": f"sample{source_index}",
            "source_row_index": str(source_index),
            "speaker_id": f"speaker{source_index}",
            "accent_clean": accent,
            "emotion": "neutral",
            "audio_sha256": f"{source_index:064x}",
            "is_eligible_for_split": eligible,
        }

    def test_chunk_filters_then_sorts_by_numeric_source_index(self):
        rows = [self._row(10), self._row(2), self._row(1, eligible="False")]
        selected, eligible_count = select_eligible_chunk(rows, offset=0, chunk_size=2)
        self.assertEqual(eligible_count, 2)
        self.assertEqual([row["source_row_index"] for row in selected], ["2", "10"])

    def test_chunk_offset_is_applied_after_eligibility_filter(self):
        rows = [self._row(index) for index in range(5)]
        selected, _ = select_eligible_chunk(rows, offset=2, chunk_size=2)
        self.assertEqual([row["sample_id"] for row in selected], ["sample2", "sample3"])

    def test_final_partial_chunk_is_allowed(self):
        rows = [self._row(index) for index in range(3)]
        selected, _ = select_eligible_chunk(rows, offset=2, chunk_size=10)
        self.assertEqual([row["sample_id"] for row in selected], ["sample2"])

    def test_selection_hash_changes_with_order_or_lineage(self):
        rows = [self._row(1), self._row(2)]
        selected, _ = select_eligible_chunk(rows, offset=0, chunk_size=2)
        original = selection_sha256(selected)
        self.assertEqual(original, selection_sha256(selected))
        self.assertNotEqual(original, selection_sha256(list(reversed(selected))))
        changed = [row.copy() for row in selected]
        changed[0]["audio_sha256"] = "f" * 64
        self.assertNotEqual(original, selection_sha256(changed))

    def test_out_of_range_offset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            select_eligible_chunk([self._row(0)], offset=1, chunk_size=1)


if __name__ == "__main__":
    unittest.main()
