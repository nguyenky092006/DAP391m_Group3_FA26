import copy
import csv
import json
import unittest
from pathlib import Path

from fedecai.features.plan_wav2vec2_full_run import (
    build_chunk_plan,
    plan_sha256,
    validate_chunk_plan,
)


class TestWav2Vec2FullRunPlan(unittest.TestCase):
    def test_repository_plan_only_evidence(self):
        repo_root = Path(__file__).resolve().parents[1]
        evidence_dir = repo_root / "reports/tables/b4"
        report = json.loads(
            (evidence_dir / "b4_wav2vec2_full_chunk_plan.json").read_text(
                encoding="utf-8"
            )
        )
        with (evidence_dir / "b4_wav2vec2_full_chunk_plan.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            plan = list(csv.DictReader(stream))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["mode"], "plan_only")
        self.assertFalse(report["inference_started"])
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["audio_loaded"])
        self.assertEqual(report["chunk_count"], 20)
        self.assertEqual(report["final_chunk_size"], 128)
        self.assertEqual(report["coverage"]["covered_rows"], 4992)
        self.assertEqual(report["coverage"]["unique_covered_samples"], 4992)
        self.assertEqual((report["coverage"]["gaps"], report["coverage"]["overlaps"]), (0, 0))
        self.assertEqual(
            report["plan_sha256"],
            "06bc28e91464a23e8ce8743d8f6ef40f8480be603f05d81c065cec58003a3cb9",
        )
        self.assertEqual(len(plan), 20)
        self.assertEqual(plan[0]["eligible_offset"], "0")
        self.assertEqual(plan[-1]["eligible_offset"], "4864")
        self.assertEqual(plan[-1]["actual_size"], "128")

    def _rows(self, count: int) -> list[dict[str, str]]:
        return [
            {
                "sample_id": f"sample{index}",
                "source_row_index": str(index),
                "speaker_id": f"speaker{index}",
                "accent": "south",
                "emotion": "neutral",
                "audio_sha256": f"{index:064x}",
            }
            for index in range(count)
        ]

    def test_plan_covers_full_and_partial_chunks(self):
        rows = self._rows(5)
        plan = build_chunk_plan(rows, chunk_size=2)
        self.assertEqual([chunk["eligible_offset"] for chunk in plan], [0, 2, 4])
        self.assertEqual([chunk["actual_size"] for chunk in plan], [2, 2, 1])
        result = validate_chunk_plan(rows, plan)
        self.assertEqual(result["covered_rows"], 5)
        self.assertEqual((result["gaps"], result["overlaps"]), (0, 0))

    def test_gap_or_overlap_is_rejected(self):
        rows = self._rows(5)
        plan = build_chunk_plan(rows, chunk_size=2)
        invalid = copy.deepcopy(plan)
        invalid[1]["eligible_offset"] = 1
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            validate_chunk_plan(rows, invalid)

    def test_selection_tampering_is_rejected(self):
        rows = self._rows(3)
        plan = build_chunk_plan(rows, chunk_size=2)
        invalid = copy.deepcopy(plan)
        invalid[0]["selection_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "selection SHA-256"):
            validate_chunk_plan(rows, invalid)

    def test_plan_hash_is_deterministic(self):
        plan = build_chunk_plan(self._rows(5), chunk_size=2)
        self.assertEqual(plan_sha256(plan), plan_sha256(copy.deepcopy(plan)))

    def test_expected_4992_by_256_geometry(self):
        rows = self._rows(4992)
        plan = build_chunk_plan(rows, chunk_size=256)
        self.assertEqual(len(plan), 20)
        self.assertEqual(plan[-1]["eligible_offset"], 4864)
        self.assertEqual(plan[-1]["actual_size"], 128)
        self.assertEqual(validate_chunk_plan(rows, plan)["covered_rows"], 4992)


if __name__ == "__main__":
    unittest.main()
