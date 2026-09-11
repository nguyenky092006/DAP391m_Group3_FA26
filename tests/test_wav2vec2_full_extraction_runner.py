import csv
import json
import tempfile
import unittest
from pathlib import Path

from fedecai.features.plan_wav2vec2_full_run import build_chunk_plan
from fedecai.features.run_wav2vec2_full_extraction import validate_chunk_evidence


class TestWav2Vec2FullExtractionRunner(unittest.TestCase):
    def test_chunk_evidence_accepts_matching_lineage_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            rows = [
                {
                    "sample_id": "sample0",
                    "source_row_index": "0",
                    "speaker_id": "speaker0",
                    "accent": "south",
                    "emotion": "neutral",
                    "audio_sha256": "a" * 64,
                }
            ]
            chunk = build_chunk_plan(rows, 256)[0]
            report_path = repo_root / chunk["report_path"]
            index_path = repo_root / chunk["evidence_index_path"]
            report_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_hash = "b" * 64
            report_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "failures": 0,
                        "processed": 1,
                        "resumed": 0,
                        "invalidated": 0,
                        "selected_samples": ["sample0"],
                        "selection": {
                            "selection_sha256": chunk["selection_sha256"],
                            "requested_offset": 0,
                            "actual_size": 1,
                        },
                        "artifact_hashes": {"sample0": artifact_hash},
                    }
                ),
                encoding="utf-8",
            )
            with index_path.open("w", encoding="utf-8", newline="") as stream:
                fields = (
                    "sample_id",
                    "source_row_index",
                    "speaker_id",
                    "emotion",
                    "accent",
                    "audio_sha256",
                    "feature_family",
                    "feature_version",
                    "model_revision",
                    "artifact_path",
                    "artifact_sha256",
                    "shape",
                    "dtype",
                    "finite_fraction",
                    "status",
                    "error",
                )
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "sample0",
                        "source_row_index": "0",
                        "speaker_id": "speaker0",
                        "emotion": "neutral",
                        "accent": "south",
                        "audio_sha256": "a" * 64,
                        "feature_family": "wav2vec2_embedding",
                        "feature_version": "b4_features_v1",
                        "model_revision": "revision",
                        "artifact_path": "data/interim/sample0.npy",
                        "artifact_sha256": artifact_hash,
                        "shape": "768",
                        "dtype": "float32",
                        "finite_fraction": "1.00000000",
                        "status": "ok",
                        "error": "",
                    }
                )
            report, index = validate_chunk_evidence(repo_root, chunk, rows)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(len(index), 1)

    def test_chunk_evidence_rejects_report_index_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            rows = [
                {
                    "sample_id": "sample0",
                    "source_row_index": "0",
                    "speaker_id": "speaker0",
                    "accent": "south",
                    "emotion": "neutral",
                    "audio_sha256": "a" * 64,
                }
            ]
            chunk = build_chunk_plan(rows, 256)[0]
            report_path = repo_root / chunk["report_path"]
            index_path = repo_root / chunk["evidence_index_path"]
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "failures": 0,
                        "selected_samples": ["sample0"],
                        "selection": {
                            "selection_sha256": chunk["selection_sha256"],
                            "requested_offset": 0,
                            "actual_size": 1,
                        },
                        "artifact_hashes": {"sample0": "b" * 64},
                    }
                ),
                encoding="utf-8",
            )
            with index_path.open("w", encoding="utf-8", newline="") as stream:
                fields = (
                    "sample_id",
                    "source_row_index",
                    "speaker_id",
                    "emotion",
                    "accent",
                    "audio_sha256",
                    "feature_family",
                    "feature_version",
                    "model_revision",
                    "artifact_path",
                    "artifact_sha256",
                    "shape",
                    "dtype",
                    "finite_fraction",
                    "status",
                    "error",
                )
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "sample0",
                        "source_row_index": "0",
                        "speaker_id": "speaker0",
                        "emotion": "neutral",
                        "accent": "south",
                        "audio_sha256": "a" * 64,
                        "feature_family": "wav2vec2_embedding",
                        "feature_version": "b4_features_v1",
                        "model_revision": "revision",
                        "artifact_path": "data/interim/sample0.npy",
                        "artifact_sha256": "c" * 64,
                        "shape": "768",
                        "dtype": "float32",
                        "finite_fraction": "1.00000000",
                        "status": "ok",
                        "error": "",
                    }
                )
            with self.assertRaisesRegex(ValueError, "report/index hash mismatch"):
                validate_chunk_evidence(repo_root, chunk, rows)


if __name__ == "__main__":
    unittest.main()
