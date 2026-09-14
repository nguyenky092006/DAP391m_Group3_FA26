import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fedecai.eda.b3_eda_sql import (
    build_database,
    build_summary,
    execute_queries,
    load_named_queries,
    validate_summary,
)


def make_row(index, *, accent, emotion, speaker, split, eligible=True):
    return {
        "sample_id": f"sample_{index}",
        "source_row_index": str(index),
        "speaker_id": str(speaker),
        "duration_sec_measured": str(1.0 + index / 10),
        "accent": accent,
        "accent_clean": accent,
        "emotion": emotion,
        "gender": "female",
        "gender_clean": "female",
        "is_eligible_for_split": str(eligible),
        "split": split if eligible else "excluded",
        "possible_clipping": "False",
        "high_silence_ratio": "False",
        "silent_frame_ratio": "0.1",
        "peak_amplitude_ratio": "0.5",
        "cleaning_action": "keep" if eligible else "exclude_duplicate_copy",
        "exclusion_reason": "" if eligible else "duplicate_audio",
        "audio_sha256": f"hash_{index}",
    }


class B3EdaSqlTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.sql_path = self.repo_root / "sql/B3_rq_queries.sql"
        self.rows = [
            make_row(0, accent="south", emotion="happy", speaker=0, split="train"),
            make_row(1, accent="north", emotion="neutral", speaker=1, split="validation"),
            make_row(2, accent="central", emotion="angry", speaker=2, split="test"),
            make_row(3, accent="central", emotion="sad", speaker=2, split="excluded", eligible=False),
        ]
        self.client_rows = [
            {
                "speaker_id": "0",
                "split_version": "speaker_disjoint_v2",
                "main_partition_version": "main_mixed_accent_v1",
                "main_client_id": "main_client_00",
                "stress_partition_version": "stress_accent_domain_v1",
                "stress_client_id": "stress_client_south",
                "accent": "south",
                "samples": "1",
            }
        ]

    def test_named_queries_include_cte_and_window_functions(self):
        queries = load_named_queries(self.sql_path)
        self.assertEqual(
            set(queries),
            {
                "rq1_rq2_samples_speakers_by_accent",
                "rq2_emotion_distribution_by_accent",
                "rq1_rq2_speaker_concentration",
                "rq1_rq2_quality_by_accent_emotion",
                "b5_split_coverage_check",
                "evaluation_speaker_overlap_across_splits",
                "rq5_federated_client_distributions",
                "rq1_rq2_manifest_quality_join",
            },
        )
        combined = "\n".join(queries.values()).upper()
        self.assertIn("WITH ", combined)
        self.assertIn(" OVER (", combined)
        self.assertIn("JOIN QUALITY_RECORDS", combined)

    def test_database_queries_and_summary_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database_path = root / "metadata.sqlite"
            output_dir = root / "tables"
            build_database(self.rows, database_path, self.client_rows)
            row_counts = execute_queries(database_path, self.sql_path, output_dir)
            summary = build_summary(database_path, row_counts)
            validate_summary(summary)

            self.assertEqual(summary["source_rows"], 4)
            self.assertEqual(summary["eligible_rows"], 3)
            self.assertEqual(summary["excluded_rows"], 1)
            self.assertEqual(summary["eligible_speakers"], 3)
            self.assertEqual(row_counts["rq2_emotion_distribution_by_accent"], 12)
            self.assertEqual(row_counts["b5_split_coverage_check"], 36)
            self.assertEqual(row_counts["rq1_rq2_samples_speakers_by_accent"], 3)
            self.assertEqual(row_counts["evaluation_speaker_overlap_across_splits"], 3)
            self.assertEqual(row_counts["rq5_federated_client_distributions"], 2)
            self.assertEqual(row_counts["rq1_rq2_manifest_quality_join"], 3)
            self.assertFalse(summary["split_cross_stratum_complete"])

            connection = sqlite3.connect(database_path)
            try:
                eligible = connection.execute("SELECT COUNT(*) FROM eligible_utterances").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(eligible, 3)

            with (output_dir / "b5_split_coverage_check.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                coverage = list(csv.DictReader(stream))
            self.assertEqual(len(coverage), 36)
            self.assertTrue(any(row["utterances"] == "0" for row in coverage))

            with (output_dir / "evaluation_speaker_overlap_across_splits.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                overlap = list(csv.DictReader(stream))
            self.assertEqual(
                [row["overlapping_speakers"] for row in overlap], ["0", "0", "0"]
            )


if __name__ == "__main__":
    unittest.main()
