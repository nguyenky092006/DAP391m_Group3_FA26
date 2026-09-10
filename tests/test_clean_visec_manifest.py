import unittest

from fedecai.data.clean_visec_manifest import (
    _majority,
    apply_cleaning_policy,
    normalized_text_sha256,
)


def make_row(
    index,
    speaker,
    audio_hash,
    group="",
    issues="",
    accent="south",
    gender="female",
    emotion="happy",
    audio_readable="True",
):
    return {
        "source_row_index": str(index),
        "sample_id": f"sample_{index}",
        "speaker_id": str(speaker),
        "audio_sha256": audio_hash,
        "duplicate_content_group": group,
        "audit_issue": issues,
        "accent": accent,
        "gender": gender,
        "emotion": emotion,
        "audio_readable": audio_readable,
    }


class TestCleanViSECManifest(unittest.TestCase):
    def test_majority_rejects_tie(self):
        with self.assertRaises(ValueError):
            _majority(["north", "south"])

    def test_consistent_duplicate_keeps_lowest_row(self):
        rows = [
            make_row(2, 1, "same", "dup_1", "duplicate_audio"),
            make_row(1, 1, "same", "dup_1", "duplicate_audio"),
            make_row(3, 2, "unique"),
        ]
        cleaned, report = apply_cleaning_policy(rows)
        eligibility = {row["sample_id"]: row["is_eligible_for_split"] for row in cleaned}
        self.assertTrue(eligibility["sample_1"])
        self.assertFalse(eligibility["sample_2"])
        self.assertEqual(report["eligible_rows"], 2)

    def test_conflicting_duplicate_excludes_whole_group(self):
        issue = "duplicate_audio;conflicting_duplicate_metadata"
        rows = [
            make_row(1, 1, "same", "dup_1", issue, emotion="happy"),
            make_row(2, 1, "same", "dup_1", issue, emotion="sad"),
            make_row(3, 2, "unique"),
        ]
        cleaned, report = apply_cleaning_policy(rows)
        excluded = [row for row in cleaned if not row["is_eligible_for_split"]]
        self.assertEqual(len(excluded), 2)
        self.assertEqual(report["eligible_rows"], 1)

    def test_invalid_records_are_retained_but_excluded(self):
        rows = [
            make_row(1, 1, "bad_audio", audio_readable="False"),
            make_row(2, "", "missing_speaker"),
            make_row(3, 3, "bad_emotion", emotion="unknown"),
            make_row(4, 4, "bad_accent", accent="unknown"),
            make_row(5, 5, "valid"),
        ]

        cleaned, report = apply_cleaning_policy(rows)
        actions = {row["sample_id"]: row["cleaning_action"] for row in cleaned}

        self.assertEqual(report["eligible_rows"], 1)
        self.assertEqual(actions["sample_1"], "exclude_invalid_audio")
        self.assertEqual(actions["sample_2"], "exclude_missing_speaker")
        self.assertEqual(actions["sample_3"], "exclude_invalid_emotion")
        self.assertEqual(actions["sample_4"], "exclude_invalid_accent")
        self.assertTrue(all(row["exclusion_reason"] for row in cleaned[:4]))

    def test_manifest_hash_ignores_line_ending_style(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            lf = Path(directory) / "lf.csv"
            crlf = Path(directory) / "crlf.csv"
            lf.write_bytes(b"a,b\n1,2\n")
            crlf.write_bytes(b"a,b\r\n1,2\r\n\r\n")
            self.assertEqual(normalized_text_sha256(lf), normalized_text_sha256(crlf))


if __name__ == "__main__":
    unittest.main()
