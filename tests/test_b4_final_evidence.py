import csv
import json
import unittest
from pathlib import Path


class TestB4FinalEvidence(unittest.TestCase):
    def test_final_reports_figures_tables_and_dashboard(self):
        repo_root = Path(__file__).resolve().parents[1]
        tables = repo_root / "reports/tables/b4"
        figures = repo_root / "reports/figures/b4"
        handcrafted = json.loads(
            (tables / "b4_handcrafted_full_extraction_report.json").read_text(
                encoding="utf-8"
            )
        )
        wav2vec = json.loads(
            (tables / "b4_wav2vec2_full_extraction_report.json").read_text(
                encoding="utf-8"
            )
        )
        deliverables = json.loads(
            (tables / "b4_deliverables_report.json").read_text(encoding="utf-8")
        )
        mel_3ch = json.loads(
            (tables / "b4_mel_3ch_contract_report.json").read_text(encoding="utf-8")
        )
        accent_probe = json.loads(
            (tables / "b4_accent_probe_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(handcrafted["status"], "PASS")
        self.assertEqual(handcrafted["clean_records"], 19968)
        self.assertEqual(wav2vec["status"], "PASS")
        self.assertEqual(wav2vec["indexed_artifacts"], 4992)
        self.assertEqual(deliverables["status"], "PASS")
        self.assertEqual(deliverables["feature_table"]["rows"], 4992)
        self.assertEqual(deliverables["feature_table"]["columns"], 951)
        self.assertEqual(deliverables["current_contract_revision"], "b4.features.v2")
        self.assertEqual(mel_3ch["status"], "PASS")
        self.assertEqual(mel_3ch["mel_3ch_mode"], "repeat_normalized_log_mel")
        self.assertEqual(mel_3ch["validated_artifacts"], 4992)
        self.assertEqual(accent_probe["status"], "PASS")
        self.assertEqual(accent_probe["split_version"], "speaker_disjoint_v2")
        self.assertEqual(accent_probe["split_audit"]["train_validation_speaker_overlap"], 0)
        self.assertEqual(accent_probe["split_audit"]["test_samples_loaded"], 0)
        self.assertEqual(set(accent_probe["representations"]), {"handcrafted_174", "wav2vec2_768"})

        expected_figures = (
            "b4_wav2vec2_pca_by_accent_emotion.png",
            "b4_accent_emotion_embedding_shift_heatmap.png",
            "b4_accent_acoustic_radar.png",
        )
        for name in expected_figures:
            path = figures / name
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 50_000)

        with (tables / "b4_wav2vec2_pca_projection.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            self.assertEqual(sum(1 for _ in csv.reader(stream)) - 1, 4992)

        html = (repo_root / "reports/dashboard/b4_feature_dashboard.html").read_text(
            encoding="utf-8"
        )
        for token in ("Plotly.react", 'id="accent"', 'id="emotion"', "dragmode:'zoom'"):
            self.assertIn(token, html)


if __name__ == "__main__":
    unittest.main()
