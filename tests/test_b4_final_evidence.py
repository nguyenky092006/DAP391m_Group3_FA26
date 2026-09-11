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
        self.assertEqual(handcrafted["status"], "PASS")
        self.assertEqual(handcrafted["clean_records"], 19968)
        self.assertEqual(wav2vec["status"], "PASS")
        self.assertEqual(wav2vec["indexed_artifacts"], 4992)
        self.assertEqual(deliverables["status"], "PASS")
        self.assertEqual(deliverables["feature_table"]["rows"], 4992)
        self.assertEqual(deliverables["feature_table"]["columns"], 951)

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
