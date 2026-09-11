import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fedecai.features.build_b4_deliverables import (
    build_radar_profiles,
    cosine_distance,
    randomized_pca_2d,
    write_dashboard,
)


class TestB4Deliverables(unittest.TestCase):
    def test_randomized_pca_is_deterministic_and_finite(self):
        matrix = np.random.default_rng(1).normal(size=(40, 16)).astype(np.float32)
        first, first_explained = randomized_pca_2d(matrix)
        second, second_explained = randomized_pca_2d(matrix)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (40, 2))
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(first_explained, second_explained)

    def test_cosine_distance_identity(self):
        vector = np.asarray([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_distance(vector, vector), 0.0)

    def test_radar_fallback_columns_exist(self):
        rows = 12
        data = {
            "accent": ["central"] * 4 + ["north"] * 4 + ["south"] * 4,
            "pitch_voiced_fraction": np.linspace(0.1, 0.9, rows),
            "pitch_f0_mean_hz": np.linspace(100, 200, rows),
            "pitch_f0_std_hz": np.linspace(10, 30, rows),
            "mfcc_mean_01": np.linspace(-2, 2, rows),
        }
        for index in range(88):
            data[f"egemaps_{index:03d}_feature"] = np.linspace(index, index + 1, rows)
        profiles, columns = build_radar_profiles(pd.DataFrame(data))
        self.assertEqual(profiles.shape, (3, 6))
        self.assertEqual(len(columns), 6)

    def test_dashboard_contains_plotly_and_filters(self):
        frame = pd.DataFrame(
            [
                {
                    "sample_id": "sample1",
                    "speaker_id": "speaker1",
                    "accent": "central",
                    "emotion": "happy",
                    "pca_1": 1.0,
                    "pca_2": 2.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dashboard.html"
            write_dashboard(frame, {"status": "test"}, path)
            html = path.read_text(encoding="utf-8")
            self.assertIn("plotly-2.35.2.min.js", html)
            self.assertIn('id="accent"', html)
            self.assertIn('id="emotion"', html)


if __name__ == "__main__":
    unittest.main()
