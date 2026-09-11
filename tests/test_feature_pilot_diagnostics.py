import unittest

from fedecai.features.render_feature_pilot_diagnostics import (
    ACCENT_ORDER,
    EMOTION_ORDER,
    build_pilot_grid,
    expected_centered_frame_count,
)


class TestFeaturePilotDiagnostics(unittest.TestCase):
    def test_expected_frame_count_matches_known_pilot_examples(self):
        self.assertEqual(expected_centered_frame_count(1.984), 199)
        self.assertEqual(expected_centered_frame_count(3.008), 301)
        self.assertEqual(expected_centered_frame_count(8.0), 801)

    def test_balanced_grid_accepts_exactly_twelve_cells(self):
        rows = [
            {
                "sample_id": f"{accent}-{emotion}",
                "accent": accent,
                "emotion": emotion,
            }
            for accent in ACCENT_ORDER
            for emotion in EMOTION_ORDER
        ]
        grid = build_pilot_grid(rows)
        self.assertEqual(len(grid), 12)

    def test_grid_rejects_missing_cell(self):
        rows = [
            {"sample_id": "x", "accent": accent, "emotion": emotion}
            for accent in ACCENT_ORDER
            for emotion in EMOTION_ORDER
        ][:-1]
        with self.assertRaisesRegex(ValueError, "Missing accent-emotion cells"):
            build_pilot_grid(rows)


if __name__ == "__main__":
    unittest.main()
