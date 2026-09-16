import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PerformanceRerunContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "configs/models/b6_performance_v1.json").read_text(encoding="utf-8")
        )

    def test_protocol_is_explicit_and_test_is_not_used(self):
        self.assertEqual(
            self.config["evaluation_protocol"],
            "emotion_stratified_with_speaker_overlap",
        )
        self.assertFalse(self.config["input_contract"]["test_used_for_selection"])

    def test_exact_five_performance_models_and_vit_skip(self):
        self.assertEqual(
            self.config["models"],
            [
                "cnn2d_logmel", "resnet18_mel3", "mobilenet_v3_mel3",
                "wav2vec2_raw", "pitch_fusion_paper_aligned",
            ],
        )
        self.assertEqual(
            self.config["vit_b16_mel3"]["performance_rerun_status"],
            "skipped_by_human_time_constraint",
        )

    def test_performance_experiment_ids_do_not_collide_with_baselines(self):
        baseline = {
            "b6_cnn2d_logmel_seed391", "b6_resnet18_mel3_seed391",
            "b6_mobilenet_v3_mel3_seed391", "b6_wav2vec2_raw_seed391",
            "b6_pitch_fusion_paper_aligned_seed391",
        }
        performance = {
            self.config[model]["experiment_id"] for model in self.config["models"]
        }
        self.assertFalse(baseline & performance)
        self.assertEqual(len(performance), 5)

    def test_waveform_performance_models_train_transformer_layers(self):
        for model in ("wav2vec2_raw", "pitch_fusion_paper_aligned"):
            self.assertTrue(self.config[model]["feature_extractor_frozen"])
            self.assertGreater(self.config[model]["trainable_transformer_layers"], 0)


if __name__ == "__main__":
    unittest.main()
