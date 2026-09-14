import json
import unittest
from pathlib import Path

from fedecai.models.run_b6_all_models import DISPLAY_NAMES


ROOT = Path(__file__).resolve().parents[1]


class B6EightModelContractTests(unittest.TestCase):
    def test_consolidated_benchmark_has_exact_required_eight(self):
        self.assertEqual(
            set(DISPLAY_NAMES),
            {
                "svm_rbf", "xgboost", "cnn2d_logmel", "resnet18_mel3",
                "mobilenet_v3_mel3", "vit_b16_mel3",
                "wav2vec2_raw", "pitch_fusion_paper_aligned",
            },
        )

    def test_three_image_models_share_b4_mel_contract(self):
        config = json.loads(
            (ROOT / "configs/models/b6_image_models_v1.json").read_text(encoding="utf-8")
        )
        for name in ("resnet18_mel3", "mobilenet_v3_mel3", "vit_b16_mel3"):
            self.assertIn(name, config["models"])
            self.assertEqual(config["models"][name]["max_frames"], 512)
        vit = config["models"]["vit_b16_mel3"]
        self.assertTrue(vit["freeze_backbone"])
        self.assertTrue(vit["cache_embeddings"])
        self.assertEqual(vit["weights"], "IMAGENET1K_V1")

        wav2vec2 = json.loads(
            (ROOT / "configs/models/b6_wav2vec2_raw_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(wav2vec2["model"]["encoder_policy"], "fully_frozen")
        self.assertTrue(wav2vec2["training"]["embedding_cache"])

if __name__ == "__main__":
    unittest.main()
